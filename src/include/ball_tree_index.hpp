#pragma once

#include "ball_tree.hpp"

#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/execution/index/index_type.hpp"
#include "duckdb/storage/index_storage_info.hpp"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace duckdb {

class ExtensionLoader;

//! Registers the BALL_TREE index type and the ball_index_* table functions.
void RegisterBallTreeIndex(ExtensionLoader &loader);

//! One indexed point: the table row it belongs to and its coordinates in radians.
struct BallTreePoint {
	row_t row_id;
	double lat;
	double lon;
};

//! An immutable ball tree together with the table row id of each of its points. `tree` is null when the
//! index holds no points.
struct BallTreeSnapshot {
	unique_ptr<balltree::BallTree> tree;
	//! row_ids[i] is the row id of tree point i
	vector<row_t> row_ids;
	//! the same row ids, sorted, to answer "is this row in the tree?"
	vector<row_t> sorted_row_ids;

	bool Contains(row_t row_id) const;
	idx_t Count() const {
		return row_ids.size();
	}
};

struct BallTreeIndexStats {
	//! points in the current tree
	idx_t tree_points = 0;
	//! rows added since the tree was built
	idx_t pending_inserts = 0;
	//! rows removed since the tree was built
	idx_t pending_deletes = 0;
	//! true when the table changed since the tree was built, so the next query rebuilds it
	bool stale = false;
	//! how many times the tree has been rebuilt to fold in changes
	idx_t rebuilds = 0;
	idx_t leaf_size = 0;
	idx_t memory_bytes = 0;
};

//! A ball tree index over a (latitude, longitude) pair of DOUBLE/FLOAT columns, in degrees.
//!
//! DuckDB tells the index about every insert and delete on the table (an UPDATE of an indexed column is a delete
//! plus an insert; UPDATEs of other columns never reach the index). Instead of editing the tree, the index keeps
//! the tree immutable and records the change in a small delta. The tree is "stale" while a delta is pending, and
//! the next query (or checkpoint) rebuilds it from the tree plus the delta. The tree is written into the
//! database file at checkpoint, and read back as-is on load.
class BallTreeIndex : public BoundIndex {
public:
	static constexpr const char *TYPE_NAME = "BALL_TREE";
	static constexpr idx_t DEFAULT_LEAF_SIZE = 40;

	BallTreeIndex(const string &name, IndexConstraintType constraint_type, const vector<column_t> &column_ids,
	              TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
	              AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
	              const IndexStorageInfo &info = IndexStorageInfo());

	static IndexType MakeIndexType();
	//! Returns leaf_size from the WITH (...) options. With `strict` (CREATE INDEX), an unknown or invalid option
	//! throws BinderException; when loading an existing index, unknown options are ignored.
	static idx_t ParseLeafSize(const case_insensitive_map_t<Value> &options, bool strict);
	static bool IsCoordinateType(const LogicalType &type);
	//! Appends the (lat, lon, row id) triples of `keys` to `out`, skipping rows where either coordinate is NULL.
	//! Coordinates are converted to radians. With `validate`, out-of-range values throw.
	static void ExtractPoints(DataChunk &keys, Vector &row_ids, vector<BallTreePoint> &out, bool validate);

	//! Replaces the whole content (CREATE INDEX). The tree is built straight away.
	void Load(vector<BallTreePoint> &&points);
	//! The tree as of now: pending changes are folded in first if the index is stale.
	std::shared_ptr<const BallTreeSnapshot> GetSnapshot();
	//! Counters for observability. Never triggers a rebuild.
	BallTreeIndexStats GetStats();

public:
	ErrorData Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) override;
	ErrorData Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) override;
	idx_t TryDelete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers,
	                optional_ptr<SelectionVector> deleted_sel = nullptr,
	                optional_ptr<SelectionVector> non_deleted_sel = nullptr) override;
	void CommitDrop(IndexLock &index_lock) override;
	bool MergeIndexes(IndexLock &state, BoundIndex &other_index) override;
	void Vacuum(IndexLock &state) override;
	idx_t GetInMemorySize(IndexLock &state) override;
	void Verify(IndexLock &state) override;
	string ToString(IndexLock &state, bool display_ascii = false) override;
	void VerifyAllocations(IndexLock &state) override;
	void VerifyBuffers(IndexLock &lock) override;
	IndexStorageInfo SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) override;
	IndexStorageInfo SerializeToWAL(const case_insensitive_map_t<Value> &options) override;
	string GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
	                                     DataChunk &input) override {
		return "Constraint violation in BALL_TREE index";
	}

private:
	using Coord = std::pair<double, double>;

	void AddPoints(vector<BallTreePoint> &points);
	void RemovePoints(vector<BallTreePoint> &points);
	bool IsStaleLocked() const;
	//! Folds the pending changes into a fresh tree. data_lock_ must be held.
	void RebuildLocked();
	std::shared_ptr<const BallTreeSnapshot> BuildSnapshot(vector<double> &coords, vector<row_t> &&row_ids) const;
	//! Writes the current tree into the linked blocks. data_lock_ must be held.
	void PersistLocked();
	void LoadFromStorage();

	idx_t leaf_size_;

	//! Guards everything below. Queries and DuckDB's Append/Delete calls can arrive from different threads.
	mutex data_lock_;
	std::shared_ptr<const BallTreeSnapshot> tree_;
	//! rows added since tree_ was built
	std::unordered_map<row_t, Coord> added_;
	//! rows of tree_ that were removed since it was built
	std::unordered_set<row_t> deleted_;
	idx_t rebuilds_ = 0;
	//! the content changed since it was last written to storage
	bool needs_persist_ = false;

	//! Storage of the serialized tree as a chain of blocks
	unique_ptr<FixedSizeAllocator> block_allocator_;
	IndexPointer root_;
};

} // namespace duckdb
