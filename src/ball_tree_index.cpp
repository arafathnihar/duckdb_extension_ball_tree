#include "ball_tree_index.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/partial_block_manager.hpp"
#include "duckdb/storage/storage_info.hpp"
#include "duckdb/storage/table_io_manager.hpp"

#include <algorithm>
#include <cmath>

namespace duckdb {

using balltree::BallTree;

//===--------------------------------------------------------------------===//
// Serialized tree storage: a byte stream over a chain of allocator segments
//===--------------------------------------------------------------------===//

namespace {

class LinkedBlock {
public:
	static constexpr const idx_t BLOCK_SIZE = Storage::DEFAULT_BLOCK_SIZE - sizeof(validity_t);
	static constexpr const idx_t BLOCK_DATA_SIZE = BLOCK_SIZE - sizeof(IndexPointer);
	static_assert(BLOCK_SIZE > sizeof(IndexPointer), "Block size must be larger than the size of an IndexPointer");

	IndexPointer next_block;
	char data[BLOCK_DATA_SIZE] = {0};
};

class LinkedBlockReader {
public:
	LinkedBlockReader(FixedSizeAllocator &allocator, IndexPointer root_pointer)
	    : allocator(allocator), current_pointer(root_pointer), position_in_block(0) {
	}

	void ReadData(data_ptr_t buffer, idx_t length) {
		idx_t bytes_read = 0;
		while (bytes_read < length) {
			auto block = allocator.Get<const LinkedBlock>(current_pointer, false);
			const auto to_read = MinValue<idx_t>(length - bytes_read, LinkedBlock::BLOCK_DATA_SIZE - position_in_block);
			std::memcpy(buffer + bytes_read, block->data + position_in_block, to_read);
			bytes_read += to_read;
			position_in_block += to_read;
			if (position_in_block == LinkedBlock::BLOCK_DATA_SIZE) {
				position_in_block = 0;
				current_pointer = block->next_block;
			}
		}
	}

private:
	FixedSizeAllocator &allocator;
	IndexPointer current_pointer;
	idx_t position_in_block;
};

class LinkedBlockWriter {
public:
	LinkedBlockWriter(FixedSizeAllocator &allocator, IndexPointer root_pointer)
	    : allocator(allocator), current_pointer(root_pointer), position_in_block(0) {
		ClearCurrentBlock();
	}

	void WriteData(const_data_ptr_t buffer, idx_t length) {
		idx_t bytes_written = 0;
		while (bytes_written < length) {
			auto block = allocator.Get<LinkedBlock>(current_pointer, true);
			const auto to_write =
			    MinValue<idx_t>(length - bytes_written, LinkedBlock::BLOCK_DATA_SIZE - position_in_block);
			std::memcpy(block->data + position_in_block, buffer + bytes_written, to_write);
			bytes_written += to_write;
			position_in_block += to_write;
			if (position_in_block == LinkedBlock::BLOCK_DATA_SIZE) {
				position_in_block = 0;
				block->next_block = allocator.New();
				current_pointer = block->next_block;
				ClearCurrentBlock();
			}
		}
	}

private:
	void ClearCurrentBlock() {
		auto block = allocator.Get<LinkedBlock>(current_pointer, true);
		block->next_block.Clear();
		std::memset(block->data, 0, LinkedBlock::BLOCK_DATA_SIZE);
	}

	FixedSizeAllocator &allocator;
	IndexPointer current_pointer;
	idx_t position_in_block;
};

constexpr uint32_t INDEX_MAGIC = 0x42544958; // "BTIX"
constexpr uint32_t INDEX_VERSION = 1;
//! Anything larger is treated as corruption rather than allocated.
constexpr uint64_t MAX_BLOB_BYTES = uint64_t(1) << 40;

template <class T>
void PutValue(vector<uint8_t> &out, T value) {
	const auto *p = reinterpret_cast<const uint8_t *>(&value);
	out.insert(out.end(), p, p + sizeof(T));
}

} // namespace

//===--------------------------------------------------------------------===//
// BallTreeSnapshot
//===--------------------------------------------------------------------===//

bool BallTreeSnapshot::Contains(row_t row_id) const {
	return std::binary_search(sorted_row_ids.begin(), sorted_row_ids.end(), row_id);
}

//===--------------------------------------------------------------------===//
// BallTreeIndex
//===--------------------------------------------------------------------===//

bool BallTreeIndex::IsCoordinateType(const LogicalType &type) {
	return type.id() == LogicalTypeId::DOUBLE || type.id() == LogicalTypeId::FLOAT;
}

idx_t BallTreeIndex::ParseLeafSize(const case_insensitive_map_t<Value> &options, bool strict) {
	idx_t leaf_size = DEFAULT_LEAF_SIZE;
	for (auto &option : options) {
		if (!StringUtil::CIEquals(option.first, "leaf_size")) {
			if (strict) {
				throw BinderException("Unknown option for BALL_TREE index: '%s'", option.first);
			}
			continue;
		}
		const auto &value = option.second;
		if (value.IsNull() || !value.type().IsIntegral()) {
			if (strict) {
				throw BinderException("BALL_TREE index 'leaf_size' must be an integer");
			}
			continue;
		}
		const auto parsed = value.GetValue<int64_t>();
		if (parsed < 1) {
			if (strict) {
				throw BinderException("BALL_TREE index 'leaf_size' must be at least 1");
			}
			continue;
		}
		leaf_size = static_cast<idx_t>(parsed);
	}
	return leaf_size;
}

BallTreeIndex::BallTreeIndex(const string &name, IndexConstraintType constraint_type,
                             const vector<column_t> &column_ids, TableIOManager &table_io_manager,
                             const vector<unique_ptr<Expression>> &unbound_expressions, AttachedDatabase &db,
                             const case_insensitive_map_t<Value> &options, const IndexStorageInfo &info)
    : BoundIndex(name, TYPE_NAME, constraint_type, column_ids, table_io_manager, unbound_expressions, db),
      leaf_size_(ParseLeafSize(options, false)) {
	if (constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("BALL_TREE indexes do not support unique or primary key constraints");
	}
	if (logical_types.size() != 2 || !IsCoordinateType(logical_types[0]) || !IsCoordinateType(logical_types[1])) {
		throw BinderException("BALL_TREE indexes are created over exactly two DOUBLE or FLOAT columns: "
		                      "(latitude, longitude)");
	}

	block_allocator_ = make_uniq<FixedSizeAllocator>(sizeof(LinkedBlock), table_io_manager.GetIndexBlockManager());
	tree_ = std::make_shared<const BallTreeSnapshot>();

	if (info.IsValid()) {
		// an existing index that is being loaded
		root_.Set(info.root);
		D_ASSERT(info.allocator_infos.size() == 1);
		block_allocator_->Init(info.allocator_infos[0]);
		// an index with no points stores no blocks
		if (!info.allocator_infos[0].buffer_ids.empty()) {
			LoadFromStorage();
		}
	} else {
		needs_persist_ = true;
	}
}

void BallTreeIndex::ExtractPoints(DataChunk &keys, Vector &row_ids, vector<BallTreePoint> &out, bool validate) {
	const auto count = keys.size();
	if (count == 0) {
		return;
	}
	D_ASSERT(keys.ColumnCount() == 2);

	UnifiedVectorFormat lat_format, lon_format, row_format;
	keys.data[0].ToUnifiedFormat(count, lat_format);
	keys.data[1].ToUnifiedFormat(count, lon_format);
	row_ids.ToUnifiedFormat(count, row_format);
	const bool lat_is_float = keys.data[0].GetType().id() == LogicalTypeId::FLOAT;
	const bool lon_is_float = keys.data[1].GetType().id() == LogicalTypeId::FLOAT;
	const auto rows = UnifiedVectorFormat::GetData<row_t>(row_format);

	out.reserve(out.size() + count);
	for (idx_t i = 0; i < count; i++) {
		const auto lat_idx = lat_format.sel->get_index(i);
		const auto lon_idx = lon_format.sel->get_index(i);
		if (!lat_format.validity.RowIsValid(lat_idx) || !lon_format.validity.RowIsValid(lon_idx)) {
			continue; // a point without coordinates is not indexed
		}
		const double lat = lat_is_float ? UnifiedVectorFormat::GetData<float>(lat_format)[lat_idx]
		                                : UnifiedVectorFormat::GetData<double>(lat_format)[lat_idx];
		const double lon = lon_is_float ? UnifiedVectorFormat::GetData<float>(lon_format)[lon_idx]
		                                : UnifiedVectorFormat::GetData<double>(lon_format)[lon_idx];
		const row_t row_id = rows[row_format.sel->get_index(i)];
		if (validate && (!std::isfinite(lat) || !std::isfinite(lon) || lat < -90.0 || lat > 90.0)) {
			throw InvalidInputException("BALL_TREE index: invalid coordinate (latitude %f, longitude %f); latitude "
			                            "must be within [-90, 90] and both must be finite",
			                            lat, lon);
		}
		out.push_back({row_id, lat * balltree::PI / 180.0, lon * balltree::PI / 180.0});
	}
}

//===--------------------------------------------------------------------===//
// The tree and its pending changes
//===--------------------------------------------------------------------===//

std::shared_ptr<const BallTreeSnapshot> BallTreeIndex::BuildSnapshot(vector<double> &coords,
                                                                    vector<row_t> &&row_ids) const {
	auto snapshot = std::make_shared<BallTreeSnapshot>();
	if (!row_ids.empty()) {
		snapshot->tree = make_uniq<BallTree>(coords.data(), row_ids.size(), leaf_size_);
	}
	snapshot->sorted_row_ids = row_ids;
	std::sort(snapshot->sorted_row_ids.begin(), snapshot->sorted_row_ids.end());
	snapshot->row_ids = std::move(row_ids);
	return snapshot;
}

bool BallTreeIndex::IsStaleLocked() const {
	return !added_.empty() || !deleted_.empty();
}

void BallTreeIndex::RebuildLocked() {
	const idx_t tree_count = tree_->Count();
	vector<double> coords;
	vector<row_t> row_ids;
	coords.reserve((tree_count + added_.size()) * 2);
	row_ids.reserve(tree_count + added_.size());

	// the surviving points of the old tree ...
	for (idx_t i = 0; i < tree_count; i++) {
		const row_t row_id = tree_->row_ids[i];
		if (deleted_.count(row_id) || added_.count(row_id)) {
			continue; // removed, or replaced by a later add of the same row
		}
		const double *point = tree_->tree->Point(static_cast<int64_t>(i));
		coords.push_back(point[0]);
		coords.push_back(point[1]);
		row_ids.push_back(row_id);
	}
	// ... plus everything added since
	for (auto &added : added_) {
		coords.push_back(added.second.first);
		coords.push_back(added.second.second);
		row_ids.push_back(added.first);
	}

	tree_ = BuildSnapshot(coords, std::move(row_ids));
	added_.clear();
	deleted_.clear();
	rebuilds_++;
}

void BallTreeIndex::Load(vector<BallTreePoint> &&points) {
	vector<double> coords;
	vector<row_t> row_ids;
	coords.reserve(points.size() * 2);
	row_ids.reserve(points.size());
	for (auto &point : points) {
		coords.push_back(point.lat);
		coords.push_back(point.lon);
		row_ids.push_back(point.row_id);
	}
	lock_guard<mutex> guard(data_lock_);
	tree_ = BuildSnapshot(coords, std::move(row_ids));
	added_.clear();
	deleted_.clear();
	needs_persist_ = true;
}

std::shared_ptr<const BallTreeSnapshot> BallTreeIndex::GetSnapshot() {
	lock_guard<mutex> guard(data_lock_);
	if (IsStaleLocked()) {
		RebuildLocked();
	}
	return tree_;
}

void BallTreeIndex::AddPoints(vector<BallTreePoint> &points) {
	if (points.empty()) {
		return;
	}
	lock_guard<mutex> guard(data_lock_);
	for (auto &point : points) {
		added_[point.row_id] = {point.lat, point.lon};
	}
	needs_persist_ = true;
}

void BallTreeIndex::RemovePoints(vector<BallTreePoint> &points) {
	lock_guard<mutex> guard(data_lock_);
	bool changed = false;
	for (auto &point : points) {
		// a row added since the last build simply disappears from the delta ...
		changed |= added_.erase(point.row_id) > 0;
		// ... a row that is in the tree needs a tombstone; a row in neither was never indexed
		if (tree_->Contains(point.row_id)) {
			changed |= deleted_.insert(point.row_id).second;
		}
	}
	if (changed) {
		needs_persist_ = true;
	}
}

BallTreeIndexStats BallTreeIndex::GetStats() {
	lock_guard<mutex> guard(data_lock_);
	BallTreeIndexStats stats;
	stats.tree_points = tree_->Count();
	stats.pending_inserts = added_.size();
	stats.pending_deletes = deleted_.size();
	stats.stale = IsStaleLocked();
	stats.rebuilds = rebuilds_;
	stats.leaf_size = leaf_size_;
	stats.memory_bytes = (tree_->tree ? tree_->tree->MemoryUsage() : 0) +
	                     tree_->row_ids.size() * sizeof(row_t) * 2 +
	                     added_.size() * (sizeof(row_t) + sizeof(Coord)) + deleted_.size() * sizeof(row_t);
	return stats;
}

//===--------------------------------------------------------------------===//
// BoundIndex: what DuckDB calls when the table changes
//===--------------------------------------------------------------------===//

ErrorData BallTreeIndex::Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) {
	DataChunk keys;
	keys.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(entries, keys);

	vector<BallTreePoint> points;
	ExtractPoints(keys, row_identifiers, points, true);
	AddPoints(points);
	return ErrorData();
}

ErrorData BallTreeIndex::Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) {
	vector<BallTreePoint> points;
	ExtractPoints(data, row_ids, points, true);
	AddPoints(points);
	return ErrorData();
}

idx_t BallTreeIndex::TryDelete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers,
                               optional_ptr<SelectionVector> deleted_sel,
                               optional_ptr<SelectionVector> non_deleted_sel) {
	DataChunk keys;
	keys.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(entries, keys);

	vector<BallTreePoint> points;
	ExtractPoints(keys, row_identifiers, points, false);
	RemovePoints(points);
	return entries.size();
}

void BallTreeIndex::CommitDrop(IndexLock &index_lock) {
	lock_guard<mutex> guard(data_lock_);
	tree_ = std::make_shared<const BallTreeSnapshot>();
	added_.clear();
	deleted_.clear();
	block_allocator_->Reset();
	root_.Clear();
	needs_persist_ = false;
}

bool BallTreeIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	throw NotImplementedException("BallTreeIndex::MergeIndexes() is not implemented");
}

void BallTreeIndex::Vacuum(IndexLock &state) {
}

idx_t BallTreeIndex::GetInMemorySize(IndexLock &state) {
	return GetStats().memory_bytes;
}

void BallTreeIndex::Verify(IndexLock &state) {
}

string BallTreeIndex::ToString(IndexLock &state, bool display_ascii) {
	const auto stats = GetStats();
	return StringUtil::Format("BALL_TREE(points=%llu, pending_inserts=%llu, pending_deletes=%llu, stale=%s)",
	                          static_cast<unsigned long long>(stats.tree_points),
	                          static_cast<unsigned long long>(stats.pending_inserts),
	                          static_cast<unsigned long long>(stats.pending_deletes), stats.stale ? "true" : "false");
}

void BallTreeIndex::VerifyAllocations(IndexLock &state) {
}

void BallTreeIndex::VerifyBuffers(IndexLock &lock) {
	block_allocator_->VerifyBuffers();
}

//===--------------------------------------------------------------------===//
// Persistence
//===--------------------------------------------------------------------===//

void BallTreeIndex::PersistLocked() {
	// the file always holds a fresh tree, never a tree plus a delta
	if (IsStaleLocked()) {
		RebuildLocked();
	}
	if (!needs_persist_) {
		return;
	}
	// A rewrite replaces the old chain of blocks; Reset releases the ones it occupied.
	block_allocator_->Reset();
	root_.Clear();
	needs_persist_ = false;
	if (tree_->Count() == 0) {
		return; // an empty index stores nothing and reloads as empty
	}

	// layout: magic, version, leaf_size, point count, row ids (in tree point order), then the serialized tree
	vector<uint8_t> tree_bytes;
	tree_->tree->Serialize(tree_bytes);
	vector<uint8_t> blob;
	blob.reserve(24 + tree_->row_ids.size() * sizeof(row_t) + tree_bytes.size());
	PutValue<uint32_t>(blob, INDEX_MAGIC);
	PutValue<uint32_t>(blob, INDEX_VERSION);
	PutValue<uint64_t>(blob, tree_->tree->LeafSize());
	PutValue<uint64_t>(blob, tree_->row_ids.size());
	const auto *row_bytes = reinterpret_cast<const uint8_t *>(tree_->row_ids.data());
	blob.insert(blob.end(), row_bytes, row_bytes + tree_->row_ids.size() * sizeof(row_t));
	blob.insert(blob.end(), tree_bytes.begin(), tree_bytes.end());

	root_ = block_allocator_->New();
	LinkedBlockWriter writer(*block_allocator_, root_);
	const uint64_t length = blob.size();
	writer.WriteData(reinterpret_cast<const_data_ptr_t>(&length), sizeof(length));
	writer.WriteData(blob.data(), blob.size());
}

void BallTreeIndex::LoadFromStorage() {
	LinkedBlockReader reader(*block_allocator_, root_);
	uint64_t length = 0;
	reader.ReadData(reinterpret_cast<data_ptr_t>(&length), sizeof(length));
	if (length < 24 || length > MAX_BLOB_BYTES) {
		throw IOException("BALL_TREE index '%s' is corrupt: bad length", name);
	}
	vector<uint8_t> blob(length);
	reader.ReadData(blob.data(), blob.size());

	uint32_t magic, version;
	uint64_t leaf_size, count;
	std::memcpy(&magic, blob.data(), 4);
	std::memcpy(&version, blob.data() + 4, 4);
	std::memcpy(&leaf_size, blob.data() + 8, 8);
	std::memcpy(&count, blob.data() + 16, 8);
	if (magic != INDEX_MAGIC || version != INDEX_VERSION) {
		throw IOException("BALL_TREE index '%s' has an unsupported storage format", name);
	}
	const uint64_t rows_bytes = count * sizeof(row_t);
	if (count == 0 || leaf_size == 0 || count > length / sizeof(row_t) || 24 + rows_bytes > length) {
		throw IOException("BALL_TREE index '%s' is corrupt: bad header", name);
	}

	auto snapshot = std::make_shared<BallTreeSnapshot>();
	snapshot->row_ids.resize(count);
	std::memcpy(snapshot->row_ids.data(), blob.data() + 24, rows_bytes);
	try {
		snapshot->tree = make_uniq<BallTree>(BallTree::Deserialize(blob.data() + 24 + rows_bytes, length - 24 - rows_bytes));
	} catch (std::runtime_error &error) {
		throw IOException("BALL_TREE index '%s' is corrupt: %s", name, error.what());
	}
	if (snapshot->tree->Size() != count) {
		throw IOException("BALL_TREE index '%s' is corrupt: row count does not match the tree", name);
	}
	snapshot->sorted_row_ids = snapshot->row_ids;
	std::sort(snapshot->sorted_row_ids.begin(), snapshot->sorted_row_ids.end());

	leaf_size_ = leaf_size; // rebuilds keep the leaf size the tree was built with
	tree_ = std::move(snapshot);
	needs_persist_ = false;
}

IndexStorageInfo BallTreeIndex::SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) {
	lock_guard<mutex> guard(data_lock_);
	PersistLocked();

	IndexStorageInfo info;
	info.name = name;
	info.root = root_.Get();

	// hand the blocks to the checkpoint's partial block manager
	auto &block_manager = table_io_manager.GetIndexBlockManager();
	PartialBlockManager partial_block_manager(context, block_manager, PartialBlockType::FULL_CHECKPOINT);
	block_allocator_->SerializeBuffers(partial_block_manager);
	partial_block_manager.FlushPartialBlocks();
	info.allocator_infos.push_back(block_allocator_->GetInfo());
	return info;
}

IndexStorageInfo BallTreeIndex::SerializeToWAL(const case_insensitive_map_t<Value> &options) {
	lock_guard<mutex> guard(data_lock_);
	PersistLocked();

	IndexStorageInfo info;
	info.name = name;
	info.root = root_.Get();
	info.buffers.push_back(block_allocator_->InitSerializationToWAL());
	info.allocator_infos.push_back(block_allocator_->GetInfo());
	return info;
}

//===--------------------------------------------------------------------===//
// CREATE INDEX: DuckDB scans the table and feeds (lat, lon, row id) chunks to these callbacks
//===--------------------------------------------------------------------===//

namespace {

struct BuildBindData : public IndexBuildBindData {
	idx_t leaf_size = BallTreeIndex::DEFAULT_LEAF_SIZE;
};

struct BuildGlobalState : public IndexBuildGlobalState {
	unique_ptr<BallTreeIndex> index;
	mutex lock;
	vector<BallTreePoint> points;
};

struct BuildLocalState : public IndexBuildLocalState {
	vector<BallTreePoint> points;
};

unique_ptr<IndexBuildBindData> BuildBind(IndexBuildBindInput &input) {
	if (input.expressions.size() != 2) {
		throw BinderException("BALL_TREE indexes are created over exactly two columns: (latitude, longitude)");
	}
	for (auto &expression : input.expressions) {
		if (!BallTreeIndex::IsCoordinateType(expression->return_type)) {
			throw BinderException("BALL_TREE index columns must be DOUBLE or FLOAT, got %s. Cast them in the index, "
			                      "e.g. CREATE INDEX i ON t USING BALL_TREE ((lat::DOUBLE), (lon::DOUBLE))",
			                      expression->return_type.ToString());
		}
	}
	auto bind_data = make_uniq<BuildBindData>();
	bind_data->leaf_size = BallTreeIndex::ParseLeafSize(input.info.options, true);
	return std::move(bind_data);
}

unique_ptr<IndexBuildGlobalState> BuildGlobalInit(IndexBuildInitGlobalStateInput &input) {
	auto state = make_uniq<BuildGlobalState>();
	auto &storage = input.table.GetStorage();
	state->index = make_uniq<BallTreeIndex>(input.info.index_name, input.info.constraint_type, input.storage_ids,
	                                        TableIOManager::Get(storage), input.expressions, storage.db,
	                                        input.info.options);
	return std::move(state);
}

unique_ptr<IndexBuildLocalState> BuildLocalInit(IndexBuildInitLocalStateInput &input) {
	return make_uniq<BuildLocalState>();
}

void BuildSink(IndexBuildSinkInput &input, DataChunk &key_chunk, DataChunk &row_chunk) {
	auto &local_state = input.local_state.Cast<BuildLocalState>();
	BallTreeIndex::ExtractPoints(key_chunk, row_chunk.data[0], local_state.points, true);
}

void BuildCombine(IndexBuildCombineInput &input) {
	auto &global_state = input.global_state.Cast<BuildGlobalState>();
	auto &local_state = input.local_state.Cast<BuildLocalState>();
	lock_guard<mutex> guard(global_state.lock);
	global_state.points.insert(global_state.points.end(), local_state.points.begin(), local_state.points.end());
	local_state.points.clear();
}

unique_ptr<BoundIndex> BuildFinalize(IndexBuildFinalizeInput &input) {
	auto &global_state = input.global_state.Cast<BuildGlobalState>();
	global_state.index->Load(std::move(global_state.points));
	return std::move(global_state.index);
}

} // namespace

IndexType BallTreeIndex::MakeIndexType() {
	IndexType index_type;
	index_type.name = TYPE_NAME;
	index_type.create_instance = [](CreateIndexInput &input) -> unique_ptr<BoundIndex> {
		return make_uniq<BallTreeIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
		                                input.unbound_expressions, input.db, input.options, input.storage_info);
	};
	index_type.build_bind = BuildBind;
	index_type.build_global_init = BuildGlobalInit;
	index_type.build_local_init = BuildLocalInit;
	index_type.build_sink = BuildSink;
	index_type.build_combine = BuildCombine;
	index_type.build_finalize = BuildFinalize;
	return index_type;
}

} // namespace duckdb
