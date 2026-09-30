#include "ball_tree_index.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/index_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/table_index_list.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>

namespace duckdb {

using balltree::EARTH_RADIUS_KM;

namespace {

//===--------------------------------------------------------------------===//
// Finding the index
//===--------------------------------------------------------------------===//

BallTreeIndex &FindBoundIndex(ClientContext &context, IndexCatalogEntry &index_entry) {
	auto &table_entry = index_entry.schema.catalog.GetEntry<TableCatalogEntry>(context, index_entry.GetSchemaName(),
	                                                                           index_entry.GetTableName());
	auto &table_info = *table_entry.GetStorage().GetDataTableInfo();
	// after a restart the index is only a description until something needs it; bind it now
	table_info.BindIndexes(context, BallTreeIndex::TYPE_NAME);
	for (auto &index : table_info.GetIndexes().Indexes()) {
		if (!index.IsBound() || index.GetIndexType() != BallTreeIndex::TYPE_NAME) {
			continue;
		}
		auto &ball_tree_index = index.Cast<BallTreeIndex>();
		if (ball_tree_index.name == index_entry.name) {
			return ball_tree_index;
		}
	}
	throw BinderException("BALL_TREE index \"%s\" not found", index_entry.name);
}

BallTreeIndex &FindIndex(ClientContext &context, const string &index_name) {
	auto qname = QualifiedName::Parse(index_name);
	Binder::BindSchemaOrCatalog(context, qname.catalog, qname.schema);
	auto &index_entry = Catalog::GetEntry(context, CatalogType::INDEX_ENTRY, qname.catalog, qname.schema, qname.name)
	                        .Cast<IndexCatalogEntry>();
	if (index_entry.index_type != BallTreeIndex::TYPE_NAME) {
		throw BinderException("Index \"%s\" is a %s index, not a BALL_TREE index", index_name, index_entry.index_type);
	}
	return FindBoundIndex(context, index_entry);
}

//===--------------------------------------------------------------------===//
// The query, its arguments and its result
//===--------------------------------------------------------------------===//

enum class QueryKind : uint8_t { DEGREE, WITHIN, NEAREST };

//! What was asked. The answer is computed each time the query executes, not here: a prepared statement binds
//! once but executes many times, and the table (hence the index) can change in between.
struct IndexQueryBindData : public FunctionData {
	QueryKind kind = QueryKind::DEGREE;
	string index_name;
	//! query point in radians (WITHIN, NEAREST)
	double point[2] = {0, 0};
	//! DEGREE, WITHIN
	double radius_km = 0;
	//! NEAREST
	int64_t k = 0;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<IndexQueryBindData>();
		copy->kind = kind;
		copy->index_name = index_name;
		copy->point[0] = point[0];
		copy->point[1] = point[1];
		copy->radius_km = radius_km;
		copy->k = k;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<IndexQueryBindData>();
		return kind == other.kind && index_name == other.index_name && point[0] == other.point[0] &&
		       point[1] == other.point[1] && radius_km == other.radius_km && k == other.k;
	}
};

//! The answer, as parallel columns. DEGREE fills `counts`; the others fill `distances_km`.
struct IndexQueryState : public GlobalTableFunctionState {
	vector<int64_t> row_ids;
	vector<int64_t> counts;
	vector<double> distances_km;
	idx_t offset = 0;
};

string IndexNameArg(TableFunctionBindInput &input) {
	if (input.inputs[0].IsNull()) {
		throw InvalidInputException("ball_index: index name must not be NULL");
	}
	return input.inputs[0].GetValue<string>();
}

double RadiusArg(const Value &value) {
	if (value.IsNull()) {
		throw InvalidInputException("ball_index: radius_km must not be NULL");
	}
	const double radius_km = value.GetValue<double>();
	if (!(radius_km >= 0)) {
		throw InvalidInputException("ball_index: radius_km must be >= 0");
	}
	return radius_km;
}

//! Reads a (lat, lon) pair of degrees from arguments [1] and [2] and stores it in radians.
void QueryPointArgs(TableFunctionBindInput &input, IndexQueryBindData &data) {
	if (input.inputs[1].IsNull() || input.inputs[2].IsNull()) {
		throw InvalidInputException("ball_index: latitude and longitude must not be NULL");
	}
	const double lat = input.inputs[1].GetValue<double>();
	const double lon = input.inputs[2].GetValue<double>();
	if (!std::isfinite(lat) || !std::isfinite(lon) || lat < -90.0 || lat > 90.0) {
		throw InvalidInputException("ball_index: invalid query point (latitude %f, longitude %f)", lat, lon);
	}
	data.point[0] = lat * balltree::PI / 180.0;
	data.point[1] = lon * balltree::PI / 180.0;
}

// ball_index_degree(index, radius_km) -> (row_id, degree)
unique_ptr<FunctionData> DegreeBind(ClientContext &context, TableFunctionBindInput &input,
                                    vector<LogicalType> &return_types, vector<string> &names) {
	auto data = make_uniq<IndexQueryBindData>();
	data->kind = QueryKind::DEGREE;
	data->index_name = IndexNameArg(input);
	data->radius_km = RadiusArg(input.inputs[1]);
	FindIndex(context, data->index_name); // fail early if there is no such index
	return_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	names = {"row_id", "degree"};
	return std::move(data);
}

// ball_index_within(index, lat, lon, radius_km) -> (row_id, distance_km)
unique_ptr<FunctionData> WithinBind(ClientContext &context, TableFunctionBindInput &input,
                                    vector<LogicalType> &return_types, vector<string> &names) {
	auto data = make_uniq<IndexQueryBindData>();
	data->kind = QueryKind::WITHIN;
	data->index_name = IndexNameArg(input);
	QueryPointArgs(input, *data);
	data->radius_km = RadiusArg(input.inputs[3]);
	FindIndex(context, data->index_name);
	return_types = {LogicalType::BIGINT, LogicalType::DOUBLE};
	names = {"row_id", "distance_km"};
	return std::move(data);
}

// ball_index_nearest(index, lat, lon, k) -> (row_id, distance_km)
unique_ptr<FunctionData> NearestBind(ClientContext &context, TableFunctionBindInput &input,
                                     vector<LogicalType> &return_types, vector<string> &names) {
	auto data = make_uniq<IndexQueryBindData>();
	data->kind = QueryKind::NEAREST;
	data->index_name = IndexNameArg(input);
	QueryPointArgs(input, *data);
	if (input.inputs[3].IsNull()) {
		throw InvalidInputException("ball_index: k must not be NULL");
	}
	data->k = input.inputs[3].GetValue<int64_t>();
	if (data->k < 0) {
		throw InvalidInputException("ball_index: k must be >= 0");
	}
	FindIndex(context, data->index_name);
	return_types = {LogicalType::BIGINT, LogicalType::DOUBLE};
	names = {"row_id", "distance_km"};
	return std::move(data);
}

//! Runs the query against the index as it is now (rebuilding it first if it is stale).
unique_ptr<GlobalTableFunctionState> QueryInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &data = input.bind_data->Cast<IndexQueryBindData>();
	auto state = make_uniq<IndexQueryState>();
	auto snapshot = FindIndex(context, data.index_name).GetSnapshot();
	if (!snapshot->tree) {
		return std::move(state); // no points
	}
	auto &tree = *snapshot->tree;

	if (data.kind == QueryKind::DEGREE) {
		const auto n_threads = static_cast<size_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
		auto counts = tree.CountRadiusAll(data.radius_km / EARTH_RADIUS_KM, n_threads);
		state->row_ids.assign(snapshot->row_ids.begin(), snapshot->row_ids.end());
		state->counts.resize(counts.size());
		for (idx_t i = 0; i < counts.size(); i++) {
			state->counts[i] = counts[i] - 1; // a point is not its own neighbour
		}
		return std::move(state);
	}

	vector<std::pair<double, int64_t>> neighbours; // (angular distance, row id)
	if (data.kind == QueryKind::WITHIN) {
		vector<int64_t> hits;
		vector<double> distances;
		tree.QueryRadius(data.point, data.radius_km / EARTH_RADIUS_KM, hits, &distances);
		for (idx_t i = 0; i < hits.size(); i++) {
			neighbours.emplace_back(distances[i], snapshot->row_ids[hits[i]]);
		}
	} else {
		for (auto &hit : tree.QueryKnn(data.point, static_cast<size_t>(data.k))) {
			neighbours.emplace_back(hit.second, snapshot->row_ids[hit.first]);
		}
	}
	std::sort(neighbours.begin(), neighbours.end()); // nearest first, ties by row id
	for (auto &neighbour : neighbours) {
		state->row_ids.push_back(neighbour.second);
		state->distances_km.push_back(neighbour.first * EARTH_RADIUS_KM);
	}
	return std::move(state);
}

//! Points the index will hold once pending changes are folded in. Never triggers a rebuild.
idx_t EstimatedPoints(ClientContext &context, const string &index_name) {
	const auto stats = FindIndex(context, index_name).GetStats();
	return stats.tree_points + stats.pending_inserts > stats.pending_deletes
	           ? stats.tree_points + stats.pending_inserts - stats.pending_deletes
	           : 0;
}

//! Row estimates for the planner. Without them it has to guess, and the documented pattern
//! (`ball_index_within(...) w JOIN big_table ON big_table.rowid = w.row_id`) could end up hashing the big table
//! instead of the few rows the function returns.
unique_ptr<NodeStatistics> QueryCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &data = bind_data_p->Cast<IndexQueryBindData>();
	const idx_t points = EstimatedPoints(context, data.index_name);
	switch (data.kind) {
	case QueryKind::DEGREE:
		return make_uniq<NodeStatistics>(points, points); // one row per point
	case QueryKind::NEAREST: {
		const idx_t rows = MinValue<idx_t>(points, static_cast<idx_t>(data.k));
		return make_uniq<NodeStatistics>(rows, rows);
	}
	default:
		// a radius query is usually selective; guess low so the function side is the one that gets hashed
		return make_uniq<NodeStatistics>(MinValue<idx_t>(points, MaxValue<idx_t>(1, points / 1000)), points);
	}
}

void DegreeScan(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<IndexQueryState>();
	const idx_t count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.row_ids.size() - state.offset);
	auto row_ids = FlatVector::GetData<int64_t>(output.data[0]);
	auto degrees = FlatVector::GetData<int64_t>(output.data[1]);
	for (idx_t i = 0; i < count; i++) {
		row_ids[i] = state.row_ids[state.offset + i];
		degrees[i] = state.counts[state.offset + i];
	}
	state.offset += count;
	output.SetCardinality(count);
}

void DistanceScan(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<IndexQueryState>();
	const idx_t count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.row_ids.size() - state.offset);
	auto row_ids = FlatVector::GetData<int64_t>(output.data[0]);
	auto distances = FlatVector::GetData<double>(output.data[1]);
	for (idx_t i = 0; i < count; i++) {
		row_ids[i] = state.row_ids[state.offset + i];
		distances[i] = state.distances_km[state.offset + i];
	}
	state.offset += count;
	output.SetCardinality(count);
}

//===--------------------------------------------------------------------===//
// ball_index_pairs(index, radius_km [, symmetric := true]) -> (row_id_a, row_id_b, distance_km)
//
// Every pair of distinct indexed points within the radius. Unlike the functions above, the result can be far
// larger than the table, so it is streamed: threads claim work from a shared cursor and emit it a chunk at a
// time, holding a bounded amount of it each. Output order is unspecified.
//
// Two algorithms give the same pairs:
//   * 'dual' (default): a dual-tree self-join. The top of the traversal is split into independent node pairs,
//     which threads claim one at a time and walk with an explicit stack that pauses when the chunk is full.
//     Each unordered pair is found once.
//   * 'single': one radius query per point. Threads claim batches of points in tree order (for locality).
//     Each unordered pair is found twice, once from each end.
// The hidden `algorithm := 'single'|'dual'` parameter picks one, for testing and benchmarking.
//===--------------------------------------------------------------------===//

enum class PairsAlgorithm : uint8_t { SINGLE, DUAL };

struct PairsBindData : public FunctionData {
	string index_name;
	double radius_km = 0;
	//! emit (a, b) and (b, a); otherwise each pair once, with row_id_a < row_id_b
	bool symmetric = true;
	PairsAlgorithm algorithm = PairsAlgorithm::DUAL;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<PairsBindData>();
		copy->index_name = index_name;
		copy->radius_km = radius_km;
		copy->symmetric = symmetric;
		copy->algorithm = algorithm;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<PairsBindData>();
		return index_name == other.index_name && radius_km == other.radius_km && symmetric == other.symmetric &&
		       algorithm == other.algorithm;
	}
};

//! Points a thread claims from the shared cursor at a time ('single').
constexpr idx_t PAIRS_BATCH = 256;
//! Node pairs to split the dual-tree traversal into, per thread, so a thread that finishes early can take more.
constexpr idx_t PAIRS_TASKS_PER_THREAD = 32;

struct PairsGlobalState : public GlobalTableFunctionState {
	//! the tree as of the start of the scan, shared by every thread; immutable
	std::shared_ptr<const BallTreeSnapshot> snapshot;
	double radius = 0; // angular
	bool symmetric = true;
	PairsAlgorithm algorithm = PairsAlgorithm::DUAL;
	//! 'dual': the independent parts of the traversal
	vector<balltree::BallTree::NodePair> tasks;
	//! next unclaimed tree position ('single') or task ('dual')
	std::atomic<idx_t> next {0};
	idx_t max_threads = 1;

	idx_t MaxThreads() const override {
		return max_threads;
	}
};

struct PairsLocalState : public LocalTableFunctionState {
	// 'single'
	//! the claimed batch of tree positions not yet queried: [pos, end)
	idx_t pos = 0;
	idx_t end = 0;
	//! the point being emitted and its neighbours (tree ids, angular distances); `emitted` of them are done
	row_t row_a = 0;
	vector<int64_t> hits;
	vector<double> distances;
	idx_t emitted = 0;

	// 'dual'
	unique_ptr<balltree::BallTree::SelfJoin> join;
	//! `join` is part way through a task
	bool in_task = false;
	//! pairs from `join`: tree ids and angular distances
	vector<int64_t> ids_a;
	vector<int64_t> ids_b;
	vector<double> pair_distances;
};

unique_ptr<FunctionData> PairsBind(ClientContext &context, TableFunctionBindInput &input,
                                   vector<LogicalType> &return_types, vector<string> &names) {
	auto data = make_uniq<PairsBindData>();
	data->index_name = IndexNameArg(input);
	data->radius_km = RadiusArg(input.inputs[1]);
	auto symmetric = input.named_parameters.find("symmetric");
	if (symmetric != input.named_parameters.end()) {
		if (symmetric->second.IsNull()) {
			throw InvalidInputException("ball_index: symmetric must not be NULL");
		}
		data->symmetric = symmetric->second.GetValue<bool>();
	}
	auto algorithm = input.named_parameters.find("algorithm");
	if (algorithm != input.named_parameters.end()) {
		const string name = algorithm->second.IsNull() ? "NULL" : algorithm->second.GetValue<string>();
		if (StringUtil::CIEquals(name, "single")) {
			data->algorithm = PairsAlgorithm::SINGLE;
		} else if (StringUtil::CIEquals(name, "dual")) {
			data->algorithm = PairsAlgorithm::DUAL;
		} else {
			throw InvalidInputException("ball_index: algorithm must be 'single' or 'dual', got %s", name);
		}
	}
	FindIndex(context, data->index_name);
	return_types = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::DOUBLE};
	names = {"row_id_a", "row_id_b", "distance_km"};
	return std::move(data);
}

unique_ptr<GlobalTableFunctionState> PairsInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &data = input.bind_data->Cast<PairsBindData>();
	auto state = make_uniq<PairsGlobalState>();
	state->snapshot = FindIndex(context, data.index_name).GetSnapshot();
	state->radius = data.radius_km / EARTH_RADIUS_KM;
	state->symmetric = data.symmetric;
	state->algorithm = data.algorithm;
	const idx_t n_threads = NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
	if (data.algorithm == PairsAlgorithm::SINGLE) {
		state->max_threads = MinValue<idx_t>(n_threads, state->snapshot->Count() / PAIRS_BATCH + 1);
	} else if (state->snapshot->tree) {
		state->tasks = state->snapshot->tree->SelfJoinTasks(state->radius, n_threads * PAIRS_TASKS_PER_THREAD);
		state->max_threads = MinValue<idx_t>(n_threads, state->tasks.size());
	}
	state->max_threads = MaxValue<idx_t>(1, state->max_threads);
	return std::move(state);
}

unique_ptr<LocalTableFunctionState> PairsInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                   GlobalTableFunctionState *global_p) {
	auto &global = global_p->Cast<PairsGlobalState>();
	auto state = make_uniq<PairsLocalState>();
	if (global.algorithm == PairsAlgorithm::DUAL && global.snapshot->tree) {
		state->join = make_uniq<balltree::BallTree::SelfJoin>(*global.snapshot->tree, global.radius);
		state->ids_a.resize(STANDARD_VECTOR_SIZE);
		state->ids_b.resize(STANDARD_VECTOR_SIZE);
		state->pair_distances.resize(STANDARD_VECTOR_SIZE);
	}
	return std::move(state);
}

//! 'single': writes up to `capacity` rows from the neighbours of this thread's points.
idx_t PairsScanSingle(PairsGlobalState &global, PairsLocalState &local, int64_t *row_ids_a, int64_t *row_ids_b,
                      double *distances, idx_t capacity) {
	auto &snapshot = *global.snapshot;
	idx_t count = 0;
	while (count < capacity) {
		if (local.emitted < local.hits.size()) {
			// the current point's neighbours
			const row_t row_b = snapshot.row_ids[local.hits[local.emitted]];
			if (global.symmetric || local.row_a < row_b) {
				row_ids_a[count] = local.row_a;
				row_ids_b[count] = row_b;
				distances[count] = local.distances[local.emitted] * EARTH_RADIUS_KM;
				count++;
			}
			local.emitted++;
		} else if (local.pos < local.end) {
			// the next point of the batch
			const int64_t id = snapshot.tree->IdAt(local.pos++);
			local.row_a = snapshot.row_ids[id];
			snapshot.tree->QueryRadiusFrom(static_cast<size_t>(id), global.radius, local.hits, &local.distances);
			local.emitted = 0;
		} else {
			// a new batch
			const idx_t start = global.next.fetch_add(PAIRS_BATCH);
			if (start >= snapshot.Count()) {
				break;
			}
			local.pos = start;
			local.end = MinValue<idx_t>(start + PAIRS_BATCH, snapshot.Count());
		}
	}
	return count;
}

//! 'dual': writes up to `capacity` rows from the node pairs this thread claims.
idx_t PairsScanDual(PairsGlobalState &global, PairsLocalState &local, int64_t *row_ids_a, int64_t *row_ids_b,
                    double *distances, idx_t capacity) {
	auto &row_ids = global.snapshot->row_ids;
	idx_t count = 0;
	while (true) {
		// a symmetric pair takes two rows
		const idx_t room = global.symmetric ? (capacity - count) / 2 : capacity - count;
		if (room == 0) {
			break;
		}
		if (!local.in_task) {
			const idx_t task = global.next.fetch_add(1);
			if (task >= global.tasks.size()) {
				break;
			}
			local.join->Start(global.tasks[task]);
			local.in_task = true;
		}
		const idx_t found = local.join->Next(local.ids_a.data(), local.ids_b.data(), local.pair_distances.data(), room);
		if (found < room) {
			local.in_task = false;
		}
		for (idx_t i = 0; i < found; i++) {
			const row_t a = row_ids[local.ids_a[i]];
			const row_t b = row_ids[local.ids_b[i]];
			const double distance_km = local.pair_distances[i] * EARTH_RADIUS_KM;
			row_ids_a[count] = MinValue(a, b);
			row_ids_b[count] = MaxValue(a, b);
			distances[count++] = distance_km;
			if (global.symmetric) {
				row_ids_a[count] = MaxValue(a, b);
				row_ids_b[count] = MinValue(a, b);
				distances[count++] = distance_km;
			}
		}
	}
	return count;
}

//! Fills `output` with up to STANDARD_VECTOR_SIZE rows; returns an empty chunk only when this thread's work is done.
void PairsScan(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &global = input.global_state->Cast<PairsGlobalState>();
	auto &local = input.local_state->Cast<PairsLocalState>();
	idx_t count = 0;
	if (global.snapshot->tree) {
		auto row_ids_a = FlatVector::GetData<int64_t>(output.data[0]);
		auto row_ids_b = FlatVector::GetData<int64_t>(output.data[1]);
		auto distances = FlatVector::GetData<double>(output.data[2]);
		if (global.algorithm == PairsAlgorithm::SINGLE) {
			count = PairsScanSingle(global, local, row_ids_a, row_ids_b, distances, STANDARD_VECTOR_SIZE);
		} else {
			count = PairsScanDual(global, local, row_ids_a, row_ids_b, distances, STANDARD_VECTOR_SIZE);
		}
	}
	output.SetCardinality(count);
}

//! Pairs are rarely known before running: assume a handful of neighbours per point. An estimate above the table
//! size keeps the planner hashing the table, not the pairs, when both sides are joined back to it.
unique_ptr<NodeStatistics> PairsCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &data = bind_data_p->Cast<PairsBindData>();
	const idx_t points = EstimatedPoints(context, data.index_name);
	// every ordered pair, saturating (the tree allows up to 2^40 points)
	const double all_pairs = static_cast<double>(points) * static_cast<double>(points > 0 ? points - 1 : 0);
	const idx_t max_pairs = all_pairs >= static_cast<double>(NumericLimits<int64_t>::Maximum())
	                            ? NumericCast<idx_t>(NumericLimits<int64_t>::Maximum())
	                            : static_cast<idx_t>(all_pairs) / (data.symmetric ? 1 : 2);
	return make_uniq<NodeStatistics>(MinValue<idx_t>(max_pairs, points * 10), max_pairs);
}

//===--------------------------------------------------------------------===//
// ball_index_info() -> one row per BALL_TREE index, including whether it is stale
//===--------------------------------------------------------------------===//

unique_ptr<FunctionData> InfoBind(ClientContext &context, TableFunctionBindInput &input,
                                  vector<LogicalType> &return_types, vector<string> &names) {
	names = {"index_name", "table_name", "tree_points", "pending_inserts", "pending_deletes",
	         "stale",      "rebuilds",   "leaf_size",   "memory_bytes"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT,
	                LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::BOOLEAN,
	                LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::BIGINT};
	return nullptr;
}

struct InfoGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	vector<reference<IndexCatalogEntry>> entries;
};

unique_ptr<GlobalTableFunctionState> InfoInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<InfoGlobalState>();
	for (auto &schema : Catalog::GetAllSchemas(context)) {
		schema.get().Scan(context, CatalogType::INDEX_ENTRY, [&](CatalogEntry &entry) {
			auto &index_entry = entry.Cast<IndexCatalogEntry>();
			if (index_entry.index_type == BallTreeIndex::TYPE_NAME) {
				result->entries.push_back(index_entry);
			}
		});
	}
	return std::move(result);
}

void InfoScan(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<InfoGlobalState>();
	idx_t row = 0;
	while (state.offset < state.entries.size() && row < STANDARD_VECTOR_SIZE) {
		auto &index_entry = state.entries[state.offset++].get();
		const auto stats = FindBoundIndex(context, index_entry).GetStats();
		idx_t col = 0;
		output.data[col++].SetValue(row, Value(index_entry.name));
		output.data[col++].SetValue(row, Value(index_entry.GetTableName()));
		output.data[col++].SetValue(row, Value::BIGINT(static_cast<int64_t>(stats.tree_points)));
		output.data[col++].SetValue(row, Value::BIGINT(static_cast<int64_t>(stats.pending_inserts)));
		output.data[col++].SetValue(row, Value::BIGINT(static_cast<int64_t>(stats.pending_deletes)));
		output.data[col++].SetValue(row, Value::BOOLEAN(stats.stale));
		output.data[col++].SetValue(row, Value::BIGINT(static_cast<int64_t>(stats.rebuilds)));
		output.data[col++].SetValue(row, Value::BIGINT(static_cast<int64_t>(stats.leaf_size)));
		output.data[col++].SetValue(row, Value::BIGINT(static_cast<int64_t>(stats.memory_bytes)));
		row++;
	}
	output.SetCardinality(row);
}

} // namespace

void RegisterBallTreeIndex(ExtensionLoader &loader) {
	DBConfig::GetConfig(loader.GetDatabaseInstance()).GetIndexTypes().RegisterIndexType(BallTreeIndex::MakeIndexType());

	const auto V = LogicalType::VARCHAR;
	const auto B = LogicalType::BIGINT;
	const auto D = LogicalType::DOUBLE;

	TableFunction degree("ball_index_degree", {V, D}, DegreeScan, DegreeBind, QueryInit);
	degree.cardinality = QueryCardinality;
	loader.RegisterFunction(degree);

	TableFunction within("ball_index_within", {V, D, D, D}, DistanceScan, WithinBind, QueryInit);
	within.cardinality = QueryCardinality;
	loader.RegisterFunction(within);

	TableFunction nearest("ball_index_nearest", {V, D, D, B}, DistanceScan, NearestBind, QueryInit);
	nearest.cardinality = QueryCardinality;
	loader.RegisterFunction(nearest);

	TableFunction pairs("ball_index_pairs", {V, D}, PairsScan, PairsBind, PairsInitGlobal, PairsInitLocal);
	pairs.named_parameters["symmetric"] = LogicalType::BOOLEAN;
	pairs.named_parameters["algorithm"] = V;
	pairs.cardinality = PairsCardinality;
	loader.RegisterFunction(pairs);

	loader.RegisterFunction(TableFunction("ball_index_info", {}, InfoScan, InfoBind, InfoInit));
}

} // namespace duckdb
