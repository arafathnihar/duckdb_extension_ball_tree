#define DUCKDB_EXTENSION_MAIN

#include "ball_tree_extension.hpp"
#include "ball_tree.hpp"
#include "ball_tree_index.hpp"
#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/planner/binder.hpp"

#include <algorithm>
#include <cmath>

namespace duckdb {

using balltree::BallTree;
using balltree::EARTH_RADIUS_KM;

//===--------------------------------------------------------------------===//
// Reading the points
//===--------------------------------------------------------------------===//

// The points of a table, in the order they were read. The tree is rebuilt from this on every call
// ("rebuild on demand"), so there is nothing to keep in sync with the table.
struct PointSet {
	vector<int64_t> ids;
	vector<double> coords;                // [lat, lon] in radians, row-major
	unordered_map<int64_t, idx_t> row_of; // id -> first row with that id
};

static string QuotedTable(const string &table) {
	auto name = QualifiedName::Parse(table);
	string result;
	if (!name.catalog.empty()) {
		result += KeywordHelper::WriteOptionallyQuoted(name.catalog) + ".";
	}
	if (!name.schema.empty()) {
		result += KeywordHelper::WriteOptionallyQuoted(name.schema) + ".";
	}
	return result + KeywordHelper::WriteOptionallyQuoted(name.name);
}

// Reads (id, lat, lon) from `table` through a separate connection, so it only sees committed data.
// Rows with a NULL id, latitude or longitude are skipped.
static PointSet LoadPoints(ClientContext &context, const string &table, const string &id_col, const string &lat_col,
                           const string &lon_col) {
	const auto id = KeywordHelper::WriteOptionallyQuoted(id_col);
	const auto lat = KeywordHelper::WriteOptionallyQuoted(lat_col);
	const auto lon = KeywordHelper::WriteOptionallyQuoted(lon_col);
	const auto sql = StringUtil::Format("SELECT CAST(%s AS BIGINT), CAST(%s AS DOUBLE), CAST(%s AS DOUBLE) FROM %s "
	                                    "WHERE %s IS NOT NULL AND %s IS NOT NULL AND %s IS NOT NULL",
	                                    id, lat, lon, QuotedTable(table), id, lat, lon);

	Connection con(*context.db);
	auto result = con.Query(sql);
	if (result->HasError()) {
		result->ThrowError("ball_tree: could not read points: ");
	}

	PointSet points;
	while (true) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		chunk->Flatten();
		auto ids = FlatVector::GetData<int64_t>(chunk->data[0]);
		auto lats = FlatVector::GetData<double>(chunk->data[1]);
		auto lons = FlatVector::GetData<double>(chunk->data[2]);
		for (idx_t i = 0; i < chunk->size(); i++) {
			if (!std::isfinite(lats[i]) || !std::isfinite(lons[i]) || lats[i] < -90.0 || lats[i] > 90.0) {
				throw InvalidInputException("ball_tree: invalid coordinate for id %lld (latitude %f, longitude %f)",
				                            static_cast<long long>(ids[i]), lats[i], lons[i]);
			}
			points.row_of.emplace(ids[i], points.ids.size());
			points.ids.push_back(ids[i]);
			points.coords.push_back(lats[i] * balltree::PI / 180.0);
			points.coords.push_back(lons[i] * balltree::PI / 180.0);
		}
	}
	return points;
}

//===--------------------------------------------------------------------===//
// Bind helpers
//===--------------------------------------------------------------------===//

static string StringArg(TableFunctionBindInput &input, idx_t i, const char *what) {
	if (input.inputs[i].IsNull()) {
		throw InvalidInputException("ball_tree: %s must not be NULL", what);
	}
	return input.inputs[i].GetValue<string>();
}

static size_t LeafSizeArg(TableFunctionBindInput &input) {
	auto entry = input.named_parameters.find("leaf_size");
	if (entry == input.named_parameters.end()) {
		return 40;
	}
	const auto leaf_size = entry->second.GetValue<int64_t>();
	if (leaf_size < 1) {
		throw InvalidInputException("ball_tree: leaf_size must be >= 1");
	}
	return static_cast<size_t>(leaf_size);
}

static idx_t RowOf(const PointSet &points, int64_t id) {
	auto entry = points.row_of.find(id);
	if (entry == points.row_of.end()) {
		throw InvalidInputException("ball_tree: unknown id: %lld", static_cast<long long>(id));
	}
	return entry->second;
}

enum class QueryKind : uint8_t { DEGREE, WITHIN, NEAREST };

// What was asked. The answer is computed each time the query executes, not at bind time: a prepared statement
// binds once but executes many times, and the table can change in between.
struct TableQueryBindData : public FunctionData {
	QueryKind kind = QueryKind::DEGREE;
	string table, id_col, lat_col, lon_col;
	size_t leaf_size = 40;
	int64_t query_id = 0; // WITHIN, NEAREST
	double radius_km = 0; // DEGREE, WITHIN
	int64_t k = 0;        // NEAREST

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<TableQueryBindData>();
		copy->kind = kind;
		copy->table = table;
		copy->id_col = id_col;
		copy->lat_col = lat_col;
		copy->lon_col = lon_col;
		copy->leaf_size = leaf_size;
		copy->query_id = query_id;
		copy->radius_km = radius_km;
		copy->k = k;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<TableQueryBindData>();
		return kind == o.kind && table == o.table && id_col == o.id_col && lat_col == o.lat_col &&
		       lon_col == o.lon_col && leaf_size == o.leaf_size && query_id == o.query_id && radius_km == o.radius_km &&
		       k == o.k;
	}
};

// The answer, as parallel columns: DEGREE fills `counts`, WITHIN / NEAREST fill `distances_km`.
struct TableQueryState : public GlobalTableFunctionState {
	vector<int64_t> ids;
	vector<int64_t> counts;
	vector<double> distances_km;
	idx_t offset = 0;
};

static void ReadCommonArgs(TableFunctionBindInput &input, TableQueryBindData &data) {
	data.table = StringArg(input, 0, "table");
	data.id_col = StringArg(input, 1, "id_col");
	data.lat_col = StringArg(input, 2, "lat_col");
	data.lon_col = StringArg(input, 3, "lon_col");
	data.leaf_size = LeafSizeArg(input);
}

static double RadiusArg(const Value &value) {
	if (value.IsNull()) {
		throw InvalidInputException("ball_tree: radius_km must not be NULL");
	}
	const auto radius_km = value.GetValue<double>();
	if (!(radius_km >= 0)) {
		throw InvalidInputException("ball_tree: radius_km must be >= 0");
	}
	return radius_km;
}

// ball_degree(table, id_col, lat_col, lon_col, radius_km [, leaf_size := 40])
//   -> (id BIGINT, degree BIGINT): how many other points lie within radius_km of each point
static unique_ptr<FunctionData> DegreeBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	auto data = make_uniq<TableQueryBindData>();
	data->kind = QueryKind::DEGREE;
	ReadCommonArgs(input, *data);
	data->radius_km = RadiusArg(input.inputs[4]);
	return_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	names = {"id", "degree"};
	return std::move(data);
}

// ball_within(table, id_col, lat_col, lon_col, query_id, radius_km [, leaf_size := 40])
// ball_nearest(table, id_col, lat_col, lon_col, query_id, k [, leaf_size := 40])
//   -> (id BIGINT, distance_km DOUBLE), nearest first, excluding query_id itself
static unique_ptr<FunctionData> NeighboursBind(TableFunctionBindInput &input, vector<LogicalType> &return_types,
                                               vector<string> &names, QueryKind kind) {
	auto data = make_uniq<TableQueryBindData>();
	data->kind = kind;
	ReadCommonArgs(input, *data);
	if (input.inputs[4].IsNull() || input.inputs[5].IsNull()) {
		throw InvalidInputException("ball_tree: query_id and %s must not be NULL",
		                            kind == QueryKind::NEAREST ? "k" : "radius_km");
	}
	data->query_id = input.inputs[4].GetValue<int64_t>();
	if (kind == QueryKind::NEAREST) {
		data->k = input.inputs[5].GetValue<int64_t>();
		if (data->k < 0) {
			throw InvalidInputException("ball_tree: k must be >= 0");
		}
	} else {
		data->radius_km = RadiusArg(input.inputs[5]);
	}
	return_types = {LogicalType::BIGINT, LogicalType::DOUBLE};
	names = {"id", "distance_km"};
	return std::move(data);
}

static unique_ptr<FunctionData> WithinBind(ClientContext &, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	return NeighboursBind(input, return_types, names, QueryKind::WITHIN);
}

static unique_ptr<FunctionData> NearestBind(ClientContext &, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types, vector<string> &names) {
	return NeighboursBind(input, return_types, names, QueryKind::NEAREST);
}

// Reads the table and answers the query, every time it executes.
static unique_ptr<GlobalTableFunctionState> QueryInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &data = input.bind_data->Cast<TableQueryBindData>();
	auto state = make_uniq<TableQueryState>();
	auto points = LoadPoints(context, data.table, data.id_col, data.lat_col, data.lon_col);

	if (data.kind == QueryKind::DEGREE) {
		if (!points.ids.empty()) {
			BallTree tree(points.coords.data(), points.ids.size(), data.leaf_size);
			const auto n_threads = static_cast<size_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
			auto counts = tree.CountRadiusAll(data.radius_km / EARTH_RADIUS_KM, n_threads);
			for (auto &count : counts) {
				count -= 1; // a point is not its own neighbour
			}
			state->counts = std::move(counts);
			state->ids = std::move(points.ids);
		}
		return std::move(state);
	}

	const auto query_row = RowOf(points, data.query_id);
	BallTree tree(points.coords.data(), points.ids.size(), data.leaf_size);
	const double *query = tree.Point(static_cast<int64_t>(query_row));

	vector<pair<double, int64_t>> neighbours; // (angular distance, row)
	if (data.kind == QueryKind::NEAREST) {
		// k + 1 because the query point comes back as its own nearest neighbour
		for (auto &hit : tree.QueryKnn(query, static_cast<size_t>(data.k) + 1)) {
			neighbours.emplace_back(hit.second, hit.first);
		}
	} else {
		vector<int64_t> rows;
		vector<double> dists;
		tree.QueryRadius(query, data.radius_km / EARTH_RADIUS_KM, rows, &dists);
		for (idx_t i = 0; i < rows.size(); i++) {
			neighbours.emplace_back(dists[i], rows[i]);
		}
	}
	neighbours.erase(std::remove_if(neighbours.begin(), neighbours.end(),
	                                [&](const pair<double, int64_t> &n) { return n.second == int64_t(query_row); }),
	                 neighbours.end());
	for (auto &n : neighbours) {
		n.second = points.ids[n.second]; // row -> id
	}
	// Nearest first; ties by id so the output is deterministic.
	std::sort(neighbours.begin(), neighbours.end());
	if (data.kind == QueryKind::NEAREST) {
		neighbours.resize(std::min<size_t>(neighbours.size(), static_cast<size_t>(data.k)));
	}
	for (auto &n : neighbours) {
		state->ids.push_back(n.second);
		state->distances_km.push_back(n.first * EARTH_RADIUS_KM);
	}
	return std::move(state);
}

// Row estimates for the planner; without them it has to guess how these results compare with other inputs of a
// join. The table's row count is cheap to get and is an upper bound on the number of points.
static unique_ptr<NodeStatistics> QueryCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &data = bind_data_p->Cast<TableQueryBindData>();
	idx_t rows;
	try {
		auto qname = QualifiedName::Parse(data.table);
		Binder::BindSchemaOrCatalog(context, qname.catalog, qname.schema);
		auto &entry = Catalog::GetEntry<TableCatalogEntry>(context, qname.catalog, qname.schema, qname.name);
		if (!entry.IsDuckTable()) {
			return nullptr;
		}
		rows = entry.GetStorage().GetTotalRows();
	} catch (std::exception &) {
		return nullptr; // e.g. no such table: the query reports that when it runs
	}
	switch (data.kind) {
	case QueryKind::DEGREE:
		return make_uniq<NodeStatistics>(rows, rows); // one row per point
	case QueryKind::NEAREST: {
		const idx_t estimate = MinValue<idx_t>(rows, static_cast<idx_t>(data.k));
		return make_uniq<NodeStatistics>(estimate, estimate);
	}
	default:
		// a radius query is usually selective; guess low so the function side is the one that gets hashed
		return make_uniq<NodeStatistics>(MinValue<idx_t>(rows, MaxValue<idx_t>(1, rows / 1000)), rows);
	}
}

static void DegreeScan(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<TableQueryState>();
	const idx_t count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.ids.size() - state.offset);
	auto ids = FlatVector::GetData<int64_t>(output.data[0]);
	auto degrees = FlatVector::GetData<int64_t>(output.data[1]);
	for (idx_t i = 0; i < count; i++) {
		ids[i] = state.ids[state.offset + i];
		degrees[i] = state.counts[state.offset + i];
	}
	state.offset += count;
	output.SetCardinality(count);
}

static void NeighboursScan(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<TableQueryState>();
	const idx_t count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.ids.size() - state.offset);
	auto ids = FlatVector::GetData<int64_t>(output.data[0]);
	auto dists = FlatVector::GetData<double>(output.data[1]);
	for (idx_t i = 0; i < count; i++) {
		ids[i] = state.ids[state.offset + i];
		dists[i] = state.distances_km[state.offset + i];
	}
	state.offset += count;
	output.SetCardinality(count);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//

static void LoadInternal(ExtensionLoader &loader) {
	const auto V = LogicalType::VARCHAR;
	const auto B = LogicalType::BIGINT;
	const auto D = LogicalType::DOUBLE;

	TableFunction degree("ball_degree", {V, V, V, V, D}, DegreeScan, DegreeBind, QueryInit);
	degree.named_parameters["leaf_size"] = B;
	degree.cardinality = QueryCardinality;
	loader.RegisterFunction(degree);

	TableFunction within("ball_within", {V, V, V, V, B, D}, NeighboursScan, WithinBind, QueryInit);
	within.named_parameters["leaf_size"] = B;
	within.cardinality = QueryCardinality;
	loader.RegisterFunction(within);

	TableFunction nearest("ball_nearest", {V, V, V, V, B, B}, NeighboursScan, NearestBind, QueryInit);
	nearest.named_parameters["leaf_size"] = B;
	nearest.cardinality = QueryCardinality;
	loader.RegisterFunction(nearest);

	RegisterBallTreeIndex(loader);
}

void BallTreeExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string BallTreeExtension::Name() {
	return "ball_tree";
}

std::string BallTreeExtension::Version() const {
#ifdef EXT_VERSION_BALL_TREE
	return EXT_VERSION_BALL_TREE;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(ball_tree, loader) {
	duckdb::LoadInternal(loader);
}
}
