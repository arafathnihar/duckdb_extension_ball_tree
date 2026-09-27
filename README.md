# ball_tree

A DuckDB extension for great-circle (haversine) neighbour queries over a table of latitude/longitude points, backed by a ball tree. It offers a **persistent index** (`CREATE INDEX ... USING BALL_TREE`) and, for one-off use, functions that work straight from a table.

The tree is a C++ port of scikit-learn's `BallTree` (`sklearn/neighbors/_ball_tree.pyx.tp` and `_binary_tree.pxi.tp`) specialised to the haversine metric. The implementation is `src/include/ball_tree.hpp`, which has no DuckDB dependency.

Based on [duckdb/extension-template](https://github.com/duckdb/extension-template).

Jump to a [worked example](#example-good-sites-near-a-place) with real data.

## The index

```sql
LOAD ball_tree;

CREATE INDEX pts_idx ON pts USING BALL_TREE (latitude, longitude);
-- optional: WITH (leaf_size = 40)
```

The two columns are latitude and longitude in **degrees** and must be `DOUBLE` or `FLOAT`. For other types, cast in the index expressions (the extra parentheses are required by DuckDB's syntax):

```sql
CREATE INDEX pts_idx ON pts USING BALL_TREE ((latitude::DOUBLE), (longitude::DOUBLE));
```

Rows where either coordinate is NULL are not indexed. A latitude outside [-90, 90] (or a non-finite value) is rejected when the row is inserted.

### Querying

The functions take the **index name** and return the matching **row ids**; join back to the table with `rowid`. Distances are in kilometres.

| Function | Returns |
|---|---|
| `ball_index_within(index, lat, lon, radius_km)` | `(row_id, distance_km)` for every point within `radius_km` of `(lat, lon)`, nearest first |
| `ball_index_nearest(index, lat, lon, k)` | `(row_id, distance_km)` for the `k` closest points, nearest first |
| `ball_index_degree(index, radius_km)` | `(row_id, degree)` for every indexed point: how many other points are within `radius_km` |
| `ball_index_info()` | one row per BALL_TREE index: point count, pending changes, whether it is stale |

```sql
-- the point with the most neighbours within 50 km
SELECT p.*, d.degree
FROM ball_index_degree('pts_idx', 50) d JOIN pts p ON p.rowid = d.row_id
ORDER BY d.degree DESC LIMIT 1;

-- everything within 5 km of a location
SELECT p.id, w.distance_km
FROM ball_index_within('pts_idx', 40.88, -72.46, 5) w JOIN pts p ON p.rowid = w.row_id;

-- its 3 nearest points
SELECT p.id, w.distance_km
FROM ball_index_nearest('pts_idx', 40.88, -72.46, 3) w JOIN pts p ON p.rowid = w.row_id;
```

The query point and radius (or `k`) must be constants; lateral use (`FROM pts p, ball_index_within('i', p.lat, p.lon, 5)`) is not supported.

The answer is computed each time the query executes, so a prepared statement always reflects the table as it is now. The functions also give the planner row estimates, so in `ball_index_within(...) w JOIN pts p ON p.rowid = w.row_id` it hashes the few rows the function returns rather than the table.

### When the index is invalidated

DuckDB tells the index about every change to the table. The index reacts as follows:

| Change to the table | Effect on the index |
|---|---|
| `INSERT` of a row with coordinates | stale |
| `DELETE` of an indexed row | stale |
| `UPDATE` of the latitude or longitude | stale (it is a delete plus an insert) |
| `UPDATE` of any other column | nothing |
| `INSERT` / `DELETE` of a row with a NULL coordinate | nothing (it was never indexed) |
| a rolled back transaction | nothing |

"Stale" means the tree no longer matches the table. The tree itself is immutable: changes are recorded as a small list of added and removed rows. The **next query rebuilds the tree** from the old tree plus that list, and the index is fresh again. A rebuild never re-reads the table, and never happens while the index is fresh. So a burst of changes costs one rebuild, at the first query after it.

**A rebuild costs the same however few rows changed**: the tree is rebuilt as a whole, about 0.4 s at 1M points whether 1, 1,000 or 100,000 rows changed (see Performance). So the index suits data that changes in batches. If a single insert is followed by a query, over and over, every query pays a full rebuild.

`ball_index_info()` shows this:

```sql
SELECT index_name, tree_points, pending_inserts, pending_deletes, stale, rebuilds FROM ball_index_info();
```

`rebuilds` counts rebuilds since the database was opened.

### Persistence

The tree is stored in the database file at every checkpoint, and read back as it is when the database opens: **no rebuild on load**. If the index is stale at a checkpoint, it is rebuilt first, so the file always holds a fresh tree. Nothing is rewritten if the index did not change since the last checkpoint.

The index is also recorded in the write-ahead log, so an index created (or changed) after the last checkpoint survives a crash: the changes are replayed on startup, which leaves the index stale until the first query rebuilds it.

Note that DuckDB checkpoints automatically when the WAL grows past `checkpoint_threshold` (16 MB by default), and on a clean shutdown, so in practice a stale index is usually folded in by a checkpoint.

### Things to know

- **Transactions:** an insert is not visible to index queries until it commits, not even in the transaction that made it. (`SELECT ... FROM pts` sees it, `ball_index_*` does not.)
- **Cost:** about 42 bytes per point in memory and about 34 in the database file.
- **Unique indexes** and multi-column keys other than (latitude, longitude) are not supported.
- **`MergeIndexes`** is not implemented. DuckDB did not call it in any of the tests (single rows, 400k-row batches, 3M-row batches), but a load path that needs it would fail with "not implemented".
- The stored format is native-endian and versioned; it is meant to be read by the same kind of machine that wrote it.

## Example: good sites near a place

`site_quality_ranking.parquet` (a private dataset, not included) has 98,311 sites, each with `latitude`, `longitude`, a `quality_score`, a `grade` (A to F), an overall `rank` and a `state`. Run these from the directory that holds the file.

```sql
CREATE TABLE sites AS SELECT * FROM 'site_quality_ranking.parquet';
CREATE INDEX sites_idx ON sites USING BALL_TREE (latitude, longitude);
```

The columns are already `DOUBLE`, so no cast is needed. Building the index takes about 60 ms.

**Best-ranked sites within 25 km of Boston.** The function returns row ids and distances; join back to the table for everything else:

```sql
SELECT s.id, s.state, s.grade, s.rank, round(w.distance_km, 1) AS km
FROM ball_index_within('sites_idx', 42.36, -71.06, 25) w
JOIN sites s ON s.rowid = w.row_id
ORDER BY s.rank LIMIT 5;
```
```
id        state          grade  rank  km
12276242  Massachusetts  A      2     22.0
24611021  Massachusetts  A      18    15.7
38208283  Massachusetts  A      54    12.5
34676300  Massachusetts  A      118   12.4
36398569  Massachusetts  A      136   22.9
```

There are 2,476 sites within 25 km in total, which matches a brute-force SQL haversine scan.

**The nearest grade-A sites to Denver.** `ball_index_nearest` returns the nearest points of any grade, so to filter first, search a radius and filter the result:

```sql
SELECT s.id, s.state, s.quality_score, round(w.distance_km, 1) AS km
FROM ball_index_within('sites_idx', 39.74, -104.99, 200) w
JOIN sites s ON s.rowid = w.row_id
WHERE s.grade = 'A'
ORDER BY w.distance_km LIMIT 5;
```
```
id        state     quality_score  km
4303099   Colorado  94.4           21.3
42384103  Colorado  90.4           24.6
38404576  Colorado  91.5           26.1
36142896  Colorado  97.8           32.5
36213596  Colorado  99.0           43.1
```

**How crowded is each top-ranked site?** `ball_index_degree` counts, for every site, how many other sites are within a radius:

```sql
SELECT s.rank, s.id, s.state, d.degree AS sites_within_25km
FROM ball_index_degree('sites_idx', 25) d
JOIN sites s ON s.rowid = d.row_id
ORDER BY s.rank LIMIT 5;
```
```
rank  id        state          sites_within_25km
1     26164869  Tennessee      5
2     12276242  Massachusetts  2921
3     13334326  New York       2118
4     14719016  Massachusetts  2291
5     6209391   Massachusetts  2193
```

**Changing the data.** The index follows the table. Only changes to the coordinates matter:

```sql
SELECT stale, tree_points, pending_inserts FROM ball_index_info();
-- false  98311  0

INSERT INTO sites (id, state, latitude, longitude, quality_score, grade, rank)
VALUES (999999999, 'Massachusetts', 42.36, -71.06, 95.0, 'A', 1);
SELECT stale, tree_points, pending_inserts FROM ball_index_info();
-- true   98311  1        the index knows it is out of date

SELECT count(*) FROM ball_index_within('sites_idx', 42.36, -71.06, 25);
-- 2477                   this query rebuilt the tree first; it now includes the new site

UPDATE sites SET grade = 'B' WHERE id = 999999999;      -- not an indexed column: stays fresh
UPDATE sites SET latitude = 42.40 WHERE id = 999999999; -- an indexed column: stale again
```

## Functions that need no index

These read the table on every call, build a tree, answer, and throw the tree away. Use them for one-off questions. Arguments name a table and its id, latitude and longitude columns; the id must be castable to `BIGINT`. Rows with a NULL id, latitude or longitude are skipped. They only see committed data.

| Function | Returns |
|---|---|
| `ball_degree(table, id_col, lat_col, lon_col, radius_km)` | `(id, degree)` for every point |
| `ball_within(table, id_col, lat_col, lon_col, query_id, radius_km)` | `(id, distance_km)` for all points within `radius_km` of `query_id`, nearest first |
| `ball_nearest(table, id_col, lat_col, lon_col, query_id, k)` | `(id, distance_km)` for the `k` closest points to `query_id`, nearest first |

`ball_within` and `ball_nearest` never return `query_id` itself. Each takes an optional `leaf_size := 40`.

```sql
SELECT * FROM ball_degree('pts', 'id', 'latitude', 'longitude', 50) ORDER BY degree DESC LIMIT 1;
```

## Performance

Measured on an Intel i7-12700 (20 threads), median of 5 runs, on 99,907 real points (and, for the larger tables, those points resampled with about 1 km of jitter). Every method returns the same matches.

**Neighbour count of every point** (99,907 points):

| radius | scikit-learn (build + count) | extension, from-scratch function | extension, index |
|---|---|---|---|
| 1 km | 0.98 s | 116 ms | 77 ms |
| 50 km | 2.9 s | 278 ms | 253 ms |
| 250 km | 3.4 s | 343 ms | 366 ms |

The index and the from-scratch function are close here: the index only saves reading the table and building the tree (about 40 ms at this size). Both beat scikit-learn by 8-13x, mostly because they use all cores.

**One query, everything within 5 km**, per query, including the join back to the table:

| table size | index | plain SQL scan | from-scratch function | scikit-learn, tree in memory (no SQL) |
|---|---|---|---|---|
| 10k | 0.08 ms | 0.26 ms | 2.7 ms | 0.047 ms |
| 99,907 | 0.11 ms | 0.82 ms | 33 ms | 0.085 ms |
| 1M | 0.29 ms | 7.1 ms | 425 ms | 0.47 ms |

This is where the index pays off: the scan grows linearly with the table (24x slower than the index at 1M points), and the from-scratch function rebuilds a tree on every call.

**Keeping an index up to date** (1M points):

| | time |
|---|---|
| `CREATE INDEX` (build, and write to the file) | 0.52 s |
| reload from the file in a new process | 72 ms |
| query on an up-to-date index | about 1 ms |
| first query after 1 insert | 0.39 s |
| first query after 1,000 inserts | 0.40 s |
| first query after 100,000 inserts | 0.41 s |

Reloading is 7x faster than building. But the first query after *any* change costs a full rebuild, whatever the number of changes. A possible improvement is to scan a small pending change list directly at query time, and rebuild only once it grows past a threshold; it is not implemented.

## Building

No third-party dependencies (no vcpkg). Needs `git`, `cmake`, a C++17 compiler and `ninja` (or `make`).

```sh
GEN=ninja make            # release build
```

Outputs:

```
./build/release/duckdb
./build/release/test/unittest
./build/release/extension/ball_tree/ball_tree.duckdb_extension
```

## Testing

```sh
make test
```

The SQL tests are in `test/sql/`. `ball_tree_index.test` covers the queries, every invalidation rule above, persistence across restarts, recovery from the write-ahead log, expression indexes, and bad input. The tree itself was checked against scikit-learn on 99,907 real points: radius counts at 5, 50 and 500 km match exactly, and k-nearest distances match to within 5e-13 rad.

The test harness checkpoints after every commit; the index tests turn that off (`PRAGMA wal_autocheckpoint`) where they need to observe a stale index.

## Running

```sh
./build/release/duckdb mydb.duckdb
```

The extension is statically linked into that shell. To load it into another DuckDB build, use `LOAD 'path/to/ball_tree.duckdb_extension'` (an unsigned extension needs `-unsigned` or `allow_unsigned_extensions`, and the DuckDB version must match the one it was built against, v1.5.5).

## Publishing as a community extension

This repository follows [duckdb/extension-template](https://github.com/duckdb/extension-template), so it builds with DuckDB's CI toolchain (`.github/workflows/MainDistributionPipeline.yml`). To publish it as a [community extension](https://duckdb.org/community_extensions/development):

1. Push this repository to a public GitHub repo and check that the *Main Extension Distribution Pipeline* workflow passes.
2. In `description.yml`, set `ref` (`COMMIT_SHA`) to the commit to build.
3. Fork [duckdb/community-extensions](https://github.com/duckdb/community-extensions), copy the file to `extensions/ball_tree/description.yml`, and open a pull request.

Once merged, anyone can run:

```sql
INSTALL ball_tree FROM community;
LOAD ball_tree;
```

Community extensions are built only for the latest stable DuckDB release; see `docs/UPDATING.md` for bumping the `duckdb` and `extension-ci-tools` submodules and the workflow versions.
