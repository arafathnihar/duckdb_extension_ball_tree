"""Benchmark of the ball_tree extension against scikit-learn's BallTree and a plain SQL scan.

Three experiments, all written to results.csv:

  degree     every point's neighbour count within a radius, on all 99,907 real points
             (scikit-learn: build + count; extension functions on a table: read + build + count;
              extension index: count only, the index is already built)
  point      one "what is within 5 km of here?" query, against tables of 10k / 99,907 / 1M points
             (index, from-scratch function, brute-force SQL scan, scikit-learn with the tree in memory)
  lifecycle  what an index costs to build, reload from the database file, and bring up to date after
             changes, on the 1M-point table

    BALL_TREE_BENCH_DATA=/path/to/points.parquet python bench.py

Needs numpy, pandas and scikit-learn. The parquet file must have latitude and longitude columns.
"""
import csv
import os
import re
import statistics
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

HERE = os.path.dirname(os.path.abspath(__file__))
DUCKDB = os.path.join(HERE, "..", "build", "release", "duckdb")
SRC = os.environ.get("BALL_TREE_BENCH_DATA", os.path.join(HERE, "points.parquet"))
EARTH_KM = 6371.0088
REPEATS = 5
ALL_THREADS = os.cpu_count()

DEGREE_RADII_KM = [1, 5, 10, 25, 50, 100, 250]
POINT_RADIUS_KM = 5
POINT_SIZES = [10_000, 99_907, 1_000_000]
N_QUERY_POINTS = 100  # queries per timed batch: DuckDB's timer has 1 ms resolution, so a batch gives ~0.01 ms per query
N_SCRATCH_CALLS = 3  # the from-scratch function is slow; time a few calls
LIFECYCLE_N = 1_000_000
LIFECYCLE_INSERTS = [1, 1_000, 100_000]
LIFECYCLE_REPEATS = 3

rows = []


def add(experiment, method, n, times, radius_km=None, note=""):
    rows.append(dict(experiment=experiment, method=method, n=n, radius_km=radius_km, median_s=statistics.median(times),
                     min_s=min(times), max_s=max(times), note=note))


def median_of(experiment, method, n=None, radius_km=None):
    return next(x["median_s"] for x in rows if x["experiment"] == experiment and x["method"] == method
                and (n is None or x["n"] == n) and (radius_km is None or x["radius_km"] == radius_km))


def run_duckdb(script, db=None):
    args = [DUCKDB] + ([db] if db else [])
    out = subprocess.run(args, input=script, capture_output=True, text=True, check=True).stdout
    times = [float(x) for x in re.findall(r"Run Time \(s\): real ([0-9.]+)", out)]
    ints = [int(x) for x in re.findall(r"^(\d+)$", out, flags=re.M)]
    return times, ints, out


def make_dataset(n, tmp):
    """Real points for 10k / 99,907; for 1M, the real points resampled with ~1 km of jitter."""
    df = pd.read_parquet(SRC)[["id", "latitude", "longitude"]]
    if n <= len(df):
        sub = df if n == len(df) else df.sample(n, random_state=0)
    else:
        rng = np.random.default_rng(0)
        pick = df.iloc[rng.integers(0, len(df), n)].reset_index(drop=True)
        pick["latitude"] = (pick.latitude + rng.normal(0, 0.01, n)).clip(-89.9, 89.9)
        pick["longitude"] = pick.longitude + rng.normal(0, 0.01, n)
        pick["id"] = np.arange(n)
        sub = pick
    path = os.path.join(tmp, f"pts_{n}.parquet")
    sub.reset_index(drop=True).to_parquet(path)
    return sub.reset_index(drop=True), path


LOAD = "CREATE TABLE pts AS SELECT id, latitude AS lat, longitude AS lon FROM read_parquet('{path}');"
PROLOGUE = ".mode list\n.headers off\n"


# ---------------------------------------------------------------------------------------------------------------
def experiment_degree(df, path):
    print("== degree, all points", flush=True)
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    n = len(df)

    sk_max = {}
    for r in DEGREE_RADII_KM:
        ts = []
        for _ in range(REPEATS):
            t = time.perf_counter()
            tree = BallTree(coords, metric="haversine")
            counts = tree.query_radius(coords, r / EARTH_KM, count_only=True)
            ts.append(time.perf_counter() - t)
        sk_max[r] = int(counts.max()) - 1
        add("degree", "sklearn", n, ts, r)

    q_fn = [f"SELECT max(degree) FROM ball_degree('pts','id','lat','lon',{r});" for r in DEGREE_RADII_KM for _ in range(REPEATS)]
    q_ix = [f"SELECT max(degree) FROM ball_index_degree('pts_idx',{r});" for r in DEGREE_RADII_KM for _ in range(REPEATS)]
    script = (PROLOGUE + LOAD.format(path=path) + "\nCREATE INDEX pts_idx ON pts USING BALL_TREE (lat, lon);\n.timer on\n"
              + "\n".join(q_fn + q_ix) + "\n")
    times, maxes, _ = run_duckdb(script)
    assert len(times) == len(maxes) == len(q_fn) + len(q_ix), (len(times), len(maxes))
    for method, off in (("table_fn", 0), ("index", len(q_fn))):
        for i, r in enumerate(DEGREE_RADII_KM):
            sl = slice(off + i * REPEATS, off + (i + 1) * REPEATS)
            assert set(maxes[sl]) == {sk_max[r]}, f"{method} r={r}: {set(maxes[sl])} != {sk_max[r]}"
            add("degree", method, n, times[sl], r)
    for r in DEGREE_RADII_KM:
        print(f"  r={r:>3}km  sklearn={median_of('degree', 'sklearn', radius_km=r) * 1000:8.1f}ms"
              f"  table_fn={median_of('degree', 'table_fn', radius_km=r) * 1000:8.1f}ms"
              f"  index={median_of('degree', 'index', radius_km=r) * 1000:8.1f}ms", flush=True)


# ---------------------------------------------------------------------------------------------------------------
def experiment_point(n, tmp):
    df, path = make_dataset(n, tmp)
    print(f"== point query, n={len(df):,}", flush=True)
    n = len(df)
    q = df.sample(N_QUERY_POINTS, random_state=1)
    qs = list(zip(q.id, q.latitude, q.longitude))
    R = POINT_RADIUS_KM

    # scikit-learn: the tree is already built; time only the query. (No SQL layer, so a floor, not a peer.)
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    pts_rad = np.radians(np.array([[la, lo] for _, la, lo in qs]))
    ts = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        expected = [len(tree.query_radius(p.reshape(1, -1), R / EARTH_KM)[0]) for p in pts_rad]
        ts.append((time.perf_counter() - t) / N_QUERY_POINTS)
    add("point", "sklearn", n, ts, R)
    expected_total = sum(expected)

    hav = ("2 * 6371.0088 * asin(sqrt(sin(radians(lat - {la}) / 2) ^ 2 + cos(radians({la})) * cos(radians(lat)) "
           "* sin(radians(lon - {lo}) / 2) ^ 2))")
    ix = " UNION ALL ".join(f"SELECT w.row_id FROM ball_index_within('pts_idx', {la!r}, {lo!r}, {R}) w "
                            f"JOIN pts p ON p.rowid = w.row_id" for _, la, lo in qs)
    bf = " UNION ALL ".join(f"SELECT id FROM pts WHERE {hav.format(la=repr(la), lo=repr(lo))} <= {R}" for _, la, lo in qs)
    fn = " UNION ALL ".join(f"SELECT id FROM ball_within('pts','id','lat','lon',{i},{R})" for i, _, _ in qs[:N_SCRATCH_CALLS])
    script = (PROLOGUE + LOAD.format(path=path) + "\nCREATE INDEX pts_idx ON pts USING BALL_TREE (lat, lon);\n.timer on\n"
              + "".join(f"SELECT count(*) FROM ({ix});\n" for _ in range(REPEATS))
              + "".join(f"SELECT count(*) FROM ({bf});\n" for _ in range(REPEATS))
              + "".join(f"SELECT count(*) FROM ({fn});\n" for _ in range(REPEATS)))
    times, counts, _ = run_duckdb(script)
    assert len(times) == len(counts) == 3 * REPEATS, (len(times), len(counts))
    ix_t, bf_t, fn_t = times[:REPEATS], times[REPEATS:2 * REPEATS], times[2 * REPEATS:]
    ix_c, bf_c = counts[:REPEATS], counts[REPEATS:2 * REPEATS]
    assert set(ix_c) == set(bf_c) == {expected_total}, f"counts differ: index {set(ix_c)} scan {set(bf_c)} sklearn {expected_total}"
    add("point", "index", n, [t / N_QUERY_POINTS for t in ix_t], R)
    add("point", "scan", n, [t / N_QUERY_POINTS for t in bf_t], R)
    add("point", "table_fn", n, [t / N_SCRATCH_CALLS for t in fn_t], R)
    print(f"  sklearn={median_of('point', 'sklearn', n) * 1e3:9.3f}ms  index={median_of('point', 'index', n) * 1e3:9.3f}ms  "
          f"scan={median_of('point', 'scan', n) * 1e3:9.3f}ms  table_fn={median_of('point', 'table_fn', n) * 1e3:9.3f}ms"
          f"   (per query; {expected_total} matches over {N_QUERY_POINTS} queries)", flush=True)


# ---------------------------------------------------------------------------------------------------------------
def experiment_lifecycle(tmp):
    print(f"== lifecycle, n={LIFECYCLE_N:,}", flush=True)
    df, path = make_dataset(LIFECYCLE_N, tmp)
    la, lo = float(df.latitude.iloc[0]), float(df.longitude.iloc[0])
    query = f"SELECT count(*) FROM ball_index_within('pts_idx', {la!r}, {lo!r}, {POINT_RADIUS_KM});"
    steps = {k: [] for k in ["create", "reload", "query"] + [f"fold_{m}" for m in LIFECYCLE_INSERTS]}
    for rep in range(LIFECYCLE_REPEATS):
        db = os.path.join(tmp, f"life_{rep}.db")
        # process 1: build the index and write it to the file. With DuckDB's default settings CREATE INDEX
        # already checkpoints on commit, so the write is part of its time (the CHECKPOINT after it is a no-op).
        t, _, _ = run_duckdb(PROLOGUE + LOAD.format(path=path) + "\n.timer on\nCREATE INDEX pts_idx ON pts USING BALL_TREE (lat, lon);\nCHECKPOINT;\n", db)
        steps["create"].append(t[0])
        # process 2, a fresh process: the first touch loads the tree from the file; then a query; then changes,
        # each followed by a query, which is the one that has to fold them in.
        script = (PROLOGUE + "PRAGMA disable_checkpoint_on_shutdown;\nSET wal_autocheckpoint='1TB';\n.timer on\n"
                  "SELECT tree_points FROM ball_index_info();\n" + query + "\n")
        next_id = LIFECYCLE_N
        for m in LIFECYCLE_INSERTS:
            script += (f"INSERT INTO pts SELECT {next_id} + i, {la!r} + (i % 1000) * 0.001, {lo!r} FROM range({m}) t(i);\n"
                       + query + "\n")
            next_id += m
        times, ints, out = run_duckdb(script, db)
        assert len(times) == 2 + 2 * len(LIFECYCLE_INSERTS), (len(times), out[-300:])
        steps["reload"].append(times[0])
        steps["query"].append(times[1])
        for j, m in enumerate(LIFECYCLE_INSERTS):
            steps[f"fold_{m}"].append(times[3 + 2 * j])  # times[2 + 2j] is the INSERT; the query after it folds it in
        os.remove(db)
        if os.path.exists(db + ".wal"):
            os.remove(db + ".wal")
    labels = {"create": "CREATE INDEX (build + write to file)", "reload": "reload from file",
              "query": "query on a fresh index"}
    labels.update({f"fold_{m}": f"first query after {m} insert(s)" for m in LIFECYCLE_INSERTS})
    for k, ts in steps.items():
        add("lifecycle", k, LIFECYCLE_N, ts, note=labels[k])
        print(f"  {k:14} {statistics.median(ts) * 1000:9.1f} ms", flush=True)


def main():
    df = pd.read_parquet(SRC)[["id", "latitude", "longitude"]]
    tmp = tempfile.mkdtemp(prefix="bench_")
    path = os.path.join(tmp, "pts_real.parquet")
    df.to_parquet(path)
    experiment_degree(df, path)
    for n in POINT_SIZES:
        experiment_point(n, tmp)
    experiment_lifecycle(tmp)
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results.csv")


if __name__ == "__main__":
    main()
