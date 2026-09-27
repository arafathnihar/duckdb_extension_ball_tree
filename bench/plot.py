"""Plot bench/results.csv -> bench/benchmark.png.

    python plot.py   # needs matplotlib and pandas
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))

# Chart chrome and ink (light mode). The three categorical slots are the first three of the validated palette
# (validate_palette.js --mode light --pairs all), and each keeps its meaning in every panel. The plain SQL scan is a
# baseline, not a contender, so it gets the muted grey instead of a fourth hue.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
COLORS = {"sklearn": "#2a78d6", "table_fn": "#eb6834", "index": "#1baf7a", "scan": MUTED}
LABELS = {
    "sklearn": "scikit-learn BallTree",
    "table_fn": "extension, from-scratch functions",
    "index": "extension, BALL_TREE index",
    "scan": "plain SQL scan (no tree)",
}

df = pd.read_csv(os.path.join(HERE, "results.csv"))


def fmt_time(ms):
    return f"{ms:g} ms" if ms < 1000 else f"{ms / 1000:g} s"


def fmt_label(ms):
    if ms >= 1000:
        return f"{ms / 1000:.3g} s"
    return f"{ms:.3g} ms"


def style(ax, ylim, yticks):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_time(v)))
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def line(ax, x, y_ms, key, label_end=True):
    color = COLORS[key]
    ax.plot(x, y_ms, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", marker="o",
            markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, clip_on=False)
    if label_end:
        ax.annotate(fmt_label(y_ms.iloc[-1]), (x.iloc[-1], y_ms.iloc[-1]), xytext=(11, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9, color=INK2, annotation_clip=False)


fig = plt.figure(figsize=(17, 6.6), facecolor=SURFACE)
# column 2 is an empty spacer: panel B's end labels and panel C's row labels would otherwise collide
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.3, 1.2], left=0.06, right=0.965, top=0.70, bottom=0.14, wspace=0.28)
ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in (0, 1, 3))

# --- A: every point's neighbour count, by radius (99,907 real points)
style(ax_a, (50, 5000), [50, 100, 300, 1000, 3000])
d = df[df.experiment == "degree"]
for key in ("sklearn", "table_fn", "index"):
    s = d[d.method == key].sort_values("radius_km")
    # the two extension lines end almost on top of each other, so only the scikit-learn end is labelled
    line(ax_a, s.radius_km, s.median_s * 1000, key, label_end=(key == "sklearn"))
radii = sorted(d.radius_km.unique())
ax_a.set_xscale("log")
ax_a.set_xticks(radii)
ax_a.set_xticklabels([f"{int(r)}" for r in radii])
ax_a.xaxis.set_minor_locator(plt.NullLocator())
ax_a.set_xlim(0.8, 250 * 1.1)
ax_a.set_xlabel("search radius (km)", fontsize=9, color=INK2, labelpad=8)
ax_a.set_ylabel("time, log scale (lower is faster)", fontsize=9, color=INK2, labelpad=8)
ax_a.set_title("Neighbour count of every point", loc="left", fontsize=11, color=INK, fontweight="bold", pad=22)
ax_a.text(0, 1.025, "99,907 points; the index is already built", transform=ax_a.transAxes, fontsize=9, color=MUTED)

# --- B: one radius query, by table size
style(ax_b, (0.03, 1000), [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])
p = df[df.experiment == "point"]
for key in ("scan", "sklearn", "table_fn", "index"):
    s = p[p.method == key].sort_values("n")
    line(ax_b, s.n, s.median_s * 1000, key)
ax_b.set_xscale("log")
sizes = sorted(p.n.unique())
ax_b.set_xticks(sizes)
ax_b.set_xticklabels(["10k", "100k", "1M"])
ax_b.xaxis.set_minor_locator(plt.NullLocator())
ax_b.set_xlim(sizes[0] * 0.8, sizes[-1] * 1.1)
ax_b.set_xlabel("points in the table", fontsize=9, color=INK2, labelpad=8)
ax_b.set_title("One query: everything within 5 km", loc="left", fontsize=11, color=INK, fontweight="bold", pad=22)
ax_b.text(0, 1.025, "time per query", transform=ax_b.transAxes, fontsize=9, color=MUTED)

# --- C: what an index costs to build, reload and bring up to date (1M points)
ax_c.set_facecolor(SURFACE)
for side in ("top", "right", "left"):
    ax_c.spines[side].set_visible(False)
ax_c.spines["bottom"].set_color(BASELINE)
ax_c.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
ax_c.grid(axis="x", color=GRID, linewidth=1, linestyle="-")
ax_c.set_axisbelow(True)
lc = df[df.experiment == "lifecycle"].set_index("method").median_s * 1000
steps = [
    ("create", "CREATE INDEX\n(build + write to file)"),
    ("reload", "Reload from file\n(in a new process)"),
    ("fold_1", "First query after\n1 insert"),
    ("fold_1000", "First query after\n1,000 inserts"),
    ("fold_100000", "First query after\n100,000 inserts"),
    ("query", "Query on an\nup-to-date index"),
]
ys = list(range(len(steps)))[::-1]
for y, (key, _label) in zip(ys, steps):
    v = lc[key]
    ax_c.barh(y, v, height=0.46, color=COLORS["index"], edgecolor=SURFACE, linewidth=2)
    text = "≈ 1 ms" if key == "query" else fmt_label(v)
    ax_c.text(v + 10, y, text, va="center", ha="left", fontsize=9, color=INK2)
ax_c.set_yticks(ys)
ax_c.set_yticklabels([label for _k, label in steps], fontsize=9, color=INK2)
ax_c.set_xlim(0, 860)
ax_c.set_ylim(-0.6, len(steps) - 0.4)
ax_c.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)} ms"))
ax_c.set_xticks([0, 200, 400])
# bracket around the three "fold" bars: the cost does not depend on how many rows changed
y_hi, y_lo = ys[2] + 0.3, ys[4] - 0.3
ax_c.plot([590, 590], [y_lo, y_hi], color=BASELINE, linewidth=1.5, solid_capstyle="butt")
ax_c.text(606, ys[3], "a full rebuild,\nwhatever the number\nof changes", va="center", ha="left", fontsize=9, color=INK2)
ax_c.set_title("Keeping an index: cost at 1M points", loc="left", fontsize=11, color=INK, fontweight="bold", pad=22)
ax_c.text(0, 1.025, "the first query after a change rebuilds the tree", transform=ax_c.transAxes, fontsize=9, color=MUTED)

# --- headline, from the data
n_big = sizes[-1]
pm = p[p.n == n_big].set_index("method").median_s
vs_scan = pm["scan"] / pm["index"]
fold = lc[["fold_1", "fold_1000", "fold_100000"]]
fig.text(0.06, 0.965,
         f"At 1M points the index answers a 5 km query in {pm['index'] * 1000:.2f} ms, {vs_scan:.0f}× faster than a SQL scan, "
         f"but the first query after any change rebuilds the tree ({fold.min() / 1000:.1f} s)",
         fontsize=13.5, fontweight="bold", color=INK, va="top")
fig.text(0.06, 0.918,
         "Ball tree (haversine) in a DuckDB extension against scikit-learn and plain SQL. Median of 5 runs, 20 threads. "
         "The 1M table is the real points resampled with ~1 km of jitter.",
         fontsize=10, color=INK2, va="top")

handles = [Line2D([0], [0], color=COLORS[k], linewidth=2, marker="o", markersize=9, markeredgecolor=SURFACE,
                  markeredgewidth=2, label=LABELS[k]) for k in ("index", "table_fn", "scan", "sklearn")]
leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.875), ncol=4, frameon=False, fontsize=10,
                 handlelength=2.2, columnspacing=2.4)
for t in leg.get_texts():
    t.set_color(INK2)

fig.text(0.06, 0.02,
         "Intel i7-12700 (20 threads), DuckDB v1.5.4. Whole-table times for scikit-learn include the build; from-scratch "
         "functions also read the table. The scikit-learn point query runs in-process with the tree already in memory "
         "(no SQL layer). Point queries include the join back to the table. Every method returns the same matches.",
         fontsize=8.5, color=MUTED, va="bottom", wrap=True)

out = os.path.join(HERE, "benchmark.png")
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
