#!/usr/bin/env python3
"""Measurement plots and summary (for all tasks).

Reads the CSV produced by run_series.sh and draws three figures:
  speedup.*     Sp = T1/Tp as a function of the thread count
  efficiency.*  Ep = Sp/p
  time.*        median on a logarithmic scale

T1 is taken from the sequential version (variant "seq"), as the methodology
prescribes, not from the parallel version with one thread. EACH LANGUAGE has
its own T1 - otherwise Sp would mix the speed difference between languages
with the benefit of parallelisation.

Panel grid: column = size, row = (affinity series, level). Each panel holds
EXACTLY TWO lines - Rust and C++ - so colour only has to encode the language.
This is exactly the shape of the research question: same level, same
conditions, only the language differs.

Every figure is saved both as PNG (for viewing) and PDF (vector, for the
thesis text). A summary CSV with medians and Sp/Ep is written next to them.

The task is detected from the CSV column "task": only the size label and the
throughput metric (GFLOP/s, MTEPS or Mwords/s) depend on it, everything else
is the same.

The word frequency task has a THIRD axis - strategy (partial tables or a
shared locked table). It is NOT merged with the level: the level is
"low/high" in both languages, while the strategy is an algorithmic decision,
identical for both languages and both levels. So a SEPARATE set of figures is
drawn for each strategy into its own subdirectory (plots/partial/,
plots/shared/). Merging them into one grid would leave empty panels, because
"shared" was measured at 512 MiB only.

Usage:
  .venv/bin/python plot_series.py results.csv [--out-dir directory]
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colours from the validated categorical palette (first slots). Checked with
# validate_palette.js: passes the colour-blindness separation and contrast
# thresholds.
#
# Colour encodes the LANGUAGE, because languages are what is compared. Level
# and affinity series are encoded by panel position (faceting), not by colour
# or line style.
COLORS = {"rust": "#eb6834", "cpp": "#2a78d6"}
LANG_LABELS = {"rust": "Rust", "cpp": "C++"}

# The level is what we compare; the variant is the name of the concrete tool in a language.
VARIANT_LEVEL = {"seq": "base", "threads": "low", "rayon": "high", "omp": "high"}
LEVEL_LABEL = {"low": "low level", "high": "high level", "base": "sequential"}
# Which tool corresponds to a level in which language - for the summary and labels.
LEVEL_TOOL = {
    ("rust", "low"): "std::thread",
    ("cpp", "low"): "std::thread",
    ("rust", "high"): "Rayon",
    ("cpp", "high"): "OpenMP",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"

# The affinity series gets a SEPARATE ROW, not a line style. Encoding it with
# dashes was tried - in practice solid and dashed lines are indistinguishable
# in the legend, and four lines in one panel overlap. With faceting each panel
# keeps two lines, distinguished by colour alone.
SERIES_LABEL = {"on": "with affinity", "off": "without affinity"}

# Strategy (word frequencies only). Empty label = the task has no strategies.
STRATEGY_LABEL = {
    "partial": "partial tables",
    "shared": "shared locked table",
}

# Number of directed entries in the graph (m) for each scale. It is not in the
# CSV, and MTEPS cannot be computed without it. Values from data/MANIFEST.md -
# if the data were regenerated with a different seed, update them here.
BFS_ENTRIES = {20: 31402998, 22: 128309562, 24: 520748778}

# Number of words in the corpus (n is written in mebibytes). Not in the CSV.
# Values from data/MANIFEST.md and from the `wordfreq --mode seq` output.
WORDFREQ_WORDS = {512: 101855682, 2048: 405750506, 8192: 1614459477}

# How the tasks differ in the plots. Everything else is shared.
TASKS = {
    "matmul": {
        "size_label": "N",
        "size_title": lambda n: f"N = {n}",
        "throughput_name": "GFLOPS",
        "throughput": lambda n, t: 2.0 * n ** 3 / t / 1e9,
    },
    "bfs": {
        # MTEPS (millions of traversed edges per second) - the Graph500 metric.
        # In a connected component every vertex is visited, so all m directed
        # entries are examined.
        "size_label": "scale",
        "size_title": lambda n: f"scale = {n}",
        "throughput_name": "MTEPS",
        "throughput": lambda n, t: BFS_ENTRIES[n] / t / 1e6 if n in BFS_ENTRIES else float("nan"),
    },
    "wordfreq": {
        # Throughput in words per second - an algorithmic metric, analogous
        # to MTEPS. MB/s would be misleading here, because the amount of work
        # is determined by the number of words, not bytes.
        "size_label": "corpus",
        "size_title": lambda n: f"{n} MiB" if n < 1024 else f"{n // 1024} GiB",
        "throughput_name": "Mwords_s",
        "throughput": (lambda n, t: WORDFREQ_WORDS[n] / t / 1e6
                       if n in WORDFREQ_WORDS else float("nan")),
    },
}


def parse_variant(variant):
    """Variant -> (level, strategy).

    "threads-partial" -> ("low", "partial"); "omp" -> ("high", "").
    The strategy is separated rather than folded into the level, because they
    are different things: the level compares a LANGUAGE facility (std::thread
    vs Rayon/OpenMP), while the strategy compares an ALGORITHMIC decision that
    is the same for both languages.
    """
    base, _, strategy = variant.partition("-")
    level = VARIANT_LEVEL.get(base)
    return level, strategy


def load(path):
    """Returns ({(lang, level, n, threads, series, strategy): [times]}, task).

    The variant is turned into a LEVEL right away: "rayon" and "omp" are the
    same high level in different languages, and that is exactly how they must
    be put into one panel. The strategy (word frequencies) stays a separate
    key member - see parse_variant.
    """
    runs = defaultdict(list)
    tasks = set()
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or header[0] != "lang":
            sys.exit(
                f"error: {path} has no header line.\n"
                "The file must have: lang,task,variant,n,threads,..."
            )
        for row in reader:
            if len(row) < 8:
                continue
            lang, variant = row[0], row[2]
            tasks.add(row[1])
            level, strategy = parse_variant(variant)
            if level is None:
                sys.exit(f"error: unknown variant {variant!r} in {path}")
            if strategy and strategy not in STRATEGY_LABEL:
                sys.exit(f"error: unknown strategy {strategy!r} in {path}")
            n, threads, series = int(row[3]), int(row[4]), row[5]
            runs[(lang, level, n, threads, series, strategy)].append(float(row[7]))
    if not runs:
        sys.exit(f"error: {path} has no data")
    if len(tasks) != 1:
        sys.exit(f"error: {path} mixes tasks: {sorted(tasks)}")
    task = tasks.pop()
    if task not in TASKS:
        sys.exit(f"error: unknown task {task!r} (known: {sorted(TASKS)})")
    return runs, task


def summarize(runs):
    """Medians, minima and TWO spread measures for every measurement point.

    The range (max-min)/med is sensitive to a single outlier and also grows
    with the number of repetitions, so it cannot be compared between points
    with different repetition counts. IQR/med is a robust measure: it shows
    whether the measurements THEMSELVES are widely scattered, not whether a
    single disturbance happened.

    The difference between them is informative:
      large range + small IQR = an isolated background disturbance, the median copes
      large IQR               = a genuinely unstable point, worth investigating
    """
    out = {}
    for key, times in runs.items():
        times = sorted(times)
        med = statistics.median(times)
        if len(times) >= 4:
            q1, _, q3 = statistics.quantiles(times, n=4)
            iqr = q3 - q1
        else:
            iqr = times[-1] - times[0]
        out[key] = {
            "min": times[0],
            "med": med,
            "max": times[-1],
            "amplitude": (times[-1] - times[0]) / med * 100 if med else 0.0,
            "iqr": iqr / med * 100 if med else 0.0,
            "n_reps": len(times),
        }
    return out


def baselines(stats):
    """T1 for every (lang, n, series) triple, from the sequential version.

    T1 is SEPARATE for each language. If one baseline were used for both
    languages, Sp would already contain the speed difference between the
    languages, and it would be impossible to tell whether a language loses
    because of parallelisation or because of its sequential code.
    """
    t1 = {}
    for (lang, level, n, _threads, series, _strategy), s in stats.items():
        if level == "base":
            t1[(lang, n, series)] = s["med"]
    return t1


def style_axes(ax, xticks):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])


def place_end_labels(ax, fig, items):
    """Places the end-point labels so that they do not overlap.

    Labels are sorted by y in pixel coordinates and, if two end up closer than
    min_gap, the lower one is pushed down. Without this, in the N = 1024 panel
    "17.3" and "16.9" ended up on top of each other.
    """
    if not items:
        return
    ax.autoscale_view()
    pts = sorted((ax.transData.transform((x, y))[1], x, y, txt, color)
                 for x, y, txt, color in items)
    min_gap_px = 13.0 * fig.dpi / 72.0
    last = float("-inf")
    for py, x, y, txt, color in pts:
        ty = max(py, last + min_gap_px)
        last = ty
        ax.annotate(txt, (x, y), textcoords="offset points",
                    xytext=(6, (ty - py) * 72.0 / fig.dpi - 3),
                    fontsize=8, color=INK_SECONDARY, zorder=4)


def draw(stats, t1, task, sizes, levels, series_list, threads, langs, kind,
         out_dir, strategy=""):
    """Draws one figure.

    Column = size, row = (affinity series, level), line = language.
    `strategy` selects which strategy's measurements are used (word
    frequencies); it is empty for the other tasks.
    """
    titles = {
        "speedup": ("Speedup $S_p = T_1/T_p$", "$S_p$"),
        "efficiency": ("Efficiency $E_p = S_p/p$", "$E_p$, %"),
        "time": ("Execution time (median)", "time, s"),
    }
    title, ylabel = titles[kind]
    rows = [(series, level) for series in series_list for level in levels]
    nrows, ncols = len(rows), len(sizes)

    # The largest value in the figure. Needed to decide whether the
    # ideal-scaling reference still makes sense.
    fig_max = 0.0
    if kind != "time":
        for series, level in rows:
            for n in sizes:
                for lang in langs:
                    for p_ in threads:
                        st = stats.get((lang, level, n, p_, series, strategy))
                        base = t1.get((lang, n, series))
                        if st and base:
                            v = base / st["med"]
                            fig_max = max(fig_max, v if kind == "speedup" else v / p_ * 100)

    # When all values are far below ideal (shared locked table: Sp drops to
    # 0.08), the ideal line squeezes all data into the bottom of the axis and
    # the plot shows nothing. In that case the reference is replaced with a
    # more meaningful one: Sp = 1 is the line where the parallel version
    # breaks even with the sequential one, and below it, it is slower.
    degenerate = ((kind == "speedup" and fig_max < 2.0)
                  or (kind == "efficiency" and fig_max < 20.0))

    # Reference line names - they do not go into the panel legends, but they
    # do go into the common figure legend at the bottom.
    reference_labels = {"ideal", "sequential version level ($S_p = 1$)"}

    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(4.2 * ncols, 3.5 * nrows),
                             sharey=(kind != "time"), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    for row, (series, level) in enumerate(rows):
        for col, n in enumerate(sizes):
            ax = axes[row][col]
            style_axes(ax, threads)

            # Ideal-scaling reference - recessive.
            if degenerate:
                if kind == "speedup":
                    ax.axhline(1.0, color=INK_MUTED, linewidth=1.2, linestyle=":",
                               zorder=1, label="sequential version level ($S_p = 1$)")
            elif kind == "speedup":
                ax.plot(threads, threads, color=INK_MUTED, linewidth=1.2,
                        linestyle=":", zorder=1, label="ideal")
            elif kind == "efficiency":
                ax.axhline(100, color=INK_MUTED, linewidth=1.2, linestyle=":",
                           zorder=1, label="ideal")

            end_labels = []
            for lang in langs:
                xs, ys = [], []
                for p in threads:
                    s = stats.get((lang, level, n, p, series, strategy))
                    if s is None:
                        continue
                    base = t1.get((lang, n, series))
                    if kind == "time":
                        ys.append(s["med"])
                    elif base is None:
                        continue
                    elif kind == "speedup":
                        ys.append(base / s["med"])
                    else:
                        ys.append(base / s["med"] / p * 100)
                    xs.append(p)
                if not xs:
                    continue

                tool = LEVEL_TOOL.get((lang, level), "")
                label = f"{LANG_LABELS[lang]} ({tool})" if tool else LANG_LABELS[lang]
                ax.plot(xs, ys, color=COLORS[lang], linewidth=2.0,
                        marker="o", markersize=5, markeredgecolor=SURFACE,
                        markeredgewidth=1.0, zorder=3, label=label)

                # Direct label on the last point: the reader does not have to
                # glance back and forth to the legend.
                if kind != "time":
                    txt = f"{ys[-1]:.1f}"
                elif ys[-1] < 1.0:
                    txt = f"{ys[-1]*1000:.1f} ms"
                else:
                    txt = f"{ys[-1]:.2f} s"
                end_labels.append((xs[-1], ys[-1], txt, COLORS[lang]))

            if kind == "time":
                ax.set_yscale("log")
            place_end_labels(ax, fig, end_labels)

            # The legend is built from EVERY row, because the tool name differs
            # between levels (Rayon / OpenMP), so the labels belong in the row.
            if col == ncols - 1:
                handles, labels = ax.get_legend_handles_labels()
                pairs = [(h, l) for h, l in zip(handles, labels)
                         if l not in reference_labels]
                if pairs:
                    # In a falling curve ("shared") the upper left is taken by
                    # the data, so the legend moves to the right.
                    loc = "upper right" if degenerate else "upper left"
                    ax.legend([h for h, _ in pairs], [l for _, l in pairs],
                              loc=loc, frameon=False, fontsize=8,
                              labelcolor=INK_SECONDARY)

            if row == 0:
                ax.set_title(TASKS[task]["size_title"](n), fontsize=11,
                             color=INK, pad=8)
            if row == nrows - 1:
                ax.set_xlabel("number of threads $p$", fontsize=9, color=INK_SECONDARY)
            if col == 0:
                lbl = ylabel
                if nrows > 1:
                    lbl = f"{SERIES_LABEL[series]}, {LEVEL_LABEL[level]}\n{ylabel}"
                ax.set_ylabel(lbl, fontsize=9, color=INK_SECONDARY)

    # In the time plot the y scale is not shared across the whole figure (the
    # sizes differ by hundreds of times), but WITHIN A COLUMN it must match -
    # otherwise the two affinity series would be drawn on different scales and
    # comparing them by eye would be misleading.
    if kind == "time" and nrows > 1:
        for col in range(ncols):
            lo = min(axes[r][col].get_ylim()[0] for r in range(nrows))
            hi = max(axes[r][col].get_ylim()[1] for r in range(nrows))
            for r in range(nrows):
                axes[r][col].set_ylim(lo, hi)

    # Only the ideal-scaling reference is left at the bottom: language labels
    # are already in every row (the tool name differs there).
    handles, labels = axes[0][0].get_legend_handles_labels()
    ideal = [(h, l) for h, l in zip(handles, labels) if l in reference_labels]
    if ideal:
        fig.legend([h for h, _ in ideal], [l for _, l in ideal], loc="lower center",
                   frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
                   bbox_to_anchor=(0.5, -0.005))

    if strategy:
        title = f"{title} - {STRATEGY_LABEL[strategy]}"
    fig.suptitle(title, fontsize=13, color=INK, y=0.997)
    fig.tight_layout(rect=[0, 0.02 if ideal else 0.0, 1, 0.975])

    names = {"speedup": "speedup", "efficiency": "efficiency", "time": "time"}
    written = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{names[kind]}.{ext}"
        fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def write_summary(stats, t1, task, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        throughput_name = TASKS[task]["throughput_name"]
        throughput = TASKS[task]["throughput"]
        w.writerow(["lang", "level", "strategy", "tool", "n", "threads",
                    "affinity", "min_s", "median_s", "max_s", "iqr_pct",
                    "range_pct", "Sp", "Ep_pct", throughput_name,
                    "reps"])
        # Sorted by (n, level, series, lang, threads): this places next to
        # each other exactly the two entries being compared - the same point
        # in both languages.
        def order(kv):
            (lang, level, n, p, series, strategy), _ = kv
            return (n, strategy, level, series, lang, p)

        for (lang, level, n, p, series, strategy), s in sorted(stats.items(), key=order):
            base = t1.get((lang, n, series))
            sp = base / s["med"] if base else ""
            ep = sp / p * 100 if base else ""
            tput = throughput(n, s["med"])
            w.writerow([
                lang, level, strategy or "-", LEVEL_TOOL.get((lang, level), "-"),
                n, p, series,
                f"{s['min']:.6f}", f"{s['med']:.6f}", f"{s['max']:.6f}",
                f"{s['iqr']:.1f}", f"{s['amplitude']:.1f}",
                f"{sp:.2f}" if sp != "" else "",
                f"{ep:.0f}" if ep != "" else "",
                f"{tput:.1f}", s["n_reps"],
            ])


def main():
    ap = argparse.ArgumentParser(description="Measurement series plots")
    ap.add_argument("csv", help="CSV produced by run_series.sh")
    ap.add_argument("--out-dir", default=None, help="where to put the figures (default: next to the CSV)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / (csv_path.stem + "-plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, task = load(csv_path)
    stats = summarize(runs)
    t1 = baselines(stats)

    # Key: (lang, level, n, threads, series, strategy)
    langs = [x for x in ("rust", "cpp") if any(k[0] == x for k in stats)]
    # The sequential version is always p = 1, so it must not be included in
    # the thread axis: otherwise a p = 1 point would appear where there are no
    # parallel measurements.
    threads = sorted({k[3] for k in stats if k[1] != "base"})
    series_list = [s for s in ("on", "off") if any(k[4] == s for k in stats)]
    levels = [x for x in ("low", "high") if any(k[1] == x for k in stats)]
    # Strategies that have PARALLEL measurements. The sequential version has
    # no strategy (its key is an empty label), and it is common to both.
    strategies = sorted({k[5] for k in stats if k[1] != "base"})

    if not t1:
        print("WARNING: no sequential version (variant 'seq'), so Sp and Ep")
        print("         are not computed. Run --mode seq as well.", file=sys.stderr)
    else:
        # T1 must exist for both languages; otherwise one language's panels would stay empty.
        all_sizes = sorted({k[2] for k in stats})
        missing = [(lang, n, s) for lang in langs for n in all_sizes for s in series_list
                   if (lang, n, s) not in t1]
        if missing:
            print(f"WARNING: missing T1 baselines: {len(missing)}, e.g. {missing[:3]}",
                  file=sys.stderr)

    written = []
    kinds = ["time"] + (["speedup", "efficiency"] if t1 else [])
    sizes_by_strategy = {}
    for strategy in strategies:
        # Sizes are computed SEPARATELY for each strategy: "shared" was
        # measured at 512 MiB only, and a common list would leave empty
        # panels.
        sizes = sorted({k[2] for k in stats if k[1] != "base" and k[5] == strategy})
        sizes_by_strategy[strategy] = sizes
        target = out_dir / strategy if strategy else out_dir
        target.mkdir(parents=True, exist_ok=True)
        for kind in kinds:
            written += draw(stats, t1, task, sizes, levels, series_list, threads,
                            langs, kind, target, strategy)

    # ONE summary for all strategies, with a "strategy" column: this leaves a
    # single table for the thesis text, in which all points can be sorted and
    # compared with each other.
    summary = out_dir / "summary.csv"
    write_summary(stats, t1, task, summary)
    written.append(summary)

    print(f"Task:      {task}")
    print(f"Languages: {[LANG_LABELS[x] for x in langs]}")
    for strategy in strategies:
        pref = f"{STRATEGY_LABEL[strategy]}: " if strategy else ""
        print(f"Sizes:     {pref}{TASKS[task]['size_label']} = {sizes_by_strategy[strategy]}")
    print(f"Levels:    {levels}")
    print(f"Series:    {series_list}")
    print(f"Threads:   {threads}")
    print()
    for p in written:
        print(f"  {p}")

    # Unstable points are judged by IQR, not by range: the range is raised by
    # a single background disturbance and also grows with the repetition count.
    label = TASKS[task]["size_label"]
    unstable = [(k, s) for k, s in stats.items() if s["iqr"] > 10]
    outliers = [(k, s) for k, s in stats.items() if s["iqr"] <= 10 and s["amplitude"] > 25]

    if unstable:
        print()
        print(f"UNSTABLE POINTS ({len(unstable)}): IQR > 10 %, i.e. the measurements")
        print("themselves are scattered, not a single disturbance. Worth investigating.")
        for (lang, level, n, p, series, strategy), s in sorted(
                unstable, key=lambda kv: -kv[1]["iqr"])[:8]:
            tool = LEVEL_TOOL.get((lang, level), "seq")
            print(f"  {lang:4s} {tool:11s} {strategy:8s} {label}={n:<5d} p={p:<3d} "
                  f"{series:3s}  IQR {s['iqr']:5.1f} %  (range {s['amplitude']:.0f} %)")

    if outliers:
        print()
        print(f"ISOLATED OUTLIERS ({len(outliers)}): small IQR, but range > 25 % -")
        print("a background disturbance during measurement. The median copes, results are valid.")
        for (lang, level, n, p, series, strategy), s in sorted(
                outliers, key=lambda kv: -kv[1]["amplitude"])[:5]:
            tool = LEVEL_TOOL.get((lang, level), "seq")
            print(f"  {lang:4s} {tool:11s} {strategy:8s} {label}={n:<5d} p={p:<3d} "
                  f"{series:3s}  range {s['amplitude']:5.0f} %  (IQR {s['iqr']:.1f} %)")


if __name__ == "__main__":
    main()
