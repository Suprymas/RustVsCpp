#!/usr/bin/env bash
#
# run_memory.sh - peak memory (VmHWM) measurement for all three tasks.
#
# A SEPARATE script from run_series.sh, on purpose. Timing needs the full
# protocol - locked frequency, warm-up, 5-30 repetitions, shuffled order,
# alternating languages within a point. Peak memory needs NONE of these:
#
#   * VmHWM does not depend on CPU frequency or thermal state, so
#     preflight.sh lock is not required here;
#   * it does not depend on thread affinity, so the affinity axis is skipped;
#   * it is almost deterministic - repeating the same run the value varies
#     by about 0.7 % (measured: matmul N=512 rust 12884-12988 KiB). The
#     remainder comes from ASLR, thread stack placement and mimalloc page
#     rounding. So the median of THREE runs is taken, not ten - that is
#     enough, because the differences under study (7-20 %) are ten times
#     larger than the noise.
#
# Hence three runs per (language, variant, size, threads) combination are
# enough, rather than the full timing protocol.
#
# The PROCESS peak is measured, i.e. it includes the input read into memory
# (the BFS graph, the corpus). So an "input_kib" column is written next to
# it - subtracting it leaves what the implementation itself allocates.
#
# Usage:
#   ./run_memory.sh --task matmul|bfs|wordfreq [--out file.csv]
#                   [--sizes ...] [--threads 1,2,4,8,16,32] [--langs rust,cpp]
#                   [--reps 1] [--passes 3]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DATA="$ROOT/data"

TASK=""
OUT=""
SIZES=""
THREADS="1,2,4,8,16,32"
LANGS="rust,cpp"
REPS=1
PASSES=3

while [ $# -gt 0 ]; do
    case "$1" in
        --task)    TASK="$2"; shift 2 ;;
        --out)     OUT="$2"; shift 2 ;;
        --sizes)   SIZES="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --langs)   LANGS="$2"; shift 2 ;;
        # Repetitions here are not statistics but the program's INTERNAL
        # loop repetitions. Default 1: so the peak reflects the cost of ONE
        # computation. With more repetitions mimalloc may not return memory to
        # the system and the peak would look larger than what is really needed.
        --reps)    REPS="$2"; shift 2 ;;
        # Number of runs the median is taken from (see the comment at the top).
        --passes)  PASSES="$2"; shift 2 ;;
        -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

# shellcheck source=tasks.sh
. "$HERE/tasks.sh"
task_config "$TASK"

[ -n "$SIZES" ] || SIZES="$DEF_SIZES"
[ -n "$OUT" ] || OUT="$HERE/results/$TASK-memory-$(date +%Y%m%d-%H%M%S).csv"

# Size of the input files in kibibytes. Needed so that the memory table can
# separate the input read into memory from what the program itself allocates.
input_kib() {
    local n="$1" total=0 f
    local files=()
    case "$TASK" in
        matmul)   files=("$DATA/matrix_A_$n.bin" "$DATA/matrix_B_$n.bin") ;;
        bfs)      files=("$DATA/graph_s$n.csr") ;;
        wordfreq) files=("$DATA/corpus_$n.txt") ;;
    esac
    for f in "${files[@]}"; do
        [ -f "$f" ] && total=$(( total + $(stat -c %s "$f") ))
    done
    echo $(( total / 1024 ))
}

IFS=, read -ra size_list <<< "$SIZES"
IFS=, read -ra thread_list <<< "$THREADS"
IFS=, read -ra lang_list <<< "$LANGS"

for lang in "${lang_list[@]}"; do
    bin=$(bin_for "$lang")
    [ -x "$bin" ] || { echo "error: $bin does not exist" >&2; exit 1; }
done
[ -d "$DATA" ] || { echo "error: data directory $DATA does not exist" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"

# List of points. The order is NOT shuffled: the measurement is
# deterministic, and a fixed order makes the progress readable by eye.
points=()
for n in "${size_list[@]}"; do
    points+=("base $n 1")
    for level in $LEVELS; do
        level_applies "$level" "$n" || continue
        for p in "${thread_list[@]}"; do
            points+=("$level $n $p")
        done
    done
done

total=$(( ${#points[@]} * ${#lang_list[@]} ))
echo "Task: $TASK. Memory points: $total (languages: $LANGS), $PASSES runs each"
echo "Output: $OUT"
echo

echo "lang,task,variant,n,threads,level,rss_kib,input_kib,spread_pct" > "$OUT"

i=0
start_ts=$(date +%s)
for point in "${points[@]}"; do
    read -r level n p <<< "$point"
    input=$(input_kib "$n")
    key=$(csv_size "$n")
    for lang in "${lang_list[@]}"; do
        i=$((i + 1))
        bin=$(bin_for "$lang")
        variant=$(variant_for "$lang" "$level")

        printf "[%3d/%d] %-4s %-16s %s=%-5s p=%-2s " \
               "$i" "$total" "$lang" "$variant$(strategy_args "$level" | sed 's/--strategy /-/')" \
               "$SIZE_LABEL" "$n" "$p"

        vals=()
        for _ in $(seq "$PASSES"); do
            # shellcheck disable=SC2046  # strategy_args is split into two words on purpose
            v=$("$bin" --mode "$variant" $(strategy_args "$level") \
                       "$SIZE_FLAG" "$n" --threads "$p" \
                       --reps "$REPS" --warmup 0 --data "$DATA" \
                | awk '/^peak memory/ {print $3}')
            [ -n "$v" ] || { echo "FAILED (no 'peak memory' line found)" >&2; exit 1; }
            vals+=("$v")
        done

        # Median and spread. The spread is written to the CSV: if it turned out
        # large anywhere, that would mean peak memory at that point is not
        # stable, and the number would need to be investigated separately.
        read -r rss spread < <(printf '%s\n' "${vals[@]}" | sort -n | awk '
            {a[NR] = $1}
            END {
                med = (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2
                printf "%d %.2f\n", med, (med > 0 ? (a[NR] - a[1]) * 100 / med : 0)
            }')
        echo "$lang,$TASK,$variant,$key,$p,$level,$rss,$input,$spread" >> "$OUT"
        printf "%'d KiB  (input %'d KiB, spread %s%%)\n" "$rss" "$input" "$spread"
    done
done

elapsed=$(( $(date +%s) - start_ts ))
echo
echo "Finished in $((elapsed / 60)) min $((elapsed % 60)) s. Results: $OUT"
