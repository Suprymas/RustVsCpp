#!/usr/bin/env bash
#
# run_series.sh - the full measurement series of one task (Rust and C++).
#
# Common engine for all three tasks. Task specifics (binary paths, size flag,
# default sizes and repetitions) are gathered in one place - tasks.sh.
# Everything else - order shuffling, environment check, CSV layout, result
# check - must be IDENTICAL for all tasks, otherwise their results would not
# be comparable with each other.
#
# Normally invoked via the wrappers: ./run_matmul.sh, ./run_bfs.sh
#
# Runs both languages x all levels x all sizes x all thread counts x both
# affinity series and collects the results into a single CSV file.
#
# THE ORDER IS SHUFFLED, and the unit of work is a MEASUREMENT POINT (level,
# n, p, series), not a single run. Within each point both languages are run
# one after the other in random order. Reason: if the series ran in a fixed
# order, the thermal state would correlate with the language - the language
# that goes first would be measured on a cooler machine and win by a few
# percent for no reason. The actual order is written to the CSV column
# "order" so it can be checked.
#
# Usage:
#   ./run_series.sh --task matmul|bfs|wordfreq [--out file.csv] [--reps 10]
#                   [--sizes ...] [--threads 1,2,4,8,16,32] [--series on,off]
#                   [--langs rust,cpp] [--quick]
#
# Required before running: sudo ./preflight.sh lock

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DATA="$ROOT/data"
CPU_ROOT="/sys/devices/system/cpu"

TASK=""
OUT=""
REPS_DEFAULT=10
REPS_MAP=""
REPS_SET=0
WARMUP=1
SIZES=""
SIZES_SET=0
THREADS="1,2,4,8,16,32"
SERIES="on,off"
LANGS="rust,cpp"
QUICK=0

while [ $# -gt 0 ]; do
    case "$1" in
        --task)    TASK="$2"; shift 2 ;;
        --out)     OUT="$2"; shift 2 ;;
        # --reps 10             10 for every size
        # --reps 512=30,1024=15 per size (the others - default)
        --reps)
            case "$2" in
                *=*) REPS_MAP="$2" ;;
                *)   REPS_DEFAULT="$2"; REPS_MAP="" ;;
            esac
            REPS_SET=1
            shift 2 ;;
        --warmup)  WARMUP="$2"; shift 2 ;;
        --sizes)   SIZES="$2"; SIZES_SET=1; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --series)  SERIES="$2"; shift 2 ;;
        --langs)   LANGS="$2"; shift 2 ;;
        # Quick check: one series, 3 repetitions, smallest sizes only.
        --quick)   QUICK=1; shift ;;
        -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

# ------------------------------------------------------- task settings

# Task specifics (binary paths, size flag, levels, default repetitions) live
# in tasks.sh - the memory measurement script shares it too.
# shellcheck source=tasks.sh
. "$HERE/tasks.sh"
task_config "$TASK"

[ -n "$SIZES" ] || SIZES="$DEF_SIZES"
[ "$REPS_SET" = "1" ] || REPS_MAP="$DEF_REPS_MAP"
[ -n "$OUT" ] || OUT="$HERE/results/$TASK-$(date +%Y%m%d-%H%M%S).csv"

if [ "$QUICK" = "1" ]; then
    REPS_DEFAULT=3
    REPS_MAP=""
    SERIES="on"
    # One "shared" repetition is enough for a technical check: at p=32 one run
    # takes ~40 s, so with the default 5 repetitions a "quick" check would
    # take about 40 min.
    SHARED_REPS=1
    [ "$SIZES_SET" = "1" ] || SIZES="$QUICK_SIZES"
fi

IFS=, read -ra size_list <<< "$SIZES"
IFS=, read -ra thread_list <<< "$THREADS"
IFS=, read -ra series_list <<< "$SERIES"
IFS=, read -ra lang_list <<< "$LANGS"

for lang in "${lang_list[@]}"; do
    bin=$(bin_for "$lang")
    if [ ! -x "$bin" ]; then
        echo "error: $bin does not exist" >&2
        case "$lang" in
            rust) echo "       run: cd rust/$TASK && cargo build --release" >&2 ;;
            cpp)  echo "       run: cd cpp && ./build.sh" >&2 ;;
        esac
        exit 1
    fi
done
[ -d "$DATA" ] || { echo "error: data directory $DATA does not exist" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"

# ------------------------------------------------------------ environment check

# Number of repetitions for a given size: REPS_MAP is searched first, then the
# default value is returned.
reps_for() {
    local n="$1" level="${2:-}" v=""
    # The "shared" strategy is tens of times slower, so it gets fewer repetitions.
    case "$level" in
        *-shared) [ -n "${SHARED_REPS:-}" ] && { echo "$SHARED_REPS"; return; } ;;
    esac
    [ -n "$REPS_MAP" ] && v=$(printf '%s\n' "$REPS_MAP" | tr ',' '\n' | awk -F= -v n="$n" '$1 == n {print $2}')
    echo "${v:-$REPS_DEFAULT}"
}

cur_freq_mhz() { echo $(( $(cat "$CPU_ROOT/cpu0/cpufreq/scaling_cur_freq" 2>/dev/null || echo 0) / 1000 )); }
loadavg()      { awk '{print $1}' /proc/loadavg; }

freq_locked() {
    [ "$(cat "$CPU_ROOT/cpufreq/boost" 2>/dev/null || echo 1)" = "0" ] || return 1
    local mn mx
    mn=$(cat "$CPU_ROOT/cpu0/cpufreq/scaling_min_freq"); mx=$(cat "$CPU_ROOT/cpu0/cpufreq/scaling_max_freq")
    [ "$mn" = "$mx" ]
}

if freq_locked; then
    echo "Frequency locked at $(cur_freq_mhz) MHz - good."
else
    echo "WARNING: frequency NOT locked. Results will reflect thermals, not languages." >&2
    echo "         Run: sudo $HERE/preflight.sh lock" >&2
    printf "Continue anyway? [y/N] " >&2
    read -r ans
    case "$ans" in [yY]*) ;; *) echo "aborted"; exit 1 ;; esac
fi

load_start=$(loadavg)
if awk -v l="$load_start" 'BEGIN{exit !(l > 0.5)}'; then
    echo "WARNING: system load $load_start - something is running in the background." >&2
fi

# ------------------------------------------------------------ list of points

# One point = "level n p series". The baseline is run with a single thread
# only: it is T1, and the thread count is meaningless for it.
points=()
for n in "${size_list[@]}"; do
    for s in "${series_list[@]}"; do
        points+=("base $n 1 $s")
        for level in $LEVELS; do
            # "shared" levels are measured only at the given sizes (see SHARED_SIZES).
            level_applies "$level" "$n" || continue
            for p in "${thread_list[@]}"; do
                points+=("$level $n $p $s")
            done
        done
    done
done

total_points=${#points[@]}
total_runs=$(( total_points * ${#lang_list[@]} ))
echo "Task: $TASK. Measurement points: $total_points, runs: $total_runs (languages: $LANGS)"
echo "Output: $OUT"
for n in "${size_list[@]}"; do
    echo "  $SIZE_LABEL = $n: $(reps_for "$n") repetitions each"
done
echo

# ------------------------------------------------------------ running

echo "lang,task,variant,n,threads,affinity,rep,time_s,checksum,bit_hash,freq_MHz,load,order" > "$OUT"

i=0
start_ts=$(date +%s)
point_no=0

while read -r level n p series; do
    point_no=$((point_no + 1))
    reps=$(reps_for "$n" "$level")
    printf "[%3d/%d] %-7s %s=%-4s p=%-2s affinity=%-3s x%-2s\n" \
           "$point_no" "$total_points" "$level" "$SIZE_LABEL" "$n" "$p" "$series" "$reps"

    # The order of languages within a point is shuffled separately - see the comment at the top.
    while read -r lang; do
        i=$((i + 1))
        bin=$(bin_for "$lang")
        variant=$(variant_for "$lang" "$level")

        aff=()
        [ "$series" = "on" ] && aff=(--affinity)

        # OpenMP affinity goes through environment variables (CLAUDE.md), not
        # through the program code.
        #
        # The environment is set ONLY for the omp variant. The reason was
        # tested: when OMP_PROC_BIND is set, libgomp pins the INITIAL thread
        # to one place and thereby narrows the affinity mask of the whole
        # process (with places=cores down to {0,16}). The sequential and
        # std::thread variants do not use libgomp, but they inherit the mask,
        # so all their threads pile onto one core and the speedup vanishes
        # (measured: threads p=8 dropped 1.3 -> 8.0 ms, i.e. to the level of
        # the sequential version).
        #
        # OMP_PLACES=cores, not =threads. Tested: libgomp builds the place
        # list by SMT pairs (place0=cpu0, place1=cpu16 - the second thread of
        # the same pair), so with =threads and close two threads land on one
        # physical core and p=2,8,16 were exactly 2x slower. With =cores one
        # place is one physical core, and thread t lands on core t - exactly
        # how core_affinity behaves on the Rust side (cpu0-15 are separate
        # cores, their SMT pairs are cpu16-31).
        env_args=()
        if [ "$lang" = "cpp" ] && [ "$variant" = "omp" ]; then
            if [ "$series" = "on" ]; then
                env_args=(env OMP_PLACES=cores OMP_PROC_BIND=close)
            else
                env_args=(env OMP_PROC_BIND=false)
            fi
        fi

        freq=$(cur_freq_mhz)
        load=$(loadavg)

        printf "         %-4s %-7s " "$lang" "$variant"

        # shellcheck disable=SC2046  # strategy_args is split into two words on purpose
        if ! "${env_args[@]}" "$bin" --mode "$variant" $(strategy_args "$level") \
                    "$SIZE_FLAG" "$n" --threads "$p" \
                    --reps "$reps" --warmup "$WARMUP" --data "$DATA" "${aff[@]}" --csv \
             | awk -v f="$freq" -v l="$load" -v e="$i" -F, '{print $0","f","l","e}' >> "$OUT"; then
            echo "FAILED" >&2
            exit 1
        fi

        # Median and spread briefly on screen, so progress and disturbances are visible.
        read -r med spread < <(tail -n "$reps" "$OUT" | awk -F, '{print $8}' | sort -n | awk '
            {a[NR] = $1}
            END {
                med = (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2
                printf "%.6f %.0f\n", med, (med > 0 ? (a[NR] - a[1]) * 100 / med : 0)
            }')
        printf "median %s s  spread %s%%\n" "$med" "$spread"
    done < <(printf '%s\n' "${lang_list[@]}" | shuf)
done < <(printf '%s\n' "${points[@]}" | shuf)

elapsed=$(( $(date +%s) - start_ts ))
echo
echo "Finished in $((elapsed / 60)) min $((elapsed % 60)) s."

# ------------------------------------------------------------ check

# For every size all results must match - both between variants and ACROSS
# LANGUAGES. It is the bit hash that is checked, not a plain sum: the sum is
# about 2e9, and differences in individual elements fall below its resolution
# (tested - the C++ version without -ffp-contract=off gives the same sum even
# though the elements differ).
echo
echo "Result agreement check (bit hash, across both languages):"
bad=0
for n in "${size_list[@]}"; do
    key=$(csv_size "$n")
    uniq_hashes=$(awk -F, -v n="$key" 'NR>1 && $4==n {print $10}' "$OUT" | sort -u)
    count=$(echo "$uniq_hashes" | grep -c . || true)
    if [ "$count" -eq 1 ]; then
        echo "  $SIZE_LABEL=$n  OK  ($uniq_hashes)"
    else
        echo "  $SIZE_LABEL=$n  ERROR: $count different hashes:" >&2
        awk -F, -v n="$key" 'NR>1 && $4==n {print $10, $1"/"$3" p="$5" "$6}' "$OUT" \
            | sort -u | sed 's/^/    /' >&2
        bad=1
    fi
done
[ "$bad" = "1" ] && { echo "Results are UNRELIABLE." >&2; exit 1; }

echo
echo "Plots: $HERE/.venv/bin/python $HERE/plot_series.py $OUT"
