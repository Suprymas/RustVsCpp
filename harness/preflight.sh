#!/usr/bin/env bash
#
# preflight.sh - preparing, checking and restoring the measurement environment.
#
# Reason: this CPU's boost range is 2.0-5.19 GHz, and core rankings vary
# 166..236 (CCD0 systematically better than CCD1). In practice the same
# sequential N=2048 run took 0.61 s on a cooled-down machine and 1.05 s on a
# hot one. That is 1.7x and more than any plausible difference between the
# languages, so measurements without a fixed frequency are worthless.
#
# Usage:
#   ./preflight.sh status     - show the current state (no root needed)
#   ./preflight.sh lock       - disable boost and lock the frequency (root needed)
#   ./preflight.sh check      - measure frequency stability under load
#   ./preflight.sh restore    - restore the original state (root needed)
#
# Usual sequence: lock -> check -> measurements -> restore

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$HERE/.freq-state"
LOG_DIR="$HERE/logs"
CPU_ROOT="/sys/devices/system/cpu"
BOOST_FILE="$CPU_ROOT/cpufreq/boost"

# Load duration and sampling interval in "check" mode.
CHECK_SECONDS="${CHECK_SECONDS:-30}"
CHECK_INTERVAL="${CHECK_INTERVAL:-1}"
# Threshold above which frequency variation is considered too large (percent).
SPREAD_LIMIT="${SPREAD_LIMIT:-2.0}"

cpus() { ls -d "$CPU_ROOT"/cpu[0-9]* 2>/dev/null | sort -V; }

khz_to_mhz() { echo $(( ${1:-0} / 1000 )); }

need_root() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "Root privileges needed - re-running via sudo." >&2
        exec sudo -- "$0" "$@"
    fi
}

# ---------------------------------------------------------------- status

read_temps() {
    local h name
    for h in /sys/class/hwmon/hwmon*; do
        [ -r "$h/name" ] || continue
        name=$(cat "$h/name")
        case "$name" in
            k10temp|acpitz)
                [ -r "$h/temp1_input" ] && \
                    printf "  %-10s %s C\n" "$name" "$(( $(cat "$h/temp1_input") / 1000 ))"
                ;;
        esac
    done
}

do_status() {
    echo "=== CPU frequency state ==="
    if [ -r "$BOOST_FILE" ]; then
        local b; b=$(cat "$BOOST_FILE")
        echo "  boost                 $b  ($([ "$b" = "0" ] && echo disabled || echo ENABLED))"
    else
        echo "  boost                 n/a (no $BOOST_FILE)"
    fi
    echo "  driver                $(cat "$CPU_ROOT/cpu0/cpufreq/scaling_driver" 2>/dev/null || echo n/a)"
    echo "  governor              $(cat "$CPU_ROOT/cpu0/cpufreq/scaling_governor" 2>/dev/null || echo n/a)"
    echo "  EPP                   $(cat "$CPU_ROOT/cpu0/cpufreq/energy_performance_preference" 2>/dev/null || echo n/a)"
    echo "  prefcore              $(cat "$CPU_ROOT/amd_pstate/prefcore" 2>/dev/null || echo n/a)"

    local min max pinned=1 first_min="" first_max=""
    for c in $(cpus); do
        [ -r "$c/cpufreq/scaling_min_freq" ] || continue
        min=$(cat "$c/cpufreq/scaling_min_freq"); max=$(cat "$c/cpufreq/scaling_max_freq")
        [ -z "$first_min" ] && { first_min=$min; first_max=$max; }
        # Considered locked only when all cores have the same
        # min = max value.
        if [ "$min" != "$max" ] || [ "$min" != "$first_min" ]; then pinned=0; fi
    done
    echo "  scaling_min_freq      $(khz_to_mhz "$first_min") MHz"
    echo "  scaling_max_freq      $(khz_to_mhz "$first_max") MHz"
    if [ "$pinned" = "1" ]; then
        echo "  frequency             LOCKED at $(khz_to_mhz "$first_min") MHz"
    else
        echo "  frequency             free (min != max or differs between cores)"
    fi

    echo
    echo "=== Temperature ==="
    read_temps

    echo
    if [ -f "$STATE_FILE" ]; then
        echo "Saved original state: $STATE_FILE (to restore: ./preflight.sh restore)"
    else
        echo "No original state saved - 'lock' has not been run yet."
    fi
}

# ---------------------------------------------------------------- lock

do_lock() {
    need_root "$@"

    if [ -f "$STATE_FILE" ]; then
        echo "Original state already saved, will not be overwritten: $STATE_FILE"
    else
        echo "Saving original state -> $STATE_FILE"
        {
            echo "# preflight.sh saved state, $(date -Iseconds)"
            [ -r "$BOOST_FILE" ] && echo "boost $(cat "$BOOST_FILE")"
            for c in $(cpus); do
                [ -r "$c/cpufreq/scaling_min_freq" ] || continue
                echo "cpu $(basename "$c") $(cat "$c/cpufreq/scaling_min_freq") $(cat "$c/cpufreq/scaling_max_freq") $(cat "$c/cpufreq/scaling_governor")"
            done
        } > "$STATE_FILE"
    fi

    # 1. Disable boost. This removes the upper ceiling and at the same time
    #    equalises the cores: ranking differences reflect boost headroom, and
    #    at the base frequency all cores are the same.
    if [ -w "$BOOST_FILE" ]; then
        echo 0 > "$BOOST_FILE"
        echo "boost disabled"
    else
        echo "WARNING: $BOOST_FILE not writable - boost not disabled" >&2
    fi

    # Wait for the driver to recompute the new limits.
    sleep 1

    # 2. Governor and EPP to performance.
    for c in $(cpus); do
        [ -w "$c/cpufreq/scaling_governor" ] && echo performance > "$c/cpufreq/scaling_governor" || true
        [ -w "$c/cpufreq/energy_performance_preference" ] && \
            echo performance > "$c/cpufreq/energy_performance_preference" || true
    done

    # 3. Lock the frequency. Disabling boost sets a ceiling but does not stop
    #    the frequency from dropping at a thermal or power limit, so min is set
    #    equal to max.
    local target=""
    for c in $(cpus); do
        [ -w "$c/cpufreq/scaling_min_freq" ] || continue
        local max; max=$(cat "$c/cpufreq/scaling_max_freq")
        [ -z "$target" ] && target=$max
        echo "$target" > "$c/cpufreq/scaling_min_freq"
        echo "$target" > "$c/cpufreq/scaling_max_freq"
    done
    echo "frequency locked at $(khz_to_mhz "$target") MHz on all cores"

    echo
    do_status
    echo
    echo "Next: ./preflight.sh check  (check stability under load)"
}

# ---------------------------------------------------------------- restore

do_restore() {
    need_root "$@"
    if [ ! -f "$STATE_FILE" ]; then
        echo "error: no $STATE_FILE - nothing to restore" >&2
        exit 1
    fi

    # Boost is restored FIRST. While it is off, the kernel clips
    # scaling_max_freq at the base frequency, and writing anything higher
    # would fail - the system would stay locked at 3000 MHz.
    local saved_boost
    saved_boost=$(awk '/^boost /{print $2}' "$STATE_FILE")
    if [ -n "$saved_boost" ] && [ -w "$BOOST_FILE" ]; then
        echo "$saved_boost" > "$BOOST_FILE"
        sleep 1
    fi

    # Only now are the ceilings lifted, so that the old min values always fit.
    for c in $(cpus); do
        [ -w "$c/cpufreq/scaling_max_freq" ] && \
            cat "$c/cpufreq/cpuinfo_max_freq" > "$c/cpufreq/scaling_max_freq" || true
    done

    # Line format: cpu <name> <min> <max> <governor> - five fields.
    while read -r kind name min max _gov; do
        case "$kind" in
            cpu)
                local dir="$CPU_ROOT/$name/cpufreq"
                [ -w "$dir/scaling_min_freq" ] && echo "$min" > "$dir/scaling_min_freq" || true
                [ -w "$dir/scaling_max_freq" ] && echo "$max" > "$dir/scaling_max_freq" || true
                ;;
        esac
    done < <(grep -v '^#' "$STATE_FILE")

    # Governors separately, because their column is the fifth.
    awk '/^cpu /{print $2, $5}' "$STATE_FILE" | while read -r name gov; do
        [ -n "$gov" ] && [ -w "$CPU_ROOT/$name/cpufreq/scaling_governor" ] && \
            echo "$gov" > "$CPU_ROOT/$name/cpufreq/scaling_governor" || true
    done

    # Verification: without it a silently failed write would go unnoticed
    # (exactly that happened - the system stayed locked at 3000 MHz while the
    # script reported success).
    local ok=1 want_min want_max got_min got_max got_boost
    want_min=$(awk '/^cpu cpu0 /{print $3}' "$STATE_FILE")
    want_max=$(awk '/^cpu cpu0 /{print $4}' "$STATE_FILE")
    got_min=$(cat "$CPU_ROOT/cpu0/cpufreq/scaling_min_freq")
    got_max=$(cat "$CPU_ROOT/cpu0/cpufreq/scaling_max_freq")
    got_boost=$(cat "$BOOST_FILE" 2>/dev/null || echo "")

    [ "$got_min" = "$want_min" ] || { echo "FAILED: scaling_min_freq $(khz_to_mhz "$got_min") MHz, expected $(khz_to_mhz "$want_min")" >&2; ok=0; }
    [ "$got_max" = "$want_max" ] || { echo "FAILED: scaling_max_freq $(khz_to_mhz "$got_max") MHz, expected $(khz_to_mhz "$want_max")" >&2; ok=0; }
    [ -z "$saved_boost" ] || [ "$got_boost" = "$saved_boost" ] || { echo "FAILED: boost $got_boost, expected $saved_boost" >&2; ok=0; }

    if [ "$ok" = "1" ]; then
        rm -f "$STATE_FILE"
        echo "Original state restored and verified."
    else
        echo "State NOT FULLY restored - $STATE_FILE is kept, you can try again." >&2
    fi
    echo
    do_status
    [ "$ok" = "1" ]
}

# ---------------------------------------------------------------- check

# Load all threads and watch whether the frequency drops. If it does, there
# is no point starting measurements - the results will reflect thermals, not
# languages.
do_check() {
    local ncpu; ncpu=$(nproc)
    local ts; ts=$(date +%Y%m%d-%H%M%S)
    mkdir -p "$LOG_DIR"
    local log="$LOG_DIR/preflight-$ts.log"

    echo "Load: $ncpu threads, $CHECK_SECONDS s. Log: $log"
    echo

    local pids=()
    local i
    for ((i = 0; i < ncpu; i++)); do
        ( while :; do :; done ) &
        pids+=($!)
    done
    # The load is stopped in any case, including on Ctrl-C.
    trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT INT TERM

    local samples=() n=0 sum=0 lo=99999999 hi=0
    local elapsed=0
    while [ "$elapsed" -lt "$CHECK_SECONDS" ]; do
        sleep "$CHECK_INTERVAL"
        elapsed=$((elapsed + CHECK_INTERVAL))

        # Mean frequency across all cores at this moment.
        local total=0 count=0 f
        for c in $(cpus); do
            [ -r "$c/cpufreq/scaling_cur_freq" ] || continue
            f=$(cat "$c/cpufreq/scaling_cur_freq" 2>/dev/null || echo 0)
            total=$((total + f)); count=$((count + 1))
        done
        [ "$count" -eq 0 ] && continue
        local avg=$((total / count / 1000))
        samples+=("$avg")
        sum=$((sum + avg)); n=$((n + 1))
        [ "$avg" -lt "$lo" ] && lo=$avg
        [ "$avg" -gt "$hi" ] && hi=$avg

        local temp="n/a"
        for h in /sys/class/hwmon/hwmon*; do
            [ -r "$h/name" ] && [ "$(cat "$h/name")" = "k10temp" ] && \
                temp="$(( $(cat "$h/temp1_input") / 1000 ))" && break
        done
        printf "  %3ds  %5d MHz  %s C\n" "$elapsed" "$avg" "$temp"
    done

    kill "${pids[@]}" 2>/dev/null || true
    trap - EXIT INT TERM

    [ "$n" -eq 0 ] && { echo "error: failed to read the frequency" >&2; exit 1; }

    local mean=$((sum / n))
    local spread; spread=$(awk -v h="$hi" -v l="$lo" -v m="$mean" \
        'BEGIN { if (m > 0) printf "%.2f", (h - l) * 100.0 / m; else print "0" }')

    {
        echo "# preflight check, $(date -Iseconds)"
        echo "duration_s $CHECK_SECONDS"
        echo "threads $ncpu"
        echo "freq_min_MHz $lo"
        echo "freq_max_MHz $hi"
        echo "freq_mean_MHz $mean"
        echo "spread_pct $spread"
        echo "kernel $(uname -r)"
        echo "rustc $(rustc --version 2>/dev/null || echo n/a)"
        echo "gcc $(g++ --version 2>/dev/null | head -1 || echo n/a)"
        echo "boost $(cat "$BOOST_FILE" 2>/dev/null || echo n/a)"
        echo "governor $(cat "$CPU_ROOT/cpu0/cpufreq/scaling_governor" 2>/dev/null || echo n/a)"
        printf "samples"; printf " %s" "${samples[@]}"; printf "\n"
    } > "$log"

    echo
    echo "  min $lo MHz, max $hi MHz, mean $mean MHz, spread $spread %"
    echo

    if awk -v s="$spread" -v lim="$SPREAD_LIMIT" 'BEGIN { exit !(s <= lim) }'; then
        echo "OK: frequency stable (spread <= $SPREAD_LIMIT %). Ready to measure."
    else
        echo "WARNING: frequency varies by $spread % (limit $SPREAD_LIMIT %)."
        echo "Possible causes: 'lock' not run, fan curve too low, or a power"
        echo "limit. Consider setting the Performance fan profile and a fixed"
        echo "Sustained Power Limit in the BIOS."
    fi
    echo
    echo "Environment record for the appendices: $log"
}

# ---------------------------------------------------------------- main

case "${1:-status}" in
    status)  do_status ;;
    lock)    do_lock "$@" ;;
    restore) do_restore "$@" ;;
    check)   do_check ;;
    *)
        echo "Usage: $0 {status|lock|check|restore}" >&2
        exit 1
        ;;
esac
