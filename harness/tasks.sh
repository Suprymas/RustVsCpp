#!/usr/bin/env bash
#
# tasks.sh - the ONLY place where the tasks differ.
#
# This file is sourced by both run_series.sh (timing) and run_memory.sh (peak
# memory). The reason is the same one that once led to run_matmul.sh being
# merged with BFS: if binary paths, size labels or level lists lived in two
# places, sooner or later they would drift apart, and then the timing and
# memory tables would call different things by the same name.
#
# Usage:
#   . "$HERE/tasks.sh"
#   task_config matmul|bfs|wordfreq

# Sets the task variables. ROOT must be set beforehand.
task_config() {
    case "$1" in
        matmul)
            RUST_BIN="$ROOT/rust/matmul/target/release/matmul"
            CPP_BIN="$ROOT/cpp/matmul/matmul"
            SIZE_FLAG="--n"
            SIZE_LABEL="N"
            DEF_SIZES="512,1024,2048"
            # At N=512 one run takes about 1 ms, so the relative noise (thread
            # start-up, scheduler, timer resolution) is largest, while the cost
            # is lowest - 512 is the fastest size.
            DEF_REPS_MAP="512=30"
            QUICK_SIZES="512,1024"
            ;;
        bfs)
            RUST_BIN="$ROOT/rust/bfs/target/release/bfs"
            CPP_BIN="$ROOT/cpp/bfs/bfs"
            SIZE_FLAG="--scale"
            SIZE_LABEL="scale"
            DEF_SIZES="20,22,24"
            # The opposite ratio to matmul: here the most expensive size is
            # scale 24 (sequential version ~3 s per run + reading a 2 GiB
            # file), so repetitions are reduced there and increased for the
            # smallest one.
            DEF_REPS_MAP="20=15,22=10,24=5"
            QUICK_SIZES="20"
            ;;
        wordfreq)
            RUST_BIN="$ROOT/rust/wordfreq/target/release/wordfreq"
            CPP_BIN="$ROOT/cpp/wordfreq/wordfreq"
            SIZE_FLAG="--size"
            SIZE_LABEL="corpus"
            # The size is passed as a label ("512M") and written to the CSV
            # as a number (512 MiB) - so the "n" column stays NUMERIC in all
            # three tasks.
            DEF_SIZES="512M,2G,8G"
            DEF_REPS_MAP="512M=10,2G=10,8G=5"
            QUICK_SIZES="512M"
            # In this task a level has TWO strategies, so they are put into
            # the level name: "low-partial" etc. See variant_for and
            # strategy_args.
            LEVELS="low-partial high-partial low-shared high-shared"
            # "shared" is the naive strategy and it does not scale: at 512M
            # with p=32 one run takes ~40 s, at 8G it would take ~11 min. The
            # full matrix would take several days, and the shape of the curve
            # is already known (Sp falls monotonically from 0.82 to 0.08). So
            # it is measured at 512M only.
            SHARED_SIZES="512M"
            SHARED_REPS=5
            ;;
        "")
            echo "error: --task matmul|bfs|wordfreq is required" >&2; return 1 ;;
        *)
            echo "error: unknown task: $1" >&2; return 1 ;;
    esac

    TASK="$1"
    # Default levels for the tasks that have no strategies.
    [ -n "${LEVELS:-}" ] || LEVELS="low high"
}

# Binary for a language.
bin_for() {
    case "$1" in
        rust) echo "$RUST_BIN" ;;
        cpp)  echo "$CPP_BIN" ;;
        *) echo "unknown language: $1" >&2; return 1 ;;
    esac
}

# Variant name for a language and level. The level is what we compare; the
# variant name is just the name of the concrete tool in that language.
variant_for() {
    # The level may carry a strategy suffix ("low-partial") - it is stripped
    # here and passed to the program as a separate flag (see strategy_args).
    set -- "$1" "${2%%-*}"
    case "$1:$2" in
        rust:base|cpp:base) echo "seq" ;;
        rust:low|cpp:low) echo "threads" ;;
        rust:high) echo "rayon" ;;
        cpp:high)  echo "omp" ;;
        *) echo "unknown level: $2" >&2; return 1 ;;
    esac
}

# The size as it goes into the CSV "n" column.
#
# In the word frequency task the program gets a label ("512M"), while the CSV
# gets mebibytes (512), so that the "n" column stays NUMERIC in all three
# tasks - otherwise the plotting script could not sort the sizes.
csv_size() {
    case "$TASK" in
        wordfreq)
            case "$1" in
                *M) echo "${1%M}" ;;
                *G) echo $(( ${1%G} * 1024 )) ;;
                *)  echo "$1" ;;
            esac
            ;;
        *) echo "$1" ;;
    esac
}

# Strategy flag from the level name. Empty if the level has no suffix.
strategy_args() {
    case "$1" in
        *-*) echo "--strategy ${1##*-}" ;;
        *)   echo "" ;;
    esac
}

# Whether this level is measured at this size. Only "shared" outside
# SHARED_SIZES does not apply.
level_applies() {
    case "$1" in
        *-shared)
            [ -n "${SHARED_SIZES:-}" ] || return 1
            printf '%s\n' "$SHARED_SIZES" | tr ',' '\n' | grep -qx "$2" || return 1
            ;;
    esac
    return 0
}
