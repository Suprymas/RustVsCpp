#!/usr/bin/env bash
#
# build.sh - builds the C++ side through a wrapper script.
#
# CLAUDE.md decision 6: compilation goes through a script that records error
# and warning codes. On the Rust side the same is done by
# `cargo build --release --message-format=json`.
#
# The C++ side uses `-fdiagnostics-format=sarif-stderr`. Note: GCC 16 NO
# LONGER has the `json` value (only sarif-file and sarif-stderr are
# available), so the `-fdiagnostics-format=json` written in CLAUDE.md is
# outdated. SARIF is even more convenient here - every diagnostic has a
# `ruleId` (e.g. "-Wunused-variable"), i.e. exactly what rustc lint names
# provide, and the histogram becomes comparable across languages.
#
# The log goes to harness/logs/build-cpp-<date>.sarif.json (thesis section 2.1).
#
# Usage:
#   ./build.sh [--clean] [--quiet]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD="$HERE/build"
MIMALLOC="$HERE/third_party/mimalloc"
LOGDIR="$ROOT/harness/logs"
LOG="$LOGDIR/build-cpp-$(date +%Y%m%d-%H%M%S).sarif.json"

CLEAN=0
QUIET=0
while [ $# -gt 0 ]; do
    case "$1" in
        --clean) CLEAN=1; shift ;;
        --quiet) QUIET=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

CXX=${CXX:-g++}
CC=${CC:-gcc}

# Matmul compiler flags. Must match the description in CLAUDE.md.
#
# -ffp-contract=off is REQUIRED: by default GCC fuses a*b+c into FMA, while
# rustc does not. Without this flag the C++ and Rust checksums would not
# match, and decision 9 (bit-for-bit) would not hold.
#
# -Wall -Wextra are the counterpart of rustc's default lints: without them the
# C++ side would stay silent where the Rust side warns, and the warning
# counts would not be comparable.
CXXFLAGS=(
    -std=c++20
    -O3
    -march=native
    -flto
    -fopenmp
    -pthread
    -ffp-contract=off
    -Wall
    -Wextra
    -I"$MIMALLOC/include"
)
LDFLAGS=(-flto -fopenmp -pthread)

# mimalloc is compiled SEPARATELY and with different flags - on purpose. On
# the Rust side it is compiled by the `cc` crate, which gets neither the
# RUSTFLAGS `-C target-cpu=native` nor LTO. Matched literally with
# libmimalloc-sys/build.rs; see third_party/mimalloc/VERSION.md.
MIFLAGS=(
    -O3
    -DMI_DEBUG=0
    -DMI_BUILD_RELEASE
    -DNDEBUG
    -ftls-model=initial-exec
    -I"$MIMALLOC/include"
    -I"$MIMALLOC/src"
)

[ "$CLEAN" = "1" ] && rm -rf "$BUILD" "$HERE/matmul/matmul" "$HERE/bfs/bfs" "$HERE/wordfreq/wordfreq"
mkdir -p "$BUILD" "$LOGDIR"

say() { [ "$QUIET" = "1" ] || echo "$@"; }

# --------------------------------------------------------------- mimalloc

MI_LIB="$BUILD/libmimalloc.a"
if [ ! -f "$MI_LIB" ] || [ "$MIMALLOC/src/static.c" -nt "$MI_LIB" ]; then
    say "mimalloc: compiling (once)"
    "$CC" "${MIFLAGS[@]}" -c "$MIMALLOC/src/static.c" -o "$BUILD/mimalloc.o"
    ar rcs "$MI_LIB" "$BUILD/mimalloc.o"
else
    say "mimalloc: already built"
fi

# --------------------------------------------------------------- programs

# GCC emits a separate SARIF document for every translation unit, so they are
# merged into one array with a note of which file each belongs to.
echo "[" > "$LOG"
first=1
status=0
had_diag=0

# build_program <output path> <sources...>
#
# Object files are named with a program prefix, because the programs each
# have seq.cpp, io.cpp and main.cpp - without the prefix they would overwrite
# each other.
build_program() {
    local out="$1"; shift
    local tag; tag="$(basename "$(dirname "$out")")"
    local objs=()
    local src obj diag

    for src in "$@"; do
        obj="$BUILD/${tag}-$(basename "${src%.cpp}").o"
        objs+=("$obj")
        say "compiling $tag/$(basename "$src")"
        diag="$BUILD/${tag}-$(basename "${src%.cpp}").sarif.json"
        if ! "$CXX" "${CXXFLAGS[@]}" -fdiagnostics-format=sarif-stderr -c "$src" -o "$obj" 2> "$diag"; then
            status=1
        fi

        [ "$first" = "0" ] && echo "," >> "$LOG"
        printf '{"program": "%s", "file": "%s", "sarif": ' "$tag" "$(basename "$src")" >> "$LOG"
        cat "$diag" >> "$LOG"
        echo "}" >> "$LOG"
        first=0

        # SARIF is always emitted, even with no diagnostics at all, so we check
        # whether the `results` array is non-empty. If it is, the same file is
        # compiled again syntax-only so the on-screen text is human-readable.
        if grep -q '"ruleId"' "$diag"; then
            had_diag=1
            "$CXX" "${CXXFLAGS[@]}" -fsyntax-only "$src" 2>&1 | head -40 >&2 || true
        fi
    done

    if [ "$status" != "0" ]; then
        return
    fi
    say "linking $tag"
    "$CXX" "${LDFLAGS[@]}" "${objs[@]}" "$MI_LIB" -o "$out"
    say "done: $out"
}

build_program "$HERE/matmul/matmul" \
    "$HERE/matmul/src/seq.cpp" \
    "$HERE/matmul/src/threads.cpp" \
    "$HERE/matmul/src/omp_impl.cpp" \
    "$HERE/common/affinity.cpp" \
    "$HERE/matmul/src/io.cpp" \
    "$HERE/matmul/src/main.cpp"

build_program "$HERE/bfs/bfs" \
    "$HERE/bfs/src/seq.cpp" \
    "$HERE/bfs/src/threads.cpp" \
    "$HERE/bfs/src/omp_impl.cpp" \
    "$HERE/common/affinity.cpp" \
    "$HERE/bfs/src/io.cpp" \
    "$HERE/bfs/src/main.cpp"

build_program "$HERE/wordfreq/wordfreq" \
    "$HERE/wordfreq/src/threads.cpp" \
    "$HERE/wordfreq/src/omp_impl.cpp" \
    "$HERE/common/affinity.cpp" \
    "$HERE/wordfreq/src/io.cpp" \
    "$HERE/wordfreq/src/main.cpp"

echo "]" >> "$LOG"

if [ "$status" != "0" ]; then
    echo "BUILD FAILED. Diagnostics: $LOG" >&2
    exit 1
fi

say "diagnostics log: $LOG"

# The summary is printed here, not put into the log - otherwise it would break the JSON.
if [ "$QUIET" != "1" ]; then
    if [ "$had_diag" = "0" ]; then
        echo "diagnostics: 0 errors, 0 warnings"
    else
        errs=$(grep -c '"level": "error"' "$LOG" || true)
        warns=$(grep -c '"level": "warning"' "$LOG" || true)
        echo "diagnostics: $errs errors, $warns warnings"
        grep -o '"ruleId": "[^"]*"' "$LOG" | sed 's/.*: "//; s/"$//' | sort | uniq -c | sort -rn
    fi
fi
