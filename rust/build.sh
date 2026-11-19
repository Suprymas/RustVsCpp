#!/usr/bin/env bash
#
# rust/build.sh - builds all three Rust programs with a diagnostics log.
#
# Symmetric counterpart of cpp/build.sh (decision 6). On the C++ side the
# diagnostics come from GCC's SARIF output (`-fdiagnostics-format=sarif-stderr`),
# here - from `cargo build --message-format=json`. Both sides have a field from
# which the error-code histogram for section 2.1 is built:
#   GCC   -> "ruleId", e.g. "-Wunused-variable"
#   rustc -> "code.code", e.g. "unused_variables" or "E0502"
#
# The log goes to harness/logs/build-rust-<date>.json.
#
# IMPORTANT NOTE ON THE LIMITS OF THIS LOG. It records only the diagnostics of
# THAT run. Errors made and fixed while writing the code do not end up here -
# just as they did not end up in the C++ logs (all fourteen of them are
# empty, because the final state of the code compiles cleanly). So the
# histogram shows the STATE of the code, not the obstacles the programmer
# ran into; the latter are recorded by hand (see CLAUDE.md "Log of obstacles
# encountered").
#
# Usage:
#   ./build.sh              all three programs
#   ./build.sh matmul bfs   only the listed ones
#   ./build.sh --clean      remove target/

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LOGDIR="$ROOT/harness/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/build-rust-$(date +%Y%m%d-%H%M%S).json"

PROGRAMS=(matmul bfs wordfreq)

if [ "${1:-}" = "--clean" ]; then
    for p in "${PROGRAMS[@]}"; do
        rm -rf "$HERE/$p/target"
        echo "removed: rust/$p/target"
    done
    exit 0
fi

if [ $# -gt 0 ]; then
    PROGRAMS=("$@")
fi

# Same flags as in the Cargo.toml profile; target-cpu=native matches the C++
# side's -march=native.
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"

echo "[" > "$LOG"
first=1
status=0

for prog in "${PROGRAMS[@]}"; do
    dir="$HERE/$prog"
    [ -d "$dir" ] || { echo "error: $dir does not exist" >&2; exit 1; }
    echo "== rust/$prog"

    raw="$LOGDIR/.build-rust-$prog.jsonl"
    if ! (cd "$dir" && cargo build --release --message-format=json) > "$raw" 2>/dev/null; then
        status=1
    fi

    # Only compiler diagnostics are taken from the cargo stream. Each is
    # recorded with the program name so the log shows where it came from.
    python3 - "$raw" "$prog" >> "$LOG" <<'PY'
import json, sys
path, prog = sys.argv[1], sys.argv[2]
out = []
for line in open(path):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        continue
    if rec.get("reason") != "compiler-message":
        continue
    m = rec.get("message", {})
    code = (m.get("code") or {}).get("code")
    out.append({
        "program": prog,
        "level": m.get("level"),
        "code": code,
        "message": m.get("message"),
        "location": [s.get("file_name") for s in m.get("spans", []) if s.get("is_primary")],
    })
print(json.dumps(out, ensure_ascii=False, indent=2)[1:-1].rstrip() + ("," if out else ""))
PY
    rm -f "$raw"
    first=0
done

# Remove the dangling comma in the last entry.
python3 - "$LOG" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read().rstrip()
s = re.sub(r",\s*$", "", s) + "\n]\n"
open(p, "w").write(s)
PY

echo
echo "Log: $LOG"
echo
echo "Diagnostics histogram (rustc codes, counterpart of GCC ruleId):"
python3 - "$LOG" <<'PY'
import json, sys
from collections import Counter
d = json.load(open(sys.argv[1]))
if not d:
    print("  (none) - all three programs compile cleanly")
    print("  This measures the STATE of the code, not obstacles encountered; see the top of this script.")
else:
    c = Counter((x["level"], x["code"] or "-") for x in d)
    for (lvl, code), n in c.most_common():
        print(f"  {n:4d}  {lvl:8s} {code}")
PY

exit $status
