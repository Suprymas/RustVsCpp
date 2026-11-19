#!/usr/bin/env python3
"""Code size (LOC) and cyclomatic complexity (CC).

The third evaluation criterion of the thesis. Until now the matmul and BFS
numbers were counted BY HAND, so they could be neither reproduced nor
checked. This script applies the same rule to all three tasks and both
languages.

What is counted
---------------
Not the whole project, only the PARALLELISATION code: the files that hold
thread creation, work splitting and merging. All variants call the same
compute kernel (`seq`/`par`), so these files are pure parallelisation code -
exactly what a programmer would write in addition when parallelising an
already working algorithm. Reading, CLI and measurement bookkeeping (`io`,
`main`) are not part of the metric: they are the same for all variants and
would only dilute the difference.

LOC rule (LOGICAL body lines)
-----------------------------
Excluded:
  * comments - line (`//`, `///`, `//!`) and block comments;
  * blank lines;
  * module imports (`use ...;`, `#include ...`, `#pragma once`);
  * the namespace wrapper (`namespace X {`) - on the Rust side the file
    ITSELF is a module, so there is no counterpart, and keeping it would give
    C++ a line for nothing;
  * lines consisting only of brackets and semicolons (`}`, `});`, `};`).

LOGICAL, not physical, lines are counted: physical lines are joined until the
brackets balance, and also when the next line starts with a dot (method
chain), a comma or a closing bracket.

Without this step the metric would measure formatting rather than code, and
to the disadvantage of C++ in particular: the function signature in
`cpp/matmul/src/omp_impl.cpp` is spread over SIX physical lines (`__restrict`
parameters), while the Rust counterpart fits on one. The ratio of physical
lines would be 2.60, of logical lines 2.67, but the first number would come
from signatures broken differently in the two files, the second from the
OpenMP loop running over integers so that the band start has to be computed
by hand. The second statement is the one the thesis wants to make.

CC rule (McCabe)
----------------
CC = 1 + the number of decision points in the file. Decision points: `if`,
`for`, `while`, `case`, `catch`, `&&`, `||`, the ternary `?`, the Rust `?`
operator (error propagation is a real branch) and `match` arms (`=>`).
`else` is not counted (McCabe), because it adds no new path - the `if`
already opened it.

A Rust pitfall that must be taken into account: `||` is not only a logical
"or" but also an ARGUMENT-LESS CLOSURE (`move || { ... }`), and in this code
there is one on every `spawn` line. Counting them would roughly double CC and
make Rust look more complex purely because of syntax. So closures are
recognised and not counted; the remaining `||` are treated as logical.

Usage:
  python3 code_metrics.py [--csv file.csv] [--detail]
"""

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (task, part, lang) -> files. "part" is what we compare:
#   low     - std::thread in both languages
#   high    - Rayon / OpenMP
#   common  - code shared by both parallel variants
#   helper  - a counterpart the language does NOT have (C++ affinity; Rust
#             gets it from the core_affinity library, so 0 lines on the Rust
#             side)
FILES = {
    ("matmul", "low", "rust"): ["rust/matmul/src/threads.rs"],
    ("matmul", "low", "cpp"): ["cpp/matmul/src/threads.cpp"],
    ("matmul", "high", "rust"): ["rust/matmul/src/rayon_impl.rs"],
    ("matmul", "high", "cpp"): ["cpp/matmul/src/omp_impl.cpp"],
    ("matmul", "common", "rust"): [],
    ("matmul", "common", "cpp"): ["cpp/matmul/src/par.hpp"],
    ("bfs", "low", "rust"): ["rust/bfs/src/threads.rs"],
    ("bfs", "low", "cpp"): ["cpp/bfs/src/threads.cpp"],
    ("bfs", "high", "rust"): ["rust/bfs/src/rayon_impl.rs"],
    ("bfs", "high", "cpp"): ["cpp/bfs/src/omp_impl.cpp"],
    ("bfs", "common", "rust"): ["rust/bfs/src/par.rs"],
    ("bfs", "common", "cpp"): ["cpp/bfs/src/par.hpp"],
    ("wordfreq", "low", "rust"): ["rust/wordfreq/src/threads.rs"],
    ("wordfreq", "low", "cpp"): ["cpp/wordfreq/src/threads.cpp"],
    ("wordfreq", "high", "rust"): ["rust/wordfreq/src/rayon_impl.rs"],
    ("wordfreq", "high", "cpp"): ["cpp/wordfreq/src/omp_impl.cpp"],
    ("wordfreq", "common", "rust"): ["rust/wordfreq/src/par.rs"],
    ("wordfreq", "common", "cpp"): ["cpp/wordfreq/src/par.hpp"],
}

# Affinity: Rust gets it from a library, C++ has to write it itself. It must
# be included in ΔLOC - otherwise the C++ side would look cheaper than it is.
AFFINITY = {"rust": [], "cpp": ["cpp/common/affinity.cpp"]}

TASKS = ["matmul", "bfs", "wordfreq"]
PARTS = ["low", "high", "common"]
LANGS = ["rust", "cpp"]

BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT = re.compile(r"//.*$", re.M)
STRING = re.compile(r'"(?:\\.|[^"\\])*"')
CHAR = re.compile(r"'(?:\\.|[^'\\])'")
IMPORT = re.compile(r"^\s*(use\s|#include\b|#pragma\s+once\b|extern\s+crate\b)")
ONLY_PUNCT = re.compile(r"^[\s{}()\[\];,]*$")
NAMESPACE = re.compile(r"^\s*namespace\s+[\w:]*\s*\{?\s*$")
CONTINUES = re.compile(r"^\s*[.,)\]?]")
CLOSURE = re.compile(r"(?:\bmove\s*)?\|\|")

DECISIONS = [
    re.compile(r"\bif\b"),
    re.compile(r"\bfor\b"),
    re.compile(r"\bwhile\b"),
    re.compile(r"\bcase\b"),
    re.compile(r"\bcatch\b"),
    re.compile(r"&&"),
    re.compile(r"=>"),          # Rust match sakos
]


def strip_noise(text, lang):
    """Removes comments and string literals - so that their content ends up
    in neither LOC nor CC (a "||" inside an error message is not a branch)."""
    text = BLOCK_COMMENT.sub(" ", text)
    text = LINE_COMMENT.sub("", text)
    text = STRING.sub('""', text)
    if lang == "cpp":
        text = CHAR.sub("' '", text)
    return text


def logical_lines(text):
    """Joins physical lines into logical ones (see the module docstring).

    Lines are joined until the brackets are balanced OR the next line is
    obviously a continuation (starts with a dot, a comma, a closing bracket).
    But an opening BLOCK always ends a logical line: without this exception
    `std::thread::scope(|scope| {` would swallow the whole thread body, because
    that open parenthesis is only closed by `});` eleven lines later.
    """
    raw = [ln for ln in text.splitlines() if ln.strip()]

    def balance(line):
        return (line.count("(") - line.count(")")
                + line.count("[") - line.count("]"))

    out, i = [], 0
    while i < len(raw):
        buf, depth = raw[i], balance(raw[i])
        i += 1
        while i < len(raw):
            end = buf.rstrip()
            closed = end.endswith(("{", "}", ";"))
            if not ((depth > 0 and not closed) or CONTINUES.match(raw[i])):
                break
            buf = f"{buf} {raw[i].strip()}"
            depth += balance(raw[i])
            i += 1
        out.append(buf)
    return out


def body_lines(text):
    out = []
    for line in logical_lines(text):
        if IMPORT.match(line):
            continue
        if NAMESPACE.match(line):
            continue
        if ONLY_PUNCT.match(line):
            continue
        out.append(line)
    return out


def complexity(text, lang):
    """CC = 1 + decision points. Also returns the breakdown - so that the
    number can be checked by eye rather than taken on trust from the script."""
    parts = {}
    body = "\n".join(body_lines(text))

    # Argument-less closures are removed FIRST, so that their "||" is not
    # taken for a logical "or" (see the module docstring). ONLY on the Rust
    # side: in C++ "||" is always a logical "or" (a lambda is written "[](...)"
    # there), and applying the same exception to that language would make CC
    # too small.
    closures = len(CLOSURE.findall(body)) if lang == "rust" else 0
    body_no_closure = CLOSURE.sub("@CLOSURE@", body) if lang == "rust" else body

    total = 0
    for rx in DECISIONS:
        k = rx.pattern.strip("\\b")
        n = len(rx.findall(body_no_closure))
        if n:
            parts[k] = n
        total += n

    n_or = len(re.findall(r"\|\|", body_no_closure))
    if n_or:
        parts["||"] = n_or
    total += n_or

    # Rust `?` is error propagation, C++ `?` the ternary operator. Both are
    # real branches, so they are counted the same way.
    n_q = len(re.findall(r"\?", body_no_closure))
    if n_q:
        parts["?"] = n_q
    total += n_q

    parts["_closures"] = closures
    return 1 + total, parts


def measure(rel_paths, lang):
    loc, cc_total, detail = 0, 0, {}
    for rel in rel_paths:
        path = ROOT / rel
        if not path.is_file():
            sys.exit(f"error: {path} does not exist")
        text = strip_noise(path.read_text(encoding="utf-8"), lang)
        n = len(body_lines(text))
        cc, parts = complexity(text, lang)
        loc += n
        # File CCs are added up discarding the repeated base (every file adds
        # its own 1), so that a set with several files is not penalised.
        cc_total += cc - 1
        detail[rel] = (n, cc, parts)
    return loc, cc_total + 1, detail


def main():
    ap = argparse.ArgumentParser(description="Code size and cyclomatic complexity")
    ap.add_argument("--csv", default=str(ROOT / "harness/results/code-metrics.csv"))
    ap.add_argument("--detail", action="store_true", help="show every file separately")
    args = ap.parse_args()

    rows = []
    for task in TASKS:
        for part in PARTS:
            for lang in LANGS:
                files = FILES[(task, part, lang)]
                loc, cc, detail = measure(files, lang)
                rows.append({"task": task, "part": part, "lang": lang,
                             "loc": loc, "cc": cc if files else 0,
                             "files": ";".join(files)})
                if args.detail and files:
                    for rel, (n, c, parts) in detail.items():
                        bits = " ".join(f"{k}={v}" for k, v in sorted(parts.items()) if v)
                        print(f"    {rel:38s} LOC {n:3d}  CC {c:3d}   {bits}")

    for lang in LANGS:
        loc, cc, _ = measure(AFFINITY[lang], lang)
        rows.append({"task": "-", "part": "affinity", "lang": lang,
                     "loc": loc, "cc": cc if AFFINITY[lang] else 0,
                     "files": ";".join(AFFINITY[lang])})

    def get(task, part, lang, field):
        for r in rows:
            if r["task"] == task and r["part"] == part and r["lang"] == lang:
                return r[field]
        return 0

    print("LOC (body lines: no comments, blank lines, imports or lone brackets)")
    print()
    print(f"{'task':10s} {'part':9s} {'Rust':>6s} {'C++':>6s} {'C++/Rust':>9s}"
          f" | {'CC Rust':>8s} {'CC C++':>7s}")
    for task in TASKS:
        for part in PARTS:
            r, c = get(task, part, "rust", "loc"), get(task, part, "cpp", "loc")
            rc, cc_ = get(task, part, "rust", "cc"), get(task, part, "cpp", "cc")
            ratio = f"{c / r:.2f}" if r else "-"
            print(f"{task:10s} {part:9s} {r:6d} {c:6d} {ratio:>9s} | {rc:8d} {cc_:7d}")
        print()
    r, c = get("-", "affinity", "rust", "loc"), get("-", "affinity", "cpp", "loc")
    print(f"{'(all)':10s} {'affinity':9s} {r:6d} {c:6d}"
          "   <- Rust takes it from the core_affinity library")

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "part", "lang", "loc", "cc", "files"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
