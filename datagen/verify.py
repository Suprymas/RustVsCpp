#!/usr/bin/env python3
"""Independent check of the generated data.

The script reads the datagen output files directly from their binary format
and checks the structural properties that the Rust and C++ implementations
rely on. Used once after the data are generated; takes no part in the
measurements.

Usage:  python3 verify.py data/
"""

import struct
import sys
from pathlib import Path

MATRIX_MAGIC = b"RMTX"
CSR_MAGIC = b"RCSR"


def fail(msg):
    print(f"  ERROR: {msg}")
    return False


def check_matrix(path):
    print(f"{path.name}:")
    with open(path, "rb") as f:
        head = f.read(24)
        magic, version, rows, cols = struct.unpack("<4sIQQ", head)
        if magic != MATRIX_MAGIC:
            return fail(f"bad magic {magic!r}")
        expected = 24 + rows * cols * 8
        actual = path.stat().st_size
        if actual != expected:
            return fail(f"file size {actual}, expected {expected}")

        # Check the first and the last row.
        first = struct.unpack(f"<{cols}d", f.read(cols * 8))
        f.seek(24 + (rows - 1) * cols * 8)
        last = struct.unpack(f"<{cols}d", f.read(cols * 8))
        vals = first + last
        if not all(0.0 <= v < 1.0 for v in vals):
            return fail("values outside the interval [0, 1)")
        mean = sum(vals) / len(vals)
        print(f"  v{version} {rows}x{cols}, sample mean {mean:.4f} (expected ~0.5)")
    return True


def check_csr(path):
    print(f"{path.name}:")
    with open(path, "rb") as f:
        magic, version, n, m, s = struct.unpack("<4sIQQQ", f.read(32))
        if magic != CSR_MAGIC:
            return fail(f"bad magic {magic!r}")
        expected = 32 + (n + 1) * 8 + m * 4 + s * 4
        actual = path.stat().st_size
        if actual != expected:
            return fail(f"file size {actual}, expected {expected}")

        offsets = struct.unpack(f"<{n + 1}Q", f.read((n + 1) * 8))
        neighbors = struct.unpack(f"<{m}I", f.read(m * 4))
        sources = struct.unpack(f"<{s}I", f.read(s * 4))

    if offsets[0] != 0 or offsets[-1] != m:
        return fail("offsets array endpoints are wrong")
    if any(offsets[i] > offsets[i + 1] for i in range(n)):
        return fail("offsets array is not non-decreasing")
    if any(v >= n for v in neighbors):
        return fail("neighbour index out of range")
    if any(v >= n for v in sources):
        return fail("search source out of range")

    # Symmetry, absence of self-loops, sorted neighbours.
    adj = [set(neighbors[offsets[v]:offsets[v + 1]]) for v in range(n)]
    for v in range(n):
        if v in adj[v]:
            return fail(f"self-loop at vertex {v}")
        row = neighbors[offsets[v]:offsets[v + 1]]
        if list(row) != sorted(set(row)):
            return fail(f"neighbours not sorted or duplicated at vertex {v}")
        for w in row:
            if v not in adj[w]:
                return fail(f"missing reverse edge {w}->{v}")

    # Connectivity.
    seen = [False] * n
    seen[0] = True
    queue, head = [0], 0
    while head < len(queue):
        v = queue[head]
        head += 1
        for w in neighbors[offsets[v]:offsets[v + 1]]:
            if not seen[w]:
                seen[w] = True
                queue.append(w)
    if len(queue) != n:
        return fail(f"graph not connected: reached {len(queue)} of {n}")

    degrees = [offsets[v + 1] - offsets[v] for v in range(n)]
    print(f"  v{version} n={n} m={m}, degree: mean {m / n:.2f}, max {max(degrees)}, "
          f"min {min(degrees)}; sources {list(sources)}")
    return True


def check_corpus(path):
    print(f"{path.name}:")
    with open(path, "rb") as f:
        head = f.read(4096)
        f.seek(max(0, path.stat().st_size - 4096))
        tail = f.read(4096)
    if not all(b in b"abcdefghijklmnopqrstuvwxyz \n" for b in head):
        return fail("found characters that are not a lower-case letter, space or line end")
    if not tail.endswith(b"\n"):
        return fail("file does not end with a line end")
    print(f"  {path.stat().st_size} bytes, start: {head[:60].decode()!r}")
    return True


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    data = Path(sys.argv[1])
    ok = True
    files = sorted(data.iterdir())
    for p in files:
        if p.suffix == ".bin" and p.name.startswith("matrix_"):
            ok &= check_matrix(p)
        elif p.suffix == ".csr":
            ok &= check_csr(p)
        elif p.suffix == ".txt" and p.name.startswith("corpus_"):
            ok &= check_corpus(p)
    print("\nALL CHECKS PASSED" if ok else "\nCHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
