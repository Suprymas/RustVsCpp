#!/usr/bin/env python3
"""Parquet -> one .txt file per book (input for datagen corpus).

WHY NOT DIRECTLY FROM PROJECT GUTENBERG. The PG rsync mirrors give ~43 KiB/s
on this connection (aleph, pglaf, xmission, ibiblio were tried), while the
connection itself manages ~1.5-2.9 MiB/s. The same texts in the Hugging Face
data set `sedthh/gutenberg_english` are served via a CDN at 2.9 MiB/s, i.e.
~68 times faster. The content is the same - PG English-language books.

Files are named by `text_id` with leading zeros, because `datagen corpus`
sorts by name, and this way the result does not depend on the order in which
the file system returns the files.

Usage:
    python3 parquet_to_txt.py <parquet_directory> <output_directory> [--max-gib N]
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

MIN_CHARS = 2000  # shorter entries (tables of contents, errors) are skipped


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    max_gib = None
    if "--max-gib" in sys.argv:
        max_gib = float(sys.argv[sys.argv.index("--max-gib") + 1])

    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("*.parquet"))
    if not shards:
        print(f"no .parquet files found in directory {src}")
        return 1

    written = 0
    total_bytes = 0
    skipped = 0

    for shard in shards:
        pf = pq.ParquetFile(shard)
        for group in range(pf.num_row_groups):
            table = pf.read_row_group(group, columns=["TEXT", "METADATA"])
            texts = table.column("TEXT").to_pylist()
            metas = table.column("METADATA").to_pylist()
            for text, meta in zip(texts, metas):
                if not text or len(text) < MIN_CHARS:
                    skipped += 1
                    continue
                try:
                    text_id = int(json.loads(meta)["text_id"])
                except Exception:
                    skipped += 1
                    continue
                path = dst / f"pg{text_id:06d}.txt"
                data = text.encode("utf-8", "replace")
                path.write_bytes(data)
                written += 1
                total_bytes += len(data)
        print(f"  {shard.name}: {written} books in total, {total_bytes / 2**30:.2f} GiB", flush=True)
        if max_gib is not None and total_bytes / 2**30 >= max_gib:
            print(f"  reached the {max_gib} GiB limit - stopping")
            break

    print(f"wrote {written} files, {total_bytes / 2**30:.2f} GiB, skipped {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
