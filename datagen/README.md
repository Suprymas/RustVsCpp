# datagen — test data generator

The program implements the requirement described in section 2.4 of the thesis:
all test data are generated **once** by this separate helper program and
written to files that the Rust and C++ implementations only read. This removes
the risk that the languages' different pseudo-random number generators would
produce different input values from the same seed.

## Building and running

```bash
cd datagen
cargo build --release
cargo test --release        # 13 unit tests, including the PRNG reference sequence

./target/release/datagen matrix --sizes 512,1024,2048 --out data
./target/release/datagen graph  --scales 20,22,24 --edgefactor 16 --out data
./target/release/datagen corpus --input gutenberg_raw --sizes 0.5G,2G,8G --out data
./target/release/datagen checksums --out data

python3 verify.py data      # independent check of the generated files
```

Common options: `--out <directory>` (default `data`), `--seed <num>`
(default `20260101`). Every command appends an entry to `data/MANIFEST.md` —
that is the reproducibility information intended for the appendices
(algorithm, seed, parameters, file sizes, SHA-256 sums, graph statistics, the
list of works used).

## Pseudo-random number generator

**Xoshiro256++** (Blackman, Vigna) is used, embedded directly in the project
rather than taken from an external library — this way reproducibility does not
depend on a library version. The initial state is expanded from a 64-bit seed
with the SplitMix64 algorithm, as recommended by the authors. The test
`rng::tests::reference_sequence` checks the output against the authors'
reference C code.

Every file is generated from a **separate stream**: the stream seed is derived
from the master seed and the file label (`graph/24/edges`, `matrix/1024/A` and
so on). So the content of one file does not depend on the order in which, or
how many, files were generated — a single file can be regenerated and will
have the same values.

## File formats

All numbers are little-endian.

### `matrix_{A,B}_{n}.bin` (magic `RMTX`, v1)

| Offset | Field |
|---|---|
| 0..4 | `RMTX` |
| 4..8 | version `u32` = 1 |
| 8..16 | number of rows `u64` |
| 16..24 | number of columns `u64` |
| 24.. | `rows*cols` × `f64`, by rows (row-major) |

Elements are uniformly distributed in the interval [0, 1).

### `graph_s{scale}.csr` (magic `RCSR`, v1)

| Offset | Field |
|---|---|
| 0..4 | `RCSR` |
| 4..8 | version `u32` = 1 |
| 8..16 | number of vertices `n` (`u64`) |
| 16..24 | number of directed entries `m` (`u64`) |
| 24..32 | number of search sources `s` (`u64`) = 8 |
| 32.. | offsets array `(n+1)` × `u64` |
| … | neighbours array `m` × `u32` |
| … | search sources `s` × `u32` |

The neighbours array is sorted in ascending order within each row. The graph
is symmetric, without self-loops or duplicate edges, and connected. The offsets
array starts at byte 32, so `u64` alignment is preserved and the file can be
memory-mapped (`mmap`) without copying.

The eight search sources are written into the file itself, so that both
implementations use exactly the same vertices.

### `corpus_{size}.txt`

Plain ASCII text: lower-case letters only, words separated by a single space,
a line end every 16 words. A file always ends with a line end and is never cut
in the middle of a word.

## Reading examples

**Rust:**

```rust
let bytes = std::fs::read("data/matrix_A_1024.bin")?;
assert_eq!(&bytes[0..4], b"RMTX");
let rows = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
let cols = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
let mut a = vec![0f64; rows * cols];
for (i, v) in a.iter_mut().enumerate() {
    let o = 24 + i * 8;
    *v = f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
}
```

**C++:**

```cpp
std::ifstream f("data/matrix_A_1024.bin", std::ios::binary);
char magic[4]; uint32_t version; uint64_t rows, cols;
f.read(magic, 4);
f.read(reinterpret_cast<char*>(&version), 4);
f.read(reinterpret_cast<char*>(&rows), 8);
f.read(reinterpret_cast<char*>(&cols), 8);
std::vector<double> a(rows * cols);
f.read(reinterpret_cast<char*>(a.data()), rows * cols * 8);
```

On `x86-64` both languages use little-endian and IEEE-754 `binary64`, so the
byte representation of `f64` / `double` is the same and no extra conversion is
needed.

## Notes to keep in mind when writing the thesis

1. **The actual graph size does not match 2^scale.** R-MAT with edge factor 16
   leaves quite a few isolated vertices; keeping the largest connected
   component leaves about 60 % of the vertices, and the mean degree grows
   accordingly (for scale 20: 645,783 vertices out of 1,048,576, mean degree
   48.6). These actual numbers must be given in the thesis — they are written
   to `MANIFEST.md`.
2. **The wording "mean vertex degree 16" is inaccurate.** In the Graph500 sense
   this is the edge factor (16·n edges are generated), not the mean degree of
   the result. It should be named that way in section 2.4.
3. **Vertex number permutation** is applied by default (as in Graph500), so
   that the generation order does not create artificial memory locality. It
   can be disabled with `--no-permute`, but then the BFS memory access pattern
   becomes unrealistically favourable.
4. **The corpus is not downloaded automatically.** The works from Project
   Gutenberg have to be downloaded separately into one directory (`.txt`,
   UTF-8 or ASCII); `datagen corpus` processes them sorted by file name, so the
   result does not depend on the order returned by the file system. The name,
   size, word count and SHA-256 of every source file go into the manifest —
   that is exactly what the appendix needs as the "list of works used".
5. **Reference results.** By default `datagen corpus` computes the total word
   count and the 20 most frequent words (disabled with `--no-reference`,
   because it takes a while for the 8 GB file). This is an independent
   reference against which both implementations are compared — better than
   comparing them only with each other. Separately:
   `datagen reference --file data/corpus_8G.txt`.
6. **Normalisation rule.** Non-ASCII-letter characters are treated as
   separators, so "don't" becomes two words, "don" and "t". This has to be
   mentioned in the thesis — it is a matter of definition, not an error.
7. **There is no reference for matrix multiplication** — the sequential
   implementation provides it. Since T₁ is defined as the sequential version
   without parallelisation primitives, it has to be written in both languages
   anyway, so generating a separate reference is not worth it.

## Approximate generation times

Measured on a single vCPU thread; your machine will be considerably faster.

| Data | Duration | File size |
|---|---|---|
| `graph_s20.csr` | ~7 s | 125 MiB |
| `graph_s24.csr` | ~2–4 min (extrapolated) | ~2 GiB |
| `matrix_*_2048.bin` | < 1 s | 32 MiB |
| `corpus_512M.txt` | ~2 s | 512 MiB |

The largest memory requirement is the `scale 24` graph: the edge list takes
about 4.3 GB during sorting, the whole process peaks at about 10 GB.
