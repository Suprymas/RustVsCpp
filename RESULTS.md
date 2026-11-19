# What changed since the thesis text was written: measurement results and statements to correct

Justas Macys, PRIfs-23. Prepared 2026-08-27.

This document gathers in one place everything that came to light after the
measurements were done, and is meant to be carried over into
`Mokslinis darbas Rust_p.docx`. It consists of two kinds of material:

- **results** that are not in the thesis text yet (all three tasks, four
  evaluation criteria);
- **statements to correct** that are already written in the thesis text but
  that the measurements refuted or refined.

All numbers come from the CSV files in `harness/results/`; every table says
which one. All time values are medians, measured with the frequency locked at
3 GHz.

---

## 1. Test environment — three statements to correct

### 1.1. Main memory: 31 GiB, not 128 GB

The thesis says "128 GB LPDDR5x-8000". Physically that much is present, but
**31 GiB is available to the operating system**. Checked:

| Source | Value |
|---|---|
| `/proc/meminfo` → `MemTotal` | 32,631,608 kB ≈ 31.1 GiB |
| `/sys/class/drm/card0/device/mem_info_vram_total` | 103,079,215,104 B = 96 GiB |

Strix Halo is a unified memory chip, and the BIOS assigned 96 GiB to the
integrated Radeon 8060S.

This does not affect the results: the largest process (word frequencies, an
8 GiB corpus plus the tables) reaches ~9.4 GiB and fits with room to spare, and
the swap file was not touched during the measurements. But the thesis has to
state the number precisely, because for reproducibility it matters how much
memory there actually was.

### 1.2. L3 cache: 2 × 32 MB, not 64 MB

The manufacturer advertises 64 MB, but that is **the sum of two separate 32 MB
regions**. Cores 0–7 (SMT pairs 16–23) share one, 8–15 (24–31) the other; data
travels between the CCDs over Infinity Fabric. The effective shared L3 is
32 MB.

This matters, because it is exactly what explains the break in the speedup
curve at 8/16 threads: at N = 2048 a single f64 matrix takes 32 MB, so it does
not fit even into one CCD.

The whole hierarchy was checked via `/sys/devices/system/cpu/cpu*/cache/`, not
from a specification table:

| Level | Size | Sets × ways | Belongs to |
|---|---|---|---|
| L1d | 48 KB | 64 × 12 | the core |
| L2 | 1 MB | 1024 × 16 | the core |
| L3 | 32 MB | — | an eight-core CCD |

### 1.3. The CPU frequency was locked during the measurements

The thesis does not mention this. Boost was disabled, and `scaling_min_freq` of
all 32 cores was set equal to `scaling_max_freq` (`harness/preflight.sh lock`).
The actual frequency during the series was **2888–3013 MHz** (CSV column
`freq_MHz`); before that the free range was 2000–5187 MHz.

Justification — three measured facts:

1. the same sequential N = 2048 run took 0.61 s on a cooled-down machine and
   1.05 s on a hot one (**1.7×**, i.e. more than the expected difference
   between the languages);
2. under an all-core load Tjmax (100 °C) is reached within 6 s, and over 20 s
   the frequency drops 4346 → 4181 MHz (3.88 %) and has not settled yet;
3. the cores are not identical: `prefcore` rankings are 206–236 on CCD0 and
   166–201 on CCD1, so pinning thread t → core t at p = 16 would move half the
   threads to slower cores.

**The cost has to be stated honestly:** the absolute GFLOP/s are lower than in
"real-world use". Example: Rust matmul N = 512 sequentially gives
30.2 GFLOP/s with boost and 17.5 GFLOP/s with the frequency locked. But **the
measured ratios do not change** — the C++ advantage remains in both cases.

Independently of the lock, `harness/` always shuffles the run order and runs
both languages one after the other at the same measurement point, so that any
residual thermal drift affects them equally.

---

## 2. Corrections to the methodology

### 2.1. A checksum does not prove bit-for-bit equality — a bit hash does

The methodology section of the thesis claims that agreement of the results
across languages is proven by a checksum. **That is not true, and it only came
to light when comparing the languages.**

At N = 2048 every C element ≈ 512, the total ≈ 2.1·10⁹, and the f64 resolution
at that magnitude ≈ 5·10⁻⁷. Differences in individual elements (~10⁻¹¹) fall
below it. Measured: the C++ program compiled **without** `-ffp-contract=off`
gives **the same** sum for all three sizes, even though `vfmadd` instructions
are visible in the object file and the actual elements differ. The sum checks
~7 significant digits, not bits.

Therefore `io::bit_hash` was added in both languages — an FNV-1a hash of the
whole matrix over `to_bits`, with an extra `h ^= h >> 33` (without it,
differences in the top bits would stay only in the top bits). It is what is
used for the check between repetitions, in the CSV column `bit_hash` and in the
end-of-series check.

A good methodological example for the thesis: **the checking tool was too
coarse for what it was supposed to check.**

### 2.2. The matrix row stride is padded: `lda = n + 8`

This is the most important change to the methodology, and without it the
N = 2048 results were not valid.

**The problem.** At N = 2048 the row stride is exactly 16 KiB = 2¹⁴. The L1d set
is determined by address bits [11:6], i.e. `address mod 4096`;
16384 mod 4096 = **0**, so all 64 rows of a tile used to fall into *the same*
L1 set, which has only 12 ways. Tiling — the whole point of B = 64 — did not
work. Whether a given process suffered was decided by the placement of its
physical pages, which is different every time.

**The symptom it produced:** the same `--mode seq --n 2048` in ten separate
processes:

| N | Rust spread between processes | C++ spread |
|---|---|---|
| 512 | 0.7 % | 2 % |
| 1024 | 6 % | 1 % |
| **2048** | **30 %** (1.45–1.89 s) | **24 %** (0.87–1.08 s) |

Within each process 10 repetitions gave a spread of ~0 %. So the within-process
spread figures at N = 2048 measured nothing.

**Proof that the cause is memory, not instructions** (`perf stat`, the same
binary, a fast and a slow run):

| | fast | slow |
|---|---|---|
| instructions | 29.3 G | 29.3 G (identical) |
| cycles | 12.0 G | 19.5–21.6 G |
| IPC | 2.45 | 1.36–1.50 |
| `cache-misses` | **78 M** | **656–895 M** |

**Why exactly +8.** 8 f64 = 64 B = one cache line; 2056·8 = 16448, and
16448 mod 4096 = 64, so each successive matrix row shifts by one set and the
64 rows spread over 64 different sets. Microbenchmark
(`harness/lda_microbench.c`, n = 2048, median of 5):

| padding | +0 B | +8 B | +16 B | +32 B | **+64 B (n+8)** | +128 B | +512 B |
|---|---|---|---|---|---|---|---|
| time | 0.888 s | 0.970 | 0.902 | 0.640 | **0.292** | 0.287 | 0.280 |

Padding smaller than 64 B does not help at all (+8 and +16 B even hurt),
because it does not move the row into another L1 set. Larger padding adds
nothing and only wastes memory. The instruction count does not change by a
single instruction.

**Result.** Spread between processes at N = 2048: Rust 30 % → **0.7 %**,
C++ 24 % → **0.5 %**. The summation order does not change, so bit-for-bit
equality was preserved — all hashes match those of the unpadded version
(`234fad36d5b3e957` / `4b652903272e71d4` / `a41a68a804002b40`).

Padding is done while reading (`io::read_rmtx(path, pad)`), i.e. outside the
timer; the file format does not change; `checksum` and `bit_hash` skip the
padding columns.

### 2.3. A separate T₁ for each language

The speedup Sp = T₁/Tp is computed against the sequential version **in the
same language**. With one common baseline Sp would already contain the speed
difference between the languages, and it would be impossible to tell whether a
language loses because of parallelisation or because of its sequential code.

**Because of this the thesis must not rely on speedup curves alone.** An
extreme example from matrix multiplication, N = 2048, p = 32: Rust Sp = 16.9 is
**higher** than C++ Sp = 15.3, yet the absolute time is 62 % worse (58.2 vs
35.9 ms). That is because Rust's T₁ is worse. Absolute times must be shown
alongside.

### 2.4. Spread is judged by IQR, not by range

The range (max−min)/median is not suitable as the main indicator: a single
measurement raises it, and it also grows with the number of repetitions, so
points with different repetition counts cannot be compared. The difference
between the two measures is informative:

| Pattern | Meaning |
|---|---|
| large range + small IQR | an isolated background disturbance, the median copes |
| large IQR | a genuinely unstable point, worth investigating |

---

## 3. Data

Generated once (`datagen/`, seed 20260101), SHA-256 sums in
`data/MANIFEST.md`; both implementations only read them.

### 3.1. Matrices

Six files (2 / 8 / 32 MiB × A and B), sizes 512 / 1024 / 2048, U[0,1),
format `RMTX v1`. After regeneration the sums match — generation is
deterministic.

### 3.2. Graph — the numbers written in the thesis are inaccurate

Section 2.4 of the thesis says "2²⁰, 2²², 2²⁴ vertices". **After extracting the
largest connected component 53–62 % remain.** The actual numbers:

| scale | vertices (LCC) | share | directed entries m | mean degree | max degree | file |
|---|---|---|---|---|---|---|
| 20 | 645,783 | 61.6 % | 31,402,998 | 48.63 | 64,574 | 125 MiB |
| 22 | 2,394,832 | 57.1 % | 128,309,562 | 53.58 | 162,815 | 508 MiB |
| 24 | 8,865,091 | 52.8 % | 520,748,778 | 58.74 | 406,621 | 2.01 GiB |

R-MAT parameters a = 0.57; b = c = 0.19; d = 0.05, symmetrised, without
self-loops or duplicates.

Also: **"mean vertex degree 16" is inaccurate.** In the Graph500 sense 16 is
the *edge factor* (16·n edges are generated), not the mean degree of the
result; the actual one is ~48–59.

### 3.3. Corpus

| file | size | words | distinct words |
|---|---|---|---|
| `corpus_512M.txt` | 512 MiB | 101,855,682 | 359,478 |
| `corpus_2G.txt` | 2.00 GiB | 405,750,506 | 729,042 |
| `corpus_8G.txt` | 8.00 GiB | 1,614,459,477 | 1,596,670 |

Source: 19,396 Project Gutenberg English-language books, 6.69 GiB of raw text.
The 8 GiB file is the corpus repeated 2 times.

**Obtaining the source — a methodological note.** The texts were taken not
directly from the PG rsync mirrors but from `sedthh/gutenberg_english`
(Hugging Face), because the measured speeds were: PG rsync (aleph) 43 KiB/s,
four parallel connections 81 KiB/s, gutenberg.org over HTTP 147 KiB/s, and
**the Hugging Face CDN 2.9 MiB/s**. The connection's own limit is 1.56 MiB/s,
i.e. the PG mirrors delivered ~3 % of what the line can do. The content is the
same; for reproducibility the source must be named precisely.

The normalisation rule is also worth mentioning: **"don't" becomes two words.**

---

## 4. First criterion — execution time

Three full series, 156 / 156 / 102 measurement points, frequency locked.

| Task | File |
|---|---|
| matrix multiplication | `harness/results/matmul-20260823-175750.csv` |
| BFS | `harness/results/bfs-20260822-192110.csv` |
| word frequencies | `harness/results/wordfreq-20260824-102653.csv` |

In all three the bit hashes match across both languages, all variants and all
thread counts.

### 4.1. Matrix multiplication

**T₁ (sequential version, affinity `on`):**

| N | Rust | C++ | C++ advantage |
|---|---|---|---|
| 512 | 14.83 ms | 8.00 ms | **1.85×** |
| 1024 | 119.20 ms | 64.67 ms | **1.84×** |
| 2048 | 980.60 ms | 548.05 ms | **1.79×** |

The gap is **the same across all sizes** — that is precisely the sign that the
cache conflict has been removed and only the code generation difference is
left. Before the padding it varied between 1.12 and 1.47 purely because of
which regime a given process ended up in.

**p = 32 (affinity `on`), Sp in parentheses:**

| N | Rust low | C++ low | Rayon | OpenMP |
|---|---|---|---|---|
| 512 | 1.397 ms (10.6) | 1.025 ms (7.8) | 1.004 ms (14.8) | 0.641 ms (12.5) |
| 1024 | 8.68 ms (13.7) | 4.82 ms (13.4) | 7.43 ms (16.1) | 4.98 ms (13.0) |
| 2048 | 58.2 ms (16.9) | **35.9 ms** (15.3) | 58.2 ms (16.9) | 39.1 ms (14.0) |

**The best performance achieved — C++ `std::thread` at N = 2048:
478.5 GFLOP/s, i.e. 62 % of the theoretical peak** (16 cores × 3 GHz × 8 f64 ×
2 op = 768 GFLOP/s). Rust at the same point gives 295.2 GFLOP/s (38 %).

Sequentially: Rust 17.5 GFLOP/s, C++ 31.3 GFLOP/s.

**Benefit of affinity** (off/on, > 1 = affinity helps):

| Variant | N | p=8 | p=16 | p=32 |
|---|---|---|---|---|
| `std::thread` | 512 | 1.00 | **0.63** | 0.92 |
| `std::thread` | 1024/2048 | ~1.00 | ~1.00 | ~0.98 |
| Rayon | 512 | 1.30 | **1.80** | 1.41 |
| Rayon | 1024/2048 | ~1.00 | ~1.01 | 1.12 |

**Rayon gains a lot from affinity, `std::thread` nothing.** Explanation: in the
`std::thread` version a thread gets a fixed band, so the scheduler has nothing
to move around; Rayon steals work and without affinity migrates between cores.
The effect is largest at the smallest N, where the migration cost is
relatively higher.

### 4.2. Breadth-first search

| scale | Rust T₁ | C++ T₁ | Rust low | C++ low | Rayon | OpenMP |
|---|---|---|---|---|---|---|
| 20 | 83.0 ms | 76.8 ms | 8.61 ms (9.6) | 8.44 ms (9.1) | 10.40 ms (8.0) | 7.76 ms (9.9) |
| 22 | 457 ms | 442 ms | 32.5 ms (14.1) | 33.1 ms (13.4) | 38.2 ms (12.0) | 32.2 ms (13.7) |
| 24 | 2.706 s | 2.446 s | **148.7 ms** (18.2) | 157.7 ms (15.5) | 244.9 ms (11.1) | 157.0 ms (15.6) |

MTEPS at scale 24, p = 32: Rust `std::thread` **3502**, OpenMP 3317,
C++ `std::thread` 3302, Rayon 2126. Sequentially — 192 and 213 MTEPS.

Three findings:

1. **The superlinear Sp has disappeared.** Previously at scale 24 with p = 16
   it was 16.0; with the frequency locked it is 14.8. It was a thermal
   artefact: the sequential version runs for 2.7 s and has time to heat the
   machine, whereas a 0.19 s parallel run does not. A direct, quantified
   justification of the frequency lock.
2. **Rayon's lag at scale 24 is real**, not an artefact: 244.9 vs 157.0 ms
   (**56 %**), whereas at scale 20 the difference is 34 %, and in matrix
   multiplication Rayon does not lag at all.
3. **In BFS the Rust low level wins at scale 24** (148.7 vs 157.7 ms), even
   though it loses by 10 % in sequential code. This is the opposite of matrix
   multiplication.

**A note on interpreting Sp.** Rust `seq` at scale 24 is 5 % slower than its
own `threads` with p = 1, and this is reproducible across separate processes.
`perf stat` shows that the misses are practically the same in all four variants
(1509–1545 M), i.e. the task is bound by memory latency (IPC ~0.5), and the
whole difference comes from code generation, which goes both ways:

| | instructions | cycles | IPC | `cache-misses` | time |
|---|---|---|---|---|---|
| rust `seq` | 13.6 G | 28.4 G | 0.48 | 1542 M | 3.112 s |
| rust `threads` | 15.3 G | 26.5 G | 0.58 | 1545 M | 2.921 s |
| cpp `seq` | 11.7 G | 27.2 G | 0.43 | 1509 M | 2.962 s |
| cpp `threads` | 15.1 G | 30.4 G | 0.50 | 1510 M | 3.444 s |

In sequential code Rust issues 16 % more instructions; in atomic code the
counts are equal, but C++ spends 15 % more cycles. **So all Rust BFS Sp values
at scale 24 are optimistic by ~8 %.**

### 4.3. Word frequencies

The series is exceptionally stable: out of 204 groups **only one** has
IQR > 10 %. The reason is clear — even the shortest run takes 0.25 s, so thread
creation and background disturbances no longer carry any weight.

**T₁ — the direction flips:**

| corpus | Rust | C++ | Rust advantage |
|---|---|---|---|
| 512 MiB | 2.808 s | 4.118 s | **1.47×** |
| 2 GiB | 10.525 s | 17.991 s | **1.71×** |
| 8 GiB | 42.084 s | 68.203 s | **1.62×** |

The cause is the container, not the language: Rust's `HashMap` is hashbrown
(SwissTable, open addressing, SIMD probing), whereas `std::unordered_map` **is
required by the standard** to be node-based, with a separate allocation for
every entry and stable references. This is a property of the standard library.

**`partial` strategy, 8 GiB, affinity `on`, Sp in parentheses:**

| p | Rust low | C++ low | Rayon | OpenMP |
|---|---|---|---|---|
| 1 | 43.50 s (0.97) | 71.27 (0.96) | 41.30 (1.02) | 66.54 (1.03) |
| 2 | 21.61 s (1.95) | 37.74 (1.81) | 21.62 (1.95) | 38.75 (1.76) |
| 4 | 11.32 s (3.72) | 20.73 (3.29) | 11.38 (3.70) | 20.48 (3.33) |
| 8 | 6.088 s (6.91) | 12.70 (5.37) | 6.239 (6.75) | 11.96 (5.70) |
| 16 | 3.460 s (12.16) | 6.934 (9.84) | 3.411 (12.34) | 6.277 (10.87) |
| 32 | **3.405 s** (12.36) | 6.162 (11.07) | **3.169** (13.28) | 5.485 (12.43) |

**512 MiB, `partial`:**

| p | Rust low | C++ low | Rayon | OpenMP |
|---|---|---|---|---|
| 8 | 0.411 s (6.82) | 0.698 (5.90) | 0.406 (6.92) | 0.647 (6.36) |
| 16 | **0.272 s** (10.31) | 0.500 (8.23) | **0.253** (11.12) | 0.392 (10.51) |
| 32 | 0.318 s (8.84) | 0.546 (7.54) | 0.259 (10.83) | 0.393 (10.48) |

Throughput at 8 GiB, p = 32: Rayon **509.4** Mwords/s, Rust `std::thread`
474.1, OpenMP 294.3, C++ `std::thread` 262.0. Sequentially — 38.4 and
23.7 Mwords/s.

Four findings:

1. **Rust's advantage holds across the whole thread range and even grows.**
   The C++/Rust ratio: 1.47–1.71 sequentially, reaching **1.84–2.09** at
   p = 8…16. Node allocations load the allocator, which at 32 threads becomes
   a point of contention even with mimalloc. Since the allocator is the same
   on both sides, this is a second independent argument that the data
   structure is to blame.
2. **Saturation at p = 16, not p = 32.** At 512 MiB `std::thread` even **slows
   down** from 16 to 32 threads (0.272 → 0.318 s). The limit is SMT.
3. **The high level consistently beats the low level at large p, in both
   languages** (512 MiB, p = 32: Rayon 0.259 vs 0.318 s, i.e. 19 %; OpenMP
   0.393 vs 0.546 s, i.e. 28 %). The reason is that the pool is created
   outside the timer.
4. **Affinity gives nothing in this task** — the off/on ratio is 0.91–1.06 at
   every point. The unit of work is a corpus band whose processing takes
   hundreds of milliseconds, so the migration cost is relatively zero.

**`shared` strategy — the key example of "parallelisation without thinking"**
(512 MiB, affinity `on`, Sp in parentheses):

| p | Rust low | C++ low | Rayon | OpenMP |
|---|---|---|---|---|
| 1 | 2.882 s (0.97) | 4.633 (0.89) | 2.902 (0.97) | 4.374 (0.94) |
| 2 | 8.556 s (0.33) | 12.92 (0.32) | 8.606 (0.33) | 12.99 (0.32) |
| 4 | 13.57 s (0.21) | 16.33 (0.25) | 12.50 (0.22) | 15.75 (0.26) |
| 8 | 19.51 s (0.14) | 18.05 (0.23) | 19.46 (0.14) | 18.11 (0.23) |
| 16 | 30.93 s (0.09) | 40.33 (0.10) | 30.85 (0.09) | 40.00 (0.10) |
| 32 | 34.17 s (0.08) | 42.74 (0.10) | 34.67 (0.08) | 42.92 (0.10) |

How many times slower `shared` is than `partial` at the same point:

| p | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Rust low | 6.4× | 18.7× | 47.4× | **113.6×** | 107.6× |
| C++ low | 7.1× | 13.3× | 25.9× | **80.6×** | 78.2× |
| Rayon | 6.5× | 18.0× | 47.9× | **122.2×** | 133.8× |
| OpenMP | 7.0× | 12.9× | 28.0× | **102.1×** | 109.3× |

Four findings:

1. **The speedup decreases monotonically — every added thread makes things
   worse.** At p = 32 Rust is **12.2×** and C++ **10.4× slower than the
   sequential version**. Already at p = 1 Sp < 1 (0.89–0.97), because the lock
   is taken for every word even without contention: 101.9 million
   `lock`/`unlock` pairs.
2. **The only point in the whole task where C++ is faster than Rust** — at
   p = 8 (18.05 vs 19.51 s). When the lock dominates all the time, the
   container's advantage has nowhere to show.
3. **Rust suffers relatively more** (113× vs 80×) precisely because its
   `partial` version is faster. The absolute time is equally bad on both
   sides: **bad synchronisation drags both languages down to the same floor**,
   and no language or library advantage makes up for it.
4. **The level does not matter** — Rayon and `std::thread` differ by < 3 %.
   When everyone contends for one lock, the quality of the scheduler no longer
   matters.

**This is the strongest quantitative argument for the thesis conclusion:** the
same language, the same parallelisation level and the same corpus give either
a **13× speedup** (`partial`) or a **12× slowdown** (`shared`) — a difference of
about 160 times, arising purely from the choice of data structure. The
difference between the languages (1.5–2×) is secondary next to that.

---

## 5. Second criterion — memory consumption

Measured with `harness/run_memory.sh`, **the median of three runs** for every
combination, `--reps 1 --warmup 0`. 258 points, 50 min.

**Methodological note:** the timing protocol is not needed here. VmHWM does not
depend on frequency, thermal state or thread affinity, so the lock is not
required and the affinity axis is skipped. It is not deterministic, though —
repeating the same run the value varies by ~0.5 % (ASLR, thread stack placement,
mimalloc page rounding), so the median of three is taken and the spread is
written to the CSV. Almost everywhere it is < 3 %; the largest is 14 % (Rust
Rayon, matmul N = 512, p = 4), and that is a finding in itself: **the memory
footprint of the Rayon pool is the least reproducible of all measured
quantities.**

All tables show **RSS minus the input read into memory**.

### 5.1. Matrix multiplication — memory does not distinguish the languages

| N | p | Rust low | C++ low | Rayon | OpenMP |
|---|---|---|---|---|---|
| 512 | 32 | 9 MiB | 11 | **15** | 11 |
| 1024 | 32 | 19 MiB | 19 | **23** | 19 |
| 2048 | 32 | 47 MiB | 47 | **51** | 47 |

Sequentially at N = 2048 — 46.6 / 46.8 MiB (theoretical requirement: three
padded matrices, 96.4 MiB, of which 64 MiB is the input itself).
Parallelisation costs almost nothing, because the bands are slices of the same
`C` array. **The only visible extra is the Rayon pool.**

### 5.2. BFS — parallelisation is expensive, but not because of the choice of language

| scale | p | Rust low | C++ low | Rayon | OpenMP | Rayon/OpenMP |
|---|---|---|---|---|---|---|
| 20 | 32 | 124 MiB | 120 | **213** | 135 | 1.58 |
| 22 | 32 | 255 MiB | 259 | **358** | 259 | 1.38 |
| 24 | 8 | 225 MiB | 243 | **319** | 239 | 1.33 |
| 24 | 16 | 323 MiB | 323 | **461** | 305 | 1.51 |
| 24 | 32 | 447 MiB | 465 | **652** | 455 | 1.43 |

Sequentially — 118.3 / 122.8 MiB (scale 24). With 32 threads consumption grows
~3.8×: every thread has its own local frontier buffer.

1. **At the low level the languages do not differ** — at scale 24, p = 16 the
   difference is 323.0 vs 322.9 MiB, i.e. 0.03 %.
2. **The whole difference is Rayon vs OpenMP: 1.2–1.6×.** This matches the
   timing finding where Rayon was 56 % slower at scale 24, and supports the
   suspicion that the `fold`/`reduce` merge is to blame:
   `par_chunks(64).fold(..).reduce(..)` allocates an intermediate `Vec` for
   every chunk.

### 5.3. Word frequencies — memory tells the opposite story to time

Sequential version:

| corpus | Rust | C++ | Rust/C++ |
|---|---|---|---|
| 512 MiB | 87 MiB | 47 MiB | **1.85** |
| 2 GiB | 101 MiB | 83 MiB | 1.21 |
| 8 GiB | 131 MiB | **163 MiB** | **0.80** |

**The ratio flips between 2 GiB and 8 GiB.** Both sides reserve the same
capacity up front (2²⁰ = 1,048,576; `HashMap::with_capacity` and
`unordered_map::reserve`), so the comparison is symmetric. What differs is
where the cost arises:

- **Rust hashbrown** has a *fixed* cost — 2²⁰ slots of ~33 B (`Vec<u8>` 24 B
  + `u64` 8 B + control byte) ≈ 33 MiB, regardless of how many words there
  actually are. At 512 MiB only 34 % of the slots are occupied, and the
  reservation is pure waste.
- **C++ `unordered_map`** has a *proportional* cost — one node per entry
  (pointer 8 + cached hash 8 + `std::string` 32 + `u64` 8 ≈ 64 B after
  mimalloc rounding). At 512 MiB that is cheaper, at 8 GiB more expensive.

**Rust's time advantage with a small vocabulary is paid for with a 1.85× larger
memory footprint, while with a large vocabulary Rust wins on both criteria.**
This is the only place in the whole study where two criteria point in opposite
directions, and it is exactly what justifies a multi-criteria evaluation.

`partial` vs `shared` (512 MiB):

| p | Rust `partial` | Rust `shared` | C++ `partial` | C++ `shared` |
|---|---|---|---|---|
| 1 | 87 MiB | 87 | 49 | 49 |
| 8 | 107 MiB | 89 | 105 | 49 |
| 16 | 127 MiB | 89 | 135 | 49 |
| 32 | **145 MiB** | **91** | **167** | **51** |

**`shared` consumption does not depend on the thread count, `partial` grows
linearly**, because the partial tables **duplicate the vocabulary** — the most
frequent words appear in the tables of all 32 threads. At p = 32 `shared` is
1.6× (Rust) and 3.3× (C++) more economical.

This is an important refinement: in terms of time `shared` is a disaster, and
it was easy to write it off as simply a bad decision. In terms of memory it is
**better**, and that is its only justification.

Rayon vs OpenMP here too (`partial`):

| corpus | p | Rayon | OpenMP | ratio |
|---|---|---|---|---|
| 512 MiB | 1 | 155 MiB | 71 | **2.18** |
| 8 GiB | 8 | 586 MiB | 358 | 1.64 |
| 8 GiB | 32 | 992 MiB | 561 | 1.77 |

### 5.4. Overall conclusion on memory

1. **At the low level the language has no effect on memory** — in all three
   tasks the Rust and C++ `std::thread` versions agree to within a percent.
   Memory is determined by the algorithm and the data structure, not by the
   language.
2. **At the high level there is a difference, and it goes one way** — Rayon
   uses 1.2–2.2× more than OpenMP, in all three tasks, at all sizes and all
   thread counts. This is the price of work stealing, which in terms of time
   was an advantage (matmul, affinity series) and a drawback (BFS scale 24).
3. **The only place where the languages themselves differ is the standard
   library container**, and there the direction depends on the data size.

---

## 6. Third criterion — code size and cyclomatic complexity

Computed by `harness/code_metrics.py`; the result is
`harness/results/code-metrics.csv`. **The earlier hand-counted numbers (matmul
12/15, 4/7; BFS 55/42, 17/22, 13/31) are no longer valid** — the rule changed,
although the direction and the conclusions remained.

**What is counted.** Not the whole project, only the parallelisation code:
`threads`, `rayon_impl`/`omp_impl` and the shared `par`. All variants call the
same compute kernel, so these files are exactly what a programmer would write
in addition when parallelising an already working algorithm. `io` and `main`
are not included — they are the same for all variants and would only dilute
the difference.

**Logical, not physical, lines are counted.** Physical lines would measure
formatting, and to the disadvantage of C++ in particular: the function
signature in `cpp/matmul/src/omp_impl.cpp` is spread over six physical lines
(`__restrict` parameters), while the Rust counterpart fits on one. The
`namespace X {` line is also excluded — in Rust the file itself is a module, so
there is no counterpart.

| task | part | Rust LOC | C++ LOC | C++/Rust | CC Rust | CC C++ |
|---|---|---|---|---|---|---|
| matmul | low | 9 | 12 | 1.33 | 3 | 4 |
| matmul | high | **3** | 7 | 2.33 | 1 | 3 |
| matmul | common | 0 | 5 | — | 0 | 2 |
| BFS | low | **49** | **38** | **0.78** | 7 | 9 |
| BFS | high | 14 | 19 | 1.36 | 2 | 4 |
| BFS | common | 11 | 18 | 1.64 | 5 | 5 |
| wordfreq | low | 24 | 37 | 1.54 | 6 | 11 |
| wordfreq | high | 14 | 23 | 1.64 | 1 | 6 |
| wordfreq | common | 46 | 53 | 1.15 | 23 | 22 |
| **total** | | **170** | **212** | **1.25** | **48** | **66** |

On the C++ side another **15 lines of `cpp/common/affinity.cpp`** are added —
Rust gets the same from the `core_affinity` library. With them the ratio is
**227 / 170 = 1.34**.

Five points:

1. **The overall ratio of 1.34 is lower than expected**, and that is a result
   in itself: the claim "safety costs a lot more code" is not confirmed by
   these measurements. It costs about a third more, and that cost falls on the
   C++ side.
2. **BFS is the only place where Rust is longer (49 vs 38).** There are two
   reasons, both concrete:
   - `std::barrier` accepts a **completion function**, run after all threads
     have arrived and before they are released, so merging the buffers fits
     into the same barrier. Rust's `std::sync::Barrier` does not have this:
     `wait()` only returns `is_leader()`, and the leader works after all
     threads have already been released, so **a second barrier is needed**.
     Here the C++ standard library is more expressive than Rust's.
   - C++ threads reach the shared `frontier` state directly through the
     lambda's reference; safe Rust needs `RwLock<Vec<u32>>` and
     `Vec<Mutex<Vec<u32>>>`, even though the barrier already guarantees that
     the phases do not overlap. The lock there is not a means of
     synchronisation but **the invariant written down as a type**. Run-time
     cost — one atomic operation per thread per level, i.e. practically zero.
3. **At the high level Rust wins in all three tasks** (3/7, 14/19, 14/23). The
   extreme case is `rust/wordfreq/src/rayon_impl.rs`, whose **cyclomatic
   complexity is 1**: it contains not a single branch, because
   `fold(..).reduce(..)` is one expression. The C++ counterpart has CC 6.
4. **CC is higher on the C++ side in 7 comparisons out of 9** (48 vs 66), and
   the reason is the same everywhere: the OpenMP loop runs over **integers**,
   so the band start, height and bounds have to be computed by hand, and every
   such computation is an extra `for` or `if`. Rust iterators express the same
   without branches.
5. **The only place where both sides are almost equal is `wordfreq/par`**
   (46/53, CC 23/22). And that is right: it holds the word-splitting scanner,
   i.e. **algorithm** code, not parallelisation code. Where the code really is
   the same, the numbers match — a good internal check that the metric
   measures what it should.

**Methodological note on CC.** Rust `||` is not only a logical "or" but also an
argument-less closure (`move || { … }`), and in this code there is one on every
`spawn` line. Counting them would roughly double Rust's CC purely because of
syntax. So closures are recognised and not counted — but **only on the Rust
side**: in C++ `||` is always a logical "or" (a lambda is written `[](…)`
there).

### A qualitative difference that LOC does not measure

Rust `chunks_mut` returns non-overlapping slices and **the compiler itself
checks** that no two threads write the same element; in C++ the same split has
to be expressed with pointer arithmetic (`c + row * n`), and a mistake here
would be a data race without any resistance from the compiler. Likewise
`join()` plays the role of the `scope` barrier, but it is guaranteed by the
author's discipline, not by a type.

`__restrict` on the C++ side is the manual counterpart of what Rust's `noalias`
gets from the borrow checker. Without it the comparison would be unfair, but
the difference is fundamental: in Rust it is **proven**, in C++ it is **an
unchecked promise**.

---
## 7. Fourth criterion — programming obstacles encountered

This indicator is recorded by hand. **The compiler diagnostics histogram does
not replace it:** all 14 C++ SARIF logs and the Rust log are empty — the final
state of the code compiles without a single diagnostic in both languages with
`-Wall -Wextra`. The logs capture the **state** of the code, not the obstacles
encountered while writing it, which were fixed and survive nowhere. This has to
be named in the thesis as a limitation of the methodology.

Obstacles recorded:

**7.1. Rust: a two-dimensional tile decomposition required `unsafe`, so it was
redesigned.** A 64×64 tile in a flat array is 64 separate fragments; it would
have to be shared either via a raw pointer with `unsafe`, or collected into a
`Vec<&mut [f64]>` (which would slow down specifically the Rust side and distort
the comparison). The decomposition was redesigned into **bands of C rows**, and
as a result there is **not a single `unsafe`** anywhere in the parallel code. A
band boundary always coincides with the start of a row, so two threads never
share the same cache line. An obstacle that ended in a better design decision.

**7.2. C++: the obstacles were not in the language's type system but in the
tools and the environment.** Both were diagnosed only with separate probe
programs:

- **`OMP_PLACES=cores`, not `threads`.** libgomp orders `threads` places by SMT
  pairs (place0 = cpu0, place1 = cpu16), so `close` puts two threads on each
  physical core — exactly 2× slower at p = 2, 8, 16.
- **`OMP_PROC_BIND` is set ONLY for the `omp` variant.** Any value of this
  variable makes libgomp pin the **initial thread** as well, which narrows the
  affinity mask of the whole process down to one CPU; the `std::thread`
  children inherit it and `cpp threads p=8` starts running at sequential speed
  (measured: 1.3 → 8.0 ms).

This is a **qualitatively different kind of obstacle** from the Rust side, and
it is worth presenting it that way in the thesis rather than lumping both into
one number.

**7.3. False sharing in buffer headers — the same for both languages.** At
first the BFS threads in both languages did `push` straight into the shared
array's `buffers[t]`. A `Vec`/`std::vector` header is 24 bytes, so two or three
fit into one cache line, and every `push` updates the `len` field in the shared
line. Fixed with a local buffer on the thread's stack and a single `swap` per
level:

| | p = 8 | p = 16 | p = 32 |
|---|---|---|---|
| Rust `std::thread` before | 16.4 ms | 10.4 ms | 8.5 ms |
| Rust `std::thread` after | 15.0 ms | **9.0 ms** | 7.8 ms |
| C++ `std::thread` before | 18.4 ms | 11.7 ms | 8.5 ms |
| C++ `std::thread` after | 15.0 ms | **10.0 ms** | 7.9 ms |

**The type system protects neither language from false sharing.**

**7.4. `std::thread` with affinity is unstable at p ≳ 14 — fragility, not a
flaw.** A slice by thread count (matmul N = 512, 30 repetitions, IQR):

| p | 8–12 | 14 | 15 | 16 | 17 | 18–32 |
|---|---|---|---|---|---|---|
| `std::thread` with affinity | 3–5 % | 60 % | 38 % | 4–11 % | 48 % | 11–19 % |
| `std::thread` without affinity | 3–5 % | 5 % | 4 % | 4 % | 10 % | 5–9 % |
| Rayon with affinity | 9 % | 2 % | 1 % | 1 % | 22 % | 3 % |

The cause is **the combination of static splitting and pinning**. Every thread
has a fixed band and is pinned to a core; if even one thread is delayed it
becomes a straggler, the `scope` barrier makes everyone wait, and there is no
way to recover. Rayon is pinned the same way, but work stealing lets another
thread take over a chunk that has not been started. The threshold at p ≈ 14
appears because once 14 of the 16 physical cores are occupied, system activity
has nowhere left to go.

Rejected hypotheses (all checked by measurement): thread creation cost; CCD
crossing; the main thread on an SMT sibling; the order of `get_core_ids()`; C3
wake-up latency.

**For the thesis this is a good result, not a flaw:** it shows that pinning
without dynamic balancing is fragile, and that the work stealing of a
high-level library provides a resilience that hand-written low-level code does
not have.

---

## 8. The most important methodological finding: separating the language from the compiler

The 1.79× gap in matrix multiplication looked like a difference between the
languages. **It is not a difference between the languages**, and this can be
shown directly: by compiling **the same C++ side with `clang++`** (the same
LLVM back end as rustc).

N = 2048, `seq`, the same flags:

| | instructions | cycles | IPC | time |
|---|---|---|---|---|
| rustc (LLVM) | 29.9 G | 11.8 G | **2.53** | 0.990 s |
| clang++ (LLVM) | 24.4 G | 9.6 G | **2.53** | 0.801 s |
| g++ (GCC) | 25.6 G | 6.6 G | **3.87** | 0.551 s |

**Both LLVM back ends give exactly the same IPC of 2.53, while GCC gives
3.87.** So the IPC difference is purely a property of the back end and has
nothing to do with Rust. `clang++` issues even **fewer** instructions than
`g++` (24.4 vs 25.6 G) and is still 46 % slower — solely because of IPC.

So 1.79× decomposes into two independent factors:

| component | comparison | factor | cause |
|---|---|---|---|
| **language** | rustc vs clang++ (same LLVM) | **1.23×** | +22 % instructions: bounds checks, iterators |
| **compiler back end** | clang++ vs g++ (same C++) | **1.45×** | IPC 2.53 vs 3.87 |
| product | rustc vs g++ | 1.79× | matches the measured value |

Reproduced in four runs in a row (1.22–1.24 × 1.45–1.46 = 1.78–1.80).

The fourth cell was filled in as well — `rustc_codegen_gcc` (Rust with the GCC
back end). N = 2048, `seq`, no LTO, `-ffp-contract=off` everywhere:

| | LLVM | GCC |
|---|---|---|
| **Rust** | 0.988 s — 29.3 G ins, IPC 2.49 | 0.862 s — **45.1 G** ins, IPC **4.36** |
| **C++** | 0.801 s — 24.4 G ins, IPC 2.53 | 0.549 s — 25.6 G ins, IPC 3.86 |

Three conclusions:

1. **IPC is a property of the back end — confirmed a second time.** LLVM gives
   2.49 and 2.53, GCC 4.36 and 3.86, regardless of the language.
2. **But there is an interaction, and it does not favour the language.**
   `rustc_codegen_gcc` issues 45.1 G instructions vs 29.3 G for rustc+LLVM
   (+54 %), while on the C++ side both back ends are similar. Such an asymmetry
   points not to a cost of the language but to `rustc_codegen_gcc` not yet
   knowing how to optimise Rust's intermediate representation.
3. **A great IPC lesson:** Rust+GCC executes 54 % more instructions and is
   still **faster** than Rust+LLVM (0.862 vs 0.988 s), because its IPC is 4.36
   vs 2.49. Judging by instruction count alone would be a mistake.

**What this means for the thesis conclusions.** A language comparison in which
one side is compiled by LLVM and the other by GCC measures both things at once.
So in the thesis:

- **the main language comparison is the LLVM row** (rustc vs clang++, both
  sides mature): **1.23×**;
- **the practical comparison is the best toolchain of each language**
  (rustc+LLVM vs g++), i.e. what a programmer would actually get: **1.79×**;
- the difference between these two numbers is explained by the back-end
  factor of 1.45×.

Without the `clang++` control measurement the thesis would have attributed to
the language what belongs to the compiler.

A note on `rustc_codegen_gcc`: it **violates Rust's floating-point
semantics** — by default it fuses `a*b+c` into FMA, although Rust never does
that, and the result differs bit for bit. Fixed with
`-Cllvm-args=-ffp-contract=off`. **It was the bit hash that caught this** — a
good example of the methodology paying off: the check found a real compiler
flaw that a timing measurement would not have revealed (the FMA version would
simply have looked faster).

---

## 9. Limitations that must be named in the thesis

1. **The locked frequency lowers the absolute numbers.** The ratios do not
   change, but the GFLOP/s are lower than what is achievable with boost.
2. **Affinity asymmetry.** Rayon is pinned inside the program
   (`start_handler`), OpenMP from outside via environment variables. This is a
   property of the library interfaces themselves, not a flaw in the
   measurement, but it has to be named.
3. **Thread creation stays inside the measured time at the low level.** The
   `std::thread` versions create the threads anew for every measurement
   (~14 µs per thread, ~0.44 ms for 32 threads), whereas Rayon and OpenMP
   create their pool once outside the timer. The language comparison does not
   suffer (both sides do the same), but **between levels** there is an
   asymmetry, and at N = 512 it accounts for a large part of the difference.
   At N = 2048 the same absolute cost is only ~1 %. **The thesis must state
   this with numbers**, otherwise the N = 512 results will look like a measure
   of parallelisation rather than of thread creation.
4. **Only the two extreme strategies were studied for word frequencies.** The
   right practical solution is most likely a third one — *sharding* (the
   vocabulary split by hash into independent parts with separate locks), which
   should have the speed of `partial` and the memory footprint of `shared`. It
   was not measured.
5. **`shared` was measured at 512 MiB only.** At p = 32 one run takes 42.9 s;
   at 8 GiB that would be ~11 min, and about a day for the whole matrix. The
   shape of the curve is invariant to size (contention depends on the thread
   count, not on the corpus length), so the extra information would be zero.
6. **The compiler diagnostics histogram is empty** (section 7) — the obstacles
   criterion relies on a hand-kept log.
7. **One compiler per language in the main series.** The `clang++` and
   `rustc_codegen_gcc` measurements were done only for the sequential version
   at N = 2048, not for the whole matrix.

---

## 10. Concrete corrections to the thesis text

Quoted phrases are given in English; the original Lithuanian wording in the
thesis is in parentheses where it helps to locate it.

| Place | What is written | Replace with |
|---|---|---|
| Test environment | "128 GB" | 128 GB installed, **31 GiB available to the OS** (96 GiB iGPU) — sec. 1.1 |
| Test environment | "64 MB L3" | **2 × 32 MB**, one per CCD — sec. 1.2 |
| Test environment | (absent) | **Frequency locking**, 2888–3013 MHz — sec. 1.3 |
| Methodology | agreement of the results is proven by a checksum | **the bit hash**; the sum checks ~7 digits — sec. 2.1 |
| Methodology | (absent) | **`lda = n + 8`** and why — sec. 2.2 |
| Section 2.1 | `(?????)`, `(???)` | **ΔLOC and CC table** — sec. 6 |
| Section 2.1 | error-code histogram | **empty; obstacles recorded by hand** — sec. 7 |
| Section 2.4 | "2²⁰, 2²², 2²⁴ vertices" | **645,783 / 2,394,832 / 8,865,091** — sec. 3.2 |
| Section 2.4 | "mean degree 16" („vidutinis laipsnis 16“) | 16 is the **edge factor**; the actual degree is ~48–59 |
| Third task | (the whole chapter is missing) | sec. 4.3, 5.3 |
| Results | (no memory results) | sec. 5 |
| Conclusions | (absent) | **separating the language from the compiler** — sec. 8 |
| Data | (no source given) | Hugging Face, not PG rsync; normalisation rule — sec. 3.3 |

---

## 11. Files this document relies on

| File | Content |
|---|---|
| `harness/results/matmul-20260823-175750.csv` | matmul series (156 points) |
| `harness/results/bfs-20260822-192110.csv` | BFS series (156 points) |
| `harness/results/wordfreq-20260824-102653.csv` | word frequency series (102 points) |
| `harness/results/*-memory-20260827-*.csv` | memory series (258 points) |
| `harness/results/code-metrics.csv` | ΔLOC and CC |
| `harness/results/*-plots/` | figures (PNG + PDF) and `summary.csv` |
| `data/MANIFEST.md` | SHA-256 sums and statistics of the data |
| `harness/logs/build-*.json` | build diagnostics (empty) |

Plots were generated for all three tasks; for word frequencies they are split
into `partial/` and `shared/` subdirectories, because the strategy is a
separate dimension, not a parallelisation level.
