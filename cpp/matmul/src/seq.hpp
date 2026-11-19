// Tiled matrix multiplication kernel (C++). Rust counterpart - seq.rs.
//
// This file is the T1 baseline AND the common kernel of all three variants:
// both threads.cpp and omp_impl.cpp call the same band_kernel, so the
// algorithm is guaranteed identical, and the parallelisation files contain
// nothing but parallelisation code (important for the dLOC metric).

#pragma once

#include <cstddef>

namespace matmul {

// Tile size. Must match seq::B on the Rust side.
//
// Three 64x64 f64 tiles take 3*64*64*8 = 96 KiB, i.e. ~9 % of the private
// 1 MiB L2 cache; two SMT threads share the same L2, so 2*96 KiB is still
// comfortable. 64 divides 512, 1024 and 2048 without remainder.
constexpr std::size_t B = 64;

// Row padding in elements. Must match seq::PAD on the Rust side.
//
// 8 double = 64 B = one cache line. Without padding, at n = 2048 the row
// stride would be exactly 16384 B, and the L1 set is determined by address
// bits [11:6] (64 sets of 12 ways), so 16384 mod 4096 = 0 - all 64 rows of a
// tile would fall into THE SAME L1 set and tiling would stop working. With
// PAD = 8 each successive row shifts by one set. This is the SMALLEST padding
// that works: +8 and +16 B do not help at all (tested with a microbenchmark).
//
// Padding does NOT change the summation order, so bit-for-bit equality holds.
constexpr std::size_t PAD = 8;

// Computes one band of C rows: C_band += A_band * B.
//
// `rows` - how many rows the band has. On the Rust side this number is derived
// from the slice length (`c.len() / lda`), but a raw C++ pointer has no
// length, so it is passed separately. `lda` - the actual row stride (n + PAD).
//
// The pointers are marked `__restrict` not as an optimisation trick, but so
// that both sides get the same information: on the Rust side `&[f64]` and
// `&mut [f64]` mean guaranteed non-overlap (LLVM `noalias`), and the borrow
// checker PROVES that guarantee. C++ does not have that information - without
// `__restrict` the compiler must assume that `c` and `b` may overlap, and it
// vectorises more cautiously. Without this qualifier we would be comparing
// the availability of aliasing information rather than languages. The
// difference is that on the C++ side it is an UNCHECKED promise by the
// programmer - exactly the kind of case the study evaluates.
void band_kernel(const double* __restrict a,
                 const double* __restrict b,
                 double* __restrict c,
                 std::size_t n,
                 std::size_t lda,
                 std::size_t rows);

// Sequential multiplication C = A * B without any parallelisation primitives (T1 baseline).
void multiply(const double* __restrict a,
              const double* __restrict b,
              double* __restrict c,
              std::size_t n,
              std::size_t lda);

}  // namespace matmul
