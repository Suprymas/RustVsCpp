// Declarations of the parallel variants. Implementations - threads.cpp (low
// level) and omp_impl.cpp (high level).

#pragma once

#include <cstddef>
#include <vector>

namespace matmul {

// Low level: std::thread. Rust counterpart - threads.rs.
// Empty `core_ids` = no affinity (control series).
void threads_multiply(const double* __restrict a,
                      const double* __restrict b,
                      double* __restrict c,
                      std::size_t n,
                      std::size_t lda,
                      std::size_t threads,
                      const std::vector<int>& core_ids);

// High level: OpenMP. Rust counterpart - rayon_impl.rs.
// No pinning here - it is handled by OMP_PLACES / OMP_PROC_BIND.
void omp_multiply(const double* __restrict a,
                 const double* __restrict b,
                 double* __restrict c,
                 std::size_t n,
                 std::size_t lda,
                 std::size_t threads);

// Band height: n / p rounded up. Identical in all three places (threads.cpp,
// omp_impl.cpp and on the Rust side `n.div_ceil(threads).max(1)`), so the
// split is guaranteed to be the same.
inline std::size_t rows_per_band(std::size_t n, std::size_t threads) {
    const std::size_t rows = (n + threads - 1) / threads;
    return rows > 0 ? rows : 1;
}

}  // namespace matmul
