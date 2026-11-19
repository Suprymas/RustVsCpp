// High-level parallelisation: OpenMP. Rust counterpart - rayon_impl.rs.
//
// Same decomposition: bands of C rows, n/p rows each, B = 64 tiling inside the
// band. The only difference is that thread creation, distribution and
// synchronisation are handled by the library.
//
// `schedule(static)` corresponds to the explicitly given `par_chunks_mut` band
// size on the Rust side: the amount of work is identical in every band, so
// work stealing would gain nothing here, and dynamic splitting would mean we
// are comparing scheduling strategies rather than languages.
//
// A difference from the Rust side worth noting in the thesis: Rayon has
// `par_chunks_mut` - an iterator that simply hands out non-overlapping slices.
// The OpenMP loop runs over INTEGERS, so the band start and height have to be
// computed by hand, and that is exactly where the extra lines come from.
//
// The thread pool is created during the first parallel region, and the first
// region runs in the warm-up run (at least 1 by default), so pool creation
// does not end up in the measurements - just like the Rayon pool, which is
// created outside the timer.
//
// NO pinning is done here: OpenMP has OMP_PLACES / OMP_PROC_BIND for that,
// and they are set by the harness. On the Rayon side the same is done inside
// the program (`start_handler`) - a property of the library interface itself,
// not a flaw in the measurement.

#include "par.hpp"

#include <algorithm>

#include "seq.hpp"

namespace matmul {

void omp_multiply(const double* __restrict a,
                 const double* __restrict b,
                 double* __restrict c,
                 std::size_t n,
                 std::size_t lda,
                 std::size_t threads) {
    const std::size_t rows_per = rows_per_band(n, threads);
    const std::size_t bands = (n + rows_per - 1) / rows_per;

#pragma omp parallel for schedule(static) num_threads(static_cast<int>(threads))
    for (std::size_t t = 0; t < bands; ++t) {
        const std::size_t row = t * rows_per;
        band_kernel(a + row * lda, b, c + row * lda, n, lda, std::min(rows_per, n - row));
    }
}

}  // namespace matmul
