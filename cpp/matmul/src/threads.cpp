// Low-level parallelisation: std::thread without any parallelisation
// library. Rust counterpart - threads.rs.
//
// Work is split into bands of C rows, n/p rows each. On the Rust side the
// bands are produced by `chunks_mut`, and the compiler ITSELF checks that no
// two threads write the same element. On the C++ side the same split has to
// be expressed with pointer arithmetic (`c + row * n`), and non-overlap is
// guaranteed only by the bands being computed correctly - there is no check,
// and a mistake here would be a data race without any resistance from the
// compiler.
//
// `join()` plays the same barrier role as the end of `scope` on the Rust side:
// the threads borrow `a`, `b` and `c` from the calling function, so they must
// finish before those do. On the Rust side this condition is guaranteed by
// the `scope` type, here - by the author's discipline.
//
// Threads are created INSIDE the measured time (as on the Rust side) - a
// deliberate decision that preserves the symmetry between the languages at
// each level.

#include "par.hpp"

#include <algorithm>
#include <thread>
#include <vector>

#include "../../common/affinity.hpp"
#include "seq.hpp"

namespace matmul {

void threads_multiply(const double* __restrict a,
                      const double* __restrict b,
                      double* __restrict c,
                      std::size_t n,
                      std::size_t lda,
                      std::size_t threads,
                      const std::vector<int>& core_ids) {
    const std::size_t rows_per = rows_per_band(n, threads);

    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (std::size_t row = 0, t = 0; row < n; row += rows_per, ++t) {
        const std::size_t rows = std::min(rows_per, n - row);
        pool.emplace_back([=, &core_ids] {
            if (!core_ids.empty()) {
                common::affinity::pin_current(core_ids[t % core_ids.size()]);
            }
            band_kernel(a + row * lda, b, c + row * lda, n, lda, rows);
        });
    }
    for (std::thread& worker : pool) {
        worker.join();
    }
}

}  // namespace matmul
