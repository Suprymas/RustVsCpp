// High-level parallelisation: OpenMP. Rust counterpart - rayon_impl.rs.
//
// Same algorithm as the low level: level-synchronous search, frontier split
// dynamically into chunks of CHUNK vertices. The only difference is that the
// thread team, dynamic split and the barrier between levels are handled by
// the library.
//
// The structure is chosen to match the Rust side: a new parallel region is
// started for every level, and every thread has its own LOCAL buffer created
// inside the region. Rayon `fold(Vec::new, ...)` does exactly the same - a new
// buffer per work split. This matters: a shared array of buffers (as in the
// low level) would mean threads write to adjacent vector header fields and
// share the same cache line.
//
// The thread pool is created during the first parallel region, and the first
// region runs in the warm-up run (at least 1 by default), so pool creation
// does not end up in the measurements - just like the Rayon pool, which is
// created outside the timer.
//
// NO pinning is done here: OpenMP has OMP_PLACES / OMP_PROC_BIND for that,
// and they are set by the harness.

#include "par.hpp"

#include <omp.h>

#include <cstddef>
#include <utility>

namespace bfs {

std::uint32_t omp_bfs(const io::Csr& g,
                      std::uint32_t source,
                      AtomicDist* dist,
                      std::size_t threads) {
    dist[source].store(0, std::memory_order_relaxed);

    std::vector<std::uint32_t> frontier{source};
    std::vector<std::uint32_t> next;
    std::uint32_t level = 0;

    while (!frontier.empty()) {
        const std::uint32_t next_level = level + 1;
        const std::ptrdiff_t len = static_cast<std::ptrdiff_t>(frontier.size());
        next.clear();

#pragma omp parallel num_threads(static_cast<int>(threads))
        {
            std::vector<std::uint32_t> local;
#pragma omp for schedule(dynamic, CHUNK) nowait
            for (std::ptrdiff_t i = 0; i < len; ++i) {
                scan_chunk(g, frontier.data() + i, 1, dist, next_level, local);
            }
#pragma omp critical
            next.insert(next.end(), local.begin(), local.end());
        }

        std::swap(frontier, next);
        level = next_level;
    }

    return level;
}

}  // namespace bfs
