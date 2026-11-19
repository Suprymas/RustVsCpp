// Part shared by both parallel variants. Rust counterpart - par.rs.
//
// Only what must be IDENTICAL in the low and high level lives here: the
// unvisited-vertex marker, the frontier split granularity and the kernel that
// processes one chunk. This keeps threads.cpp and omp_impl.cpp pure
// parallelisation code, suitable for the dLOC metric.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "io.hpp"

namespace bfs {

// Marker of an unvisited vertex. Also used by the sequential version.
constexpr std::uint32_t UNVISITED = UINT32_MAX;

// Frontier split granularity (CLAUDE.md decision 11).
//
// R-MAT vertex degrees follow a power law (scale 24: mean 58.7, maximum
// 406,621), so a static split would be badly unbalanced - one thread would
// get a vertex with 400 thousand neighbours. The split is dynamic, 64
// vertices at a time: OpenMP `schedule(dynamic, 64)`, and in the low level -
// a shared atomic chunk counter.
constexpr std::size_t CHUNK = 64;

// Type of the distance array in the parallel variants.
using AtomicDist = std::atomic<std::uint32_t>;

// Processes one frontier chunk: all neighbours of the chunk's vertices are
// visited, new ones are claimed with CAS and written to `out`.
//
// Memory ordering - `relaxed` in both languages. That is enough, because the
// only synchronisation needed is between levels, and the barrier provides it.
//
// In the header and `inline`, so that it is inlined into both variants
// independently of LTO - on the Rust side `#[inline]` gives the same guarantee.
inline void scan_chunk(const io::Csr& g,
                       const std::uint32_t* chunk,
                       std::size_t chunk_len,
                       AtomicDist* dist,
                       std::uint32_t next_level,
                       std::vector<std::uint32_t>& out) {
    for (std::size_t i = 0; i < chunk_len; ++i) {
        const std::size_t v = chunk[i];
        const std::uint32_t* nbr = g.neighbors_begin(v);
        const std::size_t deg = g.degree(v);
        for (std::size_t k = 0; k < deg; ++k) {
            const std::uint32_t w = nbr[k];
            AtomicDist& cell = dist[w];
            // Cheap read first. The mean degree is ~49-59, so most neighbours
            // are already visited, and a CAS on them would be a needless
            // exclusive acquisition of the cache line.
            if (cell.load(std::memory_order_relaxed) != UNVISITED) {
                continue;
            }
            std::uint32_t expected = UNVISITED;
            if (cell.compare_exchange_strong(expected, next_level, std::memory_order_relaxed,
                                             std::memory_order_relaxed)) {
                out.push_back(w);
            }
        }
    }
}

// Low level: std::thread + std::barrier. Rust counterpart - threads.rs.
// Empty `core_ids` = no affinity (control series).
// Returns the number of levels.
std::uint32_t threads_bfs(const io::Csr& g,
                          std::uint32_t source,
                          AtomicDist* dist,
                          std::size_t threads,
                          const std::vector<int>& core_ids);

// High level: OpenMP. Rust counterpart - rayon_impl.rs.
// No pinning here - it is handled by OMP_PLACES / OMP_PROC_BIND.
std::uint32_t omp_bfs(const io::Csr& g,
                      std::uint32_t source,
                      AtomicDist* dist,
                      std::size_t threads);

}  // namespace bfs
