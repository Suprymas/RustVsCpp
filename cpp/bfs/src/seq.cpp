// Sequential breadth-first search - the T1 baseline. Literal counterpart of
// seq.rs on the Rust side.
//
// The algorithm is the same as in the parallel versions - level-synchronous
// search with an explicit frontier array (CLAUDE.md decision 11). The only
// difference is that a vertex is claimed with a plain comparison rather than
// a CAS. The parallel version pays for atomic operations even on one thread,
// and that cost has to show up in the speedup curve at p = 1.

#include "seq.hpp"

#include <utility>

#include "par.hpp"

namespace bfs {

std::uint32_t seq_bfs(const io::Csr& g, std::uint32_t source, std::uint32_t* dist) {
    dist[source] = 0;

    std::vector<std::uint32_t> frontier{source};
    std::vector<std::uint32_t> next;
    std::uint32_t level = 0;

    while (!frontier.empty()) {
        next.clear();
        for (const std::uint32_t v : frontier) {
            const std::uint32_t* nbr = g.neighbors_begin(v);
            const std::size_t deg = g.degree(v);
            for (std::size_t k = 0; k < deg; ++k) {
                const std::uint32_t w = nbr[k];
                if (dist[w] == UNVISITED) {
                    dist[w] = level + 1;
                    next.push_back(w);
                }
            }
        }
        std::swap(frontier, next);
        ++level;
    }

    return level;
}

}  // namespace bfs
