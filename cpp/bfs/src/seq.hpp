// Sequential breadth-first search - the T1 baseline (CLAUDE.md decision 1).
// Rust counterpart - seq.rs.

#pragma once

#include <cstdint>
#include <vector>

#include "io.hpp"

namespace bfs {

// Breadth-first search from vertex `source`.
//
// `dist` is a PLAIN uint32 array, not atomic: the sequential version must not
// contain a single parallelisation primitive. It must be pre-filled with
// UNVISITED (this is done OUTSIDE the timer).
//
// Returns the number of levels.
std::uint32_t seq_bfs(const io::Csr& g, std::uint32_t source, std::uint32_t* dist);

}  // namespace bfs
