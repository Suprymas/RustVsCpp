//! Sequential breadth-first search - the T1 baseline (decision 1).
//!
//! There MUST NOT be a single parallelisation primitive here, so `dist` is a
//! plain `Vec<u32>`, not `Vec<AtomicU32>`. This is a fair baseline: the
//! parallel version pays for atomic operations even on one thread, and that
//! cost has to show up in the speedup curve at p = 1.
//!
//! The algorithm is the same as in the parallel versions - level-synchronous
//! search with an explicit frontier array (decision 11). The only difference
//! is that a vertex is claimed with a plain comparison rather than a CAS.

use crate::io::Csr;
use crate::par::UNVISITED;

/// Breadth-first search from vertex `source`. `dist` must be pre-filled with
/// `UNVISITED` (this is done OUTSIDE the timer, like zeroing C on the matmul
/// side).
///
/// Returns the number of levels (maximum distance + 1).
pub fn bfs(g: &Csr, source: u32, dist: &mut [u32]) -> u32 {
    dist[source as usize] = 0;

    let mut frontier: Vec<u32> = vec![source];
    let mut next: Vec<u32> = Vec::new();
    let mut level: u32 = 0;

    while !frontier.is_empty() {
        next.clear();
        for &v in &frontier {
            for &w in g.neighbors_of(v as usize) {
                // One indexing operation, then work through the reference.
                // Writing `dist[w] == UNVISITED` and `dist[w] = ...` separately
                // makes safe Rust perform TWO bounds checks on the same
                // element, and with random access this shows up in the
                // measurements (see CLAUDE.md).
                let cell = &mut dist[w as usize];
                if *cell == UNVISITED {
                    *cell = level + 1;
                    next.push(w);
                }
            }
        }
        std::mem::swap(&mut frontier, &mut next);
        level += 1;
    }

    level
}
