//! High-level parallelisation: Rayon. C++ counterpart - OpenMP.
//!
//! Same algorithm as the low level: level-synchronous search, frontier split
//! into chunks of `CHUNK` vertices. The only difference is that the thread
//! team, dynamic split, buffers and the barrier between levels are handled by
//! the library.
//!
//! `fold` + `reduce` (not `flat_map`) because `fold` allocates one buffer per
//! work split, whereas `flat_map_iter` would allocate one per 64-vertex chunk
//! - at scale 24 that would be hundreds of thousands of small allocations and
//! we would be measuring the allocator rather than the parallelisation.

use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::io::Csr;
use crate::par::{scan_chunk, CHUNK};

/// Breadth-first search from vertex `source`. Returns the number of levels.
pub fn bfs(g: &Csr, source: u32, dist: &[AtomicU32]) -> u32 {
    dist[source as usize].store(0, Ordering::Relaxed);

    let mut frontier: Vec<u32> = vec![source];
    let mut level: u32 = 0;

    while !frontier.is_empty() {
        let next_level = level + 1;
        frontier = frontier
            .par_chunks(CHUNK)
            .fold(Vec::new, |mut out, chunk| {
                scan_chunk(g, chunk, dist, next_level, &mut out);
                out
            })
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });
        level = next_level;
    }

    level
}
