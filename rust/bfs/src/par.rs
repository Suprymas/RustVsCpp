//! Part shared by both parallel variants. C++ counterpart - par.hpp.
//!
//! Only what must be IDENTICAL in the low and high level lives here: the type
//! of the distance array, the frontier split granularity and the kernel that
//! processes one chunk. This keeps `threads.rs` and `rayon_impl.rs` pure
//! parallelisation code, suitable for the DeltaLOC metric (like
//! `seq::band_kernel` on the matmul side).

use std::sync::atomic::{AtomicU32, Ordering};

use crate::io::Csr;

/// Marker of an unvisited vertex. Also used by the sequential version.
pub const UNVISITED: u32 = u32::MAX;

/// Frontier split granularity (decision 11).
///
/// R-MAT vertex degrees follow a power law (scale 24: mean 58.7, maximum
/// 406,621), so a static split would be badly unbalanced - one thread would
/// get a vertex with 400 thousand neighbours. The split is dynamic, 64
/// vertices at a time: OpenMP `schedule(dynamic, 64)`, Rayon
/// `par_chunks(64)`, and in the low level - a shared atomic chunk counter.
pub const CHUNK: usize = 64;

/// Processes one frontier chunk: all neighbours of the chunk's vertices are
/// visited, new ones are claimed with CAS and written to `out`.
///
/// Memory ordering - `Relaxed` in both languages. That is enough, because the
/// only synchronisation needed is between levels, and the barrier provides it
/// (low level - `Barrier`, high level - the end of the Rayon `collect`). The
/// CAS here protects only the distance cell itself: the winning thread claims
/// it, the losers do not add the vertex to their buffer, so every vertex
/// enters the next level's frontier exactly once.
#[inline]
pub fn scan_chunk(g: &Csr, chunk: &[u32], dist: &[AtomicU32], next_level: u32, out: &mut Vec<u32>) {
    for &v in chunk {
        for &w in g.neighbors_of(v as usize) {
            let cell = &dist[w as usize];
            // Cheap read first. The mean degree is ~49-59, so most neighbours
            // are already visited, and a CAS on them would be a needless
            // exclusive acquisition of the cache line.
            if cell.load(Ordering::Relaxed) != UNVISITED {
                continue;
            }
            if cell
                .compare_exchange(UNVISITED, next_level, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                out.push(w);
            }
        }
    }
}
