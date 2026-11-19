//! Low-level parallelisation: `std::thread` + `Barrier`. C++ counterpart
//! - `std::thread` + `std::barrier`.
//!
//! Unlike matrix multiplication, here the threads are created ONCE for the
//! whole search, not for every level. The reason is purely quantitative: the
//! R-MAT graph diameter is small (~7 levels), but creating 32 threads costs
//! ~0.44 ms, and creating them at every level would turn the measurement into
//! a measurement of thread creation. A persistent thread team with a barrier
//! between levels is also what a programmer actually writes. The C++ side does
//! exactly the same, so the symmetry between languages is preserved.
//!
//! Worth noting in the thesis: this is exactly where it shows what working
//! without a high-level library means. The dynamic split counter, buffer
//! merging and termination logic all have to be written by hand - in
//! `rayon_impl.rs` all of it fits into a single
//! `par_chunks(...).fold(...).reduce(...)` expression.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex, RwLock};

use crate::io::Csr;
use crate::par::{scan_chunk, CHUNK};

/// State shared by all threads.
///
/// The `RwLock` around the frontier is a formal requirement, not a means of
/// synchronisation: the barriers already guarantee that the read and write
/// phases do not overlap. But safe Rust cannot prove that, so the invariant
/// has to be WRITTEN DOWN as a type. The C++ side does not need this - a plain
/// vector suffices there, and correctness rests solely on the author's
/// discipline. The cost here is one atomic operation per thread per level,
/// i.e. practically zero.
struct Shared<'a> {
    dist: &'a [AtomicU32],
    frontier: RwLock<Vec<u32>>,
    /// One buffer per thread. A thread writes here only ONCE per level (see
    /// `worker`), so adjacent headers in the same cache line cause no
    /// problem.
    buffers: Vec<Mutex<Vec<u32>>>,
    /// Dynamic split cursor - start of the next chunk in the frontier array.
    cursor: AtomicUsize,
    barrier: Barrier,
    level: AtomicU32,
    done: AtomicBool,
}

/// Breadth-first search from vertex `source` with `threads` threads.
///
/// `dist` must be pre-filled with `UNVISITED` (outside the timer).
/// Returns the number of levels.
pub fn bfs(
    g: &Csr,
    source: u32,
    dist: &[AtomicU32],
    threads: usize,
    core_ids: &[core_affinity::CoreId],
) -> u32 {
    dist[source as usize].store(0, Ordering::Relaxed);

    let shared = Shared {
        dist,
        frontier: RwLock::new(vec![source]),
        buffers: (0..threads).map(|_| Mutex::new(Vec::new())).collect(),
        cursor: AtomicUsize::new(0),
        barrier: Barrier::new(threads),
        level: AtomicU32::new(0),
        done: AtomicBool::new(false),
    };
    let sh = &shared;

    std::thread::scope(|scope| {
        for t in 0..threads {
            scope.spawn(move || {
                if !core_ids.is_empty() {
                    core_affinity::set_for_current(core_ids[t % core_ids.len()]);
                }
                worker(g, sh, t);
            });
        }
    });

    shared.level.load(Ordering::Relaxed)
}

fn worker(g: &Csr, sh: &Shared, t: usize) {
    // The local buffer lives for the whole search on this thread's stack, so
    // `push` does not touch shared memory. The thread's header in the shared
    // `buffers` array is updated only once per level (via `swap`) - otherwise
    // 32 threads would keep bouncing the same cache line around.
    let mut local: Vec<u32> = Vec::new();

    loop {
        // Phase 1: scan the current frontier. Chunks are taken dynamically,
        // because vertex degrees are very uneven.
        {
            let frontier = sh.frontier.read().unwrap();
            let next_level = sh.level.load(Ordering::Relaxed) + 1;
            loop {
                let start = sh.cursor.fetch_add(CHUNK, Ordering::Relaxed);
                if start >= frontier.len() {
                    break;
                }
                let end = (start + CHUNK).min(frontier.len());
                scan_chunk(g, &frontier[start..end], sh.dist, next_level, &mut local);
            }
        }
        // Results are published with a single swap: `local` receives the
        // vector freed by the merge (empty, but with the same capacity).
        std::mem::swap(&mut *sh.buffers[t].lock().unwrap(), &mut local);

        // Phase 2: merge. TWO barriers are needed, because Rust's
        // `Barrier::wait` releases the leader (`is_leader`) together with all
        // the other threads, and without the second barrier they would start
        // reading a frontier that has not been merged yet. On the C++ side
        // `std::barrier` accepts a completion function that runs BETWEEN
        // arrival and release, so one barrier suffices there - a real
        // difference in the expressiveness of the standard libraries.
        if sh.barrier.wait().is_leader() {
            let mut frontier = sh.frontier.write().unwrap();
            frontier.clear();
            for buf in &sh.buffers {
                frontier.append(&mut buf.lock().unwrap());
            }
            sh.cursor.store(0, Ordering::Relaxed);
            sh.level.fetch_add(1, Ordering::Relaxed);
            sh.done.store(frontier.is_empty(), Ordering::Relaxed);
        }
        sh.barrier.wait();

        if sh.done.load(Ordering::Relaxed) {
            return;
        }
    }
}
