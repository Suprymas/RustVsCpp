//! Low-level parallelisation: `std::thread` without any parallelisation
//! library. C++ counterpart - `std::thread`.
//!
//! Work is split into bands of C rows, `n / p` rows each. A band is contiguous
//! in memory, so `chunks_mut` splits it into non-overlapping mutable slices
//! and the compiler itself checks that no two threads write the same
//! element - there is not a single `unsafe` anywhere in this file.
//!
//! A band boundary always coincides with the start of a row (n*8 bytes), which
//! is a multiple of the 64 B cache line, so there is no false sharing.
//!
//! `std::thread::scope` is needed because the threads borrow `a`, `b` and `c`
//! from the calling function; without it everything would have to go via `Arc`.

use crate::seq;

/// Parallel multiplication C = A * B using `threads` threads.
///
/// `core_ids` - cores the threads are pinned to (empty list = no affinity,
/// control series). The list is passed in from outside rather than queried
/// here: `core_affinity::get_core_ids()` makes a system call, and inside the
/// timer that would be a setup cost the Rayon version does not pay.
pub fn multiply(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    n: usize,
    lda: usize,
    threads: usize,
    core_ids: &[core_affinity::CoreId],
) {
    let rows_per = n.div_ceil(threads).max(1);

    std::thread::scope(|scope| {
        let bands = c.chunks_mut(rows_per * lda).zip(a.chunks(rows_per * lda));
        for (t, (c_band, a_band)) in bands.enumerate() {
            scope.spawn(move || {
                if !core_ids.is_empty() {
                    core_affinity::set_for_current(core_ids[t % core_ids.len()]);
                }
                seq::band_kernel(a_band, b, c_band, n, lda);
            });
        }
    });
}
