//! High-level parallelisation: Rayon. C++ counterpart - OpenMP.
//!
//! The decomposition is the same as in the low-level version: bands of C rows,
//! `n / p` rows each, with B = 64 tiling inside the band. The only difference
//! is that thread creation, distribution and synchronisation are handled by
//! the library.
//!
//! The band size is given explicitly (`par_chunks_mut`) rather than left to
//! Rayon's discretion. Reason: the amount of work is identical in every band,
//! so work stealing gains nothing here, and a fixed split matches the C++
//! side's `schedule(static)` - otherwise we would be comparing scheduling
//! strategies rather than languages.

use rayon::prelude::*;

use crate::seq;

/// Parallel multiplication C = A * B in the given Rayon thread pool.
///
/// The pool is created in `main.rs` (together with affinity via
/// `start_handler`); here the work is only launched - so thread creation does
/// not end up in the measured time.
pub fn multiply(a: &[f64], b: &[f64], c: &mut [f64], n: usize, lda: usize, threads: usize) {
    let rows_per = n.div_ceil(threads).max(1);

    c.par_chunks_mut(rows_per * lda)
        .zip(a.par_chunks(rows_per * lda))
        .for_each(|(c_band, a_band)| seq::band_kernel(a_band, b, c_band, n, lda));
}
