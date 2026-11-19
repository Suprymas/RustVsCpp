//! Sequential tiled matrix multiplication - the T1 baseline of the study.
//!
//! This file contains not a single parallelisation primitive: no threads, no
//! atomics, no locks. This matches the decision that T1 is measured with the
//! sequential version, not with the parallel one on a single thread.
//!
//! Tiled algorithm with tile size B = 64. Three 64x64 f64 tiles take 96 KiB
//! and fit into the core's 1 MiB L2 cache even when two SMT threads share
//! the same core (2 x 96 KiB = 192 KiB). Therefore every element fetched from
//! memory is used B times before eviction, and memory traffic drops from
//! O(n^3) to O(n^3 / B).
//!
//! Loop order i-k-j: the innermost loop walks C and B rows sequentially, so
//! the compiler vectorises it (AVX-512).
//!
//! THE ROW STRIDE IS `lda`, NOT `n` (decision 12). Matrices are stored with a
//! stride of `lda = n + PAD`, i.e. every row ends with PAD unused elements.
//! Without it, at n = 2048 the stride would be exactly 16384 B, and the L1 set
//! is determined by address bits [11:6] (64 sets of 12 ways), so
//! 16384 mod 4096 = 0 - all 64 rows of a tile would fall into THE SAME L1 set
//! and tiling would stop working. With PAD = 8 (64 B = one cache line) each
//! successive row shifts by one set, and the 64 rows spread over 64 sets.
//! Padding does NOT change the summation order, so bit-for-bit equality holds.

/// Tile edge. Identical in both languages - change only together with the C++ side.
pub const B: usize = 64;

/// Row padding in elements. 8 f64 = 64 B = one cache line. This is the
/// SMALLEST padding that works: anything smaller does not move a row into a
/// different L1 set (tested - +8 and +16 B do not help at all).
pub const PAD: usize = 8;

/// Sequential multiplication C = A * B. Used as the T1 baseline and as the
/// correctness reference for the parallel variants.
pub fn multiply(a: &[f64], b: &[f64], c: &mut [f64], n: usize, lda: usize) {
    band_kernel(a, b, c, n, lda);
}

/// Computes a band of C rows.
///
/// `a` and `c` are slices of the same row band (same length, same starting
/// row), `b` is the whole matrix. All three have row stride `lda`, and only the
/// first `n` elements of each row are used. The number of rows is derived from
/// `c.len() / lda`.
///
/// The same function is used in all three variants. This guarantees that the
/// differences between them come only from the parallelisation method, not
/// from the algorithm: the sequential run is the same code with a single band
/// covering the whole matrix.
///
/// The summation order of every C element (increasing k) does not depend on
/// the band split, so the result is bit-for-bit identical for any number of
/// threads. This makes it possible to compare checksums directly.
pub fn band_kernel(a: &[f64], b: &[f64], c: &mut [f64], n: usize, lda: usize) {
    let rows = c.len() / lda;

    for ii in (0..rows).step_by(B) {
        let i_max = (ii + B).min(rows);
        for kk in (0..n).step_by(B) {
            let k_max = (kk + B).min(n);
            for jj in (0..n).step_by(B) {
                let j_max = (jj + B).min(n);

                for i in ii..i_max {
                    let a_row = &a[i * lda..i * lda + n];
                    let c_row = &mut c[i * lda + jj..i * lda + j_max];

                    for k in kk..k_max {
                        let aik = a_row[k];
                        let b_row = &b[k * lda + jj..k * lda + j_max];
                        // Innermost loop: adjacent C and B elements.
                        for (cv, bv) in c_row.iter_mut().zip(b_row.iter()) {
                            *cv += aik * bv;
                        }
                    }
                }
            }
        }
    }
}
