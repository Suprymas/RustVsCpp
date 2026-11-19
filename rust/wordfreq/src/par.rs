//! Part shared by both parallel levels and both strategies.
//!
//! C++ counterpart - par.hpp. Only what does NOT depend on the parallelisation
//! method lives here: word splitting and counting one band. This keeps
//! `threads.rs` and `rayon_impl.rs` pure parallelisation code, as in the other
//! two tasks (important for the dLOC metric).

use std::sync::Mutex;

use crate::map::Counts;

/// Counts the words of a band into a LOCAL map.
///
/// The corpus is normalised: lowercase ASCII letters only, words separated by
/// a single space or a line end. So splitting is a simple scan, and it is
/// literally the same in both languages.
///
/// A new word is allocated only the first time it is seen (`to_vec`); later
/// occurrences allocate nothing. The C++ side does the same via heterogeneous
/// lookup (`is_transparent`); otherwise it would allocate a `std::string` for
/// every word and we would be measuring the allocator, not parallelisation.
#[inline]
pub fn count_chunk(data: &[u8], out: &mut Counts) {
    let mut i = 0usize;
    let n = data.len();
    while i < n {
        // Skip separators.
        while i < n && (data[i] == b' ' || data[i] == b'\n') {
            i += 1;
        }
        let start = i;
        while i < n && data[i] != b' ' && data[i] != b'\n' {
            i += 1;
        }
        if i > start {
            let word = &data[start..i];
            if let Some(c) = out.get_mut(word) {
                *c += 1;
            } else {
                out.insert(word.to_vec(), 1);
            }
        }
    }
}

/// Same, but writes straight into the SHARED locked table.
///
/// The lock is taken for EVERY word - that is precisely the naive strategy
/// whose cost the study measures. Batching (taking the lock less often) would
/// simply be the partial-tables strategy, written badly.
#[inline]
pub fn count_chunk_shared(data: &[u8], shared: &Mutex<Counts>) {
    let mut i = 0usize;
    let n = data.len();
    while i < n {
        while i < n && (data[i] == b' ' || data[i] == b'\n') {
            i += 1;
        }
        let start = i;
        while i < n && data[i] != b' ' && data[i] != b'\n' {
            i += 1;
        }
        if i > start {
            let word = &data[start..i];
            let mut guard = shared.lock().unwrap();
            if let Some(c) = guard.get_mut(word) {
                *c += 1;
            } else {
                guard.insert(word.to_vec(), 1);
            }
        }
    }
}

/// Splits the corpus into `parts` bands, cutting ONLY at separators.
///
/// Computed OUTSIDE the timer and passed in, like the core list on the matmul
/// side: this is preparation, not measured work. The split is static, because
/// the amount of work is proportional to the number of bytes and the bands
/// are equal.
pub fn split_bounds(data: &[u8], parts: usize) -> Vec<(usize, usize)> {
    let n = data.len();
    let parts = parts.max(1);
    let mut bounds = Vec::with_capacity(parts);
    let mut start = 0usize;
    for p in 0..parts {
        let mut end = if p + 1 == parts { n } else { n * (p + 1) / parts };
        // The boundary is pushed forward to the nearest separator so that a
        // word is not cut in half and counted in both bands as two bogus words.
        while end < n && data[end] != b' ' && data[end] != b'\n' {
            end += 1;
        }
        if end > start {
            bounds.push((start, end));
        }
        start = end;
    }
    bounds
}
