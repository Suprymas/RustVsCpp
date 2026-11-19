//! High-level parallelisation: Rayon. C++ counterpart - OpenMP.
//!
//! The pool is created in `main.rs` outside the timer, as in the other tasks.

use rayon::prelude::*;
use std::sync::Mutex;

use crate::map::{merge_into, new_counts, Counts};
use crate::par::{count_chunk, count_chunk_shared};

/// Shared locked table.
pub fn count_shared(data: &[u8], bounds: &[(usize, usize)], capacity: usize) -> Counts {
    let shared = Mutex::new(new_counts(capacity));
    bounds
        .par_iter()
        .for_each(|&(start, end)| count_chunk_shared(&data[start..end], &shared));
    shared.into_inner().unwrap()
}

/// Partial tables with a merge.
///
/// `fold` gives one table to each Rayon work unit, `reduce` merges them.
/// Unlike the low-level version, there is no need to create an array of
/// tables up front or to write a merge loop - the expression itself does both.
/// That is exactly the benefit of a high-level library that the dLOC metric
/// measures.
pub fn count_partial(data: &[u8], bounds: &[(usize, usize)], capacity: usize) -> Counts {
    let per_thread = capacity / bounds.len().max(1);
    bounds
        .par_iter()
        .fold(
            || new_counts(per_thread),
            |mut acc, &(start, end)| {
                count_chunk(&data[start..end], &mut acc);
                acc
            },
        )
        .reduce(
            || new_counts(per_thread),
            |mut a, b| {
                merge_into(&mut a, b);
                a
            },
        )
}
