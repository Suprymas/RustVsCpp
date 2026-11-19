//! Sequential word frequency count - the T1 baseline (decision 1).
//!
//! No threads, no atomics, no locks. Calls the same `par::count_chunk` as both
//! parallel variants, so the algorithm is guaranteed identical and the
//! difference comes solely from the parallelisation method.

use crate::map::{new_counts, Counts};
use crate::par::count_chunk;

pub fn count(data: &[u8], capacity: usize) -> Counts {
    let mut counts = new_counts(capacity);
    count_chunk(data, &mut counts);
    counts
}
