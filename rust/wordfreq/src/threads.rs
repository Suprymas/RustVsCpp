//! Low-level parallelisation: `std::thread` without any library.
//! C++ counterpart - threads.cpp.
//!
//! Both strategies live here, because they differ only in where the result is
//! accumulated: `shared` - a single locked table, `partial` - one table per
//! thread with a merge at the end. The merge IS part of the algorithm, so it
//! is included in the measured time.

use std::sync::Mutex;

use crate::map::{merge_into, new_counts, Counts};
use crate::par::{count_chunk, count_chunk_shared};

/// Shared locked table. The lock is taken for every word.
pub fn count_shared(
    data: &[u8],
    bounds: &[(usize, usize)],
    capacity: usize,
    core_ids: &[core_affinity::CoreId],
) -> Counts {
    let shared = Mutex::new(new_counts(capacity));

    std::thread::scope(|scope| {
        for (t, &(start, end)) in bounds.iter().enumerate() {
            let shared = &shared;
            scope.spawn(move || {
                if !core_ids.is_empty() {
                    core_affinity::set_for_current(core_ids[t % core_ids.len()]);
                }
                count_chunk_shared(&data[start..end], shared);
            });
        }
    });

    shared.into_inner().unwrap()
}

/// Partial tables with a merge at the end.
///
/// Every thread has its own table that nobody else touches, so no lock is
/// needed at all. On the Rust side this is visible in the types: a thread gets
/// `&mut Counts`, and the compiler itself checks that no two threads get the
/// same one. `chunks_mut` plays the same role here as in matrix
/// multiplication.
pub fn count_partial(
    data: &[u8],
    bounds: &[(usize, usize)],
    capacity: usize,
    core_ids: &[core_affinity::CoreId],
) -> Counts {
    let per_thread = capacity / bounds.len().max(1);
    let mut locals: Vec<Counts> = (0..bounds.len()).map(|_| new_counts(per_thread)).collect();

    std::thread::scope(|scope| {
        for ((t, &(start, end)), local) in bounds.iter().enumerate().zip(locals.iter_mut()) {
            scope.spawn(move || {
                if !core_ids.is_empty() {
                    core_affinity::set_for_current(core_ids[t % core_ids.len()]);
                }
                count_chunk(&data[start..end], local);
            });
        }
    });

    // Merge: the first table becomes the result, the others are merged into it.
    let mut it = locals.into_iter();
    let mut result = it.next().unwrap_or_else(|| new_counts(capacity));
    for local in it {
        merge_into(&mut result, local);
    }
    result
}
