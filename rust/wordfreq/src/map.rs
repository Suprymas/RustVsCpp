//! Hash map definition shared by both languages.
//!
//! WHY NOT THE DEFAULT `HashMap`: Rust uses SipHash-1-3 by default (resistant
//! to deliberately chosen keys), whereas libstdc++ `std::unordered_map` uses a
//! fast but non-resistant hash. Keeping the defaults would measure a
//! difference in security policy, not parallelisation. So both languages use
//! THE SAME FNV-1a hash (same constants as `io::bit_hash`).
//!
//! The cost of the default SipHash does not disappear from the study - it
//! remains as a separate side measurement (`--default-hasher`), because it is
//! an interesting Rust design decision, just not one about concurrency.
//!
//! IMPORTANT DETAIL so that both sides compute THE SAME: Rust's
//! `Hash for [u8]` first writes a length prefix (`write_length_prefix` -> 8
//! native-endian bytes), and only then the bytes. The C++ hash function does
//! exactly the same - first 8 `size_t` bytes, then the word. Otherwise the
//! Rust side would do ~2x more hashing work for short words and the
//! comparison would be unfair.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const OFFSET: u64 = 0xcbf29ce484222325;
const PRIME: u64 = 0x100000001b3;

#[derive(Clone, Copy)]
pub struct Fnv1a(u64);

impl Default for Fnv1a {
    #[inline]
    fn default() -> Self {
        Fnv1a(OFFSET)
    }
}

impl Hasher for Fnv1a {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
        self.0 = h;
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// Word -> number of occurrences.
pub type Counts = HashMap<Vec<u8>, u64, BuildHasherDefault<Fnv1a>>;

/// Empty map with pre-allocated capacity.
///
/// The capacity is given on purpose: without it the first thousands of
/// insertions would trigger several rehashes, and they would enter the
/// measured time unevenly for different thread counts (partial tables are
/// smaller, so fewer rehashes). The C++ side does the same via `reserve`.
pub fn new_counts(capacity: usize) -> Counts {
    Counts::with_capacity_and_hasher(capacity, BuildHasherDefault::<Fnv1a>::default())
}

/// Merges `src` into `dst`. Used by the partial-tables strategy.
pub fn merge_into(dst: &mut Counts, src: Counts) {
    for (word, count) in src {
        *dst.entry(word).or_insert(0) += count;
    }
}
