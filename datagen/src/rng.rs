//! Xoshiro256++ pseudo-random number generator.
//!
//! The implementation follows the reference C code by D. Blackman and
//! S. Vigna (https://prng.di.unimi.it/xoshiro256plusplus.c). The generator is
//! embedded directly in this project rather than taken from an external
//! library, so that data reproducibility does not depend on a library
//! version.
//!
//! The initial state is expanded from a 64-bit seed with the SplitMix64
//! algorithm, as recommended by the authors.

/// SplitMix64 - used only to initialise the Xoshiro256++ state and to derive
/// the seeds of separate streams.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Xoshiro256++ generator.
pub struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    /// Creates a generator from a 64-bit seed.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            *v = sm.next_u64();
        }
        // xoshiro does not support an all-zero state; it practically never
        // happens, but the check costs nothing.
        if s.iter().all(|&v| v == 0) {
            s[0] = 0x9E37_79B9_7F4A_7C15;
        }
        Self { s }
    }

    /// An independent stream derived from the master seed and a text label.
    /// This way generating each file does not depend on the order in which
    /// the files were generated.
    pub fn stream(master_seed: u64, label: &str) -> Self {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325; // FNV-1a 64 offset basis
        for b in label.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3); // FNV-1a 64 prime
        }
        let mut sm = SplitMix64::new(master_seed ^ h);
        Self::seed_from_u64(sm.next_u64())
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }

    /// Uniformly distributed f64 in the interval [0, 1).
    /// Uses the 53 most significant bits - the standard approach that ensures
    /// every representable value gets the correct probability.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 * SCALE
    }

    /// Uniformly distributed integer in the interval [0, n).
    /// Lemire's method, division-free in most cases, without modulo bias.
    #[inline]
    pub fn next_bounded(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        let mut x = self.next_u64();
        let mut m = (x as u128) * (n as u128);
        let mut l = m as u64;
        if l < n {
            let threshold = n.wrapping_neg() % n;
            while l < threshold {
                x = self.next_u64();
                m = (x as u128) * (n as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference sequence: state {1, 2, 3, 4} as in xoshiro256plusplus.c.
    #[test]
    fn reference_sequence() {
        let mut r = Xoshiro256pp { s: [1, 2, 3, 4] };
        let expected: [u64; 5] = [
            41943041,
            58720359,
            3588806011781223,
            3591011842654386,
            9228616714210784205,
        ];
        for e in expected {
            assert_eq!(r.next_u64(), e);
        }
    }

    #[test]
    fn f64_in_range() {
        let mut r = Xoshiro256pp::seed_from_u64(20260101);
        for _ in 0..100_000 {
            let v = r.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn bounded_in_range() {
        let mut r = Xoshiro256pp::seed_from_u64(7);
        for _ in 0..100_000 {
            assert!(r.next_bounded(1000) < 1000);
        }
    }

    #[test]
    fn streams_differ_but_are_stable() {
        let a1 = Xoshiro256pp::stream(1, "matrix_A_512").next_u64();
        let a2 = Xoshiro256pp::stream(1, "matrix_A_512").next_u64();
        let b = Xoshiro256pp::stream(1, "matrix_B_512").next_u64();
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
    }
}
