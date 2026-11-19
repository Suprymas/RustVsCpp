//! Reader for the RMTX v1 format.
//!
//! Format (all numbers little-endian):
//!   bytes  0..4    magic "RMTX"
//!   bytes  4..8    format version (u32) = 1
//!   bytes  8..16   number of rows (u64)
//!   bytes 16..24   number of columns (u64)
//!   bytes 24..     rows*cols f64 values by rows (row-major)

use std::fs;
use std::io::{Error, ErrorKind, Result as IoResult};
use std::path::{Path, PathBuf};

pub const MAGIC: &[u8; 4] = b"RMTX";
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 24;

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    /// Actual row stride in memory (`cols + pad`), see `seq::PAD`.
    pub lda: usize,
    pub data: Vec<f64>,
}

fn bad(msg: String) -> Error {
    Error::new(ErrorKind::InvalidData, msg)
}

/// Reads RMTX v1 into a PADDED buffer: the row stride is `cols + pad`, and
/// the `pad` elements at the end of each row stay zero and take no part in
/// the computation. Reason - cache set conflicts at power-of-two strides; see
/// `seq::PAD` and CLAUDE.md decision 12.
///
/// Padding is done while reading, i.e. OUTSIDE the timer, so it costs the
/// measurements nothing.
pub fn read_rmtx(path: &Path, pad: usize) -> IoResult<Matrix> {
    let bytes = fs::read(path)?;
    if bytes.len() < HEADER_BYTES {
        return Err(bad(format!("{}: file too short", path.display())));
    }
    if &bytes[0..4] != MAGIC {
        return Err(bad(format!("{}: bad magic (expected RMTX)", path.display())));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(bad(format!("{}: unsupported version {}", path.display(), version)));
    }
    let rows = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let cols = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;

    let expected = HEADER_BYTES + rows * cols * 8;
    if bytes.len() != expected {
        return Err(bad(format!(
            "{}: size {} bytes, expected {} ({}x{})",
            path.display(),
            bytes.len(),
            expected,
            rows,
            cols
        )));
    }

    let lda = cols + pad;
    let mut data = vec![0.0f64; rows * lda];
    for (r, dst) in data.chunks_exact_mut(lda).enumerate() {
        let src = &bytes[HEADER_BYTES + r * cols * 8..HEADER_BYTES + (r + 1) * cols * 8];
        for (cell, w) in dst[..cols].iter_mut().zip(src.chunks_exact(8)) {
            *cell = f64::from_le_bytes(w.try_into().unwrap());
        }
    }

    Ok(Matrix { rows, cols, lda, data })
}

/// Locates the data directory: the given path is checked first, then "data"
/// is searched for while walking upwards. This way the program works when
/// started from rust/matmul, from the project root, and via the harness.
pub fn find_data_dir(explicit: Option<&str>) -> IoResult<PathBuf> {
    if let Some(p) = explicit {
        let path = PathBuf::from(p);
        if path.is_dir() {
            return Ok(path);
        }
        return Err(bad(format!("specified data directory not found: {}", p)));
    }
    for prefix in ["data", "../data", "../../data", "../../../data"] {
        let path = PathBuf::from(prefix);
        if path.is_dir() {
            return Ok(path);
        }
    }
    Err(bad("data directory not found (use --data)".to_string()))
}

/// Checksum: a plain sum in a fixed order.
///
/// The summation order of each C element in the kernel does not depend on the
/// number of threads, so this sum is identical across all variants as well.
/// It is printed as the hexadecimal bit pattern to avoid text-conversion
/// errors.
///
/// IMPORTANT: this sum does NOT prove bit-for-bit equality - see `bit_hash`.
/// `lda` is the row stride, `n` the number of meaningful columns. Padding
/// elements are NOT included, so the value does not depend on PAD and matches
/// the one produced by the unpadded version.
pub fn checksum(c: &[f64], n: usize, lda: usize) -> f64 {
    let mut s = 0.0f64;
    for row in c.chunks_exact(lda) {
        for &v in &row[..n] {
            s += v;
        }
    }
    s
}

/// Bit hash of the whole matrix (FNV-1a with extra mixing).
///
/// Why `checksum` is not enough: at N = 2048 every C element is about 512, so
/// the total is about 2.1e9, and the f64 resolution at that magnitude is about
/// 5e-7. Differences in individual elements (~1e-11) simply fall below that
/// resolution and the sum stays the same. Verified in practice: the C++
/// version compiled WITHOUT `-ffp-contract=off` gives the same sum for all
/// three sizes, even though the actual elements differ. So the sum checks
/// about 7 significant digits, not bits.
///
/// This hash is sensitive to every bit, so it is the actual proof for
/// decision 9 (bit-for-bit across languages). The implementation must be
/// literally the same on the C++ side.
/// As with `checksum`, padding columns are skipped - so the hash does not
/// depend on PAD and is comparable with earlier series.
pub fn bit_hash(c: &[f64], n: usize, lda: usize) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    for row in c.chunks_exact(lda) {
        for &v in &row[..n] {
            h ^= v.to_bits();
            h = h.wrapping_mul(PRIME);
            // Extra mixing: without it the multiplication would propagate
            // differences in the top bits only into the top bits, and they
            // would quickly be shifted out.
            h ^= h >> 33;
        }
    }
    h
}

/// Peak resident memory of the process (VmHWM) in kibibytes.
pub fn peak_rss_kib() -> u64 {
    let status = match fs::read_to_string("/proc/self/status") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Some(num) = rest.split_whitespace().next() {
                return num.parse().unwrap_or(0);
            }
        }
    }
    0
}
