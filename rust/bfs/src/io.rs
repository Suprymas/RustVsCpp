//! Reader for the RCSR v1 format and helper measurement functions.
//!
//! Format (all numbers little-endian):
//!   bytes  0..4    magic "RCSR"
//!   bytes  4..8    format version (u32) = 1
//!   bytes  8..16   number of vertices n (u64)
//!   bytes 16..24   number of directed entries m (u64)
//!   bytes 24..32   number of search sources s (u64)
//!   then           offsets array (n+1) x u64
//!   then           neighbours array m x u32
//!   then           search source vertices s x u32

use std::fs::File;
use std::io::{BufReader, Error, ErrorKind, Read, Result as IoResult};
use std::path::{Path, PathBuf};

pub const MAGIC: &[u8; 4] = b"RCSR";
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 32;

/// Graph in compressed sparse row (CSR) form.
pub struct Csr {
    pub offsets: Vec<u64>,
    pub neighbors: Vec<u32>,
    pub sources: Vec<u32>,
}

impl Csr {
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    pub fn num_entries(&self) -> usize {
        self.neighbors.len()
    }

    #[inline]
    pub fn neighbors_of(&self, v: usize) -> &[u32] {
        &self.neighbors[self.offsets[v] as usize..self.offsets[v + 1] as usize]
    }
}

fn bad(msg: String) -> Error {
    Error::new(ErrorKind::InvalidData, msg)
}

/// Read buffer size. The file is read in chunks rather than all at once: at
/// scale 24 the file takes 2 GiB, and `fs::read` would temporarily double the
/// memory requirement. Peak memory (VmHWM) is one of the study's metrics, so
/// this has to be done properly.
const CHUNK_BYTES: usize = 1 << 20;

/// Reads exactly `count` u64 values.
///
/// For comparison: on the C++ side a single line suffices,
/// `in.read(reinterpret_cast<char*>(v.data()), bytes)`, because the byte
/// stream can be directed straight into the vector's memory. Safe Rust does
/// not allow that without `unsafe`, so the conversion is done in chunks. Cost
/// - a few extra lines; benefit - no way to get sizes or alignment wrong.
fn read_u64s<R: Read>(r: &mut R, count: usize) -> IoResult<Vec<u64>> {
    let mut out = Vec::with_capacity(count);
    let mut buf = vec![0u8; CHUNK_BYTES];
    let mut left = count;
    while left > 0 {
        let take = left.min(CHUNK_BYTES / 8);
        let bytes = take * 8;
        r.read_exact(&mut buf[..bytes])?;
        out.extend(buf[..bytes].chunks_exact(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())));
        left -= take;
    }
    Ok(out)
}

fn read_u32s<R: Read>(r: &mut R, count: usize) -> IoResult<Vec<u32>> {
    let mut out = Vec::with_capacity(count);
    let mut buf = vec![0u8; CHUNK_BYTES];
    let mut left = count;
    while left > 0 {
        let take = left.min(CHUNK_BYTES / 4);
        let bytes = take * 4;
        r.read_exact(&mut buf[..bytes])?;
        out.extend(buf[..bytes].chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())));
        left -= take;
    }
    Ok(out)
}

pub fn read_rcsr(path: &Path) -> IoResult<Csr> {
    let file = File::open(path)?;
    let expected_size = file.metadata()?.len();
    let mut r = BufReader::with_capacity(CHUNK_BYTES, file);

    let mut header = [0u8; HEADER_BYTES];
    r.read_exact(&mut header)
        .map_err(|_| bad(format!("{}: file too short", path.display())))?;
    if &header[0..4] != MAGIC {
        return Err(bad(format!("{}: bad magic (expected RCSR)", path.display())));
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(bad(format!("{}: unsupported version {}", path.display(), version)));
    }
    let n = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
    let m = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;
    let s = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;

    let want = HEADER_BYTES as u64 + (n as u64 + 1) * 8 + m as u64 * 4 + s as u64 * 4;
    if expected_size != want {
        return Err(bad(format!(
            "{}: size {} bytes, expected {} (n={}, m={}, s={})",
            path.display(),
            expected_size,
            want,
            n,
            m,
            s
        )));
    }

    let offsets = read_u64s(&mut r, n + 1)?;
    let neighbors = read_u32s(&mut r, m)?;
    let sources = read_u32s(&mut r, s)?;

    if offsets[n] as usize != m {
        return Err(bad(format!(
            "{}: last offset {} does not match m = {}",
            path.display(),
            offsets[n],
            m
        )));
    }

    Ok(Csr { offsets, neighbors, sources })
}

/// Locates the data directory: the given path is checked first, then "data"
/// is searched for while walking upwards.
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

/// Bit hash of the distance array (FNV-1a with extra mixing).
///
/// Same implementation as on the matmul side, only the input is u32. This
/// hash, not the sum of distances, is the proof for decision 9 (bit-for-bit
/// across languages): the sum would not notice two distances changing at each
/// other's expense.
///
/// For BFS the result is independent of the thread count in principle: a
/// level-synchronous search assigns every vertex its shortest distance, which
/// is unique. So the hash must match in ALL variants and both languages.
pub fn bit_hash_u32(d: &[u32]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    for &v in d {
        h ^= v as u64;
        h = h.wrapping_mul(PRIME);
        h ^= h >> 33;
    }
    h
}

/// Peak resident memory of the process (VmHWM) in kibibytes.
pub fn peak_rss_kib() -> u64 {
    let status = match std::fs::read_to_string("/proc/self/status") {
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
