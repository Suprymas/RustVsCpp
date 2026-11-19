//! Corpus reading and helper measurement functions.
//! C++ counterpart - io.hpp / io.cpp.

use std::fs::{self, File};
use std::io::{Error, ErrorKind, Read, Result as IoResult};
use std::path::{Path, PathBuf};

use crate::map::Counts;

fn bad(msg: String) -> Error {
    Error::new(ErrorKind::InvalidData, msg)
}

/// Reads the whole corpus into memory.
///
/// Read into a pre-allocated buffer rather than via `fs::read`: the latter
/// grows its buffer by doubling, so for an 8 GiB file peak memory would
/// briefly reach ~12 GiB and the VmHWM metric would lie. Same decision as on
/// the BFS side.
pub fn read_corpus(path: &Path) -> IoResult<Vec<u8>> {
    let len = fs::metadata(path)?.len() as usize;
    let mut file = File::open(path)?;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

/// Finds the corpus file by its size label (e.g. "512M", "2G", "8G").
pub fn corpus_path(dir: &Path, size: &str) -> IoResult<PathBuf> {
    let direct = dir.join(format!("corpus_{}.txt", size));
    if direct.is_file() {
        return Ok(direct);
    }
    // If the label does not match exactly, show what is there - makes the error useful.
    let mut found: Vec<String> = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("corpus_") && name.ends_with(".txt") {
                found.push(name);
            }
        }
    }
    found.sort();
    Err(bad(format!(
        "{} not found; the directory contains: {}",
        direct.display(),
        if found.is_empty() { "nothing".to_string() } else { found.join(", ") }
    )))
}

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

/// Bit hash of the result: FNV-1a over the SORTED (word, count) list.
///
/// Sorting is required, because hash map iteration order is unspecified and
/// differs between languages and between runs. Once sorted the result is
/// unique, and the hash can be used to compare Rust with C++ - as in
/// decision 9.
///
/// The implementation must be literally the same on the C++ side.
pub fn result_hash(counts: &Counts) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut items: Vec<(&[u8], u64)> = counts.iter().map(|(w, &c)| (w.as_slice(), c)).collect();
    items.sort_unstable();

    let mut h = OFFSET;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
            h ^= h >> 33;
        }
    };
    for (word, count) in items {
        feed(word);
        feed(&count.to_le_bytes());
    }
    h
}

/// Total number of words (sum of all frequencies).
pub fn total_words(counts: &Counts) -> u64 {
    counts.values().sum()
}

/// The `n` most frequent words, sorted by frequency descending, then by word
/// ascending - so that the order is unique and matches the `datagen`
/// reference.
pub fn top_n(counts: &Counts, n: usize) -> Vec<(String, u64)> {
    let mut items: Vec<(&[u8], u64)> = counts.iter().map(|(w, &c)| (w.as_slice(), c)).collect();
    items.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    items
        .into_iter()
        .take(n)
        .map(|(w, c)| (String::from_utf8_lossy(w).into_owned(), c))
        .collect()
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
