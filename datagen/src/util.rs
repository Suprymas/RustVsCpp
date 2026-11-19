//! Helper functions: checksums, size formatting, writing the
//! reproducibility manifest.

use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Result as IoResult, Write};
use std::path::Path;

/// Computes a file's SHA-256 checksum without reading the whole file into memory.
pub fn sha256_file(path: &Path) -> IoResult<String> {
    let f = File::open(path)?;
    let mut reader = BufReader::with_capacity(1 << 20, f);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Prints a byte count in human-readable form.
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{} {}", n, UNITS[0])
    } else {
        format!("{:.2} {}", v, UNITS[i])
    }
}

/// Short size label for a file name: 536870912 -> "512M", 2 GiB -> "2G".
pub fn size_label(n: u64) -> String {
    const UNITS: [(u64, &str); 4] = [(1 << 40, "T"), (1 << 30, "G"), (1 << 20, "M"), (1 << 10, "K")];
    for (m, s) in UNITS {
        if n >= m {
            let q = n as f64 / m as f64;
            return if (q - q.round()).abs() < 1e-9 {
                format!("{}{}", q.round() as u64, s)
            } else {
                format!("{:.1}{}", q, s)
            };
        }
    }
    format!("{}B", n)
}

/// Parses a size from text: "512", "0.5G", "2G", "8GiB", "700M".
/// Binary multipliers are used (1G = 2^30).
pub fn parse_size(s: &str) -> Result<u64, String> {
    let t = s.trim().to_ascii_lowercase();
    let t = t.trim_end_matches("ib").trim_end_matches('b').to_string();
    let (num, mult) = match t.chars().last() {
        Some('k') => (&t[..t.len() - 1], 1u64 << 10),
        Some('m') => (&t[..t.len() - 1], 1u64 << 20),
        Some('g') => (&t[..t.len() - 1], 1u64 << 30),
        Some('t') => (&t[..t.len() - 1], 1u64 << 40),
        _ => (&t[..], 1u64),
    };
    let v: f64 = num
        .parse()
        .map_err(|_| format!("cannot parse size: {}", s))?;
    if v < 0.0 {
        return Err(format!("negative size: {}", s));
    }
    Ok((v * mult as f64) as u64)
}

/// Manifest with the reproducibility information given in the appendices.
pub struct Manifest {
    lines: Vec<String>,
}

impl Manifest {
    pub fn new(title: &str, seed: u64) -> Self {
        let mut lines = Vec::new();
        lines.push(format!("# {}", title));
        lines.push(String::new());
        lines.push(format!("Generator: datagen {}", env!("CARGO_PKG_VERSION")));
        lines.push("PRNG algorithm: Xoshiro256++ (seed expansion SplitMix64)".to_string());
        lines.push(format!("Master seed: {}", seed));
        lines.push(String::new());
        Self { lines }
    }

    pub fn section(&mut self, name: &str) {
        self.lines.push(String::new());
        self.lines.push(format!("## {}", name));
        self.lines.push(String::new());
    }

    pub fn kv(&mut self, key: &str, value: &str) {
        self.lines.push(format!("{}: {}", key, value));
    }

    pub fn file_entry(&mut self, path: &Path, size: u64, sha: &str, note: &str) {
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        self.lines.push(format!(
            "{}  {}  sha256={}{}",
            name,
            human_bytes(size),
            sha,
            if note.is_empty() {
                String::new()
            } else {
                format!("  ({})", note)
            }
        ));
    }

    /// Appends the manifest content to a file (or creates a new one).
    pub fn append_to(&self, path: &Path) -> IoResult<()> {
        let mut f = OpenOptions::new().create(true).append(true).open(path)?;
        for l in &self.lines {
            writeln!(f, "{}", l)?;
        }
        writeln!(f)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vector() {
        let mut h = Sha256::new();
        h.update(b"abc");
        assert_eq!(
            hex(&h.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn labels() {
        assert_eq!(size_label(1 << 29), "512M");
        assert_eq!(size_label(2 << 30), "2G");
        assert_eq!(size_label(8u64 << 30), "8G");
        assert_eq!(size_label(1536 << 20), "1.5G");
    }

    #[test]
    fn sizes() {
        assert_eq!(parse_size("512").unwrap(), 512);
        assert_eq!(parse_size("2G").unwrap(), 2 << 30);
        assert_eq!(parse_size("0.5G").unwrap(), 1 << 29);
        assert_eq!(parse_size("8GiB").unwrap(), 8u64 << 30);
        assert_eq!(parse_size("700M").unwrap(), 700 << 20);
    }
}
