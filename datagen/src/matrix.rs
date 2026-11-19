//! Data for the matrix multiplication task.
//!
//! For every size n two square f64 matrices A and B are generated, with
//! elements uniformly distributed in the interval [0, 1).
//!
//! File format (all numbers little-endian):
//!   bytes  0..4    magic "RMTX"
//!   bytes  4..8    format version (u32) = 1
//!   bytes  8..16   number of rows (u64)
//!   bytes 16..24   number of columns (u64)
//!   bytes 24..     rows*cols f64 values stored by rows (row-major)

use crate::rng::Xoshiro256pp;
use crate::util::{human_bytes, sha256_file, Manifest};
use std::fs::File;
use std::io::{BufWriter, Result as IoResult, Write};
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"RMTX";
pub const VERSION: u32 = 1;
#[allow(dead_code)]
pub const HEADER_BYTES: u64 = 24;

fn write_matrix(path: &Path, n: usize, seed: u64, label: &str) -> IoResult<()> {
    let mut rng = Xoshiro256pp::stream(seed, label);
    let f = File::create(path)?;
    let mut w = BufWriter::with_capacity(1 << 20, f);

    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(n as u64).to_le_bytes())?;
    w.write_all(&(n as u64).to_le_bytes())?;

    // Written row by row, so that the memory requirement does not depend on n^2.
    let mut row = vec![0u8; n * 8];
    for _ in 0..n {
        for j in 0..n {
            let v = rng.next_f64();
            row[j * 8..(j + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }
        w.write_all(&row)?;
    }
    w.flush()?;
    Ok(())
}

pub fn generate(sizes: &[usize], out_dir: &Path, seed: u64, manifest: &mut Manifest) -> IoResult<()> {
    manifest.section("Matrix multiplication data");
    manifest.kv("Distribution", "uniform U[0, 1), f64");
    manifest.kv(
        "Format",
        "RMTX v1: magic(4) + version(u32) + rows(u64) + cols(u64) + f64 row-major",
    );

    for &n in sizes {
        for label_letter in ["A", "B"] {
            let name = format!("matrix_{}_{}.bin", label_letter, n);
            let path = out_dir.join(&name);
            let stream_label = format!("matrix/{}/{}", n, label_letter);
            write_matrix(&path, n, seed, &stream_label)?;

            let size = path.metadata()?.len();
            let sha = sha256_file(&path)?;
            println!("  {}  {}  {}", name, human_bytes(size), &sha[..16]);
            manifest.file_entry(&path, size, &sha, &format!("stream \"{}\"", stream_label));
        }
    }
    Ok(())
}
