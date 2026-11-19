//! Test data generator.
//!
//! The data are generated ONCE by this program and written to files that the
//! Rust and C++ implementations only read. This removes the risk that the
//! languages' different pseudo-random number generators would produce
//! different input values from the same seed.
//!
//! Usage:
//!   datagen matrix    [--sizes 512,1024,2048]
//!   datagen graph     [--scales 20,22,24] [--edgefactor 16] [--no-permute]
//!   datagen corpus    --input <directory> [--sizes 0.5G,2G,8G] [--no-reference]
//!   datagen reference --file <file> [--top 20]
//!   datagen checksums
//!
//! Common options: --out <directory> (default ./data), --seed <num>

mod corpus;
mod graph;
mod matrix;
mod rng;
mod util;

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use util::{human_bytes, parse_size, sha256_file, Manifest};

const DEFAULT_SEED: u64 = 20_260_101;
const DEFAULT_OUT: &str = "data";
const MANIFEST_NAME: &str = "MANIFEST.md";
const CHECKSUM_NAME: &str = "checksums.sha256";

fn usage() {
    eprintln!(
        "datagen - test data generator\n\
         \n\
         Commands:\n\
         \x20 matrix     [--sizes 512,1024,2048]\n\
         \x20 graph      [--scales 20,22,24] [--edgefactor 16] [--no-permute]\n\
         \x20 corpus     --input <directory> [--sizes 0.5G,2G,8G] [--keep-tmp] [--no-reference]\n\
         \x20 reference  --file <file> [--top 20]\n\
         \x20 checksums\n\
         \n\
         Common options:\n\
         \x20 --out <directory>   output directory (default: {})\n\
         \x20 --seed <num>        master seed (default: {})",
        DEFAULT_OUT, DEFAULT_SEED
    );
}

/// Simple argument parsing: --key value, --flag.
fn parse_args(args: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some(key) = a.strip_prefix("--") {
            let next_is_value = args
                .get(i + 1)
                .map(|v| !v.starts_with("--"))
                .unwrap_or(false);
            if next_is_value {
                map.insert(key.to_string(), args[i + 1].clone());
                i += 2;
            } else {
                map.insert(key.to_string(), String::new());
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    map
}

fn parse_list<T: std::str::FromStr>(s: &str) -> Result<Vec<T>, String> {
    s.split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<T>().map_err(|_| format!("bad value: {}", x)))
        .collect()
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 {
        usage();
        return Err("no command given".into());
    }
    let cmd = argv[1].clone();
    let opts = parse_args(&argv[2..]);

    let out_dir = PathBuf::from(opts.get("out").filter(|s| !s.is_empty()).map(|s| s.as_str()).unwrap_or(DEFAULT_OUT));
    let seed: u64 = match opts.get("seed") {
        Some(s) if !s.is_empty() => s.parse().map_err(|_| "bad --seed value".to_string())?,
        _ => DEFAULT_SEED,
    };
    fs::create_dir_all(&out_dir).map_err(|e| format!("failed to create {}: {}", out_dir.display(), e))?;
    let manifest_path = out_dir.join(MANIFEST_NAME);

    match cmd.as_str() {
        "matrix" => {
            let sizes: Vec<usize> = parse_list(
                opts.get("sizes").filter(|s| !s.is_empty()).map(|s| s.as_str()).unwrap_or("512,1024,2048"),
            )?;
            println!("Matrices: {:?}, seed {}", sizes, seed);
            let mut m = Manifest::new("Data set: matrices", seed);
            matrix::generate(&sizes, &out_dir, seed, &mut m).map_err(|e| e.to_string())?;
            m.append_to(&manifest_path).map_err(|e| e.to_string())?;
        }
        "graph" => {
            let scales: Vec<u32> = parse_list(
                opts.get("scales").filter(|s| !s.is_empty()).map(|s| s.as_str()).unwrap_or("20,22,24"),
            )?;
            let degree: u64 = match opts.get("edgefactor").or_else(|| opts.get("degree")) {
                Some(s) if !s.is_empty() => s.parse().map_err(|_| "bad --edgefactor value".to_string())?,
                _ => 16,
            };
            let permute = !opts.contains_key("no-permute");
            println!("Graphs: scale {:?}, edge factor {}, seed {}", scales, degree, seed);
            let mut m = Manifest::new("Data set: graph", seed);
            graph::generate(&scales, degree, &out_dir, seed, permute, &mut m).map_err(|e| e.to_string())?;
            m.append_to(&manifest_path).map_err(|e| e.to_string())?;
        }
        "corpus" => {
            let input = opts
                .get("input")
                .filter(|s| !s.is_empty())
                .ok_or("--input <directory with .txt files> is required")?;
            let sizes: Vec<u64> = opts
                .get("sizes")
                .filter(|s| !s.is_empty())
                .map(|s| s.as_str())
                .unwrap_or("0.5G,2G,8G")
                .split(',')
                .map(|x| parse_size(x))
                .collect::<Result<_, _>>()?;
            let keep_tmp = opts.contains_key("keep-tmp");
            let reference = !opts.contains_key("no-reference");
            println!(
                "Corpus from {}: sizes {:?}",
                input,
                sizes.iter().map(|&s| human_bytes(s)).collect::<Vec<_>>()
            );
            let mut m = Manifest::new("Data set: corpus", seed);
            corpus::generate(
                Path::new(input),
                &sizes,
                &out_dir,
                keep_tmp,
                reference,
                &mut m,
            )
            .map_err(|e| e.to_string())?;
            m.append_to(&manifest_path).map_err(|e| e.to_string())?;
        }
        "reference" => {
            let file = opts
                .get("file")
                .filter(|s| !s.is_empty())
                .ok_or("--file <file> is required")?;
            let top: usize = match opts.get("top") {
                Some(s) if !s.is_empty() => s.parse().map_err(|_| "bad --top value".to_string())?,
                _ => 20,
            };
            let (total, items) =
                corpus::reference_counts(Path::new(file), top).map_err(|e| e.to_string())?;
            println!("Total words: {}", total);
            for (i, (w, c)) in items.iter().enumerate() {
                println!("{:3}. {:20} {}", i + 1, w, c);
            }
        }
        "checksums" => {
            let mut entries: Vec<PathBuf> = fs::read_dir(&out_dir)
                .map_err(|e| e.to_string())?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.is_file()
                        && p.file_name().map(|n| n != MANIFEST_NAME && n != CHECKSUM_NAME).unwrap_or(false)
                })
                .collect();
            entries.sort();
            let mut out = String::new();
            for p in &entries {
                let sha = sha256_file(p).map_err(|e| e.to_string())?;
                let name = p.file_name().unwrap().to_string_lossy();
                println!("{}  {}", sha, name);
                out.push_str(&format!("{}  {}\n", sha, name));
            }
            fs::write(out_dir.join(CHECKSUM_NAME), out).map_err(|e| e.to_string())?;
            println!("Written: {}", out_dir.join(CHECKSUM_NAME).display());
        }
        "--help" | "-h" | "help" => {
            usage();
        }
        other => {
            usage();
            return Err(format!("unknown command: {}", other));
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::FAILURE
        }
    }
}
