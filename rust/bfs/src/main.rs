//! Breadth-first search (BFS) benchmark program (Rust).
//!
//! Three variants:
//!   seq     - sequential version without parallelisation primitives (T1 baseline)
//!   threads - low level: std::thread + Barrier
//!   rayon   - high level: Rayon
//!
//! Usage:
//!   bfs --mode seq|threads|rayon --scale 20 [--source 0] [--threads 8]
//!       [--reps 10] [--warmup 1] [--data <directory>] [--affinity]
//!       [--verify] [--csv]
//!
//! ONLY the search is timed. File reading, memory allocation, filling `dist`
//! and Rayon pool creation stay outside the timer.

mod io;
mod par;
mod rayon_impl;
mod seq;
mod threads;

use std::collections::HashMap;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use par::UNVISITED;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const DEFAULT_REPS: usize = 10;
const DEFAULT_WARMUP: usize = 1;

fn usage() {
    eprintln!(
        "bfs - breadth-first search benchmarks (Rust)\n\
         \n\
         Required:\n\
         \x20 --mode seq|threads|rayon\n\
         \x20 --scale <20|22|24>\n\
         \n\
         Optional:\n\
         \x20 --source <idx>    index of the search source in the file, 0..7 (default: 0)\n\
         \x20 --threads <num>   number of threads (default: number of cores)\n\
         \x20 --reps <num>      measured repetitions (default: {})\n\
         \x20 --warmup <num>    warm-up runs, not included in results (default: {})\n\
         \x20 --data <dir>      data directory (default: searches ./data, ../data, ...)\n\
         \x20 --affinity        pin threads to cores (main series)\n\
         \x20 --verify          compare distances bit-for-bit with the sequential run\n\
         \x20 --csv             output CSV lines instead of human-readable text",
        DEFAULT_REPS, DEFAULT_WARMUP
    );
}

/// Simple argument parsing: --key value, --flag.
fn parse_args(args: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(key) = args[i].strip_prefix("--") {
            let next_is_value = args.get(i + 1).map(|v| !v.starts_with("--")).unwrap_or(false);
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

fn opt<'a>(o: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
    o.get(key).map(|s| s.as_str()).filter(|s| !s.is_empty())
}

fn num<T: std::str::FromStr>(o: &HashMap<String, String>, key: &str, def: T) -> Result<T, String> {
    match opt(o, key) {
        Some(s) => s.parse().map_err(|_| format!("bad --{} value: {}", key, s)),
        None => Ok(def),
    }
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 || argv[1] == "--help" || argv[1] == "-h" {
        usage();
        return if argv.len() < 2 { Err("no arguments given".into()) } else { Ok(()) };
    }
    let o = parse_args(&argv[1..]);

    let mode = opt(&o, "mode").ok_or("--mode seq|threads|rayon is required")?.to_string();
    if !matches!(mode.as_str(), "seq" | "threads" | "rayon") {
        return Err(format!("unknown --mode: {}", mode));
    }
    let scale: u32 = num(&o, "scale", 0)?;
    if scale == 0 {
        return Err("--scale <20|22|24> is required".into());
    }
    let source_idx: usize = num(&o, "source", 0)?;
    let threads: usize = num(
        &o,
        "threads",
        std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1),
    )?;
    if threads == 0 {
        return Err("--threads must be at least 1".into());
    }
    let reps: usize = num(&o, "reps", DEFAULT_REPS)?;
    let warmup: usize = num(&o, "warmup", DEFAULT_WARMUP)?;
    let affinity = o.contains_key("affinity");
    let verify = o.contains_key("verify");
    let csv = o.contains_key("csv");

    // The sequential version always has 1 thread - otherwise Sp = T1/Tp would be meaningless.
    let threads = if mode == "seq" { 1 } else { threads };

    let data_dir = io::find_data_dir(opt(&o, "data")).map_err(|e| e.to_string())?;
    let path = data_dir.join(format!("graph_s{}.csr", scale));
    let g = io::read_rcsr(&path).map_err(|e| e.to_string())?;

    if source_idx >= g.sources.len() {
        return Err(format!(
            "--source {} out of range: the file has {} search sources (0..{})",
            source_idx,
            g.sources.len(),
            g.sources.len() - 1
        ));
    }
    let source = g.sources[source_idx];
    let n = g.num_vertices();

    // Two separate distance arrays: a plain one for the sequential version, an
    // atomic one for the parallel ones. A direct consequence of decision 1,
    // not a matter of convenience.
    let mut dist_seq: Vec<u32> = Vec::new();
    let mut dist_par: Vec<AtomicU32> = Vec::new();
    if mode == "seq" {
        dist_seq = vec![UNVISITED; n];
    } else {
        dist_par = (0..n).map(|_| AtomicU32::new(UNVISITED)).collect();
    }

    // The core list is queried once: the system call must not end up in the
    // measured time.
    let core_ids = if affinity { core_affinity::get_core_ids().unwrap_or_default() } else { Vec::new() };

    // The Rayon pool is created up front so that thread start-up does not end
    // up in the measured time. Pinning to cores - via start_handler.
    let pool = if mode == "rayon" {
        let ids = core_ids.clone();
        let builder = rayon::ThreadPoolBuilder::new().num_threads(threads).start_handler(move |idx| {
            if !ids.is_empty() {
                core_affinity::set_for_current(ids[idx % ids.len()]);
            }
        });
        Some(builder.build().map_err(|e| format!("failed to create Rayon pool: {}", e))?)
    } else {
        None
    };

    // Affinity for the sequential version: one thread on the first core.
    if mode == "seq" && affinity {
        if let Some(id) = core_ids.first().copied() {
            core_affinity::set_for_current(id);
        }
    }

    // Filling `dist` stays OUTSIDE the timer - like zeroing C on the matmul side.
    let search = |dist_seq: &mut Vec<u32>, dist_par: &mut Vec<AtomicU32>| {
        if mode == "seq" {
            dist_seq.fill(UNVISITED);
        } else {
            for cell in dist_par.iter() {
                cell.store(UNVISITED, Ordering::Relaxed);
            }
        }
        let t0 = Instant::now();
        let levels = match mode.as_str() {
            "seq" => seq::bfs(&g, source, dist_seq),
            "threads" => threads::bfs(&g, source, dist_par, threads, &core_ids),
            "rayon" => pool.as_ref().unwrap().install(|| rayon_impl::bfs(&g, source, dist_par)),
            _ => unreachable!(),
        };
        (t0.elapsed().as_secs_f64(), levels)
    };

    for _ in 0..warmup {
        search(&mut dist_seq, &mut dist_par);
    }

    let binding = if affinity { "on" } else { "off" };
    let mut times = Vec::with_capacity(reps);
    let mut levels = 0u32;
    let mut dist_sum: Option<u64> = None;
    let mut hash: Option<u64> = None;

    for rep in 1..=reps {
        let (secs, lv) = search(&mut dist_seq, &mut dist_par);
        levels = lv;

        // The distance array is copied into a plain vector OUTSIDE the timer.
        let d = snapshot(&mode, &dist_seq, &dist_par);
        let sum: u64 = d.iter().filter(|&&x| x != UNVISITED).map(|&x| x as u64).sum();
        let h = io::bit_hash_u32(&d);
        match hash {
            None => {
                hash = Some(h);
                dist_sum = Some(sum);
            }
            Some(prev) if prev != h => {
                return Err(format!(
                    "repetitions give different results: bit hash {:016x} != {:016x} \
                     (data race?)",
                    prev, h
                ))
            }
            _ => {}
        }
        times.push(secs);
        if csv {
            println!(
                "rust,bfs,{},{},{},{},{},{:.9},{:016x},{:016x}",
                mode,
                scale,
                threads,
                binding,
                rep,
                secs,
                dist_sum.unwrap(),
                h
            );
        }
    }

    let d = snapshot(&mode, &dist_seq, &dist_par);
    let visited = d.iter().filter(|&&x| x != UNVISITED).count();
    if visited != n {
        return Err(format!(
            "only {} of {} vertices reached - the graph should be connected (largest component)",
            visited, n
        ));
    }

    // Correctness check: distances must match the sequential ones.
    if verify && mode != "seq" {
        let mut reference = vec![UNVISITED; n];
        seq::bfs(&g, source, &mut reference);
        if reference != d {
            let diff = d.iter().zip(&reference).filter(|(x, y)| x != y).count();
            return Err(format!("distances do not match the sequential ones: {} vertices differ", diff));
        }
        eprintln!("check: distances match the sequential ones bit-for-bit");
    }

    if !csv {
        let mut sorted = times.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let min = sorted[0];
        let median = sorted[sorted.len() / 2];
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        // MTEPS (millions of traversed edges per second) - the Graph500 metric.
        // In a connected component every vertex is visited, so all m directed
        // entries are examined.
        let mteps = g.num_entries() as f64 / median / 1e6;
        println!("variant     {} ({} threads, affinity {})", mode, threads, binding);
        println!("graph       scale {}  n = {}  m = {}", scale, n, g.num_entries());
        println!("source      vertex {} (index {})", source, source_idx);
        println!("levels      {}", levels);
        println!("reps        {} (+ {} warm-up)", reps, warmup);
        println!("min         {:.6} s", min);
        println!("median      {:.6} s   ({:.1} MTEPS)", median, mteps);
        println!("mean        {:.6} s", mean);
        println!("dist sum    {}", dist_sum.unwrap());
        println!("bit hash    {:016x}", hash.unwrap());
        println!("peak memory {} KiB", io::peak_rss_kib());
    }

    Ok(())
}

/// Copy of the distance array as plain u32 - so that checks and the hash are
/// computed the same way in every variant.
fn snapshot(mode: &str, dist_seq: &[u32], dist_par: &[AtomicU32]) -> Vec<u32> {
    if mode == "seq" {
        dist_seq.to_vec()
    } else {
        dist_par.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }
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
