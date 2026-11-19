//! Matrix multiplication benchmark program (Rust).
//!
//! Three variants, all using the same tiled kernel (`seq::band_kernel`), so
//! that differences come only from the parallelisation method:
//!   seq     - sequential version without parallelisation primitives (T1 baseline)
//!   threads - low level: std::thread::scope
//!   rayon   - high level: Rayon
//!
//! Usage:
//!   matmul --mode seq|threads|rayon --n 1024 [--threads 8] [--reps 10]
//!          [--warmup 1] [--data <directory>] [--affinity] [--verify] [--csv]
//!
//! ONLY the multiplication is timed. File reading, memory allocation, zeroing
//! of C and Rayon pool creation stay outside the timer.

mod io;
mod rayon_impl;
mod seq;
mod threads;

use std::collections::HashMap;
use std::process::ExitCode;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const DEFAULT_REPS: usize = 10;
const DEFAULT_WARMUP: usize = 1;

fn usage() {
    eprintln!(
        "matmul - matrix multiplication benchmarks (Rust)\n\
         \n\
         Required:\n\
         \x20 --mode seq|threads|rayon\n\
         \x20 --n <512|1024|2048>\n\
         \n\
         Optional:\n\
         \x20 --threads <num>   number of threads (default: number of cores)\n\
         \x20 --reps <num>      measured repetitions (default: {})\n\
         \x20 --warmup <num>    warm-up runs, not included in results (default: {})\n\
         \x20 --data <dir>      data directory (default: searches ./data, ../data, ...)\n\
         \x20 --affinity        pin threads to cores (main series)\n\
         \x20 --verify          compare result bit-for-bit with the sequential one\n\
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
    let n: usize = num(&o, "n", 0)?;
    if n == 0 {
        return Err("--n <size> is required".into());
    }
    let threads: usize = num(&o, "threads", std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1))?;
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
    let path_a = data_dir.join(format!("matrix_A_{}.bin", n));
    let path_b = data_dir.join(format!("matrix_B_{}.bin", n));

    let ma = io::read_rmtx(&path_a, seq::PAD).map_err(|e| e.to_string())?;
    let mb = io::read_rmtx(&path_b, seq::PAD).map_err(|e| e.to_string())?;
    if ma.rows != n || ma.cols != n || mb.rows != n || mb.cols != n {
        return Err(format!(
            "wrong dimensions: A {}x{}, B {}x{}, expected {0}x{0}",
            ma.rows, ma.cols, mb.rows, mb.cols
        ));
    }
    // The row stride is taken from the reader itself, so that padding is
    // defined in a single place (io::read_rmtx) and cannot diverge.
    let lda = ma.lda;
    if mb.lda != lda {
        return Err(format!("A and B row strides differ: {} != {}", lda, mb.lda));
    }
    let (a, b) = (ma.data, mb.data);
    let mut c = vec![0.0f64; n * lda];

    // The core list is queried once: the system call must not end up in the
    // measured time (see the comment on threads::multiply).
    let core_ids = if affinity {
        core_affinity::get_core_ids().unwrap_or_default()
    } else {
        Vec::new()
    };

    // The Rayon pool is created up front so that thread start-up does not end
    // up in the measured time. Pinning to cores - via start_handler.
    let pool = if mode == "rayon" {
        let ids = core_ids.clone();
        let builder = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .start_handler(move |idx| {
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
        if let Some(id) = core_affinity::get_core_ids().and_then(|v| v.first().copied()) {
            core_affinity::set_for_current(id);
        }
    }

    let multiply = |c: &mut Vec<f64>| {
        c.fill(0.0);
        let t0 = Instant::now();
        match mode.as_str() {
            "seq" => seq::multiply(&a, &b, c, n, lda),
            "threads" => threads::multiply(&a, &b, c, n, lda, threads, &core_ids),
            "rayon" => pool
                .as_ref()
                .unwrap()
                .install(|| rayon_impl::multiply(&a, &b, c, n, lda, threads)),
            _ => unreachable!(),
        }
        t0.elapsed().as_secs_f64()
    };

    for _ in 0..warmup {
        multiply(&mut c);
    }

    let binding = if affinity { "on" } else { "off" };
    let mut times = Vec::with_capacity(reps);
    let mut sum_bits: Option<u64> = None;
    let mut hash: Option<u64> = None;

    for rep in 1..=reps {
        let secs = multiply(&mut c);
        let bits = io::checksum(&c, n, lda).to_bits();
        // The bit hash checks every bit, so it is what catches data races;
        // the sum is not that sensitive (see io::bit_hash).
        let h = io::bit_hash(&c, n, lda);
        match hash {
            None => hash = Some(h),
            Some(prev) if prev != h => {
                return Err(format!(
                    "repetitions give different results: bit hash {:016x} != {:016x} \
                     (data race?)",
                    prev, h
                ))
            }
            _ => {}
        }
        if sum_bits.is_none() {
            sum_bits = Some(bits);
        }
        times.push(secs);
        if csv {
            println!(
                "rust,matmul,{},{},{},{},{},{:.9},{:016x},{:016x}",
                mode, n, threads, binding, rep, secs, bits, h
            );
        }
    }

    // Correctness check: the result must match the sequential one bit-for-bit.
    if verify && mode != "seq" {
        let mut c_ref = vec![0.0f64; n * lda];
        seq::multiply(&a, &b, &mut c_ref, n, lda);
        if c_ref != c {
            let diff = c.iter().zip(&c_ref).filter(|(x, y)| x != y).count();
            return Err(format!("result does not match the sequential one: {} elements differ", diff));
        }
        eprintln!("check: result matches the sequential one bit-for-bit");
    }

    if !csv {
        let mut sorted = times.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let min = sorted[0];
        let median = sorted[sorted.len() / 2];
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let gflops = 2.0 * (n as f64).powi(3) / median / 1e9;
        println!("variant     {} ({} threads, affinity {})", mode, threads, binding);
        println!("size        {0}x{0}", n);
        println!("reps        {} (+ {} warm-up)", reps, warmup);
        println!("min         {:.6} s", min);
        println!("median      {:.6} s   ({:.2} GFLOP/s)", median, gflops);
        println!("mean        {:.6} s", mean);
        println!("checksum    {:016x}", sum_bits.unwrap());
        println!("bit hash    {:016x}", hash.unwrap());
        println!("peak memory {} KiB", io::peak_rss_kib());
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
