//! Word frequencies - the Rust side.
//!
//! Five variants in a single binary:
//!   seq                     sequential version (T1 baseline)
//!   threads + shared        low level, shared locked table
//!   threads + partial       low level, partial tables with a merge
//!   rayon   + shared        high level, shared locked table
//!   rayon   + partial       high level, partial tables with a merge
//!
//! ONLY counting and merging are timed. File reading, computing band
//! boundaries and Rayon pool creation stay outside the timer - as in the
//! other two tasks.

mod io;
mod map;
mod par;
mod rayon_impl;
mod seq;
mod threads;

use std::collections::HashMap;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const DEFAULT_REPS: usize = 10;
const DEFAULT_WARMUP: usize = 1;
/// Default table capacity. Same value on the C++ side.
const DEFAULT_CAPACITY: usize = 1 << 20;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

fn opt<'a>(o: &'a HashMap<String, String>, k: &str) -> Option<&'a str> {
    o.get(k).map(|s| s.as_str())
}

fn num<T: std::str::FromStr>(o: &HashMap<String, String>, k: &str, d: T) -> Result<T, String> {
    match o.get(k) {
        None => Ok(d),
        Some(v) => v.parse::<T>().map_err(|_| format!("bad --{} value: {}", k, v)),
    }
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut o: HashMap<String, String> = HashMap::new();
    let mut i = 1;
    while i < argv.len() {
        let a = &argv[i];
        if let Some(key) = a.strip_prefix("--") {
            if i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                o.insert(key.to_string(), argv[i + 1].clone());
                i += 2;
            } else {
                o.insert(key.to_string(), String::new());
                i += 1;
            }
        } else {
            return Err(format!("unknown argument: {}", a));
        }
    }
    if o.contains_key("help") {
        eprintln!(
            "usage: wordfreq --mode seq|threads|rayon [--strategy shared|partial]\n\
             \x20      --size 512M|2G|8G [--threads N] [--reps N] [--warmup N]\n\
             \x20      [--capacity N] [--data DIR] [--affinity] [--verify] [--csv] [--top N]"
        );
        return Ok(());
    }

    let mode = opt(&o, "mode").unwrap_or("seq").to_string();
    if !["seq", "threads", "rayon"].contains(&mode.as_str()) {
        return Err(format!("--mode must be seq|threads|rayon, got: {}", mode));
    }
    let strategy = opt(&o, "strategy").unwrap_or("partial").to_string();
    if !["shared", "partial"].contains(&strategy.as_str()) {
        return Err(format!("--strategy must be shared|partial, got: {}", strategy));
    }
    let size = opt(&o, "size").unwrap_or("512M").to_string();

    let threads: usize = num(&o, "threads", num_cpus_fallback())?;
    if threads == 0 {
        return Err("--threads must be at least 1".into());
    }
    // The sequential version always has 1 thread - otherwise Sp = T1/Tp would be meaningless.
    let threads = if mode == "seq" { 1 } else { threads };
    let reps: usize = num(&o, "reps", DEFAULT_REPS)?;
    let warmup: usize = num(&o, "warmup", DEFAULT_WARMUP)?;
    let capacity: usize = num(&o, "capacity", DEFAULT_CAPACITY)?;
    let top = num(&o, "top", 20usize)?;
    let affinity = o.contains_key("affinity");
    let verify = o.contains_key("verify");
    let csv = o.contains_key("csv");

    let data_dir = io::find_data_dir(opt(&o, "data")).map_err(|e| e.to_string())?;
    let path = io::corpus_path(&data_dir, &size).map_err(|e| e.to_string())?;
    let data = io::read_corpus(&path).map_err(|e| e.to_string())?;

    // Band boundaries and the core list - preparation, not measured work.
    let bounds = par::split_bounds(&data, threads);
    let core_ids = if affinity {
        core_affinity::get_core_ids().unwrap_or_default()
    } else {
        Vec::new()
    };

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

    if mode == "seq" && affinity {
        if let Some(id) = core_affinity::get_core_ids().and_then(|v| v.first().copied()) {
            core_affinity::set_for_current(id);
        }
    }

    let count = || -> (f64, map::Counts) {
        let t0 = Instant::now();
        let counts = match (mode.as_str(), strategy.as_str()) {
            ("seq", _) => seq::count(&data, capacity),
            ("threads", "shared") => threads::count_shared(&data, &bounds, capacity, &core_ids),
            ("threads", _) => threads::count_partial(&data, &bounds, capacity, &core_ids),
            ("rayon", "shared") => {
                pool.as_ref().unwrap().install(|| rayon_impl::count_shared(&data, &bounds, capacity))
            }
            _ => pool
                .as_ref()
                .unwrap()
                .install(|| rayon_impl::count_partial(&data, &bounds, capacity)),
        };
        (t0.elapsed().as_secs_f64(), counts)
    };

    for _ in 0..warmup {
        count();
    }

    let binding = if affinity { "on" } else { "off" };
    // Strategy is meaningless for the sequential version - "-" is written.
    let strategy_col = if mode == "seq" { "-" } else { strategy.as_str() };
    // The CSV "variant" column is kept uniform with the other two tasks: the
    // strategy goes into the variant name, so that the harness and
    // plot_series.py stay the same for all three tasks (10 columns, not 11).
    let variant_col = if mode == "seq" { "seq".to_string() } else { format!("{}-{}", mode, strategy) };
    // The size in the CSV is always a number, like matmul (512) and BFS (20).
    let n_mib = size_mib(&size)?;
    let mut times = Vec::with_capacity(reps);
    let mut hash: Option<u64> = None;
    let mut total: u64 = 0;
    let mut distinct: usize = 0;
    let mut last: Option<map::Counts> = None;

    for rep in 1..=reps {
        let (secs, counts) = count();
        let h = io::result_hash(&counts);
        match hash {
            None => {
                hash = Some(h);
                total = io::total_words(&counts);
                distinct = counts.len();
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
                "rust,wordfreq,{},{},{},{},{},{:.9},{:016x},{:016x}",
                variant_col, n_mib, threads, binding, rep, secs, total, h
            );
        }
        last = Some(counts);
    }

    if verify && mode != "seq" {
        let reference = seq::count(&data, capacity);
        let rh = io::result_hash(&reference);
        if Some(rh) != hash {
            return Err(format!(
                "result does not match the sequential one: {:016x} != {:016x}",
                hash.unwrap(),
                rh
            ));
        }
        eprintln!("check: result matches the sequential one bit-for-bit");
    }

    if !csv {
        let mut sorted = times.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let min = sorted[0];
        let median = sorted[sorted.len() / 2];
        let mb_per_s = data.len() as f64 / median / 1e6;
        println!("variant     {} / {} ({} threads, affinity {})", mode, strategy_col, threads, binding);
        println!("corpus      {} ({:.2} GiB)", size, data.len() as f64 / (1 << 30) as f64);
        println!("reps        {} (+ {} warm-up)", reps, warmup);
        println!("min         {:.6} s", min);
        println!("median      {:.6} s   ({:.1} MB/s)", median, mb_per_s);
        println!("words       {}", total);
        println!("distinct    {}", distinct);
        println!("bit hash    {:016x}", hash.unwrap());
        println!("peak memory {} KiB", io::peak_rss_kib());
        if top > 0 {
            if let Some(counts) = &last {
                let items = io::top_n(counts, top);
                let s: Vec<String> = items.iter().map(|(w, c)| format!("{}={}", w, c)).collect();
                println!("top {}      {}", top, s.join(", "));
            }
        }
    }

    Ok(())
}

/// Size label -> mebibytes. "512M" -> 512, "2G" -> 2048, "8G" -> 8192.
fn size_mib(label: &str) -> Result<u64, String> {
    let (num, unit) = label.split_at(label.len() - 1);
    let v: u64 = num.parse().map_err(|_| format!("bad --size value: {}", label))?;
    match unit {
        "M" => Ok(v),
        "G" => Ok(v * 1024),
        _ => Err(format!("bad --size unit: {}", label)),
    }
}

fn num_cpus_fallback() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}
