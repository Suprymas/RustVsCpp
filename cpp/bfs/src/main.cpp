// Breadth-first search (BFS) benchmark program (C++). Mirror of rust/bfs.
//
// Three variants:
//   seq     - sequential version without parallelisation primitives (T1 baseline)
//   threads - low level: std::thread + std::barrier
//   omp     - high level: OpenMP
//
// Usage:
//   bfs --mode seq|threads|omp --scale 20 [--source 0] [--threads 8]
//       [--reps 10] [--warmup 1] [--data <directory>] [--affinity]
//       [--verify] [--csv]
//
// ONLY the search is timed. File reading, memory allocation, filling `dist`
// and OpenMP pool creation stay outside the timer.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// All operator new / operator delete variants are redirected to mimalloc.
// Corresponds to #[global_allocator] on the Rust side. Must be included
// EXACTLY ONCE in the whole program.
#include <mimalloc-new-delete.h>

#include "../../common/affinity.hpp"
#include "io.hpp"
#include "par.hpp"
#include "seq.hpp"

namespace {

constexpr std::size_t DEFAULT_REPS = 10;
constexpr std::size_t DEFAULT_WARMUP = 1;

void usage() {
    std::fprintf(stderr,
                 "bfs - breadth-first search benchmarks (C++)\n"
                 "\n"
                 "Required:\n"
                 "  --mode seq|threads|omp\n"
                 "  --scale <20|22|24>\n"
                 "\n"
                 "Optional:\n"
                 "  --source <idx>    index of the search source in the file, 0..7 (default: 0)\n"
                 "  --threads <num>   number of threads (default: number of cores)\n"
                 "  --reps <num>      measured repetitions (default: %zu)\n"
                 "  --warmup <num>    warm-up runs, not included in results (default: %zu)\n"
                 "  --data <dir>      data directory (default: searches ./data, ../data, ...)\n"
                 "  --affinity        pin threads to cores (main series)\n"
                 "                    in the OpenMP variant pinning is handled by OMP_PLACES/OMP_PROC_BIND\n"
                 "  --verify          compare distances bit-for-bit with the sequential run\n"
                 "  --csv             output CSV lines instead of human-readable text\n",
                 DEFAULT_REPS, DEFAULT_WARMUP);
}

// Simple argument parsing: --key value, --flag.
std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> opts;
    for (int i = 1; i < argc;) {
        const std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) {
            ++i;
            continue;
        }
        const std::string key = arg.substr(2);
        const bool next_is_value =
            (i + 1 < argc) && std::string(argv[i + 1]).rfind("--", 0) != 0;
        if (next_is_value) {
            opts[key] = argv[i + 1];
            i += 2;
        } else {
            opts[key] = "";
            ++i;
        }
    }
    return opts;
}

bool has(const std::map<std::string, std::string>& o, const std::string& key) {
    return o.find(key) != o.end();
}

std::string opt(const std::map<std::string, std::string>& o, const std::string& key) {
    const auto it = o.find(key);
    return it == o.end() ? std::string() : it->second;
}

std::size_t num(const std::map<std::string, std::string>& o, const std::string& key,
                std::size_t def) {
    const std::string v = opt(o, key);
    if (v.empty()) {
        return def;
    }
    try {
        return static_cast<std::size_t>(std::stoull(v));
    } catch (const std::exception&) {
        throw std::runtime_error("bad --" + key + " value: " + v);
    }
}

int run(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        usage();
        return argc < 2 ? 1 : 0;
    }
    const auto o = parse_args(argc, argv);

    const std::string mode = opt(o, "mode");
    if (mode != "seq" && mode != "threads" && mode != "omp") {
        throw std::runtime_error(mode.empty() ? "--mode seq|threads|omp is required"
                                              : "unknown --mode: " + mode);
    }
    const std::size_t scale = num(o, "scale", 0);
    if (scale == 0) {
        throw std::runtime_error("--scale <20|22|24> is required");
    }
    const std::size_t source_idx = num(o, "source", 0);
    std::size_t threads = num(o, "threads", static_cast<std::size_t>(omp_get_num_procs()));
    if (threads == 0) {
        throw std::runtime_error("--threads must be at least 1");
    }
    // The sequential version always has 1 thread - otherwise Sp = T1/Tp would be meaningless.
    if (mode == "seq") {
        threads = 1;
    }
    const std::size_t reps = num(o, "reps", DEFAULT_REPS);
    const std::size_t warmup = num(o, "warmup", DEFAULT_WARMUP);
    const bool affinity = has(o, "affinity");
    const bool verify = has(o, "verify");
    const bool csv = has(o, "csv");

    const std::string data_dir = bfs::io::find_data_dir(opt(o, "data"));
    const bfs::io::Csr g =
        bfs::io::read_rcsr(data_dir + "/graph_s" + std::to_string(scale) + ".csr");

    if (source_idx >= g.sources.size()) {
        throw std::runtime_error("--source " + std::to_string(source_idx) + " out of range: the file has " +
                                 std::to_string(g.sources.size()) + " search sources");
    }
    const std::uint32_t source = g.sources[source_idx];
    const std::size_t n = g.num_vertices();

    // Two separate distance arrays: a plain one for the sequential version, an
    // atomic one for the parallel ones. A direct consequence of decision 1,
    // not a matter of convenience.
    std::vector<std::uint32_t> dist_seq;
    std::vector<bfs::AtomicDist> dist_par;
    if (mode == "seq") {
        dist_seq.assign(n, bfs::UNVISITED);
    } else {
        dist_par = std::vector<bfs::AtomicDist>(n);
    }

    // The core list is queried once: the system call must not end up in the
    // measured time.
    const std::vector<int> core_ids =
        affinity ? common::affinity::core_ids() : std::vector<int>();

    // Affinity for the sequential version: one thread on the first core.
    if (mode == "seq" && !core_ids.empty()) {
        common::affinity::pin_current(core_ids.front());
    }

    // The OpenMP pool is created here, before the timer.
    if (mode == "omp") {
#pragma omp parallel num_threads(static_cast<int>(threads))
        {
        }
    }

    std::uint32_t levels = 0;
    // Filling `dist` stays OUTSIDE the timer - like zeroing C on the matmul side.
    auto search = [&]() {
        if (mode == "seq") {
            std::fill(dist_seq.begin(), dist_seq.end(), bfs::UNVISITED);
        } else {
            for (bfs::AtomicDist& cell : dist_par) {
                cell.store(bfs::UNVISITED, std::memory_order_relaxed);
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (mode == "seq") {
            levels = bfs::seq_bfs(g, source, dist_seq.data());
        } else if (mode == "threads") {
            levels = bfs::threads_bfs(g, source, dist_par.data(), threads, core_ids);
        } else {
            levels = bfs::omp_bfs(g, source, dist_par.data(), threads);
        }
        const std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;
        return dt.count();
    };

    // Copy of the distance array as plain uint32 - so that checks and the
    // hash are computed the same way in every variant.
    auto snapshot = [&]() {
        if (mode == "seq") {
            return dist_seq;
        }
        std::vector<std::uint32_t> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = dist_par[i].load(std::memory_order_relaxed);
        }
        return out;
    };

    for (std::size_t i = 0; i < warmup; ++i) {
        search();
    }

    const char* binding = affinity ? "on" : "off";
    std::vector<double> times;
    times.reserve(reps);
    bool have_hash = false;
    std::uint64_t dist_sum = 0;
    std::uint64_t hash = 0;

    for (std::size_t rep = 1; rep <= reps; ++rep) {
        const double secs = search();

        const std::vector<std::uint32_t> d = snapshot();
        std::uint64_t sum = 0;
        for (const std::uint32_t v : d) {
            if (v != bfs::UNVISITED) {
                sum += v;
            }
        }
        const std::uint64_t h = bfs::io::bit_hash_u32(d);
        if (!have_hash) {
            hash = h;
            dist_sum = sum;
            have_hash = true;
        } else if (hash != h) {
            char msg[192];
            std::snprintf(msg, sizeof(msg),
                          "repetitions give different results: bit hash %016lx != %016lx "
                          "(data race?)",
                          static_cast<unsigned long>(hash), static_cast<unsigned long>(h));
            throw std::runtime_error(msg);
        }
        times.push_back(secs);
        if (csv) {
            std::printf("cpp,bfs,%s,%zu,%zu,%s,%zu,%.9f,%016lx,%016lx\n", mode.c_str(), scale,
                        threads, binding, rep, secs, static_cast<unsigned long>(dist_sum),
                        static_cast<unsigned long>(h));
        }
    }

    const std::vector<std::uint32_t> d = snapshot();
    const std::size_t visited =
        static_cast<std::size_t>(std::count_if(d.begin(), d.end(), [](std::uint32_t v) {
            return v != bfs::UNVISITED;
        }));
    if (visited != n) {
        throw std::runtime_error(
            "only " + std::to_string(visited) + " of " + std::to_string(n) +
            " vertices reached - the graph should be connected (largest component)");
    }

    // Correctness check: distances must match the sequential ones.
    if (verify && mode != "seq") {
        std::vector<std::uint32_t> reference(n, bfs::UNVISITED);
        bfs::seq_bfs(g, source, reference.data());
        std::size_t diff = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (reference[i] != d[i]) {
                ++diff;
            }
        }
        if (diff != 0) {
            throw std::runtime_error("distances do not match the sequential ones: " +
                                     std::to_string(diff) + " vertices differ");
        }
        std::fprintf(stderr, "check: distances match the sequential ones bit-for-bit\n");
    }

    if (!csv) {
        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        const double min = sorted.front();
        const double median = sorted[sorted.size() / 2];
        const double mean =
            std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
        // MTEPS (millions of traversed edges per second) - the Graph500 metric.
        const double mteps = static_cast<double>(g.num_entries()) / median / 1e6;
        std::printf("variant     %s (%zu threads, affinity %s)\n", mode.c_str(), threads, binding);
        std::printf("graph       scale %zu  n = %zu  m = %zu\n", scale, n, g.num_entries());
        std::printf("source      vertex %u (index %zu)\n", source, source_idx);
        std::printf("levels      %u\n", levels);
        std::printf("reps        %zu (+ %zu warm-up)\n", reps, warmup);
        std::printf("min         %.6f s\n", min);
        std::printf("median      %.6f s   (%.1f MTEPS)\n", median, mteps);
        std::printf("mean        %.6f s\n", mean);
        std::printf("dist sum    %lu\n", static_cast<unsigned long>(dist_sum));
        std::printf("bit hash    %016lx\n", static_cast<unsigned long>(hash));
        std::printf("peak memory %lu KiB\n",
                    static_cast<unsigned long>(bfs::io::peak_rss_kib()));
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
