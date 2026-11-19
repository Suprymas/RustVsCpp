// Matrix multiplication benchmark program (C++). Mirror of rust/matmul.
//
// Three variants, all using the same tiled kernel (matmul::band_kernel), so
// that differences come only from the parallelisation method:
//   seq     - sequential version without parallelisation primitives (T1 baseline)
//   threads - low level: std::thread
//   omp     - high level: OpenMP
//
// Usage:
//   matmul --mode seq|threads|omp --n 1024 [--threads 8] [--reps 10]
//          [--warmup 1] [--data <directory>] [--affinity] [--verify] [--csv]
//
// ONLY the multiplication is timed. File reading, memory allocation, zeroing
// of C and OpenMP pool creation stay outside the timer.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <numeric>
#include <string>
#include <vector>

// All operator new / operator delete variants are redirected to mimalloc.
// Corresponds to #[global_allocator] on the Rust side: std::vector allocates
// through it. Must be included EXACTLY ONCE in the whole program.
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
                 "matmul - matrix multiplication benchmarks (C++)\n"
                 "\n"
                 "Required:\n"
                 "  --mode seq|threads|omp\n"
                 "  --n <512|1024|2048>\n"
                 "\n"
                 "Optional:\n"
                 "  --threads <num>   number of threads (default: number of cores)\n"
                 "  --reps <num>      measured repetitions (default: %zu)\n"
                 "  --warmup <num>    warm-up runs, not included in results (default: %zu)\n"
                 "  --data <dir>      data directory (default: searches ./data, ../data, ...)\n"
                 "  --affinity        pin threads to cores (main series)\n"
                 "                    in the OpenMP variant pinning is handled by OMP_PLACES/OMP_PROC_BIND\n"
                 "  --verify          compare result bit-for-bit with the sequential one\n"
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
    const std::size_t n = num(o, "n", 0);
    if (n == 0) {
        throw std::runtime_error("--n <size> is required");
    }
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

    const std::string data_dir = matmul::io::find_data_dir(opt(o, "data"));
    const std::string suffix = std::to_string(n) + ".bin";
    const matmul::io::Matrix ma = matmul::io::read_rmtx(data_dir + "/matrix_A_" + suffix, matmul::PAD);
    const matmul::io::Matrix mb = matmul::io::read_rmtx(data_dir + "/matrix_B_" + suffix, matmul::PAD);
    if (ma.rows != n || ma.cols != n || mb.rows != n || mb.cols != n) {
        throw std::runtime_error("wrong dimensions: A " + std::to_string(ma.rows) + "x" +
                                 std::to_string(ma.cols) + ", B " + std::to_string(mb.rows) + "x" +
                                 std::to_string(mb.cols) + ", expected " + std::to_string(n) + "x" +
                                 std::to_string(n));
    }
    const double* a = ma.data.data();
    const double* b = mb.data.data();
    // The row stride is taken from the reader itself, so that padding is
    // defined in a single place (io::read_rmtx) and cannot diverge.
    const std::size_t lda = ma.lda;
    if (mb.lda != lda) {
        throw std::runtime_error("A and B row strides differ");
    }
    std::vector<double> c(n * lda, 0.0);

    // The core list is queried once: the system call must not end up in the
    // measured time (see the comment in affinity.hpp).
    const std::vector<int> core_ids =
        affinity ? common::affinity::core_ids() : std::vector<int>();

    // Affinity for the sequential version: one thread on the first core.
    if (mode == "seq" && !core_ids.empty()) {
        common::affinity::pin_current(core_ids.front());
    }

    // The OpenMP pool is created here, before the timer, so that thread
    // start-up does not end up in the measured time. Corresponds to the
    // rayon::ThreadPoolBuilder call outside the timer on the Rust side. An
    // empty region is all it takes.
    if (mode == "omp") {
#pragma omp parallel num_threads(static_cast<int>(threads))
        {
        }
    }

    auto multiply = [&](std::vector<double>& out) {
        std::fill(out.begin(), out.end(), 0.0);
        const auto t0 = std::chrono::steady_clock::now();
        if (mode == "seq") {
            matmul::multiply(a, b, out.data(), n, lda);
        } else if (mode == "threads") {
            matmul::threads_multiply(a, b, out.data(), n, lda, threads, core_ids);
        } else {
            matmul::omp_multiply(a, b, out.data(), n, lda, threads);
        }
        const std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;
        return dt.count();
    };

    for (std::size_t i = 0; i < warmup; ++i) {
        multiply(c);
    }

    const char* binding = affinity ? "on" : "off";
    std::vector<double> times;
    times.reserve(reps);
    bool have_sum = false;
    std::uint64_t sum_bits = 0;
    std::uint64_t hash = 0;

    for (std::size_t rep = 1; rep <= reps; ++rep) {
        const double secs = multiply(c);
        const std::uint64_t bits = matmul::io::to_bits(matmul::io::checksum(c, n, lda));
        // The bit hash checks every bit, so it is what catches data races;
        // the sum is not that sensitive (see io.hpp).
        const std::uint64_t h = matmul::io::bit_hash(c, n, lda);
        if (!have_sum) {
            sum_bits = bits;
            hash = h;
            have_sum = true;
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
            std::printf("cpp,matmul,%s,%zu,%zu,%s,%zu,%.9f,%016lx,%016lx\n", mode.c_str(), n,
                        threads, binding, rep, secs, static_cast<unsigned long>(bits),
                        static_cast<unsigned long>(h));
        }
    }

    // Correctness check: the result must match the sequential one bit-for-bit.
    if (verify && mode != "seq") {
        std::vector<double> c_ref(n * lda, 0.0);
        matmul::multiply(a, b, c_ref.data(), n, lda);
        std::size_t diff = 0;
        for (std::size_t i = 0; i < c.size(); ++i) {
            if (matmul::io::to_bits(c[i]) != matmul::io::to_bits(c_ref[i])) {
                ++diff;
            }
        }
        if (diff != 0) {
            throw std::runtime_error("result does not match the sequential one: " +
                                     std::to_string(diff) + " elements differ");
        }
        std::fprintf(stderr, "check: result matches the sequential one bit-for-bit\n");
    }

    if (!csv) {
        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        const double min = sorted.front();
        // On the Rust side the median is sorted[len / 2] (the upper one for an
        // even count). Repeated literally so both languages use the same metric.
        const double median = sorted[sorted.size() / 2];
        const double mean =
            std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
        const double nd = static_cast<double>(n);
        const double gflops = 2.0 * nd * nd * nd / median / 1e9;
        std::printf("variant     %s (%zu threads, affinity %s)\n", mode.c_str(), threads, binding);
        std::printf("size        %zux%zu\n", n, n);
        std::printf("reps        %zu (+ %zu warm-up)\n", reps, warmup);
        std::printf("min         %.6f s\n", min);
        std::printf("median      %.6f s   (%.2f GFLOP/s)\n", median, gflops);
        std::printf("mean        %.6f s\n", mean);
        std::printf("checksum    %016lx\n", static_cast<unsigned long>(sum_bits));
        std::printf("bit hash    %016lx\n", static_cast<unsigned long>(hash));
        std::printf("peak memory %lu KiB\n",
                    static_cast<unsigned long>(matmul::io::peak_rss_kib()));
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
