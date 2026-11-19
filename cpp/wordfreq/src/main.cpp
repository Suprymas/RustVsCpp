// Word frequencies - the C++ side. Mirror of rust/wordfreq/src/main.rs:
// same parameters, same CSV line, same timing.
//
// ONLY counting and merging are timed. File reading, computing band
// boundaries and OpenMP pool start-up stay outside the timer.

#include <mimalloc-new-delete.h>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../common/affinity.hpp"
#include "io.hpp"
#include "par.hpp"

namespace {

constexpr std::size_t DEFAULT_REPS = 10;
constexpr std::size_t DEFAULT_WARMUP = 1;
// Default table capacity. Same value on the Rust side.
constexpr std::size_t DEFAULT_CAPACITY = 1u << 20;

std::string opt(const std::map<std::string, std::string>& o, const char* k,
                const std::string& def) {
    const auto it = o.find(k);
    return it == o.end() || it->second.empty() ? def : it->second;
}

std::size_t num(const std::map<std::string, std::string>& o, const char* k, std::size_t def) {
    const auto it = o.find(k);
    if (it == o.end() || it->second.empty()) {
        return def;
    }
    return static_cast<std::size_t>(std::strtoull(it->second.c_str(), nullptr, 10));
}

// Size label -> mebibytes. "512M" -> 512, "2G" -> 2048, "8G" -> 8192.
std::size_t size_mib(const std::string& label) {
    if (label.empty()) {
        throw std::runtime_error("empty --size value");
    }
    const char unit = label.back();
    const std::size_t v = static_cast<std::size_t>(std::strtoull(label.c_str(), nullptr, 10));
    if (unit == 'M') return v;
    if (unit == 'G') return v * 1024;
    throw std::runtime_error("bad --size unit: " + label);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> o;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) != 0) {
                throw std::runtime_error("unknown argument: " + a);
            }
            const std::string key = a.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                o[key] = argv[++i];
            } else {
                o[key] = "";
            }
        }
        if (o.count("help")) {
            std::cerr << "usage: wordfreq --mode seq|threads|omp [--strategy shared|partial]\n"
                         "       --size 512M|2G|8G [--threads N] [--reps N] [--warmup N]\n"
                         "       [--capacity N] [--data DIR] [--affinity] [--verify] [--csv]"
                         " [--top N]\n";
            return 0;
        }

        const std::string mode = opt(o, "mode", "seq");
        if (mode != "seq" && mode != "threads" && mode != "omp") {
            throw std::runtime_error("--mode must be seq|threads|omp, got: " + mode);
        }
        const std::string strategy = opt(o, "strategy", "partial");
        if (strategy != "shared" && strategy != "partial") {
            throw std::runtime_error("--strategy must be shared|partial, got: " + strategy);
        }
        const std::string size = opt(o, "size", "512M");

        std::size_t threads = num(o, "threads", static_cast<std::size_t>(omp_get_max_threads()));
        if (threads == 0) {
            throw std::runtime_error("--threads must be at least 1");
        }
        // The sequential version always has 1 thread.
        if (mode == "seq") {
            threads = 1;
        }
        const std::size_t reps = num(o, "reps", DEFAULT_REPS);
        const std::size_t warmup = num(o, "warmup", DEFAULT_WARMUP);
        const std::size_t capacity = num(o, "capacity", DEFAULT_CAPACITY);
        const std::size_t top = num(o, "top", 20);
        const bool affinity = o.count("affinity") > 0;
        const bool verify = o.count("verify") > 0;
        const bool csv = o.count("csv") > 0;

        const std::string data_dir = wordfreq::io::find_data_dir(opt(o, "data", ""));
        const std::string path = wordfreq::io::corpus_path(data_dir, size);
        const std::string data = wordfreq::io::read_corpus(path);
        const std::string_view view(data);

        // Band boundaries and the core list - preparation, not measured work.
        const auto bounds = wordfreq::split_bounds(view, threads);
        std::vector<int> core_ids;
        if (affinity) {
            core_ids = common::affinity::core_ids();
        }
        if (mode == "seq" && affinity && !core_ids.empty()) {
            common::affinity::pin_current(core_ids[0]);
        }

        const auto count = [&]() -> std::pair<double, wordfreq::Counts> {
            const auto t0 = std::chrono::steady_clock::now();
            wordfreq::Counts counts;
            if (mode == "seq") {
                counts = wordfreq::new_counts(capacity);
                wordfreq::count_chunk(view, counts);
            } else if (mode == "threads" && strategy == "shared") {
                counts = wordfreq::threads_shared(view, bounds, capacity, core_ids);
            } else if (mode == "threads") {
                counts = wordfreq::threads_partial(view, bounds, capacity, core_ids);
            } else if (strategy == "shared") {
                counts = wordfreq::omp_shared(view, bounds, capacity, threads);
            } else {
                counts = wordfreq::omp_partial(view, bounds, capacity, threads);
            }
            const auto t1 = std::chrono::steady_clock::now();
            return {std::chrono::duration<double>(t1 - t0).count(), std::move(counts)};
        };

        for (std::size_t i = 0; i < warmup; ++i) {
            count();
        }

        const char* binding = affinity ? "on" : "off";
        const std::string strategy_col = (mode == "seq") ? "-" : strategy;
        // The CSV "variant" column is kept uniform with the other two tasks.
        const std::string variant_col = (mode == "seq") ? "seq" : (mode + "-" + strategy);
        const std::size_t n_mib = size_mib(size);
        std::vector<double> times;
        times.reserve(reps);
        bool have_hash = false;
        std::uint64_t hash = 0;
        std::uint64_t total = 0;
        std::size_t distinct = 0;
        wordfreq::Counts last;

        for (std::size_t rep = 1; rep <= reps; ++rep) {
            auto [secs, counts] = count();
            const std::uint64_t h = wordfreq::io::result_hash(counts);
            if (!have_hash) {
                have_hash = true;
                hash = h;
                total = wordfreq::io::total_words(counts);
                distinct = counts.size();
            } else if (h != hash) {
                std::fprintf(stderr,
                             "repetitions give different results: bit hash %016lx != %016lx "
                             "(data race?)\n",
                             hash, h);
                return 1;
            }
            times.push_back(secs);
            if (csv) {
                std::printf("cpp,wordfreq,%s,%zu,%zu,%s,%zu,%.9f,%016lx,%016lx\n",
                            variant_col.c_str(), n_mib, threads, binding, rep, secs, total, h);
            }
            last = std::move(counts);
        }

        if (verify && mode != "seq") {
            wordfreq::Counts reference = wordfreq::new_counts(capacity);
            wordfreq::count_chunk(view, reference);
            const std::uint64_t rh = wordfreq::io::result_hash(reference);
            if (rh != hash) {
                std::fprintf(stderr, "result does not match the sequential one: %016lx != %016lx\n", hash,
                             rh);
                return 1;
            }
            std::fprintf(stderr, "check: result matches the sequential one bit-for-bit\n");
        }

        if (!csv) {
            std::vector<double> sorted = times;
            std::sort(sorted.begin(), sorted.end());
            const double mn = sorted.front();
            const double median = sorted[sorted.size() / 2];
            const double mb_per_s = static_cast<double>(data.size()) / median / 1e6;
            std::printf("variant     %s / %s (%zu threads, affinity %s)\n", mode.c_str(),
                        strategy_col.c_str(), threads, binding);
            std::printf("corpus      %s (%.2f GiB)\n", size.c_str(),
                        static_cast<double>(data.size()) / (1024.0 * 1024.0 * 1024.0));
            std::printf("reps        %zu (+ %zu warm-up)\n", reps, warmup);
            std::printf("min         %.6f s\n", mn);
            std::printf("median      %.6f s   (%.1f MB/s)\n", median, mb_per_s);
            std::printf("words       %lu\n", total);
            std::printf("distinct    %zu\n", distinct);
            std::printf("bit hash    %016lx\n", hash);
            std::printf("peak memory %lu KiB\n", wordfreq::io::peak_rss_kib());
            if (top > 0) {
                const auto items = wordfreq::io::top_n(last, top);
                std::printf("top %zu      ", top);
                for (std::size_t i = 0; i < items.size(); ++i) {
                    std::printf("%s%s=%lu", i ? ", " : "", items[i].first.c_str(),
                                items[i].second);
                }
                std::printf("\n");
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
