#include "io.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>

namespace wordfreq::io {

std::string read_corpus(const std::string& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error(path + ": cannot determine size");
    }
    std::string buf;
    buf.resize(static_cast<std::size_t>(size));

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(path + ": cannot open");
    }
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    if (static_cast<std::size_t>(in.gcount()) != buf.size()) {
        throw std::runtime_error(path + ": failed to read all data");
    }
    return buf;
}

std::string corpus_path(const std::string& dir, const std::string& size) {
    namespace fs = std::filesystem;
    const std::string direct = dir + "/corpus_" + size + ".txt";
    if (fs::is_regular_file(direct)) {
        return direct;
    }
    std::vector<std::string> found;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        const std::string name = e.path().filename().string();
        if (name.rfind("corpus_", 0) == 0 && name.size() > 4 &&
            name.compare(name.size() - 4, 4, ".txt") == 0) {
            found.push_back(name);
        }
    }
    std::sort(found.begin(), found.end());
    std::string list;
    for (std::size_t i = 0; i < found.size(); ++i) {
        list += (i ? ", " : "") + found[i];
    }
    throw std::runtime_error(direct + " not found; the directory contains: " +
                             (list.empty() ? "nothing" : list));
}

std::string find_data_dir(const std::string& explicit_path) {
    namespace fs = std::filesystem;
    if (!explicit_path.empty()) {
        if (fs::is_directory(explicit_path)) {
            return explicit_path;
        }
        throw std::runtime_error("specified data directory not found: " + explicit_path);
    }
    for (const char* prefix : {"data", "../data", "../../data", "../../../data"}) {
        if (fs::is_directory(prefix)) {
            return prefix;
        }
    }
    throw std::runtime_error("data directory not found (use --data)");
}

std::uint64_t result_hash(const Counts& counts) {
    std::vector<std::pair<std::string_view, std::uint64_t>> items;
    items.reserve(counts.size());
    for (const auto& [word, count] : counts) {
        items.emplace_back(std::string_view(word), count);
    }
    std::sort(items.begin(), items.end());

    std::uint64_t h = FNV_OFFSET;
    const auto feed = [&h](const unsigned char* p, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            h ^= static_cast<std::uint64_t>(p[i]);
            h *= FNV_PRIME;
            h ^= h >> 33;
        }
    };
    for (const auto& [word, count] : items) {
        feed(reinterpret_cast<const unsigned char*>(word.data()), word.size());
        unsigned char cb[sizeof(std::uint64_t)];
        // On the Rust side `count.to_le_bytes()` - so little-endian is required here.
        for (std::size_t i = 0; i < sizeof(cb); ++i) {
            cb[i] = static_cast<unsigned char>((count >> (8 * i)) & 0xff);
        }
        feed(cb, sizeof(cb));
    }
    return h;
}

std::uint64_t total_words(const Counts& counts) {
    std::uint64_t total = 0;
    for (const auto& [word, count] : counts) {
        (void)word;
        total += count;
    }
    return total;
}

std::vector<std::pair<std::string, std::uint64_t>> top_n(const Counts& counts, std::size_t n) {
    std::vector<std::pair<std::string_view, std::uint64_t>> items;
    items.reserve(counts.size());
    for (const auto& [word, count] : counts) {
        items.emplace_back(std::string_view(word), count);
    }
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });
    std::vector<std::pair<std::string, std::uint64_t>> out;
    const std::size_t take = std::min(n, items.size());
    out.reserve(take);
    for (std::size_t i = 0; i < take; ++i) {
        out.emplace_back(std::string(items[i].first), items[i].second);
    }
    return out;
}

std::uint64_t peak_rss_kib() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            return std::strtoull(line.c_str() + 6, nullptr, 10);
        }
    }
    return 0;
}

}  // namespace wordfreq::io
