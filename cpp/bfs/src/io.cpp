#include "io.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace bfs::io {

namespace {

std::uint32_t le_u32(const unsigned char* p) {
    std::uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;  // x86-64 is little-endian, so no conversion is needed
}

std::uint64_t le_u64(const unsigned char* p) {
    std::uint64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

template <typename T>
void read_exact(std::ifstream& in, std::vector<T>& out, std::size_t count, const std::string& path) {
    out.resize(count);
    if (count == 0) {
        return;
    }
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) {
        throw std::runtime_error(path + ": failed to read data");
    }
}

}  // namespace

Csr read_rcsr(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error(path + ": cannot open file");
    }
    const std::streamsize size = in.tellg();
    in.seekg(0);
    if (size < static_cast<std::streamsize>(HEADER_BYTES)) {
        throw std::runtime_error(path + ": file too short");
    }

    unsigned char header[HEADER_BYTES];
    in.read(reinterpret_cast<char*>(header), HEADER_BYTES);
    if (std::memcmp(header, "RCSR", 4) != 0) {
        throw std::runtime_error(path + ": bad magic (expected RCSR)");
    }
    const std::uint32_t version = le_u32(header + 4);
    if (version != VERSION) {
        throw std::runtime_error(path + ": unsupported version " + std::to_string(version));
    }

    const std::size_t n = static_cast<std::size_t>(le_u64(header + 8));
    const std::size_t m = static_cast<std::size_t>(le_u64(header + 16));
    const std::size_t s = static_cast<std::size_t>(le_u64(header + 24));

    const std::streamsize want =
        static_cast<std::streamsize>(HEADER_BYTES + (n + 1) * 8 + m * 4 + s * 4);
    if (size != want) {
        throw std::runtime_error(path + ": size " + std::to_string(size) + " bytes, expected " +
                                 std::to_string(want) + " (n=" + std::to_string(n) +
                                 ", m=" + std::to_string(m) + ", s=" + std::to_string(s) + ")");
    }

    Csr g;
    read_exact(in, g.offsets, n + 1, path);
    read_exact(in, g.neighbors, m, path);
    read_exact(in, g.sources, s, path);

    if (g.offsets[n] != m) {
        throw std::runtime_error(path + ": last offset " + std::to_string(g.offsets[n]) +
                                 " does not match m = " + std::to_string(m));
    }
    return g;
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

std::uint64_t bit_hash_u32(const std::vector<std::uint32_t>& d) {
    constexpr std::uint64_t OFFSET = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t PRIME = 0x100000001b3ULL;
    std::uint64_t h = OFFSET;
    for (const std::uint32_t v : d) {
        h ^= static_cast<std::uint64_t>(v);
        h *= PRIME;  // uint64 overflow is well-defined in C++ (modulo 2^64)
        h ^= h >> 33;
    }
    return h;
}

std::uint64_t peak_rss_kib() {
    std::ifstream status("/proc/self/status");
    if (!status) {
        return 0;
    }
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            std::uint64_t kib = 0;
            if (std::sscanf(line.c_str(), "VmHWM: %" SCNu64, &kib) == 1) {
                return kib;
            }
            return 0;
        }
    }
    return 0;
}

}  // namespace bfs::io
