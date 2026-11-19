#include "io.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace matmul::io {

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

}  // namespace

Matrix read_rmtx(const std::string& path, std::size_t pad) {
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
    if (std::memcmp(header, "RMTX", 4) != 0) {
        throw std::runtime_error(path + ": bad magic (expected RMTX)");
    }
    const std::uint32_t version = le_u32(header + 4);
    if (version != VERSION) {
        throw std::runtime_error(path + ": unsupported version " + std::to_string(version));
    }

    Matrix m;
    m.rows = static_cast<std::size_t>(le_u64(header + 8));
    m.cols = static_cast<std::size_t>(le_u64(header + 16));

    const std::streamsize expected =
        static_cast<std::streamsize>(HEADER_BYTES + m.rows * m.cols * sizeof(double));
    if (size != expected) {
        throw std::runtime_error(path + ": size " + std::to_string(size) + " bytes, expected " +
                                 std::to_string(expected) + " (" + std::to_string(m.rows) + "x" +
                                 std::to_string(m.cols) + ")");
    }

    m.lda = m.cols + pad;
    m.data.assign(m.rows * m.lda, 0.0);
    // Read row by row, because in memory the row stride is lda, while in the
    // file it is cols. Padding elements stay zero. The Rust side does exactly
    // the same (io.rs `chunks_exact_mut(lda)`).
    const auto row_bytes = static_cast<std::streamsize>(m.cols * sizeof(double));
    for (std::size_t r = 0; r < m.rows; ++r) {
        in.read(reinterpret_cast<char*>(m.data.data() + r * m.lda), row_bytes);
        if (!in) {
            throw std::runtime_error(path + ": failed to read data");
        }
    }
    return m;
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

double checksum(const std::vector<double>& c, std::size_t n, std::size_t lda) {
    double s = 0.0;
    for (std::size_t r = 0; r * lda < c.size(); ++r) {
        for (std::size_t j = 0; j < n; ++j) {
            s += c[r * lda + j];
        }
    }
    return s;
}

std::uint64_t bit_hash(const std::vector<double>& c, std::size_t n, std::size_t lda) {
    constexpr std::uint64_t OFFSET = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t PRIME = 0x100000001b3ULL;
    std::uint64_t h = OFFSET;
    for (std::size_t r = 0; r * lda < c.size(); ++r) {
        for (std::size_t j = 0; j < n; ++j) {
            h ^= to_bits(c[r * lda + j]);
            h *= PRIME;  // uint64 overflow is well-defined in C++ (modulo 2^64)
            // Extra mixing: without it the multiplication would propagate
            // differences in the top bits only into the top bits, and they
            // would quickly be shifted out.
            h ^= h >> 33;
        }
    }
    return h;
}

std::uint64_t to_bits(double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
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

}  // namespace matmul::io
