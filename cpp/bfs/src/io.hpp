// Reader for the RCSR v1 format and helper measurement functions (C++).
// Rust counterpart - io.rs.
//
// Format (all numbers little-endian):
//   bytes  0..4    magic "RCSR"
//   bytes  4..8    format version (uint32) = 1
//   bytes  8..16   number of vertices n (uint64)
//   bytes 16..24   number of directed entries m (uint64)
//   bytes 24..32   number of search sources s (uint64)
//   then           offsets array (n+1) x uint64
//   then           neighbours array m x uint32
//   then           search source vertices s x uint32

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bfs::io {

constexpr std::size_t HEADER_BYTES = 32;
constexpr std::uint32_t VERSION = 1;

// Graph in compressed sparse row (CSR) form.
struct Csr {
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint32_t> neighbors;
    std::vector<std::uint32_t> sources;

    std::size_t num_vertices() const { return offsets.size() - 1; }
    std::size_t num_entries() const { return neighbors.size(); }

    const std::uint32_t* neighbors_begin(std::size_t v) const {
        return neighbors.data() + offsets[v];
    }
    std::size_t degree(std::size_t v) const {
        return static_cast<std::size_t>(offsets[v + 1] - offsets[v]);
    }
};

// Reads an RCSR file. On error throws std::runtime_error.
//
// Read straight into the vectors' memory (`read(reinterpret_cast<char*>(...))`),
// because x86-64 is little-endian and alignment is guaranteed. On the Rust
// side this is impossible without `unsafe`, so there the same work is done in
// chunks via `chunks_exact` - extra lines with no run-time benefit at all.
// Peak memory is the same on both sides: the whole file is never read into
// memory at once.
Csr read_rcsr(const std::string& path);

// Locates the data directory: the given path is checked first, then "data"
// is searched for while walking upwards. Empty `explicit_path` = auto search.
std::string find_data_dir(const std::string& explicit_path);

// Bit hash of the distance array (FNV-1a with extra mixing).
//
// The implementation is literally the same as io::bit_hash_u32 on the Rust
// side. This hash, not the sum of distances, is the proof of bit-for-bit
// equality.
std::uint64_t bit_hash_u32(const std::vector<std::uint32_t>& d);

// Peak resident memory of the process (VmHWM) in kibibytes.
std::uint64_t peak_rss_kib();

}  // namespace bfs::io
