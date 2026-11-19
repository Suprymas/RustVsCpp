// Reader for the RMTX v1 format and helper measurement functions (C++).
// Rust counterpart - io.rs.
//
// Format (all numbers little-endian):
//   bytes  0..4    magic "RMTX"
//   bytes  4..8    format version (uint32) = 1
//   bytes  8..16   number of rows (uint64)
//   bytes 16..24   number of columns (uint64)
//   bytes 24..     rows*cols double values by rows (row-major)

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace matmul::io {

constexpr std::size_t HEADER_BYTES = 24;
constexpr std::uint32_t VERSION = 1;

struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    // Actual row stride in memory (cols + pad), see matmul::PAD.
    std::size_t lda = 0;
    std::vector<double> data;
};

// Reads an RMTX file. On error throws std::runtime_error with a message,
// just as the Rust side returns Err with the same message.
// `pad` elements are appended to every row; they stay zero and take no part
// in the computation. Padding is done while reading, i.e. OUTSIDE the
// timer.
Matrix read_rmtx(const std::string& path, std::size_t pad);

// Locates the data directory: the given path is checked first, then "data"
// is searched for while walking upwards. Empty `explicit_path` = auto search.
std::string find_data_dir(const std::string& explicit_path);

// Checksum: a plain sum in a fixed order.
//
// IMPORTANT: this sum does NOT prove bit-for-bit equality - see bit_hash.
// `lda` - row stride, `n` - number of meaningful columns. Padding elements
// are NOT included, so the value does not depend on PAD.
double checksum(const std::vector<double>& c, std::size_t n, std::size_t lda);

// Bit hash of the whole matrix (FNV-1a with extra mixing).
//
// Why `checksum` is not enough: at N = 2048 every C element is about 512, so
// the total is about 2.1e9, and the double resolution at that magnitude is
// about 5e-7. Differences in individual elements (~1e-11) fall below that
// resolution. Tested: this program compiled WITHOUT -ffp-contract=off gives
// THE SAME sum for all three sizes, even though vfmadd instructions are
// visible in the object file and the actual elements differ. The sum checks
// about 7 significant digits, not bits.
//
// The implementation is literally the same as io::bit_hash on the Rust side -
// otherwise a comparison between languages would be meaningless.
std::uint64_t bit_hash(const std::vector<double>& c, std::size_t n, std::size_t lda);

// Bit pattern of a double. memcpy, not reinterpret_cast: the latter would
// violate the strict aliasing rule, and with -O3 the result would be undefined.
std::uint64_t to_bits(double v);

// Peak resident memory of the process (VmHWM) in kibibytes.
std::uint64_t peak_rss_kib();

}  // namespace matmul::io
