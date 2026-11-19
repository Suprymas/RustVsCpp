// Corpus reading and helper measurement functions (C++).
// Rust counterpart - io.rs.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "map.hpp"

namespace wordfreq::io {

// Reads the whole corpus into memory.
std::string read_corpus(const std::string& path);

// Finds the corpus file by its size label ("512M", "2G", "8G").
std::string corpus_path(const std::string& dir, const std::string& size);

// Locates the data directory. Empty `explicit_path` = auto search.
std::string find_data_dir(const std::string& explicit_path);

// Bit hash of the result: FNV-1a over the SORTED (word, count) list.
//
// Sorting is required, because hash map iteration order is unspecified and
// differs between languages and between runs. The implementation is literally
// the same as io::result_hash on the Rust side - otherwise a comparison
// between languages would be meaningless.
std::uint64_t result_hash(const Counts& counts);

// Total number of words (sum of all frequencies).
std::uint64_t total_words(const Counts& counts);

// The `n` most frequent words: by frequency descending, then by word ascending.
std::vector<std::pair<std::string, std::uint64_t>> top_n(const Counts& counts, std::size_t n);

// Proceso atminties pikas (VmHWM) kibibaitais.
std::uint64_t peak_rss_kib();

}  // namespace wordfreq::io
