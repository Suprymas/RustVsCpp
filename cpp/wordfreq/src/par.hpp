// Part shared by both parallel levels and both strategies.
// Rust counterpart - par.rs.
//
// Only what does NOT depend on the parallelisation method lives here: word
// splitting and counting one band. This keeps threads.cpp and omp_impl.cpp
// pure parallelisation code (important for the dLOC metric).
//
// The functions are `inline` in a header - so that they are inlined
// independently of LTO, like `#[inline]` on the Rust side.

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

#include "map.hpp"

namespace wordfreq {

inline bool is_sep(char c) { return c == ' ' || c == '\n'; }

// Counts the words of a band into a LOCAL map.
//
// The corpus is normalised: lowercase ASCII letters only, words separated by
// a single space or a line end. A new word is allocated only the first time
// it is seen (heterogeneous lookup), as on the Rust side.
inline void count_chunk(std::string_view data, Counts& out) {
    std::size_t i = 0;
    const std::size_t n = data.size();
    while (i < n) {
        while (i < n && is_sep(data[i])) {
            ++i;
        }
        const std::size_t start = i;
        while (i < n && !is_sep(data[i])) {
            ++i;
        }
        if (i > start) {
            const std::string_view word = data.substr(start, i - start);
            auto it = out.find(word);
            if (it != out.end()) {
                ++it->second;
            } else {
                out.emplace(std::string(word), 1);
            }
        }
    }
}

// Same, but writes straight into the SHARED locked table.
//
// The lock is taken for EVERY word - that is precisely the naive strategy
// whose cost the study measures.
inline void count_chunk_shared(std::string_view data, Counts& shared, std::mutex& lock) {
    std::size_t i = 0;
    const std::size_t n = data.size();
    while (i < n) {
        while (i < n && is_sep(data[i])) {
            ++i;
        }
        const std::size_t start = i;
        while (i < n && !is_sep(data[i])) {
            ++i;
        }
        if (i > start) {
            const std::string_view word = data.substr(start, i - start);
            std::lock_guard<std::mutex> guard(lock);
            auto it = shared.find(word);
            if (it != shared.end()) {
                ++it->second;
            } else {
                shared.emplace(std::string(word), 1);
            }
        }
    }
}

// Splits the corpus into `parts` bands, cutting ONLY at separators.
// Computed OUTSIDE the timer and passed in.
inline std::vector<std::pair<std::size_t, std::size_t>> split_bounds(std::string_view data,
                                                                     std::size_t parts) {
    const std::size_t n = data.size();
    if (parts < 1) {
        parts = 1;
    }
    std::vector<std::pair<std::size_t, std::size_t>> bounds;
    bounds.reserve(parts);
    std::size_t start = 0;
    for (std::size_t p = 0; p < parts; ++p) {
        std::size_t end = (p + 1 == parts) ? n : n * (p + 1) / parts;
        // The boundary is pushed forward to the nearest separator so that a
        // word is not cut in half and counted in both bands as two bogus words.
        while (end < n && !is_sep(data[end])) {
            ++end;
        }
        if (end > start) {
            bounds.emplace_back(start, end);
        }
        start = end;
    }
    return bounds;
}

// Low level: std::thread. Rust counterpart - threads.rs.
Counts threads_shared(std::string_view data,
                      const std::vector<std::pair<std::size_t, std::size_t>>& bounds,
                      std::size_t capacity,
                      const std::vector<int>& core_ids);
Counts threads_partial(std::string_view data,
                       const std::vector<std::pair<std::size_t, std::size_t>>& bounds,
                       std::size_t capacity,
                       const std::vector<int>& core_ids);

// High level: OpenMP. Rust counterpart - rayon_impl.rs.
Counts omp_shared(std::string_view data,
                  const std::vector<std::pair<std::size_t, std::size_t>>& bounds,
                  std::size_t capacity,
                  std::size_t threads);
Counts omp_partial(std::string_view data,
                   const std::vector<std::pair<std::size_t, std::size_t>>& bounds,
                   std::size_t capacity,
                   std::size_t threads);

}  // namespace wordfreq
