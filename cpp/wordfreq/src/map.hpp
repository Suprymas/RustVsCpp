// Hash map definition shared by both languages. Rust counterpart - map.rs.
//
// WHY NOT THE DEFAULT HASH: Rust uses SipHash-1-3 by default (resistant to
// deliberately chosen keys), whereas libstdc++ std::hash<std::string> is fast
// but not resistant. Keeping the defaults would measure a difference in
// security policy, not parallelisation. So both languages use THE SAME FNV-1a
// hash.
//
// IMPORTANT DETAIL so that both sides compute THE SAME: Rust's
// `Hash for [u8]` first writes a length prefix (8 native-endian bytes), and
// only then the bytes. Exactly the same is done here - otherwise the C++ side
// would do less hashing work for short words and the comparison would be
// unfair.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>

namespace wordfreq {

inline constexpr std::uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
inline constexpr std::uint64_t FNV_PRIME = 0x100000001b3ULL;

// Transparent (is_transparent) hash: allows lookup by std::string_view
// without allocating a std::string. Without it every word - even one already
// seen - would require a new std::string, and we would be measuring the
// allocator rather than parallelisation. On the Rust side
// `HashMap::get_mut(&[u8])` does the same.
struct Fnv1a {
    using is_transparent = void;

    static std::uint64_t feed(std::uint64_t h, const unsigned char* p, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            h ^= static_cast<std::uint64_t>(p[i]);
            h *= FNV_PRIME;
        }
        return h;
    }

    std::size_t operator()(std::string_view s) const {
        std::uint64_t h = FNV_OFFSET;
        // Length prefix - exactly like Rust `write_length_prefix`.
        std::size_t len = s.size();
        unsigned char lb[sizeof(std::size_t)];
        std::memcpy(lb, &len, sizeof(len));
        h = feed(h, lb, sizeof(lb));
        h = feed(h, reinterpret_cast<const unsigned char*>(s.data()), s.size());
        return static_cast<std::size_t>(h);
    }

    std::size_t operator()(const std::string& s) const {
        return (*this)(std::string_view(s));
    }
};

struct Eq {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const { return a == b; }
    bool operator()(const std::string& a, std::string_view b) const { return a == b; }
    bool operator()(std::string_view a, const std::string& b) const { return a == b; }
    bool operator()(const std::string& a, const std::string& b) const { return a == b; }
};

using Counts = std::unordered_map<std::string, std::uint64_t, Fnv1a, Eq>;

// Empty map with pre-allocated capacity. On the Rust side - map::new_counts.
inline Counts new_counts(std::size_t capacity) {
    Counts m;
    m.reserve(capacity);
    return m;
}

// Merges `src` into `dst`. Used by the partial-tables strategy.
inline void merge_into(Counts& dst, const Counts& src) {
    for (const auto& [word, count] : src) {
        dst[word] += count;
    }
}

}  // namespace wordfreq
