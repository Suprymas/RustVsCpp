// Low-level parallelisation: std::thread without any library.
// Rust counterpart - threads.rs.
//
// Both strategies live here, because they differ only in where the result is
// accumulated. The merge IS part of the algorithm, so it is included in the
// measured time.
//
// A DIFFERENCE worth describing in the thesis: in `count_partial` every
// thread writes into its own table, and correctness rests on the index `t`
// being unique. On the C++ side this is the author's discipline - no language
// feature checks that two threads will not get the same index. On the Rust
// side the same split is expressed with `iter_mut()`, and the compiler checks
// that every thread gets a separate `&mut Counts`. A mistake here on the C++
// side would be a data race without any resistance from the compiler.

#include "par.hpp"

#include <mutex>
#include <thread>

#include "../../common/affinity.hpp"

namespace wordfreq {

Counts threads_shared(std::string_view data,
                      const std::vector<std::pair<std::size_t, std::size_t>>& bounds,
                      std::size_t capacity,
                      const std::vector<int>& core_ids) {
    Counts shared = new_counts(capacity);
    std::mutex lock;

    std::vector<std::thread> pool;
    pool.reserve(bounds.size());
    for (std::size_t t = 0; t < bounds.size(); ++t) {
        pool.emplace_back([&, t] {
            if (!core_ids.empty()) {
                common::affinity::pin_current(core_ids[t % core_ids.size()]);
            }
            const auto [start, end] = bounds[t];
            count_chunk_shared(data.substr(start, end - start), shared, lock);
        });
    }
    for (std::thread& worker : pool) {
        worker.join();
    }
    return shared;
}

Counts threads_partial(std::string_view data,
                       const std::vector<std::pair<std::size_t, std::size_t>>& bounds,
                       std::size_t capacity,
                       const std::vector<int>& core_ids) {
    const std::size_t parts = bounds.empty() ? 1 : bounds.size();
    const std::size_t per_thread = capacity / parts;

    std::vector<Counts> locals;
    locals.reserve(parts);
    for (std::size_t i = 0; i < bounds.size(); ++i) {
        locals.push_back(new_counts(per_thread));
    }

    std::vector<std::thread> pool;
    pool.reserve(bounds.size());
    for (std::size_t t = 0; t < bounds.size(); ++t) {
        pool.emplace_back([&, t] {
            if (!core_ids.empty()) {
                common::affinity::pin_current(core_ids[t % core_ids.size()]);
            }
            const auto [start, end] = bounds[t];
            count_chunk(data.substr(start, end - start), locals[t]);
        });
    }
    for (std::thread& worker : pool) {
        worker.join();
    }

    // Merge: the first table becomes the result, the others are merged into it.
    if (locals.empty()) {
        return new_counts(capacity);
    }
    Counts result = std::move(locals[0]);
    for (std::size_t i = 1; i < locals.size(); ++i) {
        merge_into(result, locals[i]);
    }
    return result;
}

}  // namespace wordfreq
