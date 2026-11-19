// High-level parallelisation: OpenMP. Rust counterpart - rayon_impl.rs.
//
// NO pinning is done here: OpenMP has OMP_PLACES / OMP_PROC_BIND for that,
// and they are set by the harness. On the Rayon side the same is done inside
// the program (`start_handler`) - a property of the library interface itself,
// not a flaw in the measurement.
//
// Worth comparing with the Rust side: there the `partial` strategy is a
// single expression (`fold(..).reduce(..)`), which itself creates one table
// per work unit and itself merges them. Here a local table has to be declared
// by hand in a `parallel` region and merged via `critical`. This shows up
// directly in the dLOC metric.

#include "par.hpp"

#include <omp.h>

#include <utility>

namespace wordfreq {

Counts omp_shared(std::string_view data,
                  const std::vector<std::pair<std::size_t, std::size_t>>& bounds,
                  std::size_t capacity,
                  std::size_t threads) {
    Counts shared = new_counts(capacity);
    std::mutex lock;
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(bounds.size());

#pragma omp parallel for schedule(static) num_threads(static_cast<int>(threads))
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const auto [start, end] = bounds[static_cast<std::size_t>(i)];
        count_chunk_shared(data.substr(start, end - start), shared, lock);
    }
    return shared;
}

Counts omp_partial(std::string_view data,
                   const std::vector<std::pair<std::size_t, std::size_t>>& bounds,
                   std::size_t capacity,
                   std::size_t threads) {
    const std::size_t parts = bounds.empty() ? 1 : bounds.size();
    const std::size_t per_thread = capacity / parts;
    Counts result = new_counts(capacity);
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(bounds.size());

#pragma omp parallel num_threads(static_cast<int>(threads))
    {
        Counts local = new_counts(per_thread);
#pragma omp for schedule(static) nowait
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            const auto [start, end] = bounds[static_cast<std::size_t>(i)];
            count_chunk(data.substr(start, end - start), local);
        }
#pragma omp critical
        merge_into(result, local);
    }
    return result;
}

}  // namespace wordfreq
