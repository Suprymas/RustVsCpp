// Low-level parallelisation: std::thread + std::barrier.
// Rust counterpart - threads.rs.
//
// Unlike matrix multiplication, here the threads are created ONCE for the
// whole search, not for every level. The reason is purely quantitative: the
// R-MAT graph diameter is small (~7 levels), but creating 32 threads costs
// ~0.44 ms, and creating them at every level would turn the measurement into
// a measurement of thread creation. The Rust side does exactly the same, so
// the symmetry between languages is preserved.
//
// AN IMPORTANT DIFFERENCE worth describing in the thesis: ONE barrier per
// level is enough here, because `std::barrier` accepts a completion function
// that runs after all threads have arrived and BEFORE they are released.
// Rust's `std::sync::Barrier` has no such facility - its `wait()` only
// returns `is_leader()`, and the leader works after all threads have already
// been released, so a SECOND barrier is needed after the merge. In this
// respect the C++ standard library is more expressive than the Rust standard
// library.
//
// Second difference: `frontier` here is a plain vector accessed directly by
// all threads. On the Rust side the same spot requires an `RwLock` - not for
// synchronisation (the barrier already provides it), but because safe Rust
// has no way to prove that the read and write phases do not overlap. It is
// not needed here, but correctness rests solely on the author's discipline.

#include "par.hpp"

#include <algorithm>
#include <barrier>
#include <thread>
#include <vector>

#include "../../common/affinity.hpp"

namespace bfs {

std::uint32_t threads_bfs(const io::Csr& g,
                          std::uint32_t source,
                          AtomicDist* dist,
                          std::size_t threads,
                          const std::vector<int>& core_ids) {
    dist[source].store(0, std::memory_order_relaxed);

    std::vector<std::uint32_t> frontier{source};
    std::vector<std::vector<std::uint32_t>> buffers(threads);
    std::atomic<std::size_t> cursor{0};
    std::uint32_t level = 0;
    bool done = false;

    // Completion function: merges the thread buffers into the next level's
    // frontier. Run by one thread once all have arrived at the barrier.
    auto merge = [&]() noexcept {
        frontier.clear();
        for (std::vector<std::uint32_t>& buf : buffers) {
            frontier.insert(frontier.end(), buf.begin(), buf.end());
            buf.clear();
        }
        cursor.store(0, std::memory_order_relaxed);
        ++level;
        done = frontier.empty();
    };
    std::barrier sync(static_cast<std::ptrdiff_t>(threads), merge);

    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            if (!core_ids.empty()) {
                common::affinity::pin_current(core_ids[t % core_ids.size()]);
            }
            // The local buffer lives for the whole search on this thread's
            // stack, so `push_back` does not touch shared memory. The thread's
            // header in the shared `buffers` array is updated only once per
            // level (via `swap`) - otherwise 32 threads would keep bouncing
            // the same cache line around (a vector header is 24 bytes, i.e.
            // two or three of them fit in one line).
            std::vector<std::uint32_t> local;
            for (;;) {
                // Phase 1: scan the current frontier. Chunks are taken
                // dynamically, because vertex degrees are very uneven.
                const std::uint32_t next_level = level + 1;
                for (;;) {
                    const std::size_t start = cursor.fetch_add(CHUNK, std::memory_order_relaxed);
                    if (start >= frontier.size()) {
                        break;
                    }
                    const std::size_t end = std::min(start + CHUNK, frontier.size());
                    scan_chunk(g, frontier.data() + start, end - start, dist, next_level, local);
                }
                // Results are published with a single swap: `local` receives
                // the buffer freed by the merge (empty, but same capacity).
                buffers[t].swap(local);

                // Phase 2: barrier + merge in a single call.
                sync.arrive_and_wait();
                if (done) {
                    return;
                }
            }
        });
    }
    for (std::thread& worker : pool) {
        worker.join();
    }

    return level;
}

}  // namespace bfs
