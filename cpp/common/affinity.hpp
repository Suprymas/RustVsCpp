// Pinning threads to cores (C++).
//
// On the Rust side this is done by the `core_affinity` crate, i.e. no code
// written by the author is needed (0 lines). The C++ standard library has no
// counterpart, so the POSIX calls are written by hand - the size of this file
// is a real cost of the C++ side and must be counted as such in the dLOC
// metric.
//
// Placed in cpp/common on purpose: the same function is needed by both the
// matmul and the bfs tasks, while threads.cpp has to stay pure
// parallelisation code so that it is comparable with threads.rs on the Rust
// side.

#pragma once

#include <cstddef>
#include <vector>

namespace common::affinity {

// Numbers of the available cores in ascending order.
//
// Corresponds to `core_affinity::get_core_ids()`. Queried ONCE before the
// timer: a system call inside the measured time would be a setup cost that
// the high-level variant does not pay (the same flaw was fixed on the Rust
// side too).
std::vector<int> core_ids();

// Pins the current thread to the given core.
// Corresponds to `core_affinity::set_for_current()`.
void pin_current(int core);

}  // namespace common::affinity
