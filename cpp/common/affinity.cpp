#include "affinity.hpp"

#include <pthread.h>
#include <sched.h>

namespace common::affinity {

std::vector<int> core_ids() {
    cpu_set_t set;
    CPU_ZERO(&set);
    std::vector<int> ids;
    // Take the cores ALLOWED for the process, not all cores of the system:
    // when started via `taskset` the measurement must stay within that subset.
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return ids;
    }
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &set)) {
            ids.push_back(cpu);
        }
    }
    return ids;
}

void pin_current(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

}  // namespace common::affinity
