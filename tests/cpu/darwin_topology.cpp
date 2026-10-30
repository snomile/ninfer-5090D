#include "core/cpu/system.h"
#include <hwloc.h>
#include <algorithm>
#include <cerrno>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <system_error>

#ifndef __APPLE__
#error "This test-only topology adapter must not be used outside Darwin"
#endif

namespace ninfer {
namespace {
void check(int result, const char* operation) {
    if (result < 0) throw std::system_error(errno, std::generic_category(), operation);
}
[[noreturn]] void affinity_unsupported() {
    throw std::system_error(std::make_error_code(std::errc::operation_not_supported),
                            "test-only Darwin topology does not implement thread affinity");
}
}
struct CpuTopology::Impl {
    hwloc_topology_t topology = nullptr;
    CpuTopologyInfo info;
    ~Impl() { if (topology) hwloc_topology_destroy(topology); }
};
CpuTopology::CpuTopology() : impl_(std::make_unique<Impl>()) {
    static std::once_flag report;
    std::call_once(report, [] {
        std::cerr << "TEST-ONLY Darwin hwloc discovery; affinity test EXCLUDED\n";
    });
    check(hwloc_topology_init(&impl_->topology), "initialize test topology");
    check(hwloc_topology_set_flags(impl_->topology, HWLOC_TOPOLOGY_FLAG_DONT_CHANGE_BINDING),
          "configure test topology");
    check(hwloc_topology_set_io_types_filter(impl_->topology, HWLOC_TYPE_FILTER_KEEP_NONE),
          "disable test topology I/O");
    check(hwloc_topology_load(impl_->topology), "discover test topology");
    const auto allowed = hwloc_topology_get_allowed_cpuset(impl_->topology);
    hwloc_obj_t pu = nullptr;
    while ((pu = hwloc_get_next_obj_by_type(impl_->topology, HWLOC_OBJ_PU, pu))) {
        if (!hwloc_bitmap_isset(allowed, pu->os_index)) continue;
        CpuProcessor processor{.cpu = pu->os_index};
        for (auto parent = pu->parent; parent; parent = parent->parent)
            if (parent->type == HWLOC_OBJ_CORE) processor.core = parent->logical_index;
        impl_->info.processors.push_back(processor);
    }
    if (impl_->info.processors.empty()) throw std::runtime_error("no CPUs in test topology");
}
CpuTopology::~CpuTopology() = default;
const CpuTopologyInfo& CpuTopology::info() const noexcept { return impl_->info; }
void validate_cpu_affinity(const CpuTopologyInfo& topology, std::span<const std::uint32_t> cpus) {
    if (cpus.empty()) throw std::invalid_argument("CPU affinity must not be empty");
    std::set<std::uint32_t> seen;
    for (const auto cpu : cpus) {
        const auto found = std::find_if(topology.processors.begin(), topology.processors.end(),
                                       [=](const auto& p) { return p.cpu == cpu; });
        if (found == topology.processors.end() || !seen.insert(cpu).second)
            throw std::invalid_argument("CPU affinity is unavailable or duplicated");
    }
}
CpuAffinity current_thread_affinity(const CpuTopology&) { affinity_unsupported(); }
void bind_current_thread(const CpuTopology& topology, std::span<const std::uint32_t> cpus) {
    validate_cpu_affinity(topology.info(), cpus);
    affinity_unsupported();
}
} // namespace ninfer
