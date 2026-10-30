#pragma once

#include "core/memory.h"
#include <cstddef>
#include <cstdint>

namespace ninfer {

// Raw observations. available_bytes() is the platform's allocation-planning estimate.
struct DeviceMemorySnapshot {
    std::size_t cuda_free_bytes = 0, total_bytes = 0;
    std::size_t system_available_bytes = 0, system_total_bytes = 0;
    bool shared_physical_memory = false;

    [[nodiscard]] std::size_t available_bytes() const noexcept {
        if (!shared_physical_memory) return cuda_free_bytes;
        return system_available_bytes < total_bytes ? system_available_bytes : total_bytes;
    }
};

[[nodiscard]] DeviceMemorySnapshot device_memory_snapshot(Device);
[[nodiscard]] inline std::int64_t device_memory_growth(DeviceMemorySnapshot before,
                                                       DeviceMemorySnapshot after) {
    return static_cast<std::int64_t>(before.available_bytes()) -
           static_cast<std::int64_t>(after.available_bytes());
}

} // namespace ninfer
