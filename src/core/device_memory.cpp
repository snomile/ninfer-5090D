#include "core/device.h"

namespace ninfer {

std::int64_t device_memory_growth(DeviceMemorySnapshot before, DeviceMemorySnapshot after) {
    return static_cast<std::int64_t>(before.free_bytes) -
           static_cast<std::int64_t>(after.free_bytes);
}

} // namespace ninfer
