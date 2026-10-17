#pragma once

#include <cstdint>
#include <string>

namespace ninfer::serve {

struct ServerLogEnvironment {
    int device = 0;
    std::string gpu_name;
    std::string gpu_uuid;
    std::uint64_t total_device_memory_bytes = 0;
    int compute_capability_major            = 0;
    int compute_capability_minor            = 0;
    std::string cuda_compile_version;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
};

ServerLogEnvironment query_server_log_environment(int device);

} // namespace ninfer::serve
