#include "serve/request_log.h"

#include <cuda_runtime.h>

#include <iomanip>
#include <filesystem>
#include <chrono>
#include <sstream>

namespace ninfer::serve {
namespace {
std::string cuda_version_string(int version) {
    if (version <= 0) return {};
    return std::to_string(version / 1000) + '.' + std::to_string((version % 1000) / 10);
}

std::string cuda_uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream out;
    out << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out << '-';
        out << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return out.str();
}
} // namespace

ServerLogEnvironment query_server_log_environment(int device) {
    ServerLogEnvironment environment;
    environment.device = device;
    environment.cuda_compile_version = cuda_version_string(CUDART_VERSION);
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess)
        environment.cuda_runtime_version = cuda_version_string(runtime_version);
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess)
        environment.cuda_driver_version = cuda_version_string(driver_version);
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        environment.gpu_name = properties.name;
        environment.gpu_uuid = cuda_uuid_string(properties.uuid);
        environment.total_device_memory_bytes = properties.totalGlobalMem;
        environment.compute_capability_major = properties.major;
        environment.compute_capability_minor = properties.minor;
    }
    return environment;
}

void JsonlRequestLog::write_server_start(const product::ServeOptions& options,
                                       const EngineOptions& engine_options,
                                       const ModelSamplingDefaults& sampling_defaults,
                                       const std::string& public_model_id,
                                       const EngineDescription& description,
                                       const StartupReport& startup,
                                       const EngineSnapshot& snapshot) {
    if (!enabled()) return;
    std::error_code error;
    const auto size = std::filesystem::file_size(options.launch.engine.artifact_path, error);
    const std::optional<std::uint64_t> artifact_size =
        error ? std::nullopt : std::optional<std::uint64_t>(size);
    const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    append(format_server_start_json(server_instance_id_, timestamp, options, engine_options,
        sampling_defaults, public_model_id, description, startup, snapshot,
        query_server_log_environment(options.launch.engine.device), artifact_size));
}
} // namespace ninfer::serve
