#pragma once

// Optional full-precision JSONL measurement log. Human operational rendering is owned separately
// by operational_log.* and never consumes these serialized records.

#include "serve/request_events.h"
#include "product/config/options.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace spdlog {
class logger;
}

namespace ninfer::serve {

inline constexpr int kRequestLogSchemaVersion        = 31;
inline constexpr const char* kRequestLogArtifactType = "ninfer_serve_request_log";

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

// Pure JSON formatters are public to repository tests. Each return value is one complete JSON
// object without a trailing newline.
std::string format_startup_config_json(const std::string& server_instance_id,
                                       std::uint64_t timestamp_unix_ms,
                                       const product::ServeOptions& options);
std::string format_server_start_json(
    const std::string& server_instance_id, std::uint64_t timestamp_unix_ms,
    const product::ServeOptions& options, const ninfer::EngineOptions& engine_options,
    const ninfer::ModelSamplingDefaults& sampling_defaults, const std::string& public_model_id,
    const EngineDescription& description, const StartupReport& startup,
    const EngineSnapshot& snapshot, const ServerLogEnvironment& environment,
    std::optional<std::uint64_t> artifact_size_bytes);
std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context);
std::string
format_request_scheduling_json(const std::string& server_instance_id,
                               std::uint64_t timestamp_unix_ms, std::uint64_t request_id,
                               const std::string& http_request_id,
                               const ninfer::GenerationSchedulingObservation& observation);
std::string format_request_rejected_json(const std::string& server_instance_id,
                                         std::uint64_t timestamp_unix_ms,
                                         const RequestRejectionLogContext& context);
std::string format_request_done_json(const std::string& server_instance_id,
                                     std::uint64_t timestamp_unix_ms,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome);
std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context, const std::string& message);
std::string format_throughput_json(const std::string& server_instance_id,
                                   std::uint64_t timestamp_unix_ms, const ThroughputReport& report);

ServerLogEnvironment query_server_log_environment(int device);

struct RequestLogStatistics {
    std::uint64_t written_records = 0;
    std::uint64_t dropped_records = 0;
    std::uint64_t write_errors = 0;
    std::size_t queued_records = 0;
    std::size_t queued_bytes = 0;
};

// Opens in append mode so one campaign file can contain multiple independently started MTP/model
// blocks. Every line carries server_instance_id because request ids restart at one per process.
// Producers only enqueue bounded owned records; one worker writes and flushes every 250ms or
// 256KiB. Saturation drops new records, reports cumulative loss, and never waits for disk.
// Destruction drains accepted records and flushes; abrupt process termination can lose the tail.
class JsonlRequestLog {
public:
    static constexpr std::size_t kMaxQueuedRecords = 4096;
    static constexpr std::size_t kMaxQueuedBytes = 8 * 1024 * 1024;

    explicit JsonlRequestLog(std::string server_instance_id, const std::string& path,
                             const std::string& protected_artifact_path = {},
                             std::shared_ptr<spdlog::logger> logger     = {});
    ~JsonlRequestLog();

    JsonlRequestLog(const JsonlRequestLog&)            = delete;
    JsonlRequestLog& operator=(const JsonlRequestLog&) = delete;

    [[nodiscard]] bool enabled() const noexcept;
    [[nodiscard]] RequestLogStatistics statistics() const;

    [[nodiscard]] const std::string& server_instance_id() const noexcept {
        return server_instance_id_;
    }

    void write_startup_config(const product::ServeOptions& options);
    void write_server_start(const product::ServeOptions& options,
                            const ninfer::EngineOptions& engine_options,
                            const ninfer::ModelSamplingDefaults& sampling_defaults,
                            const std::string& public_model_id,
                            const EngineDescription& description, const StartupReport& startup,
                            const EngineSnapshot& snapshot);
    void write_request_start(const RequestLogContext& context);
    void write_request_scheduling(std::uint64_t request_id, const std::string& http_request_id,
                                  const ninfer::GenerationSchedulingObservation& observation);
    void write_request_rejected(const RequestRejectionLogContext& context);
    void write_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void write_request_error(const RequestLogContext& context, const std::string& message);
    void write_throughput(const ThroughputReport& report);

private:
    void append(std::string record);

    struct Impl;
    std::string server_instance_id_;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::serve
