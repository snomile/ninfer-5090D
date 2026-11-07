#include "serve/request_log.h"
#include "product/logging/logging.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::serve;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Predicate>
void eventually(Predicate predicate, const char* message) {
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        require(Clock::now() < deadline, message);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

std::filesystem::path path(const char* name) {
    return std::filesystem::temp_directory_path() /
           ("ninfer-async-" + std::to_string(::getpid()) + "-" + name);
}

// The writer uses a real blocking FIFO. No reader consumes data until producers finish.
class SlowOutput {
public:
    explicit SlowOutput(const char* name) : file(path(name)) {
        require(::mkfifo(file.c_str(), 0600) == 0, "mkfifo failed");
        fd = ::open(file.c_str(), O_RDWR | O_NONBLOCK);
        require(fd >= 0, "FIFO reader open failed");
    }
    ~SlowOutput() {
        finished = true;
        if (reader.joinable()) reader.join();
        ::close(fd);
        std::filesystem::remove(file);
    }
    void drain() {
        reader = std::thread([this] {
            char buffer[32768];
            for (;;) {
                const auto count = ::read(fd, buffer, sizeof(buffer));
                if (count > 0) data.append(buffer, count);
                else if (finished.load()) break;
                else std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }
    void finish() {
        finished = true;
        reader.join();
    }
    std::filesystem::path file;
    int fd = -1;
    std::string data;
private:
    std::atomic<bool> finished{false};
    std::thread reader;
};

void ordering_and_shutdown() {
    RequestLogContext required{.id = 1};
    required.tool_choice.mode = ToolChoiceMode::Required;
    require(Json::parse(format_request_start_json("required", 0, required))
                ["request"]["tool_choice"] == "required", "required tool choice logged as unknown");
    const auto file = path("ordered.jsonl");
    {
        JsonlRequestLog log("ordered", file.string());
        std::vector<std::thread> producers;
        for (unsigned p = 0; p < 4; ++p) producers.emplace_back([&, p] {
            for (unsigned n = 0; n < 100; ++n) {
                GenerationSchedulingObservation value{
                    .transition = GenerationSchedulingTransition::Admitted,
                    .engine_request_id = p * 100 + n,
                };
                const auto identity = std::to_string(p);
                log.write_request_scheduling(value.engine_request_id, identity, value);
            }
        });
        for (auto& thread : producers) thread.join();
        require(log.statistics().dropped_records == 0, "ordinary logging dropped records");
    }
    std::ifstream input(file);
    unsigned count = 0, per_producer[4]{};
    for (std::string line; std::getline(input, line);) {
        const auto record = Json::parse(line);
        const auto request = record["request"]["request_id"].get<unsigned>();
        const unsigned p = request / 100;
        require(request % 100 == per_producer[p]++, "per-producer order changed");
        require(record["request"]["http_request_id"] == std::to_string(p), "deferred identity was borrowed");
        ++count;
    }
    require(count == 400, "shutdown did not drain every accepted record");
    std::filesystem::remove(file);
}

void saturated_queue(bool byte_limit) {
    SlowOutput sink(byte_limit ? "bytes.fifo" : "count.fifo");
    std::uint64_t dropped = 0;
    const unsigned attempts = byte_limit ? 256 : 16384;
    {
        JsonlRequestLog log("saturation", sink.file.string());
        RequestLogContext context{.id = 1};
        const std::string message(byte_limit ? 96 * 1024 : 0, 'x');
        const auto start = Clock::now();
        for (unsigned n = 0; n < attempts; ++n) {
            if (byte_limit) log.write_request_error(context, message);
            else log.write_request_scheduling(n, "proxy", {
                .transition = GenerationSchedulingTransition::FirstToken,
                .engine_request_id = n,
            });
        }
        const auto milliseconds = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        const auto stats = log.statistics();
        dropped = stats.dropped_records;
        require(dropped > 0, "slow disk did not saturate the bounded queue");
        require(stats.queued_records <= JsonlRequestLog::kMaxQueuedRecords, "count limit exceeded");
        require(stats.queued_bytes <= JsonlRequestLog::kMaxQueuedBytes, "byte limit exceeded");
        if (byte_limit) require(stats.queued_records < 256, "byte bound not exercised");
        require(milliseconds < 5000, "producer waited for the disk consumer");
        std::cout << (byte_limit ? "byte-bound" : "count-bound") << " enqueue_ms=" << milliseconds
                  << " attempts=" << attempts << " dropped=" << dropped << '\n';
        sink.drain();
    }
    sink.finish();
    std::istringstream input(sink.data);
    unsigned accepted = 0;
    std::uint64_t reported = 0;
    for (std::string line; std::getline(input, line);) {
        const auto record = Json::parse(line);
        if (record["event"] == "request_log_status")
            reported = record["dropped_records"].get<std::uint64_t>();
        else ++accepted;
    }
    require(accepted + dropped == attempts, "accepted log tail lost during drain");
    require(reported == dropped, "queue loss was not recorded in JSONL");
}

void write_failure() {
    const auto file = path("failure.jsonl");
    const auto child = ::fork();
    require(child >= 0, "fork failed");
    if (child == 0) {
        ::signal(SIGXFSZ, SIG_IGN);
        const rlimit limit{0, 0};
        if (::setrlimit(RLIMIT_FSIZE, &limit) != 0) ::_exit(2);
        try {
            JsonlRequestLog log("failure", file.string());
            log.write_request_start({.id = 1});
            eventually([&] { return !log.enabled(); }, "disk failure did not disable logging");
            const auto stats = log.statistics();
            require(stats.write_errors == 1, "disk error not exposed");
            require(stats.queued_bytes == 0 && stats.queued_records == 0, "failed queue retained memory");
        } catch (...) { ::_exit(3); }
        ::_exit(0);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "write failure handling failed");
    std::filesystem::remove(file);
}

void operational_stderr() {
    SlowOutput sink("stderr.fifo");
    const int saved = ::dup(STDERR_FILENO);
    const int output = ::open(sink.file.c_str(), O_WRONLY);
    require(saved >= 0 && output >= 0 && ::dup2(output, STDERR_FILENO) >= 0, "stderr redirect failed");
    ::close(output);
    {
        product::LoggingRuntime runtime({.logger_name = "async-test",
            .color = product::LogColorMode::Never, .presentation = product::LogPresentation::Service});
        for (unsigned n = 0; n < 8192; ++n) runtime.logger()->info("record {} | {}", n, std::string(256, 'x'));
        require(runtime.dropped_records() > 0, "operational logger did not bound a slow stderr sink");
        sink.drain();
    }
    ::dup2(saved, STDERR_FILENO);
    ::close(saved);
    sink.finish();
    require(sink.data.find("operational log dropped") != std::string::npos,
            "operational overflow was silent");
    std::istringstream lines(sink.data);
    unsigned accepted = 0;
    unsigned previous = 0;
    for (std::string line; std::getline(lines, line);) {
        const auto marker = line.find("record ");
        if (marker != std::string::npos) {
            const auto current = std::stoul(line.substr(marker + 7));
            require(!accepted || current > previous, "operational log order changed");
            previous = current;
            ++accepted;
        }
    }
    require(accepted > 0, "operational shutdown discarded every record");
}
} // namespace

int main() {
    try {
        ordering_and_shutdown();
        saturated_queue(false);
        saturated_queue(true);
        write_failure();
        operational_stderr();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
