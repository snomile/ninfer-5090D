#include "serve/observations.h"

#include <chrono>
#include <iostream>
#include <stdexcept>

using namespace ninfer;
using namespace ninfer::serve;
using Json = nlohmann::json;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
Json next(const std::shared_ptr<MonitorEvents::Subscription>& subscription) {
    const auto event = subscription->next(std::chrono::milliseconds(10));
    require(event.has_value(), "missing request event");
    const auto begin = event->find("data: ") + 6;
    return Json::parse(event->substr(begin));
}
}

int main() {
    try {
        ServiceObservations observations;
        auto subscription = observations.events.subscribe();
        RequestLogMetadata metadata{.http_request_id = "proxy-request-123", .model = "qwen"};
        observations.preparing(1, "openai", metadata);
        const auto prepare = next(subscription);
        require(prepare["details"]["http_request_id"] == metadata.http_request_id,
                "preparation must correlate to proxy request");
        RequestLogContext context{.engine_request_id = 7, .id = 1,
                                  .http_request_id = metadata.http_request_id, .model = "qwen"};
        observations.prepared(context);
        next(subscription);
        const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        GenerationSchedulingObservation boundary{
            .transition = GenerationSchedulingTransition::PrefillStarted,
            .engine_request_id = 7,
            .steady_ns = static_cast<std::uint64_t>(now_ns - 2000000000),
            .elapsed_ns = 1000000000,
            .reused_prompt_tokens = 128,
            .host_to_device_bytes = 1048576,
        };
        observations.scheduling(1, boundary);
        const auto prefill = next(subscription);
        require(prefill["kind"] == "prefill_started", "normal prefill boundary missing");
        require(prefill["details"]["cached_tokens"] == 128, "cached tokens lost");
        require(prefill["details"]["host_to_device_bytes"] == 1048576, "owned transfer lost");
        const auto unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const auto lag = unix_ms - prefill["timestamp_unix_ms"].get<std::int64_t>();
        require(lag >= 1900 && lag <= 2200, "worker boundary replaced by consumer clock");
        GenerationOutcome outcome;
        outcome.prompt_tokens = 1024;
        outcome.completion_tokens = 10;
        outcome.stats.computed_prefill_tokens = 896;
        outcome.stats.reused_prompt_tokens = 128;
        outcome.stats.timings.prefill_seconds = 0.5;
        outcome.stats.timings.control_seconds = 0.02;
        observations.done(1, outcome);
        const auto done = next(subscription);
        require(done["details"]["http_request_id"] == metadata.http_request_id,
                "terminal event lost request identity after removal");
        require(done["details"]["prefill_execution_seconds"] == 0.5,
                "native execution timing missing");
        require(done["details"]["control_execution_seconds"] == 0.02,
                "control exposure missing");
        require(done["sequence"] == 4, "event sequence not monotonic");
        require(done.dump().find("generated_token_ids") == std::string::npos,
                "token contents leaked to telemetry");
        require(observations.snapshot(true).requests.empty(), "completed request stayed active");
        subscription->close();
        require(!observations.events.has_subscribers(), "closed subscriber still requested serialization");
        observations.preparing(2, "openai", metadata);
        observations.first_token(2, {.prepare_seconds = 0.1, .elapsed_since_submit_seconds = 0.2});
        observations.done(2, outcome);
        require(observations.snapshot().counters.completed == 2,
                "disabled live events stopped lifecycle counters");
        auto replacement = observations.events.subscribe();
        require(replacement && !replacement->closed(), "closed subscriber slot was not reusable");
        observations.preparing(3, "openai", metadata);
        require(next(replacement)["details"]["http_request_id"] == metadata.http_request_id,
                "resubscribing lost the next event");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
