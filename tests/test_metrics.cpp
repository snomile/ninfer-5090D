#include "serve/metrics.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::serve;

namespace {

double sample(const std::string& text, const std::string& name) {
    std::istringstream lines(text);
    lines.imbue(std::locale::classic());
    std::string key;
    std::string line;
    while (std::getline(lines, line)) {
        std::istringstream fields(line);
        fields.imbue(std::locale::classic());
        double value = 0;
        if (fields >> key >> value && key == name) { return value; }
    }
    throw std::runtime_error("missing metric sample: " + name);
}

int test_active_seconds() {
    RuntimeStats baseline;
    baseline.host_work = {.engine_boundary_ns      = 1'000'000'000,
                          .program_submit_ns       = 2'000'000'000,
                          .program_post_ns         = 3'000'000'000,
                          .engine_commit_output_ns = 4'000'000'000,
                          .engine_maintenance_ns   = 5'000'000'000,
                          .device_wait_ns          = 6'000'000'000,
                          .decode_host_ns          = 7'000'000'000,
                          .decode_device_wait_ns   = 8'000'000'000,
                          .prefill_host_ns         = 9'000'000'000,
                          .prefill_device_wait_ns  = 10'000'000'000,
                          .replay_host_ns          = 11'000'000'000,
                          .replay_device_wait_ns   = 12'000'000'000,
                          .control_host_ns         = 13'000'000'000,
                          .control_device_wait_ns  = 14'000'000'000};
    EngineOptions options;
    options.context_cache.device_state_slots = 1;
    Metrics metrics;
    metrics.configure("active-seconds", options, {}, baseline);
    struct Scenario {
        const char* name;
        std::uint64_t RuntimeHostWorkStats::* field;
        double prefill;
        double decode;
    };
    const Scenario scenarios[] = {
        {"prefill host", &RuntimeHostWorkStats::prefill_host_ns, 1, 0},
        {"prefill wait", &RuntimeHostWorkStats::prefill_device_wait_ns, 1, 0},
        {"decode host", &RuntimeHostWorkStats::decode_host_ns, 0, 1},
        {"decode wait", &RuntimeHostWorkStats::decode_device_wait_ns, 0, 1},
        {"control host", &RuntimeHostWorkStats::control_host_ns, 1, 1},
        {"control wait", &RuntimeHostWorkStats::control_device_wait_ns, 1, 1},
        {"replay host", &RuntimeHostWorkStats::replay_host_ns, 0, 0},
        {"replay wait", &RuntimeHostWorkStats::replay_device_wait_ns, 0, 0},
        {"engine boundary", &RuntimeHostWorkStats::engine_boundary_ns, 0, 0},
        {"device wait aggregate", &RuntimeHostWorkStats::device_wait_ns, 0, 0},
    };
    int failures = 0;
    const auto expect = [&](const RuntimeStats& stats, double prefill, double decode,
                            const char* name) {
        const auto text = metrics.render(stats, true);
        if (std::abs(sample(text, "ninfer_prefill_active_seconds_total") - prefill) > 1e-12 ||
            std::abs(sample(text, "ninfer_decode_active_seconds_total") - decode) > 1e-12) {
            std::cerr << "active seconds mismatch: " << name << '\n';
            ++failures;
        }
        if (metrics.render(stats, true) != text) {
            std::cerr << "active seconds scrape consumed counters: " << name << '\n';
            ++failures;
        }
    };
    expect(baseline, 0, 0, "warmup only");
    auto combined = baseline;
    for (const auto& scenario : scenarios) {
        auto current = baseline;
        current.host_work.*scenario.field += 1'000'000'000;
        combined.host_work.*scenario.field += 1'000'000'000;
        expect(current, scenario.prefill, scenario.decode, scenario.name);
    }
    expect(combined, 4, 4, "all work classes");
    options.max_concurrency = 3;
    options.context_cache.device_state_slots = 7;
    RuntimeStats occupied;
    occupied.device_state_occupied_slots = 2;
    occupied.host_state_occupied_slots = 5;
    for (bool host_images : {false, true}) {
        options.host_state_images = host_images;
        Metrics capacity;
        capacity.configure("state-capacity", options, {}, {});
        const auto text = capacity.render(occupied, true);
        if (sample(text, "ninfer_device_state_capacity_slots") != (host_images ? 3 : 10) ||
            sample(text, "ninfer_device_state_used_slots") != 2 ||
            sample(text, "ninfer_host_state_images") != 5) {
            std::cerr << "state capacity does not distinguish Host images from Device slots\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    int failures     = test_active_seconds();
    const auto check = [&](bool pass, const char* message) {
        if (!pass) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    EngineOptions options;
    options.max_concurrency                  = 2;
    options.context_cache.device_state_slots = 1;
    MemorySummary memory;
    memory.kv_capacity_page_groups     = 256;
    memory.host_context_capacity_bytes = 1048576;
    RuntimeStats baseline;
    baseline.generated_tokens         = 4;
    baseline.speculative_draft_tokens = 3;
    Metrics metrics;
    metrics.configure("custom\"model\\name\nline", options, memory, baseline);
    RuntimeStats running = baseline;
    running.generated_tokens += 5;
    running.speculative_draft_tokens += 6;
    running.speculative_accepted_tokens = 3;
    running.running_requests            = 1;
    running.host_context_occupied_bytes = 4096;
    running.host_context_reserved_bytes = 1024;
    running.kv_pools = {{0, "text", 2, 8}, {1, "draft", 1, 4}, {2, "index", 3, 6}};
    running.context_transfers.kv.resize(3);
    running.context_transfers.kv[2][1].bytes = 8192;
    metrics.first_token({.prepare_seconds = 0.05, .elapsed_since_submit_seconds = 0.2});
    const auto live = metrics.render(running, true);
    check(live.find("ninfer_generation_tokens_total 5\n") != std::string::npos,
          "live generated count must exclude startup warmup");
    check(live.find("ninfer_spec_decode_draft_tokens_total 6\n") != std::string::npos,
          "speculative work must be observable before request completion");
    check(live.find("ninfer_time_to_first_token_seconds_count 1\n") != std::string::npos &&
              live.find("ninfer_requests_total{outcome=\"completed\"} 0\n") != std::string::npos,
          "TTFT must be visible before completion");
    check(live.find("_bucket{le=\"0.25\"} 1\n") != std::string::npos &&
              live.find("ninfer_time_to_first_token_seconds_sum 0.25\n") != std::string::npos,
          "TTFT must include preparation and use inclusive histogram buckets");
    check(live.find("model_name=\"custom\\\"model\\\\name\\nline\"") != std::string::npos,
          "model label must escape quotes, backslashes and newlines");
    check(metrics.render(running, true) == live, "scrapes must not consume or reset counters");
    check(live.find("ninfer_host_context_used_bytes 4096\n") != std::string::npos,
          "reserved bytes must not be added to occupancy twice");

    check(live.find("ninfer_device_kv_used_page_groups{pool_id=\"2\"} 3\n") != std::string::npos &&
              live.find("ninfer_context_transfer_bytes_total{resource=\"kv\",pool_id=\"2\","
                        "direction=\"h2d\"} 8192\n") != std::string::npos,
          "pool metrics dropped the third pool or used a model role as its identity");

    GenerationOutcome outcome;
    outcome.finish_reason                            = FinishReason::OutputLimit;
    outcome.metrics.total_seconds                    = 2.0;
    outcome.metrics.engine_timing.queue_wait_seconds = 0.1;
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) {
        writers.emplace_back([&] {
            for (int n = 0; n < 100; ++n) { metrics.done(outcome); }
        });
    }
    for (auto& writer : writers) { writer.join(); }
    outcome.finish_reason = FinishReason::Cancelled;
    metrics.done(outcome);
    metrics.failed(true);
    metrics.failed(false);
    metrics.rejected();
    const auto final = metrics.render(running, false);
    check(final.find("ninfer_engine_ready 0\n") != std::string::npos,
          "unavailable engine must remain observable");
    check(final.find("ninfer_requests_total{outcome=\"completed\"} 400\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"cancelled\"} 2\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"failed\"} 1\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"rejected\"} 1\n") != std::string::npos,
          "concurrent settlements and terminal classifications must be preserved");
    check(final.find("ninfer_request_duration_seconds_bucket{le=\"+Inf\"} 401\n") !=
                  std::string::npos &&
              final.find("ninfer_request_duration_seconds_count 401\n") != std::string::npos &&
              final.find("ninfer_request_duration_seconds_sum 802\n") != std::string::npos,
          "histogram counts and sums must include cancelled outcomes without fabricating failure "
          "durations");
    check(final.find("ninfer_time_to_first_token_seconds_count 1\n") != std::string::npos,
          "request settlement must not count the first token again");
    outcome.constraint = ConstraintObservation{.complete          = true,
                                               .terminated        = false,
                                               .cache             = ConstraintCacheAccess::Built,
                                               .mask_positions    = 5,
                                               .mask_upload_bytes = 128};
    metrics.done(outcome);
    const auto constrained = metrics.render(running, false);
    check(constrained.find(
              "ninfer_constraint_requests_total{outcome=\"complete_interrupted\"} 1\n") !=
                  std::string::npos &&
              constrained.find("ninfer_constraint_mask_positions_total 5\n") != std::string::npos &&
              constrained.find("ninfer_constraint_mask_upload_bytes_total 128\n") !=
                  std::string::npos,
          "interrupted complete constraint lost its state or actual work");
    return failures == 0 ? 0 : 1;
}
