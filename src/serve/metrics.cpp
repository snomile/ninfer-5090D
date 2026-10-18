#include "serve/metrics.h"

#include "product/speculative_options.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string_view>
#include <utility>

#if defined(__linux__)
#    include <features.h>
#    if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#        if __GLIBC_PREREQ(2, 33)
#            include <malloc.h>
#            define NINFER_HAS_MALLINFO2 1
#        endif
#    endif
#endif

namespace ninfer::serve {
namespace {

constexpr std::array kBuckets{0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1,   0.25,  0.5,   1.0,
                              2.5,   5.0,   10.0,  20.0, 40.0,  80.0, 160.0, 320.0, 640.0, 1280.0};

std::string escape_label(std::string_view value) {
    std::string out;
    for (char c : value) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

struct AllocatorStats {
    bool available              = false;
    std::size_t arena_bytes     = 0;
    std::size_t allocated_bytes = 0;
    std::size_t free_bytes      = 0;
    std::size_t mmap_bytes      = 0;
    std::size_t mmap_regions    = 0;
};

AllocatorStats allocator_stats() noexcept {
#if defined(NINFER_HAS_MALLINFO2)
    const struct mallinfo2 info = mallinfo2();
    return AllocatorStats{
        .available       = true,
        .arena_bytes     = info.arena,
        .allocated_bytes = info.uordblks,
        .free_bytes      = info.fordblks,
        .mmap_bytes      = info.hblkhd,
        .mmap_regions    = info.hblks,
    };
#else
    return {};
#endif
}

} // namespace

void Metrics::Histogram::observe(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) { return; }
    ++bins[std::lower_bound(kBuckets.begin(), kBuckets.end(), seconds) - kBuckets.begin()];
    sum += seconds;
}

void Metrics::configure(std::string model, const EngineOptions& options,
                        const MemorySummary& memory, RuntimeStats baseline) {
    model_    = std::move(model);
    options_  = options;
    memory_   = memory;
    baseline_ = baseline;
    start_seconds_ =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void Metrics::first_token(const GenerationFirstTokenObservation& observation) {
    std::lock_guard lock(mutex_);
    requests_.ttft.observe(observation.prepare_seconds + observation.elapsed_since_submit_seconds);
}

void Metrics::done(const GenerationOutcome& outcome) {
    std::lock_guard lock(mutex_);
    if (outcome.finish_reason == FinishReason::Cancelled) {
        ++requests_.cancelled;
    } else {
        ++requests_.completed;
    }
    if (outcome.constraint) {
        const auto& c = *outcome.constraint;
        ++requests_.constraint_outcomes[c.terminated ? 0 : c.complete ? 1 : 2];
        ++requests_.constraint_cache[static_cast<unsigned>(c.cache)];
        requests_.constraint_prepare_seconds += c.prepare_seconds;
        requests_.constraint_mask_seconds += c.mask_seconds;
        requests_.constraint_matcher_seconds += c.matcher_seconds;
        requests_.constraint_positions += c.mask_positions;
        requests_.constraint_upload_bytes += c.mask_upload_bytes;
    }
    requests_.duration.observe(outcome.metrics.total_seconds);
    requests_.queue.observe(outcome.metrics.engine_timing.queue_wait_seconds);
}

void Metrics::rejected() {
    std::lock_guard lock(mutex_);
    ++requests_.rejected;
}

void Metrics::failed(bool cancelled) {
    std::lock_guard lock(mutex_);
    if (cancelled) {
        ++requests_.cancelled;
    } else {
        ++requests_.failed;
    }
}

void Metrics::response_failed() {
    std::lock_guard lock(mutex_);
    ++requests_.response_failures;
}

std::string Metrics::render(const RuntimeStats& stats, const MediaCacheSummary& media_cache,
                            bool ready) const {
    Requests requests;
    {
        std::lock_guard lock(mutex_);
        requests = requests_;
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    const auto header = [&](std::string_view name, std::string_view type, std::string_view help) {
        out << "# HELP ninfer_" << name << ' ' << help << '\n'
            << "# TYPE ninfer_" << name << ' ' << type << '\n';
    };
    const auto gauge = [&](std::string_view name, auto value, std::string_view help) {
        header(name, "gauge", help);
        out << "ninfer_" << name << ' ' << value << '\n';
    };
    const auto counter = [&](std::string_view name, auto current, auto baseline,
                             std::string_view help) {
        header(name, "counter", help);
        out << "ninfer_" << name << ' ' << current - baseline << '\n';
    };
    const AllocatorStats allocator = allocator_stats();
    header("constraint_requests_total", "counter",
           "Settled constrained requests by language completion.");
    const char* outcomes[]    = {"terminated", "complete_interrupted", "incomplete_interrupted"};
    const char* cache_names[] = {"hit", "built", "waited"};
    for (std::size_t i = 0; i < 3; ++i)
        out << "ninfer_constraint_requests_total{outcome=\"" << outcomes[i] << "\"} "
            << requests.constraint_outcomes[i] << '\n';
    header("constraint_cache_total", "counter",
           "Compilation cache access for settled constrained requests.");
    for (std::size_t i = 0; i < 3; ++i)
        out << "ninfer_constraint_cache_total{result=\"" << cache_names[i] << "\"} "
            << requests.constraint_cache[i] << '\n';
    counter("constraint_prepare_seconds_total", requests.constraint_prepare_seconds, 0.0,
            "Observed constraint preparation work.");
    counter("constraint_mask_seconds_total", requests.constraint_mask_seconds, 0.0,
            "Observed CPU mask work, including lookahead rollback.");
    counter("constraint_matcher_seconds_total", requests.constraint_matcher_seconds, 0.0,
            "Observed matcher acceptance and discard work.");
    counter("constraint_mask_positions_total", requests.constraint_positions, 0,
            "Evaluated constrained prediction positions.");
    counter("constraint_mask_upload_bytes_total", requests.constraint_upload_bytes, 0,
            "Submitted mask payload bytes.");
    counter("constraint_draft_wait_seconds_total", stats.host_work.constraint_draft_wait_ns * 1e-9,
            baseline_.host_work.constraint_draft_wait_ns * 1e-9,
            "Draft-ready wait, counted once per batch.");
    gauge("engine_ready", ready ? 1 : 0, "Whether Engine can accept work.");
    gauge("server_start_time_seconds", start_seconds_,
          "Unix time at service attachment after warmup.");
    header("model_info", "gauge", "Resident model and speculative backend.");
    out << "ninfer_model_info{model_name=\"" << escape_label(model_) << "\",speculative_backend=\""
        << product::speculative_backend_name(options_.speculative.backend) << "\"} 1\n";
    gauge("max_concurrency", options_.max_concurrency, "Maximum resident execution lanes.");
    gauge("max_context_tokens", options_.max_context, "Per-request logical context limit.");
    gauge("spec_decode_draft_window", options_.speculative.draft_tokens,
          "Configured speculative draft window.");
    gauge("media_cache_capacity_bytes", media_cache.capacity_bytes,
          "Maximum retained preprocessed media payload bytes; zero disables reuse.");
    gauge("media_cache_retained_bytes", media_cache.retained_bytes,
          "Preprocessed media payload bytes retained for cross-request reuse.");
    gauge(
        "media_cache_live_capacity_bytes", media_cache.live_capacity_bytes,
        "Maximum live preprocessed media payload bytes, including retained and request-held data.");
    gauge("media_cache_live_bytes", media_cache.live_bytes,
          "Live preprocessed media payload bytes, including retained and request-held data.");
    gauge("media_cache_entries", media_cache.entries,
          "Preprocessed media entries retained for cross-request reuse.");
    gauge("media_cache_inflight", media_cache.inflight,
          "Distinct media items currently being preprocessed.");
    gauge("media_preprocess_tasks_queued", media_cache.queued_tasks,
          "Media preprocessing tasks waiting for a worker.");
    gauge("media_preprocess_tasks_active", media_cache.active_tasks,
          "Media preprocessing tasks currently running.");
    gauge("media_preprocess_threads", media_cache.preprocess_threads,
          "Configured media preprocessing worker threads.");
    counter("media_cache_hits_total", media_cache.hits, std::uint64_t{0},
            "Preprocessed media lookups served from retained entries.");
    counter("media_cache_misses_total", media_cache.misses, std::uint64_t{0},
            "Preprocessed media lookups that started new preprocessing work.");
    counter("media_cache_singleflight_waits_total", media_cache.singleflight_waits,
            std::uint64_t{0},
            "Preprocessed media lookups joined matching work already in progress.");
    counter("media_cache_evictions_total", media_cache.evictions, std::uint64_t{0},
            "Retained preprocessed media entries evicted to honor the cache capacity.");
    counter("media_cache_oversize_bypasses_total", media_cache.oversize_bypasses, std::uint64_t{0},
            "Preprocessed media results not retained because they exceeded cache capacity.");

    gauge("process_allocator_stats_available", allocator.available ? 1 : 0,
          "Whether glibc mallinfo2 allocator statistics are available.");
    gauge("process_allocator_arena_bytes", allocator.arena_bytes,
          "Bytes obtained by glibc for non-mmap allocation arenas.");
    gauge("process_allocator_allocated_bytes", allocator.allocated_bytes,
          "Bytes currently allocated from glibc non-mmap arenas.");
    gauge("process_allocator_free_bytes", allocator.free_bytes,
          "Free bytes retained inside glibc non-mmap arenas.");
    gauge("process_allocator_mmap_bytes", allocator.mmap_bytes,
          "Bytes currently held in glibc mmap-backed allocation regions.");
    gauge("process_allocator_mmap_regions", allocator.mmap_regions,
          "Current glibc mmap-backed allocation region count.");
#define GAUGE(field, name, help)   gauge(name, stats.field, help)
#define COUNTER(field, name, help) counter(name, stats.field, baseline_.field, help)
    GAUGE(running_requests, "requests_running", "Resident requests, including prefill and decode.");
    GAUGE(waiting_requests, "requests_waiting", "Requests waiting for initial admission.");
    GAUGE(paused_requests, "requests_paused", "Paused requests waiting for recovery.");
    GAUGE(prefilling_requests, "requests_prefilling", "Resident requests in initial prefill.");
    GAUGE(decode_ready_requests, "requests_decode_ready", "Resident requests ready for decode.");
    GAUGE(replaying_requests, "requests_replaying",
          "Resident requests rebuilding committed history.");
    GAUGE(materializing_requests, "requests_materializing",
          "Requests with a binding transaction in progress.");
    COUNTER(prompt_tokens, "prompt_tokens_total",
            "Full input tokens counted once on initial binding.");
    COUNTER(reused_prompt_tokens, "prompt_tokens_cached_total",
            "Input tokens supplied by exact checkpoints.");
    COUNTER(computed_prefill_tokens, "prefill_tokens_total",
            "Initial input tokens actually computed, excluding Replay.");
    COUNTER(generated_tokens, "generation_tokens_total",
            "Committed output tokens, including first and control tokens.");
    COUNTER(committed_decode_tokens, "decode_tokens_total",
            "Committed decode/control tokens, excluding the first token.");
    counter("prefill_active_seconds_total",
            (stats.host_work.prefill_host_ns + stats.host_work.prefill_device_wait_ns +
             stats.host_work.control_host_ns + stats.host_work.control_device_wait_ns) *
                1e-9,
            (baseline_.host_work.prefill_host_ns + baseline_.host_work.prefill_device_wait_ns +
             baseline_.host_work.control_host_ns + baseline_.host_work.control_device_wait_ns) *
                1e-9,
            "Engine-active Prefill and Control wall time, excluding Replay.");
    counter("decode_active_seconds_total",
            (stats.host_work.decode_host_ns + stats.host_work.decode_device_wait_ns +
             stats.host_work.control_host_ns + stats.host_work.control_device_wait_ns) *
                1e-9,
            (baseline_.host_work.decode_host_ns + baseline_.host_work.decode_device_wait_ns +
             baseline_.host_work.control_host_ns + baseline_.host_work.control_device_wait_ns) *
                1e-9,
            "Engine-active Decode and Control wall time, excluding Replay.");
    COUNTER(replayed_tokens, "replayed_tokens_total", "History tokens recomputed during recovery.");
    COUNTER(decode_rounds, "decode_rounds_total", "Decode batch executions.");
    COUNTER(decode_row_rounds, "decode_row_rounds_total", "Sum of decode batch sizes.");
    COUNTER(preemptions, "preemptions_total", "Resource-pressure pauses.");
    COUNTER(snapshot_restores, "snapshot_restores_total",
            "Restores from complete paused snapshots.");
    COUNTER(replay_restores, "replay_restores_total", "Recovery operations that rebuild history.");
    COUNTER(root_selections, "root_selections_total", "Initial bindings without checkpoint reuse.");
    COUNTER(checkpoint_selections, "checkpoint_selections_total",
            "Initial bindings with checkpoint reuse.");
    COUNTER(speculative_rounds, "spec_decode_rounds_total",
            "Native speculative verification rounds.");
    COUNTER(speculative_draft_tokens, "spec_decode_draft_tokens_total",
            "Draft tokens evaluated by native verification.");
    COUNTER(speculative_accepted_tokens, "spec_decode_accepted_tokens_total",
            "Draft tokens accepted by native verification.");
    COUNTER(speculative_fallback_steps, "spec_decode_fallback_steps_total",
            "Speculative fallback steps without drafts.");
    GAUGE(device_state_occupied_slots, "device_state_used_slots",
          "Occupied device StateImage slots.");
    gauge("device_state_capacity_slots",
          options_.max_concurrency + options_.context_cache.device_state_slots.value(),
          "Total device StateImage slots, including resident lanes.");
    header("device_kv_used_page_groups", "gauge", "Occupied physical KV page groups by pool.");
    header("device_kv_capacity_page_groups", "gauge", "Physical KV page-group capacity by pool.");
    for (const auto& pool : stats.kv_pools) {
        out << "ninfer_device_kv_used_page_groups{pool_id=\"" << pool.id << "\"} "
            << pool.occupied_page_groups << '\n';
        out << "ninfer_device_kv_capacity_page_groups{pool_id=\"" << pool.id << "\"} "
            << pool.capacity_page_groups << '\n';
    }
    GAUGE(host_context_occupied_bytes, "host_context_used_bytes",
          "Occupied Host backing, including reservations.");
    GAUGE(host_context_reserved_bytes, "host_context_reserved_bytes",
          "Reserved Host bytes, a subset of used bytes.");
    gauge("host_context_capacity_bytes", memory_.host_context_capacity_bytes,
          "Shared pinned Host backing capacity.");
    GAUGE(host_context_peak_occupied_bytes, "host_context_peak_bytes",
          "Host backing high-water mark, including reservations.");
    GAUGE(host_state_occupied_slots, "host_state_images",
          "StateImages in the shared Host backing.");
    GAUGE(host_kv_occupied_bytes, "host_kv_used_bytes",
          "KV bytes in the shared Host backing, a subset of used bytes.");
#undef GAUGE
#undef COUNTER

    header("context_transfer_bytes_total", "counter", "Completed context payload transfers.");
    constexpr std::array directions{"d2h", "h2d", "d2d"};
    for (std::size_t i = 0; i < directions.size(); ++i) {
        out << "ninfer_context_transfer_bytes_total{resource=\"state\",direction=\""
            << directions[i] << "\"} "
            << stats.context_transfers.state[i].bytes - baseline_.context_transfers.state[i].bytes
            << '\n';
        for (const auto& pool : stats.kv_pools) {
            const auto baseline = pool.id < baseline_.context_transfers.kv.size()
                                      ? baseline_.context_transfers.kv[pool.id][i].bytes
                                      : 0;
            out << "ninfer_context_transfer_bytes_total{resource=\"kv\",pool_id=\"" << pool.id
                << "\",direction=\"" << directions[i] << "\"} "
                << stats.context_transfers.kv.at(pool.id)[i].bytes - baseline << '\n';
        }
    }
    counter("context_transfer_seconds_total", stats.actual_context_transfer_seconds,
            baseline_.actual_context_transfer_seconds,
            "Accumulated context transfer operation time.");
    header("host_work_seconds_total", "counter",
           "Exclusive Engine Host phases, excluding device waits.");
    const auto host = [&](std::string_view phase, std::uint64_t value, std::uint64_t baseline) {
        out << "ninfer_host_work_seconds_total{phase=\"" << phase << "\"} "
            << static_cast<double>(value - baseline) * 1e-9 << '\n';
    };
#define HOST(field) host(#field, stats.host_work.field##_ns, baseline_.host_work.field##_ns)
    HOST(engine_boundary);
    HOST(program_submit);
    HOST(program_post);
    HOST(engine_commit_output);
    HOST(engine_maintenance);
#undef HOST
    counter("device_wait_seconds_total", stats.host_work.device_wait_ns * 1e-9,
            baseline_.host_work.device_wait_ns * 1e-9,
            "Engine wall time waiting for device work, not kernel time.");
    header("requests_total", "counter",
           "Generation attempts entering preparation, by terminal outcome.");
    for (const auto& [outcome, count] : std::array{
             std::pair{"completed", requests.completed}, std::pair{"cancelled", requests.cancelled},
             std::pair{"failed", requests.failed}, std::pair{"rejected", requests.rejected}}) {
        out << "ninfer_requests_total{outcome=\"" << outcome << "\"} " << count << '\n';
    }
    counter("response_failures_total", requests.response_failures, 0U,
            "Response rendering, storage or transport failures after generation settlement.");
    const auto histogram = [&](std::string_view name, const Histogram& value,
                               std::string_view help) {
        header(name, "histogram", help);
        std::uint64_t count = 0;
        for (std::size_t i = 0; i < value.bins.size(); ++i) {
            count += value.bins[i];
            out << "ninfer_" << name << "_bucket{le=\"";
            if (i == kBuckets.size()) {
                out << "+Inf";
            } else {
                out << kBuckets[i];
            }
            out << "\"} " << count << '\n';
        }
        out << "ninfer_" << name << "_sum " << value.sum << '\n'
            << "ninfer_" << name << "_count " << count << '\n';
    };
    histogram("time_to_first_token_seconds", requests.ttft,
              "Preparation through first committed token; observed once, before completion.");
    histogram("request_duration_seconds", requests.duration,
              "Generation request duration through settlement, including cancellation.");
    histogram("request_queue_seconds", requests.queue,
              "Initial Engine queue wait of settled generation requests.");
    return out.str();
}

} // namespace ninfer::serve

#if defined(NINFER_HAS_MALLINFO2)
#    undef NINFER_HAS_MALLINFO2
#endif
