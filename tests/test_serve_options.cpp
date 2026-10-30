#include "product/config/options.h"
#include "serve/translate.h"
#include "serve/openai_chat.h"
#include "product/logging/configuration.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;
using ninfer::product::ServeOptions;

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return ninfer::product::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void rejects(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid server configuration was accepted");
}

void test_resources_and_units() {
    const auto defaults = parse({"ninfer-serve", "model.ninfer"});
    const auto& engine  = defaults.launch.engine;
    require(engine.context_cache.enabled && !engine.context_cache.host_capacity_bytes &&
                engine.memory_headroom_bytes == (512ULL << 20),
            "server resource defaults were resolved before model preparation");
    require(defaults.launch.generation.max_tokens == 8192 &&
                !defaults.launch.generation.sampling.seed &&
                !defaults.launch.generation.enable_thinking &&
                defaults.launch.logging.stats_interval_ms == 5000,
            "server defaults changed");
    require(!defaults.model_id_override, "public model alias is unexpectedly set by default");
    for (auto [quantity, bytes] : std::vector<std::pair<std::string, std::uint64_t>>{
             {"0B", 0},
             {"1.5 MiB", 1572864},
             {"0.00000095367431640625MiB", 1},
             {"18446744073709551615B", UINT64_MAX},
             {"17592186044415.99999904632568359375MiB", UINT64_MAX}}) {
        const auto parsed =
            parse({"ninfer-serve", "model.ninfer", "--host-cache-capacity", quantity});
        require(parsed.launch.engine.context_cache.host_capacity_bytes == bytes,
                "capacity lost exact bytes");
    }
    const auto controls =
        parse({"ninfer-serve", "--pending-timeout", "1.25s", "model.ninfer", "--stats-interval",
               "0ms", "--media-cache-capacity", "0B", "--max-request-size", "1.5MiB",
               "--max-concurrency", "4", "--media-preprocess-threads", "auto", "--context-capacity",
               "auto", "--checkpoint-interval", "16384"});
    require(controls.launch.engine.pending_timeout_ms == 1250 &&
                controls.launch.logging.stats_interval_ms == 0 &&
                controls.max_request_bytes == 1572864 &&
                controls.launch.engine.context_cache.checkpoint_interval == 16384 &&
                controls.launch.engine.media_cache_bytes == 0 &&
                controls.launch.engine.context_capacity.mode ==
                    ninfer::ContextCapacityMode::Automatic,
            "unit, zero or automatic controls changed");
    for (const auto& suffix :
         std::vector<std::vector<std::string>>{{"--host-cache-capacity", "18446744073709551616B"},
                                               {"--context-capacity", "0B"},
                                               {"--context-capacity", "0.1B"},
                                               {"--host-cache-capacity", "0.1B"},
                                               {"--host-cache-capacity", "-1MiB"},
                                               {"--pending-timeout", "0ms"},
                                               {"--pending-timeout", "0.1ms"},
                                               {"--pending-timeout", "4294967296ms"},
                                               {"--media-live-capacity", "0B"},
                                               {"--max-concurrency", "9"},
                                               {"--checkpoint-interval", "0"},
                                               {"--port", "65536"},
                                               {"--top-k", "21"},
                                               {"--media-preprocess-threads", "0"}}) {
        std::vector<std::string> args{"ninfer-serve", "model.ninfer"};
        args.insert(args.end(), suffix.begin(), suffix.end());
        rejects([&] { parse(args); });
    }
}

void test_request_defaults() {
    const auto options =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--seed",
               "0", "--thinking-budget", "37", "--no-thinking", "--preserve-thinking"});
    GenerationRequest request;
    request.max_tokens = 1;
    auto semantics     = collect_prompt_options(request, options);
    auto execution     = to_request_options(request, options, true);
    require(
        execution.execution.sampling.temperature == 0 && execution.execution.sampling.seed == 0 &&
            execution.execution.thinking.budget == 37 && !semantics.enable_thinking &&
            semantics.defaults.enable_thinking == false && execution.execution.allow_prefix_reuse,
        "startup defaults did not reach the request");
    request.sampling.temperature = 1.1;
    request.sampling.seed        = 42;
    request.enable_thinking      = true;
    request.preserve_thinking    = false;
    semantics                    = collect_prompt_options(request, options);
    execution                    = to_request_options(request, options, false);
    require(execution.execution.sampling.temperature == 1.1F &&
                execution.execution.sampling.seed == 42 &&
                execution.execution.thinking.budget == 37 &&
                !execution.execution.allow_prefix_reuse && semantics.preserve_thinking == false,
            "explicit request fields did not override startup defaults");
    request.thinking_budget = 9;
    require(to_request_options(request, options, true).execution.thinking.budget == 9,
            "request thinking budget did not override the default");
    const auto large  = parse({"ninfer-serve", "model.ninfer", "--max-tokens", "4294967295"});
    const auto parsed = parse_chat_completion_request(
        RequestJson{{"model", "test"},
                    {"messages", RequestJson::array({{{"role", "user"}, {"content", "x"}}})}},
        {.default_max_tokens = large.launch.generation.max_tokens});
    require(parsed.generation.max_tokens == UINT32_MAX,
            "server output default overflowed at the protocol boundary");
}

void test_5090d_migration() {
    const auto value = parse({"ninfer-serve", "model.ninfer", "--max-context", "200000",
        "--context-capacity", "auto", "--host-cache-capacity", "4GiB",
        "--memory-headroom", "400MiB", "--max-concurrency", "2", "--kv-dtype", "fp8",
        "--prefill-chunk", "1024", "--spec", "dflash2", "--draft-tokens", "7",
        "--proposal-head", "optimized", "--thinking-budget", "4096", "--preserve-thinking",
        "--max-request-size", "64MiB", "--media-cache-capacity", "0B",
        "--media-live-capacity", "2GiB", "--vision", "--vision-memory-budget", "1GiB",
        "--max-vision-item-tokens", "2048",
        "--host-mapped-embedding", "--host-mapped-dflash2-codebooks", "--host-pinned-vision-weights"});
    const auto& e = value.launch.engine;
    require(e.host_mapped_embedding && e.host_mapped_dflash2_codebooks &&
                e.host_pinned_vision_weights && e.max_vision_item_tokens == 2048 &&
                e.context_cache.host_capacity_bytes == (4ULL << 30) &&
                e.memory_headroom_bytes == (400ULL << 20) &&
                e.context_capacity.mode == ninfer::ContextCapacityMode::Automatic,
            "5090D startup flags did not reach engine options");
    require(e.max_context == 200000 && e.max_concurrency == 2 && e.prefill_chunk == 1024 &&
                e.kv_cache == ninfer::KvCacheStorage::Fp8E4m3fnGroup128Ue8m0 &&
                e.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                e.speculative.draft_tokens == 7 &&
                e.speculative.proposal_head == ninfer::ProposalHead::Optimized &&
                e.media_cache_bytes == 0 && e.media_live_bytes == (2ULL << 30) &&
                e.enable_vision && e.vision_memory_budget_bytes == (1ULL << 30) &&
                value.launch.generation.thinking_budget == 4096 &&
                value.launch.generation.preserve_thinking == true &&
                value.max_request_bytes == (64ULL << 20),
            "new startup command no longer preserves the production execution settings");
    const auto defaults = parse({"ninfer-serve", "model.ninfer"}).launch.engine;
    require(!defaults.host_mapped_embedding && !defaults.host_mapped_dflash2_codebooks &&
                !defaults.host_pinned_vision_weights && defaults.max_vision_item_tokens == 16384,
            "custom optimization became enabled by default");
    const auto bytes = parse({"ninfer-serve", "model.ninfer", "--context-capacity", "300000"});
    require(bytes.launch.engine.context_capacity.explicit_bytes == 300000,
            "new capacity must be documented as bytes, not historical KV token count");
    for (const auto* count : {"0", "16385", "-1", "2048.5"})
        rejects([&] { (void)parse({"ninfer-serve", "model.ninfer", "--max-vision-item-tokens", count}); });
    for (const auto* old : {"--kv-capacity", "--device-state-slots", "--host-context-mib",
                            "--default-thinking-budget", "--max-request-mib", "--media-cache-mib"})
        rejects([&] { (void)parse({"ninfer-serve", "model.ninfer", old, "0"}); });
    const auto rendered = ninfer::product::format_configuration(
        ninfer::product::configuration_view(value));
    require(rendered.find("host_mapped_embedding") != std::string::npos &&
                rendered.find("max_item_tokens") != std::string::npos,
            "startup display omitted optimization settings");
}

void test_provenance() {
    for (auto args : std::vector<std::vector<std::string>>{
             {"ninfer-serve", "model.ninfer", "--api-key", "private-value"},
             {"ninfer-serve", "--api-key=private-value", "model.ninfer"}}) {
        const auto options = parse(args);
        require(options.api_key == "private-value",
                "authentication value was not retained for the server");
        const auto view = ninfer::product::configuration_view(options);
        for (const auto& argument : view.argv)
            require(argument.find("private-value") == std::string::npos, "argv leaked the API key");
        const auto rendered = ninfer::product::format_configuration(view);
        require(rendered.find("private-value") == std::string::npos &&
                    rendered.find("configured") != std::string::npos,
                "startup display leaked or lost authentication state");
    }
}

} // namespace

int main() {
    try {
        test_resources_and_units();
        test_request_defaults();
        test_5090d_migration();
        test_provenance();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
