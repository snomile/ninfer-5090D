#include "options.h"
#include "models/load_options.h"
#include "serve/serve_options.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Parse>
auto parse(Parse parser, std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return parser(static_cast<int>(argv.size()), argv.data());
}

template <class Fn>
void rejects(Fn fn) {
    try { fn(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid memory configuration was accepted");
}

template <class Parse>
void options(Parse parser, bool cli) {
    const auto base = [&] {
        std::vector<std::string> arguments{"ninfer", "model.ninfer"};
        if (cli) arguments.insert(arguments.end(), {"--prompt", "hello"});
        return arguments;
    };
    const auto configured = [&](std::vector<std::string> extra) {
        auto arguments = base();
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        return parse(parser, std::move(arguments));
    };
    const auto defaults = configured({});
    require(!defaults.host_mapped_embedding && !defaults.host_mapped_dflash2_codebooks &&
                !defaults.host_pinned_vision_weights &&
                defaults.max_vision_item_tokens == 16384,
            "memory defaults changed");
    for (unsigned mask = 0; mask < 8; ++mask) {
        std::vector<std::string> flags{"--vision", "--spec", "dflash2", "--draft-tokens", "7"};
        if (mask & 1) flags.push_back("--host-mapped-embedding");
        if (mask & 2) flags.push_back("--host-mapped-dflash2-codebooks");
        if (mask & 4) flags.push_back("--host-pinned-vision-weights");
        const auto value = configured(flags);
        require(value.host_mapped_embedding == bool(mask & 1) &&
                    value.host_mapped_dflash2_codebooks == bool(mask & 2) &&
                    value.host_pinned_vision_weights == bool(mask & 4),
                "independent memory flags were coupled or dropped");
    }
    for (unsigned limit : {1U, 2048U, 4096U, 16384U}) {
        require(configured({"--max-vision-item-tokens", std::to_string(limit)})
                    .max_vision_item_tokens == limit, "Vision item limit was changed");
    }
    for (const char* invalid : {"0", "16385", "-1", "4294967296", "abc"}) {
        rejects([&] { configured({"--max-vision-item-tokens", invalid}); });
    }
    rejects([&] { configured({"--max-vision-item-tokens"}); });
    rejects([&] { configured({"--host-pinned-vision-weights"}); });
    rejects([&] { configured({"--host-mapped-dflash2-codebooks"}); });
    rejects([&] {
        configured({"--spec", "dflash", "--draft-tokens", "7",
                    "--host-mapped-dflash2-codebooks"});
    });
    rejects([&] { configured({"--host-state-images"}); });
    if constexpr (requires { defaults.context_cache; }) {
        const auto disabled = configured({"--device-state-slots", "0", "--host-context-mib", "0"});
        require(disabled.context_cache.device_state_slots == 0 &&
                    disabled.context_cache.host_capacity_bytes == 0,
                "explicit zero snapshot capacity was lost");
    }
    const auto help = cli ? ninfer::cli::usage_text("ninfer") :
                            ninfer::serve::serve_usage_text("ninfer");
    for (const char* flag : {"--host-mapped-embedding", "--host-mapped-dflash2-codebooks",
                             "--host-pinned-vision-weights", "--max-vision-item-tokens"}) {
        require(help.find(flag) != std::string::npos, "help omitted a memory flag");
    }
    require(help.find("--host-state-images") == std::string::npos,
            "help still advertises the removed Host StateImage flag");
}

void loader_mapping() {
    ninfer::EngineOptions engine;
    for (unsigned mask = 0; mask < 8; ++mask) {
        engine.enable_vision = true;
        engine.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        engine.host_mapped_embedding = mask & 1;
        engine.host_mapped_dflash2_codebooks = mask & 2;
        engine.host_pinned_vision_weights = mask & 4;
        const auto load = ninfer::models::load_options(engine);
        require(load.host_mapped_embedding == bool(mask & 1) &&
                    load.host_mapped_dflash2_codebooks == bool(mask & 2) &&
                    load.host_pinned_vision_weights == bool(mask & 4),
                "Engine memory settings did not reach the loader");
        ninfer::models::validate_memory_options(load);
    }
    engine.enable_vision = false;
    rejects([&] { ninfer::models::validate_memory_options(ninfer::models::load_options(engine)); });
    engine.enable_vision = true;
    for (const auto backend : {ninfer::SpeculativeBackend::None, ninfer::SpeculativeBackend::Mtp,
                               ninfer::SpeculativeBackend::DFlash}) {
        engine.speculative.backend = backend;
        rejects([&] { ninfer::models::validate_memory_options(ninfer::models::load_options(engine)); });
    }
}

} // namespace

int main() {
    try {
        options(ninfer::cli::parse_options, true);
        options(ninfer::serve::parse_serve_options, false);
        loader_mapping();
        std::cout << "ok: independent memory flags, loader mapping and limits\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
