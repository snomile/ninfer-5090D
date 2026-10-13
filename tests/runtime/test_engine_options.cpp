#include "runtime/engine/model_instance.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    ninfer::EngineOptions defaults;
    defaults.max_concurrency = 3;
    const auto normalized_defaults =
        ninfer::runtime::normalize_engine_options(std::move(defaults));
    failures += check(normalized_defaults.context_cache.device_state_slots == 3,
                      "default extra Device StateImage capacity changed");

    ninfer::EngineOptions host_states;
    host_states.max_concurrency = 3;
    host_states.host_state_images = true;
    const auto normalized_host_states =
        ninfer::runtime::normalize_engine_options(std::move(host_states));
    failures += check(normalized_host_states.context_cache.device_state_slots == 0 &&
                          normalized_host_states.host_state_images,
                      "Host StateImage mode retained extra Device slots");

    ninfer::EngineOptions explicit_zero;
    explicit_zero.host_state_images = true;
    explicit_zero.context_cache.device_state_slots = 0;
    const auto normalized_zero =
        ninfer::runtime::normalize_engine_options(std::move(explicit_zero));
    failures += check(normalized_zero.context_cache.device_state_slots == 0,
                      "explicit zero Device StateImage slots changed");

    bool conflict_rejected = false;
    try {
        ninfer::EngineOptions conflict;
        conflict.host_state_images = true;
        conflict.context_cache.device_state_slots = 1;
        (void)ninfer::runtime::normalize_engine_options(std::move(conflict));
    } catch (const std::invalid_argument&) { conflict_rejected = true; }
    failures +=
        check(conflict_rejected, "Host StateImage mode accepted nonzero extra Device slots");

    ninfer::EngineOptions scoring;
    scoring.purpose = ninfer::EnginePurpose::CausalScoring;
    scoring.host_mapped_dflash2_codebooks = true;
    scoring.host_pinned_vision_weights = true;
    scoring.host_state_images = true;
    const auto normalized_scoring =
        ninfer::runtime::normalize_engine_options(std::move(scoring));
    failures += check(!normalized_scoring.host_mapped_dflash2_codebooks &&
                          !normalized_scoring.host_pinned_vision_weights &&
                          !normalized_scoring.host_state_images,
                      "causal scoring retained generation-only Host options");

    return failures == 0 ? 0 : 1;
}
