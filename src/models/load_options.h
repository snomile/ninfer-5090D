#pragma once

#include "ninfer/types.h"

#include <string_view>

namespace ninfer::models {

struct LoadOptions {
    EnginePurpose purpose          = EnginePurpose::Generation;
    bool vision                    = false;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;
    bool host_mapped_embedding        = false;
    bool host_mapped_dflash2_codebooks = false;
    bool host_pinned_vision_weights   = false;

    bool operator==(const LoadOptions&) const = default;

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool masked_draft() const noexcept { return dflash() || dflash2(); }

    [[nodiscard]] bool proposal_enabled() const noexcept {
        return purpose == EnginePurpose::Generation && speculative != SpeculativeBackend::None &&
               proposal_head == ProposalHead::Optimized;
    }

    [[nodiscard]] std::string_view speculative_component() const noexcept {
        switch (speculative) {
        case SpeculativeBackend::None:
            return {};
        case SpeculativeBackend::Mtp:
            return "mtp";
        case SpeculativeBackend::DFlash:
            return "dflash";
        case SpeculativeBackend::DFlash2:
            return "dflash2";
        }
        return {};
    }
};

[[nodiscard]] constexpr bool is_masked_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2;
}

[[nodiscard]] inline LoadOptions load_options(const EngineOptions& options) noexcept {
    return {.purpose       = options.purpose,
            .vision        = options.enable_vision,
            .speculative   = options.speculative.backend,
            .proposal_head = options.speculative.proposal_head,
            .host_mapped_embedding = options.host_mapped_embedding,
            .host_mapped_dflash2_codebooks = options.host_mapped_dflash2_codebooks,
            .host_pinned_vision_weights = options.host_pinned_vision_weights};
}

inline void validate_memory_options(const LoadOptions& options) {
    if (options.host_mapped_dflash2_codebooks && !options.dflash2())
        throw std::invalid_argument("Host-mapped codebooks require the DFlash2 backend");
    if (options.host_pinned_vision_weights && !options.vision)
        throw std::invalid_argument("Host-pinned Vision weights require Vision");
}

} // namespace ninfer::models
