#include "models/qwen3_5/load.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/common/qwen/vision/vision.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) return 77;
    try {
        namespace q35 = ninfer::models::qwen3_5;
        ninfer::models::LoadOptions selected;
        selected.vision = true;
        ninfer::artifact::Reader reader(artifact);
        auto definition    = q35::read_definition(reader, selected);
        const auto profile = q35::execution::describe_model(*definition);
        const auto plan    = [&](std::uint32_t bound, std::size_t budget) {
            return ninfer::models::qwen::VisionContext::plan_resources(
                *profile->config.vision, *profile->vision, bound, budget);
        };
        constexpr auto budget = 1ULL << 30;
        const auto small      = plan(262144, budget / 2);
        const auto full       = plan(262144, budget);
        const auto longer     = plan(1048576, budget);
        if (!small.max_features || small.max_features >= full.max_features ||
            small.device_bytes() > budget / 2 || full.device_bytes() > budget ||
            full.max_features != longer.max_features ||
            full.device_bytes() != longer.device_bytes())
            throw std::runtime_error(
                "Vision capacity must follow its budget, not text context length");
        selected.host_pinned_vision_weights = true;
        const auto staged_definition = q35::read_definition(reader, selected);
        const auto staged_profile = q35::execution::describe_model(*staged_definition);
        const auto capped = plan(2048, budget);
        const auto staged = ninfer::models::qwen::VisionContext::plan_resources(
            *staged_profile->config.vision, *staged_profile->vision, 2048, budget);
        const auto slot = staged_profile->vision->weight_staging_slot_bytes;
        if (!slot || staged.max_features != 2048 || capped.max_features != 2048 ||
            staged.encoder.weight_staging_bytes != 2 * slot ||
            staged.encoder.workspace_bytes != capped.encoder.workspace_bytes + 2 * slot ||
            staged.device_bytes() > budget)
            throw std::runtime_error("Vision weight slots were omitted from the activation budget");
        bool rejected = false;
        try {
            (void)ninfer::models::qwen::VisionContext::plan_resources(
                *staged_profile->config.vision, *staged_profile->vision, 2048, 2 * slot);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected)
            throw std::runtime_error("Vision accepted a budget exhausted by staged weights");
        std::cout << "Vision bounded workspace/output: " << full.encoder.workspace_bytes << "/"
                  << full.output_bytes << " bytes; features=" << full.max_features << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
