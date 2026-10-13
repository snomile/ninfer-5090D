#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/device.h"
#include "models/qwen3_5/program/vision_control.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
class VisionContext;
class VisionPrefillSession;
} // namespace ninfer::models::qwen3_5::execution

namespace ninfer::models::qwen3_5::detail {

// One state per Program workspace. It owns handoff identity and, when enabled, the shared
// weight-staging slot events used by every request session.
class VisionHandoffState {
    friend class execution::VisionContext;
    friend class execution::VisionPrefillSession;

    const execution::VisionPrefillSession* owner_ = nullptr;
    std::uint64_t generation_                     = 0;
    std::array<std::optional<CudaCompletionEvent>, 2> staging_ready_;
    std::array<std::optional<CudaCompletionEvent>, 2> staging_consumed_;
};

struct VisionUseSpan {
    std::uint32_t begin               = 0;
    std::uint32_t end                 = 0;
    std::uint32_t prepared_item_index = 0;
    std::uint32_t control_index       = 0;
};

struct VisionPrefillPlan {
    std::shared_ptr<const qwen3_5::VisionControlPlan> control_plan;
    std::shared_ptr<const qwen3_5::VisionControl> control;
    std::vector<VisionUseSpan> uses;
    std::size_t max_merged_count = 0;
};

} // namespace ninfer::models::qwen3_5::detail
