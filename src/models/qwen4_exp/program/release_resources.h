#pragma once

#include "runtime/contract/resources.h"

namespace ninfer::models::qwen4_exp::detail {

template <std::size_t Pools>
[[nodiscard]] bool release_resources_useful(const runtime::ContextResources<Pools>& released,
                                            const runtime::ContextResources<Pools>& shortage) noexcept {
    if (shortage.state_slots || shortage.first_kv_deficit()) return released.useful(shortage) != 0;
    // Host admission needs the union of actions that release either requested dimension.
    return (shortage.host_bytes && released.host_bytes) ||
           (shortage.host_state_slots && released.host_state_slots);
}

} // namespace ninfer::models::qwen4_exp::detail
