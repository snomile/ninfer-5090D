#pragma once

#include "models/common/weights.h"
#include "artifact/schema.h"
#include "core/layout.h"
#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <array>
#include <vector>

namespace ninfer::models {
class Bindings;
}

namespace ninfer::models::qwen {

struct VisionConfig {
    std::uint32_t depth                   = 0;
    std::uint32_t hidden_size             = 0;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t num_heads               = 0;
    std::uint32_t patch_size              = 0;
    std::uint32_t temporal_patch_size     = 0;
    std::uint32_t spatial_merge_size      = 0;
    std::uint32_t num_position_embeddings = 0;
    std::uint32_t position_grid_side      = 0;
    [[nodiscard]] std::uint64_t patch_width() const;
    [[nodiscard]] std::uint64_t merger_width() const;
};

struct NormWeights {
    WeightId weight, bias;
};

struct VisionBlockWeights {
    NormWeights norm1, norm2;
    WeightId query, key, value, query_bias, key_bias, value_bias;
    WeightId output, output_bias;
    WeightId fc1, fc1_bias, fc2, fc2_bias;
};

struct VisionWeights {
    WeightId patch_embedding, patch_embedding_bias, position_embedding;
    std::vector<VisionBlockWeights> layers;
    NormWeights merger_norm;
    WeightId merger_fc1, merger_fc1_bias, merger_fc2, merger_fc2_bias;
};

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    ops::SingleProjectionWeight qkv;
    Tensor qkv_bias;
    ops::SingleProjectionWeight output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    bool host_staged = false;
    std::size_t weight_staging_slot_bytes = 0;
    ops::SingleProjectionWeight patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    ops::SingleProjectionWeight merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

struct VisionBlockProfile {
    ops::LinearProfile qkv, output, fc1, fc2;
};

struct VisionProfile {
    ops::LinearProfile patch_embedding, merger_fc1, merger_fc2;
    std::vector<VisionBlockProfile> layers;
    bool host_staged = false;
    std::size_t weight_staging_slot_bytes = 0;
};

[[nodiscard]] VisionWeights bind_vision(Bindings&, const VisionConfig&,
                                        std::uint32_t output_hidden);
[[nodiscard]] VisionParameters prepare_vision(std::span<const BoundWeight>, const VisionWeights&);

template <class Input>
[[nodiscard]] VisionProfile describe_vision(const VisionWeights& weights, Input&& input,
                                           bool host_staged = false) {
    const auto linear = [&](WeightId id) {
        const auto value = input(id);
        return ops::describe_linear_weight({&value, 1});
    };
    VisionProfile result{linear(weights.patch_embedding),
                         linear(weights.merger_fc1),
                         linear(weights.merger_fc2),
                         {}};
    for (const auto& layer : weights.layers) {
        const std::array inputs{input(layer.query), input(layer.key), input(layer.value)};
        result.layers.push_back({ops::describe_linear_weight(inputs), linear(layer.output),
                                 linear(layer.fc1), linear(layer.fc2)});
    }
    result.host_staged = host_staged;
    if (host_staged) {
        LayoutBuilder phase;
        const auto matrix = [&](WeightId id) {
            const auto value = input(id);
            const auto region = contiguous_weight_region(value.weight);
            (void)phase.add(region.parent->geometry.bytes, 256, "Vision staged payload");
        };
        const auto tensor = [&](WeightId id) {
            const auto value = input(id);
            (void)phase.add(artifact::checked_mul(weight_element_count(value.weight.shape), 2,
                                                 "Vision staged tensor"),
                            256, "Vision staged tensor");
        };
        const auto norm = [&](NormWeights n) { tensor(n.weight); tensor(n.bias); };
        const auto finish = [&] {
            result.weight_staging_slot_bytes =
                std::max(result.weight_staging_slot_bytes, phase.finish(256));
            phase = LayoutBuilder{};
        };
        matrix(weights.patch_embedding);
        tensor(weights.patch_embedding_bias);
        tensor(weights.position_embedding);
        finish();
        for (const auto& layer : weights.layers) {
            norm(layer.norm1);
            norm(layer.norm2);
            const std::array qkv{input(layer.query), input(layer.key), input(layer.value)};
            WeightView combined{{0, qkv[0].weight.shape[1]}, {}};
            for (const auto& value : qkv) {
                combined.shape[0] += value.weight.shape[0];
                combined.parts.insert(combined.parts.end(), value.weight.parts.begin(),
                                       value.weight.parts.end());
            }
            (void)phase.add(contiguous_weight_region(combined).parent->geometry.bytes, 256,
                            "Vision staged QKV payload");
            std::size_t bias_bytes = 0;
            for (auto id : {layer.query_bias, layer.key_bias, layer.value_bias})
                bias_bytes += weight_element_count(input(id).weight.shape) * 2;
            (void)phase.add(bias_bytes, 256, "Vision staged QKV bias");
            matrix(layer.output); tensor(layer.output_bias);
            matrix(layer.fc1); tensor(layer.fc1_bias);
            matrix(layer.fc2); tensor(layer.fc2_bias);
            finish();
        }
        norm(weights.merger_norm);
        matrix(weights.merger_fc1); tensor(weights.merger_fc1_bias); finish();
        matrix(weights.merger_fc2); tensor(weights.merger_fc2_bias); finish();
    }
    return result;
}

} // namespace ninfer::models::qwen
