#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {

VisionWeights bind_vision(Bindings& b, const VisionConfig& config, const TextConfig& target,
                          const LoadOptions& options) {
    const auto h            = config.hidden_size;
    const auto intermediate = config.intermediate_size;
    const auto residency    = options.host_pinned_vision_weights ? artifact::Residency::HostPinned
                                                                 : artifact::Residency::Device;
    const auto direct       = [&](std::string name, artifact::Shape shape) {
        return b.direct(std::move(name), std::move(shape), QType::BF16, residency);
    };
    const auto parameter = [&](std::string name, artifact::Shape shape,
                               std::vector<std::string> inputs) {
        return b.parameter(std::move(name), std::move(shape), std::move(inputs), {}, residency);
    };
    VisionWeights out;
    out.patch_embedding =
        parameter("vision/patch_embedding", {h, config.patch_width()}, {"vision/patch_input"});
    out.patch_embedding_bias = direct("vision/patch_embedding_bias", {h});
    out.position_embedding =
        direct("vision/position_embedding", {config.num_position_embeddings, h});
    out.layers.reserve(config.depth);
    for (std::uint32_t i = 0; i < config.depth; ++i) {
        const auto p = "vision/layers/" + std::to_string(i) + "/";
        VisionBlockWeights layer;
        layer.norm1       = {direct(p + "norm1_weight", {h}), direct(p + "norm1_bias", {h})};
        layer.norm2       = {direct(p + "norm2_weight", {h}), direct(p + "norm2_bias", {h})};
        layer.query       = parameter(p + "attention/query", {h, h}, {p + "attention_input"});
        layer.key         = parameter(p + "attention/key", {h, h}, {p + "attention_input"});
        layer.value       = parameter(p + "attention/value", {h, h}, {p + "attention_input"});
        layer.query_bias  = direct(p + "attention/query_bias", {h});
        layer.key_bias    = direct(p + "attention/key_bias", {h});
        layer.value_bias  = direct(p + "attention/value_bias", {h});
        layer.output      = parameter(p + "attention/output", {h, h}, {p + "attention_output"});
        layer.output_bias = direct(p + "attention/output_bias", {h});
        layer.fc1         = parameter(p + "mlp/fc1", {intermediate, h}, {p + "mlp_input"});
        layer.fc1_bias    = direct(p + "mlp/fc1_bias", {intermediate});
        layer.fc2         = parameter(p + "mlp/fc2", {h, intermediate}, {p + "mlp_activation"});
        layer.fc2_bias    = direct(p + "mlp/fc2_bias", {h});
        out.layers.push_back(layer);
    }
    const auto merger   = config.merger_width();
    out.merger_norm     = {direct("vision/merger/norm_weight", {h}),
                           direct("vision/merger/norm_bias", {h})};
    out.merger_fc1      = parameter("vision/merger/fc1", {merger, merger}, {"vision/merger/input"});
    out.merger_fc1_bias = direct("vision/merger/fc1_bias", {merger});
    out.merger_fc2 =
        parameter("vision/merger/fc2", {target.hidden_size, merger}, {"vision/merger/activation"});
    out.merger_fc2_bias = direct("vision/merger/fc2_bias", {target.hidden_size});
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
