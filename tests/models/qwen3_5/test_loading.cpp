#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/load/bindings.h"

#include <algorithm>
#include <array>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen = ninfer::models::qwen3_5;
using namespace ninfer::test::artifact_fixture;

Json added_token(int id, const std::string& content) {
    return {{"id", id},        {"content", content}, {"special", true},    {"single_word", false},
            {"lstrip", false}, {"rstrip", false},    {"normalized", false}};
}

struct ModelFixture {
    Fixture file;

    void tensor(const std::string& id, std::vector<std::uint64_t> shape, QType format = QType::BF16,
                QuantLayout layout = QuantLayout::Contiguous) {
        const auto geometry = weight_geometry(format, layout, shape);
        const auto offset   = (file.payload.size() + 255) / 256 * 256;
        file.payload.resize(offset + geometry.bytes);
        if (format == QType::NVFP4) {
            put_word(file.payload, offset + geometry.divisor_offset, 0x3f800000, 4);
        }
        file.root["objects"].push_back({{"id", id},
                                        {"kind", "tensor"},
                                        {"shape", shape},
                                        {"format", artifact::format_name(format)},
                                        {"layout", artifact::layout_name(layout)},
                                        {"offset", offset},
                                        {"bytes", geometry.bytes}});
    }

    void resource(const std::string& role, const std::string& bytes) {
        const auto offset = file.payload.size();
        for (const auto byte : bytes) { file.payload.push_back(std::byte(byte)); }
        file.root["objects"].push_back({{"id", role},
                                        {"kind", "resource"},
                                        {"encoding", "raw_bytes_v1"},
                                        {"offset", offset},
                                        {"bytes", bytes.size()}});
        file.root["components"]["text"]["resources"][role] = role;
    }

    void use(const std::string& name, const std::string& input, std::optional<int> divisor = {}) {
        Json value{{"parameter", name}, {"input", input}, {"activation_policy", "AllowA4"}};
        if (divisor) {
            value["auxiliaries"] = {
                {"activation_input_divisor",
                 {{"parts", Json::array({{{"object", "calibration"},
                                          {"range", {*divisor, *divisor + 1}}}})}}}};
        }
        file.root["uses"].push_back(value);
    }

    void parameter(const std::string& name, std::vector<std::uint64_t> shape,
                   const std::string& input = {}) {
        tensor(name, shape);
        file.root["bindings"][name] = {{"object", name}};
        if (!input.empty()) { use(name, input); }
    }

    ModelFixture() {
        file.payload.clear();
        file.root = {{"components",
                      {{"text",
                        {{"config",
                          {{"architectures", {"Qwen3_5ForCausalLM"}},
                           {"model_type", "qwen3_5_text"},
                           {"hidden_size", 128},
                           {"vocab_size", 272},
                           {"num_hidden_layers", 1},
                           {"max_position_embeddings", 128},
                           {"tie_word_embeddings", true},
                           {"rms_norm_eps", 1e-6},
                           {"layer_types", {"full_attention"}},
                           {"num_attention_heads", 2},
                           {"num_key_value_heads", 1},
                           {"head_dim", 8},
                           {"intermediate_size", 24},
                           {"rope_parameters",
                            {{"rope_theta", 10000},
                             {"partial_rotary_factor", 0.5},
                             {"mrope_section", {1, 1, 0}}}}}}}}}},
                     {"objects", Json::array()},
                     {"bindings", Json::object()},
                     {"uses", Json::array()},
                     {"metadata", {{"name", "a-user-trained-model"}}}};
        tensor("embedding-storage", {272, 128});
        file.root["bindings"]["text/token_embedding"] = {{"object", "embedding-storage"}};
        file.root["bindings"]["text/output_head"]     = {{"object", "embedding-storage"}};
        use("text/output_head", "text/final_hidden");
        parameter("text/final_norm", {128});
        const std::string p = "text/layers/0/";
        parameter(p + "input_norm", {128});
        parameter(p + "post_attention_norm", {128});
        tensor("mixer-storage", {128, 128}, QType::NVFP4, QuantLayout::BlockScaleK16M128x4);
        tensor("calibration", {2}, QType::FP32);
        const auto calibration = file.root["objects"].back()["offset"].get<std::size_t>();
        put_word(file.payload, calibration, 0x40000000, 4);
        put_word(file.payload, calibration + 4, 0x40400000, 4);
        const std::array roles = {"query", "key", "gate", "value"};
        const std::array rows  = {0, 16, 24, 40, 48};
        for (std::size_t i = 0; i < roles.size(); ++i) {
            const auto name             = p + "attention/" + roles[i];
            file.root["bindings"][name] = {
                {"parts", Json::array({{{"object", "mixer-storage"},
                                        {"range", {rows[i] * 128, rows[i + 1] * 128}}}})}};
            use(name, p + "mixer_input", i < 2 ? 0 : 1);
        }
        parameter(p + "attention/query_norm", {8});
        parameter(p + "attention/key_norm", {8});
        parameter(p + "attention/output", {128, 16}, p + "attention/gated_output");
        parameter(p + "mlp/gate", {24, 128}, p + "ffn_input");
        parameter(p + "mlp/up", {24, 128}, p + "ffn_input");
        tensor("down-storage", {128, 24}, QType::Q6_G64_FP16, QuantLayout::RowSplit);
        file.root["bindings"][p + "mlp/down"] = {{"object", "down-storage"}};
        use(p + "mlp/down", p + "mlp/product");

        Json vocab    = Json::object();
        unsigned next = 256;
        for (unsigned byte = 0; byte < 256; ++byte) {
            const bool visible =
                (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
            const unsigned cp = visible ? byte : next++;
            std::string token;
            if (cp < 128) {
                token.push_back(static_cast<char>(cp));
            } else {
                token.push_back(static_cast<char>(0xc0 | (cp >> 6)));
                token.push_back(static_cast<char>(0x80 | (cp & 63)));
            }
            vocab[token] = byte;
        }
        Json added                = Json::array();
        Json decoder              = Json::object();
        const std::array specials = {"<|endoftext|>",  "<|im_start|>",  "<|im_end|>",
                                     "<think>",        "</think>",      "<|vision_start|>",
                                     "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"};
        for (std::size_t i = 0; i < specials.size(); ++i) {
            const auto token = added_token(static_cast<int>(256 + i), specials[i]);
            if (i < 8) { added.push_back(token); }
            decoder[std::to_string(256 + i)] = token;
        }
        resource("tokenizer.json",
                 Json{{"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", Json::array()}}},
                      {"added_tokens", added}}
                     .dump());
        resource("tokenizer_config.json", Json{{"added_tokens_decoder", decoder}}.dump());
        resource("generation_config.json", Json{{"eos_token_id", {256, 258}}}.dump());
        resource("chat_template.jinja", "a user-provided template, preserved as data");
        file.root["files"] =
            Json::array({{{"path", nullptr}, {"payload_bytes", file.payload.size()}}});
    }
};

void logical_data_and_instances() {
    ModelFixture fixture;
    fixture.file.root["components"]["vision"] = {{"config", Json::object()}, {"target", "text"}};
    fixture.file.root["components"]["text"]["config"]["rope_parameters"]["partial_rotary_factor"] =
        0.4999999999;
    fixture.file.write();
    artifact::Reader reader(fixture.file.entry);
    auto plan = qwen::plan_load(reader);
    require(plan.config().text.rope_parameters->partial_rotary_factor == 0.5F &&
                plan.config().text.rope_parameters->rotary_dim == 4,
            "rotary width was derived before normalizing the PositiveF32 config value");
    require(plan.config().text.architecture == Architecture::Qwen3_5 &&
                plan.config().text.hidden_size == 128,
            "binding chose a checkpoint-specific geometry");
    require(plan.resources().public_token_count == 265 &&
                plan.resources().tokenizer->encode("<|video_pad|>") == std::vector<int>{264},
            "tokenizer_config added tokens were not included in the public domain");
    const auto& attention = std::get<qwen::AttentionWeights>(plan.weights().text.layers[0].mixer);
    const auto& query     = plan.parameter(attention.query);
    const auto& gate      = plan.parameter(attention.gate);
    require(query.binding.parts[0].object == gate.binding.parts[0].object &&
                query.binding.parts[0].begin == 0 && gate.binding.parts[0].begin == 24 * 128,
            "Q/gate logical row correspondence changed");
    require(plan.uses(attention.query)[0].activation_input_divisor == 2.0F &&
                plan.uses(attention.gate)[0].activation_input_divisor == 3.0F,
            "shared-parent Uses contaminated one another");
    require(plan.parameter(plan.weights().text.token_embedding).binding.parts[0].object ==
                plan.parameter(plan.weights().text.output_head).binding.parts[0].object,
            "explicit shared embedding/head did not bind the same parent");
    const auto capacity = plan.materialization().device_capacity_bytes;
    rejects([&] { (void)qwen::plan_load(reader, {.host_mapped_embedding = true}); },
            "shared embedding/output-head parent accepted incompatible mapped residency");
    rejects([&] { (void)qwen::plan_load(reader, {.host_pinned_vision_weights = true}); },
            "Host-pinned Vision weights were accepted without Vision");
    rejects([&] { (void)qwen::plan_load(reader, {.host_mapped_dflash2_codebooks = true}); },
            "Host-mapped DFlash2 codebooks were accepted without DFlash2");
    require(!plan.weights().vision && plan.resources().preprocessor_config_json.empty(),
            "unused Vision was required");
    rejects([&] { (void)qwen::plan_load(reader, {.vision = true}); },
            "incomplete selected Vision was accepted");
    rejects([&] { (void)qwen::plan_load(reader, {.speculative = SpeculativeBackend::Mtp}); },
            "absent MTP was accepted");

    fixture.file.root["metadata"]["name"] = "another-training-run-and-recipe";
    fixture.file.root["components"]["text"]["config"]["num_hidden_layers"] = 2;
    fixture.file.root["components"]["text"]["config"]["layer_types"]       = {"full_attention",
                                                                              "full_attention"};
    const auto original_bindings = fixture.file.root["bindings"];
    for (const auto& [name, binding] : original_bindings.items()) {
        if (name.starts_with("text/layers/0/")) {
            auto second = name;
            second.replace(12, 1, "1");
            fixture.file.root["bindings"][second] = binding;
        }
    }
    const auto original_uses = fixture.file.root["uses"];
    for (auto use : original_uses) {
        auto parameter = use["parameter"].get<std::string>();
        if (parameter.starts_with("text/layers/0/")) {
            auto input = use["input"].get<std::string>();
            parameter.replace(12, 1, "1");
            input.replace(12, 1, "1");
            use["parameter"] = parameter;
            use["input"]     = input;
            fixture.file.root["uses"].push_back(use);
        }
    }
    fixture.file.write();
    artifact::Reader changed(fixture.file.entry);
    auto expanded = qwen::plan_load(changed);
    require(expanded.weights().text.layers.size() == 2 &&
                expanded.config().text.compact_layer_indices == std::vector<std::uint32_t>{0, 1},
            "config did not drive layer geometry and compact indices");
    require(expanded.materialization().device_capacity_bytes == capacity,
            "cross-layer sharing duplicated parent storage");
}

void native_uses() {
    const std::array<std::uint64_t, 2> shape{128, 64};
    const auto geometry = weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, shape);
    std::vector<std::byte> bytes(geometry.bytes);
    const WeightParent parent{geometry, bytes.data(), 8.0F};
    const WeightView gate{{64, 64}, {{&parent, 0, 4096}}};
    const WeightView up{{64, 64}, {{&parent, 4096, 8192}}};
    const auto joined = ops::prepare_linear_swiglu_weight({gate, ops::LinearPolicy::AllowA4, 2.0F},
                                                          {up, ops::LinearPolicy::AllowA8, 2.0F});
    require(joined.policy == ops::LinearPolicy::AllowA8 && joined.weight.payload == bytes.data() &&
                joined.weight.n == 128 && joined.weight.weight_scale_divisor == 8.0F &&
                joined.weight.input_scale_divisor == 2.0F,
            "shared quantization did not intersect Uses or preserve the complete parent");
    const WeightView full{{128, 64}, {{&parent, 0, 8192}}};
    for (const auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
        const auto no_aux = ops::prepare_linear_weight({full, policy});
        require(no_aux.policy == policy && no_aux.weight.payload == bytes.data(),
                "non-A4 NVFP4 use required an unused activation divisor");
        const auto mixed = ops::prepare_linear_swiglu_weight(
            {gate, ops::LinearPolicy::AllowA4, 2.0F}, {up, policy});
        require(mixed.policy == policy, "combined use required an unused child auxiliary");
        rejects<std::invalid_argument>(
            [&] { (void)ops::prepare_linear_weight({full, policy, 0.0F}); },
            "present activation divisor was accepted without value validation");
    }
    rejects<std::invalid_argument>(
        [&] { (void)ops::prepare_linear_weight({full, ops::LinearPolicy::AllowA4}); },
        "A4 NVFP4 use accepted a missing activation divisor");
    const auto a16 = ops::prepare_linear_swiglu_weight({gate, ops::LinearPolicy::A16Only, 2.0F},
                                                       {up, ops::LinearPolicy::AllowA4, 3.0F});
    require(a16.policy == ops::LinearPolicy::A16Only && a16.weight.input_scale_divisor == 2.0F,
            "A16-only use intersection rejected independent unused activation divisors");
    const auto second = ops::prepare_linear_weight({full, ops::LinearPolicy::A16Only, 3.0F});
    const auto first  = ops::prepare_linear_weight({full, ops::LinearPolicy::AllowA4, 2.0F});
    require(first.weight.input_scale_divisor == 2.0F && second.weight.input_scale_divisor == 3.0F &&
                first.policy == ops::LinearPolicy::AllowA4 &&
                second.policy == ops::LinearPolicy::A16Only && parent.weight_scale_divisor == 8.0F,
            "preparing one use modified another use or its shared parent");
    rejects<std::invalid_argument>(
        [&] {
            (void)ops::prepare_linear_swiglu_weight({gate, ops::LinearPolicy::AllowA4, 2.0F},
                                                    {up, ops::LinearPolicy::AllowA4, 3.0F});
        },
        "one native activation silently used differing divisors");
    rejects<std::invalid_argument>(
        [&] {
            (void)ops::prepare_linear_swiglu_weight({up, ops::LinearPolicy::A16Only, 2.0F},
                                                    {gate, ops::LinearPolicy::A16Only, 2.0F});
        },
        "reordered parent regions silently changed fused gate/up row order");
    const WeightView short_gate{{64, 64}, {{&parent, 0, 4032}}};
    const WeightView long_up{{64, 64}, {{&parent, 4032, 8192}}};
    rejects<std::invalid_argument>(
        [&] {
            (void)ops::prepare_linear_swiglu_weight({short_gate, ops::LinearPolicy::A16Only, 2.0F},
                                                    {long_up, ops::LinearPolicy::A16Only, 2.0F});
        },
        "combined coverage hid an incorrect gate/up boundary");
}

void optional_host_residencies() {
    ModelFixture fixture;
    const qwen::VisionConfig vision{
        .depth                   = 1,
        .hidden_size             = 8,
        .intermediate_size       = 12,
        .num_heads               = 1,
        .patch_size              = 2,
        .temporal_patch_size     = 1,
        .spatial_merge_size      = 2,
        .num_position_embeddings = 4,
        .position_grid_side      = 2,
    };
    const std::uint64_t h = vision.hidden_size;
    fixture.parameter("vision/patch_embedding", {h, vision.patch_width()}, "vision/patch_input");
    fixture.parameter("vision/patch_embedding_bias", {h});
    fixture.parameter("vision/position_embedding", {vision.num_position_embeddings, h});
    const std::string p = "vision/layers/0/";
    for (const auto* name : {"norm1_weight", "norm1_bias", "norm2_weight", "norm2_bias"}) {
        fixture.parameter(p + name, {h});
    }
    for (const auto* name : {"query", "key", "value"}) {
        fixture.parameter(p + "attention/" + name, {h, h}, p + "attention_input");
    }
    for (const auto* name : {"query_bias", "key_bias", "value_bias"}) {
        fixture.parameter(p + "attention/" + name, {h});
    }
    fixture.parameter(p + "attention/output", {h, h}, p + "attention_output");
    fixture.parameter(p + "attention/output_bias", {h});
    fixture.parameter(p + "mlp/fc1", {vision.intermediate_size, h}, p + "mlp_input");
    fixture.parameter(p + "mlp/fc1_bias", {vision.intermediate_size});
    fixture.parameter(p + "mlp/fc2", {h, vision.intermediate_size}, p + "mlp_activation");
    fixture.parameter(p + "mlp/fc2_bias", {h});
    fixture.parameter("vision/merger/norm_weight", {h});
    fixture.parameter("vision/merger/norm_bias", {h});
    fixture.parameter("vision/merger/fc1", {vision.merger_width(), vision.merger_width()},
                      "vision/merger/input");
    fixture.parameter("vision/merger/fc1_bias", {vision.merger_width()});
    fixture.parameter("vision/merger/fc2", {128, vision.merger_width()},
                      "vision/merger/activation");
    fixture.parameter("vision/merger/fc2_bias", {128});

    qwen::DraftConfig draft;
    draft.num_hidden_layers = 1;
    draft.dflash2           = qwen::DFlash2Config{
                  .conv_kernel_size = 2, .conv_group_size = 16, .selector_rank = 4, .selector_top_k = 2};
    const std::uint64_t rows =
        2ULL * draft.dflash2->conv_kernel_size * 128 / draft.dflash2->conv_group_size;
    for (const auto* name : {"attention_conv", "mlp_conv"}) {
        fixture.parameter("dflash2/layers/0/" + std::string(name) + "/base_kernel",
                          {2, draft.dflash2->conv_kernel_size, 128});
        fixture.parameter("dflash2/layers/0/" + std::string(name) + "/kernel_projection",
                          {rows, 128}, "dflash2/layers/0/" + std::string(name) + "_input");
    }
    fixture.parameter("dflash2/candidate_selector/hidden_projection",
                      {draft.dflash2->selector_rank, 128}, "dflash2/final_hidden");
    fixture.parameter("dflash2/candidate_selector/predecessor_codebook",
                      {272, draft.dflash2->selector_rank});
    fixture.parameter("dflash2/candidate_selector/successor_codebook",
                      {272, draft.dflash2->selector_rank});
    fixture.file.root["files"][0]["payload_bytes"] = fixture.file.payload.size();
    fixture.file.write();

    artifact::Reader reader(fixture.file.entry);
    artifact::Binder default_vision_binder(reader);
    qwen::loading::Bindings default_vision_bindings(default_vision_binder);
    (void)qwen::loading::bind_vision(default_vision_bindings, vision,
                                     qwen::TextConfig{.hidden_size = 128}, {});
    for (const auto& weight : default_vision_bindings.weights) {
        require(weight.reference.residency == artifact::Residency::Device,
                "default Vision selection changed weight residency");
    }
    const auto default_vision_count = default_vision_bindings.weights.size();
    const auto default_vision_plan  = std::move(default_vision_binder).finish();
    require(default_vision_plan.pinned_host_objects.empty() &&
                default_vision_plan.device_objects.size() == default_vision_count,
            "default Vision selection no longer uses Device parents");

    artifact::Binder vision_binder(reader);
    qwen::loading::Bindings vision_bindings(vision_binder);
    const auto vision_weights =
        qwen::loading::bind_vision(vision_bindings, vision, qwen::TextConfig{.hidden_size = 128},
                                   {.vision = true, .host_pinned_vision_weights = true});
    (void)vision_weights;
    for (const auto& weight : vision_bindings.weights) {
        require(weight.reference.residency == artifact::Residency::HostPinned,
                "Host-pinned Vision selection left a Device parameter");
    }
    const auto vision_count = vision_bindings.weights.size();
    const auto vision_plan  = std::move(vision_binder).finish();
    require(vision_plan.device_objects.empty() &&
                vision_plan.pinned_host_objects.size() == vision_count,
            "Host-pinned Vision selection retained Device parents");

    artifact::Binder dflash_binder(reader);
    qwen::loading::Bindings dflash_bindings(dflash_binder);
    qwen::DraftWeights draft_weights;
    draft_weights.layers.resize(1);
    qwen::TextConfig target;
    target.hidden_size = 128;
    target.vocab_size  = 272;
    artifact::Binder default_dflash_binder(reader);
    qwen::loading::Bindings default_dflash_bindings(default_dflash_binder);
    qwen::DraftWeights default_draft_weights;
    default_draft_weights.layers.resize(1);
    qwen::loading::bind_dflash2(default_dflash_bindings, default_draft_weights, draft, target, {});
    require(default_draft_weights.selector &&
                default_dflash_bindings.at(default_draft_weights.selector->predecessor_codebook)
                        .reference.residency == artifact::Residency::Device &&
                default_dflash_bindings.at(default_draft_weights.selector->successor_codebook)
                        .reference.residency == artifact::Residency::Device,
            "default DFlash2 selector codebooks changed residency");
    const auto default_dflash_plan = std::move(default_dflash_binder).finish();
    require(default_dflash_plan.mapped_host_objects.empty() &&
                default_dflash_plan.device_objects.size() == default_dflash_bindings.weights.size(),
            "default DFlash2 selection retained mapped Host parents");

    qwen::loading::bind_dflash2(
        dflash_bindings, draft_weights, draft, target,
        {.host_mapped_dflash2_codebooks = true, .speculative = SpeculativeBackend::DFlash2});
    require(
        draft_weights.selector &&
            dflash_bindings.at(draft_weights.selector->predecessor_codebook).reference.residency ==
                artifact::Residency::HostMapped &&
            dflash_bindings.at(draft_weights.selector->successor_codebook).reference.residency ==
                artifact::Residency::HostMapped &&
            dflash_bindings.at(draft_weights.selector->hidden_projection).reference.residency ==
                artifact::Residency::Device,
        "DFlash2 Host-mapped selection changed the wrong selector parameters");
    const auto dflash_plan = std::move(dflash_binder).finish();
    require(dflash_plan.mapped_host_objects.size() == 2 &&
                dflash_plan.device_objects.size() + 2 == dflash_bindings.weights.size(),
            "DFlash2 codebook selection did not isolate two mapped Host parents");
}

void invalid_model_data() {
    ModelFixture fixture;
    const Json original = fixture.file.root;
    const auto bad      = [&](auto mutate) {
        fixture.file.root = original;
        mutate(fixture.file.root);
        fixture.file.write();
        rejects(
            [&] {
                artifact::Reader reader(fixture.file.entry);
                (void)qwen::plan_load(reader);
            },
            "invalid model data was accepted");
    };
    bad([](Json& root) { root["components"]["text"]["config"]["layer_types"] = Json::array(); });
    bad([](Json& root) { root["components"]["text"]["config"]["num_key_value_heads"] = 3; });
    bad([](Json& root) {
        root["components"]["text"]["config"]["rope_parameters"]["mrope_section"] = {0, 2, 0};
    });
    bad([](Json& root) {
        root["bindings"]["text/layers/0/attention/query"]["parts"][0]["range"] = {0, 8 * 128};
    });
    bad([](Json& root) { root["uses"][1].erase("activation_policy"); });
    bad([](Json& root) { root["components"]["text"]["config"]["vocab_size"] = 264; });
}

} // namespace

int main() {
    try {
        logical_data_and_instances();
        native_uses();
        optional_host_residencies();
        invalid_model_data();
        std::cout << "model config, binding, resources and Use checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
