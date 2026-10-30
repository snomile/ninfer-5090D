#include "models/qwen3_5/load.h"

#include "models/common/bindings.h"

namespace ninfer::models::qwen3_5 {

ResolvedLoadPlan::ResolvedLoadPlan(const artifact::Reader& reader,
                                   std::shared_ptr<const ModelDefinition> definition,
                                   artifact::MaterializationPlan materialization,
                                   std::vector<Device> devices, std::vector<Device> bank_devices)
    : reader_(&reader), definition_(std::move(definition)),
      materialization_(std::move(materialization)), devices_(std::move(devices)),
      bank_devices_(std::move(bank_devices)) {}

ResolvedLoadPlan resolve_load(const artifact::Reader& reader,
                              std::shared_ptr<const ModelDefinition> definition, Device cuda,
                              std::optional<Device> main_experts) {
    if (!definition || definition->artifact_id() != reader.artifact_id())
        throw artifact::ArtifactError("model definition belongs to another artifact");
    const auto expert_device = main_experts.value_or(cuda);
    if (cuda.type() != DeviceType::CUDA ||
        (expert_device != cuda && expert_device != Device::cpu()))
        throw artifact::ArtifactError("weight placement requires the selected GPU or CPU");
    std::vector<Device> devices(definition->weight_data().size(), cuda);
    std::vector<Device> banks(definition->expert_banks().size(), cuda);
    const auto& options = definition->options();
    const auto& weights = definition->weights();
    std::vector<bool> mapped(devices.size());
    if (options.host_mapped_embedding) {
        mapped[weights.text.token_embedding.index] = true;
        if (weights.mtp) mapped[weights.mtp->token_embedding.index] = true;
        if (weights.draft) mapped[weights.draft->token_embedding.index] = true;
    }
    if (options.host_mapped_dflash2_codebooks) {
        if (!options.dflash2() || !weights.draft || !weights.draft->selector)
            throw artifact::ArtifactError("mapped codebooks require the DFlash2 selector");
        mapped[weights.draft->selector->predecessor_codebook.index] = true;
        mapped[weights.draft->selector->successor_codebook.index] = true;
    }
    if (options.host_pinned_vision_weights) {
        if (!weights.vision)
            throw artifact::ArtifactError("pinned Vision weights require Vision");
        for (std::size_t i = 0; i < devices.size(); ++i)
            if (definition->parameter({i}).name.starts_with("vision/"))
                devices[i] = Device::cpu();
    }
    for (const auto& layer : definition->weights().text.layers) {
        if (const auto* moe = std::get_if<MoeWeights>(&layer.ffn)) {
            banks[moe->bank.index] = expert_device;
            for (const auto& expert : moe->experts)
                for (const auto id : {expert.gate, expert.up, expert.down})
                    devices[id.index] = expert_device;
        }
    }
    std::vector<artifact::PlacementRequest> requests;
    for (std::size_t i = 0; i < devices.size(); ++i)
        for (const auto& part : definition->parameter({i}).binding.parts)
            requests.push_back({part.object, devices[i],
                                mapped[i] ? artifact::HostAccess::CudaMapped
                                : devices[i] == Device::cpu() ? artifact::HostAccess::CudaTransfer
                                                              : artifact::HostAccess::CpuOnly});
    auto storage = artifact::plan_materialization(reader, requests);
    return {reader, std::move(definition), std::move(storage), std::move(devices),
            std::move(banks)};
}

std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&& plan, DeviceContext& device,
                                              const StartupObserver* observer) {
    if (!plan.definition_) throw artifact::ArtifactError("load plan was already consumed");
    auto definition = std::move(plan.definition_);
    auto backing    = artifact::materialize(*plan.reader_, std::move(plan.materialization_),
                                            device.transfer_view(), observer);
    std::vector<PendingWeight> pending;
    pending.reserve(definition->weight_data().size());
    for (std::size_t i = 0; i < definition->weight_data().size(); ++i) {
        const auto& description = definition->weight_data()[i];
        pending.push_back(
            {definition->parameter({i}), description.uses, description.source_objects});
    }
    auto bound = resolve_weights(std::move(pending), backing, plan.devices_);
    auto banks = moe::bind_banks(definition->expert_banks(), backing, plan.bank_devices_);
    return std::unique_ptr<ModelShard>(new ModelShard(std::move(definition), std::move(backing),
                                                      std::move(bound), std::move(banks)));
}

} // namespace ninfer::models::qwen3_5
