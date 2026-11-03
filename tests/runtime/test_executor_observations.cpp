#include "runtime/executor/protocol.h"

#include <iostream>
#include <stdexcept>

namespace {
using namespace ninfer;
namespace wire = distributed::wire;
namespace protocol = runtime::protocol;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template<class F>
void rejects(F&& fn) {
    try { fn(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("truncated observation was accepted");
}

void snapshot() {
    RankSnapshot source;
    source.rank = 2;
    source.node = 7;
    source.memory.allocations.device_weights_bytes = 1000;
    source.memory.allocations.host_weights_bytes = 2000;
    source.memory.allocations.registered_weights_bytes = 1500;
    source.memory.allocations.mapped_weights_bytes = 700;
    auto& vision = source.memory.vision.emplace();
    vision.input_staging_bytes = 101;
    vision.output_active_bytes = 102;
    vision.output_peak_bytes = 103;
    vision.weight_staging_bytes = 104;
    wire::Encoder encoder;
    protocol::encode(encoder, source);
    const auto bytes = std::move(encoder).take();
    wire::Decoder decoder(bytes);
    const auto result = protocol::rank_snapshot(decoder);
    decoder.finish();
    require(result.rank == 2 && result.node == 7 &&
                result.memory.allocations.device_weights_bytes == 1000 &&
                result.memory.allocations.host_weights_bytes == 2000 &&
                result.memory.allocations.registered_weights_bytes == 1500 &&
                result.memory.allocations.mapped_weights_bytes == 700 &&
                result.memory.vision && result.memory.vision->input_staging_bytes == 101 &&
                result.memory.vision->output_active_bytes == 102 &&
                result.memory.vision->output_peak_bytes == 103 &&
                result.memory.vision->weight_staging_bytes == 104,
            "rank snapshot lost mapped, registered or staging fields");
    rejects([&] {
        wire::Decoder truncated(std::span(bytes).first(bytes.size() - 1));
        (void)protocol::rank_snapshot(truncated);
    });
}

void completion() {
    for (const bool present : {false, true}) {
        RankExecutionObservation source;
        source.path = ExecutionPath::Graph;
        if (present) {
            source.gpu = {101, 102, 103, 104, 105, 106};
            source.attention = {ExecutionKind::Decode, {2, 3, 4, 5}};
        }
        source.communication_graph.calls[0] = 17;
        source.communication_graph.payload_bytes[0] = 8192;
        wire::Encoder encoder;
        protocol::encode(encoder, source);
        const auto bytes = std::move(encoder).take();
        require(bytes.size() <= protocol::rank_execution_observation_capacity(),
                "rank completion exceeded its reserved wire capacity");
        wire::Decoder decoder(bytes);
        const auto result = protocol::rank_execution_observation(decoder);
        decoder.finish();
        require(result.path == ExecutionPath::Graph &&
                    result.communication_graph.calls[0] == 17 &&
                    result.communication_graph.payload_bytes[0] == 8192 &&
                    result.attention.has_value() == present,
                "completion changed graph work or optional attention coverage");
        for (const auto field : kGpuIntervalFields)
            require(result.gpu.*field == source.gpu.*field,
                    "GPU interval changed or lost its availability");
        if (present)
            require(result.attention->kind == ExecutionKind::Decode &&
                        result.attention->work.forwards == 2 &&
                        result.attention->work.rows == 3 &&
                        result.attention->work.valid_positions == 4 &&
                        result.attention->work.scheduled_positions == 5,
                    "completion changed attention work");
        rejects([&] {
            wire::Decoder truncated(std::span(bytes).first(bytes.size() - 1));
            (void)protocol::rank_execution_observation(truncated);
        });
    }
}
} // namespace

int main() {
    try {
        snapshot();
        completion();
        std::cout << "ok: rank memory and completion observation round trips\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
