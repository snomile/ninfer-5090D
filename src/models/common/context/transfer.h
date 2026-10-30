#pragma once

#include "core/buffer.h"
#include "core/transfer.h"
#include "models/common/context/storage.h"

#include <span>
#include <vector>

namespace ninfer::distributed {
class Group;
}

namespace ninfer::models::context {

void prepare_peer_communication(const distributed::Group&, std::size_t chunk_bytes,
                                CudaExecutionView);

// Borrowed startup resources. Every group member uses the same chunk_bytes; staging covers
// one chunk and is private to this slot. Members map group coordinates to logical ranks.
struct ContextPeerBinding {
    const distributed::Group* group = nullptr;
    std::span<const std::uint32_t> members;
    MemorySpan staging;
    std::size_t chunk_bytes = 0;
};

// One rank-local action slot. Its event and pinned mapping storage are fixed at startup. Compute
// and independent context migration use separate slots; neither owns directory reservations.
class ContextMigration {
public:
    ContextMigration(DeviceContext& device, ContextStorage& storage,
                     std::size_t mapping_capacity_elements = 0,
                     std::size_t copy_segment_capacity     = 0);
    ContextMigration(DeviceContext& device, ContextStorage& storage, CudaExecutionView execution,
                     std::size_t mapping_capacity_elements = 0,
                     std::size_t copy_segment_capacity = 0, ContextPeerBinding peer = {});
    ~ContextMigration();
    ContextMigration(const ContextMigration&)            = delete;
    ContextMigration& operator=(const ContextMigration&) = delete;

    void validate(const ContextActions& actions) const;

    // Transfer-stream work inherits all compute already submitted at this boundary.
    // A nonzero byte window bounds RecordCopy work on an independent stream. The caller
    // retains actions until finish/drain and calls advance to submit each completed window.
    // The final completion fence, observations and table order still cover the entire plan.
    void submit(const ContextActions& actions, std::size_t copy_window_bytes = 0);
    void advance();
    [[nodiscard]] bool ready() const;

    [[nodiscard]] bool occupied() const noexcept { return occupied_; }

    [[nodiscard]] std::size_t pinned_bytes() const noexcept { return mapping_.size(); }

    [[nodiscard]] std::span<const runtime::ContextTransferObservation> observations();
    void finish();
    void drain();

private:
    struct Segment {
        runtime::ContextTransferObservation observed;
        std::size_t begin = 0;
        std::size_t end   = 0;
    };

    void prepare(const ContextActions& actions);
    void enqueue(const ContextActions& actions, CudaExecutionView execution);
    void enqueue_window();
    [[nodiscard]] std::uint32_t enqueue_peer(const PeerCopy&, CudaExecutionView);

    DeviceContext* device_;
    ContextStorage* storage_;
    CudaExecutionView execution_;
    const distributed::Group* peer_group_ = nullptr;
    std::vector<std::uint32_t> peer_members_;
    MemorySpan peer_staging_;
    std::size_t peer_chunk_bytes_ = 0;
    PinnedHostBuffer mapping_;
    CudaCopyBatch copies_;
    CudaCompletionEvent source_ready_;
    CudaCompletionEvent completed_;
    std::vector<CudaEventTimer> timers_;
    std::vector<Segment> segments_;
    std::vector<runtime::ContextTransferObservation> observations_;
    cudaStream_t submitted_stream_ = nullptr;
    const ContextActions* pending_actions_ = nullptr;
    std::size_t copy_window_bytes_ = 0;
    std::size_t next_operation_ = 0;
    std::size_t copy_offset_ = 0;
    std::size_t mapping_offset_ = 0;
    std::size_t next_segment_ = 0;
    bool window_pending_ = false;
    bool occupied_                 = false;
    bool fenced_                   = false;
};

} // namespace ninfer::models::context
