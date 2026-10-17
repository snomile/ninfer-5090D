#include "core/device.h"
#include "models/common/context/migration.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace store = ninfer::models::context;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

void test_state_host_reservations(ninfer::DeviceContext& device) {
    using namespace ninfer;
    StateImageBuilder schema;
    (void)schema.add(DType::I32, {8}, 1, "state");
    LayoutBuilder builder;
    const auto layout = schema.finish(builder, 2);
    DeviceBuffer backing(device.device(), builder.finish());
    StateImageStorage physical(backing.span(), layout);
    HostContextArena arena(layout.image_stride * 8, layout.image_stride);
    store::StateImageStore images(physical, &arena, 4, 2);
    const auto check = [&](std::uint32_t slots, std::size_t reserved, std::size_t live,
                           std::string_view message) {
        expect(images.host_occupied() == slots && arena.reserved_bytes() == reserved &&
                   arena.live_bytes() == live && arena.occupied_bytes() == reserved + live,
               message);
    };
    const auto initialize = [&] {
        const auto image = images.reserve_destination().value();
        physical.clear(images.physical_slot(image), device.execution_view());
        images.complete_initialization(image);
        images.freeze(image);
        return image;
    };
    const auto first = initialize();
    const auto second = initialize();
    device.synchronize();
    const auto stride = layout.image_stride;
    expect(images.device_capacity() == 2 && images.host_capacity() == 2 &&
               images.capacity() == 4,
           "Host slot limit stays independent of larger shared byte capacity");

    {
        auto one = images.reserve_host_destination().value();
        auto two = images.reserve_host_destination().value();
        check(2, 2 * stride, 0, "pending tickets charge exactly one slot and extent each");
        expect(!images.can_reserve_host() && !images.reserve_host_destination() &&
                   !images.reserve_device_to_host(first) && arena.free_bytes() == 6 * stride,
               "pending Host slots deny over-admission even with ample backing bytes");
        store::StateImageStore::HostReservation moved(std::move(one));
        expect(!one.valid() && moved.valid(), "Host ticket move transfers sole ownership");
        moved = std::move(moved);
        check(2, 2 * stride, 0, "Host ticket self-move preserves its charge");
        moved = std::move(two);
        expect(!two.valid() && moved.valid(), "Host ticket assignment consumes its source");
        check(1, stride, 0, "Host ticket assignment returns its previous slot and bytes");
        moved.reset();
        moved.reset();
        check(0, 0, 0, "Host ticket reset is idempotent and returns capacity");
    }
    {
        auto ticket = images.reserve_host_destination().value();
        auto copy = images.reserve_device_to_host(first, std::move(ticket)).value();
        expect(!ticket.valid() && copy.valid() && images.source_pins(first) == 1 &&
                   !images.host_resident(first),
               "ticket hands its allocation to the pending copy without early publication");
        check(1, stride, 0, "ticket-to-transfer handoff retains one pending charge");
        expect(arena.allocation_count() == 1,
               "ticket handoff must not allocate a second Host extent");
        auto pending = images.reserve_host_destination().value();
        check(2, 2 * stride, 0, "capture ticket and pending DMA share the slot upper bound");
        expect(!images.reserve_host_destination(), "ticket plus DMA exhaust Host slots");
        copy.enqueue(device.transfer_view());
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
        copy.publish(true);
        expect(!copy.valid() && images.source_pins(first) == 0 &&
                   images.residency(first) == store::StateReplicaResidency::Both,
               "completed copy publishes Both residency and releases its source lease");
        check(2, stride, stride, "pending-to-live publication neither doubles nor loses a slot");
        pending.reset();
        check(1, 0, stride, "cancelling another ticket preserves the published image");
        auto reader = images.read(first);
        expect(!images.drop_host_replica(first), "read lease pins the real Host replica");
        reader.reset();
        expect(images.drop_host_replica(first), "unread redundant Host replica can retire");
        check(0, 0, 0, "redundant replica retirement returns its live slot and bytes");
    }
    for (const bool cancelled : {true, false}) {
        auto one = images.reserve_device_to_host(first).value();
        auto two = images.reserve_device_to_host(second).value();
        check(2, 2 * stride, 0, "independent pending copies count against the same Host cap");
        expect(!images.reserve_host_destination(), "in-flight copies prohibit another ticket");
        one.enqueue(device.transfer_view());
        two.enqueue(device.transfer_view());
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
        if (!cancelled) {
            one.publish(true);
            two.publish(true);
            check(2, 0, 2 * stride, "two completed copies retain two live slots");
            expect(images.drop_host_replica(first) && images.drop_host_replica(second),
                   "published replicas retire independently");
        }
    }
    check(0, 0, 0, "completed or cancelled transfers leave no pending slot charge");
    expect(images.source_pins(first) == 0 && images.source_pins(second) == 0,
           "cancelled transfer destruction releases both source leases");
    {
        auto copy = images.reserve_device_to_host(first).value();
        auto duplicate = images.reserve_host_destination().value();
        expect(!images.reserve_device_to_host(first, std::move(duplicate)),
               "a second copy cannot target an already pending image");
        check(1, stride, 0, "failed copy refunds its ticket while retaining the original DMA");
    }
    check(0, 0, 0, "cancelled original DMA returns its remaining reservation");
    {
        auto blocker = arena.allocate(arena.capacity_bytes()).value();
        expect(images.host_occupied() == 0 && !images.can_reserve_host() &&
                   !images.reserve_host_destination(),
               "shared non-State allocations deny tickets without fabricating occupied slots");
    }
    {
        store::StateImageStore foreign(physical, &arena, 3, 1);
        auto ticket = foreign.reserve_host_destination().value();
        bool rejected = false;
        try {
            (void)images.reserve_device_to_host(first, std::move(ticket));
        } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected && foreign.host_occupied() == 0 && images.host_occupied() == 0 &&
                   arena.occupied_bytes() == 0,
               "foreign Host ticket is rejected and refunded to its actual owner");
    }
    {
        auto wrong_geometry = arena.allocate(2 * stride).value();
        bool rejected = false;
        try {
            (void)images.reserve_device_to_host(first, std::move(wrong_geometry));
        } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "raw Host allocation must match the complete image geometry");
        check(0, 0, 0, "geometry rejection returns both bytes and pending slot");
    }
    {
        store::StateImageStore disabled(physical, &arena, 2, 0);
        store::StateImageStore no_host(physical, nullptr, 2);
        expect(disabled.host_capacity() == 0 && !disabled.reserve_host_destination() &&
                   no_host.host_capacity() == 0 && !no_host.reserve_host_destination(),
               "zero-slot or absent Host storage never admits a ticket");
        bool rejected = false;
        try {
            store::StateImageStore undersized(physical, &arena, 3, 2);
        } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "logical descriptors must cover Device plus Host slot capacities");
    }
    expect(images.release(first) && images.release(second) && images.occupied() == 0,
           "Host reservation fixture releases its Device images");
    {
        const auto orphan = initialize();
        device.synchronize();
        images.retain_checkpoint_reference(orphan);
        auto copy = images.reserve_device_to_host(orphan).value();
        copy.enqueue(device.transfer_view());
        expect(images.release_checkpoint_owner(orphan) && images.valid(orphan) &&
                   images.host_occupied() == 1,
               "last checkpoint retirement does not free in-flight source or destination");
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
        copy.publish(true);
        expect(!images.valid(orphan) && images.device_occupied() == 0,
               "last transfer lease retires the orphaned logical identity");
        check(0, 0, 0, "orphan transfer publication returns its live slot automatically");
    }
}

void test_state_store(ninfer::DeviceContext& device) {
    ninfer::StateImageBuilder schema;
    (void)schema.add(ninfer::DType::BF16, {8, 3}, 1, "conv");
    (void)schema.add(ninfer::DType::FP32, {4, 4, 2}, 1, "recurrent");
    const auto hidden_field = schema.add(ninfer::DType::BF16, {8}, 1, "hidden");
    (void)schema.add(ninfer::DType::BF16, {4, 8, 2}, 1, "local_k");
    (void)schema.add(ninfer::DType::FP16, {4, 8, 2}, 1, "local_v");
    ninfer::LayoutBuilder builder;
    const auto layout = schema.finish(builder, 4);
    ninfer::DeviceBuffer backing(device.device(), builder.finish());
    ninfer::StateImageStorage physical(backing.span(), layout);
    ninfer::HostContextArena host_backing(layout.image_stride * 2, layout.image_stride);
    store::StateImageStore images(physical, &host_backing, 6);
    const auto hidden_slot = [&](std::int32_t slot) {
        return physical.field(hidden_field).slice(1, slot, 1).view({8});
    };

    const auto initialize = [&] {
        auto handle = images.reserve_destination();
        if (handle) {
            physical.clear(images.physical_slot(*handle), device.execution_view());
            images.complete_initialization(*handle);
        }
        return handle;
    };
    const auto source = initialize();
    expect(source.has_value(), "state source allocation");
    const std::int32_t original_slot = images.physical_slot(*source);
    const auto source_epoch          = images.content_epoch(*source);
    images.freeze(*source);
    expect(images.content_epoch(*source) == source_epoch, "freezing retains content identity");
    images.make_mutable(*source);
    expect(images.physical_slot(*source) == original_slot &&
               images.role(*source) == store::StateImageRole::ActiveMutable,
           "private Move preserves the logical image and physical slot");
    images.freeze(*source);
    const auto destination = images.reserve_destination();
    expect(destination.has_value(), "state destination reservation");
    const store::StateImageSelectors selectors = images.begin_write(*source, *destination);
    expect(selectors.source >= 0 && selectors.destination >= 0 &&
               selectors.source != selectors.destination,
           "fork resolves distinct physical selectors");
    expect(images.role(*destination) == store::StateImageRole::Writing,
           "pending write is not readable active content");
    expect(!images.release(*source) && !images.release(*destination),
           "fork pins both logical images");
    const auto concurrent_destination = images.reserve_destination();
    expect(concurrent_destination.has_value(), "concurrent fork destination reservation");
    (void)images.begin_write(*source, *concurrent_destination);
    expect(images.source_pins(*source) == 2 && !images.release(*source),
           "independent forks each retain the source");
    images.abort_write(*source, *concurrent_destination);
    expect(images.release(*concurrent_destination), "concurrent fork destination release");
    expect(images.source_pins(*source) == 1 && !images.release(*source) &&
               !images.release(*destination),
           "aborting one reader preserves the remaining fork's pins");
    images.complete_write(*source, *destination);
    expect(images.role(*source) == store::StateImageRole::CheckpointImmutable &&
               images.role(*destination) == store::StateImageRole::ActiveMutable &&
               images.content_epoch(*destination) != source_epoch,
           "fork commit preserves source and activates destination");
    images.retain_checkpoint_reference(*source);
    expect(!images.release(*source), "checkpoint owner keeps the source alive after fork commit");
    expect(images.release_checkpoint_owner(*source) && images.release(*destination) &&
               images.occupied() == 0,
           "checkpoint and active state ownership close after release");
    expect(!images.valid(*source), "released state generation becomes stale");

    const auto reused = initialize();
    expect(reused.has_value() && *reused != *source, "state descriptor reuse advances generation");
    expect(images.release(*reused), "reused state image releases");

    for (const bool cancel_last_reader : {false, true}) {
        const auto shared       = initialize();
        const auto first_reader = images.reserve_destination();
        const auto last_reader  = images.reserve_destination();
        expect(shared && first_reader && last_reader, "shared State reader fixture allocates");
        const auto source_slot = images.physical_slot(*shared);
        const std::array<std::uint16_t, 8> content{1, 3, 5, 7, 9, 11, 13, 15};
        const auto hidden = hidden_slot(source_slot);
        CUDA_CHECK(cudaMemcpyAsync(hidden.data, content.data(), sizeof(content),
                                   cudaMemcpyHostToDevice, device.stream()));
        images.freeze(*shared);
        images.retain_checkpoint_reference(*shared); // Cache checkpoint.
        images.retain_checkpoint_reference(*shared); // Paused request snapshot.
        (void)images.begin_write(*shared, *first_reader);
        (void)images.begin_write(*shared, *last_reader);
        expect(images.release_checkpoint_owner(*shared) &&
                   images.release_checkpoint_owner(*shared) && images.valid(*shared) &&
                   images.checkpoint_references(*shared) == 0 && images.source_pins(*shared) == 2,
               "cache eviction and paused cancellation leave execution readers alive");
        expect(!images.can_release_after_checkpoint_references(*shared, 0) &&
                   images.device_occupied() == 3,
               "retired content owner does not advertise pinned State capacity as free");
        physical.copy(source_slot, images.physical_slot(*first_reader), device.execution_view());
        device.synchronize();
        images.complete_write(*shared, *first_reader);
        expect(images.valid(*shared) && images.source_pins(*shared) == 1,
               "one completed reader does not release the remaining reader's State");
        physical.copy(source_slot, images.physical_slot(*last_reader), device.execution_view());
        std::array<std::uint16_t, 8> observed{};
        const auto last_hidden = hidden_slot(images.physical_slot(*last_reader));
        CUDA_CHECK(cudaMemcpyAsync(observed.data(), last_hidden.data, sizeof(observed),
                                   cudaMemcpyDeviceToHost, device.stream()));
        device.synchronize();
        expect(observed == content, "remaining State reader retains the exact source content");
        if (cancel_last_reader) {
            images.abort_write(*shared, *last_reader);
        } else {
            images.complete_write(*shared, *last_reader);
        }
        expect(!images.valid(*shared) && images.device_occupied() == 2,
               "last reader retirement releases the orphaned State automatically");
        const auto replacement = initialize();
        expect(replacement && *replacement != *shared &&
                   images.physical_slot(*replacement) == source_slot,
               "retired State storage is reusable under a new generation");
        expect(images.release(*replacement) && images.release(*first_reader) &&
                   images.release(*last_reader) && images.occupied() == 0,
               "retired checkpoint and execution readers leave no State allocation behind");
    }

    {
        const auto shared = initialize();
        const auto reader = images.reserve_destination();
        images.freeze(*shared);
        images.retain_checkpoint_reference(*shared);
        (void)images.begin_write(*shared, *reader);
        expect(images.release_checkpoint_owner(*shared), "read lease outlives original checkpoint");
        // Pausing an unwritten fork converts its read lease into a new snapshot owner.
        images.retain_checkpoint_reference(*shared);
        images.abort_write(*shared, *reader);
        expect(images.valid(*shared) && images.checkpoint_references(*shared) == 1 &&
                   images.source_pins(*shared) == 0,
               "new snapshot ownership is installed before its read lease retires");
        expect(images.release(*reader) && images.release_checkpoint_owner(*shared) &&
                   images.occupied() == 0,
               "reowned snapshot retires without leaking its previous read lease");
    }

    for (const bool cancel_transfer : {false, true}) {
        const auto shared = initialize();
        device.synchronize();
        images.freeze(*shared);
        images.retain_checkpoint_reference(*shared);
        auto transfer = images.reserve_device_to_host(*shared);
        if (transfer) transfer->enqueue(device.transfer_view());
        expect(transfer && images.release_checkpoint_owner(*shared) && images.valid(*shared),
               "submitted State transfer retains a retired checkpoint source");
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
        if (cancel_transfer) {
            transfer.reset();
        } else {
            transfer->publish(true);
        }
        expect(!images.valid(*shared) && images.occupied() == 0 &&
                   host_backing.occupied_bytes() == 0,
               "last transfer lease retirement frees both State replicas and reservations");
    }

    {
        const auto first  = initialize();
        const auto second = initialize();
        images.freeze(*first);
        images.freeze(*second);
        const auto first_writer  = images.reserve_destination();
        const auto second_writer = images.reserve_destination();
        (void)images.begin_write(*first, *first_writer);
        (void)images.begin_write(*second, *second_writer);
        bool rejected = false;
        try {
            images.complete_write(*first, *second_writer);
        } catch (const std::logic_error&) { rejected = true; }
        expect(rejected && images.source_pins(*first) == 1 && images.source_pins(*second) == 1,
               "write completion must name its reserved source");
        images.abort_write(*first, *first_writer);
        images.abort_write(*second, *second_writer);
        expect(images.release(*first) && images.release(*second) && images.release(*first_writer) &&
                   images.release(*second_writer),
               "independent writers release their own pins");
    }

    const auto host_source = initialize();
    expect(host_source.has_value(), "Host state source allocation");
    device.synchronize();
    images.freeze(*host_source);
    {
        auto allocation     = host_backing.allocate(layout.image_stride);
        const auto occupied = host_backing.occupied_bytes();
        auto cancelled      = images.reserve_device_to_host(*host_source, std::move(*allocation));
        expect(host_backing.occupied_bytes() == occupied && host_backing.allocation_count() == 1,
               "transfer consumes the existing Host reservation without reallocating");
        expect(cancelled && host_backing.reserved_bytes() == layout.image_stride &&
                   host_backing.live_bytes() == 0 && images.source_pins(*host_source) == 1 &&
                   !images.release(*host_source),
               "State transfer reservation charges destination while its source is pinned");
    }
    expect(host_backing.occupied_bytes() == 0 && images.source_pins(*host_source) == 0 &&
               images.residency(*host_source) == store::StateReplicaResidency::DeviceOnly,
           "cancelled State transfer releases its actual Host extent and retains source state");
    auto d2h = images.reserve_device_to_host(*host_source);
    if (d2h) d2h->enqueue(device.transfer_view());
    expect(d2h.has_value() &&
               images.residency(*host_source) == store::StateReplicaResidency::DeviceOnly,
           "incomplete State D2H does not publish a Host replica");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
    d2h->publish(true);
    expect(images.residency(*host_source) == store::StateReplicaResidency::Both &&
               host_backing.live_bytes() == layout.image_stride &&
               host_backing.reserved_bytes() == 0,
           "State D2H backup publishes stable Both residency");

    const auto moved_device = images.reserve_logical_destination();
    expect(moved_device.has_value(), "State replica split destination allocation");
    images.split_device_replica_identity(*host_source, *moved_device);
    expect(images.role(*moved_device) == store::StateImageRole::ReservedDestination,
           "replica handover does not publish a writer");
    images.complete_initialization(*moved_device);
    expect(images.residency(*host_source) == store::StateReplicaResidency::HostOnly &&
               images.residency(*moved_device) == store::StateReplicaResidency::DeviceOnly &&
               images.content_epoch(*moved_device) == images.content_epoch(*host_source),
           "State replica identity split keeps old Host content and moves Device ownership");

    const auto fork_one = images.reserve_logical_destination();
    const auto fork_two = images.reserve_logical_destination();
    expect(fork_one && fork_two, "concurrent Host State forks reserve logical destinations");
    auto h2d_one = images.reserve_host_to_device(*host_source, *fork_one);
    if (h2d_one) h2d_one->enqueue(device.transfer_view());
    auto h2d_two = images.reserve_host_to_device(*host_source, *fork_two);
    if (h2d_two) h2d_two->enqueue(device.transfer_view());
    expect(h2d_one && h2d_two && images.source_pins(*host_source) == 2,
           "immutable Host State source supports multiple fork pins");
    auto h2d_source = images.reserve_host_to_device(*host_source);
    if (h2d_source) h2d_source->enqueue(device.transfer_view());
    expect(h2d_source && images.source_pins(*host_source) == 3,
           "Host replica remains shared while restoring its own Device replica");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
    bool rejected = false;
    try {
        h2d_source->publish(false);
    } catch (const std::logic_error&) { rejected = true; }
    expect(rejected && h2d_source->valid() && images.host_resident(*host_source) &&
               !images.device_resident(*host_source) && images.source_pins(*host_source) == 3,
           "restoration cannot retire Host storage retained by other transfers");
    h2d_one->publish(true);
    h2d_two->publish(true);
    h2d_source->publish(false);
    expect(images.role(*fork_one) == store::StateImageRole::ReservedDestination,
           "Host materialization is separate from native adoption");
    images.complete_initialization(*fork_one);
    images.complete_initialization(*fork_two);
    expect(images.source_pins(*host_source) == 0 &&
               images.residency(*fork_one) == store::StateReplicaResidency::DeviceOnly &&
               images.residency(*fork_two) == store::StateReplicaResidency::DeviceOnly &&
               images.content_epoch(*fork_one) == images.content_epoch(*host_source) &&
               images.content_epoch(*fork_two) == images.content_epoch(*host_source),
           "Host State forks publish independent Device destinations");
    expect(images.release(*host_source) && images.release(*moved_device) &&
               images.release(*fork_one) && images.release(*fork_two) &&
               images.host_occupied() == 0,
           "State Host/Device replica ownership closes without leaked slots");
}

void test_migration_state_fields(ninfer::DeviceContext& device) {
    using namespace ninfer;
    LayoutBuilder backing_layout;
    StateImageBuilder schema;
    (void)schema.add(DType::FP32, {4}, 1, "recurrent");
    const auto local = schema.add(DType::BF16, {8}, 2, "local_kv");
    (void)schema.add(DType::I32, {3}, 1, "tail");
    const auto group  = schema.group(local, local);
    const auto layout = schema.finish(backing_layout, 2);
    DeviceBuffer backing(device.device(), backing_layout.finish());
    StateImageStorage storage(backing.span(), layout);
    store::StateImageStore states(storage, nullptr, 2);
    store::MigrationContext executor(device, 0);
    const auto source = states.reserve_destination().value();
    std::vector<std::byte> words(storage.image_bytes());
    for (std::size_t i = 0; i < words.size(); ++i) words[i] = std::byte((i * 29 + 13) & 0xff);
    CUDA_CHECK(cudaMemcpy(storage.image(states.physical_slot(source)).data, words.data(),
                          words.size(), cudaMemcpyHostToDevice));
    states.complete_initialization(source);
    states.freeze(source);
    for (bool clear : {false, true}) {
        const auto destination = states.reserve_destination().value();
        std::vector<std::byte> expected(words.size(), std::byte{0x55});
        const auto image = storage.image(states.physical_slot(destination));
        CUDA_CHECK(
            cudaMemcpy(image.data, expected.data(), expected.size(), cudaMemcpyHostToDevice));
        store::MigrationBatch migration(executor, nullptr);
        migration.own_state(states, destination);
        if (clear) {
            migration.clear_state(storage, states);
            std::fill(expected.begin(), expected.end(), std::byte{0});
        } else {
            migration.add_state_group(states, storage, source, group);
            std::copy_n(words.begin() + group.offset, group.bytes, expected.begin() + group.offset);
        }
        migration.submit();
        migration.drain();
        migration.publish();
        std::vector<std::byte> observed(words.size());
        CUDA_CHECK(
            cudaMemcpy(observed.data(), image.data, observed.size(), cudaMemcpyDeviceToHost));
        expect(observed == expected,
               "migration changed bytes outside its State field group or clear extent");
    }
    expect(states.release(source) && states.occupied() == 0,
           "State field migration retained a reservation");
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);

    try {
        ninfer::DeviceContext device(0);
        test_state_host_reservations(device);
        test_state_store(device);
        test_migration_state_fields(device);
        device.synchronize();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
