#include "core/buffer.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "models/common/context/storage.h"
#include "models/common/context/transfer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace ninfer;
namespace context = ninfer::models::context;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void expect_rejected(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::logic_error&) { return; }
    throw std::runtime_error(message);
}

KVPageGeometry mixed_geometry() {
    return {
        .planes = {
            {DType::U8, 128, 2, 256},
            {DType::U8, 128, 2, 256},
            {DType::U8, 16, 2, 256},
            {DType::U8, 16, 2, 256},
            {.dtype = DType::BF16, .leading_extent = 128, .head_extent = 1, .tokens_per_row = 4}}};
}

void check_record_geometry() {
    const auto layout = plan_kv_page_layout(mixed_geometry());
    constexpr std::array<std::size_t, 5> offsets{0, 16384, 32768, 34816, 36864};
    constexpr std::array<std::size_t, 5> bytes{16384, 16384, 2048, 2048, 4096};
    constexpr std::array<std::size_t, 5> head_bytes{8192, 8192, 1024, 1024, 4096};
    constexpr std::array<std::array<std::int32_t, 4>, 5> shapes{
        {{128, 64, 2, 1}, {128, 64, 2, 1}, {16, 64, 2, 1}, {16, 64, 2, 1}, {128, 16, 1, 1}}};
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        expect(layout.planes[i].tensor.shape == shapes[i], "Packed KV record shape differs");
        expect(layout.planes[i].tensor.region.offset == offsets[i],
               "Packed KV record plane offset differs");
        expect(layout.planes[i].tensor.region.bytes == bytes[i],
               "Packed KV record plane span differs");
        expect(layout.planes[i].head_stride_bytes == head_bytes[i],
               "Packed KV record head stride differs");
    }
    expect(layout.page_bytes == 40960 && layout.payload_bytes() == 40960,
           "Packed KV record byte count differs");
    auto aligned                    = mixed_geometry();
    aligned.planes.back().alignment = 65536;
    const auto padded               = plan_kv_page_layout(aligned);
    expect(padded.planes.back().tensor.region.offset == 65536 && padded.page_bytes == 131072 &&
               padded.payload_bytes() == 40960,
           "Packed KV record must preserve plane alignment on both Device and Host");
    for (const std::uint32_t span : {0U, 2U, 256U}) {
        auto invalid                         = mixed_geometry();
        invalid.planes.back().tokens_per_row = span;
        expect_rejected([&] { (void)plan_kv_page_layout(invalid); },
                        "Unsupported KV row span was accepted");
    }
    auto invalid        = mixed_geometry();
    invalid.page_tokens = 63;
    expect_rejected([&] { (void)plan_kv_page_layout(invalid); },
                    "Non-divisible KV page span was accepted");
}

context::ContextRange device_range(std::size_t offset, std::size_t bytes) {
    return {context::ContextMemory::Device, static_cast<std::int64_t>(offset), bytes};
}

context::ContextRange host_range(std::size_t offset, std::size_t bytes) {
    return {context::ContextMemory::Host, static_cast<std::int64_t>(offset), bytes};
}

context::RecordCopy page_copy(context::ContextRange source, context::ContextRange destination) {
    return {source, destination, {runtime::KvPoolId{2}}, 0, 1};
}

void check_action_batches() {
    context::ContextActions plan;
    for (std::size_t i = 0; i < context::kContextActionBatchOperations + 3; ++i) {
        plan.operations.emplace_back(page_copy(device_range((1ULL << 34) + i * 256, 128),
                                               host_range((1ULL << 35) + i * 256, 128)));
    }
    context::MappingUpdate first{2, 3, 5, {}};
    for (std::size_t i = 0; i < context::kContextActionBatchOffsets + 17; ++i) {
        first.offsets.push_back((1LL << 34) + i * 1024);
    }
    plan.operations.emplace_back(first);
    plan.operations.emplace_back(context::RecordClear{device_range(1024, 512)});
    plan.operations.emplace_back(
        context::MappingUpdate{1, 0, 3, std::vector<std::int64_t>(4090, -1)});
    plan.operations.emplace_back(page_copy(device_range(64, 32), device_range(128, 32)));
    context::ContextActionCursor cursor;
    std::size_t expected_operation = 0, expected_mapping = 0;
    while (!cursor.complete(plan)) {
        const auto batch = context::take_action_batch(plan, cursor);
        expect(!batch.empty() && batch.operations.size() <= context::kContextActionBatchOperations,
               "Context batch did not respect its operation limit or make progress");
        std::size_t mapping_count = 0;
        for (const auto& action : batch.operations) {
            const auto& original = plan.operations.at(expected_operation);
            expect(action.index() == original.index(), "Context batching changed operation order");
            if (const auto* mapping = std::get_if<context::MappingUpdate>(&action)) {
                const auto& full = std::get<context::MappingUpdate>(original);
                expect(mapping->pool_id == full.pool_id && mapping->row == full.row &&
                           mapping->logical_begin == full.logical_begin + expected_mapping &&
                           std::equal(mapping->offsets.begin(), mapping->offsets.end(),
                                      full.offsets.begin() + expected_mapping),
                       "Context batching changed a mapping value or logical column");
                expected_mapping += mapping->offsets.size();
                mapping_count += mapping->offsets.size();
                if (expected_mapping == full.offsets.size()) {
                    ++expected_operation;
                    expected_mapping = 0;
                }
            } else if (const auto* copy = std::get_if<context::RecordCopy>(&action)) {
                const auto& full = std::get<context::RecordCopy>(original);
                expect(copy->source == full.source && copy->destination == full.destination &&
                           copy->resource == full.resource &&
                           copy->image_count == full.image_count &&
                           copy->page_groups == full.page_groups,
                       "Context batching changed a copy descriptor");
                ++expected_operation;
            } else {
                expect(std::get<context::RecordClear>(action).destination ==
                           std::get<context::RecordClear>(original).destination,
                       "Context batching changed a clear descriptor");
                ++expected_operation;
            }
        }
        expect(mapping_count <= context::kContextActionBatchOffsets,
               "Context batch exceeded its mapping-offset limit");
    }
    expect(expected_operation == plan.operations.size() && expected_mapping == 0 &&
               context::take_action_batch(plan, cursor).empty(),
           "Context batching lost an action or repeated a completed plan");
}

void check_peer_batches() {
    context::PeerCopy send{7, true}, receive{3, false};
    std::vector<std::byte> source(32768), destination(32768, std::byte{0});
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = std::byte(i % 251);
    constexpr std::size_t count = context::kContextActionBatchOffsets + 13;
    for (std::size_t i = 0; i < count; ++i) send.ranges.push_back(device_range(i * 4, 3));
    receive.ranges = {device_range(11, count * 3 - 17), device_range(20000, 17)};
    std::array<context::ContextActions, 2> plans;
    plans[0].operations = {context::RecordClear{device_range(0, 4)}, send};
    plans[1].operations = {receive, context::RecordClear{device_range(1, 4)}};
    std::array<context::ContextActionCursor, 2> cursors;
    const std::array ranks{context::ContextActionBatchSource{3, &plans[0], &cursors[0]},
                           context::ContextActionBatchSource{7, &plans[1], &cursors[1]}};
    std::size_t copied = 0;
    while (!cursors[0].complete(plans[0]) || !cursors[1].complete(plans[1])) {
        const auto batch = context::take_action_batches(ranks);
        std::array<std::vector<context::ContextRange>, 2> ranges;
        for (std::size_t rank = 0; rank < batch.size(); ++rank)
            for (const auto& action : batch[rank].operations)
                if (const auto* peer = std::get_if<context::PeerCopy>(&action)) {
                    expect(peer->ranges.size() <= context::kContextActionBatchOffsets,
                           "Context peer batch exceeded its descriptor capacity");
                    ranges[rank] = peer->ranges;
                }
        expect(ranges[0].empty() == ranges[1].empty(),
               "Context peer endpoints were placed in different batches");
        if (ranges[0].empty()) continue;
        std::vector<std::byte> payload;
        for (const auto& range : ranges[0])
            payload.insert(payload.end(), source.begin() + range.offset,
                           source.begin() + range.offset + range.bytes);
        std::size_t offset = 0;
        for (const auto& range : ranges[1]) {
            expect(range.bytes <= payload.size() - offset,
                   "Context peer endpoints have different byte counts");
            std::copy_n(payload.begin() + offset, range.bytes, destination.begin() + range.offset);
            offset += range.bytes;
        }
        expect(offset == payload.size(), "Context peer batch lost payload bytes");
        copied += offset;
    }
    std::vector<std::byte> expected(destination.size());
    for (std::size_t i = 0; i < count * 3; ++i) {
        const auto target = i < count * 3 - 17 ? 11 + i : 20000 + i - (count * 3 - 17);
        expected[target]  = source[(i / 3) * 4 + i % 3];
    }
    expect(copied == count * 3 && destination == expected,
           "Context peer batching changed logical bytes or untouched ranges");

    plans[0].operations = {send};
    receive.ranges.front().bytes -= 1;
    plans[1].operations = {receive};
    cursors             = {};
    expect_rejected([&] { (void)context::take_action_batches(ranks); },
                    "Context peer plan accepted different payload lengths");
    expect(cursors[0].operation == 0 && cursors[1].operation == 0 && cursors[0].peer_range == 0 &&
               cursors[1].peer_range == 0,
           "Rejected Context peer batching advanced a cursor");
}

void check_record_execution(DeviceContext& device) {
    const auto record = plan_kv_page_layout(mixed_geometry());
    const auto bytes  = record.page_bytes;
    DeviceBuffer arena(device.device(), 12 * bytes);
    PinnedHostBuffer host(8 * bytes);
    LayoutBuilder builder;
    const auto table_layout =
        plan_kv_execution_tables(builder, {.logical_page_capacity = 4, .table_rows = 2});
    DeviceBuffer table_backing(device.device(), builder.finish(256));
    context::ExecutionTableStorage tables(table_backing.span(), table_layout, arena.span(), bytes);
    const std::array table_views{context::ContextTableStorage{2, &tables}};
    context::ContextStorage storage(arena.span(), host.span(), table_views);
    context::ContextMigration migration(device, storage);

    std::vector<std::byte> original(arena.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
        original[i] = std::byte((i * 29U + (i >> 9U) + 17U) & 255U);
    }
    arena.copy_from_host(original.data(), original.size());
    std::memset(host.data(), 0xa5, host.size());
    table_backing.fill(0xff);

    const std::array<std::size_t, 3> source_offsets{256, 3 * bytes + 256, 7 * bytes + 256};
    const std::array<std::size_t, 3> host_offsets{256, 2 * bytes + 512, 5 * bytes + 768};
    context::ContextActions export_records;
    for (std::size_t i = 0; i < source_offsets.size(); ++i) {
        export_records.operations.emplace_back(
            page_copy(device_range(source_offsets[i], bytes), host_range(host_offsets[i], bytes)));
    }
    migration.submit(export_records);
    device.synchronize();
    // Transfer uses its own stream; drain is the explicit completion boundary here.
    migration.drain();
    for (std::size_t i = 0; i < source_offsets.size(); ++i) {
        expect(std::memcmp(static_cast<const std::byte*>(host.data()) + host_offsets[i],
                           original.data() + source_offsets[i], bytes) == 0,
               "Device-to-Host transfer changed packed record bytes");
        expect(static_cast<const std::byte*>(host.data())[host_offsets[i] - 1] == std::byte{0xa5} &&
                   static_cast<const std::byte*>(host.data())[host_offsets[i] + bytes] ==
                       std::byte{0xa5},
               "Record transfer touched bytes outside the destination");
    }

    const std::size_t restored = 9 * bytes;
    const std::size_t copied   = 10 * bytes;
    context::ContextActions actions;
    actions.operations.emplace_back(
        page_copy(host_range(host_offsets[1], bytes), device_range(restored, bytes)));
    actions.operations.emplace_back(
        page_copy(device_range(restored, bytes), device_range(copied, bytes)));
    actions.operations.emplace_back(context::RecordClear{device_range(11 * bytes, bytes)});
    actions.operations.emplace_back(context::MappingUpdate{
        2, 1, 1, {static_cast<std::int64_t>(copied), static_cast<std::int64_t>(restored), -1}});
    const auto observed = storage.describe(actions.operations.front());
    expect(observed.work.payload_bytes == bytes && observed.work.copy_operations == 1 &&
               observed.page_groups == 1 &&
               observed.direction == runtime::ContextTransferDirection::HostToDevice,
           "Transfer observation does not describe the actual record copy");
    migration.submit(actions);
    expect_rejected([&] { migration.submit(actions); }, "An occupied migration slot was reused");
    migration.drain();
    std::vector<std::byte> after(arena.size());
    arena.copy_to_host(after.data(), after.size());
    auto expected = original;
    std::copy_n(original.begin() + source_offsets[1], bytes, expected.begin() + restored);
    std::copy_n(original.begin() + source_offsets[1], bytes, expected.begin() + copied);
    std::fill(expected.begin() + 11 * bytes, expected.end(), std::byte{0});
    expect(after == expected, "Ordered record copies/clear changed neighboring bytes");
    std::array<std::int64_t, 8> mapping{};
    table_backing.copy_to_host(mapping.data(), sizeof(mapping));
    const std::array<std::int64_t, 8> expected_mapping{
        -1, -1, -1, -1, -1, static_cast<std::int64_t>(copied), static_cast<std::int64_t>(restored),
        -1};
    expect(mapping == expected_mapping, "I64 execution table changed logical order or other rows");

    context::ContextActions invalid;
    invalid.operations.emplace_back(context::RecordClear{device_range(0, bytes)});
    invalid.operations.emplace_back(
        context::MappingUpdate{2, 0, 0, {static_cast<std::int64_t>(arena.size())}});
    expect_rejected([&] { migration.submit(invalid); }, "An out-of-range mapping was accepted");
    arena.copy_to_host(after.data(), after.size());
    expect(after == expected && !migration.occupied(),
           "A rejected action batch partially changed physical storage");
    invalid.operations = {page_copy(device_range(0, bytes), device_range(256, bytes))};
    expect_rejected([&] { migration.submit(invalid); }, "Overlapping records were copied");

    context::ContextActions host_maintenance;
    host_maintenance.operations.emplace_back(
        page_copy(host_range(host_offsets[0], bytes), host_range(6 * bytes + 1024, bytes)));
    host_maintenance.operations.emplace_back(
        context::RecordClear{host_range(host_offsets[0], bytes)});
    migration.submit(host_maintenance);
    expect(migration.ready(), "Host maintenance did not complete synchronously");
    const auto host_observations = migration.observations();
    expect(host_observations.size() == 1 &&
               host_observations.front().direction ==
                   runtime::ContextTransferDirection::HostToHost &&
               host_observations.front().elapsed_ns > 0,
           "Host maintenance did not report its CPU copy duration");
    migration.finish();
    expect(std::memcmp(static_cast<const std::byte*>(host.data()) + 6 * bytes + 1024,
                       original.data() + source_offsets[0], bytes) == 0,
           "Host record relocation changed bytes");
    expect(std::all_of(static_cast<const std::byte*>(host.data()) + host_offsets[0],
                       static_cast<const std::byte*>(host.data()) + host_offsets[0] + bytes,
                       [](auto value) { return value == std::byte{0}; }),
           "Host clear did not follow Host copy");
}

void check_incremental_execution(DeviceContext& device) {
    // Two adjacent descriptors must be coalesced only within the submitted byte window.
    // An urgent compute-stream readback completes while later Context bytes remain unqueued.
    for (const std::size_t window : {2ULL << 20, 8ULL << 20}) {
        const auto bytes = 3 * window + 13;
        DeviceBuffer arena(device.device(), 3 * bytes + 192);
        PinnedHostBuffer host(bytes + 64), urgent(28);
        LayoutBuilder builder;
        const auto layout =
            plan_kv_execution_tables(builder, {.logical_page_capacity = 2, .table_rows = 1});
        DeviceBuffer table_backing(device.device(), builder.finish(256));
        context::ExecutionTableStorage tables(table_backing.span(), layout, arena.span(), 256);
        const std::array views{context::ContextTableStorage{2, &tables}};
        context::ContextStorage storage(arena.span(), host.span(), views);
        context::ContextMigration migration(device, storage);
        std::vector<std::byte> original(arena.size(), std::byte{0xa5});
        for (std::size_t i = 0; i < bytes; ++i) original[i] = std::byte((i * 29 + 13) & 255);
        arena.copy_from_host(original.data(), original.size());
        std::memset(host.data(), 0xa5, host.size());
        table_backing.fill(0xff);
        context::ContextActions actions;
        actions.operations = {
            page_copy(device_range(0, bytes / 2), host_range(32, bytes / 2)),
            page_copy(device_range(bytes / 2, bytes - bytes / 2),
                      host_range(32 + bytes / 2, bytes - bytes / 2)),
            page_copy(host_range(32, bytes), device_range(bytes + 64, bytes)),
            page_copy(device_range(bytes + 64, bytes), device_range(2 * bytes + 128, bytes)),
            context::RecordClear{device_range(2 * bytes + 128 + bytes - 17, 17)},
            context::MappingUpdate{2, 0, 0, {static_cast<std::int64_t>(bytes + 64),
                                           static_cast<std::int64_t>(2 * bytes + 128)}}};
        migration.submit(actions, window);
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
        expect(!migration.ready() && migration.occupied(),
               "A partial Context window retired the complete action plan");
        expect_rejected([&] { migration.finish(); }, "An unfinished Context window was reused");
        expect_rejected([&] { (void)migration.observations(); },
                        "An unfinished Context plan published observations");
        const auto* captured = static_cast<const std::byte*>(host.data());
        expect(std::memcmp(captured + 32, original.data(), window) == 0 &&
                   std::all_of(captured + 32 + window, captured + host.size(),
                               [](auto value) { return value == std::byte{0xa5}; }),
               "Context copies exceeded the first byte window");
        std::array<std::int64_t, 2> mapping{};
        table_backing.copy_to_host(mapping.data(), sizeof(mapping));
        expect(mapping == std::array<std::int64_t, 2>{-1, -1},
               "Context tables were published before their record copies");
        copy_bytes_async(urgent.span(), arena.span().subspan(0, urgent.size()),
                         device.execution_view());
        device.synchronize();
        expect(std::memcmp(urgent.data(), original.data(), urgent.size()) == 0 &&
                   !migration.ready() && captured[32 + window] == std::byte{0xa5},
               "Urgent compute readback required submitting the remaining Context plan");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!migration.ready()) {
            migration.advance();
            if (std::chrono::steady_clock::now() > deadline) {
                migration.drain();
                throw std::runtime_error("Incremental Context plan did not make progress");
            }
            std::this_thread::yield();
        }
        const auto observations = migration.observations();
        expect(observations.size() == 3 &&
                   std::all_of(observations.begin(), observations.end(), [&](const auto& item) {
                       return item.work.payload_bytes == bytes &&
                              item.work.copy_operations >= 4 && item.elapsed_ns > 0;
                   }),
               "Incremental observations lost payload bytes, windows or elapsed time");
        migration.finish();
        auto expected = original;
        std::copy_n(original.begin(), bytes, expected.begin() + bytes + 64);
        std::copy_n(original.begin(), bytes, expected.begin() + 2 * bytes + 128);
        std::fill_n(expected.begin() + 3 * bytes + 128 - 17, 17, std::byte{0});
        std::vector<std::byte> actual(arena.size());
        arena.copy_to_host(actual.data(), actual.size());
        table_backing.copy_to_host(mapping.data(), sizeof(mapping));
        expect(actual == expected &&
                   mapping == std::array<std::int64_t, 2>{
                                  static_cast<std::int64_t>(bytes + 64),
                                  static_cast<std::int64_t>(2 * bytes + 128)} &&
                   std::memcmp(captured + 32, original.data(), bytes) == 0 &&
                   captured[31] == std::byte{0xa5} && captured[32 + bytes] == std::byte{0xa5},
               "Incremental copies changed dependency order, table offsets or neighboring bytes");
        context::ContextActions export_only;
        export_only.operations = {page_copy(device_range(0, bytes), host_range(32, bytes))};
        std::memset(host.data(), 0xa5, host.size());
        migration.submit(export_only, window);
        migration.drain();
        expect(!migration.occupied() && migration.ready() &&
                   std::memcmp(captured + 32, original.data(), bytes) == 0,
               "Drain omitted unsubmitted Context windows");
        context::ContextMigration compute_migration(device, storage, device.execution_view());
        expect_rejected([&] { compute_migration.submit(export_only, window); },
                        "Compute stream accepted an incremental Context plan");
    }
}
} // namespace

int main() {
    try {
        check_record_geometry();
        check_action_batches();
        check_peer_batches();
        int devices       = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || !devices) {
            std::cout << "CUDA device unavailable\n";
            return 77;
        }
        CUDA_CHECK(status);
        DeviceContext device;
        check_record_execution(device);
        check_incremental_execution(device);
        std::cout << "Packed Context records and execution tables passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Context storage test failed: " << error.what() << '\n';
        return 1;
    }
}
