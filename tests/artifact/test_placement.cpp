#include "artifact/fixture.h"
#include "artifact/placement.h"
#include "artifact/reader.h"

#include <iostream>

namespace {
using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;

void placements() {
    Fixture fixture;
    fixture.write();
    Reader reader(fixture.entry);
    const auto matrix = reader.find("q5");
    const auto cuda = Device::cuda(0);
    for (const bool reverse : {false, true}) {
        for (const bool mapped : {false, true}) {
            std::array requests{
                PlacementRequest{matrix, cuda, mapped ? HostAccess::CudaMapped : HostAccess::CpuOnly},
                PlacementRequest{matrix, cuda},
                PlacementRequest{matrix, cuda, mapped ? HostAccess::CudaMapped : HostAccess::CpuOnly}};
            if (reverse) std::reverse(requests.begin(), requests.end());
            const auto plan = plan_materialization(reader, requests);
            require(plan.tensors.size() == 1 && plan.requests[0] == plan.requests[1] &&
                        plan.requests[1] == plan.requests[2] && plan.device_objects.size() == 1 &&
                        plan.device_capacity_bytes >= reader.geometry(matrix).bytes,
                    "shared aliases duplicated their Device parent");
            require(plan.mapped_objects.size() == (mapped ? 1 : 0) &&
                        plan.registered_host_objects.size() == (mapped ? 1 : 0),
                    "shared mapped consumer lost its registered Host parent");
        }
    }
    const std::array mapped_only{
        PlacementRequest{matrix, cuda, HostAccess::CudaMapped},
        PlacementRequest{matrix, Device::cpu(), HostAccess::CudaTransfer}};
    const auto plan = plan_materialization(reader, mapped_only);
    require(plan.device_objects.empty() && plan.device_capacity_bytes == 0 &&
                plan.mapped_objects.size() == 1 && plan.registered_host_objects.size() == 1,
            "mapped-only aliases acquired a redundant Device copy");
    const std::array separate{
        PlacementRequest{matrix, cuda, HostAccess::CudaMapped},
        PlacementRequest{reader.find("divisors"), cuda}};
    const auto untied = plan_materialization(reader, separate);
    require(untied.tensors.size() == 2 && untied.device_objects.size() == 1 &&
                untied.mapped_objects.size() == 1,
            "independent consumers merged unrelated tensors");
    const std::array invalid{PlacementRequest{matrix, Device::cpu(), HostAccess::CudaMapped}};
    rejects([&] { (void)plan_materialization(reader, invalid); },
            "mapped request without a CUDA device was accepted");
    const std::array multiple{
        PlacementRequest{matrix, cuda, HostAccess::CudaMapped},
        PlacementRequest{matrix, Device::cuda(1)}};
    rejects([&] { (void)plan_materialization(reader, multiple); },
            "one materialization accepted multiple CUDA devices");
}
} // namespace

int main() {
    try {
        placements();
        std::cout << "ok: shared, separate and mapped weight placement\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
