#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "pie/driver/testing/image.hpp"
#include "pie/driver/launch/plan.hpp"
#include "pipeline/library_region.hpp"

int main(int argc, char** argv) {
    const std::string path =
        (argc > 1 ? argv[1] : "../fixtures") +
        std::string("/nucleus_sample.launch");
    pie::driver::testing::PackageImage image;
    std::string error;
    if (!image.load(path, &error) || image.package().plans.len != 1) {
        std::fprintf(stderr, "nucleus_region_test: %s\n", error.c_str());
        return 1;
    }
    const pie::driver::launch::plan::StagePlan plan =
        pie::driver::launch::plan::adopt(0, image.package().plans.ptr[0]);
    const auto region = std::find_if(
        plan.fused.regions.begin(),
        plan.fused.regions.end(),
        [](const auto& candidate) {
            return candidate.library &&
                candidate.library_op == PTIR_LIBRARY_NUCLEUS_SAMPLE;
        });
    const std::vector<std::uint32_t> expected_nodes{
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    if (PTIR_REGION_PLAN_VERSION != 4 ||
        PTIR_COMPILER_VERSION != 3 ||
        plan.ops.size() != 17 ||
        region == plan.fused.regions.end() ||
        region->nodes != expected_nodes ||
        region->inputs != std::vector<std::uint32_t>({0, 2, 1}) ||
        region->outputs != std::vector<std::uint32_t>({15}) ||
        !region->sinks.empty()) {
        std::fprintf(
            stderr,
            "nucleus_region_test: region ABI mismatch "
            "(ops=%zu nodes=%zu inputs=%zu outputs=%zu first_input=%u)\n",
            plan.ops.size(),
            region == plan.fused.regions.end() ? 0 : region->nodes.size(),
            region == plan.fused.regions.end() ? 0 : region->inputs.size(),
            region == plan.fused.regions.end() ? 0 : region->outputs.size(),
            region == plan.fused.regions.end() || region->inputs.empty()
                ? UINT32_MAX
                : region->inputs[0]);
        return 1;
    }
    auto interleaved = *region;
    interleaved.nodes = {3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    if (pie_cuda_driver::pipeline::library_region_launch_node(interleaved) !=
        16) {
        std::fputs(
            "nucleus_region_test: library launched before final node\n",
            stderr);
        return 1;
    }
    std::puts("nucleus_region_test: OK");
    return 0;
}
