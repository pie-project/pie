#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "pie_native/ptir/bound.hpp"
#include "pie_native/ptir/plan.hpp"
#include "pipeline/library_region.hpp"

namespace {

std::vector<std::uint8_t> decode_hex(const std::string& value) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(value.size() / 2);
    for (std::size_t index = 0; index + 1 < value.size(); index += 2) {
        bytes.push_back(static_cast<std::uint8_t>(
            std::stoul(value.substr(index, 2), nullptr, 16)));
    }
    return bytes;
}

std::string field(const std::string& path, const std::string& name) {
    std::ifstream input(path);
    std::string line;
    const std::string prefix = name + ": ";
    while (std::getline(input, line)) {
        if (line.rfind(prefix, 0) == 0) return line.substr(prefix.size());
    }
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path =
        (argc > 1 ? argv[1] : "tests/golden-ptir") +
        std::string("/nucleus_sample.txt");
    const auto sidecar = decode_hex(field(path, "sidecar"));
    pie_native::ptir::bound::Bound bound;
    std::string error;
    if (!pie_native::ptir::bound::parse_sidecar(
            sidecar.data(), sidecar.size(), bound, &error) ||
        bound.plans.size() != 1) {
        std::fprintf(stderr, "nucleus_region_test: sidecar: %s\n", error.c_str());
        return 1;
    }
    pie_native::ptir::plan::StagePlan plan;
    if (!pie_native::ptir::plan::decode(
            bound.plans[0].bytes.data(),
            bound.plans[0].bytes.size(),
            plan,
            &error)) {
        std::fprintf(stderr, "nucleus_region_test: plan: %s\n", error.c_str());
        return 1;
    }
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
