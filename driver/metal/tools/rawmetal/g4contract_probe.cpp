#include <cstdio>
#include <cstdlib>
#include <string>
#include "loader/load_plan.hpp"

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "";
    try {
        // Gemma4's KV sharing, in the config's own vocabulary. Authoring moved
        // to the Rust registry, so the request carries the two counts and the
        // author subtracts; this used to pass the precomputed first shared
        // layer, which no longer crosses the boundary.
        pie::metal::model::ContractFacts facts;
        facts.num_hidden_layers = argc > 2 ? std::atoi(argv[2]) : 0;
        facts.num_kv_shared_layers = argc > 3 ? std::atoi(argv[3]) : 0;
        auto plan = pie::metal::compile_load_plan(
            dir, pie::metal::metal_device_target(),
            pie::metal::descriptor_for_testing("gemma4", facts));
        // The plan's size, not just its success: the KV-sharing facts are only
        // observable as tensors the author declined to declare, so a probe that
        // printed OK either way could not tell whether they arrived.
        const auto& view = plan.view();
        std::printf("OK: load plan compiled and verified"
                    " (tensors %zu, buffers %zu, instrs %zu, sources %zu)\n",
                    view.tensors.len, view.buffers.len, view.instrs.len,
                    view.sources.len);
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
