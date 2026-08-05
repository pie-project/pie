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
            dir, pie::metal::metal_device_target(), "gemma4", facts);
        std::printf("OK: load plan compiled and verified\n");
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
