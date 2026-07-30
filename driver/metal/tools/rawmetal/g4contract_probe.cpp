#include <cstdio>
#include <string>
#include "loader/load_plan.hpp"

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "";
    const int first_shared = argc > 2 ? std::atoi(argv[2]) : -1;
    try {
        pie::metal::model::ContractFacts facts;
        facts.first_kv_shared_layer = first_shared;
        auto plan = pie::metal::compile_load_plan(
            dir, pie::metal::metal_device_target(), "gemma4", facts);
        std::printf("OK: load plan compiled and verified\n");
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
