// Does the GPT-OSS storage schema describe the real checkpoint?
//
// The contract only sees tensor NAMES, so the one thing that can be wrong about
// it is a name -- and a tensor declared under a name nothing binds is not an
// error until forty kernels later. Compiling a load plan against the real
// checkpoint type-checks every declaration against what the files contain.
#include <cstdio>
#include <cstdlib>
#include <string>
#include "loader/load_plan.hpp"

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "";
    try {
        auto plan = pie::metal::compile_load_plan(
            dir, pie::metal::metal_device_target(), "gpt_oss", pie::metal::model::ContractFacts{});
        std::printf("OK: load plan compiled and verified\n");
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
