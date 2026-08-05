#include <cstdio>
#include "loader/load_plan.hpp"
int main(int argc, char** argv) {
    try {
        pie::metal::compile_load_plan(argv[1], pie::metal::metal_device_target(),
                                      pie::metal::descriptor_for_testing("qwen3_5"));
        std::printf("OK qwen3_5\n");
    } catch (const std::exception& e) { std::printf("FAIL: %s\n", e.what()); return 1; }
}
