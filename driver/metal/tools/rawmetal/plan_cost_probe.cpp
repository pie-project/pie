// What does this checkpoint cost at boot, and how much of it is arithmetic?
//
// A load plan is either extent writes -- bytes moved from the file into a heap
// -- or transforms, which read every weight and write a different encoding of
// it. The two are not comparable: the first is bounded by the SSD and the
// second by the CPU, and "the load is slow" means something different in each
// case. Printing the split is what turns "would an offline conversion help"
// into a number rather than an opinion.
//
// The histogram is the loader's own `stats_json`, not a table built here. The
// plan carries it precisely so a driver-side probe cannot fall out of step with
// the instruction names, and a histogram reassembled from the instruction
// stream would be exactly that mistake.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "loader/load_plan.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: plan_cost_probe <checkpoint-dir> <model_type> "
                    "[n_layers] [n_kv_shared] [quant_bits] [quant_group]\n");
        return 2;
    }
    const std::string dir = argv[1];
    const std::string model_type = argv[2];
    pie::metal::model::ContractFacts facts{};
    if (argc > 3) facts.num_hidden_layers = std::uint32_t(std::atoi(argv[3]));
    if (argc > 4) facts.num_kv_shared_layers = std::uint32_t(std::atoi(argv[4]));
    if (argc > 5) facts.quant_bits = std::uint32_t(std::atoi(argv[5]));
    if (argc > 6) facts.quant_group_size = std::uint32_t(std::atoi(argv[6]));
    try {
        // The descriptor is what crosses now. A real boot forwards the
        // artifact's own `pie.model/1`; this states the handful of fields a
        // family reads and lets the same helper the tests use synthesize one.
        auto plan = pie::metal::compile_load_plan(
            dir, pie::metal::metal_device_target(),
            pie::metal::descriptor_for_testing(model_type, facts));
        const auto& view = plan.view();
        const auto gib = [](std::uint64_t bytes) {
            return double(bytes) / double(1ull << 30);
        };
        std::printf("%s\n", pie_loader::bytes_to_string(view.summary).c_str());
        std::printf("  read %.2f GiB from the checkpoint, write %.2f GiB to the device\n",
                    gib(view.memory.checkpoint_read_bytes), gib(view.memory.device_write_bytes));
        std::printf("  resident %.2f GiB, temporary peak %.2f GiB, transform scratch peak %.2f GiB\n",
                    gib(view.memory.persistent_bytes), gib(view.memory.temporary_peak_bytes),
                    gib(view.memory.transform_scratch_peak_bytes));
        std::printf("  %s\n", pie_loader::bytes_to_string(view.stats_json).c_str());
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
