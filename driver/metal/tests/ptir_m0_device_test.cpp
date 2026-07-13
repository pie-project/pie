#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "mtl4_context.hpp"
#include "observability.hpp"
#include "pipeline/interp.hpp"

using namespace pie::metal;
using namespace pie::metal::pipeline;

namespace {

struct RngCase {
    std::uint32_t key;
    std::uint32_t counter;
    std::uint32_t index;
    std::uint32_t reserved;
};

struct ReductionResult {
    std::uint32_t sum_bits;
    std::uint32_t max_bits;
    std::uint32_t min_bits;
    std::uint32_t argmax;
};

int g_pass = 0;
int g_fail = 0;

void expect(bool condition, const std::string& message) {
    if (condition) {
        ++g_pass;
        std::printf("  PASS  %s\n", message.c_str());
    } else {
        ++g_fail;
        std::printf("  FAIL  %s\n", message.c_str());
    }
}

std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

SlotHandle handle(const SharedStorage& storage) {
    return SlotHandle{
        .buffer = storage.native_buffer,
        .contents_ptr = storage.contents,
        .gpu_address = storage.gpu_address,
        .offset = 0,
        .size = storage.size,
    };
}

}  // namespace

int main() {
    std::string kernels_dir;
    if (const char* value = std::getenv("PIE_METAL_KERNELS_DIR")) {
        kernels_dir = value;
    }
#ifdef PIE_METAL_KERNELS_DIR_DEFAULT
    if (kernels_dir.empty()) kernels_dir = PIE_METAL_KERNELS_DIR_DEFAULT;
#endif
    expect(!kernels_dir.empty(), "kernel directory resolved");

    auto context = RawMetalContext::create(4u << 20);
    if (!context) {
        expect(false, "RawMetalContext created");
        return 1;
    }
    const std::string pointer_source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void ptir_pointer_probe(const device ulong* addresses [[buffer(0)]],
                               device uint* output [[buffer(1)]],
                               uint gid [[thread_position_in_grid]]) {
  if (gid != 0) return;
  const device uint* value =
      reinterpret_cast<const device uint*>(addresses[0]);
  output[0] = value[0];
}
)MSL";
    std::string pointer_error;
    const Pso pointer_probe = context->compile_ptir_pso(
        pointer_source, "ptir_pointer_probe", &pointer_error);
    expect(
        pointer_probe.valid(),
        "Metal accepts lane-table GPU-address pointer indirection (" +
            pointer_error + ")");

    const std::string source = kernels_dir + "/ptir_m0.metal";
    std::string error;
    const Pso rng =
        context->compile_ptir_pso_from_file(source, "ptir_m0_rng_vectors", &error);
    expect(rng.valid(), "strict PTIR RNG PSO compiled (" + error + ")");
    const Pso reduce =
        context->compile_ptir_pso_from_file(source, "ptir_m0_reduce32", &error);
    expect(reduce.valid(), "strict PTIR reduction PSO compiled (" + error + ")");
    const Pso channel = context->compile_ptir_pso_from_file(
        source, "ptir_m0_channel_self_loop", &error);
    expect(channel.valid(), "strict PTIR channel PSO compiled (" + error + ")");
    expect(context->last_ptir_compile_disabled_fast_math(),
           "PTIR compile path explicitly disabled fast math");
    if (!rng.valid() || !reduce.valid() || !channel.valid()) return 1;

    constexpr RngCase rng_cases[] = {
        {0x00000000u, 0x00000000u, 0u, 0u},
        {0x00000001u, 0x00000000u, 0u, 0u},
        {0x000004d2u, 0x00000000u, 0u, 0u},
        {0x000004d2u, 0x00000000u, 1u, 0u},
        {0xffffffffu, 0xffffffffu, 31u, 0u},
        {0x12345678u, 0x9abcdef0u, 7u, 0u},
    };
    constexpr std::uint64_t expected_hashes[] = {
        0x0000000000000000ull,
        0x52375cd73dbed523ull,
        0x2db56ca5bfd5b704ull,
        0x2db56ca5bfd5b704ull,
        0x78a9666a39c1a1b5ull,
        0x3b8823c5eac7f534ull,
    };
    constexpr std::uint32_t expected_uniform_bits[] = {
        0x3f370fb2u,
        0x3f602672u,
        0x3ebcc971u,
        0x3e682006u,
        0x3f490c40u,
        0x3d2110a8u,
    };

    SlotHandle rng_input = context->create_standalone_buffer(sizeof(rng_cases));
    SlotHandle rng_hashes =
        context->create_standalone_buffer(sizeof(expected_hashes));
    SlotHandle rng_uniforms =
        context->create_standalone_buffer(sizeof(expected_uniform_bits));
    std::memcpy(rng_input.contents(), rng_cases, sizeof(rng_cases));
    context->arg_bind_ordinal(6000, 0, rng_input);
    context->arg_bind_ordinal(6000, 1, rng_hashes);
    context->arg_bind_ordinal(6000, 2, rng_uniforms);

    constexpr std::uint32_t rows = 6;
    SlotHandle reduction_input =
        context->create_standalone_buffer(rows * 32 * sizeof(float));
    SlotHandle reduction_lengths =
        context->create_standalone_buffer(rows * sizeof(std::uint32_t));
    SlotHandle reduction_results =
        context->create_standalone_buffer(rows * sizeof(ReductionResult));
    auto* values = static_cast<float*>(reduction_input.contents());
    auto* lengths =
        static_cast<std::uint32_t*>(reduction_lengths.contents());
    values[0] = 1.0e20f;
    values[1] = 1.0f;
    values[2] = -1.0e20f;
    values[3] = 1.0f;
    lengths[0] = 4;
    values[32] = std::numeric_limits<float>::quiet_NaN();
    values[33] = -3.0f;
    values[34] = std::numeric_limits<float>::quiet_NaN();
    lengths[1] = 3;
    values[64] = 4.0f;
    values[65] = 7.0f;
    values[66] = 7.0f;
    values[67] = std::numeric_limits<float>::quiet_NaN();
    lengths[2] = 4;
    lengths[3] = 0;
    values[128] = -0.0f;
    values[129] = 0.0f;
    lengths[4] = 2;
    values[160] = 0.0f;
    values[161] = -0.0f;
    lengths[5] = 2;
    context->arg_bind_ordinal(6001, 0, reduction_input);
    context->arg_bind_ordinal(6001, 1, reduction_lengths);
    context->arg_bind_ordinal(6001, 2, reduction_results);

    auto state = make_platform_channel_state(DType::U32, 1, 1);
    expect(state != nullptr && state->cells.device_visible() &&
               state->words.device_visible(),
           "authoritative channel cells and words are Metal Shared storage");
    expect(state->push(Value::u32({41})), "seeded authoritative channel ring");
    const SlotHandle channel_cells = handle(state->cells);
    const SlotHandle channel_words = handle(state->words);
    context->use_external_buffer(channel_cells);
    context->use_external_buffer(channel_words);
    context->arg_bind_ordinal(6002, 0, channel_cells);
    context->arg_bind_ordinal(6002, 1, channel_words);

    context->make_resident();
    m0_timing_counters().reset_for_tests();
    context->run_step([&](StepEncoder& encoder) {
        encoder.set_pso(rng);
        encoder.set_argtable_ordinal(6000);
        encoder.dispatch(
            Grid{static_cast<std::uint32_t>(std::size(rng_cases)), 1, 1},
            Threadgroup{1, 1, 1});
        encoder.barrier(BarrierVisibility::Device);

        encoder.set_pso(reduce);
        encoder.set_argtable_ordinal(6001);
        encoder.dispatch(Grid{32, rows, 1}, Threadgroup{32, 1, 1});
        encoder.barrier(BarrierVisibility::Device);

        encoder.set_pso(channel);
        encoder.set_argtable_ordinal(6002);
        encoder.dispatch(Grid{1, 1, 1}, Threadgroup{1, 1, 1});
    });

    expect(std::memcmp(
               rng_hashes.contents(),
               expected_hashes,
               sizeof(expected_hashes)) == 0,
           "device SplitMix64 matches pinned Rust golden hashes bitwise");
    expect(std::memcmp(
               rng_uniforms.contents(),
               expected_uniform_bits,
               sizeof(expected_uniform_bits)) == 0,
           "device hash-uniform matches pinned Rust float bits");

    const auto* reduced =
        static_cast<const ReductionResult*>(reduction_results.contents());
    expect(reduced[0].sum_bits == bits(2.0f),
           "device cancellation sum follows width-32 tree order");
    expect(reduced[0].max_bits == bits(1.0e20f) &&
               reduced[0].min_bits == bits(-1.0e20f) &&
               reduced[0].argmax == 0,
           "device finite max/min/argmax match golden");
    expect(reduced[1].max_bits == bits(-3.0f) &&
               reduced[1].min_bits == bits(-3.0f) &&
               reduced[1].argmax == 1,
           "device reductions implement pinned NaN semantics");
    expect(reduced[2].max_bits == bits(7.0f) &&
               reduced[2].min_bits == bits(4.0f) &&
               reduced[2].argmax == 1,
           "device argmax keeps the lower index on ties");
    expect(reduced[3].sum_bits == bits(0.0f) &&
               reduced[3].max_bits == bits(-INFINITY) &&
               reduced[3].min_bits == bits(INFINITY) &&
               reduced[3].argmax == 0,
           "device empty reductions use canonical identities");
    expect(
        reduced[4].max_bits == bits(0.0f) &&
            reduced[4].min_bits == bits(-0.0f) &&
            reduced[5].max_bits == bits(0.0f) &&
            reduced[5].min_bits == bits(-0.0f),
        "device max/min preserve Rust signed-zero semantics in both operand orders");

    expect(state->head() == 1 && state->tail() == 2 &&
               state->front().u[0] == 42,
           "device self-loop and host interpreter share one channel authority");
    context->release_external_buffer(channel_cells);
    context->release_external_buffer(channel_words);
    for (std::size_t iteration = 0; iteration < 64; ++iteration) {
        auto transient = make_platform_channel_state(
            DType::U32, iteration + 1, 1);
        const SlotHandle transient_cells = handle(transient->cells);
        const SlotHandle transient_words = handle(transient->words);
        context->use_external_buffer(transient_cells);
        context->use_external_buffer(transient_words);
        context->release_external_buffer(transient_cells);
        context->release_external_buffer(transient_words);
    }
    expect(
        context->external_buffer_count() == 0,
        "external residency remains bounded across growth/close stress");
    expect(m0_timing_counters().snapshot().forward_wait_samples == 1,
           "forward-wait counter records the command-buffer wait");

    std::printf(
        "\n==== ptir_m0_device_test: %d passed, %d failed ====\n",
        g_pass,
        g_fail);
    return g_fail == 0 ? 0 : 1;
}
