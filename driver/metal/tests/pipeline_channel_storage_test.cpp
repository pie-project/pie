#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#include "batch/logits_convert.hpp"
#include "pipeline/interp.hpp"

using namespace pie::metal;
using namespace pie::metal::batch;
using namespace pie::metal::pipeline;

namespace {

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

}  // namespace

int main() {
    std::printf("[authoritative shared ring]\n");
    auto state = make_host_channel_state(DType::U32, 1, 1);
    expect(state != nullptr && state->valid(), "host shared ring allocated");
    expect(state->cells.size == 2 * sizeof(std::uint32_t) &&
               state->words.size == 4 * sizeof(std::uint64_t),
           "capacity-one ring has two physical cells and four words");

    // Simulate a host writer using the endpoint ABI directly. The interpreter
    // observes exactly those bytes/words; there is no deque to synchronize.
    const std::uint32_t first = 41;
    std::memcpy(state->cells.contents, &first, sizeof(first));
    auto* words = reinterpret_cast<std::uint64_t*>(state->words.contents);
    std::atomic_ref<std::uint64_t>(words[1]).store(
        1, std::memory_order_release);
    expect(state->size() == 1 && state->front().u[0] == 41,
           "host-published endpoint cell is the interpreter front");

    Value taken;
    expect(state->pop(taken) && taken.u[0] == 41 && state->head() == 1,
           "take advances the authoritative head");
    expect(state->empty() && state->current().u[0] == 41,
           "empty channel reads the last consumed physical cell");
    expect(state->push(Value::u32({42})) && state->tail() == 2,
           "put writes the next physical cell and publishes tail");
    std::uint32_t second = 0;
    std::memcpy(
        &second,
        state->cells.contents + sizeof(std::uint32_t),
        sizeof(second));
    expect(second == 42 && state->front().u[0] == 42,
           "endpoint bytes and interpreter value remain one authority");

    std::printf("[canonical host width-32 tree]\n");
    const float cancellation[] = {1.0e20f, 1.0f, -1.0e20f, 1.0f};
    const float sum = detail::canonical_reduce(
        cancellation,
        4,
        0.0f,
        [](float left, float right) { return left + right; });
    expect(bits(sum) == bits(2.0f),
           "cancellation vector follows the pinned width-32 order");
    const float nan_max[] = {
        std::numeric_limits<float>::quiet_NaN(),
        -3.0f,
        std::numeric_limits<float>::quiet_NaN(),
    };
    expect(detail::canonical_reduce(
               nan_max, 3, detail::neg_inf(), detail::canonical_max) == -3.0f,
           "canonical max ignores NaNs");
    expect(detail::argmax_row(nan_max, 3) == 1,
           "canonical argmax ignores NaNs and selects the valid index");
    const float signed_zero[] = {-0.0f, 0.0f};
    const float reversed_zero[] = {0.0f, -0.0f};
    expect(
        bits(detail::canonical_reduce(
                 signed_zero,
                 2,
                 detail::neg_inf(),
                 detail::canonical_max)) == bits(0.0f) &&
            bits(detail::canonical_reduce(
                 signed_zero,
                 2,
                 std::numeric_limits<float>::infinity(),
                 detail::canonical_min)) == bits(-0.0f) &&
            bits(detail::canonical_reduce(
                 reversed_zero,
                 2,
                 detail::neg_inf(),
                 detail::canonical_max)) == bits(0.0f) &&
            bits(detail::canonical_reduce(
                 reversed_zero,
                 2,
                 std::numeric_limits<float>::infinity(),
                 detail::canonical_min)) == bits(-0.0f),
        "canonical max/min preserve Rust signed-zero semantics in both orders");
    cptir::Shape rank3;
    rank3.dims = {2, 3, 4};
    const std::int64_t negative_values[] = {-2, -1};
    expect(
        detail::canonical_rows(rank3) == 6 &&
            detail::argmax_row_i64(negative_values, 2) == 1,
        "host oracle uses rank-3 rows and dtype-aware integer argmax");

    std::printf("[M0 conversion observability]\n");
    m0_timing_counters().reset_for_tests();
    const std::uint16_t bf16[] = {0x3f80, 0xc000, 0x7f80};
    float converted[3] = {};
    copy_bf16_to_f32(bf16, converted, 3);
    const M0TimingSnapshot timing = m0_timing_counters().snapshot();
    expect(converted[0] == 1.0f && converted[1] == -2.0f &&
               std::isinf(converted[2]),
           "bf16 conversion keeps exact widening behavior");
    expect(timing.bf16_conversion_samples == 1,
           "bf16 conversion counter records the operation");

    std::printf(
        "\n==== pipeline_channel_storage_test: %d passed, %d failed ====\n",
        g_pass,
        g_fail);
    return g_fail == 0 ? 0 : 1;
}
