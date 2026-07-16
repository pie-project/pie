#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "masked_distribution.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n",                    \
                         __FILE__, __LINE__, #cond);                      \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

#define CHECK_EQ(lhs, rhs) CHECK((lhs) == (rhs))

void test_mask_excludes_higher_probability_tokens() {
    const std::vector<float> logits = {0.0f, 3.0f, 1.0f, 0.0f};
    // Allowed: token 0, then tokens 2 and 3. Token 1 is the unmasked winner.
    const std::vector<std::uint32_t> mask = {0u, 1u, 1u, 2u};

    const auto result = pie_cuda_driver::masked_top_k_distribution(
        logits, mask, 1.0f, 2);

    CHECK_EQ(result.token_ids.size(), 2u);
    CHECK_EQ(result.token_ids[0], 2u);
    CHECK_EQ(result.token_ids[1], 0u);
    const float denominator = std::exp(1.0f) + 2.0f;
    CHECK(std::fabs(result.probabilities[0] - std::exp(1.0f) / denominator) < 1e-6f);
    CHECK(std::fabs(result.probabilities[1] - 1.0f / denominator) < 1e-6f);
}

void test_top_k_is_clamped_to_allowed_tokens() {
    const std::vector<float> logits = {0.1f, 0.2f, 0.3f};
    const std::vector<std::uint32_t> mask = {1u, 1u};  // token 1 only

    const auto result = pie_cuda_driver::masked_top_k_distribution(
        logits, mask, 1.0f, 8);

    CHECK_EQ(result.token_ids.size(), 1u);
    CHECK_EQ(result.token_ids[0], 1u);
    CHECK(std::fabs(result.probabilities[0] - 1.0f) < 1e-6f);
}

void test_empty_mask_is_supported_by_the_pure_helper() {
    const std::vector<float> logits = {0.0f, 2.0f, 1.0f};

    const auto result = pie_cuda_driver::masked_top_k_distribution(
        logits, {}, 1.0f, 2);

    CHECK_EQ(result.token_ids.size(), 2u);
    CHECK_EQ(result.token_ids[0], 1u);
    CHECK_EQ(result.token_ids[1], 2u);
    const float denominator = std::exp(2.0f) + std::exp(1.0f) + 1.0f;
    CHECK(std::fabs(result.probabilities[0] - std::exp(2.0f) / denominator) < 1e-6f);
    CHECK(std::fabs(result.probabilities[1] - std::exp(1.0f) / denominator) < 1e-6f);
}

void test_zero_length_runs_preserve_brle_parity() {
    const std::vector<float> logits = {0.1f, 0.2f, 0.7f};
    // false=0, true=0, false=2, true=1: token 2 only.
    const std::vector<std::uint32_t> mask = {0u, 0u, 2u, 1u};

    const auto result = pie_cuda_driver::masked_top_k_distribution(
        logits, mask, 1.0f, 3);

    CHECK_EQ(result.token_ids.size(), 1u);
    CHECK_EQ(result.token_ids[0], 2u);
}

void test_empty_allowed_set_fails_closed() {
    const std::vector<float> logits = {0.4f, 0.6f};
    const std::vector<std::uint32_t> mask = {2u};  // false for both tokens

    bool threw = false;
    try {
        (void)pie_cuda_driver::masked_top_k_distribution(
            logits, mask, 1.0f, 1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_mask_before_softmax_avoids_allowed_underflow() {
    const std::vector<float> logits = {0.0f, -1000.0f, -1001.0f};
    // Exclude the dominant token 0. A full-vocabulary softmax would underflow
    // both allowed values before filtering.
    const std::vector<std::uint32_t> mask = {1u, 2u};

    const auto result = pie_cuda_driver::masked_top_k_distribution(
        logits, mask, 1.0f, 2);

    CHECK_EQ(result.token_ids.size(), 2u);
    CHECK_EQ(result.token_ids[0], 1u);
    CHECK_EQ(result.token_ids[1], 2u);
    CHECK(result.probabilities[0] > result.probabilities[1]);
    CHECK(std::fabs(
        result.probabilities[0] + result.probabilities[1] - 1.0f) < 1e-6f);
}

void test_equal_logits_tie_break_by_lower_token_id() {
    const std::vector<float> logits = {1.0f, 4.0f, 4.0f};
    const std::vector<std::uint32_t> mask = {1u, 2u};

    const auto result = pie_cuda_driver::masked_top_k_distribution(
        logits, mask, 0.6f, 1);

    CHECK_EQ(result.token_ids.size(), 1u);
    CHECK_EQ(result.token_ids[0], 1u);
}

}  // namespace

int main() {
    test_mask_excludes_higher_probability_tokens();
    test_top_k_is_clamped_to_allowed_tokens();
    test_empty_mask_is_supported_by_the_pure_helper();
    test_zero_length_runs_preserve_brle_parity();
    test_empty_allowed_set_fails_closed();
    test_mask_before_softmax_avoids_allowed_underflow();
    test_equal_logits_tie_break_by_lower_token_id();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED (%d checks)\n", g_failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
