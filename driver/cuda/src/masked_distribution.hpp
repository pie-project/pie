#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace pie_cuda_driver {

struct MaskedDistribution {
    std::vector<std::uint32_t> token_ids;
    std::vector<float> probabilities;
};

// Select the highest-probability tokens allowed by a BRLE logit mask.
// BRLE runs alternate false/true starting with false. An empty mask means
// unconstrained. Softmax is evaluated only over allowed logits so an illegal
// dominant token cannot underflow the allowed distribution. Returned
// probabilities are normalized over every allowed token, not merely over the
// returned top-K subset.
MaskedDistribution masked_top_k_distribution(
    std::span<const float> logits,
    std::span<const std::uint32_t> mask_runs,
    float temperature,
    std::size_t top_k);

}  // namespace pie_cuda_driver
