#pragma once

// Sampling for the Metal (MLX) driver.
//
// v1 implements the token-producing samplers with MLX ops (argmax /
// temperature-softmax / top-k / top-p / min-p / categorical) — all running
// on-device. A fused custom Metal kernel (`fast::metal_kernel`) can replace
// the hot path later; the MLX path is the correctness reference.
//
// Wire-format sampler type IDs mirror driver/portable/src/sampler.hpp and
// runtime/src/inference/request.rs.

#include <cstdint>
#include <vector>

#include "ops/tensor.hpp"

namespace pie_metal_driver::sampling {

enum class SamplerType : std::uint32_t {
    Distribution = 0,  // special (M10)
    Multinomial  = 1,
    TopK         = 2,
    TopP         = 3,
    MinP         = 4,
    TopKTopP     = 5,
    RawLogits    = 7,  // special (M10)
    Logprob      = 8,  // special (M10)
    Logprobs     = 9,  // special (M10)
    Entropy      = 10, // special (M10)
};

struct SamplerParams {
    SamplerType   type        = SamplerType::Multinomial;
    float         temperature = 1.0f;
    std::uint32_t top_k       = 0;     // 0 => no top-k filter
    float         top_p       = 1.0f;  // 1 => no nucleus filter
    float         min_p       = 0.0f;  // 0 => no min-p filter
    std::uint32_t seed        = 0;     // 0 => derive from base offset
};

// Temperature below this collapses a token-producing sampler to argmax.
constexpr float kGreedyTempEps = 1e-5f;

// Sample one token per row of `logits` ([n_slots, vocab]). `params` has one
// entry per slot. `base_seed` mixes determinism when a slot's own seed is 0
// (the executor passes a per-fire counter). Returns `n_slots` token ids.
std::vector<std::uint32_t> sample_tokens(
    const Tensor& logits,
    const std::vector<SamplerParams>& params,
    std::uint64_t base_seed);

// Same sampling graph as `sample_tokens`, but returns the token ids as a lazy
// device array [n_slots] (uint32) WITHOUT a host readback. The driver-owned
// pipelined decode loop (DECODE_N) feeds this token straight into the next
// step's embedding so the autoregressive feedback never leaves the GPU, which
// is what lets `mx::async_eval` hide the per-token dispatch bubble. Callers
// that need host ids should use `sample_tokens` (which drains this).
Tensor sample_token_device(
    const Tensor& logits,
    const std::vector<SamplerParams>& params,
    std::uint64_t base_seed);

}  // namespace pie_metal_driver::sampling
