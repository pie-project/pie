#pragma once

// Host-side sampler suite. Mirrors `pie_driver/model/common.py::_execute_sampler`
// for the token-producing sampler types (1=Multinomial, 2=TopK, 3=TopP,
// 4=MinP, 5=TopKTopP). Special samplers (0=Distribution, 7=RawLogits,
// 8=Logprob, 9=Logprobs, 10=Entropy) need the msgpack response writer
// and land in M10.
//
// We sample on the host (after `ggml_backend_tensor_get` pulls logits to
// CPU memory). This is the same boundary the Python driver uses; for raw
// throughput a GPU-side port of flashinfer's sampling kernels is M11.
//
// Determinism: per-row seeded PCG32. Same input + same seed → same token.
// Within-driver determinism is guaranteed; bit-exact parity with Python
// flashinfer is NOT — different RNG (we use PCG32; flashinfer uses its
// own scheme), different float-summation order in softmax / top-p prefix
// sum.

#include <cstdint>
#include <span>
#include <vector>

namespace pie_ggml_driver {

// Wire-format sampler type IDs (must match `runtime/src/inference/request.rs`).
enum class SamplerType : std::uint32_t {
    Distribution = 0,  // M10 — needs msgpack
    Multinomial  = 1,
    TopK         = 2,
    TopP         = 3,
    MinP         = 4,
    TopKTopP     = 5,
    // 6 reserved
    RawLogits    = 7,  // M10
    Logprob      = 8,  // M10
    Logprobs     = 9,  // M10
    Entropy      = 10, // M10
};

struct SamplerParams {
    SamplerType   type        = SamplerType::Multinomial;
    float         temperature = 1.0f;
    std::uint32_t top_k       = 0;     // 0 → no top-k filter
    float         top_p       = 1.0f;
    float         min_p       = 0.0f;
    std::uint32_t seed        = 0;     // 0 → use process RNG
    // Logprob / Logprobs label set (per slot). For Logprob this is a
    // single token id; for Logprobs it's the K labels to evaluate.
    std::vector<std::uint32_t> labels;
};

// One slot's sampling output. Token-producing samplers (Multinomial,
// TopK, TopP, MinP, TopKTopP) write `token`. Special samplers
// (Distribution / RawLogits / Logprob / Logprobs / Entropy) write into
// the corresponding special-output field.
struct SlotOutput {
    std::uint32_t              token = 0;

    bool                       has_dist = false;
    std::vector<std::uint32_t> dist_ids;
    std::vector<float>         dist_vals;

    std::vector<std::uint8_t>  raw_logits;       // RawLogits

    std::vector<float>         logprobs;         // Logprob (len 1) / Logprobs (len K)

    bool                       has_entropy = false;
    float                      entropy = 0.0f;
};

// Per-request aggregated output. `tokens` carries one or more accepted
// tokens (length 1 for plain sampling; 1..n_drafts+1 for spec decode).
// Special-sampler payloads (one per sampling slot) ride along too.
struct SamplerOutput {
    std::vector<std::uint32_t>   tokens;               // accepted tokens
    std::vector<SlotOutput>      special_slots;        // empty unless special samplers are used
};

// Run the requested sampler on `logits` (length = vocab_size) and write
// to `out`. Token-producing types (Multinomial / TopK / TopP / MinP /
// TopKTopP) fill `out.token`; special types fill the special-output
// fields. Throws on truly unsupported types.
void sample_slot(const float* logits,
                 std::int32_t vocab_size,
                 const SamplerParams& params,
                 SlotOutput& out);

// Convenience wrapper for token-only callers. Throws if `params.type` is
// a special sampler.
std::uint32_t sample_token(const float* logits,
                           std::int32_t vocab_size,
                           const SamplerParams& params);

// Host-side finalize for the GPU non-greedy fast path. The graph
// emitted top-K per slot already sorted descending and softmaxed at
// the requested temperature. This routine applies the per-slot top-K
// cap / top-P / min-P cutoff, renormalizes, and categorical-samples
// one token. Multinomial without filtering is not supported here
// (needs the full distribution); greedy is handled by ggml_argmax.
//
// `seed_xor` mixes extra entropy in when `params.seed == 0` — pass
// the global slot index so different slots draw different streams.
std::uint32_t sample_token_from_topk(
    const std::int32_t*  top_idx,    // [K] descending-sorted indices
    const float*         top_probs,  // [K] matching softmaxed probs
    std::int32_t         K,
    const SamplerParams& params,
    std::uint64_t        seed_xor);

// Apply a BRLE-encoded logit mask in place. Sets logits[v] = -INF for
// vocab indices the mask says are forbidden. Wire format mirrors
// `runtime/src/inference/brle.rs`:
//   - Run lengths alternate false/true starting with false
//   - Empty `runs` (n_runs == 0) means "no constraint" — no-op
//   - Beyond the final run, vocab is implicitly false (masked)
//
// `runs` is the sub-slice of `flattened_masks` that belongs to this
// request (mask_indptr[r] : mask_indptr[r+1]).
void apply_brle_logit_mask(float* logits,
                           std::int32_t vocab_size,
                           const std::uint32_t* runs,
                           std::size_t n_runs);

}  // namespace pie_ggml_driver
