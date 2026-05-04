#pragma once

// Sampler type IDs as encoded in the BPIQ wire's `A_SAMPLER_TYPES` array.
// Authoritative definition lives in the runtime
// (`runtime/src/inference/request.rs::Sampler::type_id`); this header is
// the C++ mirror so the driver can dispatch by named constant rather than
// scattered magic numbers.
//
// Two convenience predicates capture the dispatch tables that the request
// handler uses:
//
//   * `is_token_sampler(t)` — produces a single sampled token at the
//     slot's row; consumes the row-indexed sampling kernel output. The
//     flat-response path requires *every* slot in the batch to satisfy
//     this predicate.
//
//   * `is_msgpack_only(t)` — non-token output (Dist/RawLogits/Logprob/
//     Logprobs/Entropy). Triggers the msgpack response path; per-slot
//     payload comes from the matching sub-pass kernel rather than the
//     sampler kernel.
//
// Type 6 (Embedding) is reserved on the wire but not yet produced by any
// driver — neither predicate covers it.

#include <cstdint>

namespace pie_cuda_driver {

enum class SamplerType : std::uint32_t {
    Dist        = 0,
    Multinomial = 1,
    TopK        = 2,
    TopP        = 3,
    MinP        = 4,
    TopKTopP    = 5,
    Embedding   = 6,
    RawLogits   = 7,
    Logprob     = 8,
    Logprobs    = 9,
    Entropy     = 10,
};

constexpr bool is_token_sampler(std::uint32_t t) noexcept {
    return t >= static_cast<std::uint32_t>(SamplerType::Multinomial)
        && t <= static_cast<std::uint32_t>(SamplerType::TopKTopP);
}

constexpr bool is_msgpack_only(std::uint32_t t) noexcept {
    return t == static_cast<std::uint32_t>(SamplerType::Dist)
        || (t >= static_cast<std::uint32_t>(SamplerType::RawLogits)
            && t <= static_cast<std::uint32_t>(SamplerType::Entropy));
}

constexpr bool is_token_sampler(SamplerType t) noexcept {
    return is_token_sampler(static_cast<std::uint32_t>(t));
}
constexpr bool is_msgpack_only(SamplerType t) noexcept {
    return is_msgpack_only(static_cast<std::uint32_t>(t));
}

}  // namespace pie_cuda_driver
