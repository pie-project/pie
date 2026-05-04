#pragma once

// BPIS flat-mode response encoder. Mirrors `pie_driver/shmem_schema.py`'s
// `write_response_v2` fast path:
//
//     [16-byte header]
//       0:  u32 magic = 0x42504953  ('BPIS')
//       4:  u32 mode  = 0           (flat token-only)
//       8:  u32 num_requests
//       12: u32 total_tokens
//     [u32 × num_requests]   per-request token counts
//     [u32 × total_tokens]   concatenated token ids
//
// The msgpack fallback (mode = 1) is for distributions, logits, logprobs,
// entropies, spec chains. M2 lands the fallback; for M1.3 we only emit the
// flat path.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace pie_cuda_driver::response {

inline constexpr std::uint32_t MAGIC = 0x42504953;  // 'BPIS'
inline constexpr std::uint32_t MODE_FLAT = 0;
inline constexpr std::uint32_t MODE_MSGPACK = 1;
inline constexpr std::size_t HEADER_SIZE = 16;

// Compute the byte size of a flat response with the given per-request token
// counts. Returns 0 on overflow.
std::size_t flat_response_size(std::span<const std::uint32_t> per_request_counts);

// Write a flat-mode response into `buf`. Returns bytes written. Throws if
// `buf` is too small. `tokens` is the concatenated tokens array; its length
// must equal sum(per_request_counts).
std::size_t write_flat_response(
    std::span<std::uint8_t> buf,
    std::span<const std::uint32_t> per_request_counts,
    std::span<const std::uint32_t> tokens);

// Slow-path msgpack response. The Python driver writes
//   {"results": [{"tokens": [...], "dists": [...], "logits": [...],
//                  "logprobs": [...], "entropies": [...],
//                  "spec_tokens": [...], "spec_positions": [...]}, ...]}
// when any sampler type is in {Logprob, Logprobs, Entropy, RawLogits, …} or
// the request asks for spec output.
//
// First cut (this PR): we emit the same map shape but only `tokens` is
// populated; the other fields are empty arrays. Inferlets that genuinely
// require distributions / logits / logprobs / entropies will receive empty
// data — they won't crash, but the feature is silently degraded until the
// kernels that compute those values are added.
struct PerRequestMsgpack {
    std::span<const std::uint32_t> tokens;
    // One entry per sampling slot for this request. For `Sampler::RawLogits`
    // slots the entry is `vocab * 4` bytes of native-endian f32 logits;
    // empty for slots that didn't request raw logits. The runtime reads
    // them in slot order, so passing more or fewer entries than slots
    // misaligns parsing — keep the count == sampling slots even when most
    // entries are empty.
    std::vector<std::vector<std::uint8_t>> logits;
    // One fp32 entropy per `Sampler::Entropy` slot. Empty when this
    // request didn't request entropy.
    std::vector<float> entropies;
    // For Sampler::Logprob (length 1 each) / Logprobs (length K each):
    // one inner vec per sampling slot, log p(token_id) for each token id
    // the inferlet asked about.
    std::vector<std::vector<float>> logprobs;
    // For Sampler::Dist: one (token_ids, probs) pair per sampling slot.
    // Both inner vectors have the same length K (top-K, sorted by prob
    // descending). K = top_k or vocab_size if top_k == 0. Empty pairs
    // for slots that didn't request a distribution.
    std::vector<std::pair<std::vector<std::uint32_t>, std::vector<float>>> dists;
};

std::size_t write_msgpack_response(
    std::span<std::uint8_t> buf,
    std::span<const PerRequestMsgpack> per_request);

}  // namespace pie_cuda_driver::response
