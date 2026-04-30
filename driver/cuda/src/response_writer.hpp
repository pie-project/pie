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

}  // namespace pie_cuda_driver::response
