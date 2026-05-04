#pragma once

// BPIS response writers (flat + msgpack). See `runtime/src/inference/...`
// for the wire format the runtime expects on the response side of the
// shmem channel.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "sampler.hpp"

namespace pie_portable_driver {

// True if any per-request output requires the msgpack response path:
// special-sampler payloads (Distribution / RawLogits / Logprob /
// Logprobs / Entropy) or variable-length token lists from M8 speculative
// decoding (anything other than length-1 `tokens` triggers msgpack since
// the flat schema requires a uniform per-request count to be encoded too).
bool needs_msgpack_mode(const std::vector<SamplerOutput>& outs);

// Emit a `BPIS` msgpack-mode response with one ForwardPassResponse per
// request. Field shapes mirror `runtime/src/inference/request.rs` and
// Pie's Python `write_response` (rmp_serde + serde derive accepts maps
// with field-name keys).
std::size_t write_msgpack_response(std::span<std::uint8_t> dst,
                                   const std::vector<SamplerOutput>& outs);

// Emit a `BPIS` flat-mode response: 16-byte header + per-request token
// counts + concatenated tokens.
std::size_t write_flat_response(std::span<std::uint8_t> dst,
                                std::span<const std::uint32_t> tokens_per_req,
                                std::span<const std::uint32_t> all_tokens);

}  // namespace pie_portable_driver
