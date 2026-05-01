#include "response.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

#include "msgpack.hpp"

namespace pie_portable_driver {

namespace {

constexpr std::uint32_t RESP_MAGIC        = 0x42504953;  // 'BPIS'
constexpr std::uint32_t RESP_MODE_FLAT    = 0;
constexpr std::uint32_t RESP_MODE_MSGPACK = 1;

// Shared 16-byte BPIS response header. Both flat and msgpack responses
// start with this; only the `mode` field and the meaning of `total_tokens`
// differ. Returns the header size for chaining.
constexpr std::size_t BPIS_HEADER_SIZE = 16;

inline void write_bpis_header(std::span<std::uint8_t> dst,
                              std::uint32_t mode,
                              std::uint32_t n_req,
                              std::uint32_t total_tokens) {
    if (dst.size() < BPIS_HEADER_SIZE) {
        throw std::runtime_error("response: dst too small for BPIS header");
    }
    auto write_u32 = [](std::uint8_t* p, std::uint32_t v) {
        std::memcpy(p, &v, 4);
    };
    write_u32(dst.data() + 0,  RESP_MAGIC);
    write_u32(dst.data() + 4,  mode);
    write_u32(dst.data() + 8,  n_req);
    write_u32(dst.data() + 12, total_tokens);
}

}  // namespace

bool needs_msgpack_mode(const std::vector<SamplerOutput>& outs) {
    for (const auto& o : outs) {
        for (const auto& s : o.special_slots) {
            if (s.has_dist || !s.raw_logits.empty() || !s.logprobs.empty()
                || s.has_entropy) {
                return true;
            }
        }
    }
    return false;
}

std::size_t write_msgpack_response(std::span<std::uint8_t> dst,
                                   const std::vector<SamplerOutput>& outs) {
    write_bpis_header(dst, RESP_MODE_MSGPACK,
                      static_cast<std::uint32_t>(outs.size()),
                      /*total_tokens=*/ 0);  // unused in msgpack mode

    MsgpackWriter w(dst.subspan(BPIS_HEADER_SIZE));
    // {"results": [ ... ]}
    w.map_header(1);
    w.str("results");
    w.array_header(outs.size());
    for (const auto& o : outs) {
        // Aggregate the per-slot special-sampler payloads for this
        // request into the flat ForwardPassResponse field shape.
        std::size_t n_dists = 0, n_raw = 0, n_logprobs = 0, n_entropies = 0;
        for (const auto& s : o.special_slots) {
            if (s.has_dist)               ++n_dists;
            if (!s.raw_logits.empty())    ++n_raw;
            if (!s.logprobs.empty())      ++n_logprobs;
            if (s.has_entropy)            ++n_entropies;
        }

        w.map_header(7);

        w.str("tokens");
        w.array_u32(std::span<const std::uint32_t>(o.tokens));

        w.str("dists");
        w.array_header(n_dists);
        for (const auto& s : o.special_slots) {
            if (!s.has_dist) continue;
            // Each entry is a 2-tuple (Vec<u32>, Vec<f32>) — array of size 2.
            w.array_header(2);
            w.array_u32(std::span<const std::uint32_t>(s.dist_ids));
            w.array_f32(std::span<const float>(s.dist_vals));
        }

        w.str("logits");
        w.array_header(n_raw);
        for (const auto& s : o.special_slots) {
            if (s.raw_logits.empty()) continue;
            w.bin(s.raw_logits.data(), s.raw_logits.size());
        }

        w.str("logprobs");
        w.array_header(n_logprobs);
        for (const auto& s : o.special_slots) {
            if (s.logprobs.empty()) continue;
            w.array_f32(std::span<const float>(s.logprobs));
        }

        w.str("entropies");
        w.array_header(n_entropies);
        for (const auto& s : o.special_slots) {
            if (!s.has_entropy) continue;
            w.f32(s.entropy);
        }

        // spec_tokens / spec_positions — empty (drafter for next iter
        // is a separate concern; the verifier emits accepted tokens
        // through `tokens` directly).
        w.str("spec_tokens");
        w.array_header(0);
        w.str("spec_positions");
        w.array_header(0);
    }
    return BPIS_HEADER_SIZE + w.size();
}

std::size_t write_flat_response(std::span<std::uint8_t> dst,
                                std::span<const std::uint32_t> tokens_per_req,
                                std::span<const std::uint32_t> all_tokens) {
    const std::size_t n_req        = tokens_per_req.size();
    const std::size_t total_tokens = all_tokens.size();
    const std::size_t need = BPIS_HEADER_SIZE + n_req * 4 + total_tokens * 4;

    if (dst.size() < need) {
        throw std::runtime_error(
            "response: dst too small (have " + std::to_string(dst.size()) +
            ", need " + std::to_string(need) + ")");
    }

    write_bpis_header(dst, RESP_MODE_FLAT,
                      static_cast<std::uint32_t>(n_req),
                      static_cast<std::uint32_t>(total_tokens));
    std::memcpy(dst.data() + BPIS_HEADER_SIZE,
                tokens_per_req.data(), n_req * 4);
    std::memcpy(dst.data() + BPIS_HEADER_SIZE + n_req * 4,
                all_tokens.data(), total_tokens * 4);
    return need;
}

}  // namespace pie_portable_driver
