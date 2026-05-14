#include "response_writer.hpp"

#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

namespace pie_cuda_driver::response {

std::size_t flat_response_size(std::span<const std::uint32_t> per_request_counts) {
    std::uint64_t total = 0;
    for (auto c : per_request_counts) total += c;
    return HEADER_SIZE
         + 4 * per_request_counts.size()
         + 4 * total;
}

std::size_t write_flat_response(
    std::span<std::uint8_t> buf,
    std::span<const std::uint32_t> per_request_counts,
    std::span<const std::uint32_t> tokens)
{
    const auto need = flat_response_size(per_request_counts);
    if (buf.size() < need) {
        throw std::runtime_error("BPIS response buffer too small: have " +
                                 std::to_string(buf.size()) + ", need " +
                                 std::to_string(need));
    }
    const std::uint64_t expected_tokens =
        std::accumulate(per_request_counts.begin(), per_request_counts.end(), 0ULL);
    if (tokens.size() != expected_tokens) {
        throw std::runtime_error("BPIS response: tokens.size() != sum(per_request_counts)");
    }

    std::uint32_t header[4] = {
        MAGIC,
        MODE_FLAT,
        static_cast<std::uint32_t>(per_request_counts.size()),
        static_cast<std::uint32_t>(tokens.size()),
    };
    std::memcpy(buf.data(), header, HEADER_SIZE);

    std::size_t off = HEADER_SIZE;
    std::memcpy(buf.data() + off,
                per_request_counts.data(),
                per_request_counts.size() * sizeof(std::uint32_t));
    off += per_request_counts.size() * sizeof(std::uint32_t);

    std::memcpy(buf.data() + off,
                tokens.data(),
                tokens.size() * sizeof(std::uint32_t));
    off += tokens.size() * sizeof(std::uint32_t);

    return off;
}

// =============================================================================
// Minimal msgpack encoder (only the subset we need: fixmap/map16/map32,
// fixarray/array16/array32, fixstr, uint32, bin/raw bytes).
// Spec: https://github.com/msgpack/msgpack/blob/master/spec.md
// =============================================================================

namespace {

class MsgpackWriter {
public:
    explicit MsgpackWriter(std::span<std::uint8_t> dst) : buf_(dst), off_(0) {}

    void put_u8(std::uint8_t v) {
        ensure(1);
        buf_[off_++] = v;
    }

    void put_be16(std::uint16_t v) {
        ensure(2);
        buf_[off_++] = static_cast<std::uint8_t>(v >> 8);
        buf_[off_++] = static_cast<std::uint8_t>(v & 0xff);
    }

    void put_be32(std::uint32_t v) {
        ensure(4);
        buf_[off_++] = static_cast<std::uint8_t>(v >> 24);
        buf_[off_++] = static_cast<std::uint8_t>((v >> 16) & 0xff);
        buf_[off_++] = static_cast<std::uint8_t>((v >> 8) & 0xff);
        buf_[off_++] = static_cast<std::uint8_t>(v & 0xff);
    }

    void map(std::uint32_t n) {
        if (n <= 15) {
            put_u8(0x80 | static_cast<std::uint8_t>(n));
        } else if (n <= 0xffff) {
            put_u8(0xde);
            put_be16(static_cast<std::uint16_t>(n));
        } else {
            put_u8(0xdf);
            put_be32(n);
        }
    }

    void array(std::uint32_t n) {
        if (n <= 15) {
            put_u8(0x90 | static_cast<std::uint8_t>(n));
        } else if (n <= 0xffff) {
            put_u8(0xdc);
            put_be16(static_cast<std::uint16_t>(n));
        } else {
            put_u8(0xdd);
            put_be32(n);
        }
    }

    void str(std::string_view s) {
        const auto n = s.size();
        if (n <= 31) {
            put_u8(0xa0 | static_cast<std::uint8_t>(n));
        } else if (n <= 0xff) {
            put_u8(0xd9);
            put_u8(static_cast<std::uint8_t>(n));
        } else if (n <= 0xffff) {
            put_u8(0xda);
            put_be16(static_cast<std::uint16_t>(n));
        } else {
            put_u8(0xdb);
            put_be32(static_cast<std::uint32_t>(n));
        }
        ensure(n);
        std::memcpy(buf_.data() + off_, s.data(), n);
        off_ += n;
    }

    // bin8/bin16/bin32 — raw byte string. Used for the `logits` payload.
    void bin(std::span<const std::uint8_t> bytes) {
        const auto n = bytes.size();
        if (n <= 0xff) {
            put_u8(0xc4);
            put_u8(static_cast<std::uint8_t>(n));
        } else if (n <= 0xffff) {
            put_u8(0xc5);
            put_be16(static_cast<std::uint16_t>(n));
        } else {
            put_u8(0xc6);
            put_be32(static_cast<std::uint32_t>(n));
        }
        ensure(n);
        std::memcpy(buf_.data() + off_, bytes.data(), n);
        off_ += n;
    }

    // float32 (msgpack `float 32` family code 0xca, big-endian per spec).
    void f32(float v) {
        static_assert(sizeof(float) == 4);
        std::uint32_t bits;
        std::memcpy(&bits, &v, 4);
        put_u8(0xca);
        put_be32(bits);
    }

    void uint(std::uint64_t v) {
        if (v <= 0x7f) {
            put_u8(static_cast<std::uint8_t>(v));   // positive fixint
        } else if (v <= 0xff) {
            put_u8(0xcc);
            put_u8(static_cast<std::uint8_t>(v));
        } else if (v <= 0xffff) {
            put_u8(0xcd);
            put_be16(static_cast<std::uint16_t>(v));
        } else if (v <= 0xffffffffULL) {
            put_u8(0xce);
            put_be32(static_cast<std::uint32_t>(v));
        } else {
            put_u8(0xcf);
            ensure(8);
            for (int i = 7; i >= 0; --i) {
                buf_[off_++] = static_cast<std::uint8_t>((v >> (i * 8)) & 0xff);
            }
        }
    }

    std::size_t offset() const noexcept { return off_; }

private:
    void ensure(std::size_t n) {
        if (off_ + n > buf_.size()) {
            throw std::runtime_error(
                "msgpack response buffer too small (" +
                std::to_string(off_ + n) + " > " + std::to_string(buf_.size()) + ")");
        }
    }

    std::span<std::uint8_t> buf_;
    std::size_t off_;
};

}  // namespace

std::size_t write_msgpack_response(
    std::span<std::uint8_t> buf,
    std::span<const PerRequestMsgpack> per_request)
{
    if (buf.size() < HEADER_SIZE) {
        throw std::runtime_error("msgpack response buffer < header size");
    }

    // Header (16 bytes): magic, mode=1, num_requests=R, total_tokens=0.
    // (`total_tokens` is unused on the msgpack side; mirrors Python's writer.)
    std::uint32_t header[4] = {
        MAGIC, MODE_MSGPACK,
        static_cast<std::uint32_t>(per_request.size()),
        0u,
    };
    std::memcpy(buf.data(), header, HEADER_SIZE);

    auto body = buf.subspan(HEADER_SIZE);
    MsgpackWriter w(body);

    // {"results": [...]} — one map per ctx.
    w.map(1);
    w.str("results");
    w.array(static_cast<std::uint32_t>(per_request.size()));

    for (const auto& r : per_request) {
        // {"tokens", "dists", "logits", "logprobs", "entropies",
        //  "spec_tokens", "spec_positions"}
        w.map(7);

        w.str("tokens");
        w.array(static_cast<std::uint32_t>(r.tokens.size()));
        for (auto t : r.tokens) w.uint(t);

        // `dists`: array of (token_ids, probs) tuples. Rust serde encodes a
        // tuple as a fixed-size msgpack array, so each entry is `array(2)`
        // wrapping two `array(K)` payloads. Empty array if no Dist slots.
        w.str("dists");
        w.array(static_cast<std::uint32_t>(r.dists.size()));
        for (const auto& [ids, probs] : r.dists) {
            w.array(2);
            w.array(static_cast<std::uint32_t>(ids.size()));
            for (auto id : ids) w.uint(id);
            w.array(static_cast<std::uint32_t>(probs.size()));
            for (float p : probs) w.f32(p);
        }
        w.str("logits");
        w.array(static_cast<std::uint32_t>(r.logits.size()));
        for (const auto& payload : r.logits) {
            w.bin(std::span<const std::uint8_t>(payload.data(), payload.size()));
        }
        w.str("logprobs");
        w.array(static_cast<std::uint32_t>(r.logprobs.size()));
        for (const auto& slot_lps : r.logprobs) {
            w.array(static_cast<std::uint32_t>(slot_lps.size()));
            for (float v : slot_lps) w.f32(v);
        }
        w.str("entropies");
        w.array(static_cast<std::uint32_t>(r.entropies.size()));
        for (float e : r.entropies) w.f32(e);
        for (const char* k : {"spec_tokens", "spec_positions"}) {
            w.str(k);
            w.array(0);
        }
    }

    return HEADER_SIZE + w.offset();
}

}  // namespace pie_cuda_driver::response
