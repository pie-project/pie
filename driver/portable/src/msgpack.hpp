#pragma once

// Minimal msgpack v5 writer for the BPIS msgpack-mode response payload.
//
// Wire format reference: https://github.com/msgpack/msgpack/blob/master/spec.md
// We implement only the encodings we need:
//   - fixmap / map16 / map32
//   - fixarray / array16 / array32
//   - fixstr / str8 / str16 (for field names)
//   - bin8 / bin16 / bin32 (for raw f32 logits)
//   - positive fixint / uint16 / uint32  (token IDs)
//   - float32
//
// All multi-byte values are big-endian in the wire format. Output is
// written into a caller-supplied buffer; the writer throws if it'd
// overflow.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pie_portable_driver {

class MsgpackWriter {
public:
    explicit MsgpackWriter(std::span<std::uint8_t> dst) noexcept
        : dst_(dst), pos_(0) {}

    std::size_t size() const noexcept { return pos_; }

    // ---- Headers ------------------------------------------------------------
    void map_header(std::size_t n) {
        if (n <= 15) put1(static_cast<std::uint8_t>(0x80u | n));
        else if (n <= 0xFFFF) { put1(0xde); put_be16(static_cast<std::uint16_t>(n)); }
        else { put1(0xdf); put_be32(static_cast<std::uint32_t>(n)); }
    }
    void array_header(std::size_t n) {
        if (n <= 15) put1(static_cast<std::uint8_t>(0x90u | n));
        else if (n <= 0xFFFF) { put1(0xdc); put_be16(static_cast<std::uint16_t>(n)); }
        else { put1(0xdd); put_be32(static_cast<std::uint32_t>(n)); }
    }

    // ---- Scalars ------------------------------------------------------------
    void str(std::string_view s) {
        const auto n = s.size();
        if (n <= 31) put1(static_cast<std::uint8_t>(0xa0u | n));
        else if (n <= 0xFF) { put1(0xd9); put1(static_cast<std::uint8_t>(n)); }
        else if (n <= 0xFFFF) { put1(0xda); put_be16(static_cast<std::uint16_t>(n)); }
        else { put1(0xdb); put_be32(static_cast<std::uint32_t>(n)); }
        put_bytes(s.data(), n);
    }
    void u32(std::uint32_t v) {
        if (v <= 0x7f) put1(static_cast<std::uint8_t>(v));
        else if (v <= 0xFF) { put1(0xcc); put1(static_cast<std::uint8_t>(v)); }
        else if (v <= 0xFFFF) { put1(0xcd); put_be16(static_cast<std::uint16_t>(v)); }
        else { put1(0xce); put_be32(v); }
    }
    void f32(float v) {
        std::uint32_t bits;
        std::memcpy(&bits, &v, 4);
        put1(0xca);
        put_be32(bits);
    }
    void bin(const void* data, std::size_t n) {
        if (n <= 0xFF) { put1(0xc4); put1(static_cast<std::uint8_t>(n)); }
        else if (n <= 0xFFFF) { put1(0xc5); put_be16(static_cast<std::uint16_t>(n)); }
        else { put1(0xc6); put_be32(static_cast<std::uint32_t>(n)); }
        put_bytes(data, n);
    }

    // ---- Vector helpers (header + repeated scalar) --------------------------
    void array_u32(std::span<const std::uint32_t> arr) {
        array_header(arr.size());
        for (auto v : arr) u32(v);
    }
    void array_f32(std::span<const float> arr) {
        array_header(arr.size());
        for (auto v : arr) f32(v);
    }

private:
    void need(std::size_t n) {
        if (pos_ + n > dst_.size()) {
            throw std::runtime_error("msgpack: writer overflow");
        }
    }
    void put1(std::uint8_t b) {
        need(1);
        dst_[pos_++] = b;
    }
    void put_be16(std::uint16_t v) {
        need(2);
        dst_[pos_++] = static_cast<std::uint8_t>(v >> 8);
        dst_[pos_++] = static_cast<std::uint8_t>(v);
    }
    void put_be32(std::uint32_t v) {
        need(4);
        dst_[pos_++] = static_cast<std::uint8_t>(v >> 24);
        dst_[pos_++] = static_cast<std::uint8_t>(v >> 16);
        dst_[pos_++] = static_cast<std::uint8_t>(v >> 8);
        dst_[pos_++] = static_cast<std::uint8_t>(v);
    }
    void put_bytes(const void* p, std::size_t n) {
        need(n);
        std::memcpy(dst_.data() + pos_, p, n);
        pos_ += n;
    }

    std::span<std::uint8_t> dst_;
    std::size_t pos_;
};

}  // namespace pie_portable_driver
