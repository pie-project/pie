#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace pie_native {

inline constexpr std::uint8_t PTIR_HOST_NONE = 0;
inline constexpr std::uint8_t PTIR_HOST_WRITER = 1;
inline constexpr std::uint8_t PTIR_HOST_READER = 2;

struct PtirChannelDecl {
    std::uint8_t dtype = 0;
    std::vector<std::uint32_t> dims;
    std::uint32_t capacity = 0;
    std::uint8_t host_role = PTIR_HOST_NONE;
    bool seeded = false;

    std::uint32_t cell_bytes() const {
        std::uint64_t numel = dims.empty() ? 1 : 1;
        for (std::uint32_t dim : dims) numel *= dim;
        const std::uint64_t elem_bytes = dtype == 3 ? 1 : 4;
        const std::uint64_t bytes = numel * elem_bytes;
        return bytes == 0 || bytes > std::numeric_limits<std::uint32_t>::max()
            ? 0
            : static_cast<std::uint32_t>(bytes);
    }
};

namespace detail {

struct PtirCursor {
    const std::uint8_t* ptr = nullptr;
    std::size_t len = 0;
    std::size_t off = 0;

    bool read_u8(std::uint8_t& out) {
        if (off + 1 > len) return false;
        out = ptr[off++];
        return true;
    }
    bool read_u16(std::uint16_t& out) {
        if (off + 2 > len) return false;
        std::memcpy(&out, ptr + off, 2);
        off += 2;
        return true;
    }
    bool read_u32(std::uint32_t& out) {
        if (off + 4 > len) return false;
        std::memcpy(&out, ptr + off, 4);
        off += 4;
        return true;
    }
    bool skip(std::size_t bytes) {
        if (off + bytes > len) return false;
        off += bytes;
        return true;
    }
};

}  // namespace detail

inline bool decode_ptir_channels(
    const std::uint8_t* bytes,
    std::size_t len,
    std::vector<PtirChannelDecl>& channels,
    std::string* error = nullptr) {
    auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (bytes == nullptr || len < 24 || std::memcmp(bytes, "PTIR", 4) != 0) {
        return fail("invalid PTIR header");
    }
    detail::PtirCursor cursor{bytes, len, 4};
    std::uint16_t version = 0;
    std::uint16_t flags = 0;
    std::uint32_t name_count = 0;
    std::uint32_t channel_count = 0;
    std::uint32_t port_count = 0;
    std::uint32_t stage_count = 0;
    if (!cursor.read_u16(version) || !cursor.read_u16(flags) ||
        !cursor.read_u32(name_count) || !cursor.read_u32(channel_count) ||
        !cursor.read_u32(port_count) || !cursor.read_u32(stage_count)) {
        return fail("truncated PTIR header");
    }
    if (version != 1) return fail("unsupported PTIR version");
    static_cast<void>(flags);
    static_cast<void>(port_count);
    static_cast<void>(stage_count);

    for (std::uint32_t i = 0; i < name_count; ++i) {
        std::uint16_t name_len = 0;
        if (!cursor.read_u16(name_len) || !cursor.skip(name_len)) {
            return fail("truncated PTIR name table");
        }
    }

    channels.clear();
    channels.reserve(channel_count);
    for (std::uint32_t i = 0; i < channel_count; ++i) {
        PtirChannelDecl channel;
        std::uint8_t rank = 0;
        std::uint8_t seeded = 0;
        if (!cursor.read_u8(channel.dtype) || !cursor.read_u8(rank) || rank > 4) {
            return fail("invalid PTIR channel type");
        }
        channel.dims.resize(rank);
        for (std::uint8_t dim = 0; dim < rank; ++dim) {
            if (!cursor.read_u32(channel.dims[dim])) {
                return fail("truncated PTIR channel shape");
            }
        }
        if (!cursor.read_u32(channel.capacity) ||
            !cursor.read_u8(channel.host_role) ||
            !cursor.read_u8(seeded)) {
            return fail("truncated PTIR channel declaration");
        }
        const std::uint32_t cell_bytes = channel.cell_bytes();
        if (channel.dtype > 3 ||
            channel.host_role > PTIR_HOST_READER ||
            seeded > 1 ||
            channel.capacity >= 8 ||
            cell_bytes == 0 ||
            static_cast<std::uint64_t>(cell_bytes) *
                    (static_cast<std::uint64_t>(channel.capacity) + 1) >
                std::numeric_limits<std::size_t>::max()) {
            return fail("invalid PTIR channel declaration");
        }
        channel.seeded = seeded != 0;
        channels.push_back(std::move(channel));
    }
    return true;
}

}  // namespace pie_native
