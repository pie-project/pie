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
    std::uint8_t extern_dir = PIE_CHANNEL_EXTERN_NONE;
    std::string extern_name;

    std::uint32_t cell_bytes() const {
        std::uint64_t numel = 1;
        for (std::uint32_t dim : dims) {
            if (dim == 0 ||
                numel > std::numeric_limits<std::uint64_t>::max() / dim) {
                return 0;
            }
            numel *= dim;
        }
        std::uint64_t bytes = 0;
        if (dtype == PIE_CHANNEL_DTYPE_BOOL) {
            if (numel > std::numeric_limits<std::uint64_t>::max() - 7) return 0;
            bytes = (numel + 7) / 8;
        } else {
            if (numel > std::numeric_limits<std::uint64_t>::max() / 4) return 0;
            bytes = numel * 4;
        }
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
    if (version != 1 && version != 2) return fail("unsupported PTIR version");
    static_cast<void>(flags);
    std::uint32_t extern_count = 0;
    if (version == 2 && !cursor.read_u32(extern_count)) {
        return fail("truncated PTIR extern count");
    }

    std::vector<std::string> names;
    names.reserve(name_count);
    for (std::uint32_t i = 0; i < name_count; ++i) {
        std::uint16_t name_len = 0;
        if (!cursor.read_u16(name_len) ||
            name_len > cursor.len - cursor.off) {
            return fail("truncated PTIR name table");
        }
        names.emplace_back(
            reinterpret_cast<const char*>(cursor.ptr + cursor.off), name_len);
        cursor.off += name_len;
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
        if (channel.dtype > PIE_CHANNEL_DTYPE_ACT ||
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

    auto read_shape_numel = [&](std::uint64_t& numel) {
        std::uint8_t rank = 0;
        if (!cursor.read_u8(rank) || rank > 4) return false;
        numel = 1;
        for (std::uint8_t i = 0; i < rank; ++i) {
            std::uint32_t dim = 0;
            if (!cursor.read_u32(dim) || dim == 0 ||
                numel > std::numeric_limits<std::uint64_t>::max() / dim) {
                return false;
            }
            numel *= dim;
        }
        return true;
    };
    for (std::uint32_t i = 0; i < port_count; ++i) {
        std::uint8_t port = 0;
        std::uint8_t source = 0;
        if (!cursor.read_u8(port) || port > 9 || !cursor.read_u8(source)) {
            return fail("invalid PTIR port");
        }
        if (source == 0) {
            if (!cursor.skip(4)) return fail("truncated PTIR channel port");
        } else if (source == 1) {
            std::uint8_t dtype = 0;
            std::uint64_t numel = 0;
            if (!cursor.read_u8(dtype) || dtype > PIE_CHANNEL_DTYPE_BOOL ||
                !read_shape_numel(numel)) {
                return fail("invalid PTIR constant port");
            }
            const std::uint64_t elem_bytes =
                dtype == PIE_CHANNEL_DTYPE_BOOL ? 1 : 4;
            if (numel > std::numeric_limits<std::size_t>::max() / elem_bytes ||
                !cursor.skip(static_cast<std::size_t>(numel * elem_bytes))) {
                return fail("truncated PTIR constant port");
            }
        } else {
            return fail("invalid PTIR port source");
        }
    }

    auto skip_shape = [&]() {
        std::uint64_t ignored = 0;
        return read_shape_numel(ignored);
    };
    auto skip_op = [&]() {
        std::uint8_t tag = 0;
        if (!cursor.read_u8(tag)) return false;
        switch (tag) {
            case 0x01: case 0x02: case 0x03: case 0x04: case 0x05: case 0x06:
            case 0x1E: case 0x30: case 0x31: case 0x32: case 0x33: case 0x3A:
            case 0x40: case 0x41: case 0x50: case 0x64: case 0x90: case 0x91:
                return cursor.skip(4);
            case 0x07:
                return cursor.skip(5);
            case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15:
            case 0x16: case 0x17: case 0x18: case 0x19: case 0x1A: case 0x1B:
            case 0x1C: case 0x1D: case 0x1F: case 0x51: case 0x55: case 0x60:
            case 0x61: case 0x65: case 0x92:
                return cursor.skip(8);
            case 0x20: case 0x62: case 0x63:
                return cursor.skip(12);
            case 0x38: case 0x39:
                return cursor.skip(4) && skip_shape();
            case 0x58:
                return cursor.skip(9);
            case 0x70: case 0x71:
                return cursor.skip(4) && skip_shape() && cursor.skip(1);
            case 0x81:
                return cursor.skip(5);
            case 0xA0:
                return cursor.skip(3) && skip_shape();
            case 0xA1: {
                if (!cursor.skip(3) || !skip_shape()) return false;
                std::uint8_t args = 0;
                return cursor.read_u8(args) &&
                       cursor.skip(static_cast<std::size_t>(args) * 4);
            }
            case 0xA2: {
                std::uint8_t args = 0;
                return cursor.skip(2) && cursor.read_u8(args) &&
                       cursor.skip(static_cast<std::size_t>(args) * 4);
            }
            default:
                return false;
        }
    };
    for (std::uint32_t i = 0; i < stage_count; ++i) {
        std::uint8_t stage = 0;
        std::uint32_t op_count = 0;
        if (!cursor.read_u8(stage) || stage > 3 || !cursor.read_u32(op_count)) {
            return fail("invalid PTIR stage");
        }
        for (std::uint32_t op = 0; op < op_count; ++op) {
            if (!skip_op()) return fail("invalid PTIR operation");
        }
    }
    std::vector<bool> has_extern(channels.size(), false);
    for (std::uint32_t i = 0; i < extern_count; ++i) {
        std::uint16_t name = 0;
        std::uint8_t direction = 0;
        std::uint32_t channel = 0;
        if (!cursor.read_u16(name) || !cursor.read_u8(direction) ||
            !cursor.read_u32(channel) || name >= names.size() ||
            direction > 1 || channel >= channels.size() || has_extern[channel]) {
            return fail("invalid PTIR extern declaration");
        }
        has_extern[channel] = true;
        channels[channel].extern_dir = static_cast<std::uint8_t>(direction + 1);
        channels[channel].extern_name = names[name];
    }
    if (cursor.off != cursor.len) {
        return fail("trailing PTIR bytes");
    }
    return true;
}

}  // namespace pie_native
