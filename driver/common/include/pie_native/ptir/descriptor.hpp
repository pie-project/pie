#pragma once

#include <cstdint>

#include "pie_native/ptir/trace.hpp"

namespace pie_native::ptir::descriptor {

inline constexpr std::uint8_t kPortEmbedTokens = 0;
inline constexpr std::uint8_t kPortEmbedIndptr = 1;
inline constexpr std::uint8_t kPortPositions = 2;
inline constexpr std::uint8_t kPortPages = 3;
inline constexpr std::uint8_t kPortPageIndptr = 4;
inline constexpr std::uint8_t kPortKvLen = 5;
inline constexpr std::uint8_t kPortWSlot = 6;
inline constexpr std::uint8_t kPortWOff = 7;
inline constexpr std::uint8_t kPortReadout = 8;
inline constexpr std::uint8_t kPortAttnMask = 9;

inline bool is_device_geometry_trace(const Trace& trace) {
    bool has_write_desc = false;
    ChannelId pages_channel = 0;
    bool has_pages = false;
    for (const PortBinding& binding : trace.ports) {
        if (binding.is_const) continue;
        if (binding.port == kPortWSlot || binding.port == kPortWOff) {
            has_write_desc = true;
        } else if (binding.port == kPortPages) {
            pages_channel = binding.channel;
            has_pages = true;
        }
    }
    if (!has_write_desc ||
        !has_pages ||
        pages_channel >= trace.channels.size()) {
        return false;
    }
    const auto& dims = trace.channels[pages_channel].type.shape.dims;
    return dims.size() == 2 && dims[1] > 1;
}

inline std::uint32_t last_page_len(
    std::uint32_t length,
    std::uint32_t page_size) {
    return length == 0 || page_size == 0
               ? 0
               : ((length - 1) % page_size) + 1;
}

}  // namespace pie_native::ptir::descriptor
