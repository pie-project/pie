#pragma once

// Cold-path control channel between the Python wrapper and this binary.
//
// The wrapper creates a Unix `SOCK_SEQPACKET` socketpair before spawn,
// passes one fd to us via `--control-fd`, and uses the other fd for
// `copy_d2h / copy_h2d / copy_d2d / copy_h2h` requests forwarded from
// the runtime's RPC channel.
//
// Wire format (all little-endian, fixed 16-byte header followed by two
// `num_pairs`-element u32 arrays):
//
//     u32 method      // 1=D2H, 2=H2D, 3=D2D, 4=H2H
//     u32 layer
//     u32 num_pairs
//     u32 reserved
//     u32 srcs[num_pairs]
//     u32 dsts[num_pairs]
//
// Total message size = 16 + 8 * num_pairs bytes.
// Response is a single u32 status (0 = ok, non-zero = error).

#include <cstdint>
#include <functional>

namespace pie_cuda_driver {

inline constexpr std::uint32_t CTRL_METHOD_COPY_D2H = 1;
inline constexpr std::uint32_t CTRL_METHOD_COPY_H2D = 2;
inline constexpr std::uint32_t CTRL_METHOD_COPY_D2D = 3;
inline constexpr std::uint32_t CTRL_METHOD_COPY_H2H = 4;

struct CtrlRequest {
    std::uint32_t method;
    std::uint32_t layer;
    std::uint32_t num_pairs;
    const std::uint32_t* src_dst_pairs;  // length = num_pairs * 2
};

using CtrlHandler = std::function<std::uint32_t(const CtrlRequest& req)>;

// Blocking serve loop. Reads one message at a time off `fd` (SEQPACKET),
// invokes `handler`, writes the returned u32 status. Returns when the peer
// closes the socket or `stop()`-style EOF.
void serve_control_socket(int fd, const CtrlHandler& handler);

}  // namespace pie_cuda_driver
