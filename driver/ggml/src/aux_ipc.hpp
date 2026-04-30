#pragma once

// Auxiliary IPC channel between the Python wrapper and the binary.
//
// The wrapper (`pie_driver_ggml/worker.py`) runs the cold-path
// `RpcServer` that the Pie runtime calls for page-copy / swap / adapter
// ops. Some of those (`copy_d2h`, `copy_h2d`, `copy_d2d`, `copy_h2h`)
// need to touch the binary's KV pool, which lives in the binary process
// (often on a GPU backend). To bridge the two processes we use a tiny
// unix-domain socket: the wrapper writes a fixed-layout command, the
// binary's aux thread reads it, executes the requested page copies via
// `ggml_backend_tensor_{get,set}`, and writes back an ack.
//
// Wire format is fixed-size little-endian for simplicity:
//   header: AuxCmdHeader (24 bytes)
//   payload: header.n_pages × AuxPagePair (8 bytes each)
//   ack:    AuxAck (16 bytes)

#include <cstdint>

namespace pie_ggml_driver::aux {

inline constexpr std::uint32_t MAGIC      = 0x41454950u;  // 'PIEA'
inline constexpr std::uint32_t ACK_MAGIC  = 0x4B414950u;  // 'PIAK'

enum class Method : std::uint32_t {
    None        = 0,
    CopyD2H     = 1,   // (src_dev_page → dst_host_slot)
    CopyH2D     = 2,   // (dst_dev_page ← src_host_slot)
    CopyD2D     = 3,   // (dst_dev_page ← src_dev_page)
    CopyH2H     = 4,   // (dst_host_slot ← src_host_slot)
    LoadAdapter = 5,   // (M9) load LoRA from a safetensors file path
};

// LoadAdapter command payload (after AuxCmdHeader). Path is utf-8 bytes
// of `path_len`. The wrapper writes the LoRA bytes to a temp file and
// sends only the path; the binary reads + parses the file.
struct AuxLoadAdapterPayload {
    std::uint64_t adapter_id;
    std::uint32_t path_len;
    std::uint32_t reserved;
    // Followed by `path_len` bytes (no null terminator).
};
static_assert(sizeof(AuxLoadAdapterPayload) == 16, "wire layout");

struct AuxCmdHeader {
    std::uint32_t magic;     // = MAGIC
    std::uint32_t method;    // Method enum
    std::uint32_t n_pages;   // number of (src, dst) pairs that follow
    std::uint32_t reserved;
    std::uint64_t req_id;    // round-trips back in the ack (debugging)
};
static_assert(sizeof(AuxCmdHeader) == 24, "wire layout");

struct AuxPagePair {
    std::uint32_t src;
    std::uint32_t dst;
};
static_assert(sizeof(AuxPagePair) == 8, "wire layout");

enum class Status : std::uint32_t {
    Ok           = 0,
    BadMagic     = 1,
    BadMethod    = 2,
    OutOfBounds  = 3,
    NoSwapPool   = 4,
    BackendError = 5,
};

struct AuxAck {
    std::uint32_t magic;     // = ACK_MAGIC
    std::uint32_t status;    // Status enum
    std::uint64_t req_id;    // copy of header.req_id
};
static_assert(sizeof(AuxAck) == 16, "wire layout");

}  // namespace pie_ggml_driver::aux
