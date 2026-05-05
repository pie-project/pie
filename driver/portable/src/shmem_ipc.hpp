#pragma once

// Shared-memory IPC fast path for fire_batch — server side.
//
// Layout matches `runtime/src/shmem_ipc.rs` and `pie/src/pie_driver/shmem_ipc.py`.
// The Rust runtime is the client (writes requests + bumps req_seq). pie_driver_portable
// is the server (polls slots, handles requests, writes responses + bumps resp_seq).

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>

namespace pie_portable_driver {

inline constexpr std::uint32_t MAGIC = 0x50494533;  // 'PIE3'
// Bump in lockstep with `runtime/src/shmem_ipc.rs::SCHEMA_VERSION` and
// `pie_driver/shmem_ipc.py::SCHEMA_VERSION`. v2 added req_buf_size and
// resp_buf_size to the global header so the Rust client no longer
// hardcodes geometry.
inline constexpr std::uint32_t SCHEMA_VERSION = 2;
inline constexpr std::size_t HEADER_SIZE = 64;
inline constexpr std::size_t SLOT_HEADER_SIZE = 64;

inline constexpr std::uint32_t METHOD_TAG_FIRE_BATCH = 0;
inline constexpr std::uint32_t METHOD_TAG_NONE = 255;

// Slot header offsets relative to the slot start.
inline constexpr std::size_t OFF_REQ_SEQ = 0;
inline constexpr std::size_t OFF_RESP_SEQ = 8;
inline constexpr std::size_t OFF_REQ_ID = 16;
inline constexpr std::size_t OFF_METHOD_TAG = 20;
inline constexpr std::size_t OFF_REQ_LEN = 24;
inline constexpr std::size_t OFF_RESP_LEN = 28;
inline constexpr std::size_t OFF_SEND_WT = 32;
inline constexpr std::size_t OFF_RESPOND_WT = 40;

struct SlotRequest {
    std::uint32_t req_id;
    std::uint32_t method_tag;
    std::span<const std::uint8_t> payload;
};

// Handler returns the number of bytes written into `response`. Returning 0
// is fine (empty response).
using RequestHandler = std::function<std::size_t(
    const SlotRequest& req, std::span<std::uint8_t> response)>;

class ShmemServer {
public:
    // Create a new shmem region (replaces any stale one with the same name).
    // The Rust runtime opens the same name and starts firing requests.
    ShmemServer(std::string name,
                std::size_t num_slots,
                std::size_t req_buf,
                std::size_t resp_buf,
                std::uint64_t spin_us = 0);

    ~ShmemServer();

    ShmemServer(const ShmemServer&) = delete;
    ShmemServer& operator=(const ShmemServer&) = delete;

    // Block forever, dispatching requests round-robin across slots.
    // Returns when `stop()` is called from another thread.
    void serve_forever(const RequestHandler& handler);

    void stop() noexcept { stop_.store(true, std::memory_order_relaxed); }

    std::size_t num_slots() const noexcept { return num_slots_; }
    std::size_t req_buf_size() const noexcept { return req_buf_size_; }
    std::size_t resp_buf_size() const noexcept { return resp_buf_size_; }
    const std::string& name() const noexcept { return name_; }

private:
    std::uint8_t* slot_ptr(std::size_t i) noexcept {
        return base_ + HEADER_SIZE + i * slot_stride_;
    }

    std::string name_;
    std::size_t num_slots_;
    std::size_t req_buf_size_;
    std::size_t resp_buf_size_;
    std::size_t slot_stride_;
    std::size_t total_size_;
    std::uint64_t spin_us_;

#ifdef _WIN32
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
    std::uint8_t* base_ = nullptr;

    std::atomic<bool> stop_{false};
};

}  // namespace pie_portable_driver
