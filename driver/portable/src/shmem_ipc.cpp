#include "shmem_ipc.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pie_portable_driver {

namespace {

[[nodiscard]] std::uint64_t now_us() noexcept {
    using namespace std::chrono;
    return duration_cast<microseconds>(
               steady_clock::now().time_since_epoch())
        .count();
}

void atomic_store_u64(std::uint8_t* p, std::uint64_t v) noexcept {
    std::atomic_ref<std::uint64_t>{*reinterpret_cast<std::uint64_t*>(p)}.store(
        v, std::memory_order_release);
}

std::uint64_t atomic_load_u64(const std::uint8_t* p) noexcept {
    return std::atomic_ref<std::uint64_t>{
        *reinterpret_cast<std::uint64_t*>(const_cast<std::uint8_t*>(p))}
        .load(std::memory_order_acquire);
}

}  // namespace

ShmemServer::ShmemServer(std::string name,
                         std::size_t num_slots,
                         std::size_t req_buf,
                         std::size_t resp_buf,
                         std::uint64_t spin_us)
    : name_(std::move(name)),
      num_slots_(num_slots),
      req_buf_size_(req_buf),
      resp_buf_size_(resp_buf),
      slot_stride_(SLOT_HEADER_SIZE + req_buf + resp_buf),
      total_size_(HEADER_SIZE + num_slots * (SLOT_HEADER_SIZE + req_buf + resp_buf)),
      spin_us_(spin_us) {
    // Best-effort cleanup of any stale region from a previous run.
    shm_unlink(name_.c_str());

    fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd_ < 0) {
        throw std::runtime_error("shm_open(" + name_ +
                                 ") failed: " + std::strerror(errno));
    }
    if (ftruncate(fd_, static_cast<off_t>(total_size_)) != 0) {
        ::close(fd_);
        shm_unlink(name_.c_str());
        throw std::runtime_error(std::string("ftruncate failed: ") +
                                 std::strerror(errno));
    }

    void* p = mmap(nullptr, total_size_, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd_, 0);
    if (p == MAP_FAILED) {
        ::close(fd_);
        shm_unlink(name_.c_str());
        throw std::runtime_error(std::string("mmap failed: ") +
                                 std::strerror(errno));
    }
    base_ = static_cast<std::uint8_t*>(p);

    // Zero the region and write the header. Rust/Python read the header to
    // sanity-check the schema.
    std::memset(base_, 0, total_size_);
    auto write_u32 = [&](std::size_t off, std::uint32_t v) {
        std::memcpy(base_ + off, &v, sizeof(v));
    };
    write_u32(0, MAGIC);
    write_u32(4, SCHEMA_VERSION);
    write_u32(8, static_cast<std::uint32_t>(num_slots_));
    write_u32(12, static_cast<std::uint32_t>(slot_stride_));
}

ShmemServer::~ShmemServer() {
    if (base_) {
        munmap(base_, total_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
    shm_unlink(name_.c_str());
}

void ShmemServer::serve_forever(const RequestHandler& handler) {
    std::vector<std::uint64_t> last_seen(num_slots_, 0);

    while (!stop_.load(std::memory_order_relaxed)) {
        bool did_work = false;

        for (std::size_t i = 0; i < num_slots_; ++i) {
            std::uint8_t* slot = slot_ptr(i);

            const std::uint64_t req_seq = atomic_load_u64(slot + OFF_REQ_SEQ);
            if (req_seq == last_seen[i]) continue;

            // New request available.
            std::uint32_t req_id = 0;
            std::uint32_t method_tag = 0;
            std::uint32_t req_len = 0;
            std::memcpy(&req_id, slot + OFF_REQ_ID, 4);
            std::memcpy(&method_tag, slot + OFF_METHOD_TAG, 4);
            std::memcpy(&req_len, slot + OFF_REQ_LEN, 4);

            const std::uint8_t* req_payload = slot + SLOT_HEADER_SIZE;
            std::uint8_t* resp_payload =
                slot + SLOT_HEADER_SIZE + req_buf_size_;

            SlotRequest req{
                .req_id = req_id,
                .method_tag = method_tag,
                .payload = std::span<const std::uint8_t>(req_payload, req_len),
            };
            std::span<std::uint8_t> resp_buf(resp_payload, resp_buf_size_);

            const std::size_t resp_len = handler(req, resp_buf);

            const std::uint32_t resp_len_u32 =
                static_cast<std::uint32_t>(resp_len);
            std::memcpy(slot + OFF_RESP_LEN, &resp_len_u32, 4);

            const std::uint64_t respond_wt = now_us();
            std::memcpy(slot + OFF_RESPOND_WT, &respond_wt, 8);

            // Publish: bump resp_seq to match req_seq.
            atomic_store_u64(slot + OFF_RESP_SEQ, req_seq);

            last_seen[i] = req_seq;
            did_work = true;
        }

        if (!did_work && spin_us_ > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(spin_us_));
        } else if (!did_work) {
            // Yield CPU briefly under pure-spin mode so we don't peg a core.
            std::this_thread::yield();
        }
    }
}

}  // namespace pie_portable_driver
