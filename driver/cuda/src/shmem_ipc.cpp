#include "shmem_ipc.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pie_cuda_driver {

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
    return std::atomic_ref<const std::uint64_t>{
        *reinterpret_cast<const std::uint64_t*>(p)}
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
    write_u32(16, static_cast<std::uint32_t>(req_buf_size_));
    write_u32(20, static_cast<std::uint32_t>(resp_buf_size_));
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

// SPSC ring buffer for pending requests. IPC scanner thread is the
// producer; the forward-worker thread is the consumer. Power-of-two
// capacity keeps the modulo a single AND. Capacity = 64 covers the 8
// shmem slots with plenty of headroom.
namespace {

struct PendingFire {
    std::size_t  slot;
    std::uint32_t req_id;
    std::uint32_t method_tag;
    std::uint64_t req_seq;
    std::uint32_t req_len;
};

class PendingRing {
public:
    static constexpr std::size_t CAP = 64;
    static_assert((CAP & (CAP - 1)) == 0, "CAP must be power of two");
    bool try_push(const PendingFire& f) noexcept {
        const auto t = tail_.load(std::memory_order_relaxed);
        const auto h = head_.load(std::memory_order_acquire);
        if (t - h >= CAP) return false;
        buf_[t & (CAP - 1)] = f;
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }
    bool try_pop(PendingFire& out) noexcept {
        const auto h = head_.load(std::memory_order_relaxed);
        const auto t = tail_.load(std::memory_order_acquire);
        if (h == t) return false;
        out = buf_[h & (CAP - 1)];
        head_.store(h + 1, std::memory_order_release);
        return true;
    }
private:
    alignas(64) std::atomic<std::uint64_t> head_{0};
    alignas(64) std::atomic<std::uint64_t> tail_{0};
    std::array<PendingFire, CAP> buf_{};
};

void cpu_pause() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
}

}  // namespace

void ShmemServer::serve_forever(const RequestHandler& handler) {
    std::vector<std::uint64_t> last_seen(num_slots_, 0);

    // ── Forward worker thread ─────────────────────────────────────────
    // The IPC scanner (this thread) pushes work into `pending`; the
    // worker pops, runs the handler (which may block on the GPU for
    // 20+ ms per fire), and publishes the response. Decoupling lets
    // the scanner stage the next fire's BPIQ while the worker's
    // current fire is still running on the GPU.
    //
    // All CUDA calls happen on the worker thread — the IPC scanner
    // touches only shmem + atomic counters, so there's no CUDA-context
    // sharing concern.
    PendingRing pending;
    std::atomic<bool> worker_stop{false};

    std::thread worker([&]() {
        while (!worker_stop.load(std::memory_order_relaxed)) {
            PendingFire f;
            if (!pending.try_pop(f)) {
                cpu_pause();
                continue;
            }
            std::uint8_t* slot = slot_ptr(f.slot);
            const std::uint8_t* req_payload = slot + SLOT_HEADER_SIZE;
            std::uint8_t* resp_payload =
                slot + SLOT_HEADER_SIZE + req_buf_size_;
            SlotRequest req{
                .req_id = f.req_id,
                .method_tag = f.method_tag,
                .payload =
                    std::span<const std::uint8_t>(req_payload, f.req_len),
            };
            std::span<std::uint8_t> resp_buf(resp_payload, resp_buf_size_);

            const std::size_t resp_len = handler(req, resp_buf);

            const std::uint32_t resp_len_u32 =
                static_cast<std::uint32_t>(resp_len);
            std::memcpy(slot + OFF_RESP_LEN, &resp_len_u32, 4);
            const std::uint64_t respond_wt = now_us();
            std::memcpy(slot + OFF_RESPOND_WT, &respond_wt, 8);
            // Publish — release-ordered so the runtime sees the
            // resp_len / payload writes above before observing the
            // new resp_seq.
            atomic_store_u64(slot + OFF_RESP_SEQ, f.req_seq);
        }
    });

    while (!stop_.load(std::memory_order_relaxed)) {
        bool did_work = false;

        for (std::size_t i = 0; i < num_slots_; ++i) {
            std::uint8_t* slot = slot_ptr(i);

            const std::uint64_t req_seq = atomic_load_u64(slot + OFF_REQ_SEQ);
            if (req_seq == last_seen[i]) continue;

            // New request available. Capture all the per-slot metadata
            // we need (the worker will dereference req_payload via
            // slot index after popping).
            std::uint32_t req_id = 0;
            std::uint32_t method_tag = 0;
            std::uint32_t req_len = 0;
            std::memcpy(&req_id, slot + OFF_REQ_ID, 4);
            std::memcpy(&method_tag, slot + OFF_METHOD_TAG, 4);
            std::memcpy(&req_len, slot + OFF_REQ_LEN, 4);

            PendingFire pf{i, req_id, method_tag, req_seq, req_len};
            // Backpressure: queue holds 64; pause until worker drains
            // if we somehow get ahead (capacity 64 vs 8 slots → never
            // fires in practice, but defensive).
            while (!pending.try_push(pf)) cpu_pause();

            last_seen[i] = req_seq;
            did_work = true;
        }

        if (!did_work) {
            if (spin_us_ > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(spin_us_));
            } else {
                cpu_pause();
            }
        }
    }

    // Drain & shut down the worker. The worker may still be running a
    // fire; wait for it before destroying captures.
    worker_stop.store(true, std::memory_order_release);
    worker.join();
}

}  // namespace pie_cuda_driver
