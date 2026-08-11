#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace pie_cuda_driver {

enum class TpFireKind : std::uint8_t {
    Forward,
    MtpDraft,
};

// Every TP fire reuses the same persistent inputs, model workspace, stream,
// and communicator. A fire kind that returned false here could publish its
// successor while another rank still consumes the previous fire, making the
// ranks execute different logical collectives at the same peer-barrier epoch.
constexpr bool tp_fire_requires_device_retirement(TpFireKind) noexcept {
    return true;
}

// Consume exactly one published gate epoch. Advancing directly to `published`
// would collapse a burst of MTP notifications into one follower receive.
inline bool tp_cpu_gate_consume_one(
    std::uint64_t published,
    std::uint64_t& consumed) noexcept {
    if (published <= consumed) return false;
    ++consumed;
    return true;
}

class TpFollowerAcks {
  public:
    void mark(int rank, std::uint64_t sequence) noexcept {
        if (rank <= 0 || rank >= kMaxRanks) return;
        consumed_[static_cast<std::size_t>(rank)].store(
            sequence, std::memory_order_release);
    }

    bool all_consumed(int world_size, std::uint64_t sequence) const noexcept {
        if (world_size <= 1) return true;
        if (world_size > kMaxRanks) return false;
        for (int rank = 1; rank < world_size; ++rank) {
            if (consumed_[static_cast<std::size_t>(rank)].load(
                    std::memory_order_acquire) < sequence) {
                return false;
            }
        }
        return true;
    }

    std::uint64_t oldest_consumed(int world_size) const noexcept {
        if (world_size <= 1) return 0;
        if (world_size > kMaxRanks) return 0;
        std::uint64_t oldest = consumed_[1].load(std::memory_order_acquire);
        for (int rank = 2; rank < world_size; ++rank) {
            const auto consumed = consumed_[static_cast<std::size_t>(rank)].load(
                std::memory_order_acquire);
            if (consumed < oldest) oldest = consumed;
        }
        return oldest;
    }

  private:
    static constexpr int kMaxRanks = 64;
    std::array<std::atomic<std::uint64_t>, kMaxRanks> consumed_{};
};

class TpSequenceGate {
  public:
    std::uint64_t published() const noexcept {
        return sequence_.load(std::memory_order_acquire);
    }

    void publish() {
        {
            // The sequence update and the wait predicate share this mutex.
            // Therefore publication cannot land between a false predicate
            // check and the condition-variable wait transition.
            std::lock_guard<std::mutex> lock(mutex_);
            sequence_.fetch_add(1, std::memory_order_release);
        }
        condition_.notify_all();
    }

    // Group-wide shutdown. `stop` alone cannot release a parked waiter: it is
    // an ordinary atomic, so flipping it never wakes `condition_`, and the
    // predicate is only re-evaluated on a notification. Teardown must go
    // through here so waiters actually observe the request.
    void request_stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_.store(true, std::memory_order_release);
        }
        condition_.notify_all();
    }

    bool stopped() const noexcept {
        return stopped_.load(std::memory_order_acquire);
    }

    // Returns true when an epoch was consumed, false when the wait ended
    // because shutdown was requested.
    bool wait_one(
        std::uint64_t& consumed,
        const std::atomic<bool>& stop) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] {
            return stop.load(std::memory_order_relaxed) ||
                stopped_.load(std::memory_order_relaxed) ||
                sequence_.load(std::memory_order_acquire) > consumed;
        });
        if (stop.load(std::memory_order_relaxed) ||
            stopped_.load(std::memory_order_relaxed)) {
            return false;
        }
        return tp_cpu_gate_consume_one(
            sequence_.load(std::memory_order_acquire), consumed);
    }

  private:
    std::atomic<std::uint64_t> sequence_{0};
    std::atomic<bool> stopped_{false};
    std::mutex mutex_;
    std::condition_variable condition_;
};

}  // namespace pie_cuda_driver
