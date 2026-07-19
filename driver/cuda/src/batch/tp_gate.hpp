#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace pie_cuda_driver {

// Consume exactly one published gate epoch. Advancing directly to `published`
// would collapse a burst of MTP notifications into one follower receive.
inline bool tp_cpu_gate_consume_one(
    std::uint64_t published,
    std::uint64_t& consumed) noexcept {
    if (published <= consumed) return false;
    ++consumed;
    return true;
}

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

    bool wait_one(
        std::uint64_t& consumed,
        const std::atomic<bool>& stop) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] {
            return stop.load(std::memory_order_relaxed) ||
                sequence_.load(std::memory_order_acquire) > consumed;
        });
        if (stop.load(std::memory_order_relaxed)) return false;
        return tp_cpu_gate_consume_one(
            sequence_.load(std::memory_order_acquire), consumed);
    }

  private:
    std::atomic<std::uint64_t> sequence_{0};
    std::mutex mutex_;
    std::condition_variable condition_;
};

}  // namespace pie_cuda_driver
