#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace pie::metal {

struct M0TimingSnapshot {
    std::uint64_t cpu_epilogue_samples = 0;
    std::uint64_t cpu_epilogue_ns = 0;
    std::uint64_t bf16_conversion_samples = 0;
    std::uint64_t bf16_conversion_ns = 0;
    std::uint64_t forward_wait_samples = 0;
    std::uint64_t forward_wait_ns = 0;
    std::uint64_t forward_wait_timeouts = 0;
};

inline M0TimingSnapshot operator-(
    const M0TimingSnapshot& after,
    const M0TimingSnapshot& before) {
    return {
        after.cpu_epilogue_samples - before.cpu_epilogue_samples,
        after.cpu_epilogue_ns - before.cpu_epilogue_ns,
        after.bf16_conversion_samples - before.bf16_conversion_samples,
        after.bf16_conversion_ns - before.bf16_conversion_ns,
        after.forward_wait_samples - before.forward_wait_samples,
        after.forward_wait_ns - before.forward_wait_ns,
        after.forward_wait_timeouts - before.forward_wait_timeouts,
    };
}

class M0TimingCounters {
  public:
    using Clock = std::chrono::steady_clock;

    void record_cpu_epilogue(Clock::duration duration) {
        record(cpu_epilogue_samples_, cpu_epilogue_ns_, duration);
    }
    void record_bf16_conversion(Clock::duration duration) {
        record(bf16_conversion_samples_, bf16_conversion_ns_, duration);
    }
    void record_forward_wait(Clock::duration duration) {
        record(forward_wait_samples_, forward_wait_ns_, duration);
    }
    void record_forward_wait_timeout() {
        forward_wait_timeouts_.fetch_add(1, std::memory_order_relaxed);
    }

    M0TimingSnapshot snapshot() const {
        return {
            cpu_epilogue_samples_.load(std::memory_order_relaxed),
            cpu_epilogue_ns_.load(std::memory_order_relaxed),
            bf16_conversion_samples_.load(std::memory_order_relaxed),
            bf16_conversion_ns_.load(std::memory_order_relaxed),
            forward_wait_samples_.load(std::memory_order_relaxed),
            forward_wait_ns_.load(std::memory_order_relaxed),
            forward_wait_timeouts_.load(std::memory_order_relaxed),
        };
    }

    void reset_for_tests() {
        cpu_epilogue_samples_.store(0, std::memory_order_relaxed);
        cpu_epilogue_ns_.store(0, std::memory_order_relaxed);
        bf16_conversion_samples_.store(0, std::memory_order_relaxed);
        bf16_conversion_ns_.store(0, std::memory_order_relaxed);
        forward_wait_samples_.store(0, std::memory_order_relaxed);
        forward_wait_ns_.store(0, std::memory_order_relaxed);
        forward_wait_timeouts_.store(0, std::memory_order_relaxed);
    }

  private:
    static void record(
        std::atomic<std::uint64_t>& samples,
        std::atomic<std::uint64_t>& total_ns,
        Clock::duration duration) {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            duration)
                            .count();
        samples.fetch_add(1, std::memory_order_relaxed);
        total_ns.fetch_add(
            static_cast<std::uint64_t>(ns > 0 ? ns : 0),
            std::memory_order_relaxed);
    }

    std::atomic<std::uint64_t> cpu_epilogue_samples_{0};
    std::atomic<std::uint64_t> cpu_epilogue_ns_{0};
    std::atomic<std::uint64_t> bf16_conversion_samples_{0};
    std::atomic<std::uint64_t> bf16_conversion_ns_{0};
    std::atomic<std::uint64_t> forward_wait_samples_{0};
    std::atomic<std::uint64_t> forward_wait_ns_{0};
    std::atomic<std::uint64_t> forward_wait_timeouts_{0};
};

inline M0TimingCounters& m0_timing_counters() {
    static M0TimingCounters counters;
    return counters;
}

struct M1PreparedResourceSnapshot {
    std::uint64_t fires = 0;
    std::uint64_t external_handles = 0;
    std::uint64_t standalone_buffers = 0;
};

class M1PreparedResourceCounters {
  public:
    void acquire(std::uint64_t external_handles,
                 std::uint64_t standalone_buffers) {
        fires_.fetch_add(1, std::memory_order_relaxed);
        external_handles_.fetch_add(
            external_handles, std::memory_order_relaxed);
        standalone_buffers_.fetch_add(
            standalone_buffers, std::memory_order_relaxed);
    }

    void release(std::uint64_t external_handles,
                 std::uint64_t standalone_buffers) {
        fires_.fetch_sub(1, std::memory_order_relaxed);
        external_handles_.fetch_sub(
            external_handles, std::memory_order_relaxed);
        standalone_buffers_.fetch_sub(
            standalone_buffers, std::memory_order_relaxed);
    }

    M1PreparedResourceSnapshot snapshot() const {
        return {
            fires_.load(std::memory_order_relaxed),
            external_handles_.load(std::memory_order_relaxed),
            standalone_buffers_.load(std::memory_order_relaxed),
        };
    }

  private:
    std::atomic<std::uint64_t> fires_{0};
    std::atomic<std::uint64_t> external_handles_{0};
    std::atomic<std::uint64_t> standalone_buffers_{0};
};

inline M1PreparedResourceCounters& m1_prepared_resource_counters() {
    static M1PreparedResourceCounters counters;
    return counters;
}

}  // namespace pie::metal
