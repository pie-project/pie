#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <span>
#include <string>
#include <thread>
#include <time.h>
#include <unistd.h>

namespace pie_cuda_driver::fire_timing {

using Clock = std::chrono::steady_clock;

// `PIE_FIRE_TIMING` -- the variable four comments across this driver already
// name as the way to get these records (`dispatch.cu`, `channel_registry.hpp`,
// `frame.cpp`). Both of these were `constexpr false`, so the only way to read a
// per-fire breakdown was to edit this header and rebuild, and the rebuilt
// binary was then not the one whose numbers were being explained.
//
//   unset / 0        both off
//   1 (or anything)  `enabled` -- the per-wave driver records
//   waves | full     also `full` -- the per-fire host-stage breakdown
//
// Every call site hoists these once per fire (six in total, none per-token or
// per-lane), so the runtime read costs a cached load on a path that already
// does device work in the microseconds.
namespace detail {
inline const char* fire_timing_env() {
    static const char* const value = std::getenv("PIE_FIRE_TIMING");
    return value;
}
inline bool fire_timing_set() {
    static const bool value = [] {
        const char* const env = fire_timing_env();
        return env != nullptr && *env != '\0' && env[0] != '0';
    }();
    return value;
}
}  // namespace detail

inline bool enabled() { return detail::fire_timing_set(); }

inline bool full() {
    static const bool value = [] {
        if (!detail::fire_timing_set()) return false;
        const std::string mode = detail::fire_timing_env();
        return mode == "waves" || mode == "full";
    }();
    return value;
}

inline std::uint64_t duration_us(
    Clock::time_point start,
    Clock::time_point end) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count());
}

inline std::uint64_t monotonic_ns() {
    ::timespec value{};
    if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        std::abort();
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ull +
           static_cast<std::uint64_t>(value.tv_nsec);
}

inline std::uint64_t membership_hash(
    std::span<const std::uint64_t> logical_fire_ids) {
    std::uint64_t hash = 14'695'981'039'346'656'037ull;
    for (const std::uint64_t logical_fire_id : logical_fire_ids) {
        hash = (hash ^ logical_fire_id) * 1'099'511'628'211ull;
    }
    return hash;
}

inline void write(const std::string& record) {
    const std::string line = "[pie-fire-timing] " + record + '\n';
    static_cast<void>(::write(STDERR_FILENO, line.data(), line.size()));
}

struct SettlementRecord {
    std::uint64_t wave_id = 0;
    std::size_t fire_count = 0;
    std::uint64_t membership_hash = 0;
    std::uint64_t finish_to_settle_us = 0;
    std::uint64_t settled_monotonic_ns = 0;
    bool synchronous = false;
};

inline std::string settlement_json(const SettlementRecord& record) {
    std::ostringstream output;
    output << R"({"schema":1,"source":"cuda","event":"cuda_settled")"
           << R"(,"wave_id":)" << record.wave_id
           << R"(,"fire_count":)" << record.fire_count
           << R"(,"membership_hash":)" << record.membership_hash;
    if (record.synchronous) {
        output << R"(,"synchronous":true)";
    } else {
        output << R"(,"finish_to_settle_us":)"
               << record.finish_to_settle_us;
    }
    if (record.settled_monotonic_ns != 0) {
        output << R"(,"settled_monotonic_ns":)"
               << record.settled_monotonic_ns;
    }
    output << '}';
    return output.str();
}

class SettlementWriter {
  public:
    SettlementWriter()
        : thread_([this] { run(); }) {}

    ~SettlementWriter() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        not_empty_.notify_one();
        if (thread_.joinable()) thread_.join();
    }

    // NEVER blocks: settlement records are enqueued from cudaLaunchHostFunc
    // callbacks, where a stalled writer would stall the stream. A full ring
    // drops the record and counts it; the drain thread reports the drop
    // count loudly (RV-21).
    void enqueue(const SettlementRecord& record) {
        {
            std::lock_guard lock(mutex_);
            if (stopping_) return;
            if (count_ == records_.size()) {
                ++dropped_;
                return;
            }
            records_[tail_] = record;
            tail_ = (tail_ + 1) % records_.size();
            ++count_;
        }
        not_empty_.notify_one();
    }

  private:
    void run() {
        for (;;) {
            SettlementRecord record;
            std::uint64_t dropped = 0;
            {
                std::unique_lock lock(mutex_);
                not_empty_.wait(lock, [this] {
                    return count_ != 0 || stopping_;
                });
                if (count_ == 0 && stopping_) return;
                record = records_[head_];
                head_ = (head_ + 1) % records_.size();
                --count_;
                dropped = dropped_;
                dropped_ = 0;
            }
            if (dropped != 0) {
                write(R"({"schema":1,"source":"driver","event":"settlement_records_dropped","count":)" +
                      std::to_string(dropped) + "}");
            }
            write(settlement_json(record));
        }
    }

    static constexpr std::size_t kCapacity = 4096;
    std::array<SettlementRecord, kCapacity> records_{};
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t count_ = 0;
    std::uint64_t dropped_ = 0;
    bool stopping_ = false;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::thread thread_;
};

inline SettlementWriter& settlement_writer() {
    static SettlementWriter writer;
    return writer;
}

inline void ensure_settlement_writer() {
    static_cast<void>(settlement_writer());
}

inline void enqueue_settled(const SettlementRecord& record) noexcept {
    try {
        settlement_writer().enqueue(record);
    } catch (...) {
    }
}

inline void write_settled_synchronously(const SettlementRecord& record) noexcept {
    try {
        write(settlement_json(record));
    } catch (...) {
    }
}

}  // namespace pie_cuda_driver::fire_timing
