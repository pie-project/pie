#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "batch/tp_gate.hpp"
#include "batch/rs_metadata.hpp"

int main() {
    using pie_cuda_driver::TpFollowerPhase;
    for (const auto [phase, expected] : {
             std::pair{TpFollowerPhase::GateWait, "gate_wait"},
             std::pair{TpFollowerPhase::Header, "header"},
             std::pair{TpFollowerPhase::PayloadEnqueue, "payload_enqueue"},
             std::pair{TpFollowerPhase::GroupEnd, "group_end"},
             std::pair{TpFollowerPhase::AsyncPoll, "async_poll"},
             std::pair{TpFollowerPhase::PayloadDone, "payload_done"},
             std::pair{TpFollowerPhase::HostViews, "host_views"},
             std::pair{TpFollowerPhase::Consumed, "consumed"},
         }) {
        if (std::string_view(pie_cuda_driver::tp_follower_phase_name(phase)) !=
            expected) {
            std::fputs("TP follower phase telemetry is mislabeled\n", stderr);
            return 1;
        }
    }
    {
        pie_cuda_driver::TpFollowerAcks acks;
        acks.mark(2, 8);
        if (acks.all_consumed(4, 8)) {
            std::fputs(
                "TP mailbox released a slot before every follower consumed it\n",
                stderr);
            return 1;
        }
        acks.mark(1, 8);
        if (acks.all_consumed(4, 8)) {
            std::fputs(
                "TP mailbox released a slot while rank 3 was still unread\n",
                stderr);
            return 1;
        }
        acks.mark(3, 8);
        if (!acks.all_consumed(4, 8)) {
            std::fputs(
                "TP mailbox did not release a slot after every follower consumed it\n",
                stderr);
            return 1;
        }
    }
    {
        // Closest CPU-only reproduction seam for the integrated fire path:
        // rank 0 publishes a header into the real ring shape, the real shared
        // sequence gate wakes three followers, and every follower copies the
        // header before marking the real per-rank acknowledgement. Payload
        // NCCL enqueue/completion remains outside a CPU-only test.
        constexpr std::uint64_t kRing = 8;
        constexpr std::uint64_t kFires = 20'000;
        std::array<std::uint64_t, kRing> headers{};
        pie_cuda_driver::TpFollowerAcks acks;
        pie_cuda_driver::TpSequenceGate gate;
        std::atomic<bool> stop{false};
        std::atomic<bool> failed{false};
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(10);
        std::vector<std::thread> followers;
        for (int rank = 1; rank < 4; ++rank) {
            followers.emplace_back([&, rank] {
                std::uint64_t seen = 0;
                while (seen < kFires && gate.wait_one(seen, stop)) {
                    const auto header = headers[(seen - 1) % kRing];
                    if (header != seen) {
                        failed.store(true, std::memory_order_release);
                        break;
                    }
                    if (rank == 2 && seen % 97 == 0) {
                        std::this_thread::sleep_for(
                            std::chrono::microseconds(50));
                    }
                    acks.mark(rank, seen);
                }
            });
        }
        for (std::uint64_t fire = 1; fire <= kFires; ++fire) {
            if (fire > kRing) {
                const auto oldest = fire - kRing;
                while (!acks.all_consumed(4, oldest)) {
                    if (failed.load(std::memory_order_acquire)) break;
                    if (std::chrono::steady_clock::now() > deadline) {
                        failed.store(true, std::memory_order_release);
                        break;
                    }
                    std::this_thread::yield();
                }
            }
            if (failed.load(std::memory_order_acquire)) break;
            headers[(fire - 1) % kRing] = fire;
            gate.publish();
        }
        while (!failed.load(std::memory_order_acquire) &&
               !acks.all_consumed(4, kFires)) {
            if (std::chrono::steady_clock::now() > deadline) {
                failed.store(true, std::memory_order_release);
                break;
            }
            std::this_thread::yield();
        }
        if (failed.load(std::memory_order_acquire)) {
            stop.store(true, std::memory_order_release);
            gate.request_stop();
        }
        for (auto& follower : followers) follower.join();
        if (failed.load(std::memory_order_acquire) ||
            !acks.all_consumed(4, kFires)) {
            std::fputs(
                "integrated TP host fire delivery lost a follower\n", stderr);
            return 1;
        }
    }
    std::uint64_t consumed = 0;
    if (!pie_cuda_driver::tp_cpu_gate_consume_one(3, consumed) ||
        consumed != 1 ||
        !pie_cuda_driver::tp_cpu_gate_consume_one(3, consumed) ||
        consumed != 2 ||
        !pie_cuda_driver::tp_cpu_gate_consume_one(3, consumed) ||
        consumed != 3 ||
        pie_cuda_driver::tp_cpu_gate_consume_one(3, consumed)) {
        std::fputs("TP CPU gate collapsed a notification burst\n", stderr);
        return 1;
    }
    using pie_cuda_driver::RsExecutionMode;
    if (!pie_cuda_driver::tp_rs_metadata_shape_valid(
            RsExecutionMode::BufferFold,
            2, 2, 2, 2, 3, 3) ||
        !pie_cuda_driver::tp_rs_metadata_shape_valid(
            RsExecutionMode::BufferWrite,
            2, 2, 2, 2, 3, 3) ||
        pie_cuda_driver::tp_rs_metadata_shape_valid(
            RsExecutionMode::BufferFold,
            2, 2, 2, 1, 3, 3) ||
        pie_cuda_driver::tp_rs_metadata_shape_valid(
            RsExecutionMode::Forward,
            2, 2, 2, 2, 3, 3)) {
        std::fputs("TP RS payload metadata can diverge across ranks\n", stderr);
        return 1;
    }
    if (!pie_cuda_driver::rs_launch_requires_readiness_settlement(
            2, 2, 3, 3) ||
        pie_cuda_driver::rs_launch_requires_readiness_settlement(
            0, 0, 0, 3)) {
        std::fputs(
            "stateful RS readiness settlement policy is incomplete\n",
            stderr);
        return 1;
    }
    {
        pie_cuda_driver::TpSequenceGate gate;
        std::atomic<bool> stop{false};
        std::uint64_t seen = 0;
        gate.publish();
        if (!gate.wait_one(seen, stop) || seen != 1) {
            std::fputs("TP gate lost a publish-before-wait epoch\n", stderr);
            return 1;
        }
        for (std::uint64_t epoch = 2; epoch <= 500; ++epoch) {
            std::atomic<bool> waiting{false};
            std::atomic<bool> done{false};
            std::thread waiter([&] {
                waiting.store(true, std::memory_order_release);
                if (gate.wait_one(seen, stop)) {
                    done.store(true, std::memory_order_release);
                }
            });
            while (!waiting.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            gate.publish();
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::milliseconds(250);
            while (!done.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::yield();
            }
            if (!done.load(std::memory_order_acquire)) {
                // Unblock a broken implementation so the regression exits.
                gate.publish();
                waiter.join();
                std::fputs("TP gate lost a concurrent wakeup\n", stderr);
                return 1;
            }
            waiter.join();
            if (seen != epoch) {
                std::fputs("TP gate consumed the wrong epoch\n", stderr);
                return 1;
            }
        }
    }
    std::puts("tp_gate_test: OK");
    return 0;
}
