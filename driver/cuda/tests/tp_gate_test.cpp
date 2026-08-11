#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "batch/tp_gate.hpp"
#include "batch/tp_fire_receipts.hpp"
#include "batch/rs_metadata.hpp"

int main() {
    {
        using pie_cuda_driver::TpFireKind;
        const auto root = pie_cuda_driver::make_tp_fire_identity(
            7, TpFireKind::MtpDraft, 16, 2, -1, 4096);
        int follower_requests = 2;
#if defined(PIE_TP_TEST_PERTURB_FIRE_RECEIPT)
        ++follower_requests;
#endif
        const auto follower = pie_cuda_driver::make_tp_fire_identity(
            7, TpFireKind::MtpDraft, 16, follower_requests, -1, 4096);
        if (!(root == follower)) {
            std::fputs(
                "TP fire receipts detected divergent logical identities\n",
                stderr);
            return 1;
        }
        const auto fire_line =
            pie_cuda_driver::tp_fire_receipt_line(3, "consume", follower);
        if (fire_line.find("rank=3 seq=7 kind=mtp_draft") ==
                std::string::npos ||
            fire_line.find("requests=2") == std::string::npos) {
            std::fputs("TP fire receipt omitted logical identity fields\n", stderr);
            return 1;
        }

        pie_cuda_driver::TpReceiptBudget budget(2);
        if (!budget.take(0) || !budget.take(0) || budget.take(0) ||
            !budget.take(1) || !budget.take(1) || budget.take(1)) {
            std::fputs("TP fire receipt budget is not bounded per rank\n", stderr);
            return 1;
        }
        pie_cuda_driver::TpReceiptBudget launch_budget(1);
        const bool launch_reserved = launch_budget.take(0);
        if (!launch_reserved || launch_budget.take(0)) {
            std::fputs("Context launch receipt was not budgeted as one pair\n", stderr);
            return 1;
        }
        const auto launch_entry = pie_cuda_driver::tp_launch_receipt_line(
            0, 9, "entry", 2, std::numeric_limits<int>::min());
        const auto launch_return = pie_cuda_driver::tp_launch_receipt_line(
            0, 9, "return", 2, 0);
        if (launch_entry.find("status=") != std::string::npos ||
            launch_return.find("status=0") == std::string::npos) {
            std::fputs("Context launch receipt pair is ambiguous\n", stderr);
            return 1;
        }

        unsetenv("PIE_TP_FIRE_RECEIPTS");
        if (pie_cuda_driver::tp_fire_receipt_limit() != 0) {
            std::fputs("TP fire receipts did not default off\n", stderr);
            return 1;
        }
        setenv("PIE_TP_FIRE_RECEIPTS", "17", 1);
        if (pie_cuda_driver::tp_fire_receipt_limit() != 17) {
            std::fputs("TP fire receipt limit was not read\n", stderr);
            return 1;
        }
        setenv("PIE_TP_FIRE_RECEIPTS", "999999", 1);
        if (pie_cuda_driver::tp_fire_receipt_limit() != 4096) {
            std::fputs("TP fire receipt hard ceiling was not enforced\n", stderr);
            return 1;
        }
        setenv("PIE_TP_FIRE_RECEIPTS", "-1", 1);
        if (pie_cuda_driver::tp_fire_receipt_limit() != 0) {
            std::fputs("TP fire receipt limit accepted a signed value\n", stderr);
            return 1;
        }
        unsetenv("PIE_TP_FIRE_RECEIPTS");
    }
    {
        // Production sequence: one ordinary forward followed by two MTP draft
        // fires. All four ranks reuse the same persistent payload/workspace.
        // A direct all-reduce loop cannot reproduce this because it never
        // overwrites a fire's inputs while another rank still consumes them.
        constexpr int kWorld = 4;
        constexpr std::uint64_t kFires = 3;
        pie_cuda_driver::TpFollowerAcks retired;
        pie_cuda_driver::TpSequenceGate gate;
        std::atomic<std::uint64_t> shared_payload{0};
        std::atomic<bool> stop{false};
        std::atomic<bool> diverged{false};
        std::vector<std::thread> followers;
        for (int rank = 1; rank < kWorld; ++rank) {
            followers.emplace_back([&, rank] {
                std::uint64_t seen = 0;
                while (seen < kFires && gate.wait_one(seen, stop)) {
                    // Force the rank skew seen in Pie rather than relying on
                    // scheduler luck: rank 1 is still consuming fire N when
                    // rank 0 is ready to publish fire N+1.
                    if (rank == 1) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(2));
                    }
                    if (shared_payload.load(std::memory_order_acquire) != seen) {
                        diverged.store(true, std::memory_order_release);
                    }
                    retired.mark(rank, seen);
                }
            });
        }

        for (std::uint64_t fire = 1; fire <= kFires; ++fire) {
            const auto kind = fire == 1
                ? pie_cuda_driver::TpFireKind::Forward
                : pie_cuda_driver::TpFireKind::MtpDraft;
            if (pie_cuda_driver::tp_fire_requires_device_retirement(kind) &&
                fire > 1) {
                while (!retired.all_consumed(kWorld, fire - 1)) {
                    std::this_thread::yield();
                }
            }
            shared_payload.store(fire, std::memory_order_release);
            gate.publish();
        }
        while (!retired.all_consumed(kWorld, kFires)) {
            std::this_thread::yield();
        }
        for (auto& follower : followers) follower.join();
        if (diverged.load(std::memory_order_acquire)) {
            std::fputs(
                "successive TP fires reused device state before all four ranks retired\n",
                stderr);
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
