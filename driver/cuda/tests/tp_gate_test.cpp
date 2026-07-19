#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

#include "batch/tp_gate.hpp"
#include "batch/rs_metadata.hpp"

int main() {
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
