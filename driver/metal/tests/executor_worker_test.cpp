// Phase 3 (metal_ptir_plan.md §7, D4) executor-worker gate — pure host unit
// test, no Metal/Apple/checkpoint dependency (`ExecutorWorker` is std::thread
// only). Validates the single-owner serializer the driver routes every
// MetalExecutor / RawMetalContext touch through:
//   * every job runs on the ONE worker thread (device-object thread-affinity);
//   * jobs run in FIFO submission order from a single submitter;
//   * a job submitted re-entrantly FROM the worker runs inline (no deadlock);
//   * concurrent submitters never interleave a job body (serialization):
//     an unguarded shared counter incremented inside jobs stays exact;
//   * a throwing job does not tear down the worker (later jobs still run);
//   * run() RETHROWS a job's exception on the caller (item 3);
//   * post() returns BEFORE a deliberately delayed job finishes (item 1/5),
//     and drain() blocks until it does; ordering (a post()ed job then a run()
//     job) is FIFO (item 5 "close queues behind launch").

#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "batch/worker.hpp"

using pie::metal::batch::ExecutorWorker;

namespace {
int g_pass = 0, g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}
}  // namespace

int main() {
    std::printf("[ExecutorWorker]\n");

    ExecutorWorker worker;
    const std::thread::id caller = std::this_thread::get_id();

    // All jobs execute on the worker thread, not the caller.
    {
        std::thread::id ran_on;
        worker.run([&] { ran_on = std::this_thread::get_id(); });
        expect(ran_on == worker.worker_thread_id() && ran_on != caller,
               "job runs on the worker thread, not the caller");
    }

    // FIFO order from a single submitter.
    {
        std::vector<int> order;
        for (int i = 0; i < 16; ++i) worker.run([&, i] { order.push_back(i); });
        bool fifo = order.size() == 16;
        for (int i = 0; i < 16 && fifo; ++i) fifo = order[i] == i;
        expect(fifo, "jobs from one submitter run in FIFO order");
    }

    // Re-entrant submission from within a job runs inline (no self-deadlock).
    {
        bool inner_ran = false;
        std::thread::id inner_thread;
        worker.run([&] {
            worker.run([&] {
                inner_ran = true;
                inner_thread = std::this_thread::get_id();
            });
        });
        expect(inner_ran && inner_thread == worker.worker_thread_id(),
               "re-entrant job runs inline on the worker (no deadlock)");
    }

    // Concurrency: many submitter threads, an UNGUARDED counter incremented
    // inside each job. If the worker ever ran two job bodies at once the
    // read-modify-write would lose increments; exact total proves serialization.
    {
        constexpr int kThreads = 8;
        constexpr int kPer = 500;
        long unguarded = 0;  // deliberately not atomic — the worker serializes
        std::vector<std::thread> submitters;
        for (int t = 0; t < kThreads; ++t) {
            submitters.emplace_back([&] {
                for (int i = 0; i < kPer; ++i) worker.run([&] { unguarded += 1; });
            });
        }
        for (auto& s : submitters) s.join();
        expect(unguarded == long(kThreads) * kPer,
               "concurrent submitters never interleave a job body (serialized)");
    }

    // A throwing job is contained; the worker keeps serving later jobs.
    {
        worker.post([] { throw std::runtime_error("boom"); });
        bool after_ran = false;
        worker.run([&] { after_ran = true; });
        expect(after_ran, "a throwing posted job does not tear down the worker");
    }

    // run() RETHROWS the job's exception on the caller with the original what()
    // (item 3).
    {
        std::string caught;
        try {
            worker.run([] { throw std::runtime_error("propagate-me"); });
        } catch (const std::exception& e) {
            caught = e.what();
        }
        expect(caught == "propagate-me", "run() rethrows the job's exception (" + caught + ")");
    }

    // post() returns BEFORE a deliberately delayed job finishes (item 1/5);
    // drain() then blocks until it (and everything after) completes.
    {
        std::atomic<bool> release{false};
        std::atomic<bool> job_done{false};
        worker.post([&] {
            while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            job_done.store(true);
        });
        // The posted job is blocked on `release`; post() must have already
        // returned, so job_done is still false here.
        const bool returned_early = !job_done.load();
        release.store(true);
        worker.drain();
        expect(returned_early, "post() returns before a delayed job finishes");
        expect(job_done.load(), "drain() blocks until the delayed job completes");
    }

    // Ordering: a post()ed (delayed) job then a run() job — the run() job must
    // observe the posted job's effect (FIFO; "a close queues behind a launch").
    {
        std::atomic<bool> release{false};
        int marker = 0;
        worker.post([&] {
            while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            marker = 1;  // launch-like job runs first
        });
        std::thread releaser([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            release.store(true);
        });
        int observed = -1;
        worker.run([&] { observed = marker; });  // close-like job runs AFTER
        releaser.join();
        expect(observed == 1, "a run() job queues strictly behind a prior post()ed job");
    }

    std::printf("\n==== executor_worker_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
