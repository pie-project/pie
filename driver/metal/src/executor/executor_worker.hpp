#pragma once

// ExecutorWorker — Phase 3 (metal_ptir_plan.md §7, D4) single-owner serializer
// for every RawMetalDecoder / RawMetalContext touch AND for the driver's own
// state mutations that must not race an in-flight forward. All jobs run on ONE
// dedicated worker thread in FIFO submission order, giving:
//
//   1. Thread-affinity: Metal command queues / MTL4 command allocators are not
//      safe to use concurrently from several threads. Routing every executor
//      call through the single worker thread means the RawMetalContext is only
//      ever driven from one thread.
//   2. Serialization: a control op (copy_kv / copy_state / resize_pool /
//      close) can never run while a forward is mid-flight, and a launch's
//      settlement can never race a close of the same instance — they are the
//      same FIFO queue's jobs.
//
// Two submission modes:
//   * run(job)  — SYNCHRONOUS: enqueue, block until the job finishes, and if
//                 the job threw, RETHROW the original exception on the caller
//                 (item 3). Used by control ops (copy/resize/close) that own a
//                 synchronous ABI return code and must settle behind any queued
//                 launches.
//   * post(job) — ASYNCHRONOUS: enqueue and return immediately without waiting
//                 (item 1). Used by `pie_metal_launch` so the ABI call returns
//                 after acceptance, before the GPU forward + settlement run.
//                 A posted job MUST be self-contained (it catches its own
//                 exceptions and translates them to per-member terminal
//                 failures); the worker loop additionally captures any stray
//                 throw into the (unobserved) ticket so a single bad job can
//                 never tear down the worker thread.
//   * drain()   — block until every job submitted so far has finished (FIFO, so
//                 enqueuing one more sync no-op and waiting for it suffices).
//                 Used at teardown and by tests.
//
// Pure std::thread — NO Metal/ObjC dependency — so it compiles and is unit-
// tested on every platform (tests/executor_worker_test.cpp), independent of a
// checkpoint or an Apple build.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace pie_metal_driver::executor {

class ExecutorWorker {
  public:
    ExecutorWorker() : thread_([this] { loop(); }) {}

    ~ExecutorWorker() {
        drain();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    ExecutorWorker(const ExecutorWorker&) = delete;
    ExecutorWorker& operator=(const ExecutorWorker&) = delete;

    // Synchronous: enqueue, wait, and rethrow the job's exception (if any) on
    // the caller. Re-entrant calls from the worker thread run inline (a job
    // that submits another job) to avoid self-deadlock; an inline job's
    // exception propagates directly.
    void run(const std::function<void()>& job) {
        if (std::this_thread::get_id() == thread_.get_id()) {
            job();  // inline — exceptions propagate to the enclosing job
            return;
        }
        std::shared_ptr<Ticket> ticket = enqueue(job);
        {
            std::unique_lock<std::mutex> done_lock(ticket->done_mutex);
            ticket->done_cv.wait(done_lock, [&] { return ticket->done; });
        }
        if (ticket->error) std::rethrow_exception(ticket->error);
    }

    // Asynchronous: enqueue and return immediately. The job owns its own error
    // handling; any escaped exception is captured (and swallowed here since no
    // one waits) purely to protect the worker thread.
    void post(const std::function<void()>& job) {
        if (std::this_thread::get_id() == thread_.get_id()) {
            // Already on the worker: run inline but never let a throw escape
            // into the enclosing job's control flow — a posted job is
            // fire-and-forget by contract.
            try {
                job();
            } catch (...) {
            }
            return;
        }
        enqueue(job);
    }

    // Block until every job submitted before this call has finished.
    void drain() {
        if (std::this_thread::get_id() == thread_.get_id()) return;  // can't drain self
        run([] {});
    }

    std::thread::id worker_thread_id() const { return thread_.get_id(); }

    std::uint64_t submitted() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return submitted_;
    }

  private:
    struct Ticket {
        std::function<void()> job;
        std::mutex done_mutex;
        std::condition_variable done_cv;
        bool done = false;
        std::exception_ptr error;  // captured job exception (item 3)
    };

    std::shared_ptr<Ticket> enqueue(const std::function<void()>& job) {
        auto ticket = std::make_shared<Ticket>();
        ticket->job = job;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(ticket);
            ++submitted_;
        }
        cv_.notify_one();
        return ticket;
    }

    void loop() {
        for (;;) {
            std::shared_ptr<Ticket> ticket;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                ticket = std::move(queue_.front());
                queue_.pop_front();
            }
            // Run outside the queue lock so submitters aren't blocked on the
            // (potentially multi-millisecond) job. Capture any exception into
            // the ticket: `run` rethrows it on the (waiting) caller; `post`
            // leaves it unobserved. Either way the worker thread survives.
            try {
                ticket->job();
            } catch (...) {
                ticket->error = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> done_lock(ticket->done_mutex);
                ticket->done = true;
            }
            ticket->done_cv.notify_all();
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Ticket>> queue_;
    bool stop_ = false;
    std::uint64_t submitted_ = 0;
    std::thread thread_;
};

}  // namespace pie_metal_driver::executor
