#pragma once

// A minimal fixed-size worker pool for off-context-thread NVRTC PTX-gen (#11).
//
// The sampling-IR JIT splits a cold compile into (1) NVRTC PTX-gen — pure CPU,
// no CUDA context — and (2) finalize (`cuModuleLoadData`/`cuMemAlloc`) which MUST
// stay on the context-owning thread. Only phase (1) runs here. NVRTC serializes
// internally (it saturates at ~2 threads on CUDA 13.3 — measured ~1.9x at N=2,
// then degrades), so the pool is sized SMALL: it hides the ~10ms compile latency
// behind in-flight steps (via the prefetch hook), it is not a throughput pool.
//
// Tasks are `void()` callables. `submit` is thread-safe; the destructor drains
// the queue (joins workers after they finish the in-flight + queued tasks).

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace pie_cuda_driver::sampling_ir {

class ThreadPool {
  public:
    explicit ThreadPool(unsigned n) {
        if (n == 0) n = 1;
        workers_.reserve(n);
        for (unsigned i = 0; i < n; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread& t : workers_)
            if (t.joinable()) t.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            tasks_.push(std::move(task));
        }
        cv_.notify_one();
    }

    unsigned size() const { return static_cast<unsigned>(workers_.size()); }

  private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();  // run outside the lock
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stop_ = false;
};

}  // namespace pie_cuda_driver::sampling_ir
