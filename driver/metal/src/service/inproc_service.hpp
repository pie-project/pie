#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include <pie_ipc.h>
#include <pie_driver_abi/response_builder.hpp>
#include <pie_driver_abi/view.hpp>

#include "forward_executor.hpp"

namespace pie_driver {
class InProcServer;
struct PieInProcRequestView;
struct PieInProcResponseView;
}  // namespace pie_driver

namespace pie_metal_driver {
class IForwardExecutor;
}  // namespace pie_metal_driver

namespace pie_metal_driver::service {

// In-process service for the Metal driver.
//
// Handles the driver ABI for `pie_driver::InProcServer`:
//
//   * Health         → status 0.
//   * Forward        → when an executor is attached (`set_executor`, the real
//                      compute path — the default MLX-free raw-Metal pipeline,
//                      or the optional MLX `ModelGraph`), delegates to it.
//                      With no executor (before a model is loaded), falls back
//                      to dummy tokens / zeroed distributions shaped exactly
//                      like the per-request sampler stream (mirrors
//                      driver/dummy).
//   * Copy / KV / RS → not-supported status (no KV cache yet — delta wires it).
//   * Adapter        → no-op success.
//
// The `Executor` (and the `ModelGraph` + `PagedKvCache` it references) is
// owned by the caller (entry.cpp's serve loop) and outlives the service; the
// service only borrows it.
class InProcService {
public:
    explicit InProcService(std::uint32_t vocab_size = 1) : vocab_size_(vocab_size) {}
    ~InProcService();

    void serve_forever(pie_driver::InProcServer& server);

    // Dispatch a single in-process request (the body of the serve loop,
    // exposed for direct/unit driving). `builder`'s scratch backs the
    // response-view slices in `out` and must outlive the caller's use of them.
    void handle_request(std::uint32_t req_id,
                        const pie_driver::PieInProcRequestView& req,
                        pie_driver::PieInProcResponseView& out,
                        pie_driver::ResponseBuilder& builder);

    std::uint64_t handled() const noexcept { return handled_; }

    // Attach the live forward pipeline. When set, the Forward arm delegates to
    // `exec->run_forward`; the pointee must outlive the serve loop.
    void set_executor(IForwardExecutor* exec) noexcept { executor_ = exec; }

private:
    // ── Deferred-response completion (async pipeline, §D3) ─────────────────
    // When the attached executor `supports_deferred()`, the Forward arm calls
    // `executor_->submit` (async_eval, non-blocking) + sets `out.deferred`, so
    // serve_forever skips the inline send and hands us the (req_id, vtable) via
    // `defer_send_`. A single completion thread then `collect`s the in-flight
    // forward OFF the serve thread (so the serve loop submits N+1 while N runs)
    // and fires `send_response` (rule 5b: guaranteed send, success-or-error).
    struct Pending {
        ::PieInProcVTable vt{};
        std::uint32_t req_id = 0;
        std::uint32_t driver_id = 0;
        std::unique_ptr<IForwardExecutor::Inflight> handle;
    };
    void start_completion_thread();
    void stop_completion_thread();
    void completion_loop();
    void deliver_completion(Pending& p);
    void enqueue_completion(Pending p);

    std::uint32_t vocab_size_;
    std::uint64_t handled_ = 0;
    IForwardExecutor* executor_ = nullptr;

    // The submit stashed by handle_request, moved into the completion queue by
    // the `defer_send_` hook (both run on the serve thread, in sequence).
    std::unique_ptr<IForwardExecutor::Inflight> pending_submit_;
    std::uint32_t pending_driver_id_ = 0;

    std::thread                completion_thread_;
    std::mutex                 completion_mu_;
    std::condition_variable    completion_cv_;
    std::deque<Pending>        completion_q_;
    bool                       completion_stop_ = false;
    pie_driver::ResponseBuilder completion_builder_;  // completion-thread-only
};

}  // namespace pie_metal_driver::service
