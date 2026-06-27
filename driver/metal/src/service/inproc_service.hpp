#pragma once

#include <cstdint>

namespace pie_driver {
class InProcServer;
class ResponseBuilder;
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
    std::uint32_t vocab_size_;
    std::uint64_t handled_ = 0;
    IForwardExecutor* executor_ = nullptr;
};

}  // namespace pie_metal_driver::service
