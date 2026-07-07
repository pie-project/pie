#pragma once

// Abstract forward-executor seam for the Metal driver.
//
// `InProcService` holds an `IForwardExecutor*` and delegates the Forward arm to
// it, un-gated by any compute backend. Two implementations plug in behind the
// same wire contract (`PieForwardRequestView` → `ResponseBuilder`):
//
//   * RawMetalExecutor (default, MLX-free)  — our raw Metal-4 decode pipeline
//     (heap_bind weights + DAG + PSOs + KV/linear-state), the path the worker
//     runs e2e.
//   * Executor         (PIE_METAL_HAS_MLX)  — the MLX `ModelGraph` path, kept
//     as an optional comparison backend.
//
// The interface references only MLX-free pie_driver_abi types, so it compiles in
// both the default (no-MLX) and MLX-on builds.

#include <pie_driver_abi/response_builder.hpp>
#include <pie_driver_abi/view.hpp>

#include <memory>

namespace pie_metal_driver {

class IForwardExecutor {
public:
    virtual ~IForwardExecutor() = default;

    // Run one forward + sampling pass. Fills `out` via `builder` (whose scratch
    // backs the response-view slices until the next build()).
    virtual void run_forward(const pie_driver::PieForwardRequestView& req,
                             pie_driver::ResponseBuilder& builder,
                             pie_driver::PieForwardResponseView& out) = 0;

    // ── Deferred-response (async double-buffer) seam ───────────────────────
    // Opaque in-flight handle (kept MLX-free so this header compiles in the
    // no-MLX build; the MLX Executor's concrete handle derives from it and owns
    // the device token). Enables the serve loop to submit forward N+1 before
    // collecting N (the wave's N+1-ahead) — the deferred-send path.
    struct Inflight {
        virtual ~Inflight() = default;
    };
    // True iff submit() is genuinely non-blocking (deferred-send eligible). The
    // default (raw path, for now) is synchronous, so the serve loop uses
    // run_forward; the MLX Executor overrides this to true.
    virtual bool supports_deferred() const { return false; }
    // Enqueue the forward WITHOUT waiting (async executors); returns an opaque
    // in-flight handle. Only called when supports_deferred() is true.
    virtual std::unique_ptr<Inflight> submit(
        const pie_driver::PieForwardRequestView& /*req*/) {
        return nullptr;
    }
    // Block for the in-flight forward and marshal its response into `out` (the
    // sync point runs the eval inside the concrete executor, so the caller
    // stays MLX-free). Invoked off the serve thread by the completion.
    virtual void collect(Inflight& /*handle*/, pie_driver::ResponseBuilder& /*builder*/,
                         pie_driver::PieForwardResponseView& /*out*/) {}
};

}  // namespace pie_metal_driver
