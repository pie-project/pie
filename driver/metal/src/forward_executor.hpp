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

namespace pie_metal_driver {

class IForwardExecutor {
public:
    virtual ~IForwardExecutor() = default;

    // Run one forward + sampling pass. Fills `out` via `builder` (whose scratch
    // backs the response-view slices until the next build()).
    virtual void run_forward(const pie_driver::PieForwardRequestView& req,
                             pie_driver::ResponseBuilder& builder,
                             pie_driver::PieForwardResponseView& out) = 0;
};

}  // namespace pie_metal_driver
