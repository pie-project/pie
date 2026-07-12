#pragma once

// Compose one submitted PTIR batch into device work and fire it.

#include <cstdint>

#include "pie_native/launch_view.hpp"
#include <pie_driver_abi.h>

namespace pie_cuda_driver {

struct BatchEngine;

// Run the forward pass for one direct/PTIR fire. `req_id` is used in error
// logging. Resolves device-geometry descriptors, composes the forward
// batch, decodes/packs the attention mask, uploads explicit KV-write
// descriptors, builds the sampling-row set, dispatches the forward
// (`batch/forward.hpp`), gathers the selected logits (`batch/logits.hpp`),
// and fires the pipeline dispatch.
void handle_fire_batch(
    std::uint32_t req_id,
    const pie_native::LaunchView& view,
    BatchEngine& engine,
    const PieRuntimeCallbacks& runtime,
    PieCompletion completion);

}  // namespace pie_cuda_driver
