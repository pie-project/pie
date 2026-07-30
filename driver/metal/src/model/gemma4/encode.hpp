#pragma once

/// Gemma 4's step encoder.
///
/// Turns the DAG into commands. Two choices are per-dispatch rather than per
/// kind, and both follow the layer's attention type: which SDPA pipeline
/// (`d=256` sliding, `d=512` full) and which grid, since head_dim and the MLP
/// width vary by layer.

#include <vector>

#include "../../kernels/decode_psos.hpp"
#include "../../mtl4_context.hpp"
#include "decode_step.hpp"
#include "geometry.hpp"
#include "kernels.hpp"

namespace pie::metal::gemma4 {

/// The shared `Kernel` a gemma4 kind borrows its pipeline and weight map from.
Kernel shared_kind(Kind k);

/// The kind whose PIPELINE a gemma4 kind runs on — a different question from
/// `shared_kind`, which is the weight-map key.
Kernel pso_kind(Kind k);

/// The pipeline a dispatch runs on.
Pso pso_for(const Dispatch& d, const DecodeStepPsos& base, const Gemma4Psos& g4);

/// Its grid and threadgroup, from the geometry and this dispatch's layer.
void launch_shape(const Dispatch& d, const Gemma4Geometry& g, Grid& grid, Threadgroup& tg);

/// `run_ends[i]`: the last ordinal of the concurrency run containing `i`. What
/// the colourer needs to know about which barriers the encoder will drop.
std::vector<int> gemma4_run_ends(const std::vector<Dispatch>& dag);

/// Walk the DAG with a real encoder.
void encode_gemma4_step(StepEncoder& se, const std::vector<Dispatch>& dag,
                        const Gemma4Geometry& g, const DecodeStepPsos& base,
                        const Gemma4Psos& g4, int ordinal_base = 0);

}  // namespace pie::metal::gemma4
