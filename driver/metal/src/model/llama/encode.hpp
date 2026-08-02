#pragma once

/// The llama families' step encoder.
///
/// Turns the DAG into commands. Almost every dispatch borrows a shared
/// `Kernel`, because the shared enum's common prefix already IS a llama
/// decoder; what this file adds is the two places that is not true -- an untied
/// embedding/head, and the routed FFN.

#include <vector>

#include "../../kernels/decode_psos.hpp"
#include "../../mtl4_context.hpp"
#include "decode_step.hpp"
#include "geometry.hpp"
#include "kernels.hpp"

namespace pie::metal::llama {

/// The shared `Kernel` a llama kind borrows its WEIGHT MAP from.
///
/// Takes the geometry because two kinds' answer depends on the checkpoint
/// rather than on the kind: a tied model's embedding and head are both
/// `shared_embedding`, an untied one's are separate tensors, and asking for the
/// wrong pair is a load failure rather than a wrong number.
Kernel shared_kind(Kind k, const LlamaGeometry& g);

/// The kind whose PIPELINE a llama kind runs on -- a different question from
/// `shared_kind`, which is the weight-map key. They differ for the routed
/// matvecs, which share the expert weight names but run the ROUTED kernel.
Kernel pso_kind(Kind k);

/// The pipeline a dispatch runs on.
Pso pso_for(const Dispatch& d, const LlamaGeometry& g, const DecodeStepPsos& base,
            const LlamaPsos& ll);

/// Its grid and threadgroup.
void launch_shape(const Dispatch& d, const LlamaGeometry& g, Grid& grid, Threadgroup& tg);

/// `run_ends[i]`: the last position of the concurrency run containing `i`. What
/// the colourer needs to know about which barriers the encoder will drop.
std::vector<int> llama_run_ends(const std::vector<Dispatch>& dag);

/// Walk the DAG with a real encoder.
void encode_llama_step(StepEncoder& se, const std::vector<Dispatch>& dag,
                       const LlamaGeometry& g, const DecodeStepPsos& base, const LlamaPsos& ll,
                       int ordinal_base = 0);

}  // namespace pie::metal::llama
