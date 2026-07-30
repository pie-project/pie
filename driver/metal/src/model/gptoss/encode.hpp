#pragma once

/// GPT-OSS's step encoder.

#include <vector>

#include "../../kernels/decode_psos.hpp"
#include "../../mtl4_context.hpp"
#include "decode_step.hpp"
#include "geometry.hpp"
#include "kernels.hpp"

namespace pie::metal::gptoss {

/// The shared `Kernel` a gpt-oss kind borrows its weight map from.
Kernel shared_kind(Kind k);

/// The pipeline a dispatch runs on.
Pso pso_for(const Dispatch& d, const DecodeStepPsos& base, const GptOssPsos& go);

/// Its grid and threadgroup.
void launch_shape(const Dispatch& d, const GptOssGeometry& g, Grid& grid, Threadgroup& tg);

/// `run_ends[i]`: the last position of the concurrency run containing `i`.
std::vector<int> gptoss_run_ends(const std::vector<Dispatch>& dag);

/// Walk the DAG with a real encoder.
void encode_gptoss_step(StepEncoder& se, const std::vector<Dispatch>& dag,
                        const GptOssGeometry& g, const DecodeStepPsos& base,
                        const GptOssPsos& go, int ordinal_base = 0);

}  // namespace pie::metal::gptoss
