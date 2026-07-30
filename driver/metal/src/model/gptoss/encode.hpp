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

/// The pipeline a dispatch runs on with PAGED KV. Differs from `pso_for` for
/// exactly the two kinds whose kernel changes with the KV layout; everything
/// else falls through rather than being restated, where a copy could drift.
Pso pso_for_paged(const Dispatch& d, const DecodeStepPsos& base, const MultiBatchPsos& mb,
                  const GptOssPsos& go);

/// Its grid and threadgroup.
void launch_shape(const Dispatch& d, const GptOssGeometry& g, Grid& grid, Threadgroup& tg);

/// `run_ends[i]`: the last position of the concurrency run containing `i`.
std::vector<int> gptoss_run_ends(const std::vector<Dispatch>& dag);

/// Walk the DAG with a real encoder.
/// The pipeline a dispatch runs on at M>1, its launch shape, and the walk.
///
/// `head_rows` is how many rows the fire SAMPLES -- what `Kind::RowGather`
/// compacts, and what the tail after it runs on. 0 means every row.
Pso pso_for_mb(const Dispatch& d, const DecodeStepPsos& base, const MultiBatchPsos& mb,
               const GptOssPsos& go);
void launch_shape_mb(const Dispatch& d, const GptOssGeometry& g, int rows, Grid& grid,
                     Threadgroup& tg, int head_rows = 0);
void encode_gptoss_step_mb(StepEncoder& se, const std::vector<Dispatch>& dag,
                           const GptOssGeometry& g, int rows, const DecodeStepPsos& base,
                           const MultiBatchPsos& mb, const GptOssPsos& go,
                           int ordinal_base = 0, int head_rows = 0);

/// Encode the step against paged KV. One row -- gpt-oss has no M>1 path -- but
/// the row's history is a page list, so several sequences coexist.
void encode_gptoss_step_paged(StepEncoder& se, const std::vector<Dispatch>& dag,
                              const GptOssGeometry& g, const DecodeStepPsos& base,
                              const MultiBatchPsos& mb, const GptOssPsos& go,
                              int ordinal_base = 0);

void encode_gptoss_step(StepEncoder& se, const std::vector<Dispatch>& dag,
                        const GptOssGeometry& g, const DecodeStepPsos& base,
                        const GptOssPsos& go, int ordinal_base = 0);

}  // namespace pie::metal::gptoss
