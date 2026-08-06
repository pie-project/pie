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
Pso pso_for_mb_rows(const Dispatch& d, const GptOssGeometry& g, int rows,
                    const DecodeStepPsos& base, const MultiBatchPsos& mb,
                    const GptOssPsos& go, int head_rows = 0);
/// How many rows the activation pool must hold for `max_rows` to be paddable to
/// a whole GEMM tile.
int gptoss_qmm_pool_rows(int max_rows);
int gptoss_moe_pairs(const GptOssGeometry& g, int rows);
int gptoss_moe_tile_rows(const GptOssGeometry& g, int rows);
int gptoss_moe_sorted_rows(const GptOssGeometry& g, int rows);
void launch_shape_mb(const Dispatch& d, const GptOssGeometry& g, int rows, Grid& grid,
                     Threadgroup& tg, int head_rows = 0);
/// `[begin, end)` walks a SLICE of the step. The default is the whole DAG; a
/// narrower range is how a step is cut into ordered command buffers so the
/// host can page routed experts in between them. The walk is otherwise
/// identical, barriers included -- a segment boundary already ends a command
/// buffer, which is a stronger ordering than a barrier.
void encode_gptoss_step_mb(StepEncoder& se, const std::vector<Dispatch>& dag,
                           const GptOssGeometry& g, int rows, const DecodeStepPsos& base,
                           const MultiBatchPsos& mb, const GptOssPsos& go,
                           int ordinal_base = 0, int head_rows = 0,
                           /// The fire's request count, which with `rows` is
                           /// what decides the attention's shape. 0 means the
                           /// caller does not know, and an unknown fire keeps
                           /// the per-row kernel.
                           int requests = 0, std::size_t begin = 0,
                           std::size_t end = 0);

/// Encode a single-row step against paged KV. Wider fires use
/// `encode_gptoss_step_mb`, including the sorted routed GEMM.
void encode_gptoss_step_paged(StepEncoder& se, const std::vector<Dispatch>& dag,
                              const GptOssGeometry& g, const DecodeStepPsos& base,
                              const MultiBatchPsos& mb, const GptOssPsos& go,
                              int ordinal_base = 0);

void encode_gptoss_step(StepEncoder& se, const std::vector<Dispatch>& dag,
                        const GptOssGeometry& g, const DecodeStepPsos& base,
                        const GptOssPsos& go, int ordinal_base = 0);

}  // namespace pie::metal::gptoss
