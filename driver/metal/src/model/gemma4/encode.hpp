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
Pso pso_for(const Dispatch& d, const Gemma4Geometry& g, const DecodeStepPsos& base,
            const Gemma4Psos& g4);

/// Its grid and threadgroup, from the geometry and this dispatch's layer.
void launch_shape(const Dispatch& d, const Gemma4Geometry& g, Grid& grid, Threadgroup& tg);

/// The same, for a batch of `rows` tokens. At rows==1 it must agree with
/// `launch_shape` exactly -- one path, checked, rather than two that drift.
/// How many rows a projection's GEMM runs over, given the fire's `rows`.
///
/// Padded up to a whole `BM` tile so the GEMM engages at any batch instead of
/// only at exact multiples of it. The pool holds `gemma4_qmm_pool_rows(max)`
/// rows so the padding always lands somewhere allocated.
int gemma4_qmm_rows(const Gemma4Geometry& g, int rows);

/// How many rows the activation pool must hold for `max_rows` to be paddable.
int gemma4_qmm_pool_rows(int max_rows);

/// `head_rows` is how many rows the fire SAMPLES -- what `Kind::RowGather`
/// compacts, and what the tail after it runs on. 0 means every row.
/// `requests` is the fire's request count. With `rows` it decides the
/// attention's shape; 0 means the caller does not know, which keeps the
/// per-row kernel. `pso_for_mb` must be asked the SAME number.
void launch_shape_mb(const Dispatch& d, const Gemma4Geometry& g, int rows, Grid& grid,
                     Threadgroup& tg, int head_rows = 0, int requests = 0);

/// The pipeline a dispatch runs on at M>1. Differs from `pso_for` only where the
/// kernel itself changes with the batch; everything else falls through.
Pso pso_for_mb(const Dispatch& d, const Gemma4Geometry& g, int rows,
               const DecodeStepPsos& base, const MultiBatchPsos& mb,
               const Gemma4Psos& g4, int head_rows = 0, int requests = 0);

/// Encode the step for a batch of `rows` tokens. Same walk as
/// `encode_gemma4_step` -- the DAG, its order and its concurrency runs belong to
/// the model, not to the batch size -- differing only in shape and pipeline.
/// Whether a dispatch's weights are in the geometry's SECOND affine format.
///
/// Exactly the dense FFN's three projections and the router -- what mlx_lm's
/// quantization predicate spares on gemma-4-26B. Asked in one place so the PSO
/// choice cannot disagree with the fit check.
bool gemma4_uses_alt_quant(Kind k);

/// Whether this dispatch's GEMM reads an FP16 copy of its input, and whether it
/// is the one that stages it. Public because the binding needs the same answer
/// the encoder gets -- slot 12 has to be bound on exactly the dispatches that
/// read it.
bool gemma4_fp16_qmm(const Gemma4Geometry& g, const Dispatch& d, int m);
bool gemma4_fp16_cast_before(Kind k);

void encode_gemma4_step_mb(StepEncoder& se, const std::vector<Dispatch>& dag,
                           const Gemma4Geometry& g, int rows,
                           const DecodeStepPsos& base, const MultiBatchPsos& mb,
                           const Gemma4Psos& g4, int ordinal_base = 0, int head_rows = 0,
                           /// The tables at `g.ffn_quant`, when the checkpoint
                           /// has a second affine format. Null means it has one.
                           const DecodeStepPsos* base_alt = nullptr,
                           const MultiBatchPsos* mb_alt = nullptr,
                           /// The fire's request count, which with `rows` is
                           /// what decides the attention's shape. 0 means the
                           /// caller does not know, and an unknown fire keeps
                           /// the per-row kernel.
                           int requests = 0,
                           /// The half-open slice of the DAG to encode. The
                           /// default is the whole of it; a paging engine asks
                           /// for one segment at a time so the host can refill
                           /// the expert slab between command buffers.
                           std::size_t begin = 0, std::size_t end = 0);

/// `run_ends[i]`: the last ordinal of the concurrency run containing `i`. What
/// the colourer needs to know about which barriers the encoder will drop.
std::vector<int> gemma4_run_ends(const std::vector<Dispatch>& dag);

/// Walk the DAG with a real encoder.
void encode_gemma4_step(StepEncoder& se, const std::vector<Dispatch>& dag,
                        const Gemma4Geometry& g, const DecodeStepPsos& base,
                        const Gemma4Psos& g4, int ordinal_base = 0,
                        const DecodeStepPsos* base_alt = nullptr);

}  // namespace pie::metal::gemma4
