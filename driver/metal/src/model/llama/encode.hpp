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
///
/// `g.paged_kv_enabled` decides the two ATTENTION kinds, and it decides them
/// together. gpt-oss answers this with a second selector, `pso_for_paged`,
/// which is a switch over two kinds falling through to the first -- but the
/// choice is a property of the geometry, not of the caller, and the geometry
/// is already an argument here. `mb` is where the paged KV scatter lives; a
/// paged geometry without it is a load-time error rather than a silent run of
/// the ring kernel against page-table binds.
/// `rows` and `head_rows` must mirror what `launch_shape` was given EXACTLY.
/// A grid computed for one tiling against a pipeline compiled for another is
/// not a crash -- it is wrong numbers.
Pso pso_for(const Dispatch& d, const LlamaGeometry& g, const DecodeStepPsos& base,
            const LlamaPsos& ll, const MultiBatchPsos* mb = nullptr, int rows = 1,
            int head_rows = 1, int requests = 1);

/// Its grid and threadgroup, over `rows` tokens.
///
/// One function for M=1 and M>1, where gpt-oss and gemma4 each carry a second
/// `launch_shape_mb`. The shared `*_mb_dispatch` helpers ARE the M=1 shapes
/// with a row count folded in and agree with them exactly at `rows == 1`, so a
/// second switch over the same thirty kinds would buy nothing and would be a
/// second place for a shape to drift.
///
/// `head_rows` is how many rows the TAIL runs on -- what `RowGather` compacted
/// for the sampler. It is not `rows`: a prefill samples a few of its tokens and
/// the head is the most expensive dispatch in the step.
void launch_shape(const Dispatch& d, const LlamaGeometry& g, Grid& grid, Threadgroup& tg,
                  int rows = 1, int head_rows = 1, int requests = 1);

/// A dense projection's GEMM row count, padded up to a whole BM tile so the
/// matmul engages at any batch rather than only at exact multiples of one. The
/// padding rows compute discardable values into pool rows nothing reads.
int llama_qmm_rows(int rows, int requests = 1);

/// The pool's row count: the widest padding any batch up to `max_rows` can ask
/// for, which is what the scratch must be sized against.
int llama_qmm_pool_rows(int max_rows);

/// The (token, slot) pairs a batch of `rows` routes: `rows * experts_per_token`.
int llama_moe_pairs(const LlamaGeometry& g, int rows);

/// The rows the sort pads each expert's run to: 1 leaves the routed
/// projections matvecs, `shared_kernels::kMoeTileRows` makes them matmuls.
///
/// One question in one place. The sort kernel, the three projections' launch
/// shapes, their pipeline choice and the pool sizer all have to give the same
/// answer -- a sort that padded to 16 under a matvec launched for the unpadded
/// count would run the projection over a fraction of its input, and the model
/// would still produce text.
int llama_moe_tile_rows(const LlamaGeometry& g, int rows);

/// The routed matmul's column tile, or 0 when the batch stays a matvec.
int llama_moe_qmm_bn(Kind k, const LlamaGeometry& g, int rows);

/// Whether a kind is a dense projection, and so a GEMM candidate. NOT the
/// routed three -- each row picks its own experts, so a tile spanning rows
/// would span weights.
bool llama_is_dense_proj(Kind k);

/// The GEMM's column tile for a dense projection, or 0 to keep the matvec.
int llama_qmm_bn(Kind k, const LlamaGeometry& g, int rows, int requests = 1);

/// The K partitions a dense projection's GEMM takes, or 1 for no split.
///
/// A decode fleet is the shape that needs this: 32 rows fill exactly one row
/// block, so the whole GEMM is `out/BN` threadgroups -- one per core on an
/// M1 Max -- and the machine idles. Splitting K hands each partition its own
/// threadgroup and a reduce sums them. MLX sends every transposed non-batched
/// decode down this path for the same reason.
///
/// One question in one place: `pso_for`, `launch_shape`, `encode_llama_step`
/// and `bind_llama_splitk` must all give the same answer, or the GEMM writes
/// partials with a stride the reduce does not read back.
int llama_qmm_split(Kind k, const LlamaGeometry& g, int rows, int requests = 1);

inline constexpr int kLlamaSplitkConcurrentLanes = 3;

/// Elements the split GEMM's partials buffer must hold for a batch up to
/// `max_rows`: one maximum-sized slice per member of the widest concurrency
/// run. Q/K/V need three independent slices so one projection's reduce can
/// overlap the next projection's GEMM.
std::size_t llama_splitk_partial_elems(const LlamaGeometry& g, int max_rows);

/// `run_ends[i]`: the last position of the concurrency run containing `i`. What
/// the colourer needs to know about which barriers the encoder will drop.
std::vector<int> llama_run_ends(const std::vector<Dispatch>& dag);

/// Walk the DAG with a real encoder.
///
/// `[begin, end)` bounds which dispatches are encoded, so a caller that must
/// run the host between two parts of the step can put each part in its own
/// command buffer. `end` past the DAG means the rest of it. A cut must fall on
/// a concurrency-run boundary -- `llama_run_ends` says where those are -- since
/// a run split across two command buffers would lose the very concurrency it
/// was grouped for.
void encode_llama_step(StepEncoder& se, const std::vector<Dispatch>& dag,
                       const LlamaGeometry& g, const DecodeStepPsos& base, const LlamaPsos& ll,
                       int ordinal_base = 0, const MultiBatchPsos* mb = nullptr, int rows = 1,
                       int head_rows = 1, int requests = 1, std::size_t begin = 0,
                       std::size_t end = ~std::size_t(0), bool run_argmax = true);

}  // namespace pie::metal::llama
