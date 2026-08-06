#pragma once
// decode_psos.hpp — compile the decode DAG's kernels into a DecodeStepPsos table (beta).
//
// Each Kernel kind maps to one runtime-compiled PSO from src/kernels/*.metal. Many
// kinds share a PSO (all Qmv* -> affine_qmv_fast; Rms/FfnRms/QNorm/KNorm/FinalRms ->
// rms_single_row; Rope/RopeK -> rope_neox_decode; Residual/LayerOut -> residual_add), so we
// compile each distinct (file, entrypoint) ONCE and fan it out to every kind that uses it.
//
// Activation dtype T = bf16 (delta's M=1 ports: core_out/activation dtype = bfloat); the
// recurrent/conv GDN state is fp32 inside the kernel. lm_head + embed are 4-bit g64.

#include <string>

#include "affine_format.hpp"
#include "decode_step.hpp"
#include "mtl4_context.hpp"

namespace pie::metal {

struct DecodePsoFeatures {
    bool argmax = false;
    bool residual_qmv = false;
    bool gdn = false;
    bool gated_attention = false;
    bool sdpa_d256 = false;
    bool routed = false;
    bool untied = false;
    /// Build ONLY the two routing projections, and nothing else.
    ///
    /// For a checkpoint whose `mlp.gate` and `mlp.shared_expert_gate` are in a
    /// second affine format: those two need their own pipelines and nothing
    /// else in the model does, so asking for the whole routed set instead would
    /// demand `affine_qmv_routed` at a width no expert bank is in -- which is
    /// not instantiated, and rightly.
    bool routing_only = false;
};

// Compile the shared decode kernels plus the model's requested extensions.
// Named features keep family setup explicit without positional boolean calls.
bool load_decode_psos(RawMetalContext& ctx,
                      const std::string& kernels_dir,
                      DecodeStepPsos& out,
                      /// The checkpoint's affine width and group. REQUIRED, and
                      /// deliberately ahead of the optional flags: when these
                      /// were two defaulted trailing ints, call sites passed
                      /// the width and let the group fall back to 64. That
                      /// compiles, binds, dispatches, and answers wrongly. One
                      /// value with no default cannot be half-supplied.
                      AffineFormat quant,
                      std::string* err = nullptr,
                      DecodePsoFeatures features = {});

struct MultiBatchPsoFeatures {
    bool d512 = false;
    bool sdpa_d256 = false;
    bool gdn = false;
    bool residual = false;
    bool bias = false;
    bool routed = false;
    bool splitk = false;
    bool strided = false;
    bool fp16_precast = false;
    bool fp16_strided = false;
};

// ── M>1 multi-batch PSOs (beta, multi-batch lane) ─────────────────────────────
// M>1-only IO, attention, recurrent, and quantized-matmul variants. Kept
// separate from DecodeStepPsos so the M=1 table stays byte-untouched.
struct MultiBatchPsos {
    Pso embed_mb{};        // embed_gather_mb_4bit_bfloat16_gs_64_b_4   (per-row id[m])
    Pso rope_mb{};         // rope_neox_mb_bfloat16                     (per-row position[m])
    Pso gdn_prep_slotted{};      // gdn_prep_slotted_bfloat16
    Pso gdn_recurrent_slotted{}; // gdn_core_recurrent_slotted_bfloat16
    Pso sdpa_paged{};      // sdpa_paged_decode_bfloat16_d_256          (page-table gather)
    Pso sdpa_paged_d512{}; // sdpa_paged_decode_bfloat16_d_512          (gemma4 full-attn)
    // Same attention, one simdgroup per row instead of one threadgroup, with
    // the keys staged once per 32-row tile. Earned by row count and request
    // count together -- see `sdpa_should_tile`.
    Pso sdpa_paged_tiled{};      // sdpa_paged_tiled_bfloat16_d_256
    Pso sdpa_paged_tiled_d512{}; // sdpa_paged_tiled_bfloat16_d_512
    Pso kv_append_paged{}; // kv_append_paged_bfloat16                  (page-table scatter write)
    // affine_qmm_t: MLX's steel quantized GEMM, for the batched decode. [0] is
    // BN=32, [1] is BN=64. Selected only above `qmm_min_batch()`.
    // [bm][bn]: `kQmmBMs` rows per block x 16/32/64 columns.
    Pso qmm_t[3][3]{};
    // Same storage ABI, but casts each tile to FP16 before the simdgroup MMA.
    // Instantiated only for g64/b4; dense llama uses it on M1 where FP16 is
    // native and BF16 MMA is markedly slower.
    Pso qmm_t_fp16_precast[3][3]{};
    Pso qmm_t_residual[3][3]{};
    /// The same GEMM with a Linear's additive bias broadcast down the tile.
    /// gpt-oss biases every projection, so without it the batched path is a
    /// GEMM plus a dispatch that rewrites the whole output to add one vector.
    Pso qmm_t_bias[3][3]{};
    // affine_qmm_t_strided: the same GEMM with an explicit row pitch, for the
    // prefill, whose scratch rows are laid at a uniform `scratch_widest_elems`
    // rather than packed at `K`.
    // Split-K: [bm_wide] x {gemm, reduce}.  MLX sends every transposed
    // non-batched decode down this path; the split is chosen per shape.
    /// The mixture's batched form: the same GEMM with the weight stack indexed
    /// per TILE rather than per dispatch. `bm` is whichever width the sort
    /// padded every expert's run to -- a tile that spanned two experts would
    /// read one expert's weights for the other's rows -- so both widths are
    /// compiled and the fire picks. Three column tiles at each, as elsewhere.
    Pso qmm_routed[3][3]{};  // [tile width][bn]
    Pso qmm_t_splitk[3]{};
    Pso qmm_t_splitk_f32[3]{};
    // FP16-compute counterparts, with bf16 or float partials respectively.
    Pso qmm_t_splitk_fp16_precast[3]{};
    Pso qmm_t_splitk_fp16_precast_f32[3]{};
    Pso qmm_cast_bf16_f16{};
    Pso qmm_splitk_reduce{};
    Pso qmm_splitk_reduce_f32{};
    Pso qmm_t_strided{};
    Pso qmm_t_strided_wide{};
    Pso qmm_t_strided_wide_residual{};
    Pso qmm_t_strided_residual{};
    Pso qmm_t_strided_fp16_precast{};
    Pso qmm_t_strided_fp16_precast_wide{};
    Pso qmm_t_strided_fp16_precast_residual{};
    Pso qmm_t_strided_fp16_precast_wide_residual{};
    Pso qmm_t_strided_cast{};
    Pso qmv_wide_strided{};
    // Row-independent prefill kernels with an explicit row pitch, so a whole
    // prompt runs as one dispatch instead of one per token.  Same arithmetic as
    // the M=1 kernels beside them -- only the row's base address is computed
    // from the prefill layout's uniform pitch.
    Pso rms_strided{};
    Pso silu_mul_strided{};
    Pso gated_rms_strided{};
    // GDN over a whole prompt in one dispatch (prep is token-parallel, the
    // recurrent scan runs in registers) instead of one serialized pair per token.
    Pso gdn_prep_prefill{};
    Pso gdn_core_prefill{};
    // The model-independent M>1 substrate. Feature-specific completeness is
    // enforced by load_multibatch_psos while compiling the requested entries.
    bool valid() const {
        return embed_mb.valid() && rope_mb.valid() && kv_append_paged.valid();
    }
};

// Compile the shared M>1 substrate plus the model's requested extensions.
// Returns false if any requested entrypoint fails to compile.
bool load_multibatch_psos(RawMetalContext& ctx,
                          const std::string& kernels_dir,
                          MultiBatchPsos& out,
                          /// The checkpoint's affine width and group; see
                          /// `load_decode_psos` for why it has no default.
                          AffineFormat quant,
                          std::string* err = nullptr,
                          MultiBatchPsoFeatures features = {});

}  // namespace pie::metal
