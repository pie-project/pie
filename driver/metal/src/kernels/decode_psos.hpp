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

// Compile every decode kernel from `kernels_dir` into
// `out`, indexed by Kernel kind. `with_argmax` controls whether the optional device-argmax
// PSO is compiled (Argmax is not yet ported; default off → it is left invalid). Returns
// false if any compile fails (the failing kernel name is written to `*err` if non-null).
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
                      bool with_argmax = false,
                      std::string* err = nullptr,
                      bool fuse_residual = false,
                      bool gdn_prep = false,
                      /// Compile the routed mixture's kernels. Only a
                      /// checkpoint whose geometry has experts dispatches them.
                      bool routed = false,
                      /// Claim `EmbedUntied`/`LmHeadUntied` for the shared
                      /// embed/matvec entrypoints. Off by default, and the
                      /// default is load-bearing: the llama family compiles
                      /// its OWN kernels for these kinds at its own group size
                      /// and bit width, and looks here only as a fallback. A
                      /// table that claimed them unconditionally would hand it
                      /// a valid gs_64/b_4 PSO for a checkpoint that is
                      /// neither, and a wrong-but-valid PSO is silent.
                      bool untied = false);

// ── M>1 multi-batch PSOs (beta, multi-batch lane) ─────────────────────────────
// The 4 kernel kinds whose M>1 form differs from the M=1 PSO (the rest just grid-widen
// via decode_dispatch_mb): per-row IO (embed/rope), slot-indexed state (gdn), and the
// page-table attention read. Kept SEPARATE from DecodeStepPsos so the sealed M=1 table is
// byte-untouched; the M>1 encoder selects from here for those kinds + reuses by_kind[] (with
// N-widened grids) for everything else. Activation dtype bf16; sdpa_paged_d512 = gemma4.
struct MultiBatchPsos {
    Pso embed_mb{};        // embed_gather_mb_4bit_bfloat16_gs_64_b_4   (per-row id[m])
    Pso rope_mb{};         // rope_neox_mb_bfloat16                     (per-row position[m])
    Pso gdn_slotted{};     // gdn_core_slotted_bfloat16                 (slot_ids[b_idx])
    Pso gdn_prep_slotted{};      // gdn_prep_slotted_bfloat16
    Pso gdn_recurrent_slotted{}; // gdn_core_recurrent_slotted_bfloat16
    Pso sdpa_paged{};      // sdpa_paged_decode_bfloat16_d_256          (page-table gather)
    Pso sdpa_paged_d512{}; // sdpa_paged_decode_bfloat16_d_512          (gemma4 full-attn)
    Pso kv_append_paged{}; // kv_append_paged_bfloat16                  (page-table scatter write)
    // affine_qmm_t: MLX's steel quantized GEMM, for the batched decode. [0] is
    // BN=32, [1] is BN=64. Selected only above `kQmmMinBatch`.
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
    /// per TILE rather than per dispatch. `bm` is fixed at `kMoeTileRows`,
    /// because that is what the sort padded every expert's run to -- a tile
    /// that spanned two experts would read one expert's weights for the
    /// other's rows. Three column tiles, as elsewhere.
    Pso qmm_routed[3]{};
    Pso qmm_t_splitk[3]{};
    Pso qmm_t_splitk_f32[3]{};
    // FP16-compute counterparts, with bf16 or float partials respectively.
    Pso qmm_t_splitk_fp16_precast[3]{};
    Pso qmm_t_splitk_fp16_precast_f32[3]{};
    Pso qmm_cast_bf16_f16{};
    Pso qmm_splitk_reduce{};
    Pso qmm_splitk_reduce_residual{};
    Pso qmm_splitk_reduce_f32{};
    Pso qmm_splitk_reduce_residual_f32{};
    Pso qmm_t_strided{};
    Pso qmm_t_strided_wide{};
    Pso qmm_t_strided_wide_residual{};
    Pso qmm_t_strided_residual{};
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
    bool valid() const {
        return embed_mb.valid() && rope_mb.valid() && gdn_slotted.valid() &&
               gdn_prep_slotted.valid() &&
               gdn_recurrent_slotted.valid() && sdpa_paged.valid() && kv_append_paged.valid();
    }
};

// Compile the M>1 variants from `kernels_dir`. d512 (gemma4) is optional via `with_d512`.
// Returns false if any required compile fails (failing entrypoint written to *err).
bool load_multibatch_psos(RawMetalContext& ctx,
                          const std::string& kernels_dir,
                          MultiBatchPsos& out,
                          /// The checkpoint's affine width and group; see
                          /// `load_decode_psos` for why it has no default.
                          AffineFormat quant,
                          bool with_d512 = false,
                          std::string* err = nullptr,
                          /// Compile the mixture's batched projections. Only a
                          /// checkpoint whose geometry has experts runs them.
                          bool routed = false,
                          /// Compile dense g64/b4 BF16->FP16 staging QMMs.
                          bool fp16_precast = false);

}  // namespace pie::metal
