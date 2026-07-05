#pragma once
// decode_psos.hpp — compile the decode DAG's kernels into a DecodeStepPsos table (beta).
//
// Each Kernel kind maps to one runtime-compiled PSO from raw_metal/kernels/*.metal. Many
// kinds share a PSO (all Qmv* -> affine_qmv_fast; Rms/FfnRms/QNorm/KNorm/FinalRms ->
// rms_single_row; Rope/RopeK -> rope_neox_decode; Residual/LayerOut -> residual_add), so we
// compile each distinct (file, entrypoint) ONCE and fan it out to every kind that uses it.
//
// Activation dtype T = bf16 (delta's M=1 ports: core_out/activation dtype = bfloat); the
// recurrent/conv GDN state is fp32 inside the kernel. lm_head + embed are 4-bit g64.

#include <string>

#include "decode_step.hpp"
#include "mtl4_context.hpp"

namespace pie_metal_driver::raw_metal {

// Compile every decode kernel from `kernels_dir` (the raw_metal/kernels directory) into
// `out`, indexed by Kernel kind. `with_argmax` controls whether the optional device-argmax
// PSO is compiled (Argmax is not yet ported; default off → it is left invalid). Returns
// false if any compile fails (the failing kernel name is written to `*err` if non-null).
bool load_decode_psos(RawMetalContext& ctx,
                      const std::string& kernels_dir,
                      DecodeStepPsos& out,
                      bool with_argmax = false,
                      std::string* err = nullptr,
                      bool fuse_residual = false,
                      bool gdn_prep = false);

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
    Pso sdpa_paged{};      // sdpa_paged_decode_bfloat16_d_256          (page-table gather)
    Pso sdpa_paged_d512{}; // sdpa_paged_decode_bfloat16_d_512          (gemma4 full-attn)
    bool valid() const {
        return embed_mb.valid() && rope_mb.valid() && gdn_slotted.valid() && sdpa_paged.valid();
    }
};

// Compile the M>1 variants from `kernels_dir`. d512 (gemma4) is optional via `with_d512`.
// Returns false if any required compile fails (failing entrypoint written to *err).
bool load_multibatch_psos(RawMetalContext& ctx,
                          const std::string& kernels_dir,
                          MultiBatchPsos& out,
                          bool with_d512 = false,
                          std::string* err = nullptr);

}  // namespace pie_metal_driver::raw_metal
