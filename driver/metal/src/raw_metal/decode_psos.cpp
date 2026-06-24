// decode_psos.cpp — runtime-compile the decode kernels and fan them out by Kernel kind.

#include "decode_psos.hpp"

namespace pie_metal_driver::raw_metal {

namespace {

// One distinct PSO to compile: source file + instantiated entrypoint, and the kinds it
// serves. Entrypoints are the bf16 (`bfloat16`) instantiations (activation dtype T = bf16);
// 4-bit kernels are g64/b4, sdpa is head_dim 256.
struct PsoSpec {
    const char*         file;
    const char*         fn;
    std::vector<Kernel> kinds;
};

const std::vector<PsoSpec>& specs() {
    static const std::vector<PsoSpec> s = {
        {"embed_gather.metal", "embed_gather_4bit_bfloat16_gs_64_b_4", {Kernel::EmbedGather}},
        {"rms_norm.metal",     "rms_single_row_bfloat16",
            {Kernel::Rms, Kernel::FfnRms, Kernel::QNorm, Kernel::KNorm, Kernel::FinalRms}},
        {"quantized_qmv.metal", "affine_qmv_fast_bfloat16_gs_64_b_4",
            {Kernel::QmvIn, Kernel::QmvInZ, Kernel::QmvOut, Kernel::QmvQ, Kernel::QmvK,
             Kernel::QmvV, Kernel::QmvO, Kernel::QmvGate, Kernel::QmvUp, Kernel::QmvDown,
             Kernel::QmvLmHead}},
        {"dense_gemv.metal",   "dense_gemv_coop_bfloat16",   {Kernel::GdnInA, Kernel::GdnInB}},
        {"gdn_core.metal",     "gdn_core_bfloat16",     {Kernel::GdnCore}},
        {"gated_rms.metal",    "gated_rms_bfloat16",    {Kernel::GatedRms}},
        {"residual_add.metal", "residual_add_bfloat16", {Kernel::Residual, Kernel::LayerOut}},
        {"attn_gate.metal",    "q_gate_split_bfloat16", {Kernel::QSplit}},
        {"attn_gate.metal",    "attn_gate_bfloat16",    {Kernel::AttnGate}},
        {"rope.metal",         "rope_neox_decode_bfloat16", {Kernel::Rope, Kernel::RopeK}},
        {"kv_append.metal",    "kv_append_bfloat16",    {Kernel::KvAppend}},
        {"sdpa_vector.metal",  "sdpa_vector_decode_bfloat16_d_256", {Kernel::Sdpa}},
        {"silu_mul.metal",     "silu_mul_bfloat16",     {Kernel::SiluMul}},
    };
    return s;
}

}  // namespace

bool load_decode_psos(RawMetalContext& ctx,
                      const std::string& kernels_dir,
                      DecodeStepPsos& out,
                      bool with_argmax,
                      std::string* err,
                      bool fuse_residual) {
    const std::string dir = kernels_dir.empty() || kernels_dir.back() == '/'
                                ? kernels_dir : kernels_dir + "/";
    for (const PsoSpec& spec : specs()) {
        Pso pso = ctx.compile_pso_from_file(dir + spec.file, spec.fn);
        if (!pso.valid()) {
            if (err) *err = std::string(spec.fn) + " (" + spec.file + ")";
            return false;
        }
        for (Kernel k : spec.kinds) out[k] = pso;
    }
    if (fuse_residual) {
        // Residual-epilogue GEMV variant for QmvO/QmvOut/QmvDown (adds buffer(7) residual).
        out.qmv_residual = ctx.compile_pso_from_file(
            dir + "quantized_qmv.metal", "affine_qmv_fast_residual_bfloat16_gs_64_b_4");
        if (!out.qmv_residual.valid()) {
            if (err) *err = "affine_qmv_fast_residual_bfloat16_gs_64_b_4 (quantized_qmv.metal)";
            return false;
        }
    }
    if (with_argmax) {
        // Argmax is an optional I3 substrate; its kernel is not yet ported. When it lands,
        // compile it here and assign out[Kernel::Argmax].
    }
    return true;
}

}  // namespace pie_metal_driver::raw_metal
