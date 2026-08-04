#include "kernels.hpp"

#include <string>

namespace pie::metal::gemma4 {

bool build_gemma4_psos(RawMetalContext& ctx, const std::string& kernels_dir,
                       const Gemma4Geometry& g, Gemma4Psos& out, std::string* err) {
    // Both widths are the geometry's, not literals: see the header.
    const std::string swa_name =
        "sdpa_vector_decode_swa_bfloat16_d_" + std::to_string(g.head_dim);
    const std::string swa_global_name =
        "sdpa_vector_decode_swa_bfloat16_d_" + std::to_string(g.global_head_dim);
    const std::string dir =
        kernels_dir.empty() || kernels_dir.back() == '/' ? kernels_dir : kernels_dir + "/";
    const std::string q = g.quant.kernel_suffix();
    struct Spec {
        const char* file;
        std::string fn;
        Pso* dst;
    };
    // bf16 throughout: the activation dtype every ported M=1 kernel already uses.
    const Spec specs[] = {
        {"sdpa_sliding.metal", swa_name, &out.sdpa_swa_d256},
        {"sdpa_sliding.metal", swa_global_name, &out.sdpa_swa_d512},
        {"geglu_tanh.metal", "geglu_tanh_bfloat16", &out.geglu_tanh},
        {"logit_softcap.metal", "logit_softcap_bfloat16", &out.logit_softcap},
        {"layer_scalar.metal", "layer_scalar_mul_bfloat16", &out.layer_scalar},
        {"ple_combine.metal", "ple_combine_bfloat16", &out.ple_combine},
        {"vnorm.metal", "vnorm_single_row_bfloat16", &out.vnorm},
        {"embed_gather.metal", "embed_gather_scaled_4bit" + q, &out.embed_scaled},
        {"quantized_qmv.metal", "affine_qmv_narrow" + q, &out.qmv_narrow},
        {"rope.metal", "rope_neox_prop_decode_bfloat16", &out.rope_prop},
        {"embed_gather.metal", "embed_gather_scaled_mb_4bit" + q, &out.embed_scaled_mb},
        {"rope.metal", "rope_neox_prop_mb_bfloat16", &out.rope_prop_mb},
        {"rms_norm.metal", "rms_residual_bfloat16", &out.rms_residual},
        {"rms_norm.metal", "rms_residual_scaled_bfloat16", &out.rms_residual_scaled},
        {"geglu_tanh.metal", "geglu_tanh_strided_bfloat16", &out.geglu_strided},
        {"row_gather.metal", "row_gather_bfloat16", &out.row_gather},
    };
    for (const Spec& spec : specs) {
        std::string compile_error;
        *spec.dst = ctx.compile_pso_from_file(dir + spec.file, spec.fn.c_str(), &compile_error);
        if (!spec.dst->valid()) {
            if (err != nullptr) {
                *err = "gemma4 PSO '" + spec.fn + "' (" + spec.file +
                       "): " + compile_error;
            }
            return false;
        }
    }
    return true;
}

}  // namespace pie::metal::gemma4
