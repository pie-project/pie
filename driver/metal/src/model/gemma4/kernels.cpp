#include "kernels.hpp"

#include <string>

namespace pie::metal::gemma4 {

bool build_gemma4_psos(RawMetalContext& ctx, const std::string& kernels_dir, Gemma4Psos& out,
                       std::string* err) {
    const std::string dir =
        kernels_dir.empty() || kernels_dir.back() == '/' ? kernels_dir : kernels_dir + "/";
    struct Spec {
        const char* file;
        const char* fn;
        Pso* dst;
    };
    // bf16 throughout: the activation dtype every ported M=1 kernel already uses.
    const Spec specs[] = {
        {"sdpa_sliding.metal", "sdpa_vector_decode_swa_bfloat16_d_256", &out.sdpa_swa_d256},
        {"sdpa_sliding.metal", "sdpa_vector_decode_swa_bfloat16_d_512", &out.sdpa_swa_d512},
        {"geglu_tanh.metal", "geglu_tanh_bfloat16", &out.geglu_tanh},
        {"logit_softcap.metal", "logit_softcap_bfloat16", &out.logit_softcap},
        {"layer_scalar.metal", "layer_scalar_mul_bfloat16", &out.layer_scalar},
        {"ple_combine.metal", "ple_combine_bfloat16", &out.ple_combine},
        {"vnorm.metal", "vnorm_single_row_bfloat16", &out.vnorm},
        {"embed_gather.metal", "embed_gather_scaled_4bit_bfloat16_gs_64_b_4", &out.embed_scaled},
        {"quantized_qmv.metal", "affine_qmv_narrow_bfloat16_gs_64_b_4", &out.qmv_narrow},
        {"rope.metal", "rope_neox_prop_decode_bfloat16", &out.rope_prop},
        {"embed_gather.metal", "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4",
         &out.embed_scaled_mb},
        {"rope.metal", "rope_neox_prop_mb_bfloat16", &out.rope_prop_mb},
        {"rms_norm.metal", "rms_residual_bfloat16", &out.rms_residual},
        {"rms_norm.metal", "rms_residual_scaled_bfloat16", &out.rms_residual_scaled},
    };
    for (const Spec& spec : specs) {
        std::string compile_error;
        *spec.dst = ctx.compile_pso_from_file(dir + spec.file, spec.fn, &compile_error);
        if (!spec.dst->valid()) {
            if (err != nullptr) {
                *err = std::string("gemma4 PSO '") + spec.fn + "' (" + spec.file +
                       "): " + compile_error;
            }
            return false;
        }
    }
    return true;
}

}  // namespace pie::metal::gemma4
