#include "kernels.hpp"

#include <string>
#include <vector>

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
        {"quantized_qmv.metal", "affine_qmv_tail" + q, &out.qmv_tail},
        {"rope.metal", "rope_neox_prop_decode_bfloat16", &out.rope_prop},
        {"embed_gather.metal", "embed_gather_scaled_mb_4bit" + q, &out.embed_scaled_mb},
        {"rope.metal", "rope_neox_prop_mb_bfloat16", &out.rope_prop_mb},
        {"rms_norm.metal", "rms_residual_bfloat16", &out.rms_residual},
        {"rms_norm.metal", "rms_residual_scaled_bfloat16", &out.rms_residual_scaled},
        {"geglu_tanh.metal", "geglu_tanh_strided_bfloat16", &out.geglu_strided},
        {"row_gather.metal", "row_gather_bfloat16", &out.row_gather},
    };
    // The mixture, built only for a model that has one: compiling these for a
    // dense gemma 4 would let an unrelated shader error fail a load that would
    // otherwise have worked.
    //
    // `q` is the routed bank's format, which on the 26B is the checkpoint-wide
    // 4-bit one -- the dense FFN and the router are the tensors mlx_lm spared
    // at 8, and they are built from the second table in `simple_family.cpp`.
    std::vector<Spec> extra;
    // A second width means a second copy of the one kernel whose SELECTION
    // depends on the width: which K counts as aligned is `32 * (32/bits) * 2`,
    // so the same projection can need the tail at one width and not the other.
    if (g.has_alt_quant()) {
        extra.push_back({"quantized_qmv.metal",
                         "affine_qmv_tail" + g.ffn_quant.kernel_suffix(), &out.qmv_tail_alt});
    }
    if (g.is_moe()) {
        extra.push_back({"moe_route.metal", "router_topk_scaled_bfloat16", &out.router_topk});
        extra.push_back({"quantized_qmv.metal", "affine_qmv_routed" + q, &out.qmv_routed});
        extra.push_back({"moe_route.metal", "moe_route_sort", &out.moe_sort});
        extra.push_back({"moe_route.metal", "moe_route_gather", &out.moe_gather});
        extra.push_back({"moe_route.metal", "moe_combine_sorted", &out.moe_combine});
        extra.push_back({"residual_add.metal", "residual_add_bfloat16", &out.residual_add});
        const std::string routed_bm =
            "affine_qmm_t_routed" + q + "_bm_" + std::to_string(shared_kernels::kMoeTileRows);
        for (int i = 0; i < 3; ++i) {
            extra.push_back({"quantized_qmm_t.metal",
                             routed_bm + "_bn_" + std::to_string(16 << i),
                             &out.qmm_routed[i]});
        }
    }
    for (const Spec& spec : extra) {
        std::string compile_error;
        *spec.dst = ctx.compile_pso_from_file(dir + spec.file, spec.fn.c_str(), &compile_error);
        if (!spec.dst->valid()) {
            if (err != nullptr) {
                *err = "gemma4 PSO '" + spec.fn + "' (" + spec.file + "): " + compile_error;
            }
            return false;
        }
    }
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
