#include "kernels.hpp"

#include <string>
#include <vector>

#include "pie/kernels/entrypoint.h"

#include "../../device_tuning.hpp"

namespace pie::metal::gemma4 {

bool build_gemma4_psos(RawMetalContext& ctx, const std::string& kernels_dir,
                       const Gemma4Geometry& g, Gemma4Psos& out, std::string* err) {
    using namespace ::pie::kernels;
    // Both widths are the geometry's, not literals: see the header.
    const std::string swa_name =
        entrypoint("sdpa_vector_decode_swa", {bf16(), head_dim(g.head_dim)});
    const std::string swa_global_name =
        entrypoint("sdpa_vector_decode_swa", {bf16(), head_dim(g.global_head_dim)});
    const std::string dir =
        kernels_dir.empty() || kernels_dir.back() == '/' ? kernels_dir : kernels_dir + "/";
    struct Spec {
        const char* file;
        std::string fn;
        Pso* dst;
    };
    // bf16 throughout: the activation dtype every ported M=1 kernel already uses.
    const Spec specs[] = {
        {"attn/sdpa_sliding.metal", swa_name, &out.sdpa_swa_d256},
        {"attn/sdpa_sliding.metal", swa_global_name, &out.sdpa_swa_d512},
        {"mlp/gated.metal", "geglu_tanh_bfloat16", &out.geglu_tanh},
        {"attn/logit_softcap.metal", "logit_softcap_bfloat16", &out.logit_softcap},
        {"norm/layer_scalar.metal", "layer_scalar_mul_bfloat16", &out.layer_scalar},
        {"layout/ple_combine.metal", "ple_combine_bfloat16", &out.ple_combine},
        {"norm/vector.metal", "vnorm_single_row_bfloat16", &out.vnorm},
        {"layout/embed_gather.metal", entrypoint("embed_gather_scaled_4bit", {affine(g.quant)}), &out.embed_scaled},
        {"quant/qmv.metal", entrypoint("affine_qmv_tail", {affine(g.quant)}), &out.qmv_tail},
        {"rope/neox.metal", "neox_prop_decode_bfloat16", &out.rope_prop},
        {"layout/embed_gather.metal", entrypoint("embed_gather_scaled_mb_4bit", {affine(g.quant)}), &out.embed_scaled_mb},
        {"rope/neox.metal", "neox_prop_mb_bfloat16", &out.rope_prop_mb},
        {"norm/rms.metal", "rms_residual_bfloat16", &out.rms_residual},
        {"norm/rms.metal", "rms_residual_scaled_bfloat16", &out.rms_residual_scaled},
        {"mlp/gated.metal", "geglu_tanh_strided_bfloat16", &out.geglu_strided},
        {"layout/row_gather.metal", "row_gather_bfloat16", &out.row_gather},
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
        extra.push_back({"quant/qmv.metal",
                         entrypoint("affine_qmv_tail", {affine(g.ffn_quant)}),
                         &out.qmv_tail_alt});
    }
    if (g.is_moe()) {
        extra.push_back({"moe/route.metal", "router_topk_scaled_bfloat16", &out.router_topk});
        extra.push_back({"quant/qmv.metal", entrypoint("affine_qmv_routed", {affine(g.quant)}), &out.qmv_routed});
        extra.push_back({"moe/route.metal", "route_sort", &out.moe_sort});
        extra.push_back({"moe/route.metal", "route_gather", &out.moe_gather});
        extra.push_back({"moe/route.metal", "combine_sorted", &out.moe_combine});
        extra.push_back({"norm/residual_add.metal", "residual_add_bfloat16", &out.residual_add});
        for (int t = 0; t < 3; ++t) {
            // The routed GEMM is a NAME choice and nothing else: the FP16 form
            // takes the same buffers, the same grid and the same `tile_expert`
            // contract, so the dispatch path never learns which one it got.
            // It is the largest term in a 26B prefill -- 47.9% of it -- and on
            // a device with no bfloat matrix unit the BF16 form is emulated.
            const std::string routed =
                fp16_qmm() && g.quant.bits == 4 && g.quant.group == 64
                    ? "affine_qmm_t_routed_fp16"
                    : "affine_qmm_t_routed";
            for (int i = 0; i < 3; ++i) {
                extra.push_back({"quant/qmm_t.metal",
                                 entrypoint(routed,
                                            {affine(g.quant),
                                             tile(shared_kernels::kMoeTileWidths[t],
                                                  16 << i)}),
                                 &out.qmm_routed[t][i]});
            }
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
