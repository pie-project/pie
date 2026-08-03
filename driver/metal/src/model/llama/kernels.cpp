#include "kernels.hpp"

#include <string>
#include <vector>

namespace pie::metal::llama {

bool build_llama_psos(RawMetalContext& ctx, const std::string& kernels_dir,
                      const LlamaGeometry& g, LlamaPsos& out, std::string* err) {
    const std::string dir =
        kernels_dir.empty() || kernels_dir.back() == '/' ? kernels_dir : kernels_dir + "/";
    struct Spec {
        const char* file;
        const char* fn;
        Pso* dst;
    };
    // bf16 throughout: the activation dtype every ported M=1 kernel already uses.
    std::vector<Spec> specs = {
        {"sdpa_vector.metal", "sdpa_vector_decode_bfloat16_d_128", &out.sdpa_d128},
        {"sdpa_paged.metal", "sdpa_paged_decode_bfloat16_d_128", &out.sdpa_paged_d128},
        {"row_gather.metal", "row_gather_bfloat16", &out.row_gather},
    };
    // Only for a routed checkpoint. A dense one never dispatches these, and
    // compiling them anyway would let an unrelated shader error fail a load
    // that would otherwise have worked.
    if (g.is_moe()) {
        specs.push_back({"gptoss.metal", "router_topk_bfloat16", &out.router_topk});
        specs.push_back({"gptoss.metal", "expert_combine_bfloat16", &out.expert_combine});
        specs.push_back({"quantized_qmv.metal", "affine_qmv_routed_bfloat16_gs_64_b_4",
                         &out.qmv_routed});
        specs.push_back({"moe_route.metal", "moe_route_sort", &out.moe_sort});
        specs.push_back({"moe_route.metal", "moe_route_gather", &out.moe_gather});
        specs.push_back({"moe_route.metal", "moe_route_scatter", &out.moe_scatter});
        // The batched form's three column tiles. `bm` is `kMoeTileRows`, which
        // is what the sort padded every expert's run to -- naming it here would
        // be a second statement of the same number, so it is spelled from the
        // constant.
        static const std::string routed[3] = {
            "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_" +
                std::to_string(shared_kernels::kMoeTileRows) + "_bn_16",
            "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_" +
                std::to_string(shared_kernels::kMoeTileRows) + "_bn_32",
            "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_" +
                std::to_string(shared_kernels::kMoeTileRows) + "_bn_64",
        };
        for (int i = 0; i < 3; ++i) {
            specs.push_back({"quantized_qmm_t.metal", routed[i].c_str(), &out.qmm_routed[i]});
        }
    }
    for (const Spec& spec : specs) {
        std::string compile_error;
        *spec.dst = ctx.compile_pso_from_file(dir + spec.file, spec.fn, &compile_error);
        if (!spec.dst->valid()) {
            if (err != nullptr) {
                *err = std::string("llama PSO '") + spec.fn + "' (" + spec.file +
                       "): " + compile_error;
            }
            return false;
        }
    }
    return true;
}

}  // namespace pie::metal::llama
