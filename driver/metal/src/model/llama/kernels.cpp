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
