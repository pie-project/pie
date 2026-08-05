#include "kernels.hpp"

#include <cmath>
#include <string>

namespace pie::metal::gptoss {

bool build_gptoss_psos(RawMetalContext& ctx, const std::string& kernels_dir,
                       const GptOssGeometry& g, GptOssPsos& out, std::string* err) {
    // The attention width is the geometry's, not a literal: see the header.
    const std::string d = "_d_" + std::to_string(g.head_dim);
    // The expert bank's codec chooses its matvec. Same kernel body, same routed
    // offsets, same bias epilogue -- the codec is a template parameter, so the
    // only thing that differs here is which instantiation is named.
    const std::string routed_name = g.mxfp4_experts
                                        ? "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4"
                                        : "affine_qmv_routed_bias_bfloat16_gs_64_b_4";
    // The router's width is the checkpoint's to state. It is the same dense
    // biased matvec at either width, so an unsolved one fails by NAME here --
    // there is no instantiation to compile -- rather than silently defaulting
    // to the common one and reading the bytes at half the stride.
    const std::string router_name = "affine_qmv_tail_bias_bfloat16_gs_64_b_" +
                                    std::to_string(g.router_bits);
    const std::string sink_name = "sdpa_vector_decode_sink_bfloat16" + d;
    const std::string sink_paged_name = "sdpa_paged_decode_sink_bfloat16" + d;
    const std::string dir =
        kernels_dir.empty() || kernels_dir.back() == '/' ? kernels_dir : kernels_dir + "/";
    struct Spec {
        const char* file;
        const char* fn;
        Pso* dst;
    };
    // bf16 throughout: the activation dtype every ported M=1 kernel already uses.
    const Spec specs[] = {
        {"quantized_qmv.metal", "affine_qmv_tail_bfloat16_gs_64_b_4", &out.qmv_tail},
        {"quantized_qmv.metal", "affine_qmv_tail_bias_bfloat16_gs_64_b_4", &out.qmv_tail_bias},
        {"quantized_qmv.metal", routed_name.c_str(), &out.qmv_routed_bias},
        {"quantized_qmv.metal", router_name.c_str(), &out.qmv_router},
        {"moe_route.metal", "router_topk_bfloat16", &out.router_topk},
        {"moe_route.metal", "moe_route_sort", &out.moe_sort},
        {"moe_route.metal", "moe_route_gather", &out.moe_gather},
        {"moe_route.metal", "moe_combine_sorted", &out.moe_combine},
        {"gptoss.metal", "gptoss_swiglu_bfloat16", &out.swiglu},
        {"sdpa_sliding.metal", sink_name.c_str(), &out.sdpa_sink},
        {"sdpa_paged.metal", sink_paged_name.c_str(), &out.sdpa_sink_paged},
        {"rope.metal", "rope_neox_freqs_mb_bfloat16", &out.rope_freqs_mb},
        {"row_gather.metal", "row_gather_bfloat16", &out.row_gather},
        {"rope.metal", "rope_neox_freqs_decode_bfloat16", &out.rope_freqs},
    };
    for (const Spec& spec : specs) {
        std::string compile_error;
        *spec.dst = ctx.compile_pso_from_file(dir + spec.file, spec.fn, &compile_error);
        if (!spec.dst->valid()) {
            if (err != nullptr) {
                *err = std::string("gpt-oss PSO '") + spec.fn + "' (" + spec.file +
                       "): " + compile_error;
            }
            return false;
        }
    }
    if (g.mxfp4_experts) {
        for (int i = 0; i < 3; ++i) {
            const std::string fn =
                "mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_" +
                std::to_string(16 << i);
            std::string compile_error;
            out.qmm_routed_bias[i] = ctx.compile_pso_from_file(
                dir + "quantized_qmm_t.metal", fn, &compile_error);
            if (!out.qmm_routed_bias[i].valid()) {
                if (err != nullptr)
                    *err = "gpt-oss PSO '" + fn + "' (quantized_qmm_t.metal): " +
                           compile_error;
                return false;
            }
        }
    }
    return true;
}

/// YaRN, ported from `mlx_lm/models/rope_utils.py::YarnRoPE`.
///
/// Two frequencies exist for every dimension: the one the original context
/// implies, and the one an extended context implies (`theta * factor`). YaRN
/// interpolates between them by a ramp over the dimensions where the ORIGINAL
/// window held between `beta_slow` and `beta_fast` full rotations -- low
/// dimensions rotate fast enough that extrapolation is safe, high ones do not.
///
/// This is arithmetic over `head_dim/2` values that does not depend on the
/// position, so it happens once here rather than in every head every token.
std::vector<float> yarn_inv_freq(const GptOssGeometry& g) {
    const int half = g.head_dim / 2;
    std::vector<float> inv_freq(std::size_t(half < 1 ? 1 : half), 0.0f);
    if (half < 1) return inv_freq;

    const float base = g.rope_theta;
    const float factor = g.rope_factor > 0.0f ? g.rope_factor : 1.0f;
    const float dim = float(g.head_dim);
    const float orig = float(g.rope_original_max_position);

    // The dimension at which `rotations` full turns fit in the original window.
    const auto find_dim = [&](float rotations) {
        return dim * std::log(orig / (rotations * 2.0f * float(M_PI))) /
               (2.0f * std::log(base));
    };
    float low = std::floor(find_dim(g.rope_beta_fast));
    float high = std::ceil(find_dim(g.rope_beta_slow));
    low = std::max(low, 0.0f);
    high = std::min(high, dim - 1.0f);
    // mlx nudges the singular case rather than dividing by zero; matched so the
    // ramp is the same function on both sides.
    if (low == high) high += 0.001f;

    for (int i = 0; i < half; ++i) {
        const float d = float(2 * i);
        const float freq_extra = 1.0f / std::pow(base, d / dim);
        const float freq_inter = freq_extra / factor;
        // 1 where the dimension is safe to extrapolate, 0 where it must be
        // interpolated, ramped between.
        float ramp = high - low <= 0.0f ? 0.0f : (float(i) - low) / (high - low);
        ramp = ramp < 0.0f ? 0.0f : (ramp > 1.0f ? 1.0f : ramp);
        const float mask = 1.0f - ramp;
        inv_freq[std::size_t(i)] = freq_inter * (1.0f - mask) + freq_extra * mask;
    }
    return inv_freq;
}

/// YaRN's `mscale`, which scales q and k.
///
/// `0.1 * log(factor) + 1` -- 1.3466 at gpt-oss's factor of 32. mlx computes it
/// as a ratio of two `yarn_get_mscale` calls whose denominator is 1 for this
/// config; stated directly here, with the ratio's shape kept so a checkpoint
/// that did carry `mscale_all_dim` would be visibly unhandled rather than
/// silently wrong.
float yarn_mscale(const GptOssGeometry& g) {
    if (g.rope_factor <= 1.0f) return 1.0f;
    return 0.1f * std::log(g.rope_factor) + 1.0f;
}

}  // namespace pie::metal::gptoss
