#include "kernels.hpp"

#include <cmath>
#include <string>
#include <vector>

#include "pie/kernels/entrypoint.h"

namespace pie::metal::llama {

bool build_llama_psos(RawMetalContext& ctx, const std::string& kernels_dir,
                      const LlamaGeometry& g, LlamaPsos& out, std::string* err) {
    const std::string dir =
        kernels_dir.empty() || kernels_dir.back() == '/' ? kernels_dir : kernels_dir + "/";
    struct Spec {
        const char* file;
        std::string fn;
        Pso* dst;
    };
    // bf16 throughout: the activation dtype every ported M=1 kernel already uses.
    // The head width is the geometry's, not a literal: see `LlamaPsos::sdpa`.
    // A width with no instantiation fails here, by name, instead of running a
    // pipeline built for a different one.
    using namespace ::pie::kernels;
    const std::string sdpa_name =
        entrypoint("sdpa_vector_decode", {bf16(), head_dim(g.head_dim)});
    const std::string paged_name =
        entrypoint("sdpa_paged_decode",
                   {bf16(), head_dim(g.head_dim), page32(g.kv_page_size == 32)});
    const std::string tiled_name =
        entrypoint("sdpa_paged_tiled", {bf16(), head_dim(g.head_dim)});
    std::vector<Spec> specs = {
        {"attn/sdpa_vector.metal", sdpa_name, &out.sdpa},
        {"attn/sdpa_paged.metal", paged_name, &out.sdpa_paged},
        {"attn/sdpa_paged.metal", tiled_name, &out.sdpa_paged_tiled},
        {"layout/row_gather.metal", "row_gather_bfloat16", &out.row_gather},
    };
    if (g.head_dim == 64 && g.kv_page_size == 32) {
        specs.push_back(
            {"attn/sdpa_paged.metal", paged_name + "_sg8", &out.sdpa_paged_sg8});
    }
    if (g.rope_freq_table) {
        specs.push_back({"rope/neox.metal", "neox_freqs_decode_bfloat16", &out.rope_freqs});
        specs.push_back({"rope/neox.metal", "neox_freqs_mb_bfloat16", &out.rope_freqs_mb});
    }
    // Only for a routed checkpoint. A dense one never dispatches these, and
    // compiling them anyway would let an unrelated shader error fail a load
    // that would otherwise have worked.
    if (g.is_moe()) {
        specs.push_back({"moe/route.metal", "router_topk_bfloat16", &out.router_topk});
        specs.push_back({"quant/qmv.metal",
                         entrypoint("affine_qmv_routed", {affine(g.quant)}),
                         &out.qmv_routed});
        specs.push_back({"moe/route.metal", "route_sort", &out.moe_sort});
        specs.push_back({"moe/route.metal", "route_gather", &out.moe_gather});
        specs.push_back({"moe/route.metal", "combine_sorted", &out.moe_combine});
        // The batched form's three column tiles, at each of the two tile
        // widths `moe_tile_rows` can pick. `bm` is what the sort padded every
        // expert's run to -- naming it here would be a second statement of the
        // same number, so it is spelled from the shared table.
        for (int t = 0; t < 3; ++t) {
            for (int i = 0; i < 3; ++i) {
                specs.push_back({"quant/qmm_t.metal",
                                 entrypoint("affine_qmm_t_routed",
                                            {affine(g.quant),
                                             tile(shared_kernels::kMoeTileWidths[t],
                                                  16 << i)}),
                                 &out.qmm_routed[t][i]});
            }
        }
    }
    for (const Spec& spec : specs) {
        std::string compile_error;
        *spec.dst = ctx.compile_pso_from_file(dir + spec.file, spec.fn.c_str(), &compile_error);
        if (!spec.dst->valid()) {
            if (err != nullptr) {
                *err = "llama PSO '" + spec.fn + "' (" + spec.file +
                       "): " + compile_error;
            }
            return false;
        }
    }
    return true;
}

std::vector<float> llama3_inv_freq(const LlamaGeometry& g) {
    const int dims = g.rotary_dims();
    const int half = dims / 2;
    std::vector<float> inv_freq(std::size_t(half < 1 ? 1 : half), 0.0f);
    if (half < 1) return inv_freq;

    const float base = g.rope_theta;
    const float factor = g.rope_scaling_factor > 0.0f ? g.rope_scaling_factor : 1.0f;
    const float lo = g.rope_low_freq_factor;
    const float hi = g.rope_high_freq_factor;
    const float orig = float(g.rope_original_max_position);
    const float low_wavelen = orig / lo;
    const float high_wavelen = orig / hi;

    for (int i = 0; i < half; ++i) {
        // mlx's `_freqs` is base^(2i/dims) -- a WAVELENGTH-like quantity, the
        // reciprocal of the usual inv_freq. The schedule is expressed on it,
        // so it is computed on it and inverted once at the end.
        const float freq = std::pow(base, float(2 * i) / float(dims));
        const float wavelen = 2.0f * float(M_PI) * freq;
        float scaled = freq;
        if (wavelen > low_wavelen) {
            // Turns too slowly to extrapolate: interpolate by the whole factor.
            scaled = freq * factor;
        } else if (wavelen > high_wavelen) {
            // The ramp. Below `high_wavelen` the dimension is left alone, which
            // is the untouched `scaled = freq` this branch falls past.
            const float smooth = (orig / wavelen - lo) / (hi - lo);
            scaled = freq / ((1.0f - smooth) / factor + smooth);
        }
        inv_freq[std::size_t(i)] = scaled != 0.0f ? 1.0f / scaled : 0.0f;
    }
    return inv_freq;
}

}  // namespace pie::metal::llama
