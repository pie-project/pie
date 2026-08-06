// decode_psos.cpp — runtime-compile the decode kernels and fan them out by Kernel kind.

#include "decode_psos.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "../model/qwen3_5/decode_dispatch_mb.hpp"

namespace pie::metal {

namespace {

// One distinct PSO to compile: source file + instantiated entrypoint, and the kinds it
// serves. Entrypoints are the bf16 (`bfloat16`) instantiations (activation dtype T = bf16);
// 4-bit kernels are g64/b4, sdpa is head_dim 256.
struct PsoSpec {
    std::string         file;
    std::string         fn;
    std::vector<Kernel> kinds;
};

std::vector<PsoSpec> specs(const std::string& embed_gather_fn, const std::string& qmv_fast_fn) {
    return {
        {"embed_gather.metal", embed_gather_fn, {Kernel::EmbedGather}},
        {"rms_norm.metal",     "rms_single_row_bfloat16",
            {Kernel::Rms, Kernel::FfnRms, Kernel::QNorm, Kernel::KNorm, Kernel::FinalRms}},
        {"quantized_qmv.metal", qmv_fast_fn,
            {Kernel::QmvIn, Kernel::QmvInZ, Kernel::QmvOut, Kernel::QmvQ, Kernel::QmvK,
             Kernel::QmvV, Kernel::QmvO, Kernel::QmvGate, Kernel::QmvUp, Kernel::QmvDown,
             Kernel::QmvLmHead, Kernel::GdnInA, Kernel::GdnInB}},
        {"residual_add.metal", "residual_add_bfloat16", {Kernel::Residual, Kernel::LayerOut}},
        {"rope.metal",         "rope_neox_decode_bfloat16", {Kernel::Rope, Kernel::RopeK}},
        {"kv_append.metal",    "kv_append_bfloat16",    {Kernel::KvAppend}},
        {"silu_mul.metal",     "silu_mul_bfloat16",     {Kernel::SiluMul}},
    };
}

}  // namespace

bool load_decode_psos(RawMetalContext& ctx,
                      const std::string& kernels_dir,
                      DecodeStepPsos& out,
                      AffineFormat quant,
                      std::string* err,
                      DecodePsoFeatures features) {
    const std::string dir = kernels_dir.empty() || kernels_dir.back() == '/'
                                ? kernels_dir : kernels_dir + "/";
    const std::string q = quant.kernel_suffix();
    const std::string embed_gather_fn = "embed_gather_4bit" + q;
    const std::string qmv_fast_fn = "affine_qmv_fast" + q;
    const std::string qmv_residual_fn = "affine_qmv_fast_residual" + q;
    const std::string qmv_routed_fn = "affine_qmv_routed" + q;

    // Every entrypoint this configuration needs, gathered first so the whole
    // set compiles as one concurrent batch (and files shared by several
    // entrypoints -- attn_gate, gdn_prep, quantized_qmv -- are read and turned
    // into a library exactly once).
    std::vector<RawMetalContext::PsoFileRequest> requests;
    std::vector<std::vector<Kernel>> targets;
    auto want = [&](const std::string& file, const std::string& fn, std::vector<Kernel> kinds) {
        requests.push_back({dir + file, fn});
        targets.push_back(std::move(kinds));
    };

    for (const PsoSpec& spec : specs(embed_gather_fn, qmv_fast_fn)) {
        want(spec.file.c_str(), spec.fn.c_str(), spec.kinds);
    }
    if (features.residual_qmv) {
        // Residual-epilogue GEMV variant for QmvO/QmvOut/QmvDown (adds buffer(7) residual).
        want("quantized_qmv.metal", qmv_residual_fn.c_str(), {});
    }
    size_t residual_at =
        features.residual_qmv ? requests.size() - 1 : SIZE_MAX;
    if (features.gdn) {
        // Prep-dispatch split (PIE_GDN_PREP): GdnPrep computes the q/k path once/head;
        // GdnCore is replaced by the slimmed recurrent kernel reading prep scratch.
        // The recurrent kernel deliberately overrides the in-kernel-share gdn_core PSO,
        // so it must be applied after the base specs above.
        want("gdn_prep.metal", "gdn_prep_bfloat16", {Kernel::GdnPrep});
        want("gdn_prep.metal", "gdn_core_recurrent_bfloat16", {Kernel::GdnCore});
        want("gated_rms.metal", "gated_rms_bfloat16", {Kernel::GatedRms});
    }
    if (features.gated_attention) {
        want("attn_gate.metal", "q_gate_split_bfloat16", {Kernel::QSplit});
        want("attn_gate.metal", "attn_gate_bfloat16", {Kernel::AttnGate});
    }
    if (features.sdpa_d256) {
        want("sdpa_vector.metal", "sdpa_vector_decode_bfloat16_d_256",
             {Kernel::Sdpa});
    }
    if (features.untied) {
        // An untied checkpoint's two ends are two tensors and therefore two
        // kinds -- but the same two entrypoints, at the same shapes. Only the
        // weight name differs, which is the whole reason the kinds exist.
        //
        // Behind a flag, and the flag is load-bearing. The llama family
        // compiles its OWN kernels for these kinds, at its checkpoint's group
        // size and bit width, and consults this table only as a fallback.
        // Claiming them unconditionally handed it a valid gs_64/b_4 PSO for a
        // checkpoint that is neither -- not a load failure, just wrong numbers,
        // and it took the numerics test to say so.
        want("embed_gather.metal", embed_gather_fn.c_str(),
             {Kernel::EmbedUntied});
        want("quantized_qmv.metal", qmv_fast_fn.c_str(),
             {Kernel::LmHeadUntied});
    }
    if (features.routed) {
        // A routed checkpoint's mixture, on the same kernels the llama family
        // dispatches -- these are shared `Kernel` values and the weights for
        // them are already keyed by kind. Compiled only when the geometry says
        // the model has experts: a dense checkpoint never dispatches them, and
        // compiling them anyway would let an unrelated shader error fail a load
        // that would otherwise have worked.
        //
        // `SiluMul` is deliberately absent. Routed, it is the SHARED expert's
        // SwiGLU -- the same kernel over a different extent -- so the dense
        // entry above already serves it. The mixture's own SwiGLU, over the
        // sorted stack, is `LlExpertSiluMul` and is served by that same entry
        // for the same reason: what differs is the launch shape, not the code.
        want("silu_mul.metal", "silu_mul_bfloat16", {Kernel::LlExpertSiluMul});
        // The routing logits are an ordinary dense matvec -- one weight, one
        // output row per expert, no routing to speak of yet. It is loaded HERE
        // and not alongside the dense projections above because a kind in that
        // table is a kind every checkpoint is expected to have: a dense model
        // has no `mlp.gate`, and claiming the kind for it makes the loader
        // demand a tensor that does not exist.
        want("quantized_qmv.metal", qmv_fast_fn.c_str(), {Kernel::LlRouter});
        want("moe_route.metal", "router_topk_bfloat16", {Kernel::GoRouterTopK});
        want("moe_route.metal", "moe_route_sort", {Kernel::LlMoeSort});
        want("moe_route.metal", "moe_route_gather", {Kernel::LlMoeGather});
        want("moe_route.metal", "moe_combine_sorted", {Kernel::LlMoeCombine});
        want("quantized_qmv.metal", qmv_routed_fn.c_str(),
             {Kernel::LlExpertGate, Kernel::LlExpertUp, Kernel::LlExpertDown});
        // The shared expert. Dense projections on the dense kernel -- they are
        // separate KINDS only because a kind is a weight name here, and these
        // read `mlp.shared_expert.*`. The gate projection is the same kernel
        // producing a single output row.
        want("quantized_qmv.metal", qmv_fast_fn.c_str(),
             {Kernel::LlSharedGate, Kernel::LlSharedUp, Kernel::LlSharedDown,
              Kernel::LlSharedGateProj});
        want("moe_route.metal", "shared_expert_combine", {Kernel::LlSharedCombine});
    }
    if (features.argmax) {
        // Device argmax + EOS-compare (I3 sampling substrate). bf16 logits = lm_head out.
        want("argmax.metal", "argmax_logits_bfloat16", {Kernel::Argmax});
    }
    if (features.routing_only) {
        requests.clear();
        targets.clear();
        residual_at = SIZE_MAX;
        want("quantized_qmv.metal", qmv_fast_fn.c_str(),
             {Kernel::LlRouter, Kernel::LlSharedGateProj});
    }

    std::vector<std::string> errors;
    const std::vector<Pso> psos = ctx.compile_psos_from_files(requests, &errors);
    for (size_t i = 0; i < psos.size(); ++i) {
        if (!psos[i].valid()) {
            if (err) {
                *err = requests[i].function + " (" + requests[i].path + "): " + errors[i];
            }
            return false;
        }
        for (Kernel k : targets[i]) out[k] = psos[i];
    }
    if (residual_at != SIZE_MAX) out.qmv_residual = psos[residual_at];
    return true;
}

bool load_multibatch_psos(RawMetalContext& ctx,
                          const std::string& kernels_dir,
                          MultiBatchPsos& out,
                          AffineFormat quant,
                          std::string* err,
                          MultiBatchPsoFeatures features) {
    const std::string dir = kernels_dir.empty() || kernels_dir.back() == '/'
                                ? kernels_dir : kernels_dir + "/";
    const std::string q = quant.kernel_suffix();
    const std::string embed_mb_fn = "embed_gather_mb_4bit" + q;
    // Every remaining entrypoint, gathered into one concurrent batch. This
    // matters most for quantized_qmm_t.metal: ~25 entrypoints come out of that
    // one 1800-line source, and compiling them one at a time re-parsed the
    // whole file for each. Batching reads and front-ends it exactly once.
    std::vector<RawMetalContext::PsoFileRequest> requests;
    std::vector<Pso*> dsts;
    auto want = [&](const std::string& file, const std::string& fn, Pso* dst) {
        requests.push_back({dir + file, fn});
        dsts.push_back(dst);
    };

    const std::string qmm = "quantized_qmm_t.metal";
    // The table's first axis IS `kQmmBMs`. Spelling the extent as a literal in
    // the header is unavoidable -- it cannot see the model layer -- so the two
    // are tied here, where both are visible.
    static_assert(pie::metal::kQmmBMCount ==
                      int(sizeof(out.qmm_t) / sizeof(out.qmm_t[0])),
                  "the PSO table's row-block axis must match kQmmBMs");
    for (int w = 0; w < pie::metal::kQmmBMCount; ++w) {
        const int bm = pie::metal::kQmmBMs[w];
        for (int i = 0; i < 3; ++i) {
            const int bn = 16 << i;
            const std::string suffix = q + "_bm_" + std::to_string(bm) +
                                       "_bn_" + std::to_string(bn);
            want(qmm, "affine_qmm_t" + suffix, &out.qmm_t[w][i]);
            if (features.fp16_precast && quant.group == 64 && quant.bits == 4) {
                want(qmm, "affine_qmm_t_fp16_precast" + suffix,
                     &out.qmm_t_fp16_precast[w][i]);
            }
            if (features.residual)
                want(qmm, "affine_qmm_t_residual" + suffix,
                     &out.qmm_t_residual[w][i]);
            if (features.bias)
                want(qmm, "affine_qmm_t_bias" + suffix,
                     &out.qmm_t_bias[w][i]);
        }
        if (features.splitk) {
            want(qmm,
                 "affine_qmm_t_splitk" + q + "_bm_" + std::to_string(bm) +
                     "_bn_" + std::to_string(pie::metal::kQmmSplitBN),
                 &out.qmm_t_splitk[w]);
            want(qmm,
                 "affine_qmm_t_splitk_f32" + q + "_bm_" +
                     std::to_string(bm) + "_bn_" +
                     std::to_string(pie::metal::kQmmSplitBN),
                 &out.qmm_t_splitk_f32[w]);
            if (features.fp16_precast && quant.group == 64 && quant.bits == 4) {
                want(qmm,
                     "affine_qmm_t_splitk_fp16_precast" + q + "_bm_" +
                         std::to_string(bm) + "_bn_" +
                         std::to_string(pie::metal::kQmmSplitBN),
                     &out.qmm_t_splitk_fp16_precast[w]);
                want(qmm,
                     "affine_qmm_t_splitk_fp16_precast_f32" + q + "_bm_" +
                         std::to_string(bm) + "_bn_" +
                         std::to_string(pie::metal::kQmmSplitBN),
                     &out.qmm_t_splitk_fp16_precast_f32[w]);
            }
        }
    }
    if (features.fp16_precast && quant.group == 64 && quant.bits == 4) {
        want(qmm, "cast_qmm_input_bfloat16_to_float16", &out.qmm_cast_bf16_f16);
    }
    if (features.routed) {
        // `bm` is spelled from the shared widths rather than restated: it is the
        // number the sort padded every expert's run to, and a tile that
        // disagreed with the padding would read one expert's weights for
        // another's rows.
        for (int t = 0; t < 3; ++t) {
            for (int i = 0; i < 3; ++i) {
                want(qmm,
                     "affine_qmm_t_routed" + q + "_bm_" +
                         std::to_string(shared_kernels::kMoeTileWidths[t]) + "_bn_" +
                         std::to_string(16 << i),
                     &out.qmm_routed[t][i]);
            }
        }
    }
    if (features.splitk) {
        want(qmm, "qmm_splitk_reduce_bfloat16", &out.qmm_splitk_reduce);
        want(qmm, "qmm_splitk_reduce_f32_bfloat16",
             &out.qmm_splitk_reduce_f32);
    }
    if (features.strided) {
        want(qmm, "affine_qmm_t_strided" + q + "_bm_16_bn_32",
             &out.qmm_t_strided);
        want(qmm, "affine_qmm_t_strided" + q + "_bm_32_bn_32",
             &out.qmm_t_strided_wide);
        if (features.residual) {
            want(qmm, "affine_qmm_t_strided_residual" + q + "_bm_16_bn_32",
                 &out.qmm_t_strided_residual);
            want(qmm, "affine_qmm_t_strided_residual" + q + "_bm_32_bn_32",
                 &out.qmm_t_strided_wide_residual);
        }
    }
    if (features.fp16_strided && quant.group == 64 && quant.bits == 4) {
        want(qmm, "affine_qmm_t_strided_fp16_precast" + q + "_bm_16_bn_32",
             &out.qmm_t_strided_fp16_precast);
        want(qmm, "affine_qmm_t_strided_fp16_precast" + q + "_bm_32_bn_32",
             &out.qmm_t_strided_fp16_precast_wide);
        want(qmm, "affine_qmm_t_strided_fp16_precast_residual" + q +
                     "_bm_16_bn_32",
             &out.qmm_t_strided_fp16_precast_residual);
        want(qmm, "affine_qmm_t_strided_fp16_precast_residual" + q +
                     "_bm_32_bn_32",
             &out.qmm_t_strided_fp16_precast_wide_residual);
        want(qmm, "cast_qmm_input_strided_bfloat16_to_float16",
             &out.qmm_t_strided_cast);
        want(qmm, "affine_qmv_wide_strided_bfloat16_gs_64_b_4_v_4_kl_8",
             &out.qmv_wide_strided);
    }
    want("embed_gather.metal", embed_mb_fn, &out.embed_mb);
    want("rope.metal", "rope_neox_mb_bfloat16", &out.rope_mb);
    want("kv_append_paged.metal", "kv_append_paged_bfloat16",
         &out.kv_append_paged);
    if (features.sdpa_d256)
        want("sdpa_paged.metal", "sdpa_paged_decode_bfloat16_d_256",
             &out.sdpa_paged);
    if (features.d512)
        want("sdpa_paged.metal", "sdpa_paged_decode_bfloat16_d_512",
             &out.sdpa_paged_d512);
    if (features.sdpa_d256)
        want("sdpa_paged.metal", "sdpa_paged_tiled_bfloat16_d_256",
             &out.sdpa_paged_tiled);
    if (features.d512)
        want("sdpa_paged.metal", "sdpa_paged_tiled_bfloat16_d_512",
             &out.sdpa_paged_tiled_d512);
    if (features.gdn) {
        want("gdn_prep.metal", "gdn_prep_slotted_bfloat16",
             &out.gdn_prep_slotted);
        want("gdn_prep.metal", "gdn_core_recurrent_slotted_bfloat16",
             &out.gdn_recurrent_slotted);
        want("gdn_prep.metal", "gdn_prep_prefill_bfloat16",
             &out.gdn_prep_prefill);
        want("gdn_prep.metal", "gdn_core_recurrent_prefill_bfloat16",
             &out.gdn_core_prefill);
        want("gated_rms.metal", "gated_rms_strided_bfloat16",
             &out.gated_rms_strided);
    }
    if (features.strided) {
        want("rms_norm.metal", "rms_strided_row_bfloat16",
             &out.rms_strided);
        want("silu_mul.metal", "silu_mul_strided_bfloat16",
             &out.silu_mul_strided);
    }

    std::vector<std::string> errors;
    const std::vector<Pso> psos = ctx.compile_psos_from_files(requests, &errors);
    for (size_t i = 0; i < psos.size(); ++i) {
        if (!psos[i].valid()) {
            if (err) {
                *err = requests[i].function + " (" + requests[i].path + "): " + errors[i];
            }
            return false;
        }
        *dsts[i] = psos[i];
    }
    return true;
}

}  // namespace pie::metal
