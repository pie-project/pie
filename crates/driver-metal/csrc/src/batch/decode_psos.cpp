// decode_psos.cpp — runtime-compile the decode kernels and fan them out by Kernel kind.

#include "decode_psos.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "../device_tuning.hpp"
#include "pie/kernels/entrypoint.h"
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
        {"layout/embed_gather.metal", embed_gather_fn, {Kernel::EmbedGather}},
        {"norm/rms.metal",     "rms_single_row_bfloat16",
            {Kernel::Rms, Kernel::FfnRms, Kernel::QNorm, Kernel::KNorm, Kernel::FinalRms}},
        {"quant/qmv.metal", qmv_fast_fn,
            {Kernel::QmvIn, Kernel::QmvInZ, Kernel::QmvOut, Kernel::QmvQ, Kernel::QmvK,
             Kernel::QmvV, Kernel::QmvO, Kernel::QmvGate, Kernel::QmvUp, Kernel::QmvDown,
             Kernel::QmvLmHead, Kernel::GdnInA, Kernel::GdnInB}},
        {"norm/residual_add.metal", "residual_add_bfloat16", {Kernel::Residual, Kernel::LayerOut}},
        {"rope/neox.metal",         "neox_decode_bfloat16", {Kernel::Rope, Kernel::RopeK}},
        {"attn/kv_write.metal",    "kv_append_bfloat16",    {Kernel::KvAppend}},
        {"mlp/gated.metal",     "silu_mul_bfloat16",     {Kernel::SiluMul}},
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
    // Every name below is built by `pie::kernels::entrypoint`, which refuses
    // one no shader instantiates. That is not defensive: `affine_qmv_routed`
    // is compiled for ONE of the six affine formats, so a mixture at any other
    // used to reach the Metal compiler as a string and fail there. It now
    // fails here, saying which formats exist.
    using pie::kernels::affine;
    using pie::kernels::entrypoint;
    const std::string embed_gather_fn = entrypoint("embed_gather_4bit", {affine(quant)});
    const std::string qmv_fast_fn = entrypoint("affine_qmv_fast", {affine(quant)});
    const std::string qmv_residual_fn =
        entrypoint("affine_qmv_fast_residual", {affine(quant)});
    const std::string qmv_routed_fn = entrypoint("affine_qmv_routed", {affine(quant)});

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
        want("quant/qmv.metal", qmv_residual_fn.c_str(), {});
    }
    size_t residual_at =
        features.residual_qmv ? requests.size() - 1 : SIZE_MAX;
    if (features.gdn) {
        // Prep-dispatch split (PIE_GDN_PREP): GdnPrep computes the q/k path once/head;
        // GdnCore is replaced by the slimmed recurrent kernel reading prep scratch.
        // The recurrent kernel deliberately overrides the in-kernel-share gdn_core PSO,
        // so it must be applied after the base specs above.
        want("ssm/gdn_prep.metal", "gdn_prep_bfloat16", {Kernel::GdnPrep});
        want("ssm/gdn_prep.metal", "gdn_core_recurrent_bfloat16", {Kernel::GdnCore});
        want("norm/gated_rms.metal", "gated_rms_bfloat16", {Kernel::GatedRms});
    }
    if (features.gated_attention) {
        want("attn/gate.metal", "q_gate_split_bfloat16", {Kernel::QSplit});
        want("attn/gate.metal", "gate_bfloat16", {Kernel::AttnGate});
    }
    if (features.sdpa_d256) {
        want("attn/sdpa_vector.metal", "sdpa_vector_decode_bfloat16_d_256",
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
        want("layout/embed_gather.metal", embed_gather_fn.c_str(),
             {Kernel::EmbedUntied});
        want("quant/qmv.metal", qmv_fast_fn.c_str(),
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
        want("mlp/gated.metal", "silu_mul_bfloat16", {Kernel::LlExpertSiluMul});
        // The routing logits are an ordinary dense matvec -- one weight, one
        // output row per expert, no routing to speak of yet. It is loaded HERE
        // and not alongside the dense projections above because a kind in that
        // table is a kind every checkpoint is expected to have: a dense model
        // has no `mlp.gate`, and claiming the kind for it makes the loader
        // demand a tensor that does not exist.
        want("quant/qmv.metal", qmv_fast_fn.c_str(), {Kernel::LlRouter});
        want("moe/route.metal", "router_topk_bfloat16", {Kernel::GoRouterTopK});
        want("moe/route.metal", "route_sort", {Kernel::LlMoeSort});
        want("moe/route.metal", "route_gather", {Kernel::LlMoeGather});
        want("moe/route.metal", "combine_sorted", {Kernel::LlMoeCombine});
        want("quant/qmv.metal", qmv_routed_fn.c_str(),
             {Kernel::LlExpertGate, Kernel::LlExpertUp, Kernel::LlExpertDown});
        // The shared expert. Dense projections on the dense kernel -- they are
        // separate KINDS only because a kind is a weight name here, and these
        // read `mlp.shared_expert.*`. The gate projection is the same kernel
        // producing a single output row.
        want("quant/qmv.metal", qmv_fast_fn.c_str(),
             {Kernel::LlSharedGate, Kernel::LlSharedUp, Kernel::LlSharedDown,
              Kernel::LlSharedGateProj});
        want("moe/route.metal", "shared_expert_combine", {Kernel::LlSharedCombine});
    }
    if (features.argmax) {
        // Device argmax + EOS-compare (I3 sampling substrate). bf16 logits = lm_head out.
        want("sample/argmax.metal", "argmax_logits_bfloat16", {Kernel::Argmax});
    }
    if (features.routing_only) {
        requests.clear();
        targets.clear();
        residual_at = SIZE_MAX;
        want("quant/qmv.metal", qmv_fast_fn.c_str(),
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
    using pie::kernels::affine;
    using pie::kernels::bf16;
    using pie::kernels::chunk;
    using pie::kernels::entrypoint;
    using pie::kernels::k_unroll;
    using pie::kernels::rows;
    using pie::kernels::tile;
    const std::string embed_mb_fn = entrypoint("embed_gather_mb_4bit", {affine(quant)});
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

    const std::string qmm = "quant/qmm_t.metal";
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
            const auto at = [&](const char* base) {
                return entrypoint(base, {affine(quant), tile(bm, bn)});
            };
            want(qmm, at("affine_qmm_t"), &out.qmm_t[w][i]);
            if (features.fp16_precast && quant.group == 64 && quant.bits == 4) {
                want(qmm, at("affine_qmm_t_fp16_precast"),
                     &out.qmm_t_fp16_precast[w][i]);
            }
            if (features.residual)
                want(qmm, at("affine_qmm_t_residual"),
                     &out.qmm_t_residual[w][i]);
            if (features.residual && features.fp16_precast &&
                quant.group == 64 && quant.bits == 4) {
                want(qmm, at("affine_qmm_t_residual_fp16_precast"),
                     &out.qmm_t_residual_fp16_precast[w][i]);
            }
            if (features.bias)
                want(qmm, at("affine_qmm_t_bias"),
                     &out.qmm_t_bias[w][i]);
            if (features.bias && features.fp16_precast && quant.group == 64 &&
                quant.bits == 4) {
                want(qmm, at("affine_qmm_t_bias_fp16_precast"),
                     &out.qmm_t_bias_fp16_precast[w][i]);
            }
        }
        if (features.splitk) {
            want(qmm,
                 entrypoint("affine_qmm_t_splitk",
                            {affine(quant), tile(bm, pie::metal::kQmmSplitBN)}),
                 &out.qmm_t_splitk[w]);
            want(qmm,
                 entrypoint("affine_qmm_t_splitk_f32",
                            {affine(quant), tile(bm, pie::metal::kQmmSplitBN)}),
                 &out.qmm_t_splitk_f32[w]);
            if (features.fp16_precast && quant.group == 64 && quant.bits == 4) {
                want(qmm,
                     entrypoint("affine_qmm_t_splitk_fp16_precast",
                                {affine(quant), tile(bm, pie::metal::kQmmSplitBN)}),
                     &out.qmm_t_splitk_fp16_precast[w]);
                want(qmm,
                     entrypoint("affine_qmm_t_splitk_fp16_precast_f32",
                                {affine(quant), tile(bm, pie::metal::kQmmSplitBN)}),
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
        //
        // FP16 is a NAME choice: both forms take the same buffers, grid and
        // `tile_expert` contract, so nothing downstream of here can tell. The
        // one family that must NOT take it is llama, whose routed top-k moved
        // under FP16 in `llama_numerics_test` -- and llama builds its own
        // routed table in `model/llama/kernels.cpp`, so it never reaches this.
        const bool fp16 = fp16_qmm() && quant.bits == 4 && quant.group == 64;
        const std::string routed =
            fp16 ? "affine_qmm_t_routed_fp16" : "affine_qmm_t_routed";
        for (int t = 0; t < 3; ++t) {
            for (int i = 0; i < 3; ++i) {
                want(qmm,
                     entrypoint(routed,
                                {affine(quant),
                                 tile(shared_kernels::kMoeTileWidths[t], 16 << i)}),
                     &out.qmm_routed[t][i]);
            }
        }
    }
    if (features.splitk) {
        want(qmm, "qmm_splitk_reduce_bfloat16", &out.qmm_splitk_reduce);
        want(qmm, "qmm_splitk_reduce_f32_bfloat16",
             &out.qmm_splitk_reduce_f32);
    }
    // The three row-block rungs, in `kQmmBMs` order. A rung the checkpoint's
    // quantization has no instantiation for stays invalid and the encoder falls
    // back to the widest one that loaded.
    // The tile suffixes, as `entrypoint` takes them. Spelled through `tile()`
    // rather than as literals so that the three rungs and the GEMM above build
    // their names by the same grammar.
    const std::string kStridedBm[3] = {tile(16, 32), tile(32, 32), tile(64, 32)};
    if (features.strided) {
        for (int r = 0; r < 3; ++r) {
            want(qmm, entrypoint("affine_qmm_t_strided", {affine(quant), kStridedBm[r]}),
                 &out.qmm_t_strided[r]);
            if (features.residual) {
                want(qmm, entrypoint("affine_qmm_t_strided_residual", {affine(quant), kStridedBm[r]}),
                     &out.qmm_t_strided_residual[r]);
            }
        }
    }
    if (features.fp16_strided && quant.group == 64 && quant.bits == 4) {
        for (int r = 0; r < 3; ++r) {
            want(qmm, entrypoint("affine_qmm_t_strided_fp16_precast", {affine(quant), kStridedBm[r]}),
                 &out.qmm_t_strided_fp16_precast[r]);
            want(qmm, entrypoint("affine_qmm_t_strided_fp16_precast_residual", {affine(quant), kStridedBm[r]}),
                 &out.qmm_t_strided_fp16_precast_residual[r]);
        }
        want(qmm, "cast_qmm_input_strided_bfloat16_to_float16",
             &out.qmm_t_strided_cast);
    }
    // The wide matvec, asked for the CHECKPOINT's own format rather than only
    // for the 4-bit one the fp16 block above happens to also want. It is the
    // batched primitive for a projection the strided GEMM declines -- one
    // dispatch for the whole prompt instead of one per token -- and gating it
    // on `fp16_strided` tied it to a feature it has nothing to do with, which
    // left an alt-quant kind with no batched shape at all.
    if (features.strided && quant.group == 64 && (quant.bits == 4 || quant.bits == 8)) {
        want(qmm,
             entrypoint("affine_qmv_wide_strided",
                        {affine(quant), rows(4), k_unroll(8)}),
             &out.qmv_wide_strided);
    }
    want("layout/embed_gather.metal", embed_mb_fn, &out.embed_mb);
    want("rope/neox.metal", "neox_mb_bfloat16", &out.rope_mb);
    want("attn/kv_write.metal", "kv_append_paged_bfloat16",
         &out.kv_append_paged);
    if (features.sdpa_d256)
        want("attn/sdpa_paged.metal", "sdpa_paged_decode_bfloat16_d_256",
             &out.sdpa_paged);
    if (features.d512)
        want("attn/sdpa_paged.metal", "sdpa_paged_decode_bfloat16_d_512",
             &out.sdpa_paged_d512);
    if (features.sdpa_d256)
        want("attn/sdpa_paged.metal", "sdpa_paged_tiled_bfloat16_d_256",
             &out.sdpa_paged_tiled);
    if (features.d512)
        want("attn/sdpa_paged.metal", "sdpa_paged_tiled_bfloat16_d_512",
             &out.sdpa_paged_tiled_d512);
    if (features.gdn) {
        want("ssm/gdn_prep.metal", "gdn_prep_slotted_bfloat16",
             &out.gdn_prep_slotted);
        want("ssm/gdn_prep.metal", "gdn_core_recurrent_slotted_bfloat16",
             &out.gdn_recurrent_slotted);
        want("ssm/gdn_prep.metal", "gdn_prep_prefill_bfloat16",
             &out.gdn_prep_prefill);
        want("ssm/gdn_prep.metal",
             entrypoint("gdn_core_recurrent_prefill",
                        {bf16(), chunk(gdn_scan_lanes()), rows(gdn_scan_rows())}),
             &out.gdn_core_prefill);
        want("norm/gated_rms.metal", "gated_rms_strided_bfloat16",
             &out.gated_rms_strided);
    }
    if (features.strided) {
        want("norm/rms.metal", "rms_strided_row_bfloat16",
             &out.rms_strided);
        want("norm/rms.metal", "rms_strided_head_row_bfloat16",
             &out.rms_strided_head);
        want("rope/neox.metal", "neox_strided_bfloat16", &out.rope_strided);
        want("mlp/gated.metal", "silu_mul_strided_bfloat16",
             &out.silu_mul_strided);
        want("norm/residual_add.metal", "residual_add_strided_bfloat16",
             &out.residual_add_strided);
        want("moe/route.metal", "shared_expert_combine_strided",
             &out.shared_expert_combine_strided);
        if (features.sdpa_d256) {
            want("attn/sdpa_paged.metal", "sdpa_paged_tiled_strided_bfloat16_d_256",
                 &out.sdpa_paged_tiled_strided);
        }
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
