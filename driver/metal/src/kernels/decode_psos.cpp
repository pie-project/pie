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
    const char*         file;
    const char*         fn;
    std::vector<Kernel> kinds;
};

const std::vector<PsoSpec>& specs() {
    static const std::vector<PsoSpec> s = {
        {"embed_gather.metal", "embed_gather_4bit_bfloat16_gs_64_b_4", {Kernel::EmbedGather}},
        {"rms_norm.metal",     "rms_single_row_bfloat16",
            {Kernel::Rms, Kernel::FfnRms, Kernel::QNorm, Kernel::KNorm, Kernel::FinalRms}},
        {"quantized_qmv.metal", "affine_qmv_fast_bfloat16_gs_64_b_4",
            {Kernel::QmvIn, Kernel::QmvInZ, Kernel::QmvOut, Kernel::QmvQ, Kernel::QmvK,
             Kernel::QmvV, Kernel::QmvO, Kernel::QmvGate, Kernel::QmvUp, Kernel::QmvDown,
             Kernel::QmvLmHead}},
        {"dense_gemv.metal",   "dense_gemv_coop_bfloat16",   {Kernel::GdnInA, Kernel::GdnInB}},
        {"gdn_core.metal",     "gdn_core_bfloat16",     {Kernel::GdnCore}},
        {"gated_rms.metal",    "gated_rms_bfloat16",    {Kernel::GatedRms}},
        {"residual_add.metal", "residual_add_bfloat16", {Kernel::Residual, Kernel::LayerOut}},
        {"attn_gate.metal",    "q_gate_split_bfloat16", {Kernel::QSplit}},
        {"attn_gate.metal",    "attn_gate_bfloat16",    {Kernel::AttnGate}},
        {"rope.metal",         "rope_neox_decode_bfloat16", {Kernel::Rope, Kernel::RopeK}},
        {"kv_append.metal",    "kv_append_bfloat16",    {Kernel::KvAppend}},
        {"sdpa_vector.metal",  "sdpa_vector_decode_bfloat16_d_256", {Kernel::Sdpa}},
        {"silu_mul.metal",     "silu_mul_bfloat16",     {Kernel::SiluMul}},
    };
    return s;
}

}  // namespace

bool load_decode_psos(RawMetalContext& ctx,
                      const std::string& kernels_dir,
                      DecodeStepPsos& out,
                      bool with_argmax,
                      std::string* err,
                      bool fuse_residual,
                      bool gdn_prep,
                      bool routed,
                      bool untied) {
    const std::string dir = kernels_dir.empty() || kernels_dir.back() == '/'
                                ? kernels_dir : kernels_dir + "/";

    // Every entrypoint this configuration needs, gathered first so the whole
    // set compiles as one concurrent batch (and files shared by several
    // entrypoints -- attn_gate, gdn_prep, quantized_qmv -- are read and turned
    // into a library exactly once).
    std::vector<RawMetalContext::PsoFileRequest> requests;
    std::vector<std::vector<Kernel>> targets;
    auto want = [&](const char* file, const char* fn, std::vector<Kernel> kinds) {
        requests.push_back({dir + file, fn});
        targets.push_back(std::move(kinds));
    };

    for (const PsoSpec& spec : specs()) want(spec.file, spec.fn, spec.kinds);
    if (fuse_residual) {
        // Residual-epilogue GEMV variant for QmvO/QmvOut/QmvDown (adds buffer(7) residual).
        want("quantized_qmv.metal", "affine_qmv_fast_residual_bfloat16_gs_64_b_4", {});
    }
    const size_t residual_at = fuse_residual ? requests.size() - 1 : SIZE_MAX;
    if (gdn_prep) {
        // Prep-dispatch split (PIE_GDN_PREP): GdnPrep computes the q/k path once/head;
        // GdnCore is replaced by the slimmed recurrent kernel reading prep scratch.
        // The recurrent kernel deliberately overrides the in-kernel-share gdn_core PSO,
        // so it must be applied after the base specs above.
        want("gdn_prep.metal", "gdn_prep_bfloat16", {Kernel::GdnPrep});
        want("gdn_prep.metal", "gdn_core_recurrent_bfloat16", {Kernel::GdnCore});
    }
    if (untied) {
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
        want("embed_gather.metal", "embed_gather_4bit_bfloat16_gs_64_b_4",
             {Kernel::EmbedUntied});
        want("quantized_qmv.metal", "affine_qmv_fast_bfloat16_gs_64_b_4",
             {Kernel::LmHeadUntied});
    }
    if (routed) {
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
        want("quantized_qmv.metal", "affine_qmv_fast_bfloat16_gs_64_b_4", {Kernel::LlRouter});
        want("gptoss.metal", "router_topk_bfloat16", {Kernel::GoRouterTopK});
        want("moe_route.metal", "moe_route_sort", {Kernel::LlMoeSort});
        want("moe_route.metal", "moe_route_gather", {Kernel::LlMoeGather});
        want("moe_route.metal", "moe_combine_sorted", {Kernel::LlMoeCombine});
        want("quantized_qmv.metal", "affine_qmv_routed_bfloat16_gs_64_b_4",
             {Kernel::LlExpertGate, Kernel::LlExpertUp, Kernel::LlExpertDown});
        // The shared expert. Dense projections on the dense kernel -- they are
        // separate KINDS only because a kind is a weight name here, and these
        // read `mlp.shared_expert.*`. The gate projection is the same kernel
        // producing a single output row.
        want("quantized_qmv.metal", "affine_qmv_fast_bfloat16_gs_64_b_4",
             {Kernel::LlSharedGate, Kernel::LlSharedUp, Kernel::LlSharedDown,
              Kernel::LlSharedGateProj});
        want("moe_route.metal", "shared_expert_combine", {Kernel::LlSharedCombine});
    }
    if (with_argmax) {
        // Device argmax + EOS-compare (I3 sampling substrate). bf16 logits = lm_head out.
        want("argmax.metal", "argmax_logits_bfloat16", {Kernel::Argmax});
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
                          bool with_d512,
                          std::string* err,
                          bool routed) {
    const std::string dir = kernels_dir.empty() || kernels_dir.back() == '/'
                                ? kernels_dir : kernels_dir + "/";
    struct MbSpec { const char* file; const char* fn; Pso* dst; bool required; };
    const MbSpec specs[] = {
        {"embed_gather.metal", "embed_gather_mb_4bit_bfloat16_gs_64_b_4", &out.embed_mb,        true},
        {"rope.metal",         "rope_neox_mb_bfloat16",                   &out.rope_mb,         true},
        {"gdn_core.metal",     "gdn_core_slotted_bfloat16",               &out.gdn_slotted,     true},
        {"gdn_prep.metal",     "gdn_prep_slotted_bfloat16",               &out.gdn_prep_slotted, true},
        {"gdn_prep.metal",     "gdn_core_recurrent_slotted_bfloat16",     &out.gdn_recurrent_slotted, true},
        {"sdpa_paged.metal",   "sdpa_paged_decode_bfloat16_d_256",        &out.sdpa_paged,      true},
        {"sdpa_paged.metal",   "sdpa_paged_decode_bfloat16_d_512",        &out.sdpa_paged_d512, false},
        {"kv_append_paged.metal", "kv_append_paged_bfloat16",             &out.kv_append_paged, true},
        {"rms_norm.metal",     "rms_strided_row_bfloat16",   &out.rms_strided,       true},
        {"silu_mul.metal",     "silu_mul_strided_bfloat16",  &out.silu_mul_strided,  true},
        {"gated_rms.metal",    "gated_rms_strided_bfloat16", &out.gated_rms_strided, true},
        {"dense_gemv.metal",   "dense_gemv_coop_strided_bfloat16",
                                                             &out.dense_gemv_strided, true},
        {"gdn_prep.metal",     "gdn_prep_prefill_bfloat16",  &out.gdn_prep_prefill,  true},
        {"gdn_prep.metal",     "gdn_core_recurrent_prefill_bfloat16",
                                                             &out.gdn_core_prefill,  true},
    };
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
    for (int w = 0; w < 2; ++w) {
        for (int i = 0; i < 3; ++i) {
            const int bn = 16 << i;
            const int bm = w == 0 ? pie::metal::kQmmBM : pie::metal::kQmmBMWide;
            const std::string suffix = "_bfloat16_gs_64_b_4_bm_" + std::to_string(bm) +
                                       "_bn_" + std::to_string(bn);
            want(qmm, "affine_qmm_t" + suffix, &out.qmm_t[w][i]);
            want(qmm, "affine_qmm_t_residual" + suffix, &out.qmm_t_residual[w][i]);
            want(qmm, "affine_qmm_t_bias" + suffix, &out.qmm_t_bias[w][i]);
        }
    }
    for (int w = 0; w < 2; ++w) {
        const int bm = w == 0 ? pie::metal::kQmmBM : pie::metal::kQmmBMWide;
        want(qmm,
             "affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_" + std::to_string(bm) +
                 "_bn_" + std::to_string(pie::metal::kQmmSplitBN),
             &out.qmm_t_splitk[w]);
    }
    if (routed) {
        // `bm` is spelled from `kMoeTileRows` rather than restated: it is the
        // number the sort padded every expert's run to, and a tile that
        // disagreed with the padding would read one expert's weights for
        // another's rows.
        for (int i = 0; i < 3; ++i) {
            want(qmm,
                 "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_" +
                     std::to_string(shared_kernels::kMoeTileRows) + "_bn_" +
                     std::to_string(16 << i),
                 &out.qmm_routed[i]);
        }
    }
    want(qmm, "qmm_splitk_reduce_bfloat16", &out.qmm_splitk_reduce);
    want(qmm, "qmm_splitk_reduce_residual_bfloat16", &out.qmm_splitk_reduce_residual);
    want(qmm, "affine_qmm_t_strided_bfloat16_gs_64_b_4_bm_16_bn_32", &out.qmm_t_strided);
    want(qmm, "affine_qmm_t_strided_residual_bfloat16_gs_64_b_4_bm_16_bn_32",
         &out.qmm_t_strided_residual);
    want(qmm, "affine_qmm_t_strided_bfloat16_gs_64_b_4_bm_32_bn_32", &out.qmm_t_strided_wide);
    want(qmm, "affine_qmm_t_strided_residual_bfloat16_gs_64_b_4_bm_32_bn_32",
         &out.qmm_t_strided_wide_residual);
    for (const MbSpec& s : specs) {
        if (!s.required && !with_d512) continue;
        want(s.file, s.fn, s.dst);
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
