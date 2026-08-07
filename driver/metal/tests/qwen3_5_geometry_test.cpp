// What the Qwen3.5 / Qwen3-Next family will and will not run.
//
// This family's `DecodeGeometry` used to be default-constructed and its
// defaults were one preview checkpoint's dimensions: hidden 1024, 24 layers,
// 16 linear heads of 128, a 3584-wide FFN. Nothing in the path ever compared
// those numbers against the checkpoint being loaded, and nothing could have --
// the loader binds by NAME, and a name carries no dimension to disagree with.
// A different member of the same family therefore loaded, ran, and produced
// tokens at the wrong shape.
//
// So the test is not "the geometry is right for one config". It is that the
// geometry is DERIVED, and that everything not derivable is refused with the
// key that was missing. A driver that guesses here is a driver that is fluent
// and wrong, which is the failure this whole family's schema is arranged to
// avoid elsewhere.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "batch/forward.hpp"
#include "batch/scratch.hpp"
#include "model/qwen3_5/decode_consts.hpp"
#include "model/qwen3_5/decode_step.hpp"
#include "model/qwen3_5/decode_step_mb.hpp"
#include "model/qwen3_5/geometry_facts.hpp"
#include "model/shared_kernels.hpp"
#include "model_facts.hpp"

using pie::metal::DecodeGeometry;
using pie::metal::geometry_from_facts;
using Facts = pie::metal::batch::SetupConfig::Qwen35Facts;

namespace {

int g_pass = 0;
int g_fail = 0;

void expect(bool ok, const std::string& what) {
    if (ok) {
        ++g_pass;
        std::printf("  PASS  %s\n", what.c_str());
    } else {
        ++g_fail;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

/// Qwen3-Next-80B-A3B's shape, which is what this family exists to run.
Facts qwen3_next() {
    Facts f;
    f.n_layers = 48;
    f.hidden = 2048;
    f.vocab = 151936;
    f.n_q_heads = 16;
    f.n_kv_heads = 2;
    f.head_dim = 256;
    f.intermediate = 5120;
    f.gdn_k_heads = 16;
    f.gdn_v_heads = 32;
    f.gdn_k_dim = 128;
    f.gdn_v_dim = 128;
    f.gdn_conv_k = 4;
    f.full_attn_interval = 4;
    f.eps = 1e-6f;
    return f;
}

/// The refusal a config produces, or the empty string when it is accepted.
std::string refusal(const Facts& f) {
    DecodeGeometry g{};
    std::string err;
    if (geometry_from_facts(f, g, &err)) return {};
    return err.empty() ? "(refused with no reason)" : err;
}

void check_the_shape_comes_from_the_config() {
    DecodeGeometry g{};
    std::string err;
    const Facts f = qwen3_next();
    expect(geometry_from_facts(f, g, &err), "a complete config builds: " + err);
    expect(g.hidden == 2048 && g.n_layers == 48 && g.vocab == 151936,
           "the decoder's dimensions are the config's, not the struct's defaults");
    expect(g.head_dim == 256 && g.n_q_heads == 16 && g.n_kv_heads == 2,
           "so are the attention heads");
    expect(g.full_attn_interval == 4, "and the full-attention interval");

    // The two GDN widths are DERIVED. A config cannot state them
    // inconsistently with its head counts, so neither can this driver: the
    // convolution runs over q, k and v concatenated, and the value total is
    // what the output projection consumes.
    expect(g.gdn_v_total == 32 * 128, "the value total is v_heads x v_dim");
    expect(g.gdn_conv_dim == 2 * 16 * 128 + 32 * 128,
           "the convolution spans q and k at the key heads plus v at the value heads");
    // Which is exactly what the state strides are read off.
    expect(g.gdn_conv_stride_bytes() == std::size_t(g.gdn_conv_dim) * 4u * 4u,
           "the conv state stride follows the derived width");

    // Asymmetric head counts are the case the old defaults could not express:
    // they had 16 value heads, and Qwen3-Next has 32. A driver that kept the
    // default would stride one head's recurrent state into another's.
    Facts sym = f;
    sym.gdn_v_heads = 16;
    DecodeGeometry gs{};
    expect(geometry_from_facts(sym, gs, &err) && gs.gdn_conv_dim != g.gdn_conv_dim,
           "changing the value-head count changes the derived widths");
}

void check_head_dim_is_derived_only_when_it_can_be() {
    Facts f = qwen3_next();
    f.head_dim = 0;
    DecodeGeometry g{};
    std::string err;
    expect(geometry_from_facts(f, g, &err) && g.head_dim == 2048 / 16,
           "an absent head_dim falls back to hidden/n_q_heads");
}

void check_what_is_refused() {
    // Nothing at all.
    expect(contains(refusal(Facts{}), "no decoder shape"),
           "an empty config is refused rather than defaulted");

    // The linear-attention block. These decide the conv and recurrent state
    // strides, and a wrong stride reads one head's state as another's -- not a
    // crash, a fluent model with the wrong memory.
    for (const char* which : {"k_heads", "v_heads", "k_dim", "v_dim"}) {
        Facts f = qwen3_next();
        const std::string w = which;
        if (w == "k_heads") f.gdn_k_heads = 0;
        if (w == "v_heads") f.gdn_v_heads = 0;
        if (w == "k_dim") f.gdn_k_dim = 0;
        if (w == "v_dim") f.gdn_v_dim = 0;
        expect(contains(refusal(f), "linear-attention block needs"),
               "a missing linear " + w + " is refused");
    }
    {
        Facts f = qwen3_next();
        f.gdn_conv_k = 0;
        expect(contains(refusal(f), "linear_conv_kernel_dim"),
               "a missing convolution width is refused");
    }

    // Which layers are linear cannot be guessed, and an irregular pattern is
    // not rounded to a regular one -- that would put full attention on layers
    // that are linear and vice versa.
    {
        Facts f = qwen3_next();
        f.full_attn_interval = 0;
        expect(contains(refusal(f), "full_attention_interval"),
               "no stated layer pattern is refused");
        f.full_attn_interval = -1;
        expect(contains(refusal(f), "irregular"), "an irregular pattern is refused");
    }

    // GQA, and the dense width.
    {
        Facts f = qwen3_next();
        f.n_kv_heads = 5;
        expect(contains(refusal(f), "GQA"), "a head count GQA cannot divide is refused");
    }
    {
        Facts f = qwen3_next();
        f.intermediate = 0;
        expect(contains(refusal(f), "dense FFN needs"),
               "a dense model with no intermediate_size is refused");
    }
}

// The same checkpoint with its mixture declared. Qwen3-Next routes 10 of 512
// experts through a 512-wide FFN, and shares the rest of its shape with the
// dense fixture -- which is the point: the two differ in the mixture alone.
Facts qwen3_next_routed() {
    Facts f = qwen3_next();
    f.n_experts = 512;
    f.experts_per_token = 10;
    f.moe_intermediate = 512;
    // Qwen3-Next-80B really ships this, and so does every other routed member
    // of the family. A routed fixture without one would be a model no release
    // is, and the whole point of these fixtures is that they are the configs.
    f.shared_expert_intermediate = 512;
    return f;
}

void check_the_routed_ffn_is_bounded_by_what_the_kernels_do() {
    const auto routed = [] { return qwen3_next_routed(); };
    {
        DecodeGeometry g{};
        std::string err;
        expect(geometry_from_facts(routed(), g, &err) && g.is_moe(),
               "a routed config builds: " + err);
        expect(g.ffn_width() == 512, "and ffn_width is the per-expert width");
    }
    {
        Facts f = routed();
        f.experts_per_token = 0;
        expect(contains(refusal(f), "num_experts_per_tok"),
               "experts with no top-k is refused");
    }
    {
        Facts f = routed();
        f.experts_per_token = 600;
        expect(contains(refusal(f), "exceeds num_experts"),
               "a top-k wider than the bank is refused");
    }
    {
        Facts f = routed();
        f.experts_per_token = 17;
        expect(contains(refusal(f), "top-k limit"),
               "a top-k past what the router kernel holds is refused, not clamped");
    }
    {
        Facts f = routed();
        f.n_experts = 2048;
        f.experts_per_token = 10;
        expect(contains(refusal(f), "threadgroup can rank"),
               "an expert bank past what one threadgroup ranks is refused, not clamped");
    }
    {
        Facts f = routed();
        f.moe_intermediate = 0;
        expect(contains(refusal(f), "moe_intermediate_size"),
               "a routed FFN with no expert width is refused");
    }
    {
        // CARRIED now, not refused. The router takes its denominator from
        // `RouterParams::softmax_over_all`, so a config whose weights come
        // from the softmax over EVERY expert gets that softmax rather than a
        // refusal -- but the flag has to arrive in the geometry, because one
        // that silently defaulted to true would renormalize the weights and
        // scale the routed FFN's contribution back up.
        Facts f = routed();
        f.norm_topk_prob = false;
        DecodeGeometry g{};
        std::string err;
        expect(geometry_from_facts(f, g, &err),
               "a router that normalizes over all experts is accepted: " + err);
        expect(!g.norm_topk_prob, "and the flag reaches the geometry");

        DecodeGeometry gn{};
        expect(geometry_from_facts(routed(), gn, &err) && gn.norm_topk_prob,
               "while the default stays the renormalized one");
    }
    // A shared expert runs beside the routed bank on every token. It is CARRIED
    // now, not refused -- but it has to arrive in the geometry, because a
    // shared width that silently defaulted to zero would drop the block and run
    // the routed mixture alone, which is the failure the refusal used to
    // prevent.
    {
        DecodeGeometry g{};
        std::string err;
        expect(geometry_from_facts(routed(), g, &err), "a shared expert is accepted: " + err);
        expect(g.shared_intermediate == 512 && g.has_shared_expert(),
               "and its width reaches the geometry");
    }
    {
        // A routing that genuinely has none is still a routing.
        Facts f = routed();
        f.shared_expert_intermediate = 0;
        DecodeGeometry g{};
        std::string err;
        expect(geometry_from_facts(f, g, &err) && g.is_moe() && !g.has_shared_expert(),
               "a mixture with no shared expert is still accepted");
    }
    // Routing only some layers is a third FFN shape in the same stack, and
    // this family's DAG emits one shape per model.
    {
        Facts f = routed();
        f.decoder_sparse_step = 2;
        expect(contains(refusal(f), "decoder_sparse_step"),
               "routing every other layer is refused");
    }
    {
        Facts f = routed();
        f.mlp_only_layer_count = 3;
        expect(contains(refusal(f), "mlp_only_layers"),
               "exempting some layers from routing is refused");
    }
}

// What the routed FFN actually costs in dispatches, and that the rest of the
// layer does not notice it.
//
// The two FFN shapes are checked against EACH OTHER rather than against a
// remembered dispatch count, because the number that matters is the delta: a
// routed layer replaces the dense mixture and changes nothing above it. A
// pinned total would have to be re-pinned every time attention changed, and
// re-pinning a number is how a test stops being a question.
void check_the_routed_ffn_replaces_the_dense_one_and_nothing_else() {
    std::printf("\n-- the routed FFN, in the DAG --\n");
    using pie::metal::Dispatch;
    using pie::metal::Kernel;

    std::string err;
    DecodeGeometry dense{};
    expect(geometry_from_facts(qwen3_next(), dense, &err), "dense geometry builds");
    DecodeGeometry moe{};
    expect(geometry_from_facts(qwen3_next_routed(), moe, &err), "routed geometry builds");

    const auto d_dag = pie::metal::build_decode_dag(dense);
    const auto m_dag = pie::metal::build_decode_dag(moe);

    auto count = [](const std::vector<Dispatch>& dag, Kernel k) {
        int n = 0;
        for (const auto& d : dag) n += (d.kind == k);
        return n;
    };

    // The mixture's own dispatches: absent from a dense stack entirely, and one
    // per layer in a routed one. If any of these were emitted unconditionally
    // the dense model would dispatch a router it has no weights for.
    const Kernel routed_only[] = {Kernel::LlRouter,      Kernel::GoRouterTopK,
                                  Kernel::LlMoeSort,     Kernel::LlMoeGather,
                                  Kernel::LlExpertGate,  Kernel::LlExpertUp,
                                  Kernel::LlExpertDown,  Kernel::LlMoeCombine};
    for (Kernel k : routed_only) {
        expect(count(d_dag, k) == 0, "a dense stack dispatches no part of a mixture");
        expect(count(m_dag, k) == moe.n_layers, "a routed stack routes every layer");
    }
    // And the dense projections are gone, not merely joined -- a routed layer
    // that still ran `mlp.gate_proj` would be reading a tensor the checkpoint
    // does not have.
    for (Kernel k : {Kernel::QmvGate, Kernel::QmvUp, Kernel::QmvDown}) {
        expect(count(d_dag, k) == dense.n_layers, "a dense stack keeps its own projections");
        expect(count(m_dag, k) == 0, "a routed stack drops the dense projections");
    }
    // `SiluMul` is shared: one per layer either way -- dense it is the FFN's,
    // routed it is the SHARED expert's, and both are one row per token. The
    // mixture's own SwiGLU is a different kind precisely because it is a
    // different extent, and a routed layer runs BOTH.
    expect(count(d_dag, Kernel::SiluMul) == dense.n_layers &&
               count(m_dag, Kernel::SiluMul) == moe.n_layers,
           "both shapes run one dense-shaped SiluMul per layer");
    expect(count(d_dag, Kernel::LlExpertSiluMul) == 0 &&
               count(m_dag, Kernel::LlExpertSiluMul) == moe.n_layers,
           "only the routed shape runs a SwiGLU over the sorted stack");

    // The shared expert. Five dispatches beside the mixture, every layer, on
    // every token -- and none of them in a dense stack, which has no such
    // block because its whole FFN is the dense one.
    for (Kernel k : {Kernel::LlSharedGate, Kernel::LlSharedUp, Kernel::LlSharedDown,
                     Kernel::LlSharedGateProj, Kernel::LlSharedCombine}) {
        expect(count(d_dag, k) == 0, "a dense stack dispatches no shared expert");
        expect(count(m_dag, k) == moe.n_layers, "a routed stack runs one every layer");
    }
    // And it is dropped when the config says there is none -- otherwise the
    // load would demand `mlp.shared_expert.*` from a checkpoint without them.
    {
        Facts f = qwen3_next_routed();
        f.shared_expert_intermediate = 0;
        DecodeGeometry bare{};
        std::string e2;
        expect(geometry_from_facts(f, bare, &e2), "a bare mixture builds: " + e2);
        const auto b_dag = pie::metal::build_decode_dag(bare);
        expect(count(b_dag, Kernel::LlSharedCombine) == 0 &&
                   count(b_dag, Kernel::SiluMul) == 0,
               "a mixture with no shared expert emits neither it nor its SwiGLU");
        expect(count(b_dag, Kernel::LlExpertSiluMul) == bare.n_layers,
               "but still runs the mixture's own");
    }

    // The shared expert's shapes, which are the thing a silent default would
    // get wrong. Its gate produces ONE logit a token -- bound as `hidden -> 1`,
    // not as a width -- and its projections are its own width, not the routed
    // one, which happens to be the same number in this fixture and is not in
    // Qwen3.5-122B-A10B.
    expect(pie::metal::qmv_kn(Kernel::LlSharedGateProj, moe).N == 1,
           "the shared expert's gate is one number a token");
    // Asked at a width that is NOT the routed one. Every released member of
    // this family happens to set the two equal, so the fixture cannot tell
    // `shared_intermediate` from `moe_intermediate` -- and a projection wired
    // to the wrong one would pass on all four checkpoints and then read past
    // the end of the first one that separates them.
    DecodeGeometry skew = moe;
    skew.shared_intermediate = 3072;
    expect(pie::metal::qmv_kn(Kernel::LlSharedGate, skew).N == 3072 &&
               pie::metal::qmv_kn(Kernel::LlSharedUp, skew).N == 3072 &&
               pie::metal::qmv_kn(Kernel::LlSharedDown, skew).K == 3072,
           "and its SwiGLU runs at the SHARED width, not the routed one");
    expect(pie::metal::qmv_kn(Kernel::LlSharedDown, skew).N == skew.hidden,
           "landing back in the hidden it will be added to");

    // Tying is what decides which pair of kinds the two ends of the model
    // run, and a kind is a weight name. A routed checkpoint of this family is
    // ALWAYS untied -- 35B-A3B ships an `lm_head` beside its `embed_tokens` --
    // so getting this wrong is not an edge case, it is every MoE member: both
    // ends would ask for `shared_embedding` and the contract would declare that
    // name twice.
    {
        Facts f = qwen3_next_routed();
        f.tied_embeddings = false;
        DecodeGeometry ut{};
        std::string e3;
        expect(geometry_from_facts(f, ut, &e3), "an untied config builds: " + e3);
        const auto u_dag = pie::metal::build_decode_dag(ut);
        expect(count(u_dag, Kernel::EmbedUntied) == 1 && count(u_dag, Kernel::LmHeadUntied) == 1,
               "an untied model reads its own two tensors");
        expect(count(u_dag, Kernel::EmbedGather) == 0 && count(u_dag, Kernel::QmvLmHead) == 0,
               "and neither end asks for the shared slot");
        expect(count(m_dag, Kernel::EmbedGather) == 1 && count(m_dag, Kernel::QmvLmHead) == 1 &&
                   count(m_dag, Kernel::EmbedUntied) == 0 &&
                   count(m_dag, Kernel::LmHeadUntied) == 0,
               "a tied model still reads one table twice");
        expect(u_dag.size() == m_dag.size(),
               "and tying costs no dispatch either way -- it is a name, not a shape");
    }

    // Everything above the FFN is untouched. Attention is where this family
    // differs from every other one, and the mixture is not allowed to perturb
    // it -- a hybrid stack whose GDN/full split moved when experts appeared
    // would be a different model wearing the same config.
    for (Kernel k : {Kernel::GdnCore, Kernel::Sdpa, Kernel::QmvQ, Kernel::FfnRms}) {
        expect(count(d_dag, k) == count(m_dag, k),
               "the mixture does not disturb what runs above it");
    }

    // The residual has to survive the swap. The dense tail can fuse the add
    // into `down_proj`; `moe_combine_sorted` has no such fusion, so a routed
    // layer must still emit the standalone add -- and if it did not, every
    // layer's input would be dropped and the model would still produce text.
    const auto m_fused = pie::metal::build_decode_dag(moe, false, /*fuse_residual=*/true);
    expect(count(m_fused, Kernel::LayerOut) == moe.n_layers,
           "a routed layer adds its residual even when the dense tail would fuse it");

    // The projections run on the SORTED rows, not on the token. At M=1 the
    // sort is a pure grouping -- one row per (token, slot) pair, no padding --
    // so a routed matvec is `experts_per_token` times as wide as a dense one.
    // Launching it at the dense extent would compute slot 0 and leave the rest
    // of the pool holding whatever the previous layer wrote.
    const int pairs = moe.experts_per_token;
    const int sorted = pie::metal::shared_kernels::moe_sorted_rows(pairs, moe.n_experts);
    expect(sorted == pairs, "at M=1 the sort pads nothing");
    for (const auto& d : m_dag) {
        if (d.kind != Kernel::LlExpertGate) continue;
        expect(d.grid.x == 32u * static_cast<std::uint32_t>(sorted),
               "a routed projection is launched over every (token, slot) pair");
        break;
    }
}

// The widths the mixture's matvecs are bound at, and the trap under them.
void check_the_routed_matvecs_know_their_own_width() {
    std::printf("\n-- the mixture's matvec widths --\n");
    using pie::metal::Kernel;
    using pie::metal::qmv_kn;

    std::string err;
    DecodeGeometry g{};
    expect(geometry_from_facts(qwen3_next_routed(), g, &err), "routed geometry builds");

    // The router is one logit per expert; an expert's gate and up are
    // `moe_intermediate` wide and its down comes back to hidden. Bound from
    // `intermediate` instead -- the DENSE width, which a routed checkpoint
    // still carries a value for in some configs -- every expert would read
    // seven times its own slice and stride off the end of the bank.
    expect(qmv_kn(Kernel::LlRouter, g).K == g.hidden &&
               qmv_kn(Kernel::LlRouter, g).N == g.n_experts,
           "the router projects hidden to one logit per expert");
    expect(qmv_kn(Kernel::LlExpertGate, g).N == g.moe_intermediate &&
               qmv_kn(Kernel::LlExpertUp, g).N == g.moe_intermediate,
           "an expert's gate and up are the MoE width, not the dense one");
    expect(qmv_kn(Kernel::LlExpertDown, g).K == g.moe_intermediate &&
               qmv_kn(Kernel::LlExpertDown, g).N == g.hidden,
           "an expert's down comes back to hidden");

    // The GDN decay and beta projections are QUANTIZED, like every other
    // projection in the layer. They were bound as dense bf16 matrices, which is
    // what a Qwen3-Next preview repack shipped; both released Qwen3.5
    // checkpoints ship them as a 4-bit weight with its own scales and biases,
    // and reading those as bf16 produced NaN in a quarter of the channels and
    // plausible garbage in the rest. `qmv_kn` answering for them is what makes
    // them a matvec at all -- `is_qmv` is defined as `N != 0` -- so it decides
    // the constant binding, the batched launch shape and the PSO together.
    expect(qmv_kn(Kernel::GdnInA, g).K == g.hidden &&
               qmv_kn(Kernel::GdnInA, g).N == g.gdn_v_heads,
           "the GDN decay projection is a matvec from hidden to one scalar per v-head");
    expect(qmv_kn(Kernel::GdnInB, g).K == g.hidden &&
               qmv_kn(Kernel::GdnInB, g).N == g.gdn_v_heads,
           "the GDN beta projection is a matvec from hidden to one scalar per v-head");
    expect(pie::metal::qmv_out_size(Kernel::GdnInA, g) == g.gdn_v_heads &&
               pie::metal::qmv_out_size(Kernel::GdnInB, g) == g.gdn_v_heads,
           "the batched path launches the GDN a/b projections over their own width");

    // And the trap. `qmv_kn` answers off the geometry it is GIVEN, and the
    // predicate that decides whether a dispatch gets a K and an N at all used
    // to ask a default-constructed one. Every dense projection's width is
    // nonzero whatever the numbers are, so it worked -- but a default geometry
    // has no experts, so the mixture's matvecs would have answered "not a
    // matvec", skipped the binding, and run against whatever the pool held at
    // those ordinals. This assertion is what makes that mistake loud.
    const DecodeGeometry none{};
    expect(qmv_kn(Kernel::LlExpertGate, none).N == 0 && qmv_kn(Kernel::LlRouter, none).N == 0,
           "a geometry with no experts gives the mixture no width at all");
    expect(qmv_kn(Kernel::QmvQ, none).N != 0,
           "a dense projection has a width even then -- which is why the trap was quiet");
}

// The mixture's activations, and whether they fit.
void check_the_mixture_fits_the_scratch_pool() {
    std::printf("\n-- the mixture's activations --\n");
    using pie::metal::Kernel;
    std::string err;
    DecodeGeometry moe{};
    expect(geometry_from_facts(qwen3_next_routed(), moe, &err), "routed geometry builds");
    DecodeGeometry dense{};
    expect(geometry_from_facts(qwen3_next(), dense, &err), "dense geometry builds");

    const auto m_sched =
        pie::metal::build_scratch_schedule(pie::metal::build_decode_dag(moe), moe);
    const auto d_sched =
        pie::metal::build_scratch_schedule(pie::metal::build_decode_dag(dense), dense);

    // The colouring self-checks that no two simultaneously-live activations
    // share a buffer. The routed block is where that is hardest: the sort's
    // four outputs and the router's two are written once and still read after
    // three matvecs have allocated freely between them, so a colouring that
    // treated them like dense temporaries would recycle one under its reader.
    expect(m_sched.hazard_free, "the routed schedule is hazard-free");
    expect(d_sched.hazard_free, "the dense schedule still is");
    // And it has to fit. There is no spilling here -- a schedule needing more
    // buffers than the pool has would bind the overflow to nothing and every
    // dispatch past it would read whatever was last there.
    expect(m_sched.colors_used <= pie::metal::SCRATCH_POOL,
           "the routed schedule fits the pool: " + std::to_string(m_sched.colors_used) + " of " +
               std::to_string(pie::metal::SCRATCH_POOL));

    // The pool slot has to be sized for a SORTED stack, not for a token's
    // worth. `experts_per_token` rows of `moe_intermediate` is what the gate
    // and up actually write, and an overrun here runs into the next pool slot
    // -- another live activation -- rather than into unmapped memory, so it
    // corrupts silently instead of faulting.
    const int sorted = pie::metal::moe_sorted_rows(moe);
    expect(pie::metal::scratch_widest_elems(moe) >= sorted * moe.moe_intermediate,
           "a scratch slot holds the whole sorted gate/up stack");
    expect(pie::metal::scratch_widest_elems(moe) >= sorted * moe.hidden,
           "and the gathered rows the projections read");
    // The long live range, checked as a specific claim rather than left to the
    // colouring's global self-check. `RouterTopK` writes the routing weights
    // and `Combine` is what finally sums with them -- five dispatches and three
    // matvecs later, every one of which allocates. If the colouring recycled
    // that buffer under its reader, the mixture would be summed with whatever
    // the last matvec wrote, which is a plausible-looking blend of the right
    // experts.
    {
        const auto dag = pie::metal::build_decode_dag(moe);
        int wrote = -1, read = -1;
        for (std::size_t i = 0; i < dag.size(); ++i) {
            for (const auto& b : m_sched.per_dispatch[i].binds) {
                if (dag[i].kind == Kernel::GoRouterTopK && b.bind_index == 2 && wrote < 0) {
                    wrote = b.buffer_id;
                } else if (dag[i].kind == Kernel::LlMoeCombine && b.bind_index == 1 &&
                           read < 0) {
                    read = b.buffer_id;
                }
            }
            if (read >= 0) break;
        }
        expect(wrote >= 0 && read == wrote,
               "the routing weights survive from the router to the combine");
    }

    // A dense model pays none of it.
    expect(pie::metal::scratch_widest_elems(dense) < sorted * moe.hidden,
           "a dense model's pool is not sized for a mixture it does not have");
}

// The mixture on the batched path, where the sort starts to pay.
void check_the_batched_mixture_becomes_a_matmul() {
    std::printf("\n-- the mixture, batched --\n");
    using pie::metal::Kernel;
    std::string err;
    DecodeGeometry moe{};
    expect(geometry_from_facts(qwen3_next_routed(), moe, &err), "routed geometry builds");

    // The batched builder derives from the M=1 one, so the mixture is already
    // in it -- what it needed was a launch shape. Before this it threw
    // "missing multi-batch launch geometry", which was at least loud.
    // Wide enough that the sort pays. The threshold is not a token count: it
    // is `n_experts * tile / 2` PAIRS, so a 512-expert mixture needs 4096 of
    // them before a tile is half full, which at ten experts per token is 410
    // tokens. A 128-expert mixture would cross it at 103. The number below is
    // chosen from that arithmetic rather than guessed.
    const int wide = 512;
    const auto dag = pie::metal::build_decode_dag_mb(moe, wide, 0);
    int routed_dispatches = 0;
    for (const auto& d : dag) {
        if (d.kind == Kernel::LlExpertGate || d.kind == Kernel::LlExpertUp ||
            d.kind == Kernel::LlExpertDown) {
            ++routed_dispatches;
        }
    }
    expect(routed_dispatches == 3 * moe.n_layers, "the batched DAG routes every layer");

    // A prefill row claims a FIXED stride of argument-table ordinals, and its
    // DAG has to fit inside one. The stride is a constant; the DAG's length is
    // the model's -- about twenty-seven dispatches a layer, and a routed layer
    // is at the wide end of that. Qwen3.5-35B-A3B's forty layers emit 1083 per
    // row, which is more than twice what the stride used to be, so the whole
    // family refused to prefill on a real MoE checkpoint: "prefill DAG exceeds
    // its argument-table ordinal stride", after loading nineteen gigabytes to
    // find out. It is arithmetic on a fixture, so it belongs here, where it
    // costs nothing and needs no weights.
    const auto row = pie::metal::build_decode_dag_mb(moe, 1, 0);
    expect(int(row.size()) < pie::metal::kPrefillOrdinalStride,
           "one prefill row's routed DAG fits inside its ordinal stride");
    // And with room to spare -- half the stride, so a decoder twice this deep
    // still fits and the next MoE does not rediscover this by loading.
    expect(int(row.size()) * 2 < pie::metal::kPrefillOrdinalStride,
           "and with room for a decoder twice as deep");

    // A single-token decode routes ten pairs over five hundred and twelve
    // experts, where every tile would be one live row in sixteen -- so the
    // matvec wins outright and `qmm_bn` must stay zero. Once the batch is wide
    // enough the projections become matmuls. Getting this backwards is not a
    // crash either way: the matvec form re-reads the whole expert bank per
    // row, which at width is the difference between amortizing the weights and
    // not.
    const auto one = pie::metal::build_decode_dag_mb(moe, 1, 0);
    for (const auto& d : one) {
        if (d.kind != Kernel::LlExpertGate) continue;
        expect(d.qmm_bn == 0, "at M=1 the mixture stays matvecs");
        break;
    }
    for (const auto& d : dag) {
        if (d.kind != Kernel::LlExpertGate) continue;
        expect(d.qmm_bn > 0, "at width the mixture becomes matmuls");
        // And the tile must be what the sort padded to. A block spanning two
        // experts reads one expert's weights for the other's rows.
        expect(d.qmm_bm == pie::metal::shared_kernels::moe_tile_rows(
                               wide * moe.experts_per_token, moe.n_experts),
               "a routed tile is the tile the sort padded every run to");
        // Split-K would sum partial products across a K the routing already
        // split by expert, so it must stay off.
        expect(d.qmm_split == 1, "a routed projection is never split along K");
        break;
    }

    // The projections run over the SORTED rows, which once the sort batches is
    // neither the token count nor `tokens * k`: it pads every touched expert's
    // run to a whole tile. Launched over the tokens instead, the tail of the
    // stack keeps the previous layer's output.
    const int sorted = pie::metal::moe_sorted_rows(moe, wide);
    expect(sorted > wide * moe.experts_per_token,
           "a batched sort pads, where the M=1 sort did not");
    for (const auto& d : dag) {
        if (d.kind != Kernel::LlMoeGather) continue;
        expect(d.grid.y == static_cast<std::uint32_t>(sorted),
               "the gather fills the whole padded stack");
        break;
    }
    // The pool slot has to hold that padded stack. Every other activation in
    // this family is `n` rows of its M=1 footprint, and the mixture is the one
    // that is not: a linear bound says 5120 rows here and the sort produces
    // 12800. The extra lands in the next pool slot, which is another live
    // activation, so it corrupts rather than faults.
    expect(pie::metal::scratch_slot_elems(moe, wide) >= std::size_t(sorted) * moe.hidden,
           "a batched pool slot holds the sort's padding too");
    expect(pie::metal::scratch_slot_elems(moe, wide) >
               std::size_t(pie::metal::scratch_widest_elems(moe)) * wide,
           "which is strictly more than n rows of the M=1 footprint");

    // And the combine comes back to one row per TOKEN -- the k results per
    // token are only summed here.
    for (const auto& d : dag) {
        if (d.kind != Kernel::LlMoeCombine) continue;
        expect(d.grid.y == static_cast<std::uint32_t>(wide),
               "the combine returns one row per token");
        break;
    }
}

// A config.json this driver must read, written to a temp dir and parsed back.
//
// The whole family ships multimodal: every real Qwen3.5 release -- 0.8B through
// 122B-A10B, and the mlx repacks of each -- nests the text decoder's facts
// under `text_config` and leaves the root to the wrapper. Qwen3-Next, the older
// sibling, is flat. Both are this family, so both must parse, and reading the
// wrong scope does not fail loudly: every key is simply absent, so every fact
// keeps its struct default and the driver runs a checkpoint at some other
// model's shape. That is the exact failure this file was written for, and it
// was live -- the block read the root.
void check_a_real_config_is_read_from_whichever_scope_holds_it() {
    namespace fs = std::filesystem;
    const auto parse = [](const std::string& body) {
        const fs::path dir = fs::temp_directory_path() /
                             ("q35cfg" + std::to_string(std::rand()));
        fs::create_directories(dir);
        { std::ofstream(dir / "config.json") << body; }
        const auto facts = pie::metal::read_model_facts(dir.string());
        fs::remove_all(dir);
        return facts;
    };
    // The decoder's facts, verbatim in shape from Qwen3.5-0.8B.
    const std::string decoder = R"("model_type": "qwen3_5_text",
        "hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 8,
        "num_key_value_heads": 2, "head_dim": 256, "intermediate_size": 3584,
        "vocab_size": 248320, "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4, "linear_num_key_heads": 16,
        "linear_num_value_heads": 16, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "rms_norm_eps": 1e-06)";

    const auto nested = parse("{\"model_type\": \"qwen3_5\", \"vision_config\": {\"depth\": 24},"
                              " \"text_config\": {" + decoder + "}}");
    expect(nested.q35_hidden_size == 1024 && nested.q35_num_hidden_layers == 24 &&
               nested.q35_head_dim == 256 && nested.q35_vocab_size == 248320 &&
               nested.q35_intermediate_size == 3584,
           "a multimodal config's decoder facts are read from text_config");
    expect(nested.q35_linear_key_heads == 16 && nested.q35_linear_value_heads == 16 &&
               nested.q35_linear_conv_kernel == 4 && nested.q35_full_attn_interval == 4,
           "including the linear-attention geometry, which has no safe default");

    const auto flat = parse("{" + decoder + "}");
    expect(flat.q35_hidden_size == 1024 && flat.q35_num_hidden_layers == 24 &&
               flat.q35_linear_key_heads == 16,
           "and a flat config -- Qwen3-Next's spelling -- still reads from the root");

    // The routed half, which only the MoE members carry.
    const auto moe = parse("{\"model_type\": \"qwen3_5_moe\", \"text_config\": {" + decoder +
                           ", \"num_experts\": 256, \"num_experts_per_tok\": 8,"
                           " \"moe_intermediate_size\": 512,"
                           " \"shared_expert_intermediate_size\": 512}}");
    expect(moe.q35_num_experts == 256 && moe.q35_num_experts_per_tok == 8 &&
               moe.q35_moe_intermediate_size == 512 &&
               moe.q35_shared_expert_intermediate == 512,
           "a routed config's mixture is read from the same scope as its shape");
}

}  // namespace

int main() {
    std::printf("==== qwen3_5_geometry_test ====\n");
    check_the_shape_comes_from_the_config();
    check_head_dim_is_derived_only_when_it_can_be();
    check_what_is_refused();
    check_a_real_config_is_read_from_whichever_scope_holds_it();
    check_the_routed_ffn_is_bounded_by_what_the_kernels_do();
    check_the_routed_ffn_replaces_the_dense_one_and_nothing_else();
    check_the_routed_matvecs_know_their_own_width();
    check_the_mixture_fits_the_scratch_pool();
    check_the_batched_mixture_becomes_a_matmul();
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
