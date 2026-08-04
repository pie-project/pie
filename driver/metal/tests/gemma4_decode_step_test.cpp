// Gemma 4's decode DAG and geometry, checked with no GPU.
//
// The same thing qwen3.5's "363 dispatches verified" bought: the shape of the
// step is a pure function, so the schedule can be wrong here and be caught here
// rather than as a wrong token forty kernels later.

#include <cstdio>
#include <string>
#include <vector>

#include "model/gemma4/decode_consts.hpp"
#include "model/gemma4/bind.hpp"
#include "model/gemma4/scratch.hpp"
#include "model/gemma4/decode_step.hpp"
#include "model/gemma4/geometry.hpp"

using namespace pie::metal::gemma4;

namespace {

int g_failures = 0;

bool expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
    return ok;
}

bool expect_eq(long long got, long long want, const std::string& what) {
    return expect(got == want,
                  what + " (got " + std::to_string(got) + ", want " + std::to_string(want) + ")");
}

int count(const std::vector<Dispatch>& dag, Kind k) {
    int n = 0;
    for (const Dispatch& d : dag) n += d.kind == k ? 1 : 0;
    return n;
}

// The E2B checkpoint's schedule, read from its config.json: `layer_types` lists
// full attention at exactly 4, 9, 14, 19, 24, 29, 34.
void the_attention_schedule_matches_the_checkpoint() {
    std::printf("[attention schedule]\n");
    const Gemma4Geometry g;
    std::vector<int> full;
    for (int L = 0; L < g.n_layers; ++L) {
        if (g.is_full_attn(L)) full.push_back(L);
    }
    expect(full == std::vector<int>{4, 9, 14, 19, 24, 29, 34},
           "full attention lands on the layers config.json names");
    expect_eq(g.n_full_attn(), 7, "seven full-attention layers");
    expect(g.is_sliding(0) && !g.is_sliding(4), "sliding is the complement of full");

    // head_dim, rope base and the rotated fraction all follow the type.
    expect_eq(g.head_dim_of(0), 256, "sliding layers use head_dim");
    expect_eq(g.head_dim_of(4), 512, "full layers use global_head_dim");
    expect_eq(g.rotary_dims_of(0), 256, "sliding layers rotate the whole head");
    expect_eq(g.rotary_dims_of(4), 128, "full layers rotate a quarter of it");
    expect(g.rope_theta_of(0) == 1.0e4f && g.rope_theta_of(4) == 1.0e6f,
           "each attention type carries its own rope base");
}

void kv_sharing_covers_the_tail_of_the_stack() {
    std::printf("[kv sharing]\n");
    const Gemma4Geometry g;
    expect_eq(g.first_kv_shared(), 15, "layers 15+ share KV (35 - 20)");
    expect(!g.is_kv_shared(14) && g.is_kv_shared(15), "the split is where the config puts it");
    expect_eq(g.n_kv_owning(), 15, "fifteen layers own KV pages");

    // A shared layer reads the most recent earlier owner of its own type.
    expect_eq(g.kv_source(14), 14, "an owning layer is its own source");
    expect_eq(g.kv_source(19), 14, "a shared FULL layer reads the last owning full layer");
    const int src15 = g.kv_source(15);
    expect(src15 >= 0 && src15 < 15 && g.is_sliding(src15),
           "a shared SLIDING layer reads an owning sliding layer");

    // The MLP doubles over exactly the shared range.
    expect_eq(g.intermediate_of(14), 6144, "the base MLP width below the split");
    expect_eq(g.intermediate_of(15), 12288, "double-wide at and above it");
}

void the_dag_skips_what_a_shared_layer_does_not_have() {
    std::printf("[dag shape]\n");
    const Gemma4Geometry g;
    const std::vector<Dispatch> dag = build_gemma4_dag(g);
    const DagStats s = dag_stats(dag, g);

    expect_eq(s.kv_owning_layers, 15, "stats agree with the geometry on ownership");
    expect_eq(s.kv_shared_layers, 20, "and on sharing");

    // The five KV-side dispatches exist only on layers that own their KV.
    for (Kind k : {Kind::QmvK, Kind::QmvV, Kind::KNorm, Kind::VNorm, Kind::RopeK,
                   Kind::KvAppend}) {
        expect_eq(count(dag, k), 15, "one per KV-owning layer");
    }
    // Everything on the query side, and the whole FFN, runs on every layer.
    for (Kind k : {Kind::AttnNorm, Kind::QmvQ, Kind::QNorm, Kind::RopeQ, Kind::Sdpa,
                   Kind::QmvO, Kind::PostAttnResidual, Kind::FfnNorm,
                   Kind::QmvGate, Kind::QmvUp, Kind::GegluTanh, Kind::QmvDown,
                   Kind::PostFfnResidual}) {
        expect_eq(count(dag, k), 35, "one per layer");
    }
    // PLE: four layer-less precompute dispatches, four more per layer -- the
    // per-layer norm, its residual add and the learned scalar are one dispatch.
    expect_eq(count(dag, Kind::PleCombine), 1, "PLE precompute runs once");
    expect_eq(count(dag, Kind::PleResidualScaled), 35,
              "the PLE residual, its norm and the layer scalar are one dispatch");
    expect_eq(count(dag, Kind::LayerScalar), 0,
              "so the standalone scalar does not run when there is a PLE to ride on");

    // The tail.
    expect_eq(count(dag, Kind::RowGather), 1,
              "the sampled rows are compacted once, before the tail");
    expect_eq(count(dag, Kind::FinalRms), 1, "one final norm");
    expect_eq(count(dag, Kind::LmHead), 1, "one logits matvec");
    expect_eq(count(dag, Kind::FinalSoftcap), 1, "gemma4 softcaps its logits");

    // 1 embed + 4 PLE precompute + 35*(17 shared-safe) + owning extras + tail.
    const int per_layer_always = 13 + 4;  // attention/FFN + per-layer PLE
    const int owning_extra = 6;           // k/v proj, k/v norm, rope_k, append
    // Tail: row gather, final norm, LM head, softcap, and the argmax this DAG
    // was built with.
    const int want = 1 + 4 + g.n_layers * per_layer_always + 15 * owning_extra + 5;
    expect_eq(s.total, want, "the whole step is exactly this many dispatches");

    // Ordinals are dense and in order — they are the argument-table keys.
    bool dense = true;
    for (std::size_t i = 0; i < dag.size(); ++i) {
        dense = dense && dag[i].ordinal == static_cast<int>(i);
    }
    expect(dense, "ordinals are a dense 0..N-1 run");

    // Every layer dispatch carries its own attention type.
    bool typed = true;
    for (const Dispatch& d : dag) {
        if (d.layer >= 0) typed = typed && d.sliding == g.is_sliding(d.layer);
    }
    expect(typed, "each dispatch carries its layer's attention type");
}

void a_family_without_ple_or_softcap_drops_those_dispatches() {
    std::printf("[optional stages]\n");
    Gemma4Geometry g;
    g.per_layer_emb_dim = 0;
    g.final_softcap = 0.0f;
    g.num_kv_shared_layers = 0;
    const std::vector<Dispatch> dag = build_gemma4_dag(g, /*with_argmax=*/false);
    expect_eq(count(dag, Kind::PleCombine), 0, "no PLE precompute without a PLE table");
    expect_eq(count(dag, Kind::PleResidualScaled), 0, "and no per-layer PLE");
    expect_eq(count(dag, Kind::LayerScalar), 35,
              "the layer scalar then runs on its own, with no PLE residual to ride on");
    expect_eq(count(dag, Kind::FinalSoftcap), 0, "no softcap when the config has none");
    expect_eq(count(dag, Kind::QmvK), 35, "every layer owns its KV when none is shared");
    expect_eq(count(dag, Kind::Argmax), 0, "argmax is opt-in");
}

// The thing that makes gemma4's consts different from qwen3.5's: they depend on
// the LAYER, not just the kind. If this ever collapses to one answer per kind,
// sliding layers get a full layer's head_dim and the model is silently wrong.
void matvec_shapes_follow_the_layer() {
    std::printf("[per-layer shapes]\n");
    const Gemma4Geometry g;

    // q_proj: 8 heads x this layer's head_dim.
    expect_eq(qmv_kn(Kind::QmvQ, g, 0).N, 8 * 256, "sliding layer's q_proj is 8 x 256");
    expect_eq(qmv_kn(Kind::QmvQ, g, 4).N, 8 * 512, "full layer's q_proj is 8 x 512");
    expect_eq(qmv_kn(Kind::QmvO, g, 0).K, 8 * 256, "and o_proj reads back the same width");
    expect_eq(qmv_kn(Kind::QmvO, g, 4).K, 8 * 512, "per layer");

    // The MLP doubles exactly where the KV is shared.
    expect_eq(qmv_kn(Kind::QmvGate, g, 14).N, 6144, "gate is base width below the split");
    expect_eq(qmv_kn(Kind::QmvGate, g, 15).N, 12288, "and double-wide at it");
    expect_eq(qmv_kn(Kind::QmvDown, g, 15).K, 12288, "down_proj reads the doubled width");

    // PLE: the model projection fans out to the whole table, the per-layer ones
    // work a slice.
    expect_eq(qmv_kn(Kind::PleProjGemv, g, -1).N, 35 * 256, "PLE projection covers every layer");
    expect_eq(qmv_kn(Kind::PleGateGemv, g, 0).N, 256, "the per-layer gate is one slice");
    expect_eq(qmv_kn(Kind::PleProjLayerGemv, g, 0).K, 256, "and projects that slice back");

    expect_eq(qmv_kn(Kind::LmHead, g, -1).N, 262144, "lm_head is the vocabulary");
    expect_eq(qmv_kn(Kind::AttnNorm, g, 0).N, 0, "a norm is not a matvec");
}

// The dataflow has to be a dataflow: nothing may be read before it is written,
// and the value the sampler reads has to be the one lm_head produced. A wiring
// slip here is a wrong token forty kernels later with nothing pointing at it.
void the_dataflow_never_reads_an_unwritten_value() {
    std::printf("[dataflow]\n");
    const Gemma4Geometry g;
    const std::vector<Dispatch> dag = build_gemma4_dag(g);
    const ScratchPlan plan = build_gemma4_scratch(dag, g);

    expect(plan.value_count > 0, "the walk produced values");
    expect(!plan.uses.empty(), "and uses");

    // Uses arrive in DAG order, so a read of a value never seen written is a
    // read-before-write.
    std::vector<bool> written(std::size_t(plan.value_count), false);
    int read_before_write = 0;
    int last_index = -1;
    bool ordered = true;
    for (const Use& u : plan.uses) {
        ordered = ordered && u.index >= last_index;
        last_index = u.index;
        if (u.is_write) {
            written[std::size_t(u.value)] = true;
        } else if (!written[std::size_t(u.value)]) {
            ++read_before_write;
        }
    }
    expect(ordered, "uses come in DAG order");
    expect_eq(read_before_write, 0, "no dispatch reads a value nothing has written");

    // Every value is read by someone, or producing it was pointless.
    std::vector<bool> read(std::size_t(plan.value_count), false);
    for (const Use& u : plan.uses) {
        if (!u.is_write) read[std::size_t(u.value)] = true;
    }
    int dead = 0;
    for (int v = 0; v < plan.value_count; ++v) {
        if (!read[std::size_t(v)] && v != plan.logits_value) ++dead;
    }
    expect_eq(dead, 0, "every activation produced is consumed");

    expect(plan.logits_value >= 0, "the logits have a value");

    // The residual stream has to be threaded, not reset: the last layer's
    // scalar output must reach the final norm.
    // Positions, not ordinals: `Use::index` is the DAG position, which is what
    // the dataflow's time axis means.
    int final_rms_at = -1;
    for (std::size_t i = 0; i < dag.size(); ++i) {
        if (dag[i].kind == Kind::FinalRms) final_rms_at = int(i);
    }
    int final_rms_input = -2;
    for (const Use& u : plan.uses) {
        if (u.index == final_rms_at && !u.is_write) final_rms_input = u.value;
    }
    int last_scalar_out = -3;
    for (const Use& u : plan.uses) {
        if (u.index < final_rms_at && u.is_write) last_scalar_out = u.value;
    }
    expect_eq(final_rms_input, last_scalar_out,
              "the final norm reads what the last layer wrote");
}

// Colouring turns the dataflow into buffers. Two things must hold: no two values
// that are live at the same time may share one (that is the whole hazard
// guarantee), and the pool must be small enough to actually allocate.
void colouring_is_hazard_free_and_small() {
    std::printf("[scratch colouring]\n");
    const Gemma4Geometry g;
    const std::vector<Dispatch> dag = build_gemma4_dag(g);
    const ScratchPlan plan = build_gemma4_scratch(dag, g);
    const ScratchColoring c = color_gemma4_scratch(dag, plan);

    expect(c.hazard_free, "no two overlapping values share a buffer");
    expect(c.colors_used > 0 && c.colors_used < 32,
           "the pool is a handful of buffers, not one per value (" +
               std::to_string(c.colors_used) + ")");
    expect_eq((long long)c.per_dispatch.size(), (long long)dag.size(),
              "every dispatch has a binding list");

    // Recycling is the point: 500+ values must not become 500+ buffers.
    const ScratchColoring nr = color_gemma4_scratch(dag, plan, /*no_recycle=*/true);
    expect_eq(nr.colors_used, plan.value_count, "no_recycle gives every value its own buffer");
    expect(c.colors_used < nr.colors_used / 10, "recycling collapses that by >10x");
}

// A KV-shared layer must read pages an OWNING layer of its own attention type
// wrote. Getting the type wrong reads a sliding layer's window as a full
// layer's history, which is wrong in a way no shape check would catch.
void kv_redirect_stays_within_an_attention_type() {
    std::printf("[kv redirect]\n");
    const Gemma4Geometry g;
    int checked = 0;
    bool same_type = true, owning = true, earlier = true;
    for (int L = 0; L < g.n_layers; ++L) {
        if (!g.is_kv_shared(L)) continue;
        const int src = g.kv_source(L);
        same_type = same_type && (g.is_sliding(src) == g.is_sliding(L));
        owning = owning && !g.is_kv_shared(src);
        earlier = earlier && src < L;
        ++checked;
    }
    expect_eq(checked, 20, "twenty layers share KV");
    expect(same_type, "each reads a layer of its own attention type");
    expect(owning, "which owns its pages");
    expect(earlier, "and comes earlier in the stack");
}

// A config either describes a shape this driver can schedule, or it is refused.
// Filling in defaults for a checkpoint whose shape we guessed at would produce
// plausible-looking wrong tokens, which is the worst available outcome.
struct Facts {
    int n_layers = 35, hidden = 1536, intermediate = 6144;
    int n_q_heads = 8, n_kv_heads = 1, head_dim = 256, global_head_dim = 512;
    int sliding_window = 512, num_kv_shared_layers = 20, per_layer_emb_dim = 256;
    int full_attn_interval = 5;
    bool double_wide_mlp = true;
    float final_softcap = 30.0f;
    float rope_theta_full = 1.0e6f, rope_theta_sliding = 1.0e4f, full_partial_rotary = 0.25f;
    bool enable_moe = false;
    int n_experts = 0, experts_per_token = 0, moe_intermediate = 0;
    bool attention_k_eq_v = false;
    int n_global_kv_heads = 0;
    bool present() const { return n_layers > 0 && hidden > 0; }
};

void a_config_either_describes_a_schedulable_shape_or_is_refused() {
    std::printf("[geometry from config]\n");
    Gemma4Geometry g;
    std::string err;

    // The real E2B config.
    expect(geometry_from_facts(Facts{}, g, &err), "E2B's shape builds: " + err);
    expect_eq(g.n_layers, 35, "layers");
    expect_eq(g.head_dim_of(0), 256, "sliding head_dim");
    expect_eq(g.head_dim_of(4), 512, "full head_dim");
    expect_eq(g.first_kv_shared(), 15, "kv split");
    expect_eq(g.intermediate_of(15), 12288, "double-wide range");
    expect(g.final_softcap == 30.0f, "softcap");

    // An empty config is not a gemma4 config.
    Facts empty;
    empty.n_layers = 0;
    expect(!geometry_from_facts(empty, g, &err), "an empty text_config is refused");

    // `layer_types` that is not a regular interval: the DAG's schedule assumes
    // one, so say so rather than mis-schedule.
    Facts irregular;
    irregular.full_attn_interval = -1;
    expect(!geometry_from_facts(irregular, g, &err), "an irregular layer_types is refused");

    // Every layer shared means no layer owns KV for the shared ones to read.
    Facts all_shared;
    all_shared.num_kv_shared_layers = all_shared.n_layers;
    expect(!geometry_from_facts(all_shared, g, &err),
           "a stack where nothing owns KV is refused");
}

// The KV region is sized by which layers OWN kv and how wide each one's head is
// -- not by counting full-attention layers, which is the other family's rule and
// wrong here twice over.
void the_kv_region_is_sized_per_owning_layer() {
    std::printf("[kv region]\n");
    Gemma4Geometry g;
    std::string err;
    expect(geometry_from_facts(Facts{}, g, &err), "geometry: " + err);

    const int max_ctx = 4096;
    const int act = 2;  // bf16

    // 15 owning layers: 12 sliding at head_dim 256, 3 full at 512.
    int owning_sliding = 0, owning_full = 0;
    for (int L = 0; L < g.n_layers; ++L) {
        if (g.is_kv_shared(L)) continue;
        (g.is_full_attn(L) ? owning_full : owning_sliding) += 1;
    }
    expect_eq(owning_sliding + owning_full, g.n_kv_owning(), "owning count");
    expect_eq(g.n_kv_owning(), 15, "E2B owns kv on 15 of 35 layers");

    const std::size_t want =
        std::size_t(2) * g.n_kv_heads * max_ctx * act *
        (std::size_t(owning_sliding) * 256 + std::size_t(owning_full) * 512);
    expect_eq(static_cast<long long>(gemma4_kv_region_bytes(g, max_ctx, act)),
              static_cast<long long>(want), "region bytes");

    // A full layer's cache really is twice a sliding layer's.
    expect_eq(static_cast<long long>(gemma4_kv_bytes_per_layer(g, 4, max_ctx, act)),
              2 * static_cast<long long>(gemma4_kv_bytes_per_layer(g, 0, max_ctx, act)),
              "a full layer's head is twice a sliding layer's");

    // The other family's rule, for contrast: counting full-attn layers at one
    // head width asks for a region that is wrong in both factors.
    int n_full = 0;
    for (int L = 0; L < g.n_layers; ++L) n_full += g.is_full_attn(L) ? 1 : 0;
    const std::size_t gdn_rule =
        std::size_t(2) * n_full * g.n_kv_heads * max_ctx * g.head_dim * act;
    expect(gdn_rule != want, "the full-attn-count rule would size this wrongly");
}

// The prefill DAG must describe the same model as the decode DAG.
//
// It is the same dispatch list with argument-table ordinals shifted clear, so
// the two paths cannot come to describe different models -- which is the failure
// that produces an engine whose prefill and decode disagree.
void the_prefill_dag_is_the_decode_dag_with_room_for_its_own_binds() {
    std::printf("[prefill DAG]\n");
    Gemma4Geometry g;
    const auto decode = build_gemma4_dag(g);
    const int base = 100000;
    const auto prefill = build_gemma4_dag_mb(g, base);

    expect_eq(static_cast<long long>(prefill.size()),
              static_cast<long long>(decode.size()), "same dispatch count");
    int wrong_kind = 0, wrong_ordinal = 0, collides = 0;
    for (std::size_t i = 0; i < decode.size(); ++i) {
        if (prefill[i].kind != decode[i].kind || prefill[i].layer != decode[i].layer ||
            prefill[i].sliding != decode[i].sliding) {
            ++wrong_kind;
        }
        if (prefill[i].ordinal != decode[i].ordinal + base) ++wrong_ordinal;
        if (prefill[i].ordinal <= decode.back().ordinal) ++collides;
    }
    expect(wrong_kind == 0, "every dispatch is the same kind, layer and attention type");
    expect(wrong_ordinal == 0, "every ordinal is shifted by exactly the base");
    expect(collides == 0, "and no prefill ordinal lands on a decode one");
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("gemma4 decode DAG + geometry\n");
    the_attention_schedule_matches_the_checkpoint();
    kv_sharing_covers_the_tail_of_the_stack();
    the_dag_skips_what_a_shared_layer_does_not_have();
    a_family_without_ple_or_softcap_drops_those_dispatches();
    matvec_shapes_follow_the_layer();
    the_dataflow_never_reads_an_unwritten_value();
    colouring_is_hazard_free_and_small();
    kv_redirect_stays_within_an_attention_type();
    a_config_either_describes_a_schedulable_shape_or_is_refused();
    the_kv_region_is_sized_per_owning_layer();
    the_prefill_dag_is_the_decode_dag_with_room_for_its_own_binds();
    std::printf("\n==== gemma4_decode_step_test: %s ====\n",
                g_failures == 0 ? "all passed" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}
