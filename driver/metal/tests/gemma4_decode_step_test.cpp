// Gemma 4's decode DAG and geometry, checked with no GPU.
//
// The same thing qwen3.5's "363 dispatches verified" bought: the shape of the
// step is a pure function, so the schedule can be wrong here and be caught here
// rather than as a wrong token forty kernels later.

#include <cstdio>
#include <string>
#include <vector>

#include "model/gemma4/decode_consts.hpp"
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
                   Kind::QmvO, Kind::PostAttnNorm, Kind::AttnResidual, Kind::FfnNorm,
                   Kind::QmvGate, Kind::QmvUp, Kind::GegluTanh, Kind::QmvDown,
                   Kind::PostFfnNorm, Kind::FfnResidual, Kind::LayerScalar}) {
        expect_eq(count(dag, k), 35, "one per layer");
    }
    // PLE: five layer-less precompute dispatches, five more per layer.
    expect_eq(count(dag, Kind::PleCombine), 1, "PLE precompute runs once");
    expect_eq(count(dag, Kind::PleResidual), 35, "PLE residual runs per layer");

    // The tail.
    expect_eq(count(dag, Kind::FinalRms), 1, "one final norm");
    expect_eq(count(dag, Kind::LmHead), 1, "one logits matvec");
    expect_eq(count(dag, Kind::FinalSoftcap), 1, "gemma4 softcaps its logits");

    // 1 embed + 4 PLE precompute + 35*(21 shared-safe) + owning extras + tail.
    const int per_layer_always = 16 + 5;  // attention/FFN/scalar + per-layer PLE
    const int owning_extra = 6;           // k/v proj, k/v norm, rope_k, append
    const int want = 1 + 4 + g.n_layers * per_layer_always + 15 * owning_extra + 4;
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
    expect_eq(count(dag, Kind::PleResidual), 0, "and no per-layer PLE");
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
    int last_ordinal = -1;
    bool ordered = true;
    for (const Use& u : plan.uses) {
        ordered = ordered && u.ordinal >= last_ordinal;
        last_ordinal = u.ordinal;
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
    int final_rms_ord = -1;
    for (const Dispatch& d : dag) {
        if (d.kind == Kind::FinalRms) final_rms_ord = d.ordinal;
    }
    int final_rms_input = -2;
    for (const Use& u : plan.uses) {
        if (u.ordinal == final_rms_ord && !u.is_write) final_rms_input = u.value;
    }
    int last_scalar_out = -3;
    for (const Use& u : plan.uses) {
        if (u.ordinal < final_rms_ord && u.is_write) last_scalar_out = u.value;
    }
    expect_eq(final_rms_input, last_scalar_out,
              "the final norm reads what the last layer wrote");
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
    std::printf("\n==== gemma4_decode_step_test: %s ====\n",
                g_failures == 0 ? "all passed" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}
