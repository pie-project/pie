// GPT-OSS's decode DAG and geometry, checked with no GPU and no checkpoint.
//
// The same thing qwen3.5's "363 dispatches verified" and gemma4's equivalent
// bought: the shape of the step is a pure function, so a schedule that is wrong
// is caught here rather than as a wrong token forty kernels later.

#include <cstdio>
#include <string>
#include <vector>

#include "model/gptoss/decode_step.hpp"
#include "model/gptoss/geometry.hpp"

using namespace pie::metal::gptoss;

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
    for (const Dispatch& d : dag) n += (d.kind == k) ? 1 : 0;
    return n;
}

// gpt-oss-20b's shape, read off the checkpoint's config.json.
struct Facts {
    int n_layers = 24, hidden = 2880, vocab = 201088;
    float eps = 1e-5f;
    int n_q_heads = 64, n_kv_heads = 8, head_dim = 64;
    int sliding_window = 128;
    int n_experts = 32, experts_per_token = 4, intermediate = 2880;
    float swiglu_limit = 7.0f;
    float rope_theta = 150000.0f, rope_factor = 32.0f;
    float rope_beta_fast = 32.0f, rope_beta_slow = 1.0f;
    int rope_original_max_position = 4096;
    bool present() const { return n_layers > 0 && hidden > 0; }
};

void the_geometry_is_the_checkpoints() {
    std::printf("[geometry]\n");
    GptOssGeometry g;
    std::string err;
    expect(geometry_from_facts(Facts{}, g, &err), "gpt-oss-20b's config builds a geometry");
    expect_eq(g.q_dim(), 64 * 64, "the query projection is n_q_heads * head_dim wide");
    expect_eq(g.kv_dim(), 8 * 64, "and the key/value projections are the grouped width");
    expect_eq(g.gqa_factor(), 8, "eight query heads share each key/value head");

    // Sliding and full alternate, starting sliding. The checkpoint's
    // `layer_types` lists exactly that, and getting it backwards is a whole
    // model's worth of wrong attention that still runs.
    bool alternates = true;
    for (int L = 0; L < g.n_layers; ++L) {
        alternates = alternates && (g.is_sliding(L) == ((L % 2) == 0));
        alternates = alternates && (g.is_sliding(L) != g.is_full_attn(L));
    }
    expect(alternates, "attention alternates sliding/full, starting sliding");

    // A config this driver cannot schedule is refused, not guessed at.
    GptOssGeometry bad;
    Facts no_experts;
    no_experts.n_experts = 0;
    expect(!geometry_from_facts(no_experts, bad, &err),
           "a config with no experts is refused: " + err);
    Facts too_many;
    too_many.experts_per_token = 64;
    expect(!geometry_from_facts(too_many, bad, &err),
           "so is routing to more experts than exist: " + err);
    Facts ragged_gqa;
    ragged_gqa.n_kv_heads = 7;
    expect(!geometry_from_facts(ragged_gqa, bad, &err),
           "so is a head count grouped attention cannot divide: " + err);
    // `router_topk` clamps both of these instead of failing, and gpt-oss shares
    // that kernel with llama. A clamped k routes to fewer experts than the
    // config asks for while `go_kind_width` keeps sizing the expert stacks by
    // the configured k, so the slots past the clamp are never written and the
    // model still produces text.
    Facts wide_k;
    wide_k.n_experts = 128;
    wide_k.experts_per_token = 20;
    expect(!geometry_from_facts(wide_k, bad, &err),
           "so is a top-k wider than the router can hold: " + err);
    Facts many_experts;
    many_experts.n_experts = 2048;
    expect(!geometry_from_facts(many_experts, bad, &err),
           "so is an expert count past one threadgroup of lanes: " + err);
}

void the_step_is_this_many_dispatches() {
    std::printf("[decode DAG]\n");
    GptOssGeometry g;
    std::string err;
    if (!geometry_from_facts(Facts{}, g, &err)) return;

    const std::vector<Dispatch> dag = build_gptoss_dag(g, /*with_argmax=*/false);
    const DagStats s = dag_stats(dag, g);

    expect_eq(s.sliding_attn_layers, 12, "half the layers slide");
    expect_eq(s.full_attn_layers, 12, "and half attend everything");

    // Unlike gemma4, nothing here is conditional on the layer: every layer owns
    // its KV and every layer is an MoE, so the per-layer cost is uniform.
    for (Kind k : {Kind::AttnNorm, Kind::QmvQ, Kind::QmvK, Kind::QmvV, Kind::RopeQ,
                   Kind::RopeK, Kind::KvAppend, Kind::SdpaSink, Kind::QmvO,
                   Kind::AttnResidual, Kind::FfnNorm, Kind::RouterGemv, Kind::RouterTopK,
                   Kind::ExpertSort, Kind::ExpertGather, Kind::ExpertGate, Kind::ExpertUp,
                   Kind::ExpertSwiGlu, Kind::ExpertDown, Kind::ExpertCombine,
                   Kind::FfnResidual}) {
        expect_eq(count(dag, k), g.n_layers, "one per layer");
    }

    expect_eq(count(dag, Kind::EmbedGather), 1, "one embedding gather");
    expect_eq(count(dag, Kind::RowGather), 1,
              "the sampled rows are compacted once, before the tail");
    expect_eq(count(dag, Kind::FinalRms), 1, "one final norm");
    expect_eq(count(dag, Kind::LmHead), 1, "one logits matvec");
    expect_eq(count(dag, Kind::Argmax), 0, "argmax is opt-in");

    const int per_layer = 21;
    // Tail: the row gather, the final norm and the LM head.
    expect_eq(s.total, 1 + g.n_layers * per_layer + 3,
              "the whole step is exactly this many dispatches");

    // Three of every layer's matvecs read weights the ROUTER picked, which is
    // what stops them being a plain qmv: the operand is not known host-side.
    expect_eq(s.routed, 3 * g.n_layers, "three routed matmuls a layer");
    // Eight a layer: q/k/v/o, the router, and the three routed expert
    // projections -- plus the head.
    expect_eq(s.gemv, 1 + g.n_layers * 8, "and this many matvecs in total");

    // Ordinals are dense and in order -- they are the argument-table keys.
    bool dense = true;
    for (std::size_t i = 0; i < dag.size(); ++i) {
        dense = dense && dag[i].ordinal == static_cast<int>(i);
    }
    expect(dense, "ordinals are a dense 0..N-1 run");

    bool typed = true;
    for (const Dispatch& d : dag) {
        if (d.layer >= 0) typed = typed && d.sliding == g.is_sliding(d.layer);
    }
    expect(typed, "each dispatch carries its layer's attention type");

    // The prefill DAG is the decode DAG with its ordinals shifted clear -- the
    // dispatch LIST does not depend on the batch size. gemma4's test caught the
    // two builders' `with_argmax` defaults disagreeing within a minute of
    // existing, so this one is written the same way.
    const std::vector<Dispatch> mb = build_gptoss_dag_mb(g, 100000, /*with_argmax=*/false);
    expect_eq(int(mb.size()), int(dag.size()), "the prefill DAG is the same dispatch list");
    bool same = true;
    for (std::size_t i = 0; i < mb.size(); ++i) {
        same = same && mb[i].kind == dag[i].kind && mb[i].layer == dag[i].layer &&
               mb[i].sliding == dag[i].sliding && mb[i].ordinal == dag[i].ordinal + 100000;
    }
    expect(same, "differing only by the ordinal base");
}

void argmax_is_opt_in() {
    std::printf("[optional stages]\n");
    GptOssGeometry g;
    std::string err;
    if (!geometry_from_facts(Facts{}, g, &err)) return;
    const std::vector<Dispatch> with = build_gptoss_dag(g, /*with_argmax=*/true);
    const std::vector<Dispatch> without = build_gptoss_dag(g, /*with_argmax=*/false);
    expect_eq(count(with, Kind::Argmax), 1, "the sampler is one more dispatch");
    expect_eq(int(with.size()), int(without.size()) + 1, "and only one");
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("gptoss_decode_step_test\n");
    the_geometry_is_the_checkpoints();
    the_step_is_this_many_dispatches();
    argmax_is_opt_in();
    std::printf("\n==== gptoss_decode_step_test: %s ====\n",
                g_failures == 0 ? "all passed" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}
