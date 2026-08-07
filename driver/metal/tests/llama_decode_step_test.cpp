// The llama families' DAG, as a shape.
//
// `build_llama_dag` is a pure function of the geometry, so this test needs no
// Metal and no checkpoint — which is the point. The whole reason llama_like and
// qwen3_moe are ONE family rather than two is that the difference between them
// is three fields (`qk_norm`, `n_experts`, `experts_per_token`) and one branch
// in this function. If that claim is wrong it is wrong here, cheaply, rather
// than in an encoder against a 30B checkpoint.
//
// So what is pinned is exactly the four configurations the family exists to
// serve, and the arithmetic relating them:
//
//   llama-3          dense, no qk-norm     — the baseline
//   qwen3            dense, qk-norm        — +2 per layer
//   qwen3-moe        routed, qk-norm       — dense FFN's 4 become 7
//   mixtral-shaped   routed, no qk-norm    — proving the two axes are independent

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "batch/forward.hpp"
#include "model/llama/decode_step.hpp"
#include "model/llama/kernels.hpp"
#include "model/llama/bind.hpp"
#include "model/llama/decode_consts.hpp"
#include "model/llama/encode.hpp"
#include "model/llama/scratch.hpp"
#include "batch/simple_family.hpp"
#include "model/qwen3_5/decode_dispatch_mb.hpp"

using pie::metal::llama::Dispatch;
using pie::metal::llama::Kind;
using pie::metal::llama::LlamaGeometry;
using pie::metal::llama::build_llama_dag;
using pie::metal::llama::build_llama_dag_mb;
using pie::metal::llama::ScratchPlan;
using pie::metal::llama::Use;
using pie::metal::llama::ValueExtent;
using pie::metal::llama::build_llama_scratch;
using pie::metal::llama::dag_stats;
using pie::metal::llama::llama_value_extents;
using pie::metal::Kernel;
using pie::metal::llama::KN;
using pie::metal::llama::color_llama_scratch;
using pie::metal::llama::llama_run_ends;
using pie::metal::llama::pso_kind;
using pie::metal::llama::qmv_kn;
using pie::metal::llama::shared_kind;

namespace {

int g_pass = 0, g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}

void expect_eq(long long got, long long want, const std::string& what) {
    if (got == want) { ++g_pass; std::printf("  PASS  %s (%lld)\n", what.c_str(), got); }
    else {
        ++g_fail;
        std::printf("  FAIL  %s: got %lld, want %lld\n", what.c_str(), got, want);
    }
}

LlamaGeometry base() {
    LlamaGeometry g;
    g.n_layers = 4;
    g.hidden = 2048;
    g.vocab = 128256;
    g.n_q_heads = 16;
    g.n_kv_heads = 4;
    g.head_dim = 128;
    g.intermediate = 8192;
    return g;
}

int count(const std::vector<Dispatch>& dag, Kind k) {
    int n = 0;
    for (const Dispatch& d : dag) if (d.kind == k) ++n;
    return n;
}

/// Ordinals must be dense, ordered and unique: they are the argument-table key,
/// and a collision silently binds one dispatch's weights to another's.
void check_ordinals(const std::vector<Dispatch>& dag, int base_ord, const std::string& who) {
    bool ok = true;
    for (std::size_t i = 0; i < dag.size(); ++i) {
        if (dag[i].ordinal != base_ord + static_cast<int>(i)) ok = false;
    }
    expect(ok, who + ": ordinals dense and ordered from " + std::to_string(base_ord));
}

/// Every dispatch inside the body must name its layer, and everything outside
/// it must not. A layer index leaking into the tail would make the LM head read
/// a per-layer weight slot.
void check_layer_tags(const std::vector<Dispatch>& dag, const LlamaGeometry& g,
                      const std::string& who) {
    bool ok = true;
    std::map<int, int> per_layer;
    for (const Dispatch& d : dag) {
        const bool body = d.kind != Kind::EmbedGather && d.kind != Kind::RowGather &&
                          d.kind != Kind::FinalRms && d.kind != Kind::LmHead &&
                          d.kind != Kind::Argmax;
        if (body) {
            if (d.layer < 0 || d.layer >= g.n_layers) ok = false;
            else ++per_layer[d.layer];
        } else if (d.layer != -1) {
            ok = false;
        }
    }
    expect(ok, who + ": layer tags in range for the body, -1 for embed and tail");
    bool uniform = per_layer.size() == std::size_t(g.n_layers);
    for (const auto& [layer, n] : per_layer) {
        if (n != per_layer.begin()->second) uniform = false;
    }
    expect(uniform, who + ": every layer emits the same dispatch count");
}

/// The per-layer cost, derived rather than transcribed, so a reader can check
/// the expectation against the model instead of against the code.
int expected_per_layer(const LlamaGeometry& g) {
    int n = 0;
    n += 1;                    // AttnNorm
    n += 3;                    // Q, K, V
    n += g.qk_norm ? 2 : 0;    // QNorm, KNorm
    n += 2;                    // RopeQ, RopeK
    n += 1;                    // KvAppend
    n += 1;                    // Sdpa
    n += 1;                    // QmvO
    n += 1;                    // AttnResidual
    n += 1;                    // FfnNorm
    // router+topk+sort+gather+gate+up+silu+down+combine, or the dense
    // gate+up+silu+down. The two reordering stages are what let the routed
    // projections be matmuls: they put the rows that share an expert next to
    // each other. Nothing puts them back -- the combine reads them where they
    // lie, through the sort's inverse.
    n += g.is_moe() ? 9 : 4;
    n += 1;                    // FfnResidual
    return n;
}

void check_family(const char* who, const LlamaGeometry& g) {
    std::printf("\n-- %s --\n", who);
    const std::vector<Dispatch> dag = build_llama_dag(g);

    const int head = 1;               // EmbedGather
    const int tail = 4;               // RowGather, FinalRms, LmHead, Argmax
    const int per_layer = expected_per_layer(g);
    expect_eq(static_cast<long long>(dag.size()),
              head + static_cast<long long>(g.n_layers) * per_layer + tail,
              std::string(who) + ": total dispatches");

    check_ordinals(dag, 0, who);
    check_layer_tags(dag, g, who);

    expect_eq(count(dag, Kind::EmbedGather), 1, std::string(who) + ": one embedding");
    expect_eq(count(dag, Kind::Argmax), 1, std::string(who) + ": one argmax");
    expect_eq(count(dag, Kind::KvAppend), g.n_layers, std::string(who) + ": one kv append/layer");
    expect_eq(count(dag, Kind::QNorm), g.qk_norm ? g.n_layers : 0,
              std::string(who) + ": qk-norm presence follows the geometry");
    expect_eq(count(dag, Kind::QNorm), count(dag, Kind::KNorm),
              std::string(who) + ": q and k norms are paired");

    // Dense and routed are exclusive: emitting both would run the FFN twice and
    // the second would overwrite the first.
    const int dense = count(dag, Kind::QmvGate);
    const int routed = count(dag, Kind::ExpertGate);
    expect(dense == 0 || routed == 0, std::string(who) + ": FFN is dense XOR routed");
    expect_eq(g.is_moe() ? routed : dense, g.n_layers,
              std::string(who) + ": one FFN per layer, of the right kind");
    expect_eq(count(dag, Kind::ExpertCombine), g.is_moe() ? g.n_layers : 0,
              std::string(who) + ": expert combine only when routed");
    expect_eq(count(dag, Kind::RouterTopK), count(dag, Kind::Router),
              std::string(who) + ": every router logit vector gets a top-k");

    // Two residuals per layer: attention and FFN. A missing one is a model that
    // still runs and still produces text.
    expect_eq(count(dag, Kind::AttnResidual) + count(dag, Kind::FfnResidual), 2 * g.n_layers,
              std::string(who) + ": two residual adds per layer");

    const auto s = dag_stats(dag, g);
    const int gemv_per_layer = 4 + (g.is_moe() ? 4 : 3);  // qkvo + (router,g,u,d) or (g,u,d)
    expect_eq(s.gemv, gemv_per_layer * g.n_layers + 1, std::string(who) + ": matvec count");
    expect_eq(s.routed, g.is_moe() ? 3 * g.n_layers : 0,
              std::string(who) + ": routed matvec count");

    // The tail is what a prefill saves: everything from RowGather on runs on the
    // sampled rows, so it must come after the last layer, exactly once, in order.
    std::vector<Kind> tail_kinds;
    for (std::size_t i = dag.size() - 4; i < dag.size(); ++i) tail_kinds.push_back(dag[i].kind);
    expect(tail_kinds == std::vector<Kind>{Kind::RowGather, Kind::FinalRms, Kind::LmHead,
                                           Kind::Argmax},
           std::string(who) + ": tail is gather, norm, head, argmax");

    // The MB path must describe the SAME model — only the ordinals move.
    const std::vector<Dispatch> mb = build_llama_dag_mb(g, 50000);
    expect_eq(static_cast<long long>(mb.size()), static_cast<long long>(dag.size()),
              std::string(who) + ": MB dag has the same length");
    bool same_shape = true;
    for (std::size_t i = 0; i < mb.size(); ++i) {
        if (mb[i].kind != dag[i].kind || mb[i].layer != dag[i].layer) same_shape = false;
    }
    expect(same_shape, std::string(who) + ": MB dag is the decode dag, ordinals aside");
    check_ordinals(mb, 50000, std::string(who) + " MB");
}


/// The dataflow's own invariants. These are the ones that fail silently: a
/// value read before it is written reads whatever the pool last left there, and
/// the model still produces fluent text.
void check_scratch(const char* who, const LlamaGeometry& g) {
    std::printf("\n-- scratch: %s --\n", who);
    const std::vector<Dispatch> dag = build_llama_dag(g);
    const ScratchPlan plan = build_llama_scratch(dag, g);

    // Uses are emitted in dispatch order, which the colouring relies on to
    // treat `index` as a time axis.
    bool ordered = true;
    for (std::size_t i = 1; i < plan.uses.size(); ++i) {
        if (plan.uses[i].index < plan.uses[i - 1].index) ordered = false;
    }
    expect(ordered, std::string(who) + ": uses are in dispatch order");

    bool in_range = true;
    for (const Use& u : plan.uses) {
        if (u.value < 0 || u.value >= plan.value_count) in_range = false;
        if (u.index < 0 || u.index >= static_cast<int>(dag.size())) in_range = false;
    }
    expect(in_range, std::string(who) + ": every use names a real value and dispatch");

    // Single assignment, except for the deliberate in-place passes. Two
    // producers for one value means the second silently clobbers the first.
    std::map<int, std::vector<int>> writers;
    for (const Use& u : plan.uses) {
        if (u.is_write) writers[u.value].push_back(u.index);
    }
    bool inplace_only = true;
    for (const auto& [value, ws] : writers) {
        if (ws.size() == 1) continue;
        // The FIRST write is the value's producer. Every write AFTER it is
        // legal only if the same dispatch also reads the value -- that is what
        // "in place" means, and rope and the qk-norms are the only dispatches
        // that do it.
        for (std::size_t wi = 1; wi < ws.size(); ++wi) {
            const int w = ws[wi];
            bool reads_too = false;
            for (const Use& u : plan.uses) {
                if (u.index == w && !u.is_write && u.value == value) reads_too = true;
            }
            const Kind k = dag[std::size_t(w)].kind;
            const bool expected_inplace = k == Kind::RopeQ || k == Kind::RopeK ||
                                          k == Kind::QNorm || k == Kind::KNorm;
            if (!reads_too || !expected_inplace) inplace_only = false;
        }
    }
    expect(inplace_only, std::string(who) + ": repeated writes are in-place rope/qk-norm only");

    // No read before write. This is the one that matters.
    std::map<int, int> first_write;
    for (const Use& u : plan.uses) {
        if (u.is_write && first_write.find(u.value) == first_write.end()) {
            first_write[u.value] = u.index;
        }
    }
    bool defined = true;
    int undefined_value = -1;
    for (const Use& u : plan.uses) {
        if (u.is_write) continue;
        auto it = first_write.find(u.value);
        if (it == first_write.end() || it->second > u.index) {
            defined = false;
            undefined_value = u.value;
        }
    }
    if (!defined) std::printf("        value %d read before written\n", undefined_value);
    expect(defined, std::string(who) + ": every read follows the value's write");

    // Every value produced is consumed. A dead value is a dispatch whose result
    // nothing needs -- which on this DAG means a wiring mistake, not a saving.
    std::map<int, int> reads;
    for (const Use& u : plan.uses) {
        if (!u.is_write) ++reads[u.value];
    }
    int dead = 0;
    for (const auto& [value, ws] : writers) {
        (void)ws;
        if (reads.find(value) == reads.end()) ++dead;
    }
    // The logits are read by the argmax, so nothing should be dead at all.
    expect_eq(dead, 0, std::string(who) + ": no value is written and never read");

    expect(plan.logits_value >= 0, std::string(who) + ": the logits value is identified");

    // The routing decision must be live ACROSS the expert matvecs, not recycled
    // between them. This is the specific thing a naive colouring gets wrong.
    if (g.is_moe()) {
        // One routing decision per MIXTURE LAYER, in DAG order, and all
        // distinct: the host rewrites the one the next segment will read when
        // the experts are paged, so a stack that reported a single value would
        // let it rewrite a buffer nothing reads.
        expect_eq(int(plan.expert_ids_by_layer.size()), g.n_layers,
                  std::string(who) + ": one routing decision per mixture layer");
        std::set<int> distinct(plan.expert_ids_by_layer.begin(), plan.expert_ids_by_layer.end());
        expect_eq(int(distinct.size()), g.n_layers,
                  std::string(who) + ": no two layers share a routing value");
        // Per layer: written once by RouterTopK, read three times by the
        // expert matvecs (ids) and once by the combine (weights).
        int id_reads = 0, weight_reads = 0;
        for (const Use& u : plan.uses) {
            if (u.is_write) continue;
            const Kind k = dag[std::size_t(u.index)].kind;
            if (k == Kind::ExpertGate || k == Kind::ExpertUp || k == Kind::ExpertDown) {
                if (u.bind_index == 8) ++id_reads;
            }
            if (k == Kind::ExpertCombine && u.bind_index == 1) ++weight_reads;
        }
        expect_eq(id_reads, 3 * g.n_layers,
                  std::string(who) + ": each expert matvec reads the router's ids");
        expect_eq(weight_reads, g.n_layers,
                  std::string(who) + ": the combine reads the router's weights");
    } else {
        expect(plan.expert_ids_by_layer.empty(),
               std::string(who) + ": a dense model has no routing values");
    }

    // Extents. The expert stack is the only place a value is k times taller,
    // and getting that wrong overruns the pool by exactly that factor.
    const std::vector<ValueExtent> ext = llama_value_extents(dag, plan, g);
    expect_eq(static_cast<long long>(ext.size()), plan.value_count,
              std::string(who) + ": one extent per value");
    expect_eq(ext[std::size_t(plan.logits_value)].elems, g.vocab,
              std::string(who) + ": the logits are vocab-wide");

    int slotted = 0;
    int sorted = 0;
    for (const ValueExtent& e : ext) {
        if (e.rows_are_slots != 0) ++slotted;
        if (e.rows_are_sorted != 0) ++sorted;
    }
    // NOTHING is in slot order any more. The routed run enters the sorted
    // layout at the gather and leaves it at the combine, which reads through
    // the sort's inverse rather than after a kernel has undone it -- so the
    // [rows, k, hidden] stack that used to sit between the two is gone.
    expect_eq(slotted, 0, std::string(who) + ": nothing is slot-major any more");
    // The sort's four index outputs are one value each -- perm, row_expert,
    // tile_expert, inv -- plus the gather's rows, gate, up, silu-out and
    // down-out.
    expect_eq(sorted, g.is_moe() ? 9 * g.n_layers : 0,
              std::string(who) + ": the sorted stack runs from the gather to the combine");

    bool sized = true;
    for (std::size_t v = 0; v < ext.size(); ++v) {
        // KvAppend and Argmax write nothing, so no value can come back zero.
        if (ext[v].elems <= 0) sized = false;
    }
    expect(sized, std::string(who) + ": every value has a positive width");
}


/// The colouring, and the one property that makes it usable: hazard freedom.
///
/// A colouring that recycles a buffer while a value still lives in it is not a
/// crash. It is an activation quietly replaced by a later one, at a point where
/// the model still produces fluent text -- so it has to be checked rather than
/// observed.
void check_coloring(const char* who, const LlamaGeometry& g) {
    std::printf("\n-- coloring: %s --\n", who);
    const std::vector<Dispatch> dag = build_llama_dag(g);
    const ScratchPlan plan = build_llama_scratch(dag, g);
    const auto colored = color_llama_scratch(dag, plan);

    expect(colored.hazard_free, std::string(who) + ": colouring is hazard-free");
    expect(colored.colors_used > 0 && colored.colors_used <= plan.value_count,
           std::string(who) + ": colours are used and never exceed the value count");
    // Recycling has to actually happen, or the "pool" is just the value list.
    expect(colored.colors_used < plan.value_count,
           std::string(who) + ": buffers are reused across the stack");
    expect_eq(static_cast<long long>(colored.per_dispatch.size()),
              static_cast<long long>(dag.size()), std::string(who) + ": one entry per dispatch");

    bool assigned = true;
    for (const auto& binds : colored.per_dispatch) {
        for (const auto& sb : binds) {
            if (sb.color < 0 || sb.color >= colored.colors_used) assigned = false;
        }
    }
    expect(assigned, std::string(who) + ": every bind resolved to a real colour");

    // The concurrency runs the colourer was told about must be the ones the
    // encoder will actually drop barriers for, or the two disagree about when a
    // value dies. Runs only ever merge same-layer neighbours.
    const std::vector<int> ends = llama_run_ends(dag);
    bool runs_ok = ends.size() == dag.size();
    int merged = 0;
    for (std::size_t i = 0; i < ends.size(); ++i) {
        if (ends[i] < static_cast<int>(i) || ends[i] >= static_cast<int>(dag.size())) {
            runs_ok = false;
        }
        if (ends[i] > static_cast<int>(i)) {
            ++merged;
            for (int j = static_cast<int>(i); j <= ends[i]; ++j) {
                if (dag[std::size_t(j)].layer != dag[i].layer) runs_ok = false;
            }
        }
    }
    expect(runs_ok, std::string(who) + ": runs are forward and stay inside one layer");
    expect(merged > 0, std::string(who) + ": some dispatches share a barrier");
}

/// The kernel mapping and the constants.
void check_encode(const char* who, const LlamaGeometry& g) {
    std::printf("\n-- encode: %s --\n", who);
    const std::vector<Dispatch> dag = build_llama_dag(g);

    // The weight key and the pipeline are DIFFERENT questions, and the routed
    // matvecs are where they disagree: they bind the expert stack but run the
    // routed kernel. A binder that used `pso_kind` would ask for the wrong
    // tensor names entirely.
    expect(shared_kind(Kind::ExpertGate, g) == Kernel::LlExpertGate,
           std::string(who) + ": ExpertGate's weight key is the expert stack");
    expect(pso_kind(Kind::ExpertGate) == Kernel::LlExpertGate,
           std::string(who) + ": ExpertGate runs the routed pipeline");
    expect(pso_kind(Kind::QmvGate) != pso_kind(Kind::ExpertGate),
           std::string(who) + ": the dense and routed matvecs are different pipelines");

    // Tied vs untied is a property of the CHECKPOINT, and asking for the wrong
    // pair is a load failure rather than a wrong number.
    LlamaGeometry tied = g;
    tied.tied_embeddings = true;
    LlamaGeometry untied = g;
    untied.tied_embeddings = false;
    expect(shared_kind(Kind::EmbedGather, tied) == Kernel::EmbedGather &&
               shared_kind(Kind::LmHead, tied) == Kernel::QmvLmHead,
           std::string(who) + ": a tied model reads one table at both ends");
    expect(shared_kind(Kind::EmbedGather, untied) == Kernel::EmbedUntied &&
               shared_kind(Kind::LmHead, untied) == Kernel::LmHeadUntied,
           std::string(who) + ": an untied model reads two separate tensors");

    // Matvec shapes. The routed ones have the SAME K and N as their dense
    // counterparts -- the expert axis is a stride into the weight stack, not a
    // wider matrix -- and that is the fact a reader is most likely to doubt.
    expect_eq(qmv_kn(Kind::QmvQ, g).N, g.q_width(), std::string(who) + ": q_proj is q_width");
    expect_eq(qmv_kn(Kind::QmvK, g).N, g.kv_width(), std::string(who) + ": k_proj is kv_width");
    expect_eq(qmv_kn(Kind::QmvO, g).K, g.q_width(), std::string(who) + ": o_proj reads q_width");
    expect_eq(qmv_kn(Kind::LmHead, g).N, g.vocab, std::string(who) + ": the head is vocab-wide");
    if (g.is_moe()) {
        expect_eq(qmv_kn(Kind::Router, g).N, g.n_experts,
                  std::string(who) + ": the router is n_experts wide");
        expect_eq(qmv_kn(Kind::ExpertGate, g).N, g.moe_intermediate,
                  std::string(who) + ": an expert's gate is the per-expert width");
        expect_eq(qmv_kn(Kind::ExpertDown, g).K, g.moe_intermediate,
                  std::string(who) + ": an expert's down reads the per-expert width");
        expect_eq(qmv_kn(Kind::ExpertGate, g).K, g.hidden,
                  std::string(who) + ": an expert reads the full hidden vector");
    }

    // Every dispatch must resolve to some pipeline; a kind that fell through to
    // a default would run the wrong kernel with plausible-looking arguments.
    bool mapped = true;
    for (const Dispatch& d : dag) {
        if (d.kind == Kind::EmbedGather) continue;
        const Kernel pk = pso_kind(d.kind);
        const Kernel wk = shared_kind(d.kind, g);
        if (pk == Kernel::Rms && d.kind != Kind::AttnNorm && d.kind != Kind::FfnNorm &&
            d.kind != Kind::FinalRms && d.kind != Kind::QNorm && d.kind != Kind::KNorm) {
            mapped = false;  // the fall-through value, reached by something else
        }
        (void)wk;
    }
    expect(mapped, std::string(who) + ": no kind falls through to the default pipeline");
}

}  // namespace


/// Pool sizing. The one place a colouring pass cannot be trusted on its own.
///
/// A colour is a buffer several values share, so it must be sized by the widest
/// of them -- and on a routed model "widest" is not a property of any dispatch's
/// kind. `ExpertGate` writes `[rows * k, moe_intermediate]` where the dense
/// `QmvGate` beside it writes `[rows, intermediate]`. If the two land in one
/// colour and it is sized by kind, the last k-1 slots of the expert stack are
/// written past the end of the buffer -- into whatever colour was allocated
/// next, which is another live activation.
void check_pool(const char* who, const LlamaGeometry& g) {
    std::printf("\n-- pool: %s --\n", who);
    const std::vector<Dispatch> dag = build_llama_dag(g);
    const ScratchPlan plan = build_llama_scratch(dag, g);
    const auto colored = color_llama_scratch(dag, plan);
    const std::vector<std::size_t> elems = llama_pool_elems(dag, plan, colored, g);

    expect_eq(static_cast<long long>(elems.size()),
              static_cast<long long>(colored.colors_used),
              std::string(who) + ": one size per colour");
    expect_eq(static_cast<long long>(colored.color_of_value.size()),
              static_cast<long long>(plan.value_count),
              std::string(who) + ": every value has a colour");

    bool nonzero = true;
    for (const std::size_t e : elems) {
        if (e == 0) nonzero = false;
    }
    expect(nonzero, std::string(who) + ": no colour is sized zero");

    // The claim, stated directly: every value fits in the colour it was put in.
    const std::vector<ValueExtent> ext = llama_value_extents(dag, plan, g);
    const int k = g.is_moe() ? g.experts_per_token : 1;
    bool fits = true;
    for (int v = 0; v < plan.value_count; ++v) {
        const int c = colored.color_of_value[std::size_t(v)];
        if (c < 0) continue;
        const ValueExtent& e = ext[std::size_t(v)];
        const std::size_t need =
            std::size_t(e.rows_are_slots != 0 ? k : 1) * std::size_t(e.elems);
        if (need > elems[std::size_t(c)]) fits = false;
    }
    expect(fits, std::string(who) + ": every value fits the colour it landed in");

    // The logits are the widest thing in the model by two orders of magnitude,
    // so a pool that has not noticed them is a pool sized by the body alone.
    std::size_t widest = 0;
    for (const std::size_t e : elems) widest = std::max(widest, e);
    expect_eq(static_cast<long long>(widest), static_cast<long long>(g.vocab),
              std::string(who) + ": the widest colour is the LM head's");
}

/// The routed pool, against the dense one at the same shape.
///
/// A discriminator rather than an invariant: it fails if `rows_are_slots` is
/// ignored, which is the specific mistake that costs the last expert its
/// buffer. Both models have the same hidden size and the same expert width, so
/// the only thing that can produce a difference is the slot axis.
void check_pool_counts_the_slots() {
    std::printf("\n-- pool: the expert stack is k times taller --\n");
    LlamaGeometry dense = base();
    dense.intermediate = 768;

    LlamaGeometry routed = base();
    routed.n_experts = 128;
    routed.experts_per_token = 8;
    routed.moe_intermediate = 768;

    const auto sized = [](const LlamaGeometry& g) {
        const std::vector<Dispatch> dag = build_llama_dag(g);
        const ScratchPlan plan = build_llama_scratch(dag, g);
        return llama_pool_elems(dag, plan, color_llama_scratch(dag, plan), g);
    };
    const std::vector<std::size_t> d = sized(dense);
    const std::vector<std::size_t> r = sized(routed);

    const auto has = [](const std::vector<std::size_t>& v, std::size_t want) {
        for (const std::size_t e : v) {
            if (e == want) return true;
        }
        return false;
    };
    expect(has(d, 768), "the dense FFN asks for one 768-wide buffer");
    expect(has(r, 768 * 8), "the routed FFN asks for an 8-slot stack of them");

    std::size_t dsum = 0, rsum = 0;
    for (const std::size_t e : d) dsum += e;
    for (const std::size_t e : r) rsum += e;
    expect(rsum > dsum, "the routed pool is strictly larger at the same widths");
}

// What the geometry REFUSES, which is the half of a contract tests usually
// skip. Every one of these configs describes a model the kernels would happily
// run and get quietly wrong -- not crash, not produce garbage, just return
// plausible tokens from the wrong arithmetic. That is the failure mode a
// refusal exists to prevent, so it is worth pinning that each one still fires.
// (The shared-expert/per-expert refusal that lived here is pinned in Rust
// now — `model/common/src/mlx.rs::routed_banks_map_and_unservable_experts_are_refused` —
// beside the `routed_expert_member` rule it checks, which is where the name
// normalisation went when the C++ author died.)

/// Llama 3.1's rope schedule, on Llama-3.2-1B's own numbers.
///
/// Three bands, and the whole point of the schedule is that they differ. The
/// fast channels are left alone so short-range structure is unchanged; the slow
/// ones are divided by the whole factor so the context stretches; between them
/// a ramp. A table that scaled everything -- which is what approximating this
/// with `rope_scale` does -- would fail the first and third of these together.
void check_llama3_schedule() {
    using Facts = pie::metal::batch::SetupConfig::LlamaFacts;
    Facts f;
    f.n_layers = 16; f.hidden = 2048; f.vocab = 128256;
    f.n_q_heads = 32; f.n_kv_heads = 8; f.head_dim = 64;
    f.intermediate = 8192;
    f.rope_theta = 500000.0f;
    f.rope_scaling_kind = "llama3";
    f.rope_scale = 32.0f;
    f.rope_low_freq_factor = 1.0f;
    f.rope_high_freq_factor = 4.0f;
    f.rope_original_max_position = 8192;

    LlamaGeometry g;
    std::string err;
    if (!pie::metal::llama::geometry_from_facts(f, g, &err)) {
        expect(false, "Llama-3.2-1B's llama3 geometry builds [" + err + "]");
        return;
    }
    expect(g.rope_freq_table, "llama3 asks for a frequency table");
    // The factor divides frequencies inside the table. Leaving it in
    // `rope_scale` too would divide the position by it as well.
    expect(g.rope_scaling_factor == 32.0f, "the factor is carried as the schedule's own");
    expect(g.rope_scale == 1.0f, "and is NOT left in the linear position scale");

    const std::vector<float> table = pie::metal::llama::llama3_inv_freq(g);
    expect(int(table.size()) == g.rotary_dims() / 2,
           "the table has one entry per rotated pair");

    const auto base_inv = [&](int i) {
        return 1.0f / std::pow(g.rope_theta, float(2 * i) / float(g.rotary_dims()));
    };
    const auto close = [](float a, float b) {
        return std::abs(a - b) <= 1e-4f * std::max(1.0f, std::abs(b));
    };

    // i = 0: wavelength 2*pi, far below orig/high_freq_factor = 2048. Untouched.
    expect(close(table[0], base_inv(0)), "the fastest channel is left alone");
    // The last channel's wavelength is 2*pi*500000^(62/64) ~ 2.6e6, far above
    // orig/low_freq_factor = 8192. Divided by the whole factor.
    const int last = int(table.size()) - 1;
    expect(close(table[std::size_t(last)], base_inv(last) / 32.0f),
           "the slowest channel is divided by the whole factor");

    // And somewhere between them a channel that is neither: strictly slower
    // than untouched, strictly faster than fully scaled. Without the ramp this
    // would have to be one or the other.
    int ramped = -1;
    for (int i = 0; i <= last; ++i) {
        const float t = table[std::size_t(i)];
        if (!close(t, base_inv(i)) && !close(t, base_inv(i) / 32.0f)) { ramped = i; break; }
    }
    expect(ramped > 0, "a band between them is interpolated, not switched");
    if (ramped > 0) {
        const float t = table[std::size_t(ramped)];
        expect(t < base_inv(ramped) && t > base_inv(ramped) / 32.0f,
               "and it lands between the two bands it interpolates");
    }

    // The table is monotone: a slower channel is never given a faster rate.
    bool monotone = true;
    for (int i = 1; i <= last; ++i) {
        if (table[std::size_t(i)] > table[std::size_t(i - 1)]) monotone = false;
    }
    expect(monotone, "and no channel is turned faster than a faster one");
}

void check_refusals() {
    using Facts = pie::metal::batch::SetupConfig::LlamaFacts;
    const auto moe = [] {
        Facts f;
        f.n_layers = 48; f.hidden = 2048; f.vocab = 151936;
        f.n_q_heads = 32; f.n_kv_heads = 4; f.head_dim = 128;
        f.n_experts = 128; f.experts_per_token = 8; f.moe_intermediate = 768;
        return f;
    };
    const auto refused = [](const Facts& f, const std::string& fragment,
                            const std::string& what) {
        LlamaGeometry g;
        std::string err;
        const bool ok = pie::metal::llama::geometry_from_facts(f, g, &err);
        expect(!ok && err.find(fragment) != std::string::npos,
               what + (ok ? " [ACCEPTED]" : " [" + err + "]"));
    };
    const auto accepted = [](const Facts& f, const std::string& what) {
        LlamaGeometry g;
        std::string err;
        expect(pie::metal::llama::geometry_from_facts(f, g, &err), what + " [" + err + "]");
    };

    // Qwen3-30B-A3B exactly: 128 experts, top-8, renormalized. The baseline the
    // refusals below are each one field away from.
    accepted(moe(), "Qwen3-MoE's own shape is accepted");

    // The router softmaxes the SELECTED logits. A config asking for the softmax
    // over all of them wants weights summing to less than one, which scales the
    // FFN's contribution down -- same tokens, quieter. Qwen1.5-MoE is this.
    Facts unnormalized = moe();
    unnormalized.norm_topk_prob = false;
    refused(unnormalized, "norm_topk_prob", "norm_topk_prob=false is refused");

    // The kernel holds the chosen logits in a fixed threadgroup array and
    // clamps k to it, while every consumer keeps striding by the configured k.
    Facts wide_k = moe();
    wide_k.experts_per_token = pie::metal::shared_kernels::kRouterMaxTopK + 1;
    refused(wide_k, "top-k limit", "experts_per_token past the router's top-k is refused");
    Facts max_k = moe();
    max_k.experts_per_token = pie::metal::shared_kernels::kRouterMaxTopK;
    accepted(max_k, "experts_per_token exactly at the top-k limit is accepted");

    // One lane per expert, one threadgroup per row.
    Facts wide_n = moe();
    wide_n.n_experts = int(pie::metal::shared_kernels::kRouterMaxExperts) + 1;
    refused(wide_n, "single threadgroup can rank",
            "n_experts past a threadgroup is refused");

    // Llama 3.1's piecewise schedule IS implemented, as a frequency table.
    Facts llama3_rope = moe();
    llama3_rope.rope_scaling_kind = "llama3";
    accepted(llama3_rope, "llama3 rope_scaling is accepted");
    // Anything else still is not. Approximating a per-channel schedule with the
    // linear scale runs, and is wrong past the original context length.
    Facts longrope = moe();
    longrope.rope_scaling_kind = "longrope";
    refused(longrope, "rope_scaling", "an unimplemented rope_scaling is still refused");

    // The two degenerate schedules. `lo == hi` divides by zero in the ramp;
    // a non-positive original context makes every wavelength comparison
    // vacuous. Both produce a table of plausible-looking wrong angles.
    Facts flat_band = moe();
    flat_band.rope_scaling_kind = "llama3";
    flat_band.rope_low_freq_factor = 4.0f;
    flat_band.rope_high_freq_factor = 4.0f;
    refused(flat_band, "low_freq_factor != high_freq_factor",
            "llama3 with no ramp between its bands is refused");
    Facts no_orig = moe();
    no_orig.rope_scaling_kind = "llama3";
    no_orig.rope_original_max_position = 0;
    refused(no_orig, "positive original_max_position",
            "llama3 without an original context length is refused");

    check_llama3_schedule();

    // A routed config missing the half of itself the router needs.
    Facts no_k = moe();
    no_k.experts_per_token = 0;
    refused(no_k, "num_experts_per_tok", "a routed FFN without a top-k is refused");
    Facts no_width = moe();
    no_width.moe_intermediate = 0;
    refused(no_width, "moe_intermediate_size", "a routed FFN without a width is refused");

    // GQA's own requirement.
    Facts ragged = moe();
    ragged.n_kv_heads = 5;
    refused(ragged, "multiple", "n_q_heads not a multiple of n_kv_heads is refused");
}

/// The llama family has BOTH FFN shapes, so its streaming predicate must name
/// both banks.
///
/// It named only the dense one. A routed llama publishes `mlp.experts.gate_proj`
/// and the clause tested for `mlp.gate_proj`, which is not a substring of it, so
/// Qwen3-MoE streamed nothing at all -- while gpt-oss, whose experts have the
/// same name and the same sparse access, streamed most of its checkpoint. The
/// symptom was a 16 GB copy at load and no failure anywhere, because streaming
/// is a residency decision and a model that copies everything still answers.
///
/// Stated against the names the CONTRACT publishes, not against invented ones:
/// the author (`model/common/src/mlx.rs`) normalises both `mlp.switch_mlp.` and the fused HF form
/// onto `mlp.experts.`, and `heap_bind.cpp` hands this predicate the finalized
/// runtime name.
void check_streaming_covers_both_ffn_shapes() {
    using pie::metal::batch::SimpleFamilyEngine;
    using pie::metal::model::ModelFamily;
    const auto p = SimpleFamilyEngine::stream_predicate(true);
    expect(bool(p), "a streaming predicate is offered when the config asks");
    if (!p) return;
    expect(p("layers.3.mlp.experts.gate_proj"), "a routed bank streams");
    expect(p("layers.3.mlp.experts.down_proj"), "including the down projection");
    expect(p("layers.3.mlp.switch_mlp.gate_proj") == false,
           "a name the contract has not finalized onto mlp.experts. does not");
    // The dense FFN is read WHOLE every token, so paging it saves no traffic.
    // It used to stream under this same switch, which made one flag mean two
    // trades -- and disagreed with the CUDA driver, where every use of this
    // config key sits inside a routed expert stack.
    expect(!p("layers.3.mlp.gate_proj"), "a dense FFN does not stream");
    expect(!p("layers.3.mlp.up_proj"), "nor its up projection");
    expect(!p("layers.3.mlp.down_proj"), "nor its down projection");
    expect(!p("layers.3.mlp.experts.gate_proj.bias"),
           "the bias beside an expert stays resident");
    expect(!p("layers.3.mlp.gate.weight"), "the router itself stays resident");
    expect(!p("layers.3.self_attn.q_proj"), "attention stays resident");
    expect(!SimpleFamilyEngine::stream_predicate(false),
           "and nothing streams unless the config asks");
}

/// Why the tiled attention has to be told N.
///
/// `sdpa_paged_decode` gets a grid exactly N threadgroups tall, so a row index
/// that exists is a row that exists. `sdpa_paged_tiled` gets a grid of WHOLE
/// query tiles, which is taller than N whenever N is not a multiple of the
/// tile -- and the rows in the overhang would otherwise load a query and store
/// an attention output past the end of both.
///
/// The numerics test cannot see that: a write past a value's extent is not a
/// number it reads back. So the reason is pinned here instead, as the shape it
/// actually is -- a grid that covers more rows than the fire has, and a bound
/// slot that says how many the fire has.
void check_the_tiled_attention_is_told_its_row_count() {
    using pie::metal::Grid;
    using pie::metal::Threadgroup;
    using pie::metal::kSdpaQueryTile;
    using pie::metal::sdpa_should_tile;
    using pie::metal::llama::launch_shape;

    expect(!sdpa_should_tile(1, 1), "a decode does not tile: one row cannot fill one");
    expect(!sdpa_should_tile(kSdpaQueryTile - 1, 1), "nor does a fire one row short of a tile");
    expect(sdpa_should_tile(kSdpaQueryTile, 1), "a fire that fills a tile does");

    // The row count alone cannot decide this. A fleet of members decoding one
    // token each has the SAME rows as a prefill of the same width, and wants
    // the opposite kernel: the tiled one walks runs of equal request, so with
    // one row per request it stages the tile once per row and leaves all but
    // one simdgroup idle. Measured on llama-1B that is 370 tok/s against 728.
    expect(!sdpa_should_tile(kSdpaQueryTile, kSdpaQueryTile),
           "a fleet of one-row members does not tile, however many members");
    expect(!sdpa_should_tile(8 * kSdpaQueryTile, 8 * kSdpaQueryTile),
           "nor does a bigger fleet of them");
    expect(!sdpa_should_tile(4 * kSdpaQueryTile, kSdpaQueryTile),
           "nor four rows each, which still leaves a run far short of a tile");
    expect(sdpa_should_tile(2 * kSdpaQueryTile, 2),
           "two members prefilling a tile each still tiles");

    LlamaGeometry g = base();
    g.paged_kv_enabled = true;
    const auto dag = build_llama_dag(g, true);
    for (const int rows : {kSdpaQueryTile, kSdpaQueryTile + 1, 3 * kSdpaQueryTile - 1}) {
        for (const auto& d : dag) {
            if (d.kind != Kind::Sdpa) continue;
            Grid grid{};
            Threadgroup tg{};
            launch_shape(d, g, grid, tg, rows, rows);
            const long long covered = static_cast<long long>(grid.y) * kSdpaQueryTile;
            expect(covered >= rows,
                   "the tiled grid covers every row of the fire at " + std::to_string(rows));
            if (rows % kSdpaQueryTile != 0) {
                expect(covered > rows,
                       "and overhangs it when the rows do not fill whole tiles at " +
                           std::to_string(rows));
            }
            break;
        }
    }
}

int main() {
    std::printf("llama_decode_step_test — one family, four configurations\n");

    LlamaGeometry llama3 = base();
    check_family("llama-3 (dense, no qk-norm)", llama3);

    LlamaGeometry qwen3 = base();
    qwen3.qk_norm = true;
    check_family("qwen3 (dense, qk-norm)", qwen3);

    LlamaGeometry qwen3_moe = base();
    qwen3_moe.qk_norm = true;
    qwen3_moe.n_experts = 128;
    qwen3_moe.experts_per_token = 8;
    qwen3_moe.moe_intermediate = 768;
    check_family("qwen3-moe (routed, qk-norm)", qwen3_moe);

    LlamaGeometry moe_no_qknorm = base();
    moe_no_qknorm.n_experts = 8;
    moe_no_qknorm.experts_per_token = 2;
    moe_no_qknorm.moe_intermediate = 14336;
    check_family("routed, no qk-norm", moe_no_qknorm);

    // The two axes are independent, and the arithmetic says so: turning qk-norm
    // on costs 2 per layer whatever the FFN is, and going routed costs 3 per
    // layer whatever the norms are.
    std::printf("\n-- axis independence --\n");
    const auto n = [](const LlamaGeometry& g) {
        return static_cast<long long>(build_llama_dag(g).size());
    };
    expect_eq(n(qwen3) - n(llama3), 2 * llama3.n_layers, "qk-norm costs 2/layer (dense)");
    expect_eq(n(qwen3_moe) - n(moe_no_qknorm), 2 * llama3.n_layers, "qk-norm costs 2/layer (routed)");
    expect_eq(n(moe_no_qknorm) - n(llama3), 5 * llama3.n_layers, "routing costs 5/layer (no qk-norm)");
    expect_eq(n(qwen3_moe) - n(qwen3), 5 * llama3.n_layers, "routing costs 5/layer (qk-norm)");

    // `ffn_width` is the one geometry accessor the encoder trusts to pick
    // between the dense and per-expert intermediate size.
    expect_eq(qwen3_moe.ffn_width(), 768, "moe ffn_width is the per-expert width");
    expect_eq(llama3.ffn_width(), 8192, "dense ffn_width is the intermediate size");

    check_scratch("llama-3", llama3);
    check_scratch("qwen3", qwen3);
    check_scratch("qwen3-moe", qwen3_moe);

    check_coloring("llama-3", llama3);
    check_coloring("qwen3-moe", qwen3_moe);
    check_encode("llama-3", llama3);
    check_encode("qwen3-moe", qwen3_moe);

    check_pool("llama-3", llama3);
    check_pool("qwen3-moe", qwen3_moe);
    check_pool_counts_the_slots();

    // Sampling can be off (the caller wants logits, not a token), and that must
    // remove exactly the argmax.
    expect_eq(static_cast<long long>(build_llama_dag(llama3, false).size()), n(llama3) - 1,
              "with_argmax=false drops exactly one dispatch");

    check_refusals();
    check_streaming_covers_both_ffn_shapes();
    check_the_tiled_attention_is_told_its_row_count();

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
