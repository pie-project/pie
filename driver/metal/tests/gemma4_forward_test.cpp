// Does gemma 4 produce a forward pass on this machine, from the real checkpoint?
//
// Everything before this was structural: the DAG resolves, the shapes are legal,
// the binds cover every slot. None of it says the model computes anything. This
// loads `~/.pie-bench/gemma4-e2b-pie`, runs one token through the decode DAG and
// looks at the logits.
//
// Skipped (not failed) when the checkpoint is absent, so CI without a 2.5 GB
// download stays green while the machine that has it gets the real answer.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "mtl4_context.hpp"
#include "batch/decode_abi.hpp"
#include "batch/golden_tap.hpp"
#include "kernels/decode_psos.hpp"
#include "loader/heap_bind_metal.hpp"
#include "loader/load_plan.hpp"
#include "pie_loader/checkpoint_source.hpp"
#include "model/contract.hpp"
#include "model/gemma4/bind.hpp"
#include "model/gemma4/decode_consts.hpp"
#include "model/gemma4/decode_step.hpp"
#include "model/gemma4/encode.hpp"
#include "model/gemma4/geometry.hpp"
#include "model/gemma4/kernels.hpp"
#include "model/gemma4/scratch.hpp"

using namespace pie::metal;
using namespace pie::metal::gemma4;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

float from_bf16(std::uint16_t h) {
    const std::uint32_t bits = std::uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// E2B's shape, which `geometry_from_facts` is separately tested against.
struct Facts {
    int n_layers = 35, hidden = 1536, intermediate = 6144;
    int n_q_heads = 8, n_kv_heads = 1, head_dim = 256, global_head_dim = 512;
    int sliding_window = 512, num_kv_shared_layers = 20, per_layer_emb_dim = 256;
    int full_attn_interval = 5;
    bool double_wide_mlp = true;
    float final_softcap = 30.0f;
    float rope_theta_full = 1.0e6f, rope_theta_sliding = 1.0e4f, full_partial_rotary = 0.25f;
    bool present() const { return n_layers > 0 && hidden > 0; }
};

// ── Per-kernel taps, for the walk against mlx-lm ────────────────────────────
//
// Which bind index carries each kind's OUTPUT and how wide it is. The names
// match `tests/parity/gemma4_mlx_taps.py`, so `cosine_bisect.py` can diff the
// two trees and name the FIRST (kind, layer) that disagrees.
//
// Under `no_recycle` every activation VALUE gets its own buffer -- but the
// in-place kinds (the q/k norms, both ropes, the two post-norms, the PLE norm)
// rewrite their input's value, so they share a buffer with the tap before them
// and it only ever holds the last writer's result. Naming the earlier one too
// would publish the later tensor under it, which reads as a divergence that is
// really the dump lying. So only a colour's final writer is named.
struct Tap {
    const char* name;
    std::uint8_t out_bind;
    int width;
};

bool tap_for(const gemma4::Dispatch& d, const Gemma4Geometry& g, Tap& out) {
    const int L = d.layer;
    const int hd = L >= 0 ? g.head_dim_of(L) : g.head_dim;
    const int q_dim = g.n_q_heads * hd;
    const int kv_dim = g.n_kv_heads * hd;
    const int inter = L >= 0 ? g.intermediate_of(L) : g.intermediate;
    const int ple_all = g.n_layers * g.per_layer_emb_dim;
    switch (d.kind) {
        case Kind::EmbedGather:      out = {"embed",        4, g.hidden};  return true;
        case Kind::PleTokenGather:   out = {"ple_tok",      4, ple_all};   return true;
        case Kind::PleProjGemv:      out = {"ple_proj",     4, ple_all};   return true;
        case Kind::PleProjNorm:      out = {"ple_projnorm", 2, ple_all};   return true;
        case Kind::PleCombine:       out = {"ple",          2, ple_all};   return true;

        case Kind::AttnNorm:         out = {"attn_norm",    2, g.hidden};  return true;
        case Kind::QmvQ:             out = {"q_proj",       4, q_dim};     return true;
        case Kind::QmvK:             out = {"k_proj",       4, kv_dim};    return true;
        case Kind::QmvV:             out = {"v_proj",       4, kv_dim};    return true;
        case Kind::QNorm:            out = {"q_norm",       2, q_dim};     return true;
        case Kind::KNorm:            out = {"k_norm",       2, kv_dim};    return true;
        case Kind::VNorm:            out = {"v_norm",       1, kv_dim};    return true;
        case Kind::RopeQ:            out = {"rope_q",       0, q_dim};     return true;
        case Kind::RopeK:            out = {"rope_k",       0, kv_dim};    return true;
        case Kind::Sdpa:             out = {"sdpa",         3, q_dim};     return true;
        case Kind::QmvO:             out = {"o_proj",       4, g.hidden};  return true;
        case Kind::PostAttnNorm:     out = {"attn_postnorm", 2, g.hidden}; return true;
        case Kind::AttnResidual:     out = {"attn_resid",   2, g.hidden};  return true;

        case Kind::FfnNorm:          out = {"ffn_norm",     2, g.hidden};  return true;
        case Kind::QmvGate:          out = {"gate_proj",    4, inter};     return true;
        case Kind::QmvUp:            out = {"up_proj",      4, inter};     return true;
        case Kind::GegluTanh:        out = {"geglu",        2, inter};     return true;
        case Kind::QmvDown:          out = {"down_proj",    4, g.hidden};  return true;
        case Kind::PostFfnNorm:      out = {"ffn_postnorm", 2, g.hidden};  return true;
        case Kind::FfnResidual:      out = {"ffn_resid",    2, g.hidden};  return true;

        case Kind::PleGateGemv:      out = {"ple_gate",     4, g.per_layer_emb_dim}; return true;
        case Kind::PleGeglu:         out = {"ple_act",      2, g.per_layer_emb_dim}; return true;
        case Kind::PleProjLayerGemv: out = {"ple_back",     4, g.hidden};  return true;
        case Kind::PleNorm:          out = {"ple_norm",     2, g.hidden};  return true;
        case Kind::PleResidual:      out = {"ple_resid",    2, g.hidden};  return true;
        case Kind::LayerScalar:      out = {"layer_out",    2, g.hidden};  return true;

        case Kind::FinalRms:         out = {"final_norm",   2, g.hidden};  return true;
        case Kind::LmHead:           out = {"logits_raw",   4, g.vocab};   return true;
        case Kind::FinalSoftcap:     out = {"logits",       1, g.vocab};   return true;
        default: return false;
    }
}

void dump_gemma4_taps(const std::vector<gemma4::Dispatch>& dag, const Gemma4Geometry& g,
                      const ScratchColoring& coloring, const std::vector<SlotHandle>& pool) {
    const int n_pool = static_cast<int>(pool.size());
    auto color_of = [&](std::size_t di, std::uint8_t bind_index) {
        for (const auto& sb : coloring.per_dispatch[di]) {
            if (sb.bind_index == bind_index) return int(sb.color);
        }
        return -1;
    };

    std::vector<int> last_writer(std::size_t(n_pool < 0 ? 0 : n_pool), -1);
    for (std::size_t di = 0; di < dag.size(); ++di) {
        Tap tap{};
        if (!tap_for(dag[di], g, tap)) continue;
        const int c = color_of(di, tap.out_bind);
        if (c >= 0 && c < n_pool) last_writer[std::size_t(c)] = int(di);
    }

    for (std::size_t di = 0; di < dag.size(); ++di) {
        Tap tap{};
        if (!tap_for(dag[di], g, tap)) continue;
        const int c = color_of(di, tap.out_bind);
        if (c < 0 || c >= n_pool || !pool[std::size_t(c)].valid()) continue;
        if (last_writer[std::size_t(c)] != int(di)) continue;
        const std::string name = dag[di].layer < 0
            ? std::string(tap.name)
            : std::to_string(dag[di].layer) + "." + tap.name;
        dump_golden_bf16(name, pool[std::size_t(c)].contents(), 1, tap.width, tap.width);
    }
}

// How many argument-table slots each kind's kernel actually reads.
//
// The counterpart of `decode_step_mb_test.cpp`'s `required_slots`, and the gate
// that would have caught the four SDPA stride constants gemma4 never bound: an
// unbound `constant` slot is not a crash, it is a stride read out of whatever
// the argument table happened to hold.
int required_slots(Kind k) {
    switch (k) {
        case Kind::EmbedGather:
        case Kind::PleTokenGather:      return 7;  // w/scales/biases/id/out/hidden/scale
        case Kind::QmvQ: case Kind::QmvK: case Kind::QmvV: case Kind::QmvO:
        case Kind::QmvGate: case Kind::QmvUp: case Kind::QmvDown:
        case Kind::LmHead:
        case Kind::PleProjGemv: case Kind::PleGateGemv:
        case Kind::PleProjLayerGemv:    return 7;
        case Kind::AttnNorm: case Kind::PostAttnNorm: case Kind::FfnNorm:
        case Kind::PostFfnNorm: case Kind::FinalRms: case Kind::QNorm:
        case Kind::KNorm: case Kind::PleNorm: case Kind::PleProjNorm: return 4;
        case Kind::VNorm:               return 3;
        case Kind::RopeQ: case Kind::RopeK: return 5;
        case Kind::KvAppend:            return 8;
        case Kind::Sdpa:                return 14;
        case Kind::AttnResidual: case Kind::FfnResidual: case Kind::PleResidual:
        case Kind::GegluTanh: case Kind::PleGeglu:
        case Kind::LayerScalar: case Kind::PleCombine: return 4;
        case Kind::FinalSoftcap:        return 3;
        case Kind::Argmax:              return 4;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string ckpt = argc > 1 ? argv[1] : std::string();
    if (ckpt.empty()) {
        const char* home = std::getenv("HOME");
        if (home != nullptr) ckpt = std::string(home) + "/.pie-bench/gemma4-e2b-pie";
    }
    std::string kernels_dir = PIE_METAL_KERNELS_DIR_FOR_TEST;

    {
        std::string probe = ckpt + "/config.json";
        FILE* f = std::fopen(probe.c_str(), "rb");
        if (f == nullptr) {
            std::printf("gemma4 forward: SKIP (no checkpoint at %s)\n", ckpt.c_str());
            return 0;
        }
        std::fclose(f);
    }
    std::printf("gemma4 forward (%s)\n", ckpt.c_str());

    Gemma4Geometry g;
    std::string err;
    if (!geometry_from_facts(Facts{}, g, &err)) {
        std::printf("  FAIL  geometry: %s\n", err.c_str());
        return 1;
    }
    g.vocab = 262144;
    const int max_ctx = 1024;
    g.kv_max_ctx = max_ctx;

    auto ctx = RawMetalContext::create(std::size_t(6) << 30);
    if (!ctx) {
        std::printf("  FAIL  RawMetalContext::create\n");
        return 1;
    }

    // ── the plan the contract authors for this checkpoint ──
    pie_loader::LoadPlan plan;
    try {
        pie::metal::model::ContractFacts facts;
        facts.first_kv_shared_layer = g.first_kv_shared();
        plan = compile_load_plan(ckpt, metal_device_target(), "gemma4", facts);
    } catch (const std::exception& e) {
        std::printf("  FAIL  compile_load_plan: %s\n", e.what());
        return 1;
    }
    expect(true, "the contract compiles a load plan for the real checkpoint");

    // ── weights ──
    BoundGemma4 b;
    try {
        const auto storage = plan.view();
        pie_loader::CheckpointSource view(storage);
        StagedWeights staged =
            stage_plan_weights(*ctx, view, plan, storage.memory.persistent_bytes);
        b.weights = std::move(staged.weights);
    } catch (const std::exception& e) {
        std::printf("  FAIL  stage_plan_weights: %s\n", e.what());
        return 1;
    }
    std::printf("    (%zu tensors staged)\n", b.weights.size());
    expect(!b.weights.empty(), "the checkpoint's tensors are staged into the heap");

    // ── KV, sized per owning layer ──
    b.kv.resize(g.n_layers);
    for (int L = 0; L < g.n_layers; ++L) {
        if (g.is_kv_shared(L)) continue;
        const std::size_t bytes = gemma4_kv_bytes_per_layer(g, L, max_ctx, 2);
        b.kv[L].k = ctx->heap_alloc(bytes);
        b.kv[L].v = ctx->heap_alloc(bytes);
    }

    // ── the DAG, its dataflow, and the pool that dataflow needs ──
    const auto dag = build_gemma4_dag(g, /*with_argmax=*/false);
    const ScratchPlan sp = build_gemma4_scratch(dag, g);
    // Under a tap dump every value needs its own buffer, or a later dispatch
    // overwrites the one being read.
    const ScratchColoring coloring =
        color_gemma4_scratch(dag, sp, /*no_recycle=*/golden_taps_enabled());
    expect(coloring.hazard_free, "the activation colouring is hazard-free");

    b.pool.resize(coloring.colors_used);
    const std::size_t widest = std::size_t(g.vocab) * 2;
    for (int c = 0; c < coloring.colors_used; ++c) b.pool[c] = ctx->heap_alloc(widest);

    b.io.resize(kIoSlotCount);
    for (int i = 0; i < kIoSlotCount; ++i) b.io[i] = ctx->heap_alloc(4096);
    // Token 2 (<bos> in gemma's vocabulary) at position 0. `SeqLen` is what the
    // attention reads as its key count -- left at 0 it attends nothing and the
    // whole sublayer returns zeros, which the tap walk names as `0.sdpa`.
    *static_cast<std::int32_t*>(b.io[int(IoSlot::TokenId)].contents()) = 2;
    *static_cast<std::int32_t*>(b.io[int(IoSlot::Position)].contents()) = 0;
    *static_cast<std::int32_t*>(b.io[int(IoSlot::SeqLen)].contents()) = 1;

    Gemma4Psos psos;
    DecodeStepPsos base;
    if (!build_gemma4_psos(*ctx, kernels_dir, psos, &err) ||
        !load_decode_psos(*ctx, kernels_dir, base, /*with_argmax=*/false, &err)) {
        std::printf("  FAIL  pipelines: %s\n", err.c_str());
        return 1;
    }

    const int bound = bind_gemma4_consts(*ctx, dag, g);
    bind_gemma4_dag(*ctx, b, dag, g, coloring);
    std::printf("    (%d constants bound over %zu dispatches)\n", bound, dag.size());

    // Every slot the kernel reads, bound. An unbound one is not a crash: it is a
    // stride, a scale or a pointer read out of whatever the table happened to
    // hold, which reads downstream as wrong numbers.
    bool all_bound = true;
    int missing = 0;
    for (const gemma4::Dispatch& d : dag) {
        const int n = required_slots(d.kind);
        for (int slot = 0; slot < n; ++slot) {
            if (!ctx->arg_slot_is_bound(d.ordinal, std::uint8_t(slot))) {
                if (missing < 24) {
                    std::printf("    missing ord=%d kind=%d layer=%d slot=%d\n", d.ordinal,
                                int(d.kind), d.layer, slot);
                }
                ++missing;
                all_bound = false;
            }
        }
    }
    if (missing > 0) std::printf("    (%d unbound slots)\n", missing);
    expect(all_bound, "every slot the kernels read is bound");

    ctx->make_resident();

    ctx->run_step([&](StepEncoder& se) {
        encode_gemma4_step(se, dag, g, base, psos);
    });

    if (golden_taps_enabled()) {
        dump_gemma4_taps(dag, g, coloring, b.pool);
        std::printf("    (taps -> %s)\n", golden_tap_dir().c_str());
    }

    // ── did it compute anything? ──
    // Read the softcap's OUTPUT, named by its bind index rather than by
    // position -- `front()` is the input, and reading it would compare the wrong
    // tensor against mlx while looking like it worked.
    int logits_color = -1;
    for (const auto& sb : coloring.per_dispatch.back()) {
        if (sb.bind_index == (std::uint8_t)bind::Softcap::Out) logits_color = sb.color;
    }
    expect(logits_color >= 0, "the final dispatch has an output to read");
    if (logits_color < 0) return 1;
    const auto* logits = static_cast<const std::uint16_t*>(b.pool[logits_color].contents());
    int nonzero = 0, nan_or_inf = 0;
    float best = -1e30f;
    int argmax = -1;
    for (int i = 0; i < g.vocab; ++i) {
        const float v = from_bf16(logits[i]);
        if (v != 0.0f) ++nonzero;
        if (std::isnan(v) || std::isinf(v)) ++nan_or_inf;
        if (v > best) {
            best = v;
            argmax = i;
        }
    }
    std::printf("    (nonzero %d/%d, argmax %d, max logit %.4f)\n", nonzero, g.vocab,
                argmax, best);
    // The five the model actually prefers, so the comparison against mlx-lm is
    // about the distribution and not just its peak.
    std::vector<int> idx(g.vocab);
    for (int i = 0; i < g.vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int c) {
        return from_bf16(logits[a]) > from_bf16(logits[c]);
    });
    std::printf("    top5");
    for (int i = 0; i < 5; ++i) {
        std::printf(" (%d, %.4f)", idx[i], from_bf16(logits[idx[i]]));
    }
    std::printf("\n");
    expect(nonzero > g.vocab / 2, "the logits are populated, not a zeroed buffer");
    expect(nan_or_inf == 0, "and finite everywhere");
    expect(std::fabs(best) <= 30.0f + 1e-3f,
           "within the final softcap, so the tail of the graph ran");
    // mlx-lm's answer for this checkpoint, <bos> at position 0. It is the whole
    // point of the test: "it ran" was true while every projection was summing a
    // thirty-second of its row. Per-kernel cosine against mlx-lm is >0.9989 at
    // every one of the 35 layers' taps (tests/parity/gemma4_mlx_taps.py +
    // cosine_bisect.py --family gemma4); the argmax is what that buys.
    expect(argmax == 236761, "and the argmax is mlx-lm's for the same checkpoint");

    std::printf("\n==== gemma4_forward_test: %s ====\n",
                failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
