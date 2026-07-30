// Does gemma4 compute the same thing at M>1 as it does at M=1?
//
// `gemma4_forward_test` proves the decode path against mlx-lm one token at a
// time. The prefill path is a different set of kernels -- matmul instead of
// matvec, paged KV instead of the contiguous ring, per-row IO instead of a
// scalar -- and none of it had ever been checked against anything. Until it is,
// the executor cannot use it, and without it a second concurrent sequence
// cannot be served.
//
// So this fires the WHOLE prompt in one command buffer and compares every row
// against mlx-lm, which ran the same prompt as one parallel forward.
//
// Skipped (not failed) when the checkpoint is absent.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        // The fused kinds produce the RESIDUAL, so they are named for it; the
        // norm they contain is no longer a tensor anything can observe.
        case Kind::PostAttnResidual: out = {"attn_resid",   2, g.hidden};  return true;

        case Kind::FfnNorm:          out = {"ffn_norm",     2, g.hidden};  return true;
        case Kind::QmvGate:          out = {"gate_proj",    4, inter};     return true;
        case Kind::QmvUp:            out = {"up_proj",      4, inter};     return true;
        case Kind::GegluTanh:        out = {"geglu",        2, inter};     return true;
        case Kind::QmvDown:          out = {"down_proj",    4, g.hidden};  return true;
        case Kind::PostFfnResidual:  out = {"ffn_resid",    2, g.hidden};  return true;

        case Kind::PleGateGemv:      out = {"ple_gate",     4, g.per_layer_emb_dim}; return true;
        case Kind::PleGeglu:         out = {"ple_act",      2, g.per_layer_emb_dim}; return true;
        case Kind::PleProjLayerGemv: out = {"ple_back",     4, g.hidden};  return true;
        // Norm, residual add and the learned gain in one -- so its output is
        // exactly mlx-lm's `layer_out`.
        case Kind::PleResidualScaled: out = {"layer_out",   2, g.hidden};  return true;
        case Kind::LayerScalar:      out = {"layer_out",    2, g.hidden};  return true;

        case Kind::FinalRms:         out = {"final_norm",   2, g.hidden};  return true;
        case Kind::LmHead:           out = {"logits_raw",   4, g.vocab};   return true;
        case Kind::FinalSoftcap:     out = {"logits",       1, g.vocab};   return true;
        default: return false;
    }
}

void dump_gemma4_taps(const std::vector<gemma4::Dispatch>& dag, const Gemma4Geometry& g,
                      const ScratchColoring& coloring, const std::vector<SlotHandle>& pool,
                      int rows) {    const int n_pool = static_cast<int>(pool.size());
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
        dump_golden_bf16(name, pool[std::size_t(c)].contents(), rows, tap.width, tap.width);
    }
}

// How many barriers the step is made of, as a measurement rather than a claim.
//
// The shipped walk drops a barrier inside a concurrency run; `all` forces one
// after every dispatch and `none` drops them entirely. `none` is NOT correct --
// it races every RAW edge in the DAG -- but the three numbers together price
// what barrier drain actually costs on this step, which is the only honest way
// to decide whether fusing dispatches is worth anything. Phase 51 priced
// qwen3.5's at 5.4 us each by the same method.
enum class BarrierMode { Runs, All, None };

BarrierMode barrier_mode_from_env() {
    const char* e = std::getenv("PIE_G4_BARRIERS");
    if (e == nullptr) return BarrierMode::Runs;
    if (std::string(e) == "all") return BarrierMode::All;
    if (std::string(e) == "none") return BarrierMode::None;
    return BarrierMode::Runs;
}

int encode_gemma4_variant(StepEncoder& se, const std::vector<gemma4::Dispatch>& dag,
                          const Gemma4Geometry& g, const DecodeStepPsos& base,
                          const Gemma4Psos& g4, BarrierMode mode) {
    const std::vector<int> run_ends = gemma4_run_ends(dag);
    int barriers = 0;
    for (std::size_t i = 0; i < dag.size(); ++i) {
        const gemma4::Dispatch& d = dag[i];
        Grid grid;
        Threadgroup tg;
        launch_shape(d, g, grid, tg);
        se.set_pso(pso_for(d, base, g4));
        se.set_argtable_ordinal(d.ordinal);
        se.dispatch(grid, tg);
        const bool last = i + 1 >= dag.size();
        bool want = last || run_ends[i] == static_cast<int>(i);
        if (mode == BarrierMode::All) want = true;
        if (mode == BarrierMode::None) want = last;
        if (want) {
            se.barrier();
            ++barriers;
        }
    }
    return barriers;
}

void write_u32s(const SlotHandle& s, const std::vector<std::uint32_t>& v) {
    if (!s.valid() || s.contents() == nullptr) return;
    std::memcpy(s.contents(), v.data(), v.size() * sizeof(std::uint32_t));
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string ckpt = argc > 1 ? argv[1] : std::string();
    if (ckpt.empty()) {
        const char* home = std::getenv("HOME");
        if (home != nullptr) ckpt = std::string(home) + "/.pie-bench/gemma4-e2b-pie";
    }
    const std::string kernels_dir = PIE_METAL_KERNELS_DIR_FOR_TEST;
    {
        const std::string probe = ckpt + "/config.json";
        FILE* f = std::fopen(probe.c_str(), "rb");
        if (f == nullptr) {
            std::printf("gemma4 prefill numerics: SKIP (no checkpoint at %s)\n", ckpt.c_str());
            return 0;
        }
        std::fclose(f);
    }
    std::printf("gemma4 prefill numerics (%s)\n", ckpt.c_str());

    // The prompt `gemma4_forward_test` teacher-forces one token at a time; here
    // it goes in one fire, so the two paths are compared on the same tokens.
    std::vector<std::uint32_t> ids{2, 818, 3821, 563, 529, 476, 3625, 506};
    bool default_prompt = true;
    if (const char* env = std::getenv("PIE_G4MB_TOKENS"); env != nullptr && *env != '\0') {
        ids.clear();
        for (const char* p = env; *p != '\0';) {
            char* end = nullptr;
            const long v = std::strtol(p, &end, 10);
            if (end == p) break;
            ids.push_back(std::uint32_t(v));
            p = (*end == ',') ? end + 1 : end;
        }
        if (ids.empty()) ids.push_back(2);
        default_prompt = false;
    }
    const int N = int(ids.size());

    Gemma4Geometry g;
    std::string err;
    if (!geometry_from_facts(Facts{}, g, &err)) {
        std::printf("  FAIL  geometry: %s\n", err.c_str());
        return 1;
    }
    g.vocab = 262144;
    g.kv_page_size = 32;
    g.total_pages = 4;              // one request's worth; 8 tokens fit in one
    g.kv_max_ctx = g.kv_page_size * g.total_pages;
    g.max_tokens = N;
    g.max_requests = 1;
    g.paged_kv_enabled = true;

    // Under a tap dump every activation VALUE gets its own [N, vocab] slot, so
    // the pool is the budget rather than the weights.
    auto ctx = RawMetalContext::create(std::size_t(golden_taps_enabled() ? 14 : 8) << 30);
    if (!ctx) {
        std::printf("  FAIL  RawMetalContext::create\n");
        return 1;
    }

    pie_loader::LoadPlan plan;
    try {
        pie::metal::model::ContractFacts facts;
        facts.first_kv_shared_layer = g.first_kv_shared();
        plan = compile_load_plan(ckpt, metal_device_target(), "gemma4", facts);
    } catch (const std::exception& e) {
        std::printf("  FAIL  compile_load_plan: %s\n", e.what());
        return 1;
    }

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

    // Paged KV: one page pool per OWNING layer, laid out [page*page_size + off]
    // by (kv_head, head_dim) -- the NHD page row the paged kernels index.
    std::vector<SlotHandle> kpages(std::size_t(g.n_layers));
    std::vector<SlotHandle> vpages(std::size_t(g.n_layers));
    for (int L = 0; L < g.n_layers; ++L) {
        const int hd = g.head_dim_of(L);
        const std::size_t bytes = std::size_t(g.total_pages) * std::size_t(g.kv_page_size) *
                                  std::size_t(g.n_kv_heads) * std::size_t(hd) * 2;
        kpages[std::size_t(L)] = ctx->heap_alloc(bytes);
        vpages[std::size_t(L)] = ctx->heap_alloc(bytes);
        std::memset(kpages[std::size_t(L)].contents(), 0, bytes);
        std::memset(vpages[std::size_t(L)].contents(), 0, bytes);
    }
    b.kv.resize(std::size_t(g.n_layers));
    for (int L = 0; L < g.n_layers; ++L) {
        b.kv[std::size_t(L)].k = kpages[std::size_t(L)];
        b.kv[std::size_t(L)].v = vpages[std::size_t(L)];
    }

    const auto dag = build_gemma4_dag_mb(g, /*ordinal_base=*/0, /*with_argmax=*/false);
    const ScratchPlan sp = build_gemma4_scratch(dag, g);
    const ScratchColoring coloring =
        color_gemma4_scratch(dag, sp, /*no_recycle=*/golden_taps_enabled());
    expect(coloring.hazard_free, "the activation colouring is hazard-free");

    // Every activation is [N, width] row-major at M>1, so a pool slot is N rows
    // of the widest one.
    b.pool.resize(std::size_t(coloring.colors_used));
    const std::size_t slot_bytes = std::size_t(N) * std::size_t(g.vocab) * 2;
    for (int c = 0; c < coloring.colors_used; ++c) b.pool[std::size_t(c)] = ctx->heap_alloc(slot_bytes);

    b.io.resize(kIoSlotCount);
    for (int i = 0; i < kIoSlotCount; ++i) b.io[i] = ctx->heap_alloc(4096);

    // Per-row IO, and the page table one request needs.
    std::vector<std::uint32_t> pos, req, wpage, woff;
    pos.resize(std::size_t(N));
    req.assign(std::size_t(N), 0u);
    wpage.resize(std::size_t(N));
    woff.resize(std::size_t(N));
    for (int i = 0; i < N; ++i) {
        pos[std::size_t(i)] = std::uint32_t(i);
        wpage[std::size_t(i)] = std::uint32_t(i / g.kv_page_size);
        woff[std::size_t(i)] = std::uint32_t(i % g.kv_page_size);
    }
    std::vector<std::uint32_t> page_indices;
    page_indices.resize(std::size_t(g.total_pages));
    for (int p = 0; p < g.total_pages; ++p) page_indices[std::size_t(p)] = std::uint32_t(p);
    write_u32s(b.io[int(IoSlot::TokenId)], ids);
    write_u32s(b.io[int(IoSlot::Position)], pos);
    write_u32s(b.io[int(IoSlot::ReqOfToken)], req);
    write_u32s(b.io[int(IoSlot::WPage)], wpage);
    write_u32s(b.io[int(IoSlot::WOff)], woff);
    write_u32s(b.io[int(IoSlot::KvPageIndices)], page_indices);
    write_u32s(b.io[int(IoSlot::KvPageIndptr)], {0u, std::uint32_t(g.total_pages)});
    write_u32s(b.io[int(IoSlot::QoIndptr)], {0u, std::uint32_t(N)});
    // No dense mask: the causal bound comes from PositionIds, and the window
    // from the layer's own constant.
    write_u32s(b.io[int(IoSlot::AttnMaskStride)], {0u});
    std::memset(b.io[int(IoSlot::AttnMaskEnabled)].contents(), 0, std::size_t(N));

    Gemma4Psos psos;
    DecodeStepPsos base;
    MultiBatchPsos mb;
    if (!build_gemma4_psos(*ctx, kernels_dir, psos, &err) ||
        !load_decode_psos(*ctx, kernels_dir, base, /*with_argmax=*/false, &err) ||
        !load_multibatch_psos(*ctx, kernels_dir, mb, /*with_d512=*/true, &err)) {
        std::printf("  FAIL  pipelines: %s\n", err.c_str());
        return 1;
    }

    const int bound = bind_gemma4_consts(*ctx, dag, g, N, /*paged=*/true);
    bind_gemma4_dag_mb(*ctx, b, dag, g, coloring, kpages, vpages);
    std::printf("    (%d constants over %zu dispatches, %d rows)\n", bound, dag.size(), N);
    // Every M>1 dispatch must fit the pipeline it will actually run on. This is
    // checked against `pso_for_mb`, not `pso_for`: a 512-wide head carries twice
    // the registers per thread, so its paged instantiation can allow fewer
    // threads per threadgroup than the 256-wide one -- and a dispatch that
    // exceeds the limit does not run at all, which reads downstream as an
    // attention output of exactly zero.
    int oversized = 0;
    for (const gemma4::Dispatch& d : dag) {
        Grid gr;
        Threadgroup tg;
        launch_shape_mb(d, g, N, gr, tg);
        const Pso p = pso_for_mb(d, g, N, base, mb, psos);
        const std::uint32_t threads = tg.x * tg.y * tg.z;
        if (!p.valid()) {
            std::printf("    unresolved pipeline: kind=%d layer=%d\n", int(d.kind), d.layer);
            ++oversized;
        } else if (threads > ctx->pso_max_threads(p)) {
            std::printf("    oversized: kind=%d layer=%d tg=%u allowed=%u\n", int(d.kind),
                        d.layer, threads, ctx->pso_max_threads(p));
            ++oversized;
        }
    }
    expect(oversized == 0, "every M>1 dispatch fits the pipeline it runs on");

    ctx->make_resident();

    ctx->run_step([&](StepEncoder& se) {
        encode_gemma4_step_mb(se, dag, g, N, base, mb, psos);
    });

    if (golden_taps_enabled()) {
        dump_gemma4_taps(dag, g, coloring, b.pool, N);
        std::printf("    (taps -> %s)\n", golden_tap_dir().c_str());
    }

    // The softcap's output, by bind index. Every row is capped, so the last one
    // -- the row a prefill samples -- is readable.
    int logits_color = -1;
    for (const auto& sb : coloring.per_dispatch.back()) {
        if (sb.bind_index == (std::uint8_t)bind::Softcap::Out) logits_color = sb.color;
    }
    expect(logits_color >= 0, "the final dispatch has an output to read");
    if (logits_color < 0) return 1;
    const auto* logits = static_cast<const std::uint16_t*>(b.pool[std::size_t(logits_color)].contents());

    const auto* last = logits + std::size_t(N - 1) * std::size_t(g.vocab);
    int nonzero = 0, nan_or_inf = 0, argmax = -1;
    float best = -1e30f;
    for (int i = 0; i < g.vocab; ++i) {
        const float v = from_bf16(last[i]);
        if (v != 0.0f) ++nonzero;
        if (std::isnan(v) || std::isinf(v)) ++nan_or_inf;
        if (v > best) {
            best = v;
            argmax = i;
        }
    }
    std::printf("    (last row: nonzero %d/%d, argmax %d, max logit %.4f)\n", nonzero, g.vocab,
                argmax, best);
    expect(nonzero > g.vocab / 2, "the last row's logits are populated");
    expect(nan_or_inf == 0, "and finite everywhere");
    // What `gemma4_forward_test` gets by teacher-forcing the same eight tokens
    // one at a time, and what mlx-lm gets running them as one forward. The whole
    // point: the prefill path is the same model.
    if (default_prompt) {
        expect(argmax == 3821, "and the prefill's last row agrees with decode and mlx-lm");
    }

    std::printf("\n==== gemma4_prefill_numerics_test: %s ====\n",
                failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
