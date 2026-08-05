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
#include "model/facts.hpp"
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
    bool enable_moe = false;
    int n_experts = 0, experts_per_token = 0, moe_intermediate = 0;
    bool attention_k_eq_v = false;
    int n_global_kv_heads = 0;
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
        se.set_pso(pso_for(d, g, base, g4));
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
    g.total_pages = 4;              // one request's worth; 32 tokens fit in one
    g.kv_max_ctx = g.kv_page_size * g.total_pages;
    g.max_tokens = N;
    g.max_requests = 1;
    g.paged_kv_enabled = true;

    // Under a tap dump every activation VALUE gets its own [N, vocab] slot, so
    // the pool is the budget rather than the weights.
    // Under a tap dump every activation VALUE gets its own slot, so the pool
    // scales with the row count. 8 rows fit in 14 GB; 16 do not.
    auto ctx = RawMetalContext::create(std::size_t(golden_taps_enabled() ? 12 : 8) << 30);
    if (!ctx) {
        std::printf("  FAIL  RawMetalContext::create\n");
        return 1;
    }

    pie_loader::LoadPlan plan;
    try {
        pie::metal::model::ContractFacts facts;
        facts.num_hidden_layers = g.n_layers;
        facts.num_kv_shared_layers = g.num_kv_shared_layers;
        plan = compile_load_plan(ckpt, metal_device_target(),
                                 descriptor_for_testing("gemma4", facts));
    } catch (const std::exception& e) {
        std::printf("  FAIL  compile_load_plan: %s\n", e.what());
        return 1;
    }

    BoundGemma4 b;
    std::shared_ptr<void> weight_mapping;
    try {
        const auto storage = plan.view();
        auto view = std::make_shared<pie_loader::CheckpointSource>(storage);
        StagedWeights staged =
            stage_plan_weights(*ctx, std::move(view), plan, storage.memory.persistent_bytes);
        b.weights = std::move(staged.weights);
        // The mapping has to outlive `b.weights`: when the checkpoint places
        // its tensors where a device pointer may point, those slots are slices
        // of the checkpoint's own mmap rather than copies in the heap, and
        // dropping `staged` here would unmap them under the GPU.
        weight_mapping = std::move(staged.weight_mapping);
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
    // of its OWN width -- not of the vocabulary. Under a tap dump there is one
    // slot per dispatch, and sizing them all at the widest costs 8 GB at 16
    // rows for a model whose activations are mostly 1536 wide.
    b.pool.resize(std::size_t(coloring.colors_used));
    std::vector<std::size_t> slot_elems(std::size_t(coloring.colors_used), 0);
    for (std::size_t di = 0; di < dag.size(); ++di) {
        Tap t{};
        if (!tap_for(dag[di], g, t)) continue;
        for (const auto& sb : coloring.per_dispatch[di]) {
            if (sb.color < 0 || sb.color >= coloring.colors_used) continue;
            slot_elems[std::size_t(sb.color)] =
                std::max(slot_elems[std::size_t(sb.color)], std::size_t(t.width));
        }
    }
    // The 16-row case below fires more rows than the default prompt has, and a
    // projection pads its batch up to a whole GEMM tile on top of that, so the
    // pool is sized for the widest fire this file makes rather than for `N`.
    const int pool_rows = gemma4_qmm_pool_rows(std::max(N, 32));
    for (int c = 0; c < coloring.colors_used; ++c) {
        const std::size_t w = slot_elems[std::size_t(c)] > 0 ? slot_elems[std::size_t(c)]
                                                             : std::size_t(g.hidden);
        b.pool[std::size_t(c)] = ctx->heap_alloc(std::size_t(pool_rows) * w * 2);
    }

    b.io.resize(kIoSlotCount);
    for (int i = 0; i < kIoSlotCount; ++i) b.io[i] = ctx->heap_alloc(8192);

    // Per-row IO and the page table. `starts` gives each request's first token
    // row, so one fire can carry several of them -- which is what a mixed
    // prefill+decode batch is.
    const auto fill_io = [&](const std::vector<std::uint32_t>& toks,
                             const std::vector<std::uint32_t>& starts, int pages_per_req) {
        const int rows = int(toks.size());
        std::vector<std::uint32_t> pos, req, wpage, woff, qo, indptr, page_indices;
        pos.resize(std::size_t(rows));
        req.resize(std::size_t(rows));
        wpage.resize(std::size_t(rows));
        woff.resize(std::size_t(rows));
        for (std::size_t r = 0; r + 1 <= starts.size(); ++r) {
            const int lo = int(starts[r]);
            const int hi = (r + 1 < starts.size()) ? int(starts[r + 1]) : rows;
            for (int i = lo; i < hi; ++i) {
                const int p = i - lo;                      // position within this request
                pos[std::size_t(i)] = std::uint32_t(p);
                req[std::size_t(i)] = std::uint32_t(r);
                // The page this request's p-th token lands in, in ITS page list.
                wpage[std::size_t(i)] =
                    std::uint32_t(int(r) * pages_per_req + p / g.kv_page_size);
                woff[std::size_t(i)] = std::uint32_t(p % g.kv_page_size);
            }
        }
        for (std::uint32_t v : starts) qo.push_back(v);
        qo.push_back(std::uint32_t(rows));
        // Each request owns a contiguous run of physical pages; the page LIST is
        // walked through `kv_page_indptr`, so the two need not coincide.
        for (std::size_t r = 0; r < starts.size(); ++r) {
            indptr.push_back(std::uint32_t(int(r) * pages_per_req));
        }
        indptr.push_back(std::uint32_t(int(starts.size()) * pages_per_req));
        page_indices.resize(std::size_t(g.total_pages));
        for (int p = 0; p < g.total_pages; ++p) page_indices[std::size_t(p)] = std::uint32_t(p);
        write_u32s(b.io[int(IoSlot::TokenId)], toks);
        write_u32s(b.io[int(IoSlot::Position)], pos);
        write_u32s(b.io[int(IoSlot::ReqOfToken)], req);
        write_u32s(b.io[int(IoSlot::WPage)], wpage);
        write_u32s(b.io[int(IoSlot::WOff)], woff);
        write_u32s(b.io[int(IoSlot::KvPageIndices)], page_indices);
        write_u32s(b.io[int(IoSlot::KvPageIndptr)], indptr);
        write_u32s(b.io[int(IoSlot::QoIndptr)], qo);
        // This test samples EVERY row -- it compares all of them against mlx-lm
        // -- so the tail's gather is the identity and the rows keep the indices
        // the taps and the assertions below use.
        std::vector<std::uint32_t> sample;
        sample.resize(std::size_t(rows));
        for (int i = 0; i < rows; ++i) sample[std::size_t(i)] = std::uint32_t(i);
        write_u32s(b.io[int(IoSlot::SampleRows)], sample);
        std::memset(b.io[int(IoSlot::AttnMaskEnabled)].contents(), 0, std::size_t(rows));
    };
    fill_io(ids, {0u}, g.total_pages);
    // No dense mask: the causal bound comes from PositionIds, and the window
    // from the layer's own constant.
    write_u32s(b.io[int(IoSlot::AttnMaskStride)], {0u});

    Gemma4Psos psos;
    DecodeStepPsos base;
    MultiBatchPsos mb;
    if (!build_gemma4_psos(*ctx, kernels_dir, g, psos, &err) ||
        !load_decode_psos(*ctx, kernels_dir, base, g.quant, &err) ||
        !load_multibatch_psos(
            *ctx, kernels_dir, mb, g.quant, &err,
            MultiBatchPsoFeatures{.d512 = true, .sdpa_d256 = true})) {
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

    // ── The same prompt, one row at a time ──
    //
    // A decode step is a fire of one row, so the engine drives the M>1 path for
    // both. That only holds if replaying the prompt token by token through
    // 1-row fires lands on the same answer as firing it whole.
    if (default_prompt) {
        for (int L = 0; L < g.n_layers; ++L) {
            if (g.is_kv_shared(L)) continue;
            std::memset(kpages[std::size_t(L)].contents(), 0, kpages[std::size_t(L)].size);
            std::memset(vpages[std::size_t(L)].contents(), 0, vpages[std::size_t(L)].size);
        }
        for (int t = 0; t < N; ++t) {
            fill_io({ids[std::size_t(t)]}, {0u}, g.total_pages);
            // `fill_io` restarts positions per request, which is right for a
            // prefill and wrong for a continuation: row 0 of THIS fire is at
            // absolute position t.
            write_u32s(b.io[int(IoSlot::Position)], {std::uint32_t(t)});
            write_u32s(b.io[int(IoSlot::WPage)], {std::uint32_t(t / g.kv_page_size)});
            write_u32s(b.io[int(IoSlot::WOff)], {std::uint32_t(t % g.kv_page_size)});
            bind_gemma4_consts(*ctx, dag, g, 1, /*paged=*/true);
            ctx->run_step([&](StepEncoder& se) {
                encode_gemma4_step_mb(se, dag, g, 1, base, mb, psos);
            });
        }
        int bi = -1;
        float bv = -1e30f;
        for (int i = 0; i < g.vocab; ++i) {
            const float v = from_bf16(logits[i]);
            if (v > bv) { bv = v; bi = i; }
        }
        std::printf("    (one row at a time: argmax %d)\n", bi);
        expect(bi == 3821, "a decode step is a fire of one row, and lands where the whole "
                           "prompt does");
    }

    // ── Rebinding the constants must not consume the heap ──
    //
    // The row-dependent constants are rebound whenever a fire's row count
    // changes, which in a real batch is most fires. `bind_const` used to
    // `heap_alloc` a fresh slot every time, so a varying batch walked the heap
    // until allocation failed -- and what failed was the NEXT model's setup,
    // reporting "budget too small" about a request that had nothing to do with
    // it. gpt-oss hit it at four concurrent lanes.
    //
    // The value at a given (ordinal, index) is always the same size, so the
    // slot is allocated once and rewritten. A thousand alternations here is
    // far more than any fire sequence and costs milliseconds.
    {
        bool threw = false;
        try {
            for (int i = 0; i < 1000; ++i) {
                bind_gemma4_consts(*ctx, dag, g, (i % 2) ? 1 : N, /*paged=*/true);
            }
        } catch (const std::exception&) {
            threw = true;
        }
        expect(!threw, "rebinding the constants a thousand times does not exhaust the heap");
        // And it must still be the right answer afterwards, not merely alive.
        bind_gemma4_consts(*ctx, dag, g, N, /*paged=*/true);
    }

    // ── Sixteen rows: the batch where the GEMM engages ──
    //
    // Below twelve rows the projections are a per-row GEMV, and `qmm_bn`
    // refuses any batch that does not fill a whole 16-row tile -- so every
    // earlier case in this file, and every executor test, ran the GEMV. The
    // first fire that took the GEMM answered 147040 where mlx-lm says 476: the
    // split-K variant was dispatched and its partial sums never reduced,
    // because this family has no reduce pass and no split buffer.
    //
    // So the GEMM's batch needs a case of its own, or the next one to engage it
    // is a user's prompt.
    if (default_prompt) {
        const std::vector<std::uint32_t> ids16{2, 818, 3821, 563, 529, 476, 3625, 506,
                                               529, 476, 3625, 506, 818, 3821, 563, 529};
        const int rows16 = int(ids16.size());
        if (rows16 <= g.max_tokens || true) {
            for (int L = 0; L < g.n_layers; ++L) {
                if (g.is_kv_shared(L)) continue;
                std::memset(kpages[std::size_t(L)].contents(), 0, kpages[std::size_t(L)].size);
                std::memset(vpages[std::size_t(L)].contents(), 0, vpages[std::size_t(L)].size);
            }
            fill_io(ids16, {0u}, g.total_pages);
            bind_gemma4_consts(*ctx, dag, g, rows16, /*paged=*/true);
            ctx->run_step([&](StepEncoder& se) {
                encode_gemma4_step_mb(se, dag, g, rows16, base, mb, psos);
            });
            const auto* r = logits + std::size_t(rows16 - 1) * std::size_t(g.vocab);
            int bi = -1;
            float bv = -1e30f;
            for (int i = 0; i < g.vocab; ++i) {
                const float v = from_bf16(r[i]);
                if (v > bv) { bv = v; bi = i; }
            }
            std::printf("    (16 rows, the GEMM's batch: argmax %d)\n", bi);
            expect(bi == 476, "the GEMM path agrees with mlx-lm");
        }
    }

    // ── Thirty-two rows: the WIDE row block ──
    //
    // Above 32 rows the GEMM switches to BM=32, and `qmm_bn` takes the widest
    // column tile that divides the output -- which for every projection in this
    // checkpoint is 64. That pair was never instantiated: the loader aliased it
    // onto the BM=16 pipeline, so the grid was built for 32 rows per block and
    // the pipeline computed 16, and HALF the batch was never written. At 32 rows
    // this model's logits came back all zero.
    //
    // 16 rows does not catch it (BM=16 there) and 12 does not either, so the
    // wide block needs a case of its own.
    if (default_prompt) {
        std::vector<std::uint32_t> ids32;
        for (int rep = 0; rep < 4; ++rep) ids32.insert(ids32.end(), ids.begin(), ids.end());
        const int rows32 = int(ids32.size());
        for (int L = 0; L < g.n_layers; ++L) {
            if (g.is_kv_shared(L)) continue;
            std::memset(kpages[std::size_t(L)].contents(), 0, kpages[std::size_t(L)].size);
            std::memset(vpages[std::size_t(L)].contents(), 0, vpages[std::size_t(L)].size);
        }
        fill_io(ids32, {0u}, g.total_pages);
        bind_gemma4_consts(*ctx, dag, g, rows32, /*paged=*/true);
        ctx->run_step([&](StepEncoder& se) {
            encode_gemma4_step_mb(se, dag, g, rows32, base, mb, psos);
        });
        const auto* r = logits + std::size_t(rows32 - 1) * std::size_t(g.vocab);
        int bi = -1;
        float bv = -1e30f;
        for (int i = 0; i < g.vocab; ++i) {
            const float v = from_bf16(r[i]);
            if (v > bv) { bv = v; bi = i; }
        }
        std::printf("    (32 rows, the wide row block: argmax %d)\n", bi);
        expect(bi == 818, "the wide-BM GEMM agrees with mlx-lm");
    }

    // ── Two requests in one fire ──
    //
    // This is what a mixed prefill+decode batch is made of: several sequences
    // sharing a command buffer, each with its own position run and its own page
    // list. If a request's answer is the same whether it fires alone or beside a
    // neighbour, the CSR and the page tables keep them apart -- and that is the
    // property continuous batching rests on.
    //
    // Request 1 is a PREFIX of request 0, so its own last row has an answer this
    // test already knows: firing those four tokens ALONE gives 496 (the shape
    // the tap walk ran, `PIE_G4MB_TOKENS=2,818,3821,563`). It must give 496
    // again beside a longer sibling -- a different answer from request 0's, out
    // of the same fire, is the whole point.
    if (default_prompt) {
        const int pages_per_req = g.total_pages / 2;
        std::vector<std::uint32_t> two;
        two.insert(two.end(), ids.begin(), ids.end());          // request 0: all 8
        two.insert(two.end(), ids.begin(), ids.begin() + 4);    // request 1: first 4
        const int rows2 = int(two.size());
        // Zero the pages so neither request reads what the previous fire wrote.
        for (int L = 0; L < g.n_layers; ++L) {
            if (g.is_kv_shared(L)) continue;
            std::memset(kpages[std::size_t(L)].contents(), 0, kpages[std::size_t(L)].size);
            std::memset(vpages[std::size_t(L)].contents(), 0, vpages[std::size_t(L)].size);
        }
        fill_io(two, {0u, std::uint32_t(N)}, pages_per_req);
        bind_gemma4_consts(*ctx, dag, g, rows2, /*paged=*/true);
        ctx->run_step([&](StepEncoder& se) {
            encode_gemma4_step_mb(se, dag, g, rows2, base, mb, psos);
        });

        const auto argmax_of = [&](int row) {
            const auto* r = logits + std::size_t(row) * std::size_t(g.vocab);
            int bi = -1;
            float bv = -1e30f;
            for (int i = 0; i < g.vocab; ++i) {
                const float v = from_bf16(r[i]);
                if (v > bv) {
                    bv = v;
                    bi = i;
                }
            }
            return bi;
        };
        const int a0 = argmax_of(N - 1);          // request 0's last row
        const int a1 = argmax_of(rows2 - 1);      // request 1's last row
        std::printf("    (two requests in one fire: r0 last %d, r1 last %d)\n", a0, a1);
        expect(a0 == 3821,
               "a request's answer is unchanged by a sibling sharing its fire");
        expect(a1 == 496,
               "and the shorter request gets its OWN answer, from its own pages and "
               "positions -- not its neighbour's");
    }

    std::printf("\n==== gemma4_prefill_numerics_test: %s ====\n",
                failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
