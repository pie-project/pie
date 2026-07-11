// gemma4_encode.cpp — compile the gemma4 decode kernels + encode the per-token DAG.
//
// PSO table (load_gemma4_psos): one runtime-compiled PSO per distinct (file, entrypoint),
// fanned out to every gemma4 Kernel kind it serves. The 4-bit path reuses delta's shared
// kernels (affine_qmv for all linears, 4-bit embed_gather for the tied embed/PLE tables,
// rms_single_row, rope, kv_append, residual_add) + alpha's gemma kernels (geglu_tanh,
// vnorm, ple_combine, logit_softcap, layer_scalar, sdpa_sliding). Per-model flags
// (rms plus_one=false, sdpa window, rope theta, softcap cap) are RmsParams/IO-bound at
// the consts stage — the PSOs are flag-independent.
//
// encode_gemma4_step: mirrors beta's encode_decode_step (proven at qwen3.6 argmax-264) —
// walk the pure DAG, set pso + arg table (flat ordinal) + launch dims, dispatch + barrier.

#include "gemma4_encode.hpp"

#include <set>

#include "decode_dispatch.hpp"  // shared native launch helpers (qmv/rms/rope/sdpa/...)

namespace pie::metal::gemma4 {

namespace {

// One distinct PSO: source file + entrypoint, and the gemma4 Kernel kinds it serves.
struct PsoSpec {
    const char*         file;
    const char*         fn;
    std::vector<Kernel> kinds;
};

// 4-bit (q_bits==4) spec set — the shipped gemma4. Entrypoints are the bf16 activation /
// g64-b4 weight instantiations, head_dim 256 sdpa.
std::vector<PsoSpec> specs_4bit() {
    return {
        // 4-bit dequant gather, SCALED (gemma4): tied embed_tokens (== lm_head bundle,
        // *sqrt(hidden)) + the per-layer embed table (*sqrt(ple_dim)). The scale (buffer 6)
        // is bound per-dispatch at the consts stage; qwen's unscaled embed_gather is untouched.
        {"embed_gather.metal", "embed_gather_scaled_4bit_bfloat16_gs_64_b_4",
            {Kernel::EmbedGather, Kernel::PleTokenGather}},
        // rms_single_row serves every (plain-rms, plus_one bound false) norm.
        {"rms_norm.metal", "rms_single_row_bfloat16",
            {Kernel::PleProjNorm, Kernel::AttnNorm, Kernel::QNorm, Kernel::KNorm,
             Kernel::PostAttnNorm, Kernel::FfnNorm, Kernel::PostFfnNorm,
             Kernel::PleNorm, Kernel::FinalRms}},
        // affine_qmv: every 4-bit linear (q/k/v/o + gate/up/down + PLE projections +
        // tied lm_head logits).
        {"quantized_qmv.metal", "affine_qmv_fast_bfloat16_gs_64_b_4",
            {Kernel::PleProjGemv, Kernel::QmvQ, Kernel::QmvK, Kernel::QmvV, Kernel::QmvO,
             Kernel::QmvGate, Kernel::QmvUp, Kernel::QmvDown,
             Kernel::PleGateGemv, Kernel::PleProjLayerGemv, Kernel::LmHead}},
        // alpha's gemma-specific pointwise + sliding SDPA.
        {"vnorm.metal",        "vnorm_single_row_bfloat16",       {Kernel::VNorm}},
        {"rope.metal",         "rope_neox_decode_bfloat16",       {Kernel::RopeQ, Kernel::RopeK}},
        {"kv_append.metal",    "kv_append_bfloat16",              {Kernel::KvAppend}},
        {"sdpa_sliding.metal", "sdpa_vector_decode_swa_bfloat16_d_256", {Kernel::Sdpa}},
        {"residual_add.metal", "residual_add_bfloat16",
            {Kernel::AttnResidual, Kernel::FfnResidual, Kernel::PleResidual}},
        {"geglu_tanh.metal",   "geglu_tanh_bfloat16",             {Kernel::GegluTanh, Kernel::PleGeglu}},
        {"ple_combine.metal",  "ple_combine_bfloat16",            {Kernel::PleCombine}},
        {"layer_scalar.metal", "layer_scalar_mul_bfloat16",       {Kernel::LayerScalar}},
        {"logit_softcap.metal","logit_softcap_bfloat16",          {Kernel::FinalSoftcap}},
    };
}

}  // namespace

bool load_gemma4_psos(RawMetalContext& ctx,
                      const std::string& kernels_dir,
                      const Gemma4Geometry& geom,
                      Gemma4StepPsos& out,
                      bool with_argmax,
                      std::string* err) {
    if (geom.q_bits != 4) {
        if (err) *err = "only the 4-bit gemma4 path is wired (q_bits must be 4; bf16 "
                        "fallback was retired on accuracy grounds)";
        return false;
    }
    const std::string dir = kernels_dir.empty() || kernels_dir.back() == '/'
                                ? kernels_dir : kernels_dir + "/";
    for (const PsoSpec& spec : specs_4bit()) {
        std::string e;
        Pso pso = ctx.compile_pso_from_file(dir + spec.file, spec.fn, &e);
        if (!pso.valid()) {
            if (err) *err = std::string(spec.fn) + " (" + spec.file + "): " + e;
            return false;
        }
        for (Kernel k : spec.kinds) out[k] = pso;
    }
    // Full-attention layers (head_dim 512) need the d_512 SDPA instantiation; by_kind[Sdpa]
    // (d_256) serves the sliding layers. The encode_fn picks per-layer via is_full_attn().
    {
        std::string e;
        Pso pso = ctx.compile_pso_from_file(
            dir + "sdpa_sliding.metal", "sdpa_vector_decode_swa_bfloat16_d_512", &e);
        if (!pso.valid()) {
            if (err) *err = std::string("sdpa_vector_decode_swa_bfloat16_d_512 "
                                        "(sdpa_sliding.metal): ") + e;
            return false;
        }
        out.sdpa_full = pso;
    }
    if (with_argmax) {
        // Argmax is the optional I3 device-argmax substrate; kernel not yet ported.
        // When it lands, compile + assign out[Kernel::Argmax] here.
    }
    return true;
}

void gemma4_launch_dims(Kernel kind, int layer, const Gemma4Geometry& g,
                        Grid& grid, Threadgroup& tg) {
    const int hd      = g.head_dim_at(layer);            // 256 sliding / 512 full
    const int q_dim   = g.q_dim_at(layer);               // 2048 / 4096
    const int kv_dim  = g.kv_dim_at(layer);              // 256  / 512
    const int rotary  = g.rotary_at(layer);              // 256  / 128
    const int interm  = g.intermediate_at(layer);        // 6144 / 12288
    const int ple_w   = g.n_layers * g.per_layer_emb_dim; // 8960 (PLE table row width)

    // Elementwise pointwise: one thread per output element, 256-wide threadgroups.
    auto pointwise = [&](int n) { grid = Grid{uint32_t(n), 1, 1}; tg = Threadgroup{256, 1, 1}; };

    switch (kind) {
        // ── PLE precompute ──
        case Kernel::EmbedGather:    embed_dispatch(g.hidden, grid, tg); break;
        case Kernel::PleTokenGather: embed_dispatch(ple_w, grid, tg); break;
        case Kernel::PleProjGemv:    qmv_dispatch(ple_w, grid, tg); break;            // [hidden]->[L*ple]
        case Kernel::PleProjNorm:    rms_dispatch(g.per_layer_emb_dim, g.n_layers, grid, tg); break;
        case Kernel::PleCombine:     pointwise(ple_w); break;

        // ── attention (head_dim per layer type) ──
        case Kernel::AttnNorm:     rms_dispatch(g.hidden, 1, grid, tg); break;
        case Kernel::QmvQ:         qmv_dispatch(q_dim, grid, tg); break;
        case Kernel::QmvK:         qmv_dispatch(kv_dim, grid, tg); break;
        case Kernel::QmvV:         qmv_dispatch(kv_dim, grid, tg); break;
        case Kernel::QNorm:        rms_dispatch(hd, g.n_q_heads, grid, tg); break;
        case Kernel::KNorm:        rms_dispatch(hd, g.n_kv_heads, grid, tg); break;
        case Kernel::VNorm:        rms_dispatch(hd, g.n_kv_heads, grid, tg); break;  // vnorm == rms launch
        case Kernel::RopeQ:        rope_dispatch(rotary, g.n_q_heads, grid, tg); break;
        case Kernel::RopeK:        rope_dispatch(rotary, g.n_kv_heads, grid, tg); break;
        case Kernel::KvAppend:     kv_append_dispatch(hd, g.n_kv_heads, grid, tg); break;
        case Kernel::Sdpa:         sdpa_dispatch(g.n_q_heads, grid, tg); break;       // sliding via window bind
        case Kernel::QmvO:         qmv_dispatch(g.hidden, grid, tg); break;
        case Kernel::PostAttnNorm: rms_dispatch(g.hidden, 1, grid, tg); break;
        case Kernel::AttnResidual: residual_dispatch(g.hidden, grid, tg); break;

        // ── FFN (GeGLU-tanh; double-wide on shared layers) ──
        case Kernel::FfnNorm:     rms_dispatch(g.hidden, 1, grid, tg); break;
        case Kernel::QmvGate:     qmv_dispatch(interm, grid, tg); break;
        case Kernel::QmvUp:       qmv_dispatch(interm, grid, tg); break;
        case Kernel::GegluTanh:   pointwise(interm); break;
        case Kernel::QmvDown:     qmv_dispatch(g.hidden, grid, tg); break;
        case Kernel::PostFfnNorm: rms_dispatch(g.hidden, 1, grid, tg); break;
        case Kernel::FfnResidual: residual_dispatch(g.hidden, grid, tg); break;

        // ── PLE residual + layer scalar ──
        case Kernel::PleGateGemv:      qmv_dispatch(g.per_layer_emb_dim, grid, tg); break; // [hidden]->[ple]
        case Kernel::PleGeglu:         pointwise(g.per_layer_emb_dim); break;
        case Kernel::PleProjLayerGemv: qmv_dispatch(g.hidden, grid, tg); break;       // [ple]->[hidden]
        case Kernel::PleNorm:          rms_dispatch(g.hidden, 1, grid, tg); break;    // post_per_layer_input_norm: [hidden]
        case Kernel::PleResidual:      residual_dispatch(g.hidden, grid, tg); break;
        case Kernel::LayerScalar:      pointwise(g.hidden); break;

        // ── tail ──
        case Kernel::FinalRms:     rms_dispatch(g.hidden, 1, grid, tg); break;
        case Kernel::LmHead:       qmv_dispatch(g.vocab, grid, tg); break;
        case Kernel::FinalSoftcap: pointwise(g.vocab); break;
        case Kernel::Argmax:       grid = Grid{1024, 1, 1}; tg = Threadgroup{1024, 1, 1}; break;
    }
}

// ── Concurrency: the general dependency-driven RAW-barrier predicate (alpha deps + beta algo) ──
// gemma4 scratch is NO-RECYCLE (one fresh buffer per ordinal, never reused) ⇒ no WAR/WAW
// hazards, ONLY RAW. A `se.barrier()` is global (drains all in-flight). So the minimal correct
// schedule is greedy: emit a barrier before dispatch i IFF i reads any buffer written since the
// last barrier (an un-drained RAW). This auto-captures Gate‖Up + the Q/K/V attention cluster +
// every independent norm/rope/elementwise pair — no hand-picked pairs, can't forget one, and
// (unlike qwen's recycler) zero `concurrent_after`/coloring.
//
// `gemma4_build_io` is alpha's half: the per-dispatch activation read/write buffer-id sets,
// mirroring wire_dataflow's register tracking. Buffer-ids:
//   • activation pool slot   → the producing dispatch's ordinal (no_recycle ⇒ one writer/slot,
//                               EXCEPT in-place RoPE which re-writes qn/kn → the id is reused,
//                               which the set-based predicate handles correctly).
//   • KV[layer]              → kKvBase+layer.  KvAppend(L) WRITES it; Sdpa(L) READS kv_source(L).
//                               This is a LIVE intra-step RAW (Sdpa attends the just-appended
//                               current token) — modeling it keeps the critical KvAppend→Sdpa
//                               barrier. (Shared L15-34 read an earlier, already-drained append
//                               ⇒ correct-by-construction, costs no barrier.)
//   • logits / logits_capped → kLogits/kCapped, so FinalSoftcap can't race the LmHead matmul.
// Only true read-only operands (weights, scales, consts, pages, pos) are excluded.
struct DispatchIO { std::vector<int> reads, writes; };

std::vector<DispatchIO> gemma4_build_io(const std::vector<Gemma4Dispatch>& dag,
                                        const Gemma4Geometry& g) {
    constexpr int kKvBase = 1 << 20, kLogits = (2 << 20), kCapped = (2 << 20) + 1;
    std::vector<DispatchIO> io(dag.size());
    // buffer-id currently held by each logical register (-1 = a read-only source ⇒ no RAW)
    int hidden = -1, embed = -1, ple_token = -1, ple_proj = -1, ple_projn = -1, ple_input = -1;
    int normed = -1, q = -1, qn = -1, k = -1, v = -1, kn = -1, vn = -1, sdpa_o = -1, o = -1;
    int fn = -1, gate = -1, up = -1, geglu = -1, down = -1;
    int plegate = -1, plegated = -1, pleproj = -1, finalnorm = -1, post_tmp = -1, logits = -1;
    std::vector<int> kv_producer(g.n_layers, -1);
    for (size_t i = 0; i < dag.size(); ++i) {
        const Gemma4Dispatch& d = dag[i];
        const int id = d.ordinal;          // == i; the pool slot this dispatch allocates
        const int L  = d.layer;
        std::vector<int>& R = io[i].reads;
        std::vector<int>& W = io[i].writes;
        auto rd = [&](int p) { if (p >= 0) R.push_back(p); };
        switch (d.kind) {
            case Kernel::EmbedGather:    embed = hidden = id; W = {id}; break;
            case Kernel::PleTokenGather: ple_token = id; W = {id}; break;
            case Kernel::PleProjGemv:    rd(embed); ple_proj = id; W = {id}; break;
            case Kernel::PleProjNorm:    rd(ple_proj); ple_projn = id; W = {id}; break;
            case Kernel::PleCombine:     rd(ple_projn); rd(ple_token); ple_input = id; W = {id}; break;
            case Kernel::AttnNorm:       rd(hidden); normed = id; W = {id}; break;
            case Kernel::QmvQ:           rd(normed); q = id; W = {id}; break;
            case Kernel::QNorm:          rd(q); qn = id; W = {id}; break;
            case Kernel::QmvK:           rd(normed); k = id; W = {id}; break;
            case Kernel::QmvV:           rd(normed); v = id; W = {id}; break;
            case Kernel::KNorm:          rd(k); kn = id; W = {id}; break;
            case Kernel::VNorm:          rd(v); vn = id; W = {id}; break;
            case Kernel::RopeK:          rd(kn); W = {kn}; break;   // in-place: re-writes kn
            case Kernel::RopeQ:          rd(qn); W = {qn}; break;   // in-place: re-writes qn
            case Kernel::KvAppend:       rd(kn); rd(vn); kv_producer[L] = kKvBase + L;
                                         W = {kKvBase + L}; break;
            case Kernel::Sdpa:           rd(qn); rd(kv_producer[g.kv_source(L)]);
                                         sdpa_o = id; W = {id}; break;
            case Kernel::QmvO:           rd(sdpa_o); o = id; W = {id}; break;
            case Kernel::PostAttnNorm:   rd(o); post_tmp = id; W = {id}; break;
            case Kernel::AttnResidual:   rd(post_tmp); rd(hidden); hidden = id; W = {id}; break;
            case Kernel::FfnNorm:        rd(hidden); fn = id; W = {id}; break;
            case Kernel::QmvGate:        rd(fn); gate = id; W = {id}; break;
            case Kernel::QmvUp:          rd(fn); up = id; W = {id}; break;
            case Kernel::GegluTanh:      rd(gate); rd(up); geglu = id; W = {id}; break;
            case Kernel::QmvDown:        rd(geglu); down = id; W = {id}; break;
            case Kernel::PostFfnNorm:    rd(down); post_tmp = id; W = {id}; break;
            case Kernel::FfnResidual:    rd(post_tmp); rd(hidden); hidden = id; W = {id}; break;
            case Kernel::PleGateGemv:    rd(hidden); plegate = id; W = {id}; break;
            case Kernel::PleGeglu:       rd(plegate); rd(ple_input); plegated = id; W = {id}; break;
            case Kernel::PleProjLayerGemv: rd(plegated); pleproj = id; W = {id}; break;
            case Kernel::PleNorm:        rd(pleproj); post_tmp = id; W = {id}; break;
            case Kernel::PleResidual:    rd(post_tmp); rd(hidden); hidden = id; W = {id}; break;
            case Kernel::LayerScalar:    rd(hidden); hidden = id; W = {id}; break;
            case Kernel::FinalRms:       rd(hidden); finalnorm = id; W = {id}; break;
            case Kernel::LmHead:         rd(finalnorm); logits = kLogits; W = {kLogits}; break;
            case Kernel::FinalSoftcap:   rd(logits); W = {kCapped}; break;
            case Kernel::Argmax:         break;
        }
    }
    return io;
}

// beta's greedy algorithm (local mirror of gemma4_barrier_plan.hpp `plan_barriers`, pending the
// beta-rawmetal merge — swap to the shared header then). Returns barrier_BEFORE[i].
std::vector<char> plan_barriers_greedy(const std::vector<DispatchIO>& io) {
    const size_t N = io.size();
    std::vector<char> before(N, 0);
    std::set<int> inflight;                       // buffer-ids written since the last barrier
    for (size_t i = 0; i < N; ++i) {
        bool hazard = false;
        for (int r : io[i].reads)
            if (inflight.count(r)) { hazard = true; break; }
        if (hazard) { before[i] = 1; inflight.clear(); }
        for (int w : io[i].writes) inflight.insert(w);
    }
    return before;
}

// Resolve the per-dispatch barrier_AFTER plan for the encode loop.
// concur:  0 = barrier after every dispatch (proven baseline);
//          1 = +Gate‖Up only (the hand-picked safe pair, kept for A/B);
//          2 = general greedy RAW predicate (auto-captures all independent clusters);
//         -1 = drop ALL barriers (TIMING-ONLY ceiling — WRONG argmax, never shipped).
std::vector<char> gemma4_barrier_plan(const std::vector<Gemma4Dispatch>& dag,
                                      const Gemma4Geometry& g, int concur) {
    const size_t N = dag.size();
    if (concur < 0) return std::vector<char>(N, 0);             // ceiling
    if (concur == 0) return std::vector<char>(N, 1);            // barrier-each
    if (concur == 1) {                                          // Gate‖Up
        std::vector<char> after(N, 1);
        for (size_t i = 0; i + 1 < N; ++i)
            if (dag[i].kind == Kernel::QmvGate && dag[i + 1].kind == Kernel::QmvUp &&
                dag[i].layer == dag[i + 1].layer)
                after[i] = 0;
        return after;
    }
    // concur >= 2: greedy. Map barrier_before → barrier_after (after[i] == before[i+1]); the
    // trailing write needs no barrier (cmd-buffer completion drains it before the host read).
    std::vector<char> before = plan_barriers_greedy(gemma4_build_io(dag, g));
    std::vector<char> after(N, 0);
    for (size_t i = 0; i + 1 < N; ++i) after[i] = before[i + 1];
    return after;
}

int gemma4_count_barriers(const std::vector<char>& plan) {
    int n = 0;
    for (char c : plan) n += (c != 0);
    return n;
}

void encode_gemma4_step(StepEncoder& se,
                        const std::vector<Gemma4Dispatch>& dag,
                        const Gemma4StepPsos& psos,
                        const Gemma4Geometry& geom,
                        bool force_barriers,
                        BarrierVisibility vis,
                        int concur) {
    (void)force_barriers;
    const std::vector<char> barrier_after = gemma4_barrier_plan(dag, geom, concur);
    Grid grid; Threadgroup tg;
    for (size_t i = 0; i < dag.size(); ++i) {
        const Gemma4Dispatch& d = dag[i];
        const Pso& pso = (d.kind == Kernel::Sdpa && Gemma4Geometry::is_full_attn(d.layer))
                             ? psos.sdpa_full : psos[d.kind];
        se.set_pso(pso);
        se.set_argtable_ordinal(d.ordinal);   // flat-ordinal key (ratified)
        gemma4_launch_dims(d.kind, d.layer, geom, grid, tg);
        se.dispatch(grid, tg);
        if (barrier_after[i]) se.barrier(vis);
    }
}

// Public banner helper: barriers emitted by `concur` vs the barrier-each baseline (dag.size()).
int gemma4_plan_barrier_count(const std::vector<Gemma4Dispatch>& dag,
                              const Gemma4Geometry& g, int concur) {
    return gemma4_count_barriers(gemma4_barrier_plan(dag, g, concur));
}

}  // namespace pie::metal::gemma4
