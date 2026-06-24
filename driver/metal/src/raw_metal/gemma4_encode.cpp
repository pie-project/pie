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

#include "decode_dispatch.hpp"  // shared raw_metal:: launch helpers (qmv/rms/rope/sdpa/...)

namespace pie_metal_driver::raw_metal::gemma4 {

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

void encode_gemma4_step(StepEncoder& se,
                        const std::vector<Gemma4Dispatch>& dag,
                        const Gemma4StepPsos& psos,
                        const Gemma4Geometry& geom,
                        bool force_barriers,
                        BarrierVisibility vis) {
    (void)force_barriers;  // gemma4 has no ‖-pair concurrency model yet → barrier after each.
    Grid grid; Threadgroup tg;
    for (const Gemma4Dispatch& d : dag) {
        const Pso& pso = (d.kind == Kernel::Sdpa && Gemma4Geometry::is_full_attn(d.layer))
                             ? psos.sdpa_full : psos[d.kind];
        se.set_pso(pso);
        se.set_argtable_ordinal(d.ordinal);   // flat-ordinal key (ratified)
        gemma4_launch_dims(d.kind, d.layer, geom, grid, tg);
        se.dispatch(grid, tg);
        se.barrier(vis);
    }
}

}  // namespace pie_metal_driver::raw_metal::gemma4
