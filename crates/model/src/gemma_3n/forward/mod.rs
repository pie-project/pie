//! gemma3n's forward, declared.
//!
//! Transcribed from `gemma3n.cpp`. The second rank-K residual in the
//! tree, and it shares no statement with the first:
//!
//! * deepseek_v4's hyper-connections MIX K streams every layer —
//!   `hc_pre` reads a mix, `hc_post` writes one back, and no stream is
//!   ever singled out.
//! * gemma3n's AltUp PREDICTS the other streams from the active one,
//!   runs the layer body on that prediction, then CORRECTS all of them
//!   from the result.
//!
//! That difference is what asked the IR two new questions, and both are
//! now answered: `select` states the window the body READS, and
//! `kernel!`'s `in_place` states that the per-layer embedding's add
//! LANDS in the window it read. Neither is gemma3n-specific; both are
//! things the DSL simply could not say before.
//!
//! Both coefficient sets come from the SAME three-statement shape — norm,
//! scale, a projection through `tanh`, a projection — and then an
//! unpack. `unpack_predict_coefs` and `unpack_correct_coefs` are separate
//! symbols because they unpack different layouts, not because the shape
//! differs.
//!
//! Two more things belong to this family alone. `laurel` is a low-rank
//! branch that lands beside attention and is normed before it does. And
//! the per-layer embeddings are read PER LAYER, gated, and added to the
//! corrected streams EXCEPT the active one — which is what makes them a
//! residual input rather than a second embedding.

pub mod facts;

use self::facts::Gemma3nFacts;
use model_dsl::{self as dsl, MatW, NormW, Val, WeightRepr, matmul};
use model_ir::trace::{FireClass, ForwardPlan, NormVariant};

struct G3nLayerW {
    altup_norm: NormW,
    altup_router: MatW,
    altup_predict_coefs: MatW,
    altup_correct_norm: NormW,
    altup_correct_router: MatW,
    altup_correct_coefs: MatW,
    attn_norm: NormW,
    q_proj: MatW,
    k_proj: MatW,
    v_proj: MatW,
    q_norm: NormW,
    k_norm: NormW,
    o_proj: MatW,
    post_attn_norm: NormW,
    laurel_left: MatW,
    laurel_right: MatW,
    laurel_post_norm: NormW,
    mlp_norm: NormW,
    gate_proj: MatW,
    up_proj: MatW,
    down_proj: MatW,
    post_mlp_norm: NormW,
    ple_gate: MatW,
    ple_proj: MatW,
}

impl G3nLayerW {
    fn new(l: u32, f: &Gemma3nFacts) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let m = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr: WeightRepr::Bf16,
        };
        let n = |name: &str| NormW {
            name: w(name),
            variant: NormVariant::Gemma,
            per_head: None,
            layer: Some(l),
        };
        let k = f.altup.num_streams;
        Self {
            altup_norm: n("altup_norm"),
            altup_router: m("altup_router", k),
            altup_predict_coefs: m("altup_predict_coefs", k * k),
            altup_correct_norm: n("altup_correct_norm"),
            altup_correct_router: m("altup_correct_router", k),
            altup_correct_coefs: m("altup_correct_coefs", k),
            attn_norm: n("attn_norm"),
            q_proj: m("q_proj", f.attn.q_width()),
            k_proj: m("k_proj", f.attn.kv_width()),
            v_proj: m("v_proj", f.attn.kv_width()),
            q_norm: NormW {
                name: w("q_norm"),
                variant: NormVariant::Gemma,
                per_head: Some(f.attn.head_dim),
                layer: Some(l),
            },
            k_norm: NormW {
                name: w("k_norm"),
                variant: NormVariant::Gemma,
                per_head: Some(f.attn.head_dim),
                layer: Some(l),
            },
            o_proj: m("o_proj", f.hidden),
            post_attn_norm: n("post_attn_norm"),
            laurel_left: m("laurel_left", f.laurel_rank),
            laurel_right: m("laurel_right", f.hidden),
            laurel_post_norm: n("laurel_post_norm"),
            mlp_norm: n("mlp_norm"),
            gate_proj: m("gate_proj", f.intermediate(l)),
            up_proj: m("up_proj", f.intermediate(l)),
            down_proj: m("down_proj", f.hidden),
            post_mlp_norm: n("post_mlp_norm"),
            ple_gate: m("ple_input_gate", f.ple_width),
            ple_proj: m("ple_projection", f.hidden),
        }
    }
}

/// The coefficient shape both AltUp halves share, stated once because it
/// IS one shape: the halves differ only in which weights they read and
/// how the result unpacks.
fn altup_coefs(x: &Val, norm: &NormW, router: &MatW, coefs: &MatW, scale: &str) -> Val {
    let n = dsl::cuda::rmsnorm(x, norm);
    let n = dsl::cuda::scalar_mul(&n, scale, None);
    let modality = matmul(&n, router);
    let modality = dsl::cuda::tanh(&modality);
    matmul(&modality, coefs)
}

/// gemma3n's CUDA text for one fire class.
pub fn gemma3n_cuda(facts: &Gemma3nFacts, class: FireClass) -> ForwardPlan {
    // DECODE AND PREFILL, and the difference is one call. This family
    // served Decode only and PANICKED on anything else — not refused,
    // panicked, on the first prefill a serving deployment sends. The
    // class-dependent sites in this text number exactly one, the
    // attention op, which `dsl::cuda::attention_for` now holds.
    let family = format!("gemma3n.cuda.{}", class.suffix());
    let k = facts.altup.num_streams;
    let active = facts.altup.active;
    dsl::trace_named(&family, |t| {
        dsl::seam(t, &dsl::seam::IN, &[], None);
        let embedded = dsl::embed_with(t, "embed", facts.hidden);
        let mut streams = dsl::cuda::hc_expand(&embedded, k, facts.hidden);

        for l in 0..facts.layers() {
            // THIS LAYER's sliding window, `-1` for none — a
            // load-time fact the dispatch statements carry, where four
            // executors used to re-derive it per launch.
            let window_left = model_ir::facts::window_left_at(facts.window_left, l);
            let w = G3nLayerW::new(l, facts);
            let active_in = dsl::select(&streams, active);

            // ── AltUp.predict ────────────────────────────────────────
            let packed = altup_coefs(
                &active_in,
                &w.altup_norm,
                &w.altup_router,
                &w.altup_predict_coefs,
                &format!("layer.{l}.altup_scale"),
            );
            let pcoefs = dsl::cuda::altup_unpack_predict_coefs(&packed, k);
            let predictions = dsl::cuda::altup_predict(&streams, &pcoefs, k, facts.hidden);

            // ── The layer body, on the ACTIVE prediction ─────────────
            // `select` is the whole reason this line can exist: in
            // `gemma3n.cpp` it is `predictions + active * N * H`, a
            // pointer offset with no kernel behind it.
            let p_active = dsl::select(&predictions, active);
            let x = dsl::cuda::rmsnorm(&p_active, &w.attn_norm);

            let lau = matmul(&x, &w.laurel_left);
            let lau = matmul(&lau, &w.laurel_right);
            let lau = dsl::cuda::rmsnorm(&lau, &w.laurel_post_norm);

            let q = matmul(&x, &w.q_proj);
            let kk = matmul(&x, &w.k_proj);
            let v = matmul(&x, &w.v_proj);
            let q = dsl::cuda::rmsnorm(&q, &w.q_norm);
            let kk = dsl::cuda::rmsnorm(&kk, &w.k_norm);
            // The value takes the SCALE-LESS norm: gemma3n norms v too,
            // and with no gamma.
            let v = dsl::cuda::rmsnorm_no_scale(&v);
            let (q, kk) = dsl::cuda::rope(&q, &kk);
            let kv = dsl::Kv::at(t, l);
            dsl::cuda::write_kv_to_pages(&kk, &v, &kv);
            dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));
            let o = dsl::cuda::attention_for(class, &q, &kv, window_left, facts.attn.head_dim)
                .expect("a plain attention statement produces its value");
            let o = dsl::attention_landing(&o, &w.o_proj, l);
            let o = dsl::cuda::rmsnorm(&o, &w.post_attn_norm);
            let mid = dsl::cuda::residual_add(&p_active, &o, facts.hidden);
            let mid = dsl::cuda::residual_add(&mid, &lau, facts.hidden);
            let mid = dsl::cuda::scalar_mul(&mid, &format!("layer.{l}.laurel_scale"), None);

            // ── MLP ──────────────────────────────────────────────────
            let m = dsl::cuda::rmsnorm(&mid, &w.mlp_norm);
            let gate = matmul(&m, &w.gate_proj);
            let up = matmul(&m, &w.up_proj);
            let gate = if facts.is_sparse(l) {
                dsl::cuda::gaussian_topk(&gate, facts.intermediate(l), facts.sparsity_std_mult())
            } else {
                gate
            };
            let act = dsl::cuda::geglu_tanh_pair(&gate, &up, facts.intermediate(l));
            let mlp = matmul(&act, &w.down_proj);
            let mlp = dsl::cuda::rmsnorm(&mlp, &w.post_mlp_norm);
            let activated = dsl::cuda::residual_add(&mid, &mlp, facts.hidden);

            // ── AltUp.correct ────────────────────────────────────────
            let packed = altup_coefs(
                &activated,
                &w.altup_correct_norm,
                &w.altup_correct_router,
                &w.altup_correct_coefs,
                &format!("layer.{l}.altup_correct_scale"),
            );
            let ccoefs = dsl::cuda::altup_unpack_correct_coefs(&packed, k);
            streams = dsl::cuda::altup_correct(&predictions, &activated, &ccoefs, k, facts.hidden);

            // ── PLE: gated, added into every stream but the active ───
            // K-1 in-place adds, each through its own window — the
            // driver's `for (k) if (k != act_idx) residual_add(...)`,
            // launch for launch. `in_place` is what makes each land in
            // the window it read instead of in a value nothing reads.
            let ple = dsl::embed_with(t, &format!("layer.{l}.embed_per_layer"), facts.ple_width);
            let g = matmul(&x, &w.ple_gate);
            let ple = dsl::sigmoid_gate_mul(&ple, &g);
            let ple = matmul(&ple, &w.ple_proj);
            for s in 0..k {
                if s == active {
                    continue;
                }
                let win = dsl::select(&streams, s);
                // `let _`, because the K-1 per-stream PLE adds are IN PLACE
                // and the value the statement produces is `win` again.
                // `builder!` blanket-`#[must_use]`s what it generates
                // (`model-dsl/src/cuda/mod.rs:278`), which is right for a
                // surface where dropping a result is normally a lost launch;
                // this is the one site on the surface where it is not.
                let _ = dsl::cuda::residual_add(&win, &ple, facts.hidden);
            }
        }

        // The streams collapse by their MEAN, rescaled to the magnitude a
        // single stream would have had.
        let active_final = dsl::select(&streams, active);
        let target = dsl::cuda::compute_rms(&active_final);
        let y = dsl::cuda::mean_streams(&streams, facts.hidden);
        let y = dsl::cuda::magnitude_rescale(&y, &target, facts.hidden);
        dsl::logits_epilogue(t, &y, NormVariant::Gemma, false, facts.vocab, true);
    })
}
