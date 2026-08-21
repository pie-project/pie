pub mod facts;

use self::facts::Gemma3nFacts;
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, NativeKv};
use model_dsl::{self as dsl, MatW, NormW, Val, matmul};
use model_ir::trace::{DType, Dim, FireClass, ForwardPlan, NormVariant, Shape};

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
    fn new(l: u32, f: &Gemma3nFacts, norm_eps: f32, repr: model_dsl::WeightRepr) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let m = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr,
        };
        let n = |name: &str| NormW {
            name: w(name),
            variant: NormVariant::Gemma,
            per_head: None,
            layer: Some(l),
            eps: norm_eps,
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
                eps: norm_eps,
            },
            k_norm: NormW {
                name: w("k_norm"),
                variant: NormVariant::Gemma,
                per_head: Some(f.attn.head_dim),
                layer: Some(l),
                eps: norm_eps,
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

fn altup_coefs(x: &Val, norm: &NormW, router: &MatW, coefs: &MatW, scale: &str) -> Val {
    let n = dsl::cuda::rmsnorm(x, norm);
    let n = dsl::cuda::scalar_mul(&n, scale, None);
    let modality = matmul(&n, router);
    let modality = dsl::cuda::generated::tanh(&modality, modality.layer(), None);
    matmul(&modality, coefs)
}

pub fn gemma3n_cuda<W1: DtypeAxis, A: DtypeAxis, K: KvAxis>(
    facts: &Gemma3nFacts,
    class: FireClass,
    norm_eps: f32,
    rope_theta_global: f32,
    rope_theta_local: f32,
) -> ForwardPlan {

    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
        assert!(K::NATIVE_BF16);
    }

    let family = format!("gemma3n-{}-{}.cuda.{}", W1::NAME, K::NAME, class.suffix());
    let k = facts.altup.num_streams;
    let active = facts.altup.active;
    dsl::trace_named(&family, |t| {
        dsl::seam(t, &dsl::seam::IN, &[], None);
        let embedded = dsl::embed_with(t, "embed", facts.hidden, facts.vocab);
        let mut streams = dsl::cuda::generated::hc_expand(
            &embedded,
            (
                Shape(vec![Dim::Tokens, Dim::Const(k), Dim::Const(facts.hidden)]),
                DType::BF16,
            ),
            embedded.layer(),
            None,
        );

        let rt = dsl::rt(t);
        for l in 0..facts.layers() {

            let window_left = model_ir::facts::window_left_at(facts.window_left, l);
            let w = G3nLayerW::new(l, facts, norm_eps, W1::REPR);
            let active_in = dsl::select(&streams, active);

            let packed = altup_coefs(
                &active_in,
                &w.altup_norm,
                &w.altup_router,
                &w.altup_predict_coefs,
                &format!("layer.{l}.altup_scale"),
            );
            let pcoefs = dsl::cuda::generated::altup_unpack_predict_coefs(
                &packed,
                (
                    Shape(vec![Dim::Tokens, Dim::Const(k), Dim::Const(k)]),
                    DType::F32,
                ),
                packed.layer(),
                None,
            );
            let predictions = dsl::cuda::generated::altup_predict(
                &streams,
                &pcoefs,
                (
                    Shape(vec![Dim::Const(k), Dim::Tokens, Dim::Const(facts.hidden)]),
                    DType::BF16,
                ),
                streams.layer(),
                None,
            );

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

            let v = dsl::cuda::generated::rmsnorm_no_scale(
                &v,
                facts.attn.head_dim as i32,
                norm_eps,
                v.layer(),
                None,
            );

            let theta = if window_left < 0 {
                rope_theta_global
            } else {
                rope_theta_local
            };
            let (q, kk) = dsl::cuda::generated::rope_bf16(
                &q,
                &kk,
                facts.attn.heads as i32,
                facts.attn.kv_heads as i32,
                facts.attn.head_dim as i32,
                theta,
                false,
                &rt.positions(),
                q.layer(),
                None,
            );
            let kv = dsl::Kv::at(t, l);
            dsl::cuda::write_kv_to_pages(&kk, &v, &kv, facts.attn.kv_heads, facts.attn.head_dim);
            dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));
            let o = dsl::cuda::attention_for(
                class,
                &q,
                &kv,
                window_left,
                facts.attn.head_dim,
                0.0,
                0.0,
                facts.attn.kv_heads,
            )
            .expect("a plain attention statement produces its value");
            let o = dsl::attention_landing(&o, &w.o_proj, l);
            let o = dsl::cuda::rmsnorm(&o, &w.post_attn_norm);
            let mid = dsl::cuda::generated::residual_add(&p_active, &o, p_active.layer(), None);
            let mid = dsl::cuda::generated::residual_add(&mid, &lau, mid.layer(), None);
            let mid = dsl::cuda::scalar_mul(&mid, &format!("layer.{l}.laurel_scale"), None);

            let m = dsl::cuda::rmsnorm(&mid, &w.mlp_norm);
            let gate = matmul(&m, &w.gate_proj);
            let up = matmul(&m, &w.up_proj);
            let gate = if facts.is_sparse(l) {
                dsl::cuda::generated::gaussian_topk(
                    &gate,
                    facts.sparsity_std_mult(),
                    gate.layer(),
                    None,
                )
            } else {
                gate
            };
            let act = dsl::cuda::generated::geglu_tanh(&gate, &up, gate.layer(), None);
            let mlp = matmul(&act, &w.down_proj);
            let mlp = dsl::cuda::rmsnorm(&mlp, &w.post_mlp_norm);
            let activated = dsl::cuda::generated::residual_add(&mid, &mlp, mid.layer(), None);

            let packed = altup_coefs(
                &activated,
                &w.altup_correct_norm,
                &w.altup_correct_router,
                &w.altup_correct_coefs,
                &format!("layer.{l}.altup_correct_scale"),
            );
            let ccoefs = dsl::cuda::generated::altup_unpack_correct_coefs(
                &packed,
                (Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::F32),
                packed.layer(),
                None,
            );
            streams = dsl::cuda::generated::altup_correct(
                &predictions,
                &activated,
                &ccoefs,
                active as i32,
                predictions.layer(),
                None,
            );

            let ple = dsl::embed_with(
                t,
                &format!("layer.{l}.embed_per_layer"),
                facts.ple_width,

                facts.ple_vocab,
            );
            let g = matmul(&x, &w.ple_gate);
            let ple = dsl::sigmoid_gate_mul(&ple, &g);
            let ple = matmul(&ple, &w.ple_proj);
            for s in 0..k {
                if s == active {
                    continue;
                }
                let win = dsl::select(&streams, s);

                let _ = dsl::cuda::generated::residual_add(&win, &ple, win.layer(), None);
            }
        }

        let active_final = dsl::select(&streams, active);
        let target = dsl::cuda::generated::compute_rms(
            &active_final,
            (Shape(vec![Dim::Tokens]), DType::F32),
            active_final.layer(),
            None,
        );
        let y = dsl::cuda::generated::mean_streams(
            &streams,
            (
                Shape(vec![Dim::Tokens, Dim::Const(facts.hidden)]),
                DType::BF16,
            ),
            streams.layer(),
            None,
        );
        let y = dsl::cuda::generated::magnitude_rescale(&y, &target, y.layer(), None);
        dsl::logits_epilogue(
            t,
            &y,
            NormVariant::Gemma,
            false,
            facts.vocab,

            Some(crate::gemma_3n::project::FINAL_LOGIT_SOFTCAP),
            norm_eps,
        );
    })
}

pub type TraceFn = fn(&Gemma3nFacts, FireClass, f32, f32, f32) -> ForwardPlan;

pub type ShippedW1 = Bf16Ax;

pub type ShippedA = Bf16Ax;

pub type ShippedKv = NativeKv;

pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalogue![(
    "gemma3n-bf16-kv-bf16",
    gemma3n_cuda::<ShippedW1, ShippedA, ShippedKv>
),];
