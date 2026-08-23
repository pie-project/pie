pub mod facts;

use self::facts::{Gemma4CudaFacts, Gemma4Facts};
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, NativeKv};
use model_dsl::{self as dsl, MatW, NormW, matmul};
use model_ir::trace::{DType, Dim, FireClass, ForwardPlan, NormVariant, RopeKind, Shape};

struct Gemma4LayerW {
    attn_norm: NormW,
    post_attn_norm: NormW,
    pre_ffw_norm: NormW,
    post_ffw_norm: NormW,
    qkv: MatW,
    q_proj: MatW,
    k_proj: MatW,
    v_proj: MatW,
    o_proj: MatW,
    q_norm: NormW,
    k_norm: NormW,
    gate_up: MatW,
    gate_proj: MatW,
    up_proj: MatW,
    down: MatW,
    ple_gate: MatW,
    ple_proj: MatW,
    ple_norm: NormW,
}

impl Gemma4LayerW {
    fn new(l: u32, f: &Gemma4Facts, norm_eps: f32, repr: model_dsl::WeightRepr) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let d = f.head_dim_of(l);
        let mat = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr,
        };

        let norm = |name: &str| NormW {
            name: w(name),
            variant: NormVariant::Plain,
            per_head: None,
            layer: Some(l),
            eps: norm_eps,
        };
        let head_norm = |name: &str| NormW {
            name: w(name),
            variant: NormVariant::Plain,
            per_head: Some(d),
            layer: Some(l),
            eps: norm_eps,
        };
        Gemma4LayerW {
            attn_norm: norm("attn_norm"),
            post_attn_norm: norm("post_attn_norm"),
            pre_ffw_norm: norm("pre_ffw_norm"),
            post_ffw_norm: norm("post_ffw_norm"),
            qkv: mat("qkv", (f.q_heads + 2 * f.kv_heads) * d),
            q_proj: mat("q_proj", f.q_heads * d),
            k_proj: mat("k_proj", f.kv_heads * d),
            v_proj: mat("v_proj", f.kv_heads * d),
            o_proj: mat("o_proj", f.hidden),
            q_norm: head_norm("q_norm"),
            k_norm: head_norm("k_norm"),

            gate_up: mat("gate_up", 2 * f.intermediate_of(l)),
            gate_proj: mat("gate_proj", f.intermediate_of(l)),
            up_proj: mat("up_proj", f.intermediate_of(l)),
            down: mat("down", f.hidden),
            ple_gate: mat("ple_gate", f.ple_dim),
            ple_proj: mat("ple_proj", f.hidden),
            ple_norm: norm("ple_norm"),
        }
    }
}

pub fn gemma4_cuda<W1: DtypeAxis, A: DtypeAxis, K: KvAxis>(
    facts: &Gemma4Facts,
    cuda: &Gemma4CudaFacts,
    class: FireClass,
    norm_eps: f32,
) -> ForwardPlan {
    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
    }

    let family = format!("gemma4-{}-{}.cuda.{}", W1::NAME, K::NAME, class.suffix());
    let hidden = facts.hidden;
    dsl::trace_named(&family, |t| {
        dsl::seam(t, &dsl::seam::IN, &[], None);

        let mut y = dsl::cuda::scalar_mul(
            &dsl::embed_with(t, "embed", hidden, facts.vocab),
            "sqrt_hidden",
            Some((hidden as f32).sqrt()),
        );

        let ple_total = facts.layers * facts.ple_dim;
        let table = dsl::cuda::scalar_mul(
            &dsl::embed_with(t, "embed_per_layer", ple_total, facts.ple_vocab),
            "sqrt_ple_dim",
            Some((facts.ple_dim as f32).sqrt()),
        );

        let ple = matmul(
            &y,
            &MatW {
                name: "ple_model_proj".into(),
                width: ple_total,
                layer: None,
                repr: W1::REPR,
            },
        );
        let scaled =
            dsl::cuda::scalar_mul(&ple, "rsqrt_hidden", Some(1.0 / (hidden as f32).sqrt()));
        let normed_ple = dsl::cuda::rmsnorm(
            &scaled,
            &NormW {
                name: "ple_model_norm".into(),
                variant: NormVariant::Plain,
                per_head: Some(facts.ple_dim),
                layer: None,
                eps: norm_eps,
            },
        );

        let ple = dsl::cuda::generated::residual_add(&normed_ple, &table, normed_ple.layer(), None);
        let ple = dsl::cuda::scalar_mul(&ple, "rsqrt_2", Some(1.0 / 2f32.sqrt()));
        let ple_table = dsl::cuda::generated::transpose_bf16_nld_to_lnd(
            &ple,
            (
                Shape(vec![
                    Dim::Const(facts.layers),
                    Dim::Tokens,
                    Dim::Const(facts.ple_dim),
                ]),
                DType::BF16,
            ),
            facts.ple_dim as i32,
            None,
            None,
        );

        let mut normed = dsl::cuda::rmsnorm(
            &y,
            &Gemma4LayerW::new(0, facts, norm_eps, W1::REPR).attn_norm,
        );

        let rt = dsl::rt(t);
        for l in 0..facts.layers {
            let window_left = model_ir::facts::window_left_at(&cuda.window_left, l);
            let w = Gemma4LayerW::new(l, facts, norm_eps, W1::REPR);
            let full = facts.is_full_attn(l);
            let d = facts.head_dim_of(l);
            let shared = facts.is_kv_shared(l);

            let kv = dsl::Kv::at(t, facts.kv_source(l).unwrap_or(l));

            let fused_post =
                cuda.fused_qkv && K::NATIVE_BF16 && !full && !shared && class == FireClass::Decode;

            let attn_in = if fused_post {
                let packed = matmul(&normed, &w.qkv);
                dsl::cuda::generated::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
                    &packed,
                    (
                        Shape(vec![Dim::Tokens, Dim::Const(facts.q_heads * d)]),
                        DType::BF16,
                    ),
                    &w.q_norm.name,
                    &w.k_norm.name,
                    facts.kv_heads as i32,
                    d as i32,
                    &kv.cache(),
                    crate::gemma_4::project::ROPE_THETA_LOCAL,
                    w.q_norm.eps,
                    &rt.positions(),
                    &rt.row_valid(),
                    w.q_norm.layer,
                    Some(kv.state()),
                )
            } else if shared {
                let q = matmul(&normed, &w.q_proj);
                if full {
                    let q = dsl::cuda::rmsnorm(&q, &w.q_norm);
                    dsl::cuda::generated::rope_partial_q_bf16(
                        &q,
                        facts.global_rotary_dim as i32,
                        d as i32,
                        crate::gemma_4::project::ROPE_THETA_GLOBAL,
                        &rt.positions(),
                        q.layer(),
                        None,
                    )
                } else {
                    dsl::cuda::generated::q_rmsnorm_rope_bf16_rounded(
                        &q,
                        &w.q_norm.name,
                        d as i32,
                        crate::gemma_4::project::ROPE_THETA_LOCAL,
                        w.q_norm.eps,
                        &rt.positions(),
                        w.q_norm.layer,
                        None,
                    )
                }
            } else {
                let (q, k, v) = if cuda.fused_qkv {
                    let packed = matmul(&normed, &w.qkv);
                    dsl::split_qkv(&packed, facts.q_heads * d, facts.kv_heads * d)
                } else {
                    (
                        matmul(&normed, &w.q_proj),
                        matmul(&normed, &w.k_proj),
                        matmul(&normed, &w.v_proj),
                    )
                };

                dsl::seam(q.trace(), &dsl::seam::ATTN_QV, &[&q, &v], Some(l));
                let v =
                    dsl::cuda::generated::rmsnorm_no_scale(&v, d as i32, norm_eps, v.layer(), None);
                let (q, k) = if full {
                    let q = dsl::cuda::rmsnorm(&q, &w.q_norm);
                    let k = dsl::cuda::rmsnorm(&k, &w.k_norm);
                    dsl::rope_partial(
                        &q,
                        &k,
                        RopeKind::Standard,
                        facts.global_rotary_dim,
                        d,
                        crate::gemma_4::project::ROPE_THETA_GLOBAL,
                    )
                } else {
                    dsl::cuda::generated::qk_rmsnorm_rope_bf16_rounded(
                        &q,
                        &k,
                        &w.q_norm.name,
                        &w.k_norm.name,
                        d as i32,
                        crate::gemma_4::project::ROPE_THETA_LOCAL,
                        w.q_norm.eps,
                        &rt.positions(),
                        w.q_norm.layer,
                        None,
                    )
                };
                dsl::cuda::write_kv_to_pages(&k, &v, &kv, facts.kv_heads, d);
                q
            };

            dsl::seam(attn_in.trace(), &dsl::seam::ATTN_Q, &[&attn_in], Some(l));
            let a = match class {
                FireClass::Decode => dsl::cuda::attention_flashinfer_decode(
                    &attn_in,
                    &kv,
                    window_left,
                    facts.head_dim_of(l),
                    0.0,
                    1.0,
                ),
                FireClass::Prefill if d == 512 => dsl::cuda::attention_naive_paged(
                    &dsl::runtime::query_windows(&attn_in),
                    &kv,
                    window_left,
                    d,
                    facts.kv_heads_of(l),
                    1.0,
                    0.0,
                ),
                FireClass::Prefill => dsl::cuda::attention_flashinfer_prefill_planless(
                    &dsl::runtime::query_windows(&attn_in),
                    &kv,
                    window_left,
                    facts.head_dim_of(l),
                    0.0,
                    1.0,
                    facts.kv_heads_of(l),
                ),
            }
            .expect("the class states its attention");

            let attn_out = dsl::attention_landing(&a, &w.o_proj, l);

            let (landed, mlp_in) = dsl::cuda::generated::rmsnorm_residual_add_scale_rmsnorm_bf16(
                &attn_out,
                &w.post_attn_norm.name,
                &y,
                1.0,
                &w.pre_ffw_norm.name,
                w.post_attn_norm.eps,
                w.post_attn_norm.layer,
                None,
            );
            y = landed;

            let inter = facts.intermediate_of(l);
            let act = if cuda.gate_up_fused {
                dsl::cuda::geglu_tanh(&matmul(&mlp_in, &w.gate_up), inter, true)
            } else {
                let gate = matmul(&mlp_in, &w.gate_proj);
                let up = matmul(&mlp_in, &w.up_proj);
                dsl::cuda::generated::geglu_tanh(&gate, &up, gate.layer(), None)
            };
            let mlp_out = matmul(&act, &w.down);
            y = dsl::cuda::generated::rmsnorm_residual_add(
                &mlp_out,
                &w.post_ffw_norm.name,
                &y,
                w.post_ffw_norm.eps,
                w.post_ffw_norm.layer,
                None,
            );

            let gate = matmul(&y, &w.ple_gate);

            let slice = dsl::select(&ple_table, l);
            let gated = dsl::cuda::generated::geglu_tanh(&gate, &slice, gate.layer(), None);
            let ple_out = matmul(&gated, &w.ple_proj);
            if l + 1 < facts.layers {
                let next = Gemma4LayerW::new(l + 1, facts, norm_eps, W1::REPR);
                let (landed, next_norm) =
                    dsl::cuda::generated::rmsnorm_residual_add_scale_rmsnorm_bf16(
                        &ple_out,
                        &w.ple_norm.name,
                        &y,
                        cuda.layer_scalars.get(l as usize).copied().unwrap_or(1.0),
                        &next.attn_norm.name,
                        w.ple_norm.eps,
                        w.ple_norm.layer,
                        None,
                    );
                y = landed;
                normed = next_norm;
            } else {
                y = dsl::cuda::generated::rmsnorm_residual_add(
                    &ple_out,
                    &w.ple_norm.name,
                    &y,
                    w.ple_norm.eps,
                    w.ple_norm.layer,
                    None,
                );
            }
        }

        let normed = dsl::cuda::rmsnorm(
            &y,
            &NormW {
                name: "final_norm".into(),
                variant: NormVariant::Plain,
                per_head: None,
                layer: None,
                eps: norm_eps,
            },
        );
        let logits = dsl::lm_head_tied(t, &normed, facts.tied_embeddings, facts.vocab);
        let logits = if facts.logit_softcap > 0.0 {
            dsl::cuda::generated::logit_softcap(
                &logits,
                (
                    Shape(vec![Dim::Requests, Dim::Const(facts.vocab)]),
                    DType::BF16,
                ),
                facts.logit_softcap,
                None,
                None,
            )
        } else {
            logits
        };

        dsl::seam(t, &dsl::seam::OUT, &[&logits], None);
    })
}

pub type TraceFn = fn(&Gemma4Facts, &Gemma4CudaFacts, FireClass, f32) -> ForwardPlan;

pub type ShippedW1 = Bf16Ax;

pub type ShippedA = Bf16Ax;

pub type ShippedKv = NativeKv;

pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalogue![(
    "gemma4-bf16-kv-bf16",
    gemma4_cuda::<ShippedW1, ShippedA, ShippedKv>
),];
