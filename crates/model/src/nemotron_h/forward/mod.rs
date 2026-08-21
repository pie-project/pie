pub mod facts;

use self::facts::{NemotronHFacts, NemotronLayerKind};
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, NativeKv};
use model_dsl::{self as dsl, MatW, NormW, WeightRepr, matmul};
use model_ir::trace::{DType, Dim, FireClass, ForwardPlan, NormVariant, Shape};

struct NhLayerW {
    norm: NormW,

    in_proj: MatW,
    out_proj: MatW,
    gate_norm: NormW,

    q_proj: MatW,
    k_proj: MatW,
    v_proj: MatW,
    o_proj: MatW,

    up_proj: MatW,
    down_proj: MatW,
    router: MatW,
    shared_up: MatW,
    shared_down: MatW,
}

impl NhLayerW {
    fn new(l: u32, f: &NemotronHFacts, norm_eps: f32, repr: WeightRepr) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let m = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr,
        };
        let n = |name: &str| NormW {
            name: w(name),
            variant: NormVariant::Plain,
            per_head: None,
            layer: Some(l),
            eps: norm_eps,
        };
        Self {
            norm: n("norm"),
            in_proj: m("mamba_in_proj", f.mamba.in_proj_width()),
            out_proj: m("mamba_out_proj", f.hidden),
            gate_norm: n("mamba_norm"),
            q_proj: m("q_proj", f.attn.q_width()),
            k_proj: m("k_proj", f.attn.kv_width()),
            v_proj: m("v_proj", f.attn.kv_width()),
            o_proj: m("o_proj", f.hidden),

            up_proj: m("up_proj", f.moe.moe_intermediate),
            down_proj: m("down_proj", f.hidden),
            router: m("router", f.moe.num_experts),
            shared_up: m("shared_expert.up", f.moe.shared_intermediate),
            shared_down: m("shared_expert.down", f.hidden),
        }
    }
}

pub fn nemotron_h_cuda<W1: DtypeAxis, W2: DtypeAxis, A: DtypeAxis, K: KvAxis>(
    facts: &NemotronHFacts,
    class: FireClass,
    norm_eps: f32,
    rope_theta: f32,
) -> ForwardPlan {

    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
        assert!(K::NATIVE_BF16);
    }

    let family = format!(
        "nemotron_h-{}-{}-{}.cuda.{}",
        W1::NAME,
        W2::NAME,
        K::NAME,
        class.suffix()
    );
    let mb = facts.mamba;
    let _at = facts.attn;
    dsl::trace_named(&family, |t| {
        let mut y = dsl::embedded_prologue(t, facts.hidden, facts.vocab);

        let rt = dsl::rt(t);
        for l in 0..facts.layers() {

            let window_left = model_ir::facts::window_left_at(facts.window_left, l);
            let w = NhLayerW::new(l, facts, norm_eps, W1::REPR);
            let x = dsl::cuda::rmsnorm(&y, &w.norm);

            match facts.kind(l) {
                NemotronLayerKind::Mamba => {

                    let packed = matmul(&x, &w.in_proj);
                    let (z, conv_in, dt_raw) = dsl::cuda::generated::nemotron_mamba_split_bf16(
                        &packed,
                        (
                            Shape(vec![Dim::Tokens, Dim::Const(mb.intermediate())]),
                            DType::BF16,
                        ),
                        (
                            Shape(vec![Dim::Tokens, Dim::Const(mb.conv_dim())]),
                            DType::BF16,
                        ),
                        (
                            Shape(vec![Dim::Tokens, Dim::Const(mb.num_heads)]),
                            DType::BF16,
                        ),
                        packed.layer(),
                        None,
                    );
                    let rs = dsl::Rs::at(t, l);

                    let conv_bias = format!("layer.{l}.mamba_conv_bias");
                    let conv_out = dsl::cuda::generated::causal_conv1d_update_batched(
                        &conv_in,
                        &format!("layer.{l}.mamba_conv"),
                        Some(conv_bias.as_str()),
                        mb.conv_dim() as i32,
                        mb.conv_kernel as i32,
                        &rs.view(),
                        Some(l),
                        Some(rs.state()),
                    );

                    let head_row = || (Shape(vec![Dim::Const(mb.num_heads)]), DType::F32);
                    let (a_par, d_par, dt_bias) =
                        dsl::cuda::generated::nemotron_prepare_mamba_params(
                            t,
                            &format!("layer.{l}.mamba_a_log"),
                            &format!("layer.{l}.mamba_d"),
                            &format!("layer.{l}.mamba_dt_bias"),
                            head_row(),
                            head_row(),
                            head_row(),
                            mb.num_heads as i32,
                            Some(l),
                            None,
                        );
                    let token_row = || {
                        (
                            Shape(vec![Dim::Tokens, Dim::Const(mb.num_heads)]),
                            DType::F32,
                        )
                    };
                    let (dt, da) = dsl::cuda::generated::nemotron_prepare_mamba_dt_da(
                        &dt_raw,
                        &a_par,
                        &dt_bias,
                        token_row(),
                        token_row(),
                        dt_raw.layer(),
                        None,
                    );
                    let qo_indptr = rt.qo_indptr();
                    let core = dsl::cuda::generated::nemotron_mamba_ssm_batched_bf16(
                        &conv_out,
                        &dt,
                        &dt_raw,
                        &a_par,
                        &d_par,
                        &dt_bias,
                        &da,
                        (
                            Shape(vec![Dim::Tokens, Dim::Const(mb.intermediate())]),
                            DType::BF16,
                        ),

                        mb.num_heads as i32,
                        mb.head_dim as i32,
                        mb.state_size as i32,
                        mb.n_groups as i32,
                        mb.conv_dim() as i32,
                        &rs.view(),
                        &qo_indptr,
                        Some(l),
                        Some(rs.state()),
                    );
                    dsl::seam(core.trace(), &dsl::seam::ATTN_OUT, &[&core], Some(l));

                    let o = dsl::cuda::generated::zamba_rmsnorm_gated(
                        &core,
                        &z,
                        &w.gate_norm.name,
                        mb.n_groups as i32,
                        norm_eps,
                        core.layer(),
                        None,
                    );
                    y += matmul(&o, &w.out_proj);
                }
                NemotronLayerKind::Attention => {
                    let q = matmul(&x, &w.q_proj);
                    let k = matmul(&x, &w.k_proj);
                    let v = matmul(&x, &w.v_proj);

                    let (q, k) = dsl::cuda::generated::rope_bf16(
                        &q,
                        &k,
                        facts.attn.heads as i32,
                        facts.attn.kv_heads as i32,
                        facts.attn.head_dim as i32,
                        rope_theta,
                        false,
                        &rt.positions(),
                        q.layer(),
                        None,
                    );
                    let kv = dsl::Kv::at(t, l);
                    dsl::cuda::write_kv_to_pages(&k, &v, &kv, facts.attn.kv_heads, facts.attn.head_dim);
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
                    dsl::seam(o.trace(), &dsl::seam::ATTN_OUT, &[&o], Some(l));
                    y += matmul(&o, &w.o_proj);
                }
                NemotronLayerKind::Mlp => {

                    let up = matmul(&x, &w.up_proj);
                    let act = dsl::cuda::generated::relu2(&up, up.layer(), None);
                    y += matmul(&act, &w.down_proj);
                    continue;
                }
            }

            if facts.moe.num_experts == 0 {
                continue;
            }
            let m = dsl::cuda::rmsnorm(&y, &w.norm);

            let logits = dsl::cuda::generated::act_x_wt_bf16_out_fp32(
                &m,
                &w.router.name,
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.moe.num_experts)]),
                    DType::F32,
                ),
                m.layer(),
                None,
            );
            let (experts, weights) = dsl::cuda::generated::topk_sigmoid_bias_fp32(
                &logits,
                &format!("layer.{l}.router_bias"),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.moe.top_k)]),
                    DType::I32,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.moe.top_k)]),
                    DType::F32,
                ),
                facts.moe.norm_topk_prob,
                facts.moe.routed_scaling,
                logits.layer(),
                None,
            );
            let gate_up = dsl::cuda::generated::moe_gate_up_decode_gemv(
                &experts,
                &m,
                &format!("layer.{l}.expert.{{e}}.up"),
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(facts.moe.top_k),
                        Dim::Const(facts.moe.moe_intermediate),
                    ]),
                    DType::BF16,
                ),
                Some(l),
                None,
            );
            let act = dsl::cuda::generated::relu2(&gate_up, gate_up.layer(), None);
            let route_out = dsl::cuda::generated::moe_down_decode_gemv(
                &experts,
                &act,
                &format!("layer.{l}.expert.{{e}}.down"),
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(facts.moe.top_k),
                        Dim::Const(facts.hidden),
                    ]),
                    DType::BF16,
                ),
                Some(l),
                None,
            );
            let routed = dsl::cuda::weighted_sum(&weights, &route_out, facts.hidden, None);

            let moe_out = if facts.moe.shared_intermediate > 0 {
                let sup = matmul(&m, &w.shared_up);
                let sact = dsl::cuda::generated::relu2(&sup, sup.layer(), None);
                let shared = matmul(&sact, &w.shared_down);
                dsl::cuda::generated::residual_add(&routed, &shared, routed.layer(), None)
            } else {
                routed
            };
            y = dsl::cuda::generated::residual_add(&y, &moe_out, y.layer(), None);
        }

        dsl::logits_epilogue(
            t,
            &y,
            NormVariant::Plain,
            false,
            facts.vocab,
            None,
            norm_eps,
        );
    })
}

pub type TraceFn = fn(&NemotronHFacts, FireClass, f32, f32) -> ForwardPlan;

pub type ShippedW1 = Bf16Ax;

pub type ShippedW2 = Bf16Ax;

pub type ShippedA = Bf16Ax;

pub type ShippedKv = NativeKv;

pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalogue![(
    "nemotron_h-bf16-bf16-kv-bf16",
    nemotron_h_cuda::<ShippedW1, ShippedW2, ShippedA, ShippedKv>,
),];
