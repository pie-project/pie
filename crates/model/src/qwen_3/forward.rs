use model_dsl::{
    Classify, ForwardHybrid, GeomKind, HybridSpec, Input, Predicate, Request, Value, ops, seam,
};

use super::model::{Attn, Gdn, Head, Mixer, Mlp, Model};

pub struct Facts {
    pub qo_one: bool,
}

impl Facts {
    #[must_use]
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv);
        for w in &self.layers {
            match &w.mixer {
                Mixer::Attn(a) => {
                    c.kv(kv, a.kv.clone(), [2, a.kv_heads as u64 * a.head_dim as u64]);
                }

                Mixer::Gdn(g) => {
                    let conv_ch = u64::from(Gdn::qkv_width(g.k_heads, g.v_heads, g.k_dim, g.v_dim));
                    c.state(g.conv_state.clone(), [g.conv_kernel as u64, conv_ch]);
                    c.state(
                        g.delta_state.clone(),
                        [g.v_heads as u64, g.k_dim as u64, g.v_dim as u64],
                    );
                }
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;

        let plan_d = inputs.plan_decode();
        let plan_p = inputs.plan_prefill();
        let ids = inputs.tokens();
        let mut y = ops::layout::embed(&ids, &m.embed, m.vocab);

        for (_, w) in inputs.walk_layers(&m.layers) {
            let x = ops::elemwise::rmsnorm_plus_one(&y, &w.mixer_norm, w.mixer_norm_eps);
            let o = match &w.mixer {
                Mixer::Attn(a) => attn_mixer(&x, &inputs, &plan_d, &plan_p, a),
                Mixer::Gdn(g) => gdn_mixer(&x, &inputs, g),
            };
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            y = ops::elemwise::residual_add(&o, &y);

            let x = ops::elemwise::rmsnorm_plus_one(&y, &w.mlp_norm, w.mlp_norm_eps);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                } => ops::linear::matmul(
                    &ops::linear::mlp_swiglu(&ops::linear::matmul(&x, gate_up), *inter),
                    down,
                ),
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared_gate_up,
                    shared_down,
                    shared_gate,
                    experts,
                    top_k,
                    inter,
                    shared_inter,
                } => {
                    let (routes, weights) = ops::linear::moe_topk_softmax(
                        &ops::linear::matmul(&x, router),
                        *experts,
                        *top_k,
                    );
                    let hidden = ops::linear::mlp_swiglu(
                        &ops::linear::moe_matmul_select(&x, gate_up, &routes, *top_k),
                        *inter,
                    );
                    let routed = ops::linear::moe_weighted_sum(
                        &ops::linear::moe_matmul_select(&hidden, down, &routes, *top_k),
                        &weights,
                    );
                    let shared = ops::linear::matmul(
                        &ops::linear::mlp_swiglu(
                            &ops::linear::matmul(&x, shared_gate_up),
                            *shared_inter,
                        ),
                        shared_down,
                    );
                    ops::linear::moe_sigmoid_gate_add(
                        &routed,
                        &shared,
                        &ops::linear::matmul(&x, shared_gate),
                    )
                }
            };
            let f = if m.tp > 1 {
                ops::collective::all_reduce(&f)
            } else {
                f
            };
            y = ops::elemwise::residual_add(&f, &y);
        }

        let x = ops::elemwise::rmsnorm_plus_one(&y, &m.final_norm, m.final_norm_eps);
        match &m.head {
            Head::Tied => ops::linear::lm_head(&x, &m.embed),
            Head::Bank(bank) => ops::linear::lm_head(&x, bank),
        }
    }
}

fn attn_mixer(x: &Value, inputs: &Input<Facts>, plan_d: &Value, plan_p: &Value, a: &Attn) -> Value {
    let pages = inputs.kv(&a.kv);
    let positions = inputs.positions();
    let write_page = inputs.geometry(inputs.kv_space(), GeomKind::WritePage);
    let write_offset = inputs.geometry(inputs.kv_space(), GeomKind::WriteOffset);
    let d = a.head_dim;
    let (q, gate) = ops::layout::split_q_gate(&ops::linear::matmul(x, &a.qg_proj), d);
    let k = ops::linear::matmul(x, &a.k_proj);
    let v = ops::linear::matmul(x, &a.v_proj);
    seam::at(seam::ATTN_QV, &[&q, &v]);
    let q = ops::elemwise::rmsnorm_per_head_plus_one(&q, &a.q_norm, d, a.q_norm_eps);
    let k = ops::elemwise::rmsnorm_per_head_plus_one(&k, &a.k_norm, d, a.k_norm_eps);
    let (q, k) = ops::elemwise::rope_partial(&q, &k, &positions, a.rotary_dim, d, a.theta);
    ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
    seam::at(seam::ATTN_Q, &[&q]);
    let (dq, p) = q.split(&Facts::qo_one());
    let o = Value::merge(vec![
        ops::attn::decode(&dq, plan_d, pages, None, d, a.sm_scale),
        ops::attn::prefill(&p, plan_p, pages, None, d, a.kv_heads, a.sm_scale),
    ]);
    seam::at(seam::ATTN_OUT, &[&o]);
    ops::linear::matmul(&ops::elemwise::gate_sigmoid_mul(&o, &gate), &a.o_proj)
}

fn gdn_mixer(x: &Value, inputs: &Input<Facts>, g: &Gdn) -> Value {
    let conv_state = inputs.state(&g.conv_state);
    let delta_state = inputs.state(&g.delta_state);
    let qkvz = ops::linear::matmul(x, &g.in_qkvz);
    let ba = ops::linear::matmul(x, &g.in_ba);
    seam::at(seam::RECURRENT, &[&qkvz]);
    let width = Gdn::qkv_width(g.k_heads, g.v_heads, g.k_dim, g.v_dim);
    let one = Facts::qo_one();
    let (qkvz_d, qkvz_p) = qkvz.split(&one);
    let (ba_d, ba_p) = ba.split(&one);
    let (core_d, z_d) = {
        let (qkv, z) = ops::layout::split_rows(&qkvz_d, width);
        let qkv = ops::attn::ssm_causal_conv1d(&qkv, &g.conv, conv_state, g.conv_kernel);
        let gates = ops::attn::ssm_gdn_prep(&ba_d, &g.dt_bias, &g.a_log);
        let core = ops::attn::ssm_gated_delta(
            &qkv,
            &z,
            &gates,
            delta_state,
            g.k_heads,
            g.v_heads,
            g.k_dim,
            g.v_dim,
        );
        (core, z)
    };
    let (core_p, z_p) = {
        let (qkv, z) = ops::layout::split_rows(&qkvz_p, width);
        let qkv = ops::attn::ssm_causal_conv1d_chunked(&qkv, &g.conv, conv_state, g.conv_kernel);
        let gates = ops::attn::ssm_gdn_prep(&ba_p, &g.dt_bias, &g.a_log);
        let core = ops::attn::ssm_gated_delta_chunked(
            &qkv,
            &z,
            &gates,
            delta_state,
            g.k_heads,
            g.v_heads,
            g.k_dim,
            g.v_dim,
        );
        (core, z)
    };
    let o = Value::merge(vec![core_d, core_p]);
    let z = Value::merge(vec![z_d, z_p]);

    let o = ops::elemwise::rmsnorm_gated(&o, &z, &g.norm, g.v_dim, g.norm_eps);
    ops::linear::matmul(&o, &g.out_proj)
}
