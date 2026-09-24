use model_dsl::{Classify, ForwardHybrid, HybridSpec, Input, Predicate, Request, Value, ops, seam};

use super::model::{Model, Reading};

pub struct Facts {
    pub qo_one: bool,
    pub has_adapter: bool,
    pub block_draft: bool,
    pub captures_scores: bool,
    pub masked: bool,
}

impl Facts {
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    pub fn has_adapter() -> Predicate {
        Predicate::fact(1)
    }

    pub fn block_draft() -> Predicate {
        Predicate::fact(2)
    }

    pub fn captures_scores() -> Predicate {
        Predicate::fact(3)
    }

    pub fn masked() -> Predicate {
        Predicate::fact(4)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
            has_adapter: r.has_adapter(),
            block_draft: r.drafts_a_block(),
            captures_scores: r.captures_scores(),
            masked: r.has_custom_mask(),
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one)
            | (u64::from(self.has_adapter) << 1)
            | (u64::from(self.block_draft) << 2)
            | (u64::from(self.captures_scores) << 3)
            | (u64::from(self.masked) << 4)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv);
        let plane = self.kv_heads as u64 * self.head_dim as u64;
        for w in &self.layers {
            c.kv(kv, w.attn.kv.clone(), [plane, plane]);
        }
        if let Some(d) = &self.dflash {
            d.declare_caches(&mut c, kv);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;

        let mask = inputs.mask();
        let (_, trunk_inputs) = match &m.dflash {
            Some(_) => inputs.split(&Facts::block_draft()),
            None => (inputs.clone(), inputs.clone()),
        };
        let positions = trunk_inputs.positions();
        let classes = [
            Facts::masked(),
            Facts::captures_scores(),
            Facts::qo_one(),
            Predicate::rest(),
        ];
        let [input_m, input_s, input_d, input_p] = trunk_inputs.split(classes.clone());
        let plan_m = [
            ops::attn::plan_prefill(&input_m, m.q_heads, m.kv_heads, m.head_dim, Some(m.window)),
            ops::attn::plan_prefill(&input_m, m.q_heads, m.kv_heads, m.head_dim, None),
        ];
        let plan_s = [
            ops::attn::plan_prefill(&input_s, m.q_heads, m.kv_heads, m.head_dim, Some(m.window)),
            ops::attn::plan_prefill(&input_s, m.q_heads, m.kv_heads, m.head_dim, None),
        ];
        let plan_d = [
            ops::attn::plan_decode(&input_d, m.q_heads, m.kv_heads, m.head_dim, Some(m.window)),
            ops::attn::plan_decode(&input_d, m.q_heads, m.kv_heads, m.head_dim, None),
        ];
        let plan_p = [
            ops::attn::plan_prefill(&input_p, m.q_heads, m.kv_heads, m.head_dim, Some(m.window)),
            ops::attn::plan_prefill(&input_p, m.q_heads, m.kv_heads, m.head_dim, None),
        ];
        let ids = inputs.tokens();
        let y = ops::layout::embed(&ids, &m.embed, m.vocab);
        let (h_block, mut y) = match &m.dflash {
            Some(_) => {
                let (block, rest) = y.split(&Facts::block_draft());
                (Some(block), rest)
            }
            None => (None, y),
        };
        let mut tapped: Option<Value> = None;
        let d = m.head_dim;

        let routes = inputs.adapter_routes();
        for (l, w) in inputs.walk_layers(&m.layers) {
            let at = &w.attn;
            let pages = inputs.kv(&at.kv);
            let write_page = inputs.write_page(&at.kv);
            let write_offset = inputs.write_offset(&at.kv);

            let x = ops::elemwise::rmsnorm(&y, &w.attn_norm, w.attn_norm_eps);
            let q = ops::elemwise::add_bias(&at.q_bias, &ops::linear::matmul(&x, &at.q_proj));
            let k = ops::elemwise::add_bias(&at.k_bias, &ops::linear::matmul(&x, &at.k_proj));
            let v = ops::elemwise::add_bias(&at.v_bias, &ops::linear::matmul(&x, &at.v_proj));
            seam::at(seam::ATTN_QV, &[&q, &v]);

            let (q, k) = ops::elemwise::rope_yarn(
                &q,
                &k,
                &positions,
                d,
                at.theta,
                at.factor,
                at.beta_fast,
                at.beta_slow,
                at.attention_factor,
                at.original_max_position,
                false,
            );
            ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
            seam::at(seam::ATTN_Q, &[&q]);

            let win = match at.reading {
                Reading::Windowed => Some(m.window),
                Reading::Full => None,
            };
            let [mq, sq, dq, p] = q.split(classes.clone());
            let a = Value::merge(vec![
                {
                    let (o, lse) = ops::attn::masked_lse(
                        &mq,
                        &plan_m[at.reading as usize],
                        &mask,
                        pages,
                        win,
                        d,
                        m.kv_heads,
                        true,
                        at.sm_scale,
                    );
                    ops::attn::sink(&o, &lse, &at.sinks, d)
                },
                {
                    let (o, lse) = ops::attn::prefill_lse(
                        &sq,
                        &plan_s[at.reading as usize],
                        pages,
                        win,
                        d,
                        m.kv_heads,
                        at.sm_scale,
                    );
                    if let Reading::Full = at.reading {
                        seam::at(seam::SCORES, &[&lse]);
                    }
                    ops::attn::sink(&o, &lse, &at.sinks, d)
                },
                {
                    let (o, lse) = ops::attn::decode_lse(
                        &dq,
                        &plan_d[at.reading as usize],
                        pages,
                        win,
                        d,
                        at.sm_scale,
                    );
                    ops::attn::sink(&o, &lse, &at.sinks, d)
                },
                {
                    let (o, lse) = ops::attn::prefill_lse(
                        &p,
                        &plan_p[at.reading as usize],
                        pages,
                        win,
                        d,
                        m.kv_heads,
                        at.sm_scale,
                    );
                    ops::attn::sink(&o, &lse, &at.sinks, d)
                },
            ]);
            seam::at(seam::ATTN_OUT, &[&a]);

            let o = ops::linear::matmul(&a, &at.o_proj);
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            let o = ops::elemwise::add_bias(&at.o_bias, &o);
            let o = {
                let (adapted, _) = o.split(&Facts::has_adapter());
                let (px, _) = x.split(&Facts::has_adapter());
                ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, &routes, &adapted)
            };
            y = ops::elemwise::residual_add(&o, &y);

            let e = &w.mlp;
            let x = ops::elemwise::rmsnorm(&y, &w.mlp_norm, w.mlp_norm_eps);
            let (routes, weights) = ops::linear::moe_topk_softmax(
                &ops::elemwise::add_bias(&e.router_bias, &ops::linear::matmul(&x, &e.router)),
                e.experts,
                e.top_k,
            );
            let packed = ops::linear::moe_matmul_select_bias(
                &x,
                &e.gate_up,
                &e.gate_up_bias,
                &routes,
                e.top_k,
            );
            let act = ops::linear::mlp_swiglu_clamp_alpha(
                &packed,
                e.inter,
                e.swiglu_limit,
                e.swiglu_alpha,
            );
            let routed = ops::linear::moe_matmul_select_quant(&act, &e.down, &routes, e.top_k);
            let f = ops::linear::moe_weighted_sum(&routed, &weights);
            let f = if m.tp > 1 {
                ops::collective::all_reduce(&f)
            } else {
                f
            };
            let f = ops::linear::moe_bias_sum(&f, &e.down_bias, &routes, &weights);
            y = ops::elemwise::residual_add(&f, &y);
            if let Some(dr) = &m.dflash {
                dr.tap(l, &y, &mut tapped);
            }
        }

        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        let (x, hb) = match (&m.dflash, h_block) {
            (Some(dr), Some(block)) => {
                let tapped = tapped.as_ref().expect("a block drafter tapped the trunk");
                let hb = dr.arm(&inputs, tapped, &block, &mask, &Facts::block_draft());
                (Value::merge(vec![hb.clone(), x]), Some(hb))
            }
            _ => (x, None),
        };
        let logits = ops::linear::lm_head(&x, &m.head);
        let logits = if m.head.dim(0) < u64::from(m.vocab) {
            ops::collective::all_gather(&logits, m.tp)
        } else {
            logits
        };
        if let Some(dr) = &m.dflash {
            dr.plant_readout(&logits, &inputs, hb.as_ref(), &Facts::block_draft());
        }
        logits
    }
}
