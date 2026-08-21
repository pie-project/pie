use super::facts::{Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind};
use model_dsl::metal::{GdnShape, GdnW};
use model_dsl::{self as dsl, Kv, MatW, Val, WeightRepr};
use model_ir::trace::{
    DType, Dim, FireClass, ForwardPlan, GuardPred, NormVariant, Shape, StateRef, StateStore,
};

fn default_moe_tile() -> Option<(u32, u32)> {
    Some(crate::shared::llama_like::project::ROUTED_QMM_TILE)
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Qwen35MetalFacts {

    pub proj_repr: WeightRepr,

    pub affine_bits: u32,

    pub moe_repr: Option<WeightRepr>,

    pub moe_bits: u32,

    #[serde(default = "default_moe_tile")]
    pub moe_tile: Option<(u32, u32)>,

    pub router_repr: Option<WeightRepr>,

    pub router_bits: u32,

    pub qmm_tile: (u32, u32),

    pub qmm_fp16_precast: bool,

    pub routed_qmm_fp16: bool,

    pub qmm_multi_batch: bool,

    pub fuse_residual_gemv: bool,

    pub rms_eps: f32,

    pub rope_theta: f32,

    pub attn_scale: f32,

    pub norm_topk_prob: bool,
}

fn hidden_of(facts: &Qwen35HybridFacts) -> u32 {
    let hidden = facts.hidden();
    assert_eq!(
        facts.gdn.hidden, hidden,
        "hybrid sub-facts disagree on hidden (gdn)"
    );
    if let Qwen35MlpKind::Moe(moe) = &facts.mlp {
        assert_eq!(
            moe.hidden, hidden,
            "hybrid sub-facts disagree on hidden (moe)"
        );
    }
    hidden
}

struct Ctx<'a> {
    metal: &'a Qwen35MetalFacts,

    point: String,

    gemm_point: String,

    multi_batch: bool,

    staged: std::cell::RefCell<std::collections::HashMap<model_ir::trace::ValueId, Val>>,

    norm_variant: NormVariant,
}

impl Ctx<'_> {

    fn stage(&self, x: &Val) -> Val {
        if let Some(v) = self.staged.borrow().get(&x.key()) {
            return v.clone();
        }
        let v = dsl::metal::cast_qmm_input(x);
        self.staged.borrow_mut().insert(x.key(), v.clone());
        v
    }

    fn gemm_at(&self, x: &Val, w: &MatW, pt: &str, gpt: &str) -> Val {

        if !(self.multi_batch
            && self.metal.qmm_multi_batch
            && dsl::metal::gemm_fits(w.width, self.metal.qmm_tile))
        {
            return dsl::metal::qmv(x, w, pt);
        }
        let half = self.metal.qmm_fp16_precast.then(|| self.stage(x));
        let shape = (Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16);
        let (g, v) = dsl::guarded_value(x.trace(), w.layer, shape);
        g.arm(
            GuardPred::TokensMultipleOf(self.metal.qmm_tile.0.max(1)),
            || match &half {
                Some(h) => {
                    dsl::metal::qmm_fp16(h, w, gpt);
                }
                None => {
                    dsl::metal::qmm(x, w, gpt);
                }
            },
        )
        .otherwise(|| {
            dsl::metal::qmv(x, w, pt);
        });
        v
    }

    fn gemm(&self, x: &Val, w: &MatW) -> Val {
        self.gemm_at(x, w, &self.point, &self.gemm_point)
    }

    fn gemm_add(&self, x: &Val, w: &MatW, residual: &Val) -> Val {
        if !self.metal.fuse_residual_gemv {
            let y = self.gemm(x, w);
            return dsl::metal::generated::residual_add(&y, residual, y.layer(), None);
        }
        if !(self.multi_batch
            && self.metal.qmm_multi_batch
            && dsl::metal::gemm_fits(w.width, self.metal.qmm_tile))
        {
            return dsl::metal::qmv_residual(x, w, residual, &self.point);
        }
        let half = self.metal.qmm_fp16_precast.then(|| self.stage(x));
        let shape = (Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16);
        let (g, v) = dsl::guarded_value(x.trace(), w.layer, shape);
        g.arm(
            GuardPred::TokensMultipleOf(self.metal.qmm_tile.0.max(1)),
            || match &half {
                Some(h) => {
                    dsl::metal::qmm_residual_fp16(h, w, residual, &self.gemm_point);
                }
                None => {
                    dsl::metal::qmm_residual(x, w, residual, &self.gemm_point);
                }
            },
        )
        .otherwise(|| {
            dsl::metal::qmv_residual(x, w, residual, &self.point);
        });
        v
    }

    fn mat(&self, l: Option<u32>, name: &str, width: u32) -> MatW {
        MatW {
            name: match l {
                Some(l) => format!("layer.{l}.{name}"),
                None => name.to_string(),
            },
            width,
            layer: l,
            repr: self.metal.proj_repr,
        }
    }

    fn norm(&self, x: &Val, l: Option<u32>, name: &str, row: u32) -> Val {
        let name = match l {
            Some(l) => format!("layer.{l}.{name}"),
            None => name.to_string(),
        };
        dsl::metal::generated::rms_single_row(
            x,
            &name,
            self.metal.rms_eps,
            row as i32,
            1,
            u32::from(self.norm_variant == NormVariant::Gemma),
            1.0,
            l,
            None,
        )
    }
}

fn full_attn(c: &Ctx<'_>, l: u32, f: &Qwen35FullAttnFacts, y: &Val) -> Val {
    let q_width = f.q_heads * f.head_dim;
    let kv_width = f.kv_heads * f.head_dim;
    let x = c.norm(y, Some(l), "attn_norm", f.hidden);

    let qg = c.gemm(&x, &c.mat(Some(l), "q_proj", 2 * q_width));
    let k = c.gemm(&x, &c.mat(Some(l), "k_proj", kv_width));
    let v = c.gemm(&x, &c.mat(Some(l), "v_proj", kv_width));

    let qg_out = (Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16);
    let (q, gate) = dsl::metal::generated::q_gate_split(
        &qg,
        qg_out.clone(),
        qg_out,
        f.head_dim as i32,
        (2 * q_width) as i32,
        q_width as i32,
        f.q_heads as i32,
        qg.layer(),
        None,
    );

    let q = c.norm(&q, Some(l), "q_norm", f.head_dim);
    let k = c.norm(&k, Some(l), "k_norm", f.head_dim);

    let (q, k) = dsl::metal::rope(
        &q,
        &k,
        c.multi_batch,
        c.metal.rope_theta,
        1.0,
        f.head_dim,
        f.rotary_dim,
        false,

        false,
    );
    let kv = Kv::at(x.trace(), l);
    dsl::metal::kv_append(&k, &v, &kv,  true, f.head_dim, f.kv_heads);
    let a = dsl::metal::sdpa(
        &q,
        &kv,
        q_width,
        f.head_dim,
         true,
        f.kv_heads,

        -1,
        None,
        c.metal.attn_scale,
        c.multi_batch,
    )
    .expect("a plain attention statement produces its value");

    let gated = dsl::metal::generated::gate(&a, &gate, q_width as i32, a.layer(), None);
    c.gemm_add(&gated, &c.mat(Some(l), "o_proj", f.hidden), y)
}

fn gdn(c: &Ctx<'_>, l: u32, f: &Qwen35GdnFacts, y: &Val, class: FireClass) -> Val {
    let key_width = f.key_width();
    let v_width = f.value_width();
    let x = c.norm(y, Some(l), "attn_norm", f.hidden);

    let qkv = c.gemm(&x, &c.mat(Some(l), "in_proj_qkv", f.conv_dim()));
    let z = c.gemm(&x, &c.mat(Some(l), "in_proj_z", v_width));
    let a = c.gemm(&x, &c.mat(Some(l), "in_proj_a", f.value_heads));
    let b = c.gemm(&x, &c.mat(Some(l), "in_proj_b", f.value_heads));

    let shape = GdnShape {
        k_dim: f.key_head_dim,
        v_dim: f.value_head_dim,
        k_heads: f.key_heads,
        v_heads: f.value_heads,
        conv_dim: f.conv_dim(),
        conv_k: f.conv_kernel,
        q_off: 0,
        k_off: key_width,
        v_off: 2 * key_width,
        eps: c.metal.rms_eps,
    };
    let w = GdnW {
        conv_w: format!("layer.{l}.conv_w"),
        conv_b: format!("layer.{l}.conv_b"),
        a_log: format!("layer.{l}.a_log"),
        dt_bias: format!("layer.{l}.dt"),
    };
    let core = match class {

        FireClass::Decode => dsl::metal::generated::gdn_core_slotted(
            &qkv,
            (
                Shape(vec![Dim::Tokens, Dim::Const(shape.v_heads * shape.v_dim)]),
                DType::BF16,
            ),
            &w.conv_w,
            &w.conv_b,
            &w.a_log,
            &w.dt_bias,
            &a,
            &b,
            shape.k_dim as i32,
            shape.v_dim as i32,
            shape.k_heads as i32,
            shape.v_heads as i32,
            shape.conv_dim as i32,
            shape.conv_k as i32,
            shape.q_off as i32,
            shape.k_off as i32,
            shape.v_off as i32,
            shape.eps,

            (shape.k_dim as f32).powf(-0.5),
            &dsl::runtime::recurrent(qkv.trace(), l),
            Some(l),
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: l,
            }),
        ),

        FireClass::Prefill => {
            let (pre_q, pre_k, pre_gate) = dsl::metal::gdn_prep_prefill(&qkv, &a, &b, shape, &w, l);
            dsl::metal::gdn_core_recurrent_prefill(
                &qkv,
                &pre_q,
                &pre_k,
                &pre_gate,
                shape,
                &w,
                l,
                dsl::metal::GDN_SCAN_TILE,
            )
        }
    };

    let o = dsl::metal::generated::gated_rms(
        &core,
        &z,
        &format!("layer.{l}.gate_norm"),
        (
            Shape(vec![
                Dim::Tokens,
                Dim::Const(f.value_heads * f.value_head_dim),
            ]),
            DType::BF16,
        ),
        c.metal.rms_eps,
        f.value_head_dim as i32,
        f.value_heads as i32,
        core.layer(),
        None,
    );
    c.gemm_add(&o, &c.mat(Some(l), "o_proj", f.hidden), y)
}

fn mlp(
    c: &Ctx<'_>,
    l: u32,
    facts: &Qwen35HybridFacts,
    hidden: u32,
    y: &Val,
    class: FireClass,
) -> Val {
    let x = c.norm(y, Some(l), "mlp_norm", hidden);
    match &facts.mlp {
        Qwen35MlpKind::Dense { intermediate } => {
            let g = c.gemm(&x, &c.mat(Some(l), "gate_proj", *intermediate));
            let u = c.gemm(&x, &c.mat(Some(l), "up_proj", *intermediate));
            let h = dsl::metal::generated::silu_mul(&g, &u, g.layer(), None);
            c.gemm_add(&h, &c.mat(Some(l), "down", hidden), y)
        }
        Qwen35MlpKind::Moe(moe) => {
            let k = moe.top_k.max(1);

            let gate_repr = c.metal.router_repr.unwrap_or(c.metal.proj_repr);
            let gate_bits = if c.metal.router_repr.is_some() {
                c.metal.router_bits
            } else {
                c.metal.affine_bits
            };
            let logits = dsl::metal::qmv(
                &x,
                &MatW {
                    repr: gate_repr,
                    ..c.mat(Some(l), "router", moe.num_experts)
                },
                &dsl::metal::affine_point(gate_repr, gate_bits),
            );
            let (ids, weights) = dsl::metal::router_topk(
                &logits,
                moe.num_experts,
                k,

                None,
                c.metal.norm_topk_prob,
            );

            let tile = if class == FireClass::Prefill {
                c.metal.moe_tile
            } else {
                None
            };
            let block = tile.map_or(dsl::metal::ROUTE_BLOCK_MATVEC, |t| t.0);
            let (perm, row_expert, tile_expert, inv) =
                dsl::metal::route_sort(&ids, moe.num_experts, k, hidden, block);
            let rows = dsl::metal::route_gather(&x, &perm, moe.num_experts, k, hidden, block);
            let bank = |name: &str, width: u32| MatW {
                repr: c.metal.moe_repr.unwrap_or(c.metal.proj_repr),
                ..c.mat(Some(l), name, width)
            };
            let bits = if c.metal.moe_repr.is_some() {
                c.metal.moe_bits
            } else {
                c.metal.affine_bits
            };
            let project = |x: &Val, name: &str, width: u32, in_vec: u32| {
                if let Some(tile) = tile {
                    dsl::metal::routed_qmm(
                        x,
                        &row_expert,
                        &tile_expert,
                        &bank(name, width),
                        moe.num_experts,
                        k,
                        bits,
                        tile,
                        c.metal.routed_qmm_fp16 && bits == 4,
                    )
                } else {
                    dsl::metal::routed_qmv(
                        x,
                        &row_expert,
                        &bank(name, width),
                        k,
                        in_vec,
                        false,
                        bits,
                    )
                }
            };

            let g = project(&rows, "expert_gate", moe.moe_intermediate, hidden);
            let u = project(&rows, "expert_up", moe.moe_intermediate, hidden);

            let h = dsl::metal::generated::silu_mul(&g, &u, g.layer(), None);
            let routed = dsl::metal::combine_sorted(
                &project(&h, "expert_down", hidden, moe.moe_intermediate),
                &weights,
                &inv,
                k,
                hidden,
            );
            let blended = if moe.shared_expert_intermediate == 0 {
                routed
            } else {
                let sg = c.gemm(
                    &x,
                    &c.mat(Some(l), "shared_gate", moe.shared_expert_intermediate),
                );
                let su = c.gemm(
                    &x,
                    &c.mat(Some(l), "shared_up", moe.shared_expert_intermediate),
                );
                let shared = dsl::metal::generated::silu_mul(&sg, &su, sg.layer(), None);

                let shared_down = c.gemm(&shared, &c.mat(Some(l), "shared_down", hidden));
                let shared_gate = dsl::metal::qmv(
                    &x,
                    &MatW {
                        repr: gate_repr,
                        ..c.mat(Some(l), "shared_gate_proj", 1)
                    },
                    &dsl::metal::affine_point(gate_repr, gate_bits),
                );
                dsl::metal::generated::shared_expert_combine(
                    &routed,
                    &shared_down,
                    &shared_gate,
                    routed.layer(),
                    None,
                )
            };
            dsl::metal::generated::residual_add(&blended, y, blended.layer(), None)
        }
    }
}

#[must_use]
pub fn qwen3_5_hybrid_metal(
    facts: &Qwen35HybridFacts,
    metal: &Qwen35MetalFacts,
    class: FireClass,
) -> ForwardPlan {
    let hidden = hidden_of(facts);
    let (n_experts, moe_intermediate, shared_intermediate, intermediate) = match &facts.mlp {
        Qwen35MlpKind::Dense { intermediate } => (0, 0, 0, *intermediate),
        Qwen35MlpKind::Moe(moe) => (
            moe.num_experts,
            moe.moe_intermediate,
            moe.shared_expert_intermediate,
            0,
        ),
    };
    let shape = dsl::ModelShape {
        hidden,
        intermediate,
        n_experts,
        moe_intermediate,
        shared_intermediate,
        vocab: facts.vocab,
        head_dim: facts.attn.head_dim,
        q_width: facts.attn.q_width(),
        kv_width: facts.attn.kv_width(),
        qk_norm: model_ir::facts::QkNorm::PerHead,
        norm_variant: facts.norm_variant,
        norm_eps_micro: (metal.rms_eps * 1.0e6).round() as u32,
        tied_embeddings: facts.tied_embeddings,
        proj_repr: metal.proj_repr,
    };
    dsl::trace_metal("qwen3_5_hybrid", &shape, class, |m| {

        m.depth_window();
        let c = Ctx {
            metal,
            point: dsl::metal::affine_point(metal.proj_repr, metal.affine_bits),
            gemm_point: dsl::metal::affine_gemm_point(
                metal.proj_repr,
                metal.affine_bits,
                metal.qmm_tile,
            ),
            multi_batch: class != FireClass::Decode,
            staged: std::cell::RefCell::new(std::collections::HashMap::new()),
            norm_variant: facts.norm_variant,
        };
        let t = m.trace();
        let mut y =
            dsl::metal::embed_gather(t, "embed", hidden, c.multi_batch, metal.proj_repr, &c.point);
        for l in 0..facts.layers {
            let after = if facts.is_full_attn(l) {
                full_attn(&c, l, &facts.attn, &y)
            } else {
                gdn(&c, l, &facts.gdn, &y, class)
            };
            y = mlp(&c, l, facts, hidden, &after, class);
        }
        let normed = c.norm(&y, None, "final_norm", hidden);

        let sampled = dsl::metal::sample_rows(&normed, hidden);
        let head = if facts.tied_embeddings {
            "embed"
        } else {
            "lm_head"
        };
        let logits = dsl::metal::lm_head(&sampled, head, facts.vocab, metal.proj_repr, &c.point);

        dsl::seam(t, &dsl::seam::OUT, &[&logits], None);
    })
}
