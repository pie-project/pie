//! The fusion rules, in the order they run. A rule that forms a longer chain
//! runs before one that forms a piece of it.

use poem::ops::{attn, elemwise, layout, linear};
use poem::pattern::{Pattern, Template, template};
use poem_ir::ops::fused::PostNorm;
use poem_ir::{Dim, Dtype, Elementwise, Fused, ModulateForm, NormKind, Operation, Ty};

use super::Rule;

const W: u64 = 64;

pub(super) fn all() -> Vec<Rule> {
    vec![
        qkv_qknorm_rope_vnorm_write(),
        rmsnorm_residual_add(),
        embed_scale_add_select(),
        embed_scale_add(),
        residual_add_rmsnorm(),
        matmul_geglu(),
        lm_head_softcap(),
        matmul_bias(),
        gated_residual_norm_modulate(),
        norm_modulate(),
        rmsnorm_rope_partial_q(),
    ]
}

#[derive(Clone, Copy)]
enum Norm {
    Plain,
    PlusOne,
}

impl Norm {
    const BOTH: [Norm; 2] = [Norm::Plain, Norm::PlusOne];

    fn apply(
        self,
        p: &Pattern,
        x: &poem::Value,
        weight: &'static str,
        eps: &'static str,
    ) -> poem::Value {
        let w = p.weight(weight, [W]);
        let eps = p.f32(eps);
        match self {
            Norm::Plain => elemwise::rmsnorm(x, &w, eps),
            Norm::PlusOne => elemwise::rmsnorm_plus_one(x, &w, eps),
        }
    }
}

fn plus_one(op: &Operation) -> bool {
    match op {
        Operation::Elementwise(Elementwise::Rmsnorm { .. }) => false,
        Operation::Elementwise(Elementwise::RmsnormPlusOne { .. }) => true,
        other => panic!("{other:?} is not a row norm"),
    }
}

/// Gemma 4's attention prologue: split the packed qkv row, norm q and k per
/// head and v without a scale, rotate q and k, and write k and v to the cache.
fn qkv_qknorm_rope_vnorm_write() -> Rule {
    let rope = |partial: bool| {
        template(move |p| {
            let packed = p.rows("packed", 3 * W);
            let positions = p.value(
                "positions",
                Ty::Tensor {
                    shape: vec![Dim::Tokens],
                    dtype: Dtype::I32,
                },
            );
            let (d, eps, theta) = (p.u32("d"), p.f32("eps"), p.f32("theta"));
            let (q, k, v) = layout::split_qkv(&packed, p.u32("q_width"), p.u32("kv_width"));
            let v = elemwise::rmsnorm_no_scale(&v, d, eps);
            let q = elemwise::rmsnorm_per_head(&q, &p.weight("q_norm", [W]), d, eps);
            let k = elemwise::rmsnorm_per_head(&k, &p.weight("k_norm", [W]), d, eps);
            let (q, k) = if partial {
                elemwise::rope_partial(&q, &k, &positions, p.u32("rotary"), d, theta)
            } else {
                elemwise::rope_full(&q, &k, &positions, d, theta, false)
            };
            let cache = p.cache("cache");
            let (write_page, write_offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
            attn::kv_append(&k, &v, cache, &write_page, &write_offset);
            p.export("q", &q);
        })
    };
    Rule::new(
        "custom_cuda.qkv_fused_qknorm_rope_vnorm_write",
        [rope(true), rope(false)],
        |m| {
            let d = m.u32("d");
            Fused::QkvFusedQknormRopeVnormWrite {
                packed: m.value("packed"),
                positions: m.value("positions"),
                q_norm_weight: m.value("q_norm"),
                k_norm_weight: m.value("k_norm"),
                eps: m.f32("eps"),
                cache: m.value("cache"),
                write_page: m.value("write_page"),
                write_offset: m.value("write_offset"),
                kv_heads: m.u32("kv_width") / d,
                head_dim: d,
                theta: m.f32("theta"),
                rotary_dim: m.get_u32("rotary").unwrap_or(d),
                q: m.value("q"),
            }
        },
    )
    .when(|m| {
        let d = m.u32("d");
        let rotary = m.get_u32("rotary").unwrap_or(d);
        m.dtype("packed") == Some(Dtype::Bf16)
            && rotary > 0
            && rotary <= d
            && rotary % 2 == 0
            && m.u32("kv_width") % d == 0
    })
}

/// A row norm whose output joins the residual stream, optionally scaled and
/// normed again on the way out.
fn rmsnorm_residual_add() -> Rule {
    let chain = |scaled: bool, post: Option<Norm>| {
        template(move |p| {
            let x = p.rows("x", W);
            let y = p.rows("y", W);
            let t = elemwise::rmsnorm(&x, &p.weight("weight", [W]), p.f32("eps"));
            p.export("t", &t);
            let y_out = elemwise::residual_add(&t, &y);
            p.export("y_out", &y_out);
            let mut row = y_out;
            if scaled {
                row = elemwise::scale(&p.weight("s", [W]), &row);
                p.export("scaled", &row);
            }
            if let Some(post) = post {
                let out = post.apply(p, &row, "post_weight", "post_eps");
                p.export("out", &out);
            }
        })
    };
    let mut alternatives = Vec::new();
    for scaled in [true, false] {
        for post in [Some(Norm::Plain), Some(Norm::PlusOne)] {
            alternatives.push(chain(scaled, post));
        }
    }
    alternatives.push(chain(true, None));
    alternatives.push(chain(false, None));
    Rule::new("elementwise.rmsnorm_residual_add", alternatives, |m| {
        Fused::RmsnormResidualAdd {
            x: m.value("x"),
            weight: m.value("weight"),
            eps: m.f32("eps"),
            t: m.value("t"),
            y: m.value("y"),
            y_out: m.value("y_out"),
            scale: m.get("scaled").map(|scaled| (m.value("s"), scaled)),
            post: m.get("out").map(|out| PostNorm {
                weight: m.value("post_weight"),
                plus_one: plus_one(m.op("out")),
                eps: m.f32("post_eps"),
                out,
            }),
        }
    })
    .when(|m| m.width("t").is_some_and(|width| width <= 256 * 32))
}

/// A scaled embedding added onto the residual stream, the sum scaled again;
/// `select` names the stacked row the stream is read from.
fn embed_chain(select: bool) -> Template {
    template(|p| {
        let ids = p.value(
            "ids",
            Ty::Tensor {
                shape: vec![Dim::Tokens],
                dtype: Dtype::I32,
            },
        );
        let e = layout::embed(&ids, &p.weight("table", [W, W]), p.u32("vocab"));
        p.export("e", &e);
        let e_scaled = elemwise::mul_scalar(p.f32("embed_scale"), &e);
        p.export("e_scaled", &e_scaled);
        let y = if select {
            let y = layout::select(&p.rows("stacked", 4 * W), p.u32("layer"), p.u32("width"));
            p.name("y", &y);
            y
        } else {
            p.rows("y", W)
        };
        let y_out = elemwise::residual_add(&e_scaled, &y);
        p.export("y_out", &y_out);
        let y_scaled = elemwise::mul_scalar(p.f32("out_scale"), &y_out);
        p.export("y_scaled", &y_scaled);
    })
}

fn embed_scale_add_select() -> Rule {
    Rule::new(
        "elementwise.embed_scale_add_select",
        [embed_chain(true)],
        |m| Fused::EmbedScaleAddSelect {
            ids: m.value("ids"),
            table: m.value("table"),
            vocab: m.u32("vocab"),
            e: m.value("e"),
            embed_scale: m.f32("embed_scale"),
            e_scaled: m.value("e_scaled"),
            stacked: m.value("stacked"),
            layer: m.u32("layer"),
            width: m.u32("width"),
            y_out: m.value("y_out"),
            out_scale: m.f32("out_scale"),
            y_scaled: m.value("y_scaled"),
        },
    )
}

fn embed_scale_add() -> Rule {
    Rule::new("elementwise.embed_scale_add", [embed_chain(false)], |m| {
        Fused::EmbedScaleAdd {
            ids: m.value("ids"),
            table: m.value("table"),
            vocab: m.u32("vocab"),
            e: m.value("e"),
            embed_scale: m.f32("embed_scale"),
            e_scaled: m.value("e_scaled"),
            y: m.value("y"),
            y_out: m.value("y_out"),
            out_scale: m.f32("out_scale"),
            y_scaled: m.value("y_scaled"),
        }
    })
}

/// A residual add whose sum is normed.
fn residual_add_rmsnorm() -> Rule {
    let pair = |norm: Norm| {
        template(move |p| {
            let y_out = elemwise::residual_add(&p.rows("x", W), &p.rows("y", W));
            p.export("y_out", &y_out);
            let out = norm.apply(p, &y_out, "weight", "eps");
            p.export("out", &out);
        })
    };
    Rule::new(
        "elementwise.residual_add_rmsnorm",
        Norm::BOTH.map(pair),
        |m| Fused::ResidualAddRmsnorm {
            x: m.value("x"),
            y: m.value("y"),
            y_out: m.value("y_out"),
            weight: m.value("weight"),
            plus_one: plus_one(m.op("out")),
            eps: m.f32("eps"),
            out: m.value("out"),
        },
    )
}

fn matmul_geglu() -> Rule {
    let pattern = template(|p| {
        let packed = linear::matmul(&p.rows("act", W), &p.weight("w", [2 * W, W]));
        p.name("packed", &packed);
        let y = linear::mlp_geglu_tanh_packed(&packed, p.u32("intermediate"));
        p.export("y", &y);
    });
    Rule::new("linear.matmul_geglu", [pattern], |m| Fused::MatmulGeglu {
        act: m.value("act"),
        w: m.value("w"),
        intermediate: m.u32("intermediate"),
        packed: m.value("packed"),
        y: m.value("y"),
    })
}

fn lm_head_softcap() -> Rule {
    let pattern = template(|p| {
        let y = linear::lm_head(&p.rows("act", W), &p.weight("w", [W, W]));
        p.name("y", &y);
        let y_out = attn::logit_softcap(&y, p.f32("cap"));
        p.export("y_out", &y_out);
    });
    Rule::new("linear.lm_head_softcap", [pattern], |m| {
        Fused::LmHeadSoftcap {
            act: m.value("act"),
            w: m.value("w"),
            cap: m.f32("cap"),
            y: m.value("y"),
            y_out: m.value("y_out"),
        }
    })
}

fn matmul_bias() -> Rule {
    let pattern = template(|p| {
        let y = linear::matmul(&p.rows("act", W), &p.weight("w", [W, W]));
        p.name("y", &y);
        let y_out = elemwise::add_bias(&p.weight("bias", [W]), &y);
        p.export("y_out", &y_out);
    });
    Rule::new("linear.matmul_bias", [pattern], |m| Fused::MatmulBias {
        act: m.value("act"),
        w: m.value("w"),
        bias: m.value("bias"),
        y: m.value("y"),
        y_out: m.value("y_out"),
    })
}

#[derive(Clone, Copy)]
enum ScaleFree {
    Layernorm,
    Rmsnorm,
}

impl ScaleFree {
    const BOTH: [ScaleFree; 2] = [ScaleFree::Layernorm, ScaleFree::Rmsnorm];

    fn apply(self, p: &Pattern, x: &poem::Value) -> poem::Value {
        match self {
            ScaleFree::Layernorm => elemwise::layernorm_no_scale(x, p.f32("eps")),
            ScaleFree::Rmsnorm => elemwise::rmsnorm_no_scale(x, p.u32("head_dim"), p.f32("eps")),
        }
    }
}

fn norm_kind(op: &Operation) -> NormKind {
    match op {
        Operation::Elementwise(Elementwise::LayernormNoScale { eps, .. }) => {
            NormKind::Layernorm { eps: *eps }
        }
        Operation::Elementwise(Elementwise::RmsnormNoScale { head_dim, eps, .. }) => {
            NormKind::Rmsnorm {
                head_dim: *head_dim,
                eps: *eps,
            }
        }
        other => panic!("{other:?} is not a scale-free norm"),
    }
}

fn modulate_form(op: &Operation) -> ModulateForm {
    match op {
        Operation::Elementwise(Elementwise::Modulate { form, .. }) => *form,
        other => panic!("{other:?} is not a modulation"),
    }
}

/// The per-row (or, through `lanes`, per-lane) vectors a modulation reads.
fn modulation_inputs(p: &Pattern, per_lane: bool) -> (poem::Value, Option<poem::Value>) {
    let rows = if per_lane { Dim::Lanes } else { Dim::Tokens };
    let m = p.value(
        "m",
        Ty::Tensor {
            shape: vec![rows, Dim::Const(2 * W)],
            dtype: Dtype::Bf16,
        },
    );
    let lanes = per_lane.then(|| {
        p.value(
            "lanes",
            Ty::Tensor {
                shape: vec![Dim::Tokens],
                dtype: Dtype::I32,
            },
        )
    });
    (m, lanes)
}

/// A gated residual add, normed without a scale and modulated.
fn gated_residual_norm_modulate() -> Rule {
    let chain = |norm: ScaleFree, per_lane: bool| {
        template(move |p| {
            let (m, lanes) = modulation_inputs(p, per_lane);
            let rows = if per_lane { Dim::Lanes } else { Dim::Tokens };
            let g = p.value(
                "g",
                Ty::Tensor {
                    shape: vec![rows, Dim::Const(W)],
                    dtype: Dtype::Bf16,
                },
            );
            let r_out =
                elemwise::gated_residual_add(&p.rows("r", W), &g, &p.rows("y", W), lanes.as_ref());
            p.export("r_out", &r_out);
            let normed = norm.apply(p, &r_out);
            p.export("normed", &normed);
            p.free("form");
            let out = elemwise::modulate(&normed, &m, lanes.as_ref(), ModulateForm::ScaleShift);
            p.export("out", &out);
        })
    };
    let alternatives = ScaleFree::BOTH
        .into_iter()
        .flat_map(|norm| [chain(norm, true), chain(norm, false)]);
    Rule::new(
        "elementwise.gated_residual_norm_modulate",
        alternatives,
        |m| Fused::GatedResidualNormModulate {
            r: m.value("r"),
            g: m.value("g"),
            y: m.value("y"),
            lane_of_row: m.get("lanes"),
            r_out: m.value("r_out"),
            norm: norm_kind(m.op("normed")),
            normed: m.value("normed"),
            m: m.value("m"),
            form: modulate_form(m.op("out")),
            out: m.value("out"),
        },
    )
}

/// A norm without a scale, modulated.
fn norm_modulate() -> Rule {
    let pair = |norm: ScaleFree, per_lane: bool| {
        template(move |p| {
            let (m, lanes) = modulation_inputs(p, per_lane);
            let normed = norm.apply(p, &p.rows("x", W));
            p.export("normed", &normed);
            p.free("form");
            let y = elemwise::modulate(&normed, &m, lanes.as_ref(), ModulateForm::ScaleShift);
            p.export("y", &y);
        })
    };
    let alternatives = ScaleFree::BOTH
        .into_iter()
        .flat_map(|norm| [pair(norm, true), pair(norm, false)]);
    Rule::new("elementwise.norm_modulate", alternatives, |m| {
        Fused::NormModulate {
            x: m.value("x"),
            norm: norm_kind(m.op("normed")),
            normed: m.value("normed"),
            m: m.value("m"),
            lane_of_row: m.get("lanes"),
            form: modulate_form(m.op("y")),
            y: m.value("y"),
        }
    })
}

/// A per-head norm of q, rotated.
fn rmsnorm_rope_partial_q() -> Rule {
    let pattern = template(|p| {
        let positions = p.value(
            "positions",
            Ty::Tensor {
                shape: vec![Dim::Tokens],
                dtype: Dtype::I32,
            },
        );
        let d = p.u32("head_dim");
        let y =
            elemwise::rmsnorm_per_head(&p.rows("x", W), &p.weight("weight", [W]), d, p.f32("eps"));
        p.name("y", &y);
        let q_out =
            elemwise::rope_partial_q(&y, &positions, p.u32("rotary_dim"), d, p.f32("theta"));
        p.export("q_out", &q_out);
    });
    Rule::new("elementwise.rmsnorm_rope_partial_q", [pattern], |m| {
        Fused::RmsnormRopePartialQ {
            x: m.value("x"),
            weight: m.value("weight"),
            head_dim: m.u32("head_dim"),
            eps: m.f32("eps"),
            positions: m.value("positions"),
            rotary_dim: m.u32("rotary_dim"),
            theta: m.f32("theta"),
            y: m.value("y"),
            q_out: m.value("q_out"),
        }
    })
}
