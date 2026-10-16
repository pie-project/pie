//! The fusion rules, in the order they run. A rule that forms a longer chain
//! runs before one that forms a piece of it.

use poem::declare::Precision;
use poem::ops::{attn, elemwise, layout, linear};
use poem::pattern::{Pattern, Template, template};
use poem_ir::ops::fused::PostNorm;
use poem_ir::{
    ActQuant, Dim, Dtype, Elementwise, Fused, ModulateForm, NormKind, Operation, StructKind, Ty,
};

use super::Rule;

/// A placeholder width for pattern holes; the matcher compares op fields, not
/// the types of the values they hold.
const W: u64 = 64;

/// An NVFP4 bank against NVFP4 activations (W4A4).
const W4A4: Precision = Precision {
    dtype: Dtype::Nvfp4,
    act: Some(ActQuant::Nvfp4),
};

/// An MXFP4 bank against e4m3 activations in 1x128 blocks (W4A8).
const W4A8: Precision = Precision {
    dtype: Dtype::Mxfp4,
    act: Some(ActQuant::E4m3 { gran: 128 }),
};

pub(super) fn all() -> Vec<Rule> {
    vec![
        qkv_qknorm_rope_vnorm_write(),
        rmsnorm_residual_add(),
        embed_scale_add_select(),
        embed_scale_add(),
        residual_add_rmsnorm(),
        mlp_swiglu(),
        matmul_geglu(),
        moe_experts(),
        moe_select_glu(),
        matmul_swiglu(),
        lm_head_softcap(),
        matmul_bias(),
        matmul_split_rows(),
        gated_residual_norm_modulate(),
        norm_modulate(),
        rmsnorm_rope_partial_q(),
        norm_rope_last_append(),
        norm_rope_last(),
        qg_qknorm_rope_append(),
        split_rope_yarn_append(),
        rope_yarn_append(),
        pool_attention(),
        hc_prenorm_project(),
        lse_sink(),
    ]
}

/// A hyper-connection stream norm read only by the mix projection over it.
fn hc_prenorm_project() -> Rule {
    let pattern = template(|p| {
        let streams = p.rows("streams", W);
        let normed = elemwise::hc_rmsnorm_f32(&streams, p.f32("eps"));
        p.name("normed", &normed);
        p.free("stream_count");
        let mixes = elemwise::hc_project(&normed, &p.weight("weight", [24, W]), 4);
        p.export("mixes", &mixes);
    });
    Rule::new("elementwise.hc_prenorm_project", [pattern], |m| {
        let elementwise = |name: &str| {
            let Operation::Elementwise(op) = m.op(name) else {
                panic!("`{name}` comes from an elementwise op");
            };
            Box::new(op.clone())
        };
        Fused::HcPrenormProject {
            norm: elementwise("normed"),
            project: elementwise("mixes"),
        }
    })
}

/// DeepSeek-V4's attention: the windowed log-sum-exp attention, the
/// compressed-pool one over the same q (every visible entry, or those an
/// index selected), their merge and the sinks over it.
fn pool_attention() -> Rule {
    let chain = |selected: bool| {
        template(move |p| {
            let q = p.rows("q", W);
            let plan = p.value("plan", Ty::Struct(StructKind::AttnPrefillPlan));
            let cache = p.cache("cache");
            let d = p.u32("d");
            let sm_scale = p.f32("sm_scale");
            p.free("window");
            let (o, lse) =
                attn::prefill_lse(&q, &plan, cache, None, d, p.u32("kv_heads"), sm_scale);
            p.name("o", &o);
            p.name("lse", &lse);
            let positions = p.indices("positions");
            let requests = p.indices("requests");
            let entries = p.cache("entries");
            let (ratio, heads) = (p.u32("ratio"), p.u32("heads"));
            let (po, plse) = if selected {
                let selection = p.rows("selection", W);
                attn::pool_lse_selected(
                    &q,
                    &positions,
                    &requests,
                    &selection,
                    entries,
                    ratio,
                    p.u32("top_k"),
                    heads,
                    d,
                    sm_scale,
                )
            } else {
                attn::pool_lse(
                    &q, &positions, &requests, entries, ratio, heads, d, sm_scale,
                )
            };
            p.name("po", &po);
            p.name("plse", &plse);
            let (mo, mlse) = attn::merge_lse(&o, &lse, &po, &plse, heads, d);
            p.name("mo", &mo);
            p.name("mlse", &mlse);
            let out = attn::sink(&mo, &mlse, &p.weight("sink", [W]), d);
            p.export("o_out", &out);
        })
    };
    Rule::new(
        "attention.pool_attention",
        [chain(true), chain(false)],
        |m| {
            let attention = |name: &str| {
                let Operation::Attention(attn) = m.op(name) else {
                    panic!("`{name}` comes from an attention");
                };
                Box::new(attn.clone())
            };
            Fused::PoolAttention {
                window: attention("o"),
                pool: attention("po"),
                merge: attention("mo"),
                sink: m.value("sink"),
                head_dim: m.u32("d"),
                o_out: m.value("o_out"),
            }
        },
    )
}

/// A log-sum-exp attention and the sinks over its output.
fn lse_sink() -> Rule {
    let pair = |decode: bool| {
        template(move |p| {
            let q = p.rows("q", W);
            let cache = p.cache("cache");
            let d = p.u32("d");
            p.free("window");
            let (o, lse) = if decode {
                let plan = p.value("plan", Ty::Struct(StructKind::AttnDecodePlan));
                attn::decode_lse(&q, &plan, cache, None, d, p.f32("sm_scale"))
            } else {
                let plan = p.value("plan", Ty::Struct(StructKind::AttnPrefillPlan));
                attn::prefill_lse(
                    &q,
                    &plan,
                    cache,
                    None,
                    d,
                    p.u32("kv_heads"),
                    p.f32("sm_scale"),
                )
            };
            p.name("o", &o);
            p.name("lse", &lse);
            let out = attn::sink(&o, &lse, &p.weight("sink", [W]), d);
            p.export("o_out", &out);
        })
    };
    Rule::new("attention.lse_sink", [pair(true), pair(false)], |m| {
        let Operation::Attention(attn) = m.op("o") else {
            panic!("the sinked output comes from an attention");
        };
        Fused::LseSink {
            attn: Box::new(attn.clone()),
            sink: m.value("sink"),
            head_dim: m.u32("d"),
            o_out: m.value("o_out"),
        }
    })
}

/// A packed q/k/v row split, YaRN over its q and k, and the rotated k
/// written to the cache with its v: what `rope_yarn_append` does, reading the
/// three from the packed row rather than from a split's copies.
/// Qwen3.5's attention prologue: q cut from its gate, q and k normed per
/// head (plus-one weights), turned, and k with its v written to the cache,
/// through the `attn.qv` marker the model sets on the cut q and v.
fn qg_qknorm_rope_append() -> Rule {
    let pattern = template(|p| {
        let qg = p.rows("qg", 2 * W);
        let k = p.rows("k", W);
        let v = p.rows("v", W);
        let positions = p.indices("positions");
        let d = p.u32("d");
        let (q, gate) = layout::split_q_gate(&qg, d);
        p.name("q_split", &q);
        p.export("gate", &gate);
        let qn =
            elemwise::rmsnorm_per_head_plus_one(&q, &p.weight("q_norm", [W]), d, p.f32("q_eps"));
        p.name("q_normed", &qn);
        let kn =
            elemwise::rmsnorm_per_head_plus_one(&k, &p.weight("k_norm", [W]), d, p.f32("k_eps"));
        p.name("k_normed", &kn);
        let (q_out, k_out) = elemwise::rope_partial(
            &qn,
            &kn,
            &positions,
            p.u32("rotary"),
            d,
            p.f32("theta"),
            Some(p.u32("span")),
        );
        p.name("k_out", &k_out);
        let cache = p.cache("cache");
        let (write_page, write_offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
        attn::kv_append(&k_out, &v, cache, &write_page, &write_offset);
        p.export("q_out", &q_out);
    });
    Rule::new("elementwise.qg_qknorm_rope_append", [pattern], |m| {
        Fused::QgQknormRopeAppend {
            qg: m.value("qg"),
            k: m.value("k"),
            v: m.value("v"),
            positions: m.value("positions"),
            q_norm: m.value("q_norm"),
            k_norm: m.value("k_norm"),
            head_dim: m.u32("d"),
            q_eps: m.f32("q_eps"),
            k_eps: m.f32("k_eps"),
            rotary_dim: m.u32("rotary"),
            theta: m.f32("theta"),
            span: m.u32("span"),
            cache: m.value("cache"),
            write_page: m.value("write_page"),
            write_offset: m.value("write_offset"),
            q_split: m.value("q_split"),
            gate: m.value("gate"),
            q_normed: m.value("q_normed"),
            k_normed: m.value("k_normed"),
            q_out: m.value("q_out"),
            k_out: m.value("k_out"),
        }
    })
    .through_markers()
    .when(|m| m.dtype("qg") == Some(Dtype::Bf16))
}

fn split_rope_yarn_append() -> Rule {
    let pattern = template(|p| {
        let packed = p.rows("packed", 3 * W);
        let (q, k, v) = layout::split_qkv(&packed, p.u32("q_width"), p.u32("kv_width"));
        p.name("k_split", &k);
        p.name("v_split", &v);
        let positions = p.indices("positions");
        p.free("interleaved");
        let (q, k) = elemwise::rope_yarn(
            &q,
            &k,
            &positions,
            p.u32("d"),
            p.f32("theta"),
            p.f32("factor"),
            p.f32("beta_fast"),
            p.f32("beta_slow"),
            p.f32("attention_factor"),
            p.u32("original_max_position"),
            false,
        );
        let cache = p.cache("cache");
        let (write_page, write_offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
        attn::kv_append(&k, &v, cache, &write_page, &write_offset);
        p.export("q_out", &q);
    });
    Rule::new("elementwise.split_rope_yarn_append", [pattern], |m| {
        Fused::SplitRopeYarnAppend {
            packed: m.value("packed"),
            q_width: m.u32("q_width"),
            kv_width: m.u32("kv_width"),
            positions: m.value("positions"),
            head_dim: m.u32("d"),
            theta: m.f32("theta"),
            factor: m.f32("factor"),
            beta_fast: m.f32("beta_fast"),
            beta_slow: m.f32("beta_slow"),
            attention_factor: m.f32("attention_factor"),
            original_max_position: m.u32("original_max_position"),
            interleaved: matches!(
                m.op("q_out"),
                Operation::Elementwise(Elementwise::RopeYarn {
                    interleaved: true,
                    ..
                })
            ),
            cache: m.value("cache"),
            write_page: m.value("write_page"),
            write_offset: m.value("write_offset"),
            q_out: m.value("q_out"),
            k_out: m.value("k_split"),
            v_out: m.value("v_split"),
        }
    })
    // gpt-oss marks its split q and v with `attn.qv`, which no shell reads.
    .through_markers()
    .when(|m| {
        let d = m.u32("d");
        m.dtype("packed") == Some(Dtype::Bf16)
            && d > 0
            && d % 2 == 0
            && m.u32("q_width") % d == 0
            && m.u32("kv_width") % d == 0
    })
}

/// YaRN over q and k, and the rotated k written to the cache with its v.
fn rope_yarn_append() -> Rule {
    let pattern = template(|p| {
        let (q, k, v) = (p.rows("q", W), p.rows("k", W), p.rows("v", W));
        let positions = p.indices("positions");
        p.free("interleaved");
        let (q, k) = elemwise::rope_yarn(
            &q,
            &k,
            &positions,
            p.u32("d"),
            p.f32("theta"),
            p.f32("factor"),
            p.f32("beta_fast"),
            p.f32("beta_slow"),
            p.f32("attention_factor"),
            p.u32("original_max_position"),
            false,
        );
        p.name("k_out", &k);
        let cache = p.cache("cache");
        let (write_page, write_offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
        attn::kv_append(&k, &v, cache, &write_page, &write_offset);
        p.export("q_out", &q);
    });
    Rule::new("elementwise.rope_yarn_append", [pattern], |m| {
        Fused::RopeYarnAppend {
            q: m.value("q"),
            k: m.value("k"),
            v: m.value("v"),
            positions: m.value("positions"),
            head_dim: m.u32("d"),
            theta: m.f32("theta"),
            factor: m.f32("factor"),
            beta_fast: m.f32("beta_fast"),
            beta_slow: m.f32("beta_slow"),
            attention_factor: m.f32("attention_factor"),
            original_max_position: m.u32("original_max_position"),
            interleaved: matches!(
                m.op("q_out"),
                Operation::Elementwise(Elementwise::RopeYarn {
                    interleaved: true,
                    ..
                })
            ),
            cache: m.value("cache"),
            write_page: m.value("write_page"),
            write_offset: m.value("write_offset"),
            q_out: m.value("q_out"),
            k_out: m.value("k_out"),
        }
    })
    .when(|m| m.dtype("q") == Some(Dtype::Bf16))
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
/// head and v without a scale, rotate q and k (proportionally on a global
/// layer), and write k and v to the cache.
fn qkv_qknorm_rope_vnorm_write() -> Rule {
    let rope = |partial: bool| {
        template(move |p| {
            let packed = p.rows("packed", 3 * W);
            let positions = p.indices("positions");
            let (d, eps, theta) = (p.u32("d"), p.f32("eps"), p.f32("theta"));
            let (q, k, v) = layout::split_qkv(&packed, p.u32("q_width"), p.u32("kv_width"));
            let v = elemwise::rmsnorm_no_scale(&v, d, eps);
            let q = elemwise::rmsnorm_per_head(&q, &p.weight("q_norm", [W]), d, eps);
            let k = elemwise::rmsnorm_per_head(&k, &p.weight("k_norm", [W]), d, eps);
            let (q, k) = if partial {
                let span = Some(p.u32("span"));
                elemwise::rope_partial(&q, &k, &positions, p.u32("rotary"), d, theta, span)
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
            && m.get_u32("span").is_none_or(|span| span == d / 2)
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
    let alternatives = [true, false]
        .into_iter()
        .flat_map(|scaled| Norm::BOTH.map(|post| chain(scaled, Some(post))))
        .chain([chain(true, None), chain(false, None)]);
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
        let ids = p.indices("ids");
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

/// A dense MLP whole: the packed gate-up projection, swiglu, and the down
/// projection. Both intermediates stay inside the match, so a chain that
/// reads one of them elsewhere (Bonsai rotates `h` before `down`) stays as
/// its three primitives.
fn mlp_swiglu() -> Rule {
    let pattern = template(|p| {
        let packed = linear::matmul(&p.rows("act", W), &p.weight("gate_up", [2 * W, W]));
        p.name("packed", &packed);
        let h = linear::mlp_swiglu(&packed, p.u32("intermediate"));
        p.name("h", &h);
        let y = linear::matmul(&h, &p.weight("down", [W, W]));
        p.export("y", &y);
    });
    Rule::new("linear.mlp_swiglu", [pattern], |m| Fused::MlpSwiglu {
        act: m.value("act"),
        gate_up: m.value("gate_up"),
        down: m.value("down"),
        intermediate: m.u32("intermediate"),
        packed: m.value("packed"),
        h: m.value("h"),
        y: m.value("y"),
    })
}

/// A routed gate-up select and its GLU: the select over a dense bank, a
/// biased one (gpt-oss) or a quantized one, then SwiGLU, its clamp or its
/// clamp-alpha form, or GeGLU (Gemma 4), over the packed gate|up rows. The packed rows stay
/// inside the match.
/// A routed MLP: a gate-up select and its GLU (each `moe_select_glu` form),
/// the down select of the GLU on the same routes, and the weighted sum.
fn moe_experts() -> Rule {
    let mut alternatives = Vec::new();
    // Each leg's select: over a dense bank, a biased one, a quantized one,
    // or (both legs) a bank quantizing its activations: NVFP4 (W4A4) or
    // MXFP4 against e4m3 (W4A8).
    let legs = (0..3)
        .flat_map(|s| (0..3).map(move |d| (s, d)))
        .chain([(3, 3), (4, 4)]);
    for (select, down) in legs {
        for glu in 0..4 {
            for gated in [true, false] {
                alternatives.push(template(move |p| {
                    let x = p.rows("x", W);
                    let routes = p.value(
                        "routes",
                        Ty::Tensor {
                            shape: vec![Dim::Tokens, Dim::Const(W)],
                            dtype: Dtype::I32,
                        },
                    );
                    let top_k = p.u32("top_k");
                    let select_of = |x: &poem::Value,
                                     bank: &'static str,
                                     bias: &'static str,
                                     kind: u32| match kind {
                        0 => {
                            linear::moe_matmul_select(x, &p.weight(bank, [W, W, W]), &routes, top_k)
                        }
                        1 => linear::moe_matmul_select_bias(
                            x,
                            &p.weight(bank, [W, W, W]),
                            &p.weight(bias, [W, W]),
                            &routes,
                            top_k,
                        ),
                        2 => linear::moe_matmul_select_quant(
                            x,
                            &p.weight(bank, [W, W, W]),
                            &routes,
                            top_k,
                        ),
                        3 => linear::moe_matmul_select_quant(
                            x,
                            &p.weight_at(bank, [W, W, W], W4A4),
                            &routes,
                            top_k,
                        ),
                        _ => linear::moe_matmul_select_quant(
                            x,
                            &p.weight_at(bank, [W, W, W], W4A8),
                            &routes,
                            top_k,
                        ),
                    };
                    let packed = select_of(&x, "bank", "bias", select);
                    p.name("packed", &packed);
                    let intermediate = p.u32("intermediate");
                    let hidden = match glu {
                        0 => linear::mlp_swiglu(&packed, intermediate),
                        1 => linear::mlp_swiglu_clamp(&packed, intermediate, p.f32("limit")),
                        3 => linear::mlp_geglu_tanh_packed(&packed, intermediate),
                        _ => linear::mlp_swiglu_clamp_alpha(
                            &packed,
                            intermediate,
                            p.f32("limit"),
                            p.f32("alpha"),
                        ),
                    };
                    p.name("hidden", &hidden);
                    let routed = select_of(&hidden, "down", "down_bias", down);
                    p.name("routed", &routed);
                    let weights = p.value(
                        "weights",
                        Ty::Tensor {
                            shape: vec![Dim::Tokens, Dim::Const(W)],
                            dtype: Dtype::F32,
                        },
                    );
                    let summed = linear::moe_weighted_sum(&routed, &weights);
                    if gated {
                        // Qwen's gated shared expert, added to the sum.
                        p.name("summed", &summed);
                        let y = linear::moe_sigmoid_gate_add(
                            &summed,
                            &p.rows("shared", W),
                            &p.rows("gate", W),
                        );
                        p.export("y", &y);
                    } else {
                        p.export("y", &summed);
                    }
                }));
            }
        }
    }
    // DeepSeek's W4A8 MLP: its route weights scale the clamped GLU's rows
    // before the down leg quantizes them, and the routes are summed plainly
    // with the shared expert's row added.
    alternatives.push(template(|p| {
        let x = p.rows("x", W);
        let routes = p.value(
            "routes",
            Ty::Tensor {
                shape: vec![Dim::Tokens, Dim::Const(W)],
                dtype: Dtype::I32,
            },
        );
        let top_k = p.u32("top_k");
        let packed = linear::moe_matmul_select_quant(
            &x,
            &p.weight_at("bank", [W, W, W], W4A8),
            &routes,
            top_k,
        );
        p.name("packed", &packed);
        let weights = p.value(
            "weights",
            Ty::Tensor {
                shape: vec![Dim::Tokens, Dim::Const(W)],
                dtype: Dtype::F32,
            },
        );
        let hidden = linear::mlp_swiglu_clamp_routed(
            &packed,
            p.u32("intermediate"),
            &weights,
            p.f32("limit"),
        );
        p.name("hidden", &hidden);
        let routed = linear::moe_matmul_select_quant(
            &hidden,
            &p.weight_at("down", [W, W, W], W4A8),
            &routes,
            top_k,
        );
        p.name("routed", &routed);
        let y = linear::moe_sum_add(&routed, &p.rows("shared", W));
        p.export("y", &y);
    }));
    Rule::new("linear.moe_experts", alternatives, |m| {
        let linear_op = |name| match m.op(name) {
            Operation::Linear(op) => Box::new(op.clone()),
            other => panic!("`{name}` is computed by {other:?}, not a linear op"),
        };
        // Under a gate the sum is `summed` and the gate-add `y`.
        let (sum, gate_add) = match m.get("summed") {
            Some(_) => (linear_op("summed"), Some(linear_op("y"))),
            None => (linear_op("y"), None),
        };
        Fused::MoeExperts {
            select: linear_op("packed"),
            glu: linear_op("hidden"),
            down: linear_op("routed"),
            sum,
            gate_add,
        }
    })
}

fn moe_select_glu() -> Rule {
    let mut alternatives = Vec::new();
    for select in 0..3 {
        for glu in 0..4 {
            alternatives.push(template(|p| {
                let x = p.rows("x", W);
                let bank = p.weight("bank", [W, W, W]);
                let routes = p.value(
                    "routes",
                    Ty::Tensor {
                        shape: vec![Dim::Tokens, Dim::Const(W)],
                        dtype: Dtype::I32,
                    },
                );
                let top_k = p.u32("top_k");
                let packed = match select {
                    0 => linear::moe_matmul_select(&x, &bank, &routes, top_k),
                    1 => linear::moe_matmul_select_bias(
                        &x,
                        &bank,
                        &p.weight("bias", [W, W]),
                        &routes,
                        top_k,
                    ),
                    _ => linear::moe_matmul_select_quant(&x, &bank, &routes, top_k),
                };
                p.name("packed", &packed);
                let intermediate = p.u32("intermediate");
                let y = match glu {
                    0 => linear::mlp_swiglu(&packed, intermediate),
                    1 => linear::mlp_swiglu_clamp(&packed, intermediate, p.f32("limit")),
                    3 => linear::mlp_geglu_tanh_packed(&packed, intermediate),
                    _ => linear::mlp_swiglu_clamp_alpha(
                        &packed,
                        intermediate,
                        p.f32("limit"),
                        p.f32("alpha"),
                    ),
                };
                p.export("y", &y);
            }));
        }
    }
    Rule::new("linear.moe_select_glu", alternatives, |m| {
        let linear_op = |name| match m.op(name) {
            Operation::Linear(op) => Box::new(op.clone()),
            other => panic!("`{name}` is computed by {other:?}, not a linear op"),
        };
        Fused::MoeSelectGlu {
            select: linear_op("packed"),
            glu: linear_op("y"),
        }
    })
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

/// The packed gate-up projection and its SwiGLU, where the MLP's down
/// projection stays apart (a backend that ships `linear.mlp_swiglu` forms
/// that first).
fn matmul_swiglu() -> Rule {
    let pattern = template(|p| {
        let packed = linear::matmul(&p.rows("act", W), &p.weight("w", [2 * W, W]));
        p.name("packed", &packed);
        let y = linear::mlp_swiglu(&packed, p.u32("intermediate"));
        p.export("y", &y);
    });
    Rule::new("linear.matmul_swiglu", [pattern], |m| Fused::MatmulSwiglu {
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

/// A projection cut into two row halves (Qwen3.5's packed qkv|z), through
/// the `recurrent` marker the cut's caller sets on the packed rows.
fn matmul_split_rows() -> Rule {
    let pattern = template(|p| {
        let y = linear::matmul(&p.rows("act", W), &p.weight("w", [W, W]));
        p.name("y", &y);
        let (left, right) = layout::split_rows(&y, p.u32("width"));
        p.export("left", &left);
        p.export("right", &right);
    });
    Rule::new("linear.matmul_split_rows", [pattern], |m| {
        Fused::MatmulSplitRows {
            act: m.value("act"),
            w: m.value("w"),
            width: m.u32("width"),
            y: m.value("y"),
            left: m.value("left"),
            right: m.value("right"),
        }
    })
    .through_markers()
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

/// The per-row (or, through `lanes`, per-lane) vectors a modulation reads,
/// and the rows they are per.
fn modulation_inputs(p: &Pattern, per_lane: bool) -> (poem::Value, Option<poem::Value>, Dim) {
    let rows = if per_lane { Dim::Lanes } else { Dim::Tokens };
    let m = p.value(
        "m",
        Ty::Tensor {
            shape: vec![rows, Dim::Const(2 * W)],
            dtype: Dtype::Bf16,
        },
    );
    let lanes = per_lane.then(|| p.indices("lanes"));
    (m, lanes, rows)
}

/// A gated residual add, normed without a scale and modulated.
fn gated_residual_norm_modulate() -> Rule {
    let chain = |norm: ScaleFree, per_lane: bool| {
        template(move |p| {
            let (m, lanes, rows) = modulation_inputs(p, per_lane);
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
            let (m, lanes, _) = modulation_inputs(p, per_lane);
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

/// A head's RMS norm, scaled (the whole row one head) or scale-free (per
/// head), its last dims rotated: DeepSeek-V4's q, kv plane, compressed
/// latent and index keys.
fn norm_rope_last() -> Rule {
    let rope = |p: &poem::pattern::Pattern, y: &poem::Value| {
        p.free("interleaved");
        p.free("inverse");
        p.free("yarn");
        let positions = p.indices("positions");
        elemwise::rope_partial_last_yarn(
            y,
            &positions,
            p.u32("rotary_dim"),
            p.u32("head_dim"),
            p.f32("theta"),
            false,
            false,
            None,
        )
    };
    let scale_free = template(move |p| {
        let y = elemwise::rmsnorm_no_scale(&p.rows("x", W), p.u32("norm_dim"), p.f32("eps"));
        p.name("y", &y);
        let q_out = rope(p, &y);
        p.export("q_out", &q_out);
    });
    let scaled = template(move |p| {
        let y = elemwise::rmsnorm(&p.rows("x", W), &p.weight("weight", [W]), p.f32("eps"));
        p.name("y", &y);
        let q_out = rope(p, &y);
        p.export("q_out", &q_out);
    });
    Rule::new("elementwise.norm_rope_last", [scale_free, scaled], |m| {
        let Operation::Elementwise(Elementwise::RopePartialLast {
            interleaved,
            inverse,
            yarn,
            ..
        }) = *m.op("q_out")
        else {
            unreachable!("the pattern's last op is the rotation")
        };
        Fused::NormRopeLast {
            x: m.value("x"),
            weight: m.get("weight"),
            head_dim: m.u32("head_dim"),
            eps: m.f32("eps"),
            positions: m.value("positions"),
            rotary_dim: m.u32("rotary_dim"),
            theta: m.f32("theta"),
            interleaved,
            inverse,
            yarn,
            y: m.value("y"),
            q_out: m.value("q_out"),
        }
    })
    // The norm's heads are the rotation's: a scale-free norm per head of
    // the rotation's width, a scaled one over a row one head wide.
    .when(|m| match m.get_u32("norm_dim") {
        Some(norm_dim) => norm_dim == m.u32("head_dim"),
        None => m.width("x") == Some(u64::from(m.u32("head_dim"))),
    })
}

/// DeepSeek-V4's kv plane: a scaled norm of a row one head wide, its last
/// dims rotated, written to the cache and read by nothing else.
fn norm_rope_last_append() -> Rule {
    let pattern = template(|p| {
        p.free("interleaved");
        p.free("inverse");
        p.free("yarn");
        let positions = p.indices("positions");
        let y = elemwise::rmsnorm(&p.rows("x", W), &p.weight("weight", [W]), p.f32("eps"));
        let k = elemwise::rope_partial_last_yarn(
            &y,
            &positions,
            p.u32("rotary_dim"),
            p.u32("head_dim"),
            p.f32("theta"),
            false,
            false,
            None,
        );
        p.name("y", &k);
        let cache = p.cache("cache");
        let (write_page, write_offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
        attn::kv_append_shared(&k, cache, &write_page, &write_offset);
    });
    Rule::new("elementwise.norm_rope_last_append", [pattern], |m| {
        let Operation::Elementwise(Elementwise::RopePartialLast {
            interleaved,
            inverse,
            yarn,
            ..
        }) = *m.op("y")
        else {
            unreachable!("the rotation lands the written row")
        };
        Fused::NormRopeLastAppend {
            x: m.value("x"),
            weight: m.value("weight"),
            head_dim: m.u32("head_dim"),
            eps: m.f32("eps"),
            positions: m.value("positions"),
            rotary_dim: m.u32("rotary_dim"),
            theta: m.f32("theta"),
            interleaved,
            inverse,
            yarn,
            y: m.value("y"),
            cache: m.value("cache"),
            write_page: m.value("write_page"),
            write_offset: m.value("write_offset"),
        }
    })
    .when(|m| m.width("x") == Some(u64::from(m.u32("head_dim"))))
}

/// A per-head norm of q, rotated.
fn rmsnorm_rope_partial_q() -> Rule {
    let pattern = template(|p| {
        let positions = p.indices("positions");
        let d = p.u32("head_dim");
        let y =
            elemwise::rmsnorm_per_head(&p.rows("x", W), &p.weight("weight", [W]), d, p.f32("eps"));
        p.name("y", &y);
        let (rotary_dim, theta, span) = (p.u32("rotary_dim"), p.f32("theta"), p.u32("span"));
        let q_out = elemwise::rope_partial_q(&y, &positions, rotary_dim, d, theta, Some(span));
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
            span: m.u32("span"),
            y: m.value("y"),
            q_out: m.value("q_out"),
        }
    })
}
