//! The fusion rules' constructors, and [`all`]: every rule in every form, in
//! the order rules run. A rule that forms a longer chain runs before one that
//! forms a piece of it. A backend builds the rules it implements from these.

use poem::declare::Precision;
use poem::ops::{attn, elemwise, gather, linear, ssm};
use poem::pattern::{Pattern, Template, template};
use poem_ir::ops::fused::{Intermediates, PostNorm, Scaled};
use poem_ir::{
    ActQuant, Dim, Dtype, Elemwise, Fused, Gather, Linear, ModulateForm, NormKind, Operation,
    RouteScore, StructKind, Ty,
};
use poem_ir::{NormAxis, NormWeight, Turn};

use super::{Match, Rule};
use poem_ir::GluAct;

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

/// Every rule, in every form, that forms one of the fused ops `labels`
/// names. Panics on a label no rule forms.
#[must_use]
pub fn forming(labels: &[&str]) -> Vec<Rule> {
    let rules: Vec<Rule> = all()
        .into_iter()
        .filter(|rule| labels.contains(&rule.kernel()))
        .collect();
    for label in labels {
        assert!(
            rules.iter().any(|rule| rule.kernel() == *label),
            "`{label}` is a fused op no rule forms"
        );
    }
    rules
}

/// Every rule in every form it takes, in the order rules run.
#[must_use]
pub fn all() -> Vec<Rule> {
    vec![
        qkv_fused_qknorm_rope_vnorm_write(),
        qkv_fused_padded(),
        rmsnorm_residual_add(),
        embed_scale_add_select(),
        embed_scale_add(),
        residual_add_rmsnorm(),
        matmul_swiglu_matmul(),
        matmul_glu(&[GluAct::GegluTanh, GluAct::Swiglu]),
        moe_mega(),
        moe_experts(),
        moe_select_glu(),
        matmul_softcap(),
        matmul_bias(),
        matmul_moe_route(),
        matmul_split_rows(),
        gated_residual_norm_modulate(),
        norm_modulate(),
        norm_rope(Rotated::First),
        hc_norm_project(),
        gdn_step(),
        q_gate_norm_rope_append(Intermediates::Written),
        pool_compress(),
        norm_rope_pair(),
        index_topk_rope(),
        norm_rope_last_append(),
        norm_rope_last_append_rows(),
        norm_rope(Rotated::Last),
        q_gate_norm_rope_append(Intermediates::Unwritten),
        split_rope_yarn_append(),
        rope_yarn_append(),
        pool_attention(),
        hc_fold_pre(),
        hc_pre(),
        mla_prefill_expanded(),
        hc_prenorm_project(),
        hc_gates_norm(),
        res_blend_rmsnorm(),
        lse_sink(),
    ]
}

/// A sublayer's hyper-connection fold and the next sublayer's whole pre-mix
/// over the streams it folds ([`hc_pre`]'s chain): the folded streams read
/// on, the pre-mix's intermediates read only inside it. The first sublayer's
/// expanded streams (no fold), the last fold's (collapsed, not gated), an
/// Engram layer's (written into before they are normed) and Single-Pass
/// mHC's (whose normed row is the previous sublayer's mixes' pre-mix) match
/// none of it.
#[must_use]
pub fn hc_fold_pre() -> Rule {
    let pattern = template(|p| {
        let streams = elemwise::hc_fold(
            &p.rows("x", W),
            &p.rows("residual", 4 * W),
            &p.rows("post_in", 4),
            &p.rows("comb_in", 16),
        );
        p.export("folded", &streams);
        let normed = elemwise::hc_rmsnorm_f32(&streams, p.f32("mix_eps"));
        p.name("normed", &normed);
        p.free("stream_count");
        let mixes = elemwise::hc_project(&normed, &p.weight("plane", [24, 4 * W]), 4);
        p.name("mixes", &mixes);
        let (x, post, comb) = elemwise::hc_gates(
            &mixes,
            &streams,
            &p.weight("scale", [3]),
            &p.weight("base", [24]),
            4,
            p.f32("gate_eps"),
            p.f32("alpha"),
            p.u32("sinkhorn"),
        );
        p.name("pre", &x);
        p.export("post", &post);
        p.export("comb", &comb);
        let y = elemwise::rmsnorm(&x, &p.weight("weight", [W]), p.f32("eps"));
        p.export("y", &y);
    });
    Rule::new("fused.hc_fold_pre", [pattern], |m| {
        let elementwise = |name: &str| {
            let Operation::Elemwise(op) = m.op(name) else {
                panic!("`{name}` comes from an elementwise op");
            };
            Box::new(op.clone())
        };
        Fused::HcFoldPre {
            fold: elementwise("folded"),
            norm: elementwise("normed"),
            project: elementwise("mixes"),
            gates: elementwise("pre"),
            rmsnorm: elementwise("y"),
        }
    })
}

/// A sublayer's whole hyper-connection pre-mix: the stream norm read only by
/// the mix projection, the mixes read only by the combined gates, and their
/// pre-mixed row read only by an RMS norm.
#[must_use]
pub fn hc_pre() -> Rule {
    let pattern = template(|p| {
        let streams = p.rows("streams", 4 * W);
        let normed = elemwise::hc_rmsnorm_f32(&streams, p.f32("mix_eps"));
        p.name("normed", &normed);
        p.free("stream_count");
        let mixes = elemwise::hc_project(&normed, &p.weight("plane", [24, 4 * W]), 4);
        p.name("mixes", &mixes);
        let (x, post, comb) = elemwise::hc_gates(
            &mixes,
            &streams,
            &p.weight("scale", [3]),
            &p.weight("base", [24]),
            4,
            p.f32("gate_eps"),
            p.f32("alpha"),
            p.u32("sinkhorn"),
        );
        p.name("x", &x);
        p.export("post", &post);
        p.export("comb", &comb);
        let y = elemwise::rmsnorm(&x, &p.weight("weight", [W]), p.f32("eps"));
        p.export("y", &y);
    });
    Rule::new("fused.hc_pre", [pattern], |m| {
        let elementwise = |name: &str| {
            let Operation::Elemwise(op) = m.op(name) else {
                panic!("`{name}` comes from an elementwise op");
            };
            Box::new(op.clone())
        };
        Fused::HcPre {
            norm: elementwise("normed"),
            project: elementwise("mixes"),
            gates: elementwise("x"),
            rmsnorm: elementwise("y"),
        }
    })
}

/// The combined hyper-connection gates whose pre-mixed row only an RMS norm
/// reads.
#[must_use]
pub fn hc_gates_norm() -> Rule {
    let pattern = template(|p| {
        let mixes = p.value(
            "mixes",
            Ty::Tensor {
                shape: vec![Dim::Tokens, Dim::Const(24)],
                dtype: Dtype::F32,
            },
        );
        let streams = p.rows("streams", 4 * W);
        p.free("stream_count");
        let (x, post, comb) = elemwise::hc_gates(
            &mixes,
            &streams,
            &p.weight("scale", [3]),
            &p.weight("base", [24]),
            4,
            p.f32("gate_eps"),
            p.f32("alpha"),
            p.u32("sinkhorn"),
        );
        p.name("x", &x);
        p.export("post", &post);
        p.export("comb", &comb);
        let y = elemwise::rmsnorm(&x, &p.weight("weight", [W]), p.f32("eps"));
        p.export("y", &y);
    });
    Rule::new("fused.hc_gates_norm", [pattern], |m| {
        let elementwise = |name: &str| {
            let Operation::Elemwise(op) = m.op(name) else {
                panic!("`{name}` comes from an elementwise op");
            };
            Box::new(op.clone())
        };
        Fused::HcGatesNorm {
            gates: elementwise("x"),
            norm: elementwise("y"),
        }
    })
}

/// AttnRes's blend whose row only an RMS norm reads (Kimi-K3's sublayer
/// inputs), over however many closed blocks.
#[must_use]
pub fn res_blend_rmsnorm() -> Rule {
    let pattern = template(|p| {
        let prefix = p.rows("prefix", W);
        p.free("blocks");
        let h = elemwise::res_blend(
            &prefix,
            &[],
            &p.weight("norm", [W]),
            p.f32("eps"),
            &p.weight("proj", [W]),
        );
        p.name("h", &h);
        let y = elemwise::rmsnorm(&h, &p.weight("weight", [W]), p.f32("norm_eps"));
        p.export("y", &y);
    });
    Rule::new("fused.res_blend_rmsnorm", [pattern], |m| {
        let elementwise = |name: &str| {
            let Operation::Elemwise(op) = m.op(name) else {
                panic!("`{name}` comes from an elementwise op");
            };
            Box::new(op.clone())
        };
        Fused::ResBlendRmsnorm {
            blend: elementwise("h"),
            norm: elementwise("y"),
        }
    })
}

/// A hyper-connection stream norm read only by the mix projection over it.
#[must_use]
pub fn hc_prenorm_project() -> Rule {
    let pattern = template(|p| {
        let streams = p.rows("streams", W);
        let normed = elemwise::hc_rmsnorm_f32(&streams, p.f32("eps"));
        p.name("normed", &normed);
        p.free("stream_count");
        let mixes = elemwise::hc_project(&normed, &p.weight("weight", [24, W]), 4);
        p.export("mixes", &mixes);
    });
    Rule::new("fused.hc_prenorm_project", [pattern], |m| {
        let elementwise = |name: &str| {
            let Operation::Elemwise(op) = m.op(name) else {
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

/// DeepSeek-V4's attention: windowed and compressed-pool (all entries or the
/// indexed ones) log-sum-exp attention over one q, their merge and the sinks;
/// the window floored at each row's replay floor where it states one.
#[must_use]
pub fn pool_attention() -> Rule {
    let chain = |selected: bool, rotated: bool, floored: bool| {
        template(move |p| {
            let q = p.rows("q", W);
            let plan = p.value("plan", Ty::Struct(StructKind::AttnPrefillPlan));
            let cache = p.cache("cache");
            let d = p.u32("d");
            let sm_scale = p.f32("sm_scale");
            p.free("window");
            let floor = floored.then(|| p.indices("floor"));
            let (o, lse) = both(attn::prefill(
                &q,
                &plan,
                cache,
                None,
                d,
                p.u32("kv_heads"),
                sm_scale,
                true,
                None,
                true,
                floor.as_ref(),
                None,
                None,
                false,
                None,
                None,
                None,
            ));
            p.name("o", &o);
            p.name("lse", &lse);
            let positions = p.indices("positions");
            let requests = p.indices("requests");
            let entries = p.cache("entries");
            let (ratio, heads) = (p.u32("ratio"), p.u32("heads"));
            let (po, plse) = if selected {
                let selection = p.rows("selection", W);
                attn::pool_lse(
                    &q,
                    &positions,
                    &requests,
                    entries,
                    ratio,
                    heads,
                    d,
                    sm_scale,
                    Some(&selection),
                    Some(p.u32("top_k")),
                )
            } else {
                attn::pool_lse(
                    &q, &positions, &requests, entries, ratio, heads, d, sm_scale, None, None,
                )
            };
            p.name("po", &po);
            p.name("plse", &plse);
            let (mo, mlse) = attn::merge_lse(&o, &lse, &po, &plse, heads, d);
            p.name("mo", &mo);
            p.name("mlse", &mlse);
            let out = attn::sink(&mo, &mlse, &p.weight("sink", [W]), d);
            if rotated {
                // The sinked rows' tail turned back at the rows' positions
                // (DeepSeek-V4's inverse rotation), the turned rows read on.
                p.free("interleaved");
                p.free("inverse");
                p.free("yarn");
                p.name("o_out", &out);
                let turned = elemwise::rope(
                    &out,
                    &positions,
                    d,
                    p.f32("theta"),
                    Turn::Last {
                        rotary_dim: p.u32("rotary_dim"),
                        interleaved: false,
                        inverse: false,
                        yarn: None,
                    },
                    None,
                    None,
                )
                .0;
                p.export("o_rot", &turned);
            } else {
                p.export("o_out", &out);
            }
        })
    };
    Rule::new(
        "fused.pool_attention",
        [
            chain(true, true, false),
            chain(false, true, false),
            chain(true, false, false),
            chain(false, false, false),
            chain(true, true, true),
            chain(false, true, true),
            chain(true, false, true),
            chain(false, false, true),
        ],
        |m| {
            let attention = |name: &str| {
                let Operation::Attn(attn) = m.op(name) else {
                    panic!("`{name}` comes from an attention");
                };
                Box::new(attn.clone())
            };
            Fused::PoolAttention {
                window: attention("o"),
                pool: attention("po"),
                merge: attention("mo"),
                sink: attention("o_out"),
                rotate: m.get("o_rot").map(|_| {
                    let Operation::Elemwise(rotate) = m.op("o_rot") else {
                        unreachable!("the turned rows come from a rotation")
                    };
                    Box::new(rotate.clone())
                }),
            }
        },
    )
}

/// A latent attention's prefill: query absorb, scoring over every cached
/// latent row and output absorb through one `kv_b`, opening on the cut of
/// `q_b`'s rows where the query reaches it unturned, else on the two halves.
#[must_use]
pub fn mla_prefill_expanded() -> Rule {
    let chain = |split: bool| {
        template(move |p| {
            let (heads, rank) = (p.u32("heads"), p.u32("rank"));
            let (nope, v) = (p.u32("nope"), p.u32("v"));
            let (q_nope, q_pe) = if split {
                let q_b = p.rows("q_b", W);
                let (q_nope, q_pe) = attn::mla_split_q_b(&q_b, heads, nope, p.u32("rope"));
                p.name("q_nope", &q_nope);
                p.name("q_pe", &q_pe);
                (q_nope, q_pe)
            } else {
                (p.rows("nope_rows", W), p.rows("rope_rows", W))
            };
            let kv_b = p.weight("kv_b", [W, W]);
            let q = attn::mla_absorb_q(&q_nope, &kv_b, heads, rank, nope, v);
            p.name("q", &q);
            let plan = p.value("plan", Ty::Struct(StructKind::MlaPlan));
            let pages = p.cache("pages");
            p.free("window");
            let scored = attn::mla_prefill(
                &q,
                &plan,
                &q_pe,
                pages,
                heads,
                rank,
                p.f32("sm_scale"),
                None,
                false,
                None,
            )
            .0;
            p.name("scored", &scored);
            let o = attn::mla_absorb_out(&scored, &kv_b, heads, rank, nope, v);
            p.export("o", &o);
        })
    };
    Rule::new(
        "fused.mla_prefill_expanded",
        [chain(true), chain(false)],
        |m| {
            let attention = |name: &str| {
                let Operation::Attn(attn) = m.op(name) else {
                    panic!("`{name}` comes from an attention");
                };
                Box::new(attn.clone())
            };
            Fused::MlaPrefillExpanded {
                split: m.get("q_b").map(|_| attention("q_nope")),
                absorb_q: attention("q"),
                prefill: attention("scored"),
                absorb_out: attention("o"),
            }
        },
    )
}

/// A log-sum-exp attention and the sinks over its output.
#[must_use]
pub fn lse_sink() -> Rule {
    let pair = |decode: bool| {
        template(move |p| {
            let q = p.rows("q", W);
            let cache = p.cache("cache");
            let d = p.u32("d");
            p.free("window");
            let (o, lse) = if decode {
                let plan = p.value("plan", Ty::Struct(StructKind::AttnDecodePlan));
                both(attn::decode(
                    &q,
                    &plan,
                    cache,
                    None,
                    d,
                    p.f32("sm_scale"),
                    true,
                    None,
                    None,
                    false,
                    None,
                    None,
                    None,
                ))
            } else {
                let plan = p.value("plan", Ty::Struct(StructKind::AttnPrefillPlan));
                both(attn::prefill(
                    &q,
                    &plan,
                    cache,
                    None,
                    d,
                    p.u32("kv_heads"),
                    p.f32("sm_scale"),
                    true,
                    None,
                    true,
                    None,
                    None,
                    None,
                    false,
                    None,
                    None,
                    None,
                ))
            };
            p.name("o", &o);
            p.name("lse", &lse);
            let out = attn::sink(&o, &lse, &p.weight("sink", [W]), d);
            p.export("o_out", &out);
        })
    };
    Rule::new("fused.lse_sink", [pair(true), pair(false)], |m| {
        let Operation::Attn(attn) = m.op("o") else {
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

/// `rope_yarn_append` reading q, k and v straight from a packed q/k/v row
/// rather than from a split's copies.
#[must_use]
pub fn split_rope_yarn_append() -> Rule {
    let pattern = template(|p| {
        let packed = p.rows("packed", 3 * W);
        let (q, k, v) = gather::split_qkv(&packed, p.u32("q_width"), p.u32("kv_width"));
        p.name("k_split", &k);
        p.name("v_split", &v);
        let positions = p.indices("positions");
        p.free("interleaved");
        let (q, k) = both(elemwise::rope(
            &q,
            &positions,
            p.u32("d"),
            p.f32("theta"),
            Turn::Yarn {
                yarn: elemwise::Yarn {
                    factor: p.f32("factor"),
                    beta_fast: p.f32("beta_fast"),
                    beta_slow: p.f32("beta_slow"),
                    original_max_position: p.u32("original_max_position"),
                },
                attention_factor: p.f32("attention_factor"),
                interleaved: false,
            },
            Some(&k),
            None,
        ));
        let cache = p.cache("cache");
        let (write_page, write_offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
        attn::kv_append(&k, Some(&v), cache, &write_page, &write_offset);
        p.export("q_out", &q);
    });
    Rule::new("fused.split_rope_yarn_append", [pattern], |m| {
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
                Operation::Elemwise(Elemwise::Rope {
                    turn: Turn::Yarn {
                        interleaved: true,
                        ..
                    },
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
#[must_use]
pub fn rope_yarn_append() -> Rule {
    let pattern = template(|p| {
        let (q, k, v) = (p.rows("q", W), p.rows("k", W), p.rows("v", W));
        let positions = p.indices("positions");
        p.free("interleaved");
        let (q, k) = both(elemwise::rope(
            &q,
            &positions,
            p.u32("d"),
            p.f32("theta"),
            Turn::Yarn {
                yarn: elemwise::Yarn {
                    factor: p.f32("factor"),
                    beta_fast: p.f32("beta_fast"),
                    beta_slow: p.f32("beta_slow"),
                    original_max_position: p.u32("original_max_position"),
                },
                attention_factor: p.f32("attention_factor"),
                interleaved: false,
            },
            Some(&k),
            None,
        ));
        p.name("k_out", &k);
        let cache = p.cache("cache");
        let (write_page, write_offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
        attn::kv_append(&k, Some(&v), cache, &write_page, &write_offset);
        p.export("q_out", &q);
    });
    Rule::new("fused.rope_yarn_append", [pattern], |m| {
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
                Operation::Elemwise(Elemwise::Rope {
                    turn: Turn::Yarn {
                        interleaved: true,
                        ..
                    },
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
        Operation::Elemwise(Elemwise::Rmsnorm {
            weight: NormWeight::Scale(_),
            axis: NormAxis::Row,
            ..
        }) => false,
        Operation::Elemwise(Elemwise::Rmsnorm {
            weight: NormWeight::PlusOne(_),
            axis: NormAxis::Row,
            ..
        }) => true,
        other => panic!("{other:?} is not a row norm"),
    }
}

/// An attention prologue on the packed qkv row: split it, norm q and k per
/// head, rotate them, and write k and v to the cache. Gemma 4's norms carry
/// weights and v's a scaleless norm, rotated in part on a global layer;
/// Muse Spark's are scaleless with v as it is, its global layers unturned.
#[must_use]
pub fn qkv_fused_qknorm_rope_vnorm_write() -> Rule {
    Rule::new(
        "fused.qkv_fused_qknorm_rope_vnorm_write",
        qkv_patterns(),
        qkv_fused,
    )
    .when(qkv_admits)
}

/// [`qkv_fused_qknorm_rope_vnorm_write`]'s padded forms alone (MiMo's, its
/// v scaled, its heads zero-padded in the cache past their widths): where a
/// backend's anchored regions land the rest of the qkv writes, k and v
/// paged into the cache by the epilogue, a cache whose heads carry padding
/// past a piece's widths is one no window of the product addresses.
#[must_use]
pub fn qkv_fused_padded() -> Rule {
    let patterns: Vec<Template> = qkv_patterns().into_iter().skip(4).collect();
    Rule::new(
        "fused.qkv_fused_qknorm_rope_vnorm_write",
        patterns,
        qkv_fused,
    )
    .when(qkv_admits)
    .form("padded")
}

/// The qkv write's forms: Gemma's (the rotation: 0 none, 1 full, 2
/// partial), MiMo's (no norms, v scaled, a partial rotation).
fn qkv_patterns() -> Vec<Template> {
    let pattern = |gemma: bool, rope: u8| {
        template(move |p| {
            let packed = p.rows("packed", 3 * W);
            let positions = p.indices("positions");
            let d = p.u32("d");
            let (q, k, v) = if rope == 4 {
                let heads = [p.u32("q_heads"), p.u32("kv_heads")];
                gather::split_qkv_padded(&packed, heads, [p.u32("qk"), p.u32("vw")], d)
            } else {
                gather::split_qkv(&packed, p.u32("q_width"), p.u32("kv_width"))
            };
            if rope >= 3 {
                let v = elemwise::mul_scalar(p.f32("v_scale"), &v);
                let (rotary, theta) = (p.u32("rotary"), p.f32("theta"));
                let (q, k) = both(elemwise::rope(
                    &q,
                    &positions,
                    d,
                    theta,
                    Turn::Partial {
                        rotary_dim: rotary,
                        span: p.u32("span"),
                    },
                    Some(&k),
                    None,
                ));
                let cache = p.cache("cache");
                let (write_page, write_offset) =
                    (p.rows("write_page", 1), p.rows("write_offset", 1));
                attn::kv_append(&k, Some(&v), cache, &write_page, &write_offset);
                p.export("q", &q);
                return;
            }
            let eps = p.f32("eps");
            let (q, k, v) = if gemma {
                (
                    elemwise::rmsnorm_per_head(&q, &p.weight("q_norm", [W]), d, eps),
                    elemwise::rmsnorm_per_head(&k, &p.weight("k_norm", [W]), d, eps),
                    elemwise::rmsnorm_no_scale(&v, d, eps),
                )
            } else {
                (
                    elemwise::rmsnorm_no_scale(&q, d, eps),
                    elemwise::rmsnorm_no_scale(&k, d, eps),
                    v,
                )
            };
            let (q, k) = match rope {
                0 => (q, k),
                1 => both(elemwise::rope(
                    &q,
                    &positions,
                    d,
                    p.f32("theta"),
                    Turn::Full { interleaved: false },
                    Some(&k),
                    None,
                )),
                _ => {
                    let (rotary, span, theta) = (p.u32("rotary"), p.u32("span"), p.f32("theta"));
                    both(elemwise::rope(
                        &q,
                        &positions,
                        d,
                        theta,
                        Turn::Partial {
                            rotary_dim: rotary,
                            span,
                        },
                        Some(&k),
                        None,
                    ))
                }
            };
            let cache = p.cache("cache");
            let (write_page, write_offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
            attn::kv_append(&k, Some(&v), cache, &write_page, &write_offset);
            p.export("q", &q);
        })
    };
    vec![
        pattern(true, 2),
        pattern(true, 1),
        pattern(false, 1),
        pattern(false, 0),
        pattern(false, 3),
        pattern(false, 4),
    ]
}

/// The fused qkv write a match states.
fn qkv_fused(m: &Match) -> Fused {
    let d = m.u32("d");
    let rotated = m.get_f32("theta").is_some();
    let normed = m.get_f32("eps").is_some();
    Fused::QkvFusedQknormRopeVnormWrite {
        packed: m.value("packed"),
        positions: m.get("positions").filter(|_| rotated),
        q_norm_weight: m.get("q_norm"),
        k_norm_weight: m.get("k_norm"),
        v_norm: m.get("q_norm").is_some(),
        eps: m.get_f32("eps").unwrap_or(0.0),
        cache: m.value("cache"),
        write_page: m.value("write_page"),
        write_offset: m.value("write_offset"),
        kv_heads: m
            .get_u32("kv_heads")
            .unwrap_or_else(|| m.u32("kv_width") / d),
        head_dim: d,
        theta: m.get_f32("theta").unwrap_or(0.0),
        rotary_dim: if rotated {
            m.get_u32("rotary").unwrap_or(d)
        } else {
            0
        },
        span: if normed { d / 2 } else { m.u32("span") },
        qk_norm: normed,
        v_scale: m.get_f32("v_scale").unwrap_or(1.0),
        widths: [m.get_u32("qk").unwrap_or(d), m.get_u32("vw").unwrap_or(d)],
        q: m.value("q"),
    }
}

fn qkv_admits(m: &Match) -> bool {
    let d = m.u32("d");
    let rotary = m.get_u32("rotary").unwrap_or(d);
    m.dtype("packed") == Some(Dtype::Bf16)
        && rotary <= d
        && rotary.is_multiple_of(2)
        && m.get_u32("span").is_none_or(|span| {
            span == d / 2 || (m.get_f32("eps").is_none() && 2 * span <= d && span > 0)
        })
        && m.get_u32("kv_width").is_none_or(|w| w % d == 0)
}

/// A row norm whose output joins the residual stream, optionally scaled and
/// normed again on the way out.
#[must_use]
pub fn rmsnorm_residual_add() -> Rule {
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
    Rule::new("fused.rmsnorm_residual_add", alternatives, |m| {
        Fused::RmsnormResidualAdd {
            x: m.value("x"),
            weight: m.value("weight"),
            eps: m.f32("eps"),
            t: m.value("t"),
            y: m.value("y"),
            y_out: m.value("y_out"),
            scale: m.get("scaled").map(|out| Scaled {
                by: m.value("s"),
                out,
            }),
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
        let e = gather::embed(&ids, &p.weight("table", [W, W]), p.u32("vocab"));
        p.export("e", &e);
        let e_scaled = elemwise::mul_scalar(p.f32("embed_scale"), &e);
        p.export("e_scaled", &e_scaled);
        let y = if select {
            let y = gather::select(&p.rows("stacked", 4 * W), p.u32("layer"), p.u32("width"));
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

#[must_use]
pub fn embed_scale_add_select() -> Rule {
    Rule::new("fused.embed_scale_add_select", [embed_chain(true)], |m| {
        Fused::EmbedScaleAddSelect {
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
        }
    })
    .when(dense_table)
}

/// The fused kernels read the table's rows as plain elements; a packed bank
/// (gemma-4's per-layer tables at u4) is gathered by its own kernel.
fn dense_table(m: &Match<'_>) -> bool {
    m.weight_dtype("table").and_then(Dtype::elem).is_some()
}

#[must_use]
pub fn embed_scale_add() -> Rule {
    Rule::new("fused.embed_scale_add", [embed_chain(false)], |m| {
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
    .when(dense_table)
}

/// A residual add whose sum is normed.
#[must_use]
pub fn residual_add_rmsnorm() -> Rule {
    let pair = |norm: Norm| {
        template(move |p| {
            let y_out = elemwise::residual_add(&p.rows("x", W), &p.rows("y", W));
            p.export("y_out", &y_out);
            let out = norm.apply(p, &y_out, "weight", "eps");
            p.export("out", &out);
        })
    };
    Rule::new("fused.residual_add_rmsnorm", Norm::BOTH.map(pair), |m| {
        Fused::ResidualAddRmsnorm {
            x: m.value("x"),
            y: m.value("y"),
            y_out: m.value("y_out"),
            weight: m.value("weight"),
            plus_one: plus_one(m.op("out")),
            eps: m.f32("eps"),
            out: m.value("out"),
        }
    })
}

/// A dense MLP whole: the packed gate-up projection, swiglu, and the down
/// projection. Both intermediates stay inside the match, so a chain that
/// reads one of them elsewhere (Bonsai rotates `h` before `down`) stays as
/// its three primitives.
#[must_use]
pub fn matmul_swiglu_matmul() -> Rule {
    let pattern = template(|p| {
        let packed = linear::matmul(&p.rows("act", W), &p.weight("gate_up", [2 * W, W]));
        p.name("packed", &packed);
        let h = linear::glu(
            &packed,
            GluAct::Swiglu,
            Some(p.u32("intermediate")),
            None,
            None,
            None,
        );
        p.name("h", &h);
        let y = linear::matmul(&h, &p.weight("down", [W, W]));
        p.export("y", &y);
    });
    Rule::new("fused.matmul_swiglu_matmul", [pattern], |m| {
        Fused::MatmulSwigluMatmul {
            act: m.value("act"),
            gate_up: m.value("gate_up"),
            down: m.value("down"),
            intermediate: m.u32("intermediate"),
            packed: m.value("packed"),
            h: m.value("h"),
            y: m.value("y"),
        }
    })
}

/// A routed MLP the mega-MoE lands in one launch: a gate-up select over a
/// packed bank (gate rows then up rows per expert), its SwiGLU, the down
/// select on the same routes and the fold, nothing between. Its forms are
/// the ones the kernel takes: DeepSeek-V4's W4A8 banks, the route weights
/// scaling the clamped GLU's rows and the routes summed with the shared row
/// added; and dense banks under the weighted sum (clamped or not), or the
/// weights first under a clamp, the shared row added. A weighted sum under
/// Qwen's gated shared expert takes that gate-add along, as `moe_experts`
/// does, so a fire the kernel declines runs as the whole `MoeExperts` the
/// chain would have formed.
#[must_use]
pub fn moe_mega() -> Rule {
    let mut alternatives = Vec::new();
    // (W4A8 banks, the weights first, a clamp, a gated shared expert)
    let forms = [
        (true, true, true, false),
        (false, false, false, true),
        (false, false, false, false),
        (false, false, true, true),
        (false, false, true, false),
        (false, true, true, false),
    ];
    for (w4a8, first, clamped, gated) in forms {
        alternatives.push(template(move |p| {
            let x = p.rows("x", W);
            let routes = p.value(
                "routes",
                Ty::Tensor {
                    shape: vec![Dim::Tokens, Dim::Const(W)],
                    dtype: Dtype::I32,
                },
            );
            let weights = p.value(
                "weights",
                Ty::Tensor {
                    shape: vec![Dim::Tokens, Dim::Const(W)],
                    dtype: Dtype::F32,
                },
            );
            let top_k = p.u32("top_k");
            let select_of = |x: &poem::Value, bank: &'static str| {
                if w4a8 {
                    linear::select::quantized(
                        x,
                        &p.weight_at(bank, [W, W, W], W4A8),
                        &routes,
                        top_k,
                    )
                } else {
                    linear::select::plain(x, &p.weight(bank, [W, W, W]), &routes, top_k)
                }
            };
            let packed = select_of(&x, "bank");
            p.name("packed", &packed);
            let hidden = linear::glu(
                &packed,
                GluAct::Swiglu,
                Some(p.u32("intermediate")),
                None,
                first.then_some(&weights),
                clamped.then(|| p.f32("limit")),
            );
            p.name("hidden", &hidden);
            let routed = select_of(&hidden, "down");
            p.name("routed", &routed);
            if first {
                p.export("y", &linear::moe_sum_add(&routed, &p.rows("shared", W)));
                return;
            }
            let summed = linear::moe_weighted_sum(&routed, &weights);
            if gated {
                p.name("summed", &summed);
                let y =
                    linear::moe_sigmoid_gate_add(&summed, &p.rows("shared", W), &p.rows("gate", W));
                p.export("y", &y);
            } else {
                p.export("y", &summed);
            }
        }));
    }
    Rule::new("fused.moe_mega", alternatives, |m| {
        let linear_op = |name| match m.op(name) {
            Operation::Linear(op) => Box::new(op.clone()),
            other => panic!("`{name}` is computed by {other:?}, not a linear op"),
        };
        // Under a gate the sum is `summed` and the gate-add `y`.
        let (sum, gate_add) = match m.get("summed") {
            Some(_) => (linear_op("summed"), Some(linear_op("y"))),
            None => (linear_op("y"), None),
        };
        Fused::MoeMega {
            select: linear_op("packed"),
            glu: linear_op("hidden"),
            down: linear_op("routed"),
            sum,
            gate_add,
        }
    })
}

/// A routed MLP: a gate-up select and its GLU (each `moe_select_glu` form,
/// or SiTU; or an up select and its relu2), the down select of the GLU on the
/// same routes (each row normed under a scale where `normed`, Muse Spark's),
/// and the weighted sum.
#[must_use]
pub fn moe_experts() -> Rule {
    let mut alternatives = Vec::new();
    // Each leg's select: over a dense bank, a biased one, a quantized one,
    // or (both legs) a bank quantizing its activations: NVFP4 (W4A4) or
    // MXFP4 against e4m3 (W4A8).
    let legs = (0..3)
        .flat_map(|s| (0..3).map(move |d| (s, d)))
        .chain([(3, 3), (4, 4)]);
    for (select, down) in legs {
        for glu in 0..6 {
            for (gated, normed) in [(true, false), (false, false), (false, true)] {
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
                        0 => linear::select::plain(x, &p.weight(bank, [W, W, W]), &routes, top_k),
                        1 => linear::select::biased(
                            x,
                            &p.weight(bank, [W, W, W]),
                            &p.weight(bias, [W, W]),
                            &routes,
                            top_k,
                        ),
                        2 => {
                            linear::select::quantized(x, &p.weight(bank, [W, W, W]), &routes, top_k)
                        }
                        3 => linear::select::quantized(
                            x,
                            &p.weight_at(bank, [W, W, W], W4A4),
                            &routes,
                            top_k,
                        ),
                        _ => linear::select::quantized(
                            x,
                            &p.weight_at(bank, [W, W, W], W4A8),
                            &routes,
                            top_k,
                        ),
                    };
                    let packed = select_of(&x, "bank", "bias", select);
                    p.name("packed", &packed);
                    let hidden = match glu {
                        0 => linear::glu(
                            &packed,
                            GluAct::Swiglu,
                            Some(p.u32("intermediate")),
                            None,
                            None,
                            None,
                        ),
                        1 => linear::glu(
                            &packed,
                            GluAct::Swiglu,
                            Some(p.u32("intermediate")),
                            None,
                            None,
                            Some(p.f32("limit")),
                        ),
                        3 => linear::glu(
                            &packed,
                            GluAct::GegluTanh,
                            Some(p.u32("intermediate")),
                            None,
                            None,
                            None,
                        ),
                        4 => {
                            // Kimi-K3's SiTU, its up cap whatever it holds.
                            p.free("up_cap");
                            linear::glu(
                                &packed,
                                GluAct::Situ {
                                    beta: p.f32("beta"),
                                    up_cap: None,
                                },
                                Some(p.u32("intermediate")),
                                None,
                                None,
                                None,
                            )
                        }
                        // Nemotron-H's ungated experts: the select's rows
                        // are the up rows alone.
                        5 => linear::relu2(&packed),
                        _ => linear::glu(
                            &packed,
                            GluAct::SwigluAlpha {
                                alpha: p.f32("alpha"),
                            },
                            Some(p.u32("intermediate")),
                            None,
                            None,
                            Some(p.f32("limit")),
                        ),
                    };
                    p.name("hidden", &hidden);
                    let mut routed = select_of(&hidden, "down", "down_bias", down);
                    p.name("routed", &routed);
                    if normed {
                        let scale = p.weight("route_scale", [W]);
                        let eps = p.f32("route_eps");
                        (routed, _) =
                            elemwise::norm_gate(&routed, Some(&scale), None, eps, 1.0, None);
                        p.name("normed", &routed);
                    }
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
        let packed =
            linear::select::quantized(&x, &p.weight_at("bank", [W, W, W], W4A8), &routes, top_k);
        p.name("packed", &packed);
        let weights = p.value(
            "weights",
            Ty::Tensor {
                shape: vec![Dim::Tokens, Dim::Const(W)],
                dtype: Dtype::F32,
            },
        );
        let hidden = linear::glu(
            &packed,
            GluAct::Swiglu,
            Some(p.u32("intermediate")),
            None,
            Some(&weights),
            Some(p.f32("limit")),
        );
        p.name("hidden", &hidden);
        let routed = linear::select::quantized(
            &hidden,
            &p.weight_at("down", [W, W, W], W4A8),
            &routes,
            top_k,
        );
        p.name("routed", &routed);
        let y = linear::moe_sum_add(&routed, &p.rows("shared", W));
        p.export("y", &y);
    }));
    Rule::new("fused.moe_experts", alternatives, |m| {
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
            norm: m.get("normed").map(|_| match m.op("normed") {
                Operation::Elemwise(op) => Box::new(op.clone()),
                other => panic!("the routes' norm is computed by {other:?}"),
            }),
            sum,
            gate_add,
        }
    })
}

/// A routed gate-up select (over a dense, biased or quantized bank) and its
/// GLU (SwiGLU, its clamp or clamp-alpha form, or GeGLU) over the packed
/// gate|up rows, which stay inside the match.
#[must_use]
pub fn moe_select_glu() -> Rule {
    let mut alternatives = Vec::new();
    for select in 0..3 {
        for glu in 0..5 {
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
                    0 => linear::select::plain(&x, &bank, &routes, top_k),
                    1 => {
                        linear::select::biased(&x, &bank, &p.weight("bias", [W, W]), &routes, top_k)
                    }
                    _ => linear::select::quantized(&x, &bank, &routes, top_k),
                };
                p.name("packed", &packed);
                let y = match glu {
                    4 => linear::relu2(&packed),
                    0 => linear::glu(
                        &packed,
                        GluAct::Swiglu,
                        Some(p.u32("intermediate")),
                        None,
                        None,
                        None,
                    ),
                    1 => linear::glu(
                        &packed,
                        GluAct::Swiglu,
                        Some(p.u32("intermediate")),
                        None,
                        None,
                        Some(p.f32("limit")),
                    ),
                    3 => linear::glu(
                        &packed,
                        GluAct::GegluTanh,
                        Some(p.u32("intermediate")),
                        None,
                        None,
                        None,
                    ),
                    _ => linear::glu(
                        &packed,
                        GluAct::SwigluAlpha {
                            alpha: p.f32("alpha"),
                        },
                        Some(p.u32("intermediate")),
                        None,
                        None,
                        Some(p.f32("limit")),
                    ),
                };
                p.export("y", &y);
            }));
        }
    }
    Rule::new("fused.moe_select_glu", alternatives, |m| {
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

/// The packed gate-up projection and its GLU under each of `acts`, where
/// the MLP's down projection stays apart (a backend that holds
/// `matmul_swiglu_matmul` forms that first).
#[must_use]
pub fn matmul_glu(acts: &[GluAct]) -> Rule {
    let alternatives: Vec<Template> = acts
        .iter()
        .map(|&act| {
            template(|p| {
                let packed = linear::matmul(&p.rows("act", W), &p.weight("w", [2 * W, W]));
                p.name("packed", &packed);
                let y = linear::glu(&packed, act, Some(p.u32("intermediate")), None, None, None);
                p.export("y", &y);
            })
        })
        .collect();
    Rule::new("fused.matmul_glu", alternatives, |m| Fused::MatmulGlu {
        act: m.value("act"),
        w: m.value("w"),
        intermediate: m.u32("intermediate"),
        glu: match m.op("y") {
            Operation::Linear(poem_ir::Linear::Glu { act, .. }) => *act,
            other => panic!("`y` is computed by {other:?}, not a glu"),
        },
        packed: m.value("packed"),
        y: m.value("y"),
    })
}

#[must_use]
pub fn matmul_softcap() -> Rule {
    let pattern = template(|p| {
        let y = linear::lm_head(&p.rows("act", W), &p.weight("w", [W, W]));
        p.name("y", &y);
        let y_out = attn::logit_softcap(&y, p.f32("cap"));
        p.export("y_out", &y_out);
    });
    Rule::new("fused.matmul_softcap", [pattern], |m| {
        Fused::MatmulSoftcap {
            act: m.value("act"),
            w: m.value("w"),
            cap: m.f32("cap"),
            y: m.value("y"),
            y_out: m.value("y_out"),
        }
    })
}

#[must_use]
pub fn matmul_bias() -> Rule {
    let pattern = template(|p| {
        let y = linear::matmul(&p.rows("act", W), &p.weight("w", [W, W]));
        p.name("y", &y);
        let y_out = elemwise::add_bias(&p.weight("bias", [W]), &y);
        p.export("y_out", &y_out);
    });
    Rule::new("fused.matmul_bias", [pattern], |m| Fused::MatmulBias {
        act: m.value("act"),
        w: m.value("w"),
        bias: m.value("bias"),
        y: m.value("y"),
        y_out: m.value("y_out"),
    })
}

/// A router's unquantized product read only by the top-k it feeds, a
/// softmax, sigmoid or sqrt-softplus pick (bf16 or f32 logits alike: the
/// product is one op either way). Its bias, gain and hint pass through; a
/// hashed route, which reads its experts off a table, stays apart.
#[must_use]
pub fn matmul_moe_route() -> Rule {
    let pattern = template(|p| {
        let logits = linear::matmul(&p.rows("act", W), &p.weight("w", [W, W]));
        p.name("logits", &logits);
        for field in ["experts", "top_k", "score", "bias", "scale", "hint"] {
            p.free(field);
        }
        let (routes, weights) = linear::moe_route(
            &logits,
            W as u32,
            8,
            RouteScore::Softmax,
            None,
            None,
            None,
            None,
            None,
        );
        p.export("routes", &routes);
        p.export("weights", &weights);
    });
    Rule::new("fused.matmul_moe_route", [pattern], |m| {
        let Operation::Linear(Linear::MoeRoute {
            bias,
            scale,
            hint,
            experts,
            top_k,
            score,
            ..
        }) = m.op("routes")
        else {
            panic!("`routes` comes from a route");
        };
        Fused::MatmulMoeRoute {
            act: m.value("act"),
            w: m.value("w"),
            bias: *bias,
            scale: *scale,
            hint: *hint,
            experts: *experts,
            top_k: *top_k,
            score: *score,
            logits: m.value("logits"),
            routes: m.value("routes"),
            weights: m.value("weights"),
        }
    })
    .when(|m| {
        matches!(
            m.op("routes"),
            Operation::Linear(Linear::MoeRoute {
                score: RouteScore::Softmax
                    | RouteScore::Sigmoid { .. }
                    | RouteScore::SqrtSoftplus { .. },
                ..
            })
        )
    })
}

/// A projection cut into two row halves (Qwen3.5's packed qkv|z).
#[must_use]
pub fn matmul_split_rows() -> Rule {
    let pattern = template(|p| {
        // The product's row as wide as `ssm_gdn_step`'s `qkvz`: the split
        // width is a pattern sentinel past any small width, and the rest of
        // the row is what remains of it.
        let y = linear::matmul(&p.rows("act", W), &p.weight("w", [1 << 40, W]));
        p.name("y", &y);
        // Any cut matches: the pattern cuts at a width its rows hold, and
        // the replacement reads the trace's own.
        let (left, right) = gather::split_rows(&y, W as u32 / 2);
        p.free("width");
        p.export("left", &left);
        p.export("right", &right);
    });
    Rule::new("fused.matmul_split_rows", [pattern], |m| {
        let Operation::Gather(Gather::SplitRows { width, .. }) = m.op("left") else {
            panic!("`left` comes from a row split");
        };
        Fused::MatmulSplitRows {
            act: m.value("act"),
            w: m.value("w"),
            width: *width,
            y: m.value("y"),
            left: m.value("left"),
            right: m.value("right"),
        }
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
        Operation::Elemwise(Elemwise::Layernorm {
            eps,
            weight: None,
            bias: None,
            ..
        }) => NormKind::Layernorm { eps: *eps },
        Operation::Elemwise(Elemwise::Rmsnorm {
            axis: NormAxis::Head(head_dim),
            eps,
            weight: NormWeight::None,
            ..
        }) => NormKind::Rmsnorm {
            head_dim: *head_dim,
            eps: *eps,
        },
        other => panic!("{other:?} is not a scale-free norm"),
    }
}

fn modulate_form(op: &Operation) -> ModulateForm {
    match op {
        Operation::Elemwise(Elemwise::Modulate { form, .. }) => *form,
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
#[must_use]
pub fn gated_residual_norm_modulate() -> Rule {
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
    Rule::new("fused.gated_residual_norm_modulate", alternatives, |m| {
        Fused::GatedResidualNormModulate {
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
        }
    })
}

/// A norm without a scale, modulated.
#[must_use]
pub fn norm_modulate() -> Rule {
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
    Rule::new("fused.norm_modulate", alternatives, |m| {
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

/// The part of each head a [`norm_rope`] turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotated {
    /// Its first dims (`Turn::Partial`), the norm scaled per head: Qwen's q.
    First,
    /// Its last dims (`Turn::Last`), the norm scaled over a row one head
    /// wide or scale-free per head: DeepSeek-V4's q, kv plane, compressed
    /// latent and index keys.
    Last,
}

/// A head's RMS norm and the rotation of the part of each head `rotated`
/// names, the rotated heads the norm's plane.
#[must_use]
pub fn norm_rope(rotated: Rotated) -> Rule {
    fn replace(m: &Match) -> Fused {
        let Operation::Elemwise(Elemwise::Rope { turn, .. }) = m.op("q_out") else {
            unreachable!("the pattern's last op is the rotation")
        };
        Fused::NormRope {
            x: m.value("x"),
            weight: m.get("weight"),
            head_dim: m.u32("head_dim"),
            eps: m.f32("eps"),
            positions: m.value("positions"),
            theta: m.f32("theta"),
            turn: *turn,
            y: m.value("y"),
            q_out: m.value("q_out"),
        }
    }
    match rotated {
        Rotated::First => {
            let pattern = template(|p| {
                let positions = p.indices("positions");
                let d = p.u32("head_dim");
                let y = elemwise::rmsnorm_per_head(
                    &p.rows("x", W),
                    &p.weight("weight", [W]),
                    d,
                    p.f32("eps"),
                );
                p.name("y", &y);
                let (rotary_dim, theta, span) =
                    (p.u32("rotary_dim"), p.f32("theta"), p.u32("span"));
                let q_out = elemwise::rope(
                    &y,
                    &positions,
                    d,
                    theta,
                    Turn::Partial { rotary_dim, span },
                    None,
                    None,
                )
                .0;
                p.export("q_out", &q_out);
            });
            Rule::new("fused.norm_rope", [pattern], replace).form("first")
        }
        Rotated::Last => {
            let rope = |p: &poem::pattern::Pattern, y: &poem::Value| {
                p.free("interleaved");
                p.free("inverse");
                p.free("yarn");
                let positions = p.indices("positions");
                elemwise::rope(
                    y,
                    &positions,
                    p.u32("head_dim"),
                    p.f32("theta"),
                    Turn::Last {
                        rotary_dim: p.u32("rotary_dim"),
                        interleaved: false,
                        inverse: false,
                        yarn: None,
                    },
                    None,
                    None,
                )
                .0
            };
            let scale_free = template(move |p| {
                let y =
                    elemwise::rmsnorm_no_scale(&p.rows("x", W), p.u32("norm_dim"), p.f32("eps"));
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
            Rule::new("fused.norm_rope", [scale_free, scaled], replace)
                .form("last")
                // The norm's heads are the rotation's: a scale-free norm per
                // head of the rotation's width, a scaled one over a row one
                // head wide.
                .when(|m| match m.get_u32("norm_dim") {
                    Some(norm_dim) => norm_dim == m.u32("head_dim"),
                    None => m.width("x") == Some(u64::from(m.u32("head_dim"))),
                })
        }
    }
}

/// An index's queries rotated and ranked, the rotated queries read by the
/// ranking alone (DeepSeek-V4's indexer).
#[must_use]
pub fn index_topk_rope() -> Rule {
    let chain = |weighted: bool| {
        template(move |p| {
            p.free("interleaved");
            p.free("inverse");
            p.free("yarn");
            p.free("score");
            let positions = p.indices("positions");
            let d = p.u32("head_dim");
            let q = elemwise::rope(
                &p.rows("q", W),
                &positions,
                d,
                p.f32("theta"),
                Turn::Last {
                    rotary_dim: p.u32("rotary_dim"),
                    interleaved: false,
                    inverse: false,
                    yarn: None,
                },
                None,
                None,
            )
            .0;
            p.name("q_rot", &q);
            let weights = weighted.then(|| p.rows("weights", W));
            let sel = attn::index_topk(
                &q,
                weights.as_ref(),
                p.cache("keys"),
                p.u32("heads"),
                d,
                p.u32("top_k"),
                p.u32("ratio"),
                poem_ir::IndexScore::Pooled,
            );
            p.export("selection", &sel);
        })
    };
    Rule::new("fused.index_topk_rope", [chain(true), chain(false)], |m| {
        let Operation::Elemwise(rotate) = m.op("q_rot") else {
            unreachable!("the rotated queries come from a rotation")
        };
        let Operation::Attn(rank) = m.op("selection") else {
            unreachable!("the selection comes from the ranking")
        };
        Fused::IndexTopkRope {
            rotate: Box::new(rotate.clone()),
            rank: Box::new(rank.clone()),
        }
    })
}

/// DeepSeek-V4's q (scale-free per-head norm, tail rotated) and kv plane
/// (scaled norm of a one-head row, rotated and cached) of one layer: same
/// positions and rotation, and nothing between them reads either.
#[must_use]
pub fn norm_rope_pair() -> Rule {
    let pattern = template(|p| {
        p.free("interleaved");
        p.free("inverse");
        p.free("yarn");
        let positions = p.indices("positions");
        let (rotary, d, theta) = (p.u32("rotary_dim"), p.u32("head_dim"), p.f32("theta"));
        let qn = elemwise::rmsnorm_no_scale(&p.rows("q", W), d, p.f32("q_eps"));
        p.name("q_y", &qn);
        let q = elemwise::rope(
            &qn,
            &positions,
            d,
            theta,
            Turn::Last {
                rotary_dim: rotary,
                interleaved: false,
                inverse: false,
                yarn: None,
            },
            None,
            None,
        )
        .0;
        p.export("q_out", &q);
        let kn = elemwise::rmsnorm(&p.rows("kv", W), &p.weight("weight", [W]), p.f32("kv_eps"));
        let k = elemwise::rope(
            &kn,
            &positions,
            d,
            theta,
            Turn::Last {
                rotary_dim: rotary,
                interleaved: false,
                inverse: false,
                yarn: None,
            },
            None,
            None,
        )
        .0;
        p.name("kv_y", &k);
        let cache = p.cache("cache");
        let (write_page, write_offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
        attn::kv_append(&k, None, cache, &write_page, &write_offset);
    });
    Rule::new("fused.norm_rope_pair", [pattern], |m| {
        let rope = |name: &str| match *m.op(name) {
            Operation::Elemwise(Elemwise::Rope {
                turn:
                    Turn::Last {
                        interleaved,
                        inverse,
                        yarn,
                        ..
                    },
                ..
            }) => (interleaved, inverse, yarn),
            _ => unreachable!("`{name}` comes from a rotation"),
        };
        let (q_interleaved, q_inverse, q_yarn) = rope("q_out");
        let (kv_interleaved, kv_inverse, kv_yarn) = rope("kv_y");
        Fused::NormRopePair {
            q: Box::new(Fused::NormRope {
                x: m.value("q"),
                weight: None,
                head_dim: m.u32("head_dim"),
                eps: m.f32("q_eps"),
                positions: m.value("positions"),
                theta: m.f32("theta"),
                turn: Turn::Last {
                    rotary_dim: m.u32("rotary_dim"),
                    interleaved: q_interleaved,
                    inverse: q_inverse,
                    yarn: q_yarn,
                },
                y: m.value("q_y"),
                q_out: m.value("q_out"),
            }),
            kv: Box::new(Fused::NormRopeLastAppend {
                x: m.value("kv"),
                weight: m.value("weight"),
                head_dim: m.u32("head_dim"),
                eps: m.f32("kv_eps"),
                positions: m.value("positions"),
                rotary_dim: m.u32("rotary_dim"),
                theta: m.f32("theta"),
                interleaved: kv_interleaved,
                inverse: kv_inverse,
                yarn: kv_yarn,
                y: m.value("kv_y"),
                cache: m.value("cache"),
                write_page: m.value("write_page"),
                write_offset: m.value("write_offset"),
            }),
        }
    })
    // One layer's: the plane's row is one head of q's width.
    .when(|m| m.width("kv") == Some(u64::from(m.u32("head_dim"))))
}

/// DeepSeek-V4's compressor at its boundaries: the gathered entries normed,
/// their tail rotated at the boundaries' rope positions, and written to the
/// pool, the entries and the normed rows read by nothing else.
#[must_use]
pub fn pool_compress() -> Rule {
    let chain = |biased: bool, stated: bool| {
        template(move |p| {
            p.free("interleaved");
            p.free("inverse");
            p.free("yarn");
            let (bpos, breq) = (p.indices("bpos"), p.indices("breq"));
            let pages = p.cache("pages");
            if stated {
                // The fire's compressor state written just before the gather.
                attn::pool_state_write(
                    &p.rows("s_kv", W),
                    &p.rows("s_score", W),
                    pages,
                    &p.rows("s_page", 1),
                    &p.rows("s_off", 1),
                    p.u32("head_dim"),
                    p.u32("ratio"),
                );
            }
            let ape = biased.then(|| p.weight("ape", [W, W]));
            let gathered = attn::pool_gather(
                &bpos,
                &breq,
                pages,
                ape.as_ref(),
                p.u32("head_dim"),
                p.u32("ratio"),
                poem_ir::Dtype::Bf16,
            );
            p.name("gathered", &gathered);
            let latent = elemwise::rmsnorm(&gathered, &p.weight("weight", [W]), p.f32("eps"));
            p.name("latent", &latent);
            let brope = p.indices("brope");
            let rotated = elemwise::rope(
                &latent,
                &brope,
                p.u32("head_dim"),
                p.f32("theta"),
                Turn::Last {
                    rotary_dim: p.u32("rotary_dim"),
                    interleaved: false,
                    inverse: false,
                    yarn: None,
                },
                None,
                None,
            )
            .0;
            p.name("rotated", &rotated);
            let pool = p.cache("pool");
            let (write_page, write_offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
            attn::pool_kv_append(&rotated, &bpos, &breq, pool, &write_page, &write_offset);
        })
    };
    Rule::new(
        "fused.pool_compress",
        [
            chain(true, true),
            chain(false, true),
            chain(true, false),
            chain(false, false),
        ],
        |m| {
            let Operation::Attn(gather) = m.op("gathered") else {
                unreachable!("the entries come from the gather")
            };
            let Operation::Elemwise(rotate) = m.op("rotated") else {
                unreachable!("the rotated rows come from the rotation")
            };
            let append = m
                .trace
                .nodes
                .iter()
                .find_map(|node| match &node.op {
                    Operation::Attn(a @ poem_ir::Attn::PoolKvAppend { entries, .. })
                        if *entries == m.value("rotated") =>
                    {
                        Some(a.clone())
                    }
                    _ => None,
                })
                .expect("the rotated rows are appended");
            let state =
                m.get("s_kv").map(|kv| {
                    let op =
                        m.trace
                            .nodes
                            .iter()
                            .find_map(|node| match &node.op {
                                Operation::Attn(
                                    a @ poem_ir::Attn::PoolStateWrite { kv: at, .. },
                                ) if *at == kv => Some(a.clone()),
                                _ => None,
                            })
                            .expect("the state is written");
                    Box::new(op)
                });
            Fused::PoolCompress {
                gather: Box::new(gather.clone()),
                weight: m.value("weight"),
                eps: m.f32("eps"),
                latent: m.value("latent"),
                rotate: Box::new(rotate.clone()),
                append: Box::new(append),
                state,
            }
        },
    )
}

/// [`norm_rope_last_append`] where the cache holds DeepSeek's fp8 rows (u8
/// planes: e4m3 blocks under their scales and a bf16 rotated tail), a row
/// format no paged store of an epilogue lands; a bf16 cache's write is an
/// anchored region's.
#[must_use]
pub fn norm_rope_last_append_rows() -> Rule {
    norm_rope_last_append()
        .when(|m| {
            let cache = m.value("cache");
            match m.trace.values[cache.0 as usize].def {
                poem_ir::Def::Cache(row) => matches!(
                    m.trace.caches.get(row as usize),
                    Some(poem_ir::CacheRow::Kv {
                        dtype: Dtype::U8,
                        ..
                    })
                ),
                _ => false,
            }
        })
        .form("rows")
}

/// DeepSeek-V4's kv plane: a scaled norm of a row one head wide, its last
/// dims rotated, written to the cache and read by nothing else.
#[must_use]
pub fn norm_rope_last_append() -> Rule {
    let pattern = template(|p| {
        p.free("interleaved");
        p.free("inverse");
        p.free("yarn");
        let positions = p.indices("positions");
        let y = elemwise::rmsnorm(&p.rows("x", W), &p.weight("weight", [W]), p.f32("eps"));
        let k = elemwise::rope(
            &y,
            &positions,
            p.u32("head_dim"),
            p.f32("theta"),
            Turn::Last {
                rotary_dim: p.u32("rotary_dim"),
                interleaved: false,
                inverse: false,
                yarn: None,
            },
            None,
            None,
        )
        .0;
        p.name("y", &k);
        let cache = p.cache("cache");
        let (write_page, write_offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
        attn::kv_append(&k, None, cache, &write_page, &write_offset);
    });
    Rule::new("fused.norm_rope_last_append", [pattern], |m| {
        let Operation::Elemwise(Elemwise::Rope {
            turn:
                Turn::Last {
                    interleaved,
                    inverse,
                    yarn,
                    ..
                },
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

/// mHC's f32 stream norm read only by its mix projection.
#[must_use]
pub fn hc_norm_project() -> Rule {
    let pattern = template(|p| {
        let normed = elemwise::hc_rmsnorm_f32(&p.rows("streams", W), p.f32("eps"));
        let count = p.u32("stream_count");
        let weight = p.weight("weight", [u64::from(count), W]);
        p.export("mixes", &elemwise::hc_project(&normed, &weight, count));
    });
    Rule::new("fused.hc_norm_project", [pattern], |m| {
        Fused::HcNormProject {
            streams: m.value("streams"),
            eps: m.f32("eps"),
            weight: m.value("weight"),
            stream_count: m.u32("stream_count"),
            mixes: m.value("mixes"),
        }
    })
}

/// A gated-delta layer's decode step: the split of its in-projection, the
/// conv update, the gate prep and the recurrence, all under one guard.
#[must_use]
pub fn gdn_step() -> Rule {
    let pattern = template(|p| {
        let qkvz = p.rows("qkvz", 1 << 40);
        let (qkv, z) = gather::split_rows(&qkvz, p.u32("width"));
        p.export("qkv", &qkv);
        p.export("z", &z);
        let conv_state = p.cache("conv_state");
        let conv = ssm::causal_conv1d_dilated(
            &qkv,
            &p.weight("conv_weight", [W, W]),
            None,
            conv_state,
            p.u32("conv_width"),
            p.u32("dilation"),
        );
        p.name("conv", &conv);
        let gates = ssm::gdn_gates(
            &p.rows("ba", W),
            &p.weight("dt_bias", [W]),
            &p.weight("a_log", [W]),
        );
        p.name("gates", &gates);
        let y = ssm::gated_delta(
            &conv,
            &z,
            &gates,
            p.cache("state"),
            p.u32("k_heads"),
            p.u32("v_heads"),
            p.u32("k_dim"),
            p.u32("v_dim"),
        );
        p.export("y", &y);
    });
    Rule::new("fused.gdn_step", [pattern], |m| Fused::GdnStep {
        qkvz: m.value("qkvz"),
        width: m.u32("width"),
        qkv: m.value("qkv"),
        z: m.value("z"),
        conv_weight: m.value("conv_weight"),
        conv_state: m.value("conv_state"),
        conv_width: m.u32("conv_width"),
        dilation: m.u32("dilation"),
        conv: m.value("conv"),
        ba: m.value("ba"),
        dt_bias: m.value("dt_bias"),
        a_log: m.value("a_log"),
        gates: m.value("gates"),
        state: m.value("state"),
        k_heads: m.u32("k_heads"),
        v_heads: m.u32("v_heads"),
        k_dim: m.u32("k_dim"),
        v_dim: m.u32("v_dim"),
        y: m.value("y"),
    })
}

/// Qwen's gated attention prologue: the q/gate split, the two per-head
/// norms, the partial rotation and the kv append, all under one guard.
/// Qwen's gated attention prologue: q cut from its gate, q and k normed per
/// head (plus-one weights), turned, and k with its v written to the cache.
/// Under `Intermediates::Written` the cut q may be read past it and every
/// intermediate lands. Under `Unwritten` nothing reads them, and the gated
/// row is bf16.
#[must_use]
pub fn q_gate_norm_rope_append(intermediates: Intermediates) -> Rule {
    let written = intermediates == Intermediates::Written;
    let pattern = template(|p| {
        let d = p.u32("d");
        let (q, gate) = gather::split_q_gate(&p.rows("qg", 2 * W), d);
        if written {
            p.export("q", &q);
        } else {
            p.name("q", &q);
        }
        p.export("gate", &gate);
        let qn =
            elemwise::rmsnorm_per_head_plus_one(&q, &p.weight("q_norm", [W]), d, p.f32("q_eps"));
        p.name("qn", &qn);
        let kn = elemwise::rmsnorm_per_head_plus_one(
            &p.rows("k", W),
            &p.weight("k_norm", [W]),
            d,
            p.f32("k_eps"),
        );
        p.name("kn", &kn);
        let (q_out, k_out) = both(elemwise::rope(
            &qn,
            &p.indices("positions"),
            d,
            p.f32("theta"),
            Turn::Partial {
                rotary_dim: p.u32("rotary"),
                span: p.u32("span"),
            },
            Some(&kn),
            None,
        ));
        p.export("q_out", &q_out);
        p.name("k_out", &k_out);
        let (write_page, write_offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
        attn::kv_append(
            &k_out,
            Some(&p.rows("v", W)),
            p.cache("cache"),
            &write_page,
            &write_offset,
        );
    });
    fn replace(m: &Match, intermediates: Intermediates) -> Fused {
        Fused::QGateNormRopeAppend {
            qg: m.value("qg"),
            intermediates,
            head_dim: m.u32("d"),
            q: m.value("q"),
            gate: m.value("gate"),
            q_norm: m.value("q_norm"),
            q_eps: m.f32("q_eps"),
            qn: m.value("qn"),
            k: m.value("k"),
            k_norm: m.value("k_norm"),
            k_eps: m.f32("k_eps"),
            kn: m.value("kn"),
            positions: m.value("positions"),
            rotary_dim: m.u32("rotary"),
            theta: m.f32("theta"),
            span: m.u32("span"),
            q_out: m.value("q_out"),
            k_out: m.value("k_out"),
            v: m.value("v"),
            cache: m.value("cache"),
            write_page: m.value("write_page"),
            write_offset: m.value("write_offset"),
        }
    }
    if written {
        Rule::new("fused.q_gate_norm_rope_append", [pattern], |m| {
            replace(m, Intermediates::Written)
        })
        .form("written")
    } else {
        Rule::new("fused.q_gate_norm_rope_append", [pattern], |m| {
            replace(m, Intermediates::Unwritten)
        })
        .form("unwritten")
        .when(|m| m.dtype("qg") == Some(Dtype::Bf16))
    }
}

/// An op's two outputs, its second asked for: a rope's queries and the
/// keys it was given, an attention's output and log-sum-exp.
fn both((q, k): (poem::Value, Option<poem::Value>)) -> (poem::Value, poem::Value) {
    (q, k.expect("an output asked for lands"))
}
