use model_dsl::{Dtype, Weight};
use model_loader::contract::ModelContract;

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    pub kv: Dtype,
    pub softcap: Option<f32>,
    pub embed: Weight,
    pub ple: Option<Ple>,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,
}

pub struct Ple {
    pub dim: u32,
    pub model_proj: Weight,
    pub model_norm: Weight,
    pub model_norm_eps: f32,
    pub per_layer: Vec<PleLayer>,
}

pub struct PleLayer {
    pub table: Weight,
    pub gate: Weight,
    pub proj: Weight,
    pub norm: Weight,
    pub norm_eps: f32,
    pub scalar: Weight,
}

pub struct Layer {
    pub attn: Attn,
    pub o_proj: Weight,
    pub attn_norm: Weight,
    pub attn_norm_eps: f32,
    pub post_attn_norm: Weight,
    pub post_attn_norm_eps: f32,
    pub pre_ffw_norm: Weight,
    pub pre_ffw_norm_eps: f32,
    pub post_ffw_norm: Weight,
    pub post_ffw_norm_eps: f32,

    pub gate_up: Weight,
    pub inter: u32,
    pub down: Weight,
}

pub struct Attn {
    pub kind: AttnKind,
    pub q_heads: u32,
    pub sm_scale: f32,
    pub q_norm: Weight,
    pub q_norm_eps: f32,

    pub kv: String,
    pub banks: AttnBanks,
}

pub enum AttnKind {
    Full {
        head_dim: u32,
        kv_heads: u32,
        rotary_dim: u32,
        theta: f32,
    },
    Sliding {
        head_dim: u32,
        kv_heads: u32,
        window: u32,
        theta: f32,
    },
}

impl AttnKind {
    pub fn head_dim(&self) -> u32 {
        match *self {
            AttnKind::Full { head_dim, .. } | AttnKind::Sliding { head_dim, .. } => head_dim,
        }
    }
    pub fn kv_heads(&self) -> u32 {
        match *self {
            AttnKind::Full { kv_heads, .. } | AttnKind::Sliding { kv_heads, .. } => kv_heads,
        }
    }
    pub fn theta(&self) -> f32 {
        match *self {
            AttnKind::Full { theta, .. } | AttnKind::Sliding { theta, .. } => theta,
        }
    }
    pub fn window(&self) -> Option<u32> {
        match *self {
            AttnKind::Full { .. } => None,
            AttnKind::Sliding { window, .. } => Some(window),
        }
    }
    pub fn sliding(&self) -> bool {
        matches!(self, AttnKind::Sliding { .. })
    }
}

#[allow(clippy::large_enum_variant)]
pub enum AttnBanks {
    Owned {
        qkv: Weight,
        k_norm: Weight,
        k_norm_eps: f32,
    },
    Shared {
        q_proj: Weight,
    },
}

struct Dims {
    hidden: u32,
    layers: u32,
    full_every: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    global_head_dim: u32,
    global_kv_heads: u32,
    global_rotary_dim: u32,
    theta_local: f32,
    theta_global: f32,
    sm_scale: f32,
    intermediate: u32,
    vocab: u32,
    shared_tail: Option<u32>,
    ple_dim: Option<u32>,
    softcap: Option<f32>,
    window: u32,
    norm_eps: f32,
}

impl Model {
    pub fn e4b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        assemble(
            w,
            kv,
            tp,
            Dims {
                hidden: 2560,
                layers: 42,
                full_every: 6,
                q_heads: 8,
                kv_heads: 2,
                head_dim: 256,
                global_head_dim: 512,
                global_kv_heads: 2,
                global_rotary_dim: 128,
                theta_local: 10_000.0,
                theta_global: 1_000_000.0,
                sm_scale: 1.0,
                intermediate: 10_240,
                vocab: 262_144,
                shared_tail: Some(18),
                ple_dim: Some(256),
                softcap: Some(30.0),
                window: 512,
                norm_eps: 1e-6,
            },
        )
    }

    pub fn b31(w: Dtype, kv: Dtype, tp: u32) -> Model {
        assemble(
            w,
            kv,
            tp,
            Dims {
                hidden: 5376,
                layers: 60,
                full_every: 6,
                q_heads: 32,
                kv_heads: 16,
                head_dim: 256,
                global_head_dim: 512,
                global_kv_heads: 4,
                global_rotary_dim: 128,
                theta_local: 10_000.0,
                theta_global: 1_000_000.0,
                sm_scale: 1.0,
                intermediate: 21_504,
                vocab: 262_144,
                shared_tail: None,
                ple_dim: None,
                softcap: Some(30.0),
                window: 512,
                norm_eps: 1e-6,
            },
        )
    }
}

fn assemble(w: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
    assert!(
        matches!(tp, 1 | 2 | 4 | 8),
        "tp {tp} is not a world this catalog ships"
    );
    let q_heads = d.q_heads / tp;
    let kv_heads = d.kv_heads / tp;
    let global_kv_heads = d.global_kv_heads / tp;
    let intermediate = d.intermediate / tp;

    let hidden = d.hidden as u64;
    let full_at = |l: u32| l % d.full_every == d.full_every - 1;
    let shared_at = |l: u32| d.shared_tail.is_some_and(|tail| l >= d.layers - tail);
    let source = |l: u32| {
        (0..l)
            .rev()
            .find(|&s| !shared_at(s) && full_at(s) == full_at(l))
    };
    let owner = |l: u32| match d.shared_tail {
        None => l,
        Some(tail) if l < d.layers - tail => l,
        Some(tail) => source(l).unwrap_or_else(|| {
            panic!(
                "layer {l} borrows its kv cache and none of the {} layers \
                 before the shared tail is of its kind (full_every {}, \
                 shared_tail {tail})",
                d.layers - tail,
                d.full_every,
            )
        }),
    };
    let kind = |l: u32| {
        if full_at(l) {
            AttnKind::Full {
                head_dim: d.global_head_dim,
                kv_heads: global_kv_heads,
                rotary_dim: d.global_rotary_dim,
                theta: d.theta_global,
            }
        } else {
            AttnKind::Sliding {
                head_dim: d.head_dim,
                kv_heads,
                window: d.window,
                theta: d.theta_local,
            }
        }
    };

    let layers = (0..d.layers)
        .map(|l| {
            let n = |s: &str| format!("layer.{l}.{s}");
            let norm = |s: &str, len: u64| Weight::sym(n(s), [len], w);
            let ki = kind(l);
            let hd = ki.head_dim() as u64;
            let q_w = q_heads as u64 * hd;
            let kv_w = ki.kv_heads() as u64 * hd;
            let iw = intermediate as u64;
            Layer {
                attn: Attn {
                    q_heads,
                    sm_scale: d.sm_scale,
                    q_norm: norm("q_norm", hd),
                    q_norm_eps: d.norm_eps,
                    kv: format!("kv.{}", owner(l)),
                    banks: if shared_at(l) {
                        AttnBanks::Shared {
                            q_proj: Weight::sym(n("q_proj"), [q_w, hidden], w).columns(),
                        }
                    } else {
                        AttnBanks::Owned {
                            qkv: Weight::sym(n("qkv"), [q_w + 2 * kv_w, hidden], w)
                                .packed([q_w, kv_w, kv_w]),
                            k_norm: norm("k_norm", hd),
                            k_norm_eps: d.norm_eps,
                        }
                    },
                    kind: ki,
                },
                o_proj: Weight::sym(n("o_proj"), [hidden, q_w], w).rows(),
                attn_norm: norm("attn_norm", hidden),
                attn_norm_eps: d.norm_eps,
                post_attn_norm: norm("post_attn_norm", hidden),
                post_attn_norm_eps: d.norm_eps,
                pre_ffw_norm: norm("pre_ffw_norm", hidden),
                pre_ffw_norm_eps: d.norm_eps,
                post_ffw_norm: norm("post_ffw_norm", hidden),
                post_ffw_norm_eps: d.norm_eps,
                gate_up: Weight::sym(n("gate_up"), [2 * iw, hidden], w).packed([iw, iw]),
                inter: intermediate,
                down: Weight::sym(n("down"), [hidden, iw], w).rows(),
            }
        })
        .collect();

    Model {
        hidden: d.hidden,
        vocab: d.vocab,
        tp,
        kv,
        softcap: d.softcap,
        embed: Weight::sym("embed", [d.vocab as u64, hidden], w),
        ple: d.ple_dim.map(|dim| {
            let ple = dim as u64;
            Ple {
                dim,
                model_proj: Weight::sym("ple.model_proj", [d.layers as u64 * ple, hidden], w),
                model_norm: Weight::sym("ple.model_norm", [ple], w),
                model_norm_eps: d.norm_eps,
                per_layer: (0..d.layers)
                    .map(|l| PleLayer {
                        table: Weight::sym(
                            format!("layer.{l}.ple_table"),
                            [d.vocab as u64, ple],
                            w,
                        ),
                        gate: Weight::sym(format!("layer.{l}.ple_gate"), [ple, hidden], w),
                        proj: Weight::sym(format!("layer.{l}.ple_proj"), [hidden, ple], w),
                        norm: Weight::sym(format!("layer.{l}.ple_norm"), [hidden], w),
                        norm_eps: d.norm_eps,
                        scalar: Weight::sym(format!("layer.{l}.ple_scalar"), [1], w),
                    })
                    .collect(),
            }
        }),
        layers,
        final_norm: Weight::sym("final_norm", [hidden], w),
        final_norm_eps: d.norm_eps,
    }
}

impl Model {
    pub fn load(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, crate::contract::ModelError> {
        let mut claims = Vec::new();
        let mut claim = |w: &Weight| claims.push(crate::contract::claim(w, self.tp));

        claim(&self.embed);
        claim(&self.final_norm);

        for layer in &self.layers {
            claim(&layer.attn_norm);
            claim(&layer.post_attn_norm);
            claim(&layer.pre_ffw_norm);
            claim(&layer.post_ffw_norm);
            claim(&layer.attn.q_norm);
            match &layer.attn.banks {
                AttnBanks::Owned { qkv, k_norm, .. } => {
                    claim(k_norm);
                    claim(qkv);
                }

                AttnBanks::Shared { q_proj } => {
                    claim(q_proj);
                }
            }
            claim(&layer.o_proj);
            claim(&layer.gate_up);
            claim(&layer.down);
        }

        if let Some(ple) = &self.ple {
            claim(&ple.model_proj);
            claim(&ple.model_norm);
            for per in &ple.per_layer {
                claim(&per.table);
                claim(&per.gate);
                claim(&per.proj);
                claim(&per.norm);
                claim(&per.scalar);
            }
        }

        crate::contract::elaborate(src, claims)
    }
}
