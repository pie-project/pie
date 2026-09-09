use model_dsl::{Dtype, Weight};

use crate::drafter::dflash::{self, DFlash};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    pub q_heads: u32,
    pub sliding: Sliding,
    pub global: Global,

    pub adapters: Adapters,

    pub tower: Option<Tower>,

    pub kv: Dtype,
    pub softcap: Option<f32>,
    pub embed: Weight,
    pub ple: Option<Ple>,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,

    pub draft: Option<Draft>,

    pub assistant: Option<Assistant>,

    pub dflash: Option<DFlash>,
    pub self_cond: Option<SelfCond>,
}

pub struct SelfCond {
    pub taps: u32,
    pub pre_norm: Weight,
    pub norm_eps: f32,
    pub gate_up: Weight,
    pub inter: u32,
    pub down: Weight,
}

pub struct Assistant {
    pub depth: u32,
    pub pre_embed: Weight,
    pub pre_hidden: Weight,
    pub post: Weight,
    pub embed: Weight,
    pub norm: Weight,
    pub norm_eps: f32,
    pub layers: Vec<AssistantLayer>,
}

pub struct AssistantLayer {
    pub attn: Attn,
    pub o_proj: Weight,
    pub attn_norm: Weight,
    pub post_attn_norm: Weight,
    pub pre_ffw_norm: Weight,
    pub post_ffw_norm: Weight,
    pub gate_up: Weight,
    pub inter: u32,
    pub down: Weight,
    pub scalar: Weight,
}

pub const SELF_COND_TAPS: u32 = 64;

const ASSISTANT_HIDDEN: u32 = 1024;
const ASSISTANT_INTER: u32 = 8192;
const ASSISTANT_READINGS: [Reading; 4] = [
    Reading::Sliding,
    Reading::Sliding,
    Reading::Sliding,
    Reading::Global,
];
pub const ASSISTANT_DEPTH: u32 = 1;

pub struct Draft {
    pub fc_embed: Weight,
    pub fc_hidden: Weight,
    pub attn_norm: Weight,
    pub post_attn_norm: Weight,
    pub pre_ffw_norm: Weight,
    pub post_ffw_norm: Weight,
    pub attn: Attn,
    pub o_proj: Weight,
    pub gate_up: Weight,
    pub inter: u32,
    pub down: Weight,
    pub norm_eps: f32,
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

pub struct Tower {
    pub hidden: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub pool: u32,
    pub patch_width: u32,
    pub positions: u32,
    pub theta: f32,
    pub norm_eps: f32,
    pub sm_scale: f32,
    pub patch_embed: Weight,
    pub pos_embed: Weight,
    pub blocks: Vec<TowerBlock>,
    pub projection: Weight,
    pub std: Option<Standardization>,
}

pub struct Standardization {
    pub bias: Weight,
    pub scale: Weight,
}

pub struct Clippable {
    pub bank: Weight,
    pub clip: Option<Bounds>,
}

pub struct Bounds {
    pub in_lo: Weight,
    pub in_hi: Weight,
    pub out_lo: Weight,
    pub out_hi: Weight,
}

pub struct TowerBlock {
    pub attn_norm: Weight,
    pub post_attn_norm: Weight,
    pub pre_ffw_norm: Weight,
    pub post_ffw_norm: Weight,
    pub q: Clippable,
    pub k: Clippable,
    pub v: Clippable,
    pub o: Clippable,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub gate: Clippable,
    pub up: Clippable,
    pub down: Clippable,
}

pub use crate::adapter::Adapters;

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

    pub scalar: Option<Weight>,

    pub lora_a: Weight,
    pub lora_b: Weight,

    pub moe: Option<Moe>,
}

pub struct Moe {
    pub router_norm: Weight,
    pub router_norm_eps: f32,
    pub router: Weight,
    pub per_expert_scale: Weight,
    pub pre_ffw_norm_2: Weight,
    pub pre_ffw_norm_2_eps: f32,
    pub post_ffw_norm_1: Weight,
    pub post_ffw_norm_1_eps: f32,
    pub post_ffw_norm_2: Weight,
    pub post_ffw_norm_2_eps: f32,
    pub gate_up: Weight,
    pub down: Weight,
    pub experts: u32,
    pub top_k: u32,
    pub inter: u32,
}

pub struct Attn {
    pub reading: Reading,
    pub sm_scale: f32,
    pub q_norm: Weight,
    pub q_norm_eps: f32,

    pub kv: String,
    pub banks: AttnBanks,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Reading {
    Sliding = 0,
    Global = 1,
}

pub struct Sliding {
    pub head_dim: u32,
    pub kv_heads: u32,
    pub window: u32,
    pub theta: f32,
}

pub struct Global {
    pub head_dim: u32,
    pub kv_heads: u32,
    pub rotary_dim: u32,
    pub theta: f32,
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

#[derive(Clone, Copy)]
struct TowerDims {
    depth: u32,
    hidden: u32,
    heads: u32,
    inter: u32,
    patch_width: u32,
    pool: u32,
    positions: u32,
    out_hidden: u32,
    theta: f32,
    norm_eps: f32,
    sm_scale: f32,
    clipped: bool,
    standardize: bool,
}

impl TowerDims {
    const fn e4b() -> TowerDims {
        TowerDims {
            depth: 16,
            hidden: 768,
            heads: 12,
            inter: 3072,
            patch_width: 3 * 16 * 16,
            pool: 3,
            positions: 10_240,
            out_hidden: 2560,
            theta: 100.0,
            norm_eps: 1e-6,
            sm_scale: 1.0,
            clipped: true,
            standardize: false,
        }
    }

    const fn wide(out_hidden: u32) -> TowerDims {
        TowerDims {
            depth: 27,
            hidden: 1152,
            heads: 16,
            inter: 4304,
            patch_width: 3 * 16 * 16,
            pool: 3,
            positions: 10_240,
            out_hidden,
            theta: 100.0,
            norm_eps: 1e-6,
            sm_scale: 1.0,
            clipped: false,
            standardize: true,
        }
    }
}

const SLIDING: Option<u32> = Some(2_048);

pub const GEMMA4_26B_A4B_DFLASH: dflash::Head = dflash::Head {
    taps: &[1, 6, 11, 17, 22, 27],
    windows: &[SLIDING, SLIDING, SLIDING, SLIDING, None],
    q_heads: 32,
    kv_heads: 8,
    head_dim: 128,
    inter: 5_632,
    theta: 1_000_000.0,
    block: 16,
    mask_token: 4,
    proposals_from: 1,
    conv: None,
    readout: dflash::Readout::Argmax,
    attn_bias: false,
};

struct Dims {
    tower: Option<TowerDims>,
    self_cond: bool,
    self_cond_w: Option<Dtype>,
    draft: bool,
    assistant: bool,
    dflash: Option<&'static dflash::Head>,
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
    moe: Option<MoeDims>,
}

#[derive(Clone, Copy)]
struct MoeDims {
    experts: u32,
    top_k: u32,
    inter: u32,
}

impl Model {
    pub fn e4b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::e4b_dims())
    }

    pub fn e4b_mini(layers: u32, w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::e4b_dims();
        let owned = d.layers - d.shared_tail.unwrap_or(0);
        d.layers = layers;
        d.shared_tail = (layers > owned).then(|| layers - owned);
        Model::new(w, kv, tp, d)
    }

    fn e4b_dims() -> Dims {
        Dims {
            tower: None,
            self_cond: false,
            self_cond_w: None,
            draft: false,
            assistant: false,
            dflash: None,
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
            moe: None,
        }
    }

    pub fn e4b_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::e4b_dims();
        d.tower = Some(TowerDims::e4b());
        Model::new(w, kv, tp, d)
    }

    pub fn e4b_eagle(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::e4b_dims();
        d.draft = true;
        Model::new(w, kv, tp, d)
    }

    pub fn b31_mtp(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::b31_dims();
        d.assistant = true;
        Model::new(w, kv, tp, d)
    }

    pub fn b31(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::b31_dims())
    }

    pub fn b31_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::b31_dims();
        d.tower = Some(TowerDims::wide(d.hidden));
        Model::new(w, kv, tp, d)
    }

    fn b31_dims() -> Dims {
        Dims {
            tower: None,
            self_cond: false,
            self_cond_w: None,
            draft: false,
            assistant: false,
            dflash: None,
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
            window: 1024,
            norm_eps: 1e-6,
            moe: None,
        }
    }

    pub fn a4b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::a4b_dims())
    }

    pub fn a4b_diffusion(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::a4b_diffusion_experts(w, w, kv, tp)
    }

    pub fn a4b_diffusion_experts(w: Dtype, xw: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::a4b_dims();
        d.self_cond = true;
        Model::new_with_experts(w, xw, kv, tp, d)
    }

    pub fn a4b_diffusion_experts_self_cond(
        w: Dtype,
        xw: Dtype,
        sw: Dtype,
        kv: Dtype,
        tp: u32,
    ) -> Model {
        let mut d = Model::a4b_dims();
        d.self_cond = true;
        d.self_cond_w = Some(sw);
        Model::new_with_experts(w, xw, kv, tp, d)
    }

    pub fn a4b_mtp(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::a4b_dims();
        d.assistant = true;
        Model::new(w, kv, tp, d)
    }

    pub fn a4b_dflash(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::a4b_dims();
        d.dflash = Some(&GEMMA4_26B_A4B_DFLASH);
        Model::new(w, kv, tp, d)
    }

    pub fn a4b_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::a4b_dims();
        d.tower = Some(TowerDims::wide(d.hidden));
        Model::new(w, kv, tp, d)
    }

    fn a4b_dims() -> Dims {
        Dims {
            tower: None,
            self_cond: false,
            self_cond_w: None,
            draft: false,
            assistant: false,
            dflash: None,
            hidden: 2816,
            layers: 30,
            full_every: 6,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 256,
            global_head_dim: 512,
            global_kv_heads: 2,
            global_rotary_dim: 128,
            theta_local: 10_000.0,
            theta_global: 1_000_000.0,
            sm_scale: 1.0,
            intermediate: 2112,
            vocab: 262_144,
            shared_tail: None,
            ple_dim: None,
            softcap: Some(30.0),
            window: 1024,
            norm_eps: 1e-6,
            moe: Some(MoeDims {
                experts: 128,
                top_k: 8,
                inter: 704,
            }),
        }
    }

    fn new(w: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        Model::new_with_experts(w, w, kv, tp, d)
    }

    fn new_with_experts(w: Dtype, xw: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        for (count, what) in [
            (d.q_heads, "query heads"),
            (d.kv_heads, "sliding KV heads"),
            (d.global_kv_heads, "global KV heads"),
            (d.intermediate, "the intermediate"),
        ] {
            assert!(
                count.is_multiple_of(tp),
                "tp {tp} does not divide this text's {count} {what}; a gemma-4 \
                 row is cut by heads and by the intermediate, and every count \
                 it cuts must divide the rank count"
            );
        }
        if let Some(moe) = d.moe {
            assert!(
                moe.inter.is_multiple_of(tp),
                "tp {tp} does not divide the routed experts' {} intermediate",
                moe.inter
            );
        }
        let dense = crate::dense(w);
        let gate = match w {
            Dtype::U4g64 => Dtype::U8g64,
            other => other,
        };
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
        let sliding = Sliding {
            head_dim: d.head_dim,
            kv_heads,
            window: d.window,
            theta: d.theta_local,
        };
        let global = Global {
            head_dim: d.global_head_dim,
            kv_heads: global_kv_heads,
            rotary_dim: d.global_rotary_dim,
            theta: d.theta_global,
        };

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str, len: u64| Weight::sym(n(s), [len], dense);
                let (lora_a, lora_b) =
                    crate::adapter::banks(&format!("layer.{l}"), ADAPTERS, hidden, dense);
                let reading = if full_at(l) {
                    Reading::Global
                } else {
                    Reading::Sliding
                };
                let (head_dim, row_heads) = match reading {
                    Reading::Sliding => (sliding.head_dim, sliding.kv_heads),
                    Reading::Global => (global.head_dim, global.kv_heads),
                };
                let hd = head_dim as u64;
                let q_w = q_heads as u64 * hd;
                let kv_w = row_heads as u64 * hd;
                let iw = intermediate as u64;
                Layer {
                    attn: Attn {
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
                        reading,
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
                    scalar: d
                        .ple_dim
                        .is_none()
                        .then(|| Weight::sym(n("scalar"), [1], dense)),
                    lora_a,
                    lora_b,
                    moe: d.moe.map(|m| {
                        let mi = (m.inter / tp) as u64;
                        Moe {
                            router_norm: norm("router_norm", hidden),
                            router_norm_eps: d.norm_eps,
                            router: Weight::sym(n("router"), [m.experts as u64, hidden], gate),
                            per_expert_scale: Weight::sym(
                                n("per_expert_scale"),
                                [m.experts as u64],
                                dense,
                            )
                            .columns(),
                            pre_ffw_norm_2: norm("pre_ffw_norm_2", hidden),
                            pre_ffw_norm_2_eps: d.norm_eps,
                            post_ffw_norm_1: norm("post_ffw_norm_1", hidden),
                            post_ffw_norm_1_eps: d.norm_eps,
                            post_ffw_norm_2: norm("post_ffw_norm_2", hidden),
                            post_ffw_norm_2_eps: d.norm_eps,
                            gate_up: Weight::sym(
                                n("experts_gate_up"),
                                [m.experts as u64, 2 * mi, hidden],
                                xw,
                            )
                            .bank([mi, mi]),
                            down: Weight::sym(
                                n("experts_down"),
                                [m.experts as u64, hidden, mi],
                                xw,
                            )
                            .rows(),
                            experts: m.experts,
                            top_k: m.top_k,
                            inter: m.inter / tp,
                        }
                    }),
                }
            })
            .collect();

        let tower = d.tower.map(|t| {
            assert_eq!(
                t.out_hidden, d.hidden,
                "a tower's projection lands a TRUNK row; a mismatch would scatter a \
                 rectangle of the wrong width into the embedding"
            );
            assert_eq!(
                t.hidden % t.heads,
                0,
                "a {}-wide tower does not divide into {} heads",
                t.hidden,
                t.heads
            );
            let th = t.hidden as u64;
            let ti = t.inter as u64;
            let head_dim = t.hidden / t.heads;
            let n = |s: String| format!("vision.{s}");
            let bank = |s: String, dims: [u64; 2]| Weight::sym(n(s), dims, dense);
            let vec1 = |s: String, len: u64| Weight::sym(n(s), [len], dense);
            let clip = |s: &str, dims: [u64; 2]| Clippable {
                bank: bank(s.to_string(), dims),
                clip: t.clipped.then(|| Bounds {
                    in_lo: vec1(format!("{s}_in_lo"), 1),
                    in_hi: vec1(format!("{s}_in_hi"), 1),
                    out_lo: vec1(format!("{s}_out_lo"), 1),
                    out_hi: vec1(format!("{s}_out_hi"), 1),
                }),
            };
            Tower {
                hidden: t.hidden,
                heads: t.heads,
                head_dim,
                pool: t.pool,
                patch_width: t.patch_width,
                positions: 2 * t.positions,
                theta: t.theta,
                norm_eps: t.norm_eps,
                sm_scale: t.sm_scale,
                patch_embed: bank("patch_embed".into(), [th, u64::from(t.patch_width)]),
                pos_embed: bank("pos_embed".into(), [2 * u64::from(t.positions), th]),
                blocks: (0..t.depth)
                    .map(|l| {
                        let b = |s: &str| format!("block.{l}.{s}");
                        TowerBlock {
                            attn_norm: vec1(b("attn_norm"), th),
                            post_attn_norm: vec1(b("post_attn_norm"), th),
                            pre_ffw_norm: vec1(b("pre_ffw_norm"), th),
                            post_ffw_norm: vec1(b("post_ffw_norm"), th),
                            q: clip(&b("q"), [th, th]),
                            k: clip(&b("k"), [th, th]),
                            v: clip(&b("v"), [th, th]),
                            o: clip(&b("o"), [th, th]),
                            q_norm: vec1(b("q_norm"), u64::from(head_dim)),
                            k_norm: vec1(b("k_norm"), u64::from(head_dim)),
                            gate: clip(&b("gate"), [ti, th]),
                            up: clip(&b("up"), [ti, th]),
                            down: clip(&b("down"), [th, ti]),
                        }
                    })
                    .collect(),
                projection: Weight::sym(n("projection".into()), [hidden, th], w),
                std: t.standardize.then(|| Standardization {
                    bias: vec1("std_bias".into(), th),
                    scale: vec1("std_scale".into(), th),
                }),
            }
        });

        let draft = d.draft.then(|| {
            let hd = global.head_dim as u64;
            let q_w = q_heads as u64 * hd;
            let kv_w = global.kv_heads as u64 * hd;
            let iw = intermediate as u64;
            let n = |s: &str| format!("aux.{s}");
            let norm = |s: &str, len: u64| Weight::sym(n(s), [len], dense);
            Draft {
                fc_embed: Weight::sym(n("fc_embed"), [hidden, hidden], w),
                fc_hidden: Weight::sym(n("fc_hidden"), [hidden, hidden], w),
                attn_norm: norm("attn_norm", hidden),
                post_attn_norm: norm("post_attn_norm", hidden),
                pre_ffw_norm: norm("pre_ffw_norm", hidden),
                post_ffw_norm: norm("post_ffw_norm", hidden),
                attn: Attn {
                    sm_scale: d.sm_scale,
                    q_norm: norm("q_norm", hd),
                    q_norm_eps: d.norm_eps,
                    kv: "kv.mtp".to_string(),
                    banks: AttnBanks::Owned {
                        qkv: Weight::sym(n("qkv"), [q_w + 2 * kv_w, hidden], w)
                            .packed([q_w, kv_w, kv_w]),
                        k_norm: norm("k_norm", hd),
                        k_norm_eps: d.norm_eps,
                    },
                    reading: Reading::Global,
                },
                o_proj: Weight::sym(n("o_proj"), [hidden, q_w], w).rows(),
                gate_up: Weight::sym(n("gate_up"), [2 * iw, hidden], w).packed([iw, iw]),
                inter: intermediate,
                down: Weight::sym(n("down"), [hidden, iw], w).rows(),
                norm_eps: d.norm_eps,
            }
        });

        let assistant = d.assistant.then(|| {
            assert_eq!(tp, 1, "the assistant head is written for one rank");
            let last = |want_full: bool| {
                (0..d.layers)
                    .rev()
                    .find(|&l| !shared_at(l) && full_at(l) == want_full)
                    .map(owner)
                    .expect("the trunk has a layer of each reading")
            };
            let ah = ASSISTANT_HIDDEN as u64;
            let iw = ASSISTANT_INTER as u64;
            let n = |s: &str| format!("aux.{s}");
            let layers = ASSISTANT_READINGS
                .iter()
                .enumerate()
                .map(|(l, &reading)| {
                    let n = |s: &str| format!("aux.layer.{l}.{s}");
                    let norm = |s: &str, len: u64| Weight::sym(n(s), [len], dense);
                    let hd = match reading {
                        Reading::Sliding => sliding.head_dim,
                        Reading::Global => global.head_dim,
                    } as u64;
                    let q_w = q_heads as u64 * hd;
                    AssistantLayer {
                        attn: Attn {
                            sm_scale: d.sm_scale,
                            q_norm: norm("q_norm", hd),
                            q_norm_eps: d.norm_eps,
                            kv: format!("kv.{}", last(reading == Reading::Global)),
                            banks: AttnBanks::Shared {
                                q_proj: Weight::sym(n("q_proj.weight"), [q_w, ah], w),
                            },
                            reading,
                        },
                        o_proj: Weight::sym(n("o_proj.weight"), [ah, q_w], w),
                        attn_norm: norm("attn_norm", ah),
                        post_attn_norm: norm("post_attn_norm", ah),
                        pre_ffw_norm: norm("pre_ffw_norm", ah),
                        post_ffw_norm: norm("post_ffw_norm", ah),
                        gate_up: Weight::sym(n("gate_up.weight"), [2 * iw, ah], w).packed([iw, iw]),
                        inter: ASSISTANT_INTER,
                        down: Weight::sym(n("down.weight"), [ah, iw], w),
                        scalar: norm("scalar", 1),
                    }
                })
                .collect();
            Assistant {
                depth: ASSISTANT_DEPTH,
                pre_embed: Weight::sym(n("pre_embed.weight"), [ah, hidden], w),
                pre_hidden: Weight::sym(n("pre_hidden.weight"), [ah, hidden], w),
                post: Weight::sym(n("post.weight"), [hidden, ah], w),
                embed: Weight::sym(n("embed.weight"), [d.vocab as u64, ah], w),
                norm: Weight::sym(n("final_norm"), [ah], dense),
                norm_eps: d.norm_eps,
                layers,
            }
        });

        let banded = tp > 1 && !d.self_cond && std::env::var_os("PIE_NO_VOCAB_SHARD").is_none();
        let vocab_rows = if banded {
            (d.vocab / tp) as u64
        } else {
            d.vocab as u64
        };

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            q_heads,
            sliding,
            global,
            adapters: ADAPTERS,
            tower,
            kv,
            softcap: d.softcap,
            embed: {
                let table = Weight::sym(
                    "embed",
                    [vocab_rows, hidden],
                    if d.self_cond { dense } else { w },
                );
                if banded {
                    table.packed([vocab_rows])
                } else {
                    table
                }
            },
            ple: d.ple_dim.map(|dim| {
                let ple = dim as u64;
                Ple {
                    dim,
                    model_proj: Weight::sym("ple.model_proj", [d.layers as u64 * ple, hidden], w),
                    model_norm: Weight::sym("ple.model_norm", [ple], dense),
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
                            norm: Weight::sym(format!("layer.{l}.ple_norm"), [hidden], dense),
                            norm_eps: d.norm_eps,
                            scalar: Weight::sym(format!("layer.{l}.ple_scalar"), [1], dense),
                        })
                        .collect(),
                }
            }),
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            final_norm_eps: d.norm_eps,
            draft,
            assistant,
            self_cond: d.self_cond.then(|| {
                let iw = intermediate as u64;
                let sw = d.self_cond_w.unwrap_or(w);
                SelfCond {
                    taps: SELF_COND_TAPS,
                    pre_norm: Weight::sym("self_cond.pre_norm", [hidden], dense),
                    norm_eps: d.norm_eps,
                    gate_up: Weight::sym("self_cond.gate_up", [2 * iw, hidden], sw)
                        .packed([iw, iw]),
                    inter: intermediate,
                    down: Weight::sym("self_cond.down", [hidden, iw], sw).rows(),
                }
            }),
            dflash: d.dflash.map(|head| {
                DFlash::declare(
                    head,
                    "aux",
                    &dflash::Trunk {
                        hidden,
                        vocab: d.vocab as u64,
                        norm_eps: d.norm_eps,
                        weights: w,
                        dense,
                        tp,
                    },
                )
            }),
        }
    }
}

const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

impl Model {}
