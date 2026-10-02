use crate::drafter::dflash::{self, DFlash};
use model_dsl::{Dtype, Weight};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,

    pub adapters: Adapters,

    pub kv: Dtype,
    pub embed: Weight,
    pub head: Head,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,

    pub tower: Option<Tower>,

    pub mtp: Option<Mtp>,

    pub dflash: Option<DFlash>,
}

pub const DRAFT_DEPTH: u32 = 1;

pub struct Mtp {
    pub recipe: Recipe,
    pub pre_fc: Option<PreFc>,
    pub fc_embed: Weight,
    pub fc_hidden: Weight,
    pub mixer_norm: Weight,
    pub mixer_norm_eps: f32,
    pub attn: Attn,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: Mlp,
    pub norm: Option<Weight>,
    pub norm_eps: f32,
}

pub struct PreFc {
    pub embedding: Weight,
    pub hidden: Weight,
    pub eps: f32,
}

const SLIDING: Option<u32> = Some(2_048);

pub const QWEN36_27B_DFLASH: dflash::Head = dflash::Head {
    taps: &[1, 16, 31, 46, 61],
    windows: &[SLIDING, SLIDING, SLIDING, SLIDING, None],
    q_heads: 32,
    kv_heads: 8,
    head_dim: 128,
    inter: 17_408,
    theta: 10_000_000.0,
    block: 16,
    mask_token: 248_070,
    proposals_from: 1,
    conv: None,
    readout: dflash::Readout::Argmax,
    attn_bias: false,
};

pub const QWEN38_27B_DFLASH2: dflash::Head = dflash::Head {
    taps: &[5, 19, 33, 47, 61],
    windows: &[SLIDING; 5],
    q_heads: 32,
    kv_heads: 8,
    head_dim: 128,
    inter: 17_408,
    theta: 10_000_000.0,
    block: 8,
    mask_token: 248_070,
    proposals_from: 1,
    conv: Some(dflash::Conv { taps: 2, group: 16 }),
    readout: dflash::Readout::Selector {
        rank: 256,
        top_k: 16,
    },
    attn_bias: false,
};

pub const QWEN38_27B_DSPARK: dflash::Head = dflash::Head {
    taps: &[1, 16, 31, 46, 61],
    windows: &[None; 5],
    q_heads: 32,
    kv_heads: 8,
    head_dim: 128,
    inter: 17_408,
    theta: 10_000_000.0,
    block: 15,
    mask_token: 248_200,
    proposals_from: 0,
    conv: None,
    readout: dflash::Readout::Markov {
        rank: 256,
        top_k: 16,
    },
    attn_bias: false,
};

pub const QWEN36_35B_A3B_DFLASH: dflash::Head = dflash::Head {
    taps: &[1, 6, 11, 16, 22, 27, 32, 37],
    windows: &[
        Some(4_096),
        Some(4_096),
        Some(4_096),
        Some(4_096),
        Some(4_096),
        None,
    ],
    q_heads: 32,
    kv_heads: 8,
    head_dim: 128,
    inter: 6_144,
    theta: 10_000_000.0,
    block: 16,
    mask_token: 248_077,
    proposals_from: 1,
    conv: None,
    readout: dflash::Readout::Argmax,
    attn_bias: false,
};

pub const QWEN35_9B_DFLASH: dflash::Head = dflash::Head {
    taps: &[1, 5, 9, 13, 17, 21, 25, 29],
    windows: &[
        Some(4_096),
        Some(4_096),
        Some(4_096),
        Some(4_096),
        Some(4_096),
        None,
    ],
    q_heads: 32,
    kv_heads: 8,
    head_dim: 128,
    inter: 12_288,
    theta: 10_000_000.0,
    block: 16,
    mask_token: 248_077,
    proposals_from: 1,
    conv: None,
    readout: dflash::Readout::Argmax,
    attn_bias: false,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recipe {
    Mtp,
    Eagle,
    DFlash,
    DFlash2,
    DSpark,
}

impl Recipe {
    #[must_use]
    pub fn prefix(self) -> &'static str {
        match self {
            Recipe::Mtp => "mtp",
            Recipe::Eagle | Recipe::DFlash | Recipe::DFlash2 | Recipe::DSpark => "aux",
        }
    }

    #[must_use]
    pub fn drafts_a_block(self) -> bool {
        matches!(self, Recipe::DFlash | Recipe::DFlash2 | Recipe::DSpark)
    }
}

pub struct Tower {
    pub hidden: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub merge: u32,
    pub patch_width: u32,
    pub taps: u32,
    pub positions: u32,
    pub theta: f32,
    pub norm_eps: f32,
    pub sm_scale: f32,
    pub patch_embed: Weight,
    pub patch_embed_bias: Weight,
    pub pos_embed: Weight,
    pub blocks: Vec<TowerBlock>,
    pub merger: Merger,
}

pub struct TowerBlock {
    pub norm1: Weight,
    pub norm1_bias: Weight,
    pub qkv: Weight,
    pub qkv_bias: Weight,
    pub proj: Weight,
    pub proj_bias: Weight,
    pub norm2: Weight,
    pub norm2_bias: Weight,
    pub fc1: Weight,
    pub fc1_bias: Weight,
    pub fc2: Weight,
    pub fc2_bias: Weight,
}

pub struct Merger {
    pub norm: Weight,
    pub norm_bias: Weight,
    pub fc1: Weight,
    pub fc1_bias: Weight,
    pub fc2: Weight,
    pub fc2_bias: Weight,
}

pub enum Head {
    Tied,
    Bank(Weight),
}

pub use crate::adapter::Adapters;

pub struct Layer {
    pub mixer: Mixer,
    pub mixer_norm: Weight,
    pub mixer_norm_eps: f32,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: Mlp,
    pub lora_a: Weight,
    pub lora_b: Weight,
}

pub enum Mixer {
    Attn(Attn),
    Gdn(Gdn),
}

pub struct Attn {
    pub rotary_dim: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub qg_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    pub o_proj: Weight,
    pub q_norm: Weight,
    pub q_norm_eps: f32,
    pub k_norm: Weight,
    pub k_norm_eps: f32,
    pub kv: String,
}

pub struct Gdn {
    pub k_heads: u32,
    pub v_heads: u32,
    pub k_dim: u32,
    pub v_dim: u32,
    pub conv_kernel: u32,
    pub in_qkvz: Weight,
    pub in_ba: Weight,
    pub conv: Weight,
    pub dt_bias: Weight,
    pub a_log: Weight,
    pub norm: Weight,
    pub norm_eps: f32,
    pub out_proj: Weight,
    pub conv_state: String,
    pub delta_state: String,
}

impl Gdn {
    #[must_use]
    pub fn qkv_width(k_heads: u32, v_heads: u32, k_dim: u32, v_dim: u32) -> u32 {
        2 * k_heads * k_dim + v_heads * v_dim
    }
}

#[allow(clippy::large_enum_variant)]
pub enum Mlp {
    Dense {
        gate_up: Weight,
        down: Weight,
        inter: u32,
    },
    Routed {
        router: Weight,
        gate_up: Weight,
        down: Weight,
        shared_gate_up: Weight,
        shared_down: Weight,
        shared_gate: Weight,
        experts: u32,
        top_k: u32,
        inter: u32,
        shared_inter: u32,
    },
}

struct MoeDims {
    experts: u32,
    top_k: u32,
    inter: u32,
    shared_inter: u32,
}

enum MlpDims {
    Dense { inter: u32 },
    Routed(MoeDims),
}

#[derive(Clone, Copy)]
struct TowerDims {
    depth: u32,
    hidden: u32,
    heads: u32,
    inter: u32,
    patch_width: u32,
    merge: u32,
    positions: u32,
    out_hidden: u32,
    theta: f32,
    norm_eps: f32,
    taps: u32,
}

impl TowerDims {
    const fn qwen35() -> TowerDims {
        TowerDims {
            depth: 12,
            hidden: 768,
            heads: 12,
            inter: 3072,
            patch_width: 1536,
            merge: 2,
            positions: 2304,
            out_hidden: 1024,
            theta: 10_000.0,
            norm_eps: 1e-6,
            taps: 4,
        }
    }

    const fn qwen36() -> TowerDims {
        TowerDims {
            depth: 27,
            hidden: 1152,
            heads: 16,
            inter: 4304,
            patch_width: 1536,
            merge: 2,
            positions: 2304,
            out_hidden: 5120,
            theta: 10_000.0,
            norm_eps: 1e-6,
            taps: 4,
        }
    }
}

struct Dims {
    hidden: u32,
    layers: u32,
    attn_every: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    conv_kernel: u32,
    mlp: MlpDims,
    vocab: u32,
    tied: bool,
    norm_eps: f32,
    tower: Option<TowerDims>,
    draft: Option<Recipe>,
    dflash_head: Option<&'static dflash::Head>,
}

impl Model {
    pub fn a3b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::a3b_dims())
    }

    pub fn a3b_mtp(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::a3b_dims();
        d.draft = Some(Recipe::Mtp);
        Model::new(w, kv, tp, d)
    }

    pub fn a3b_dflash(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::a3b_dims();
        d.draft = Some(Recipe::DFlash);
        d.dflash_head = Some(&QWEN36_35B_A3B_DFLASH);
        Model::new(w, kv, tp, d)
    }

    fn a3b_dims() -> Dims {
        Dims {
            hidden: 2048,
            layers: 40,
            attn_every: 4,
            q_heads: 16,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            theta: 10_000_000.0,
            k_heads: 16,
            v_heads: 32,
            k_dim: 128,
            v_dim: 128,
            conv_kernel: 4,
            mlp: MlpDims::Routed(MoeDims {
                experts: 256,
                top_k: 8,
                inter: 512,
                shared_inter: 512,
            }),
            vocab: 248_320,
            tied: false,
            norm_eps: 1e-6,
            tower: None,
            draft: None,
            dflash_head: None,
        }
    }

    pub fn a3b_mini(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::a3b_mini_dims(16))
    }

    pub fn a3b_mini64(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::a3b_mini_dims(64))
    }

    fn a3b_mini_dims(experts: u32) -> Dims {
        let mut d = Model::a3b_dims();
        d.layers = 5;
        let MlpDims::Routed(moe) = &mut d.mlp else {
            unreachable!("the a3b dims carry a routed MLP")
        };
        moe.experts = experts;
        d
    }

    pub fn a3b_micro(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Dims {
                hidden: 512,
                layers: 4,
                attn_every: 1,
                q_heads: 8,
                kv_heads: 2,
                head_dim: 64,
                rotary_dim: 64,
                theta: 10_000_000.0,
                k_heads: 8,
                v_heads: 8,
                k_dim: 64,
                v_dim: 64,
                conv_kernel: 4,
                mlp: MlpDims::Routed(MoeDims {
                    experts: 32,
                    top_k: 4,
                    inter: 128,
                    shared_inter: 128,
                }),
                vocab: 2048,
                tied: true,
                norm_eps: 1e-6,
                tower: None,
                draft: None,
                dflash_head: None,
            },
        )
    }

    pub fn a3b_uncached_bank(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Dims {
                hidden: 2048,
                layers: 1,
                attn_every: 1,
                q_heads: 8,
                kv_heads: 2,
                head_dim: 64,
                rotary_dim: 64,
                theta: 10_000_000.0,
                k_heads: 8,
                v_heads: 8,
                k_dim: 64,
                v_dim: 64,
                conv_kernel: 4,
                mlp: MlpDims::Routed(MoeDims {
                    experts: 32,
                    top_k: 4,
                    inter: 512,
                    shared_inter: 512,
                }),
                vocab: 2048,
                tied: true,
                norm_eps: 1e-6,
                tower: None,
                draft: None,
                dflash_head: None,
            },
        )
    }

    pub fn d0_8b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d0_8b_dims(None, None))
    }

    pub fn d0_8b_eagle(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d0_8b_dims(None, Some(Recipe::Eagle)))
    }

    pub fn d0_8b_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Model::d0_8b_dims(Some(TowerDims::qwen35()), None),
        )
    }

    pub fn d0_8b_vision_eagle(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Model::d0_8b_dims(Some(TowerDims::qwen35()), Some(Recipe::Eagle)),
        )
    }

    fn d0_8b_dims(tower: Option<TowerDims>, draft: Option<Recipe>) -> Dims {
        Dims {
            hidden: 1024,
            layers: 24,
            attn_every: 4,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            theta: 10_000_000.0,
            k_heads: 16,
            v_heads: 16,
            k_dim: 128,
            v_dim: 128,
            conv_kernel: 4,
            mlp: MlpDims::Dense { inter: 3584 },
            vocab: 248_320,
            tied: true,
            norm_eps: 1e-6,
            tower,
            draft,
            dflash_head: None,
        }
    }

    pub fn tiny(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Dims {
                hidden: 256,
                layers: 4,
                attn_every: 4,
                q_heads: 4,
                kv_heads: 2,
                head_dim: 64,
                rotary_dim: 32,
                theta: 10_000_000.0,
                k_heads: 4,
                v_heads: 8,
                k_dim: 128,
                v_dim: 128,
                conv_kernel: 4,
                mlp: MlpDims::Dense { inter: 512 },
                vocab: 248_320,
                tied: true,
                norm_eps: 1e-6,
                tower: None,
                draft: None,
                dflash_head: None,
            },
        )
    }

    pub fn d3b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Dims {
                hidden: 2048,
                layers: 24,
                attn_every: 4,
                q_heads: 16,
                kv_heads: 2,
                head_dim: 256,
                rotary_dim: 64,
                theta: 10_000_000.0,
                k_heads: 16,
                v_heads: 32,
                k_dim: 128,
                v_dim: 128,
                conv_kernel: 4,
                mlp: MlpDims::Dense { inter: 8192 },
                vocab: 151_936,
                tied: true,
                norm_eps: 1e-6,
                tower: None,
                draft: None,
                dflash_head: None,
            },
        )
    }

    pub fn d2b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Dims {
                hidden: 2048,
                layers: 24,
                attn_every: 4,
                q_heads: 8,
                kv_heads: 2,
                head_dim: 256,
                rotary_dim: 64,
                theta: 10_000_000.0,
                k_heads: 16,
                v_heads: 16,
                k_dim: 128,
                v_dim: 128,
                conv_kernel: 4,
                mlp: MlpDims::Dense { inter: 6144 },
                vocab: 248_320,
                tied: true,
                norm_eps: 1e-6,
                tower: None,
                draft: None,
                dflash_head: None,
            },
        )
    }

    pub fn d4b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Dims {
                hidden: 2560,
                layers: 32,
                attn_every: 4,
                q_heads: 16,
                kv_heads: 4,
                head_dim: 256,
                rotary_dim: 64,
                theta: 10_000_000.0,
                k_heads: 16,
                v_heads: 32,
                k_dim: 128,
                v_dim: 128,
                conv_kernel: 4,
                mlp: MlpDims::Dense { inter: 9216 },
                vocab: 248_320,
                tied: true,
                norm_eps: 1e-6,
                tower: None,
                draft: None,
                dflash_head: None,
            },
        )
    }

    pub fn d9b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d9b_dims())
    }

    pub fn d9b_dflash(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::d9b_dims();
        d.draft = Some(Recipe::DFlash);
        d.dflash_head = Some(&QWEN35_9B_DFLASH);
        Model::new(w, kv, tp, d)
    }

    fn d9b_dims() -> Dims {
        {
            Dims {
                hidden: 4096,
                layers: 32,
                attn_every: 4,
                q_heads: 16,
                kv_heads: 4,
                head_dim: 256,
                rotary_dim: 64,
                theta: 10_000_000.0,
                k_heads: 16,
                v_heads: 32,
                k_dim: 128,
                v_dim: 128,
                conv_kernel: 4,
                mlp: MlpDims::Dense { inter: 12288 },
                vocab: 248_320,
                tied: false,
                norm_eps: 1e-6,
                tower: None,
                draft: None,
                dflash_head: None,
            }
        }
    }

    pub fn d27b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d27b_dims(None, Some(Recipe::Mtp)))
    }

    pub fn d27b_dflash(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d27b_dims(None, Some(Recipe::DFlash)))
    }

    pub fn d27b_dflash2(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d27b_dims(None, Some(Recipe::DFlash2)))
    }

    pub fn d27b_dspark(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d27b_dims(None, Some(Recipe::DSpark)))
    }

    pub fn d27b_undrafted(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d27b_dims(None, None))
    }

    pub fn d27b_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Model::d27b_dims(Some(TowerDims::qwen36()), Some(Recipe::Mtp)),
        )
    }

    pub fn d27b_vision_undrafted(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d27b_dims(Some(TowerDims::qwen36()), None))
    }

    fn d27b_dims(tower: Option<TowerDims>, draft: Option<Recipe>) -> Dims {
        Dims {
            hidden: 5120,
            layers: 64,
            attn_every: 4,
            q_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
            theta: 10_000_000.0,
            k_heads: 16,
            v_heads: 48,
            k_dim: 128,
            v_dim: 128,
            conv_kernel: 4,
            mlp: MlpDims::Dense { inter: 17_408 },
            vocab: 248_320,
            tied: false,
            norm_eps: 1e-6,
            tower,
            draft,
            dflash_head: draft.and_then(|r| match r {
                Recipe::DFlash => Some(&QWEN36_27B_DFLASH),
                Recipe::DFlash2 => Some(&QWEN38_27B_DFLASH2),
                Recipe::DSpark => Some(&QWEN38_27B_DSPARK),
                Recipe::Mtp | Recipe::Eagle => None,
            }),
        }
    }

    fn new(w: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        let dense = crate::dense(w);
        let gate = match w {
            Dtype::U4g64 => Dtype::U8g64,
            other => other,
        };
        let proj = match w {
            Dtype::U4g64 => Dtype::U4g64tiled,
            other => other,
        };
        let q_heads = d.q_heads / tp;
        let kv_heads = d.kv_heads / tp;
        let k_heads = d.k_heads / tp;
        let v_heads = d.v_heads / tp;
        let hidden = d.hidden as u64;
        let attn_at = |l: u32| l % d.attn_every == d.attn_every - 1;

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str, dim: u64| Weight::sym(n(s), [dim], dense);
                let (lora_a, lora_b) =
                    crate::adapter::banks(&format!("layer.{l}"), ADAPTERS, hidden, dense);
                let mixer = if attn_at(l) {
                    Mixer::Attn(gated_attn(
                        proj,
                        &d,
                        q_heads,
                        kv_heads,
                        &format!("layer.{l}"),
                        format!("kv.{l}"),
                    ))
                } else {
                    let k_w = k_heads as u64 * d.k_dim as u64;
                    let v_w = v_heads as u64 * d.v_dim as u64;
                    let qkv = u64::from(Gdn::qkv_width(k_heads, v_heads, d.k_dim, d.v_dim));
                    let qkvz = qkv + v_w;
                    Mixer::Gdn(Gdn {
                        k_heads,
                        v_heads,
                        k_dim: d.k_dim,
                        v_dim: d.v_dim,
                        conv_kernel: d.conv_kernel,
                        in_qkvz: Weight::sym(n("in_qkvz"), [qkvz, hidden], proj)
                            .packed([k_w, k_w, v_w, v_w]),
                        in_ba: Weight::sym(n("in_ba"), [2 * v_heads as u64, hidden], proj)
                            .packed([v_heads as u64, v_heads as u64]),
                        conv: Weight::sym(n("conv"), [qkv, d.conv_kernel as u64], dense)
                            .packed([k_w, k_w, v_w]),
                        dt_bias: Weight::sym(n("dt_bias"), [v_heads as u64], dense).columns(),
                        a_log: Weight::sym(n("a_log"), [v_heads as u64], Dtype::F32).columns(),
                        norm: Weight::sym(n("gdn_norm"), [d.v_dim as u64], Dtype::F32),
                        norm_eps: d.norm_eps,
                        out_proj: Weight::sym(n("out_proj"), [hidden, v_w], proj).rows(),
                        conv_state: format!("conv.{l}"),
                        delta_state: format!("delta.{l}"),
                    })
                };
                let mlp = match &d.mlp {
                    MlpDims::Dense { inter } => {
                        dense_mlp(proj, hidden, inter / tp, &format!("layer.{l}"))
                    }
                    MlpDims::Routed(m) => {
                        let inter = m.inter / tp;
                        let shared_inter = m.shared_inter / tp;
                        Mlp::Routed {
                            router: Weight::sym(n("router"), [m.experts as u64, hidden], gate),

                            gate_up: Weight::sym(
                                n("experts_gate_up"),
                                [m.experts as u64, 2 * inter as u64, hidden],
                                w,
                            )
                            .bank([inter as u64, inter as u64]),
                            down: Weight::sym(
                                n("experts_down"),
                                [m.experts as u64, hidden, inter as u64],
                                w,
                            )
                            .rows(),
                            shared_gate_up: Weight::sym(
                                n("shared_gate_up"),
                                [2 * shared_inter as u64, hidden],
                                proj,
                            )
                            .packed([shared_inter as u64, shared_inter as u64]),
                            shared_down: Weight::sym(
                                n("shared_down"),
                                [hidden, shared_inter as u64],
                                proj,
                            )
                            .rows(),
                            shared_gate: Weight::sym(n("shared_gate"), [1, hidden], gate),
                            experts: m.experts,
                            top_k: m.top_k,
                            inter,
                            shared_inter,
                        }
                    }
                };
                Layer {
                    mixer,
                    mixer_norm: norm("mixer_norm", hidden),
                    mixer_norm_eps: d.norm_eps,
                    mlp_norm: norm("mlp_norm", hidden),
                    mlp_norm_eps: d.norm_eps,
                    mlp,
                    lora_a,
                    lora_b,
                }
            })
            .collect();

        let tower = d.tower.map(|t| {
            assert_eq!(
                t.out_hidden, d.hidden,
                "a tower's `out_hidden_size` is the TRUNK's width — the merger's \
                 answer is a token row, and a mismatch would scatter a rectangle \
                 of the wrong width into the embedding"
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
            let merged = u64::from(t.merge) * u64::from(t.merge) * th;
            let head_dim = t.hidden / t.heads;
            let n = |s: String| format!("visual.{s}");
            let plane = |s: String, dims: [u64; 2]| Weight::sym(n(s), dims, dense);
            let vec1 = |s: String, len: u64| Weight::sym(n(s), [len], dense);
            Tower {
                hidden: t.hidden,
                heads: t.heads,
                head_dim,
                merge: t.merge,
                patch_width: t.patch_width,
                taps: t.taps,
                positions: t.positions,
                theta: t.theta,
                norm_eps: t.norm_eps,
                sm_scale: (head_dim as f32).sqrt().recip(),
                patch_embed: plane("patch_embed".into(), [th, u64::from(t.patch_width)]),
                patch_embed_bias: vec1("patch_embed_bias".into(), th),
                pos_embed: plane("pos_embed".into(), [u64::from(t.positions), th]),
                blocks: (0..t.depth)
                    .map(|l| {
                        let b = |s: &str| format!("block.{l}.{s}");
                        TowerBlock {
                            norm1: vec1(b("norm1"), th),
                            norm1_bias: vec1(b("norm1_bias"), th),
                            qkv: plane(b("qkv"), [3 * th, th]),
                            qkv_bias: vec1(b("qkv_bias"), 3 * th),
                            proj: plane(b("proj"), [th, th]),
                            proj_bias: vec1(b("proj_bias"), th),
                            norm2: vec1(b("norm2"), th),
                            norm2_bias: vec1(b("norm2_bias"), th),
                            fc1: plane(b("fc1"), [ti, th]),
                            fc1_bias: vec1(b("fc1_bias"), ti),
                            fc2: plane(b("fc2"), [th, ti]),
                            fc2_bias: vec1(b("fc2_bias"), th),
                        }
                    })
                    .collect(),
                merger: Merger {
                    norm: vec1("merger_norm".into(), th),
                    norm_bias: vec1("merger_norm_bias".into(), th),
                    fc1: plane("merger_fc1".into(), [merged, merged]),
                    fc1_bias: vec1("merger_fc1_bias".into(), merged),
                    fc2: plane("merger_fc2".into(), [hidden, merged]),
                    fc2_bias: vec1("merger_fc2_bias".into(), hidden),
                },
            }
        });

        let mtp = d.draft.filter(|r| !r.drafts_a_block()).map(|recipe| {
            let inter = match &d.mlp {
                MlpDims::Dense { inter } => *inter,
                MlpDims::Routed(m) => m.inter,
            } / tp;
            let p = recipe.prefix();
            let n = |s: &str| format!("{p}.{s}");
            Mtp {
                recipe,
                pre_fc: matches!(recipe, Recipe::Mtp).then(|| PreFc {
                    embedding: Weight::sym(n("pre_fc_norm_embedding"), [hidden], dense),
                    hidden: Weight::sym(n("pre_fc_norm_hidden"), [hidden], dense),
                    eps: d.norm_eps,
                }),
                fc_embed: Weight::sym(n("fc_embed"), [hidden, hidden], w),
                fc_hidden: Weight::sym(n("fc_hidden"), [hidden, hidden], w),
                mixer_norm: Weight::sym(n("mixer_norm"), [hidden], dense),
                mixer_norm_eps: d.norm_eps,
                attn: gated_attn(w, &d, q_heads, kv_heads, p, "kv.mtp".to_string()),
                mlp_norm: Weight::sym(n("mlp_norm"), [hidden], dense),
                mlp_norm_eps: d.norm_eps,
                mlp: dense_mlp(w, hidden, inter, p),
                norm: matches!(recipe, Recipe::Mtp)
                    .then(|| Weight::sym(n("norm"), [hidden], dense)),
                norm_eps: d.norm_eps,
            }
        });

        let dflash = d.draft.filter(|r| r.drafts_a_block()).map(|recipe| {
            let head = d
                .dflash_head
                .expect("a block-drafting recipe names its published head");
            DFlash::declare(
                head,
                recipe.prefix(),
                &dflash::Trunk {
                    hidden,
                    vocab: d.vocab as u64,
                    norm_eps: d.norm_eps,
                    weights: w,
                    dense,
                    tp,
                },
            )
        });

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            q_heads,
            kv_heads,
            head_dim: d.head_dim,
            adapters: ADAPTERS,
            kv,
            embed: Weight::sym("embed", [d.vocab as u64, hidden], w),
            head: if d.tied {
                Head::Tied
            } else {
                let banded = tp > 1 && std::env::var_os("PIE_NO_VOCAB_SHARD").is_none();
                let rows = if banded {
                    (d.vocab / tp) as u64
                } else {
                    d.vocab as u64
                };
                let bank = Weight::sym("lm_head", [rows, hidden], w);
                Head::Bank(if banded { bank.packed([rows]) } else { bank })
            },
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            final_norm_eps: d.norm_eps,
            tower,
            mtp,
            dflash,
        }
    }
}

const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

fn gated_attn(w: Dtype, d: &Dims, q_heads: u32, kv_heads: u32, prefix: &str, kv: String) -> Attn {
    let n = |s: &str| format!("{prefix}.{s}");
    let dense = crate::dense(w);
    let hidden = d.hidden as u64;
    let hd = d.head_dim as u64;
    Attn {
        rotary_dim: d.rotary_dim,
        theta: d.theta,
        sm_scale: (d.head_dim as f32).sqrt().recip(),
        qg_proj: Weight::sym(n("qg_proj"), [2 * q_heads as u64 * hd, hidden], w).columns(),
        k_proj: Weight::sym(n("k_proj"), [kv_heads as u64 * hd, hidden], w).columns(),
        v_proj: Weight::sym(n("v_proj"), [kv_heads as u64 * hd, hidden], w).columns(),
        o_proj: Weight::sym(n("o_proj"), [hidden, q_heads as u64 * hd], w).rows(),
        q_norm: Weight::sym(n("q_norm"), [hd], dense),
        q_norm_eps: d.norm_eps,
        k_norm: Weight::sym(n("k_norm"), [hd], dense),
        k_norm_eps: d.norm_eps,
        kv,
    }
}

fn dense_mlp(w: Dtype, hidden: u64, inter: u32, prefix: &str) -> Mlp {
    let n = |s: &str| format!("{prefix}.{s}");
    Mlp::Dense {
        gate_up: Weight::sym(n("gate_up"), [2 * inter as u64, hidden], w)
            .packed([inter as u64, inter as u64]),
        down: Weight::sym(n("down"), [hidden, inter as u64], w).rows(),
        inter,
    }
}

impl Model {}
