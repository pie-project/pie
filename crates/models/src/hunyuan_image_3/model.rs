use poem::{Dtype, Weight};

pub const TRAIN_STEPS: u32 = 1000;
pub const FLOW_SHIFT: f32 = 3.0;

pub const NORM_EPS: f32 = 1e-5;

pub const ROPE_THETA: f32 = 10_000.0;
pub const ROPE_AXES: u8 = 2;

#[must_use]
pub fn rope_x_scale(head_dim: u32) -> f32 {
    ROPE_THETA.powf(-2.0 / head_dim as f32)
}

pub const T_FREQ_DIM: u32 = 256;
pub const T_MAX_PERIOD: f32 = 10_000.0;
pub const T_FLIP_SIN_COS: bool = true;
pub const T_SCALE: f32 = 1.0;

pub const GN_GROUPS: u32 = 32;
pub const GN_EPS: f32 = 1e-5;

pub const PATCH: u32 = 1;
pub const LATENT_CHANNELS: u32 = 32;
pub const SPATIAL_COMPRESSION: u32 = 16;
pub const VAE_SCALING: f32 = 0.562_679_2;

pub const CONV3: [u32; 3] = [1, 3, 3];
pub const CONV1: [u32; 3] = [1, 1, 1];

pub mod port {
    pub const ROWS: u8 = 0;
    pub const SPECIAL: u8 = 1;
    pub const TIMESTEP: u8 = 0;
    pub const POSITIONS: u8 = 0;
    pub const LATENT_VOXELS: u8 = 0;
    pub const ROW_VOXELS: u8 = 1;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub experts: u32,
    pub top_k: u32,
    pub moe_inter: u32,
    pub shared_inter: u32,
    pub head_hidden: u32,
}

impl Dims {
    #[must_use]
    pub const fn flagship() -> Dims {
        Dims {
            hidden: 4096,
            layers: 32,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            vocab: 133_120,
            experts: 64,
            top_k: 8,
            moe_inter: 3072,
            shared_inter: 3072,
            head_hidden: 1024,
        }
    }

    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            hidden: 256,
            layers: 2,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 64,
            vocab: 133_120,
            experts: 8,
            top_k: 2,
            moe_inter: 256,
            shared_inter: 256,
            head_hidden: 64,
        }
    }

    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        [self.head_dim / 2, self.head_dim / 2, 0, 0]
    }
}

pub struct Linear {
    pub w: Weight,
    pub bias: Weight,
}

impl Linear {
    fn at(name: &str, out: u32, in_: u32, banks: Dtype) -> Linear {
        Linear {
            w: Weight::sym(name, [u64::from(out), u64::from(in_)], banks),
            bias: Weight::sym(
                format!("{name}.bias"),
                [u64::from(out)],
                crate::dense(banks),
            ),
        }
    }
}

pub struct Conv {
    pub w: Weight,
    pub bias: Weight,
    pub k: [u32; 3],
}

impl Conv {
    fn at(name: &str, c_out: u32, c_in: u32, k: [u32; 3], banks: Dtype) -> Conv {
        let taps = k[0] * k[1] * k[2];
        Conv {
            w: Weight::sym(
                name,
                [u64::from(c_out), u64::from(c_in) * u64::from(taps)],
                banks,
            )
            .conv_taps_major(c_in, taps),
            bias: Weight::sym(format!("{name}.bias"), [u64::from(c_out)], Dtype::F32),
            k,
        }
    }
}

pub struct GroupNorm {
    pub weight: Weight,
    pub bias: Weight,
}

impl GroupNorm {
    fn at(name: &str, c: u32) -> GroupNorm {
        GroupNorm {
            weight: Weight::sym(name, [u64::from(c)], Dtype::F32),
            bias: Weight::sym(format!("{name}.bias"), [u64::from(c)], Dtype::F32),
        }
    }
}

pub struct ResBlock {
    pub norm_in: GroupNorm,
    pub conv_in: Conv,
    pub emb: Linear,
    pub norm_out: GroupNorm,
    pub conv_out: Conv,
    pub skip: Option<Conv>,
}

impl ResBlock {
    fn at(prefix: &str, c_in: u32, c_out: u32, emb: u32, banks: Dtype) -> ResBlock {
        let n = |s: &str| format!("{prefix}.{s}");
        ResBlock {
            norm_in: GroupNorm::at(&n("norm_in"), c_in),
            conv_in: Conv::at(&n("conv_in"), c_out, c_in, CONV3, banks),
            emb: Linear::at(&n("emb"), 2 * c_out, emb, banks),
            norm_out: GroupNorm::at(&n("norm_out"), c_out),
            conv_out: Conv::at(&n("conv_out"), c_out, c_out, CONV3, banks),
            skip: (c_in != c_out).then(|| Conv::at(&n("skip"), c_out, c_in, CONV1, banks)),
        }
    }
}

pub struct UNetDown {
    pub conv_in: Conv,
    pub res: ResBlock,
}

pub struct UNetUp {
    pub res: ResBlock,
    pub norm_out: GroupNorm,
    pub conv_out: Conv,
}

pub struct Embedder {
    pub mlp_in: Linear,
    pub mlp_out: Linear,
}

impl Embedder {
    fn at(prefix: &str, hidden: u32, out: u32, banks: Dtype) -> Embedder {
        Embedder {
            mlp_in: Linear::at(&format!("{prefix}.in"), hidden, T_FREQ_DIM, banks),
            mlp_out: Linear::at(&format!("{prefix}.out"), out, hidden, banks),
        }
    }
}

pub struct Layer {
    pub attn_norm: Weight,
    pub qkv: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub o_proj: Weight,
    pub kv: String,
    pub mlp_norm: Weight,
    pub router: Weight,
    pub experts_gate_up: Weight,
    pub experts_down: Weight,
    pub shared_gate_up: Weight,
    pub shared_down: Weight,
}

pub struct Model {
    pub banks: Dtype,
    pub expert_banks: Dtype,
    pub kv_dtype: Dtype,
    pub dims: Dims,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub moe_inter: u32,
    pub shared_inter: u32,
    pub embed: Weight,
    pub head: Weight,
    pub final_norm: Weight,
    pub layers: Vec<Layer>,
    pub timestep_emb: Embedder,
    pub time_embed: Embedder,
    pub time_embed_2: Embedder,
    pub patch_embed: UNetDown,
    pub final_layer: UNetUp,
    pub ones: Weight,
}

impl Model {
    #[must_use]
    pub fn flagship(banks: Dtype, experts: Dtype, kv: Dtype) -> Model {
        Model::new(banks, experts, kv, Dims::flagship())
    }

    #[must_use]
    pub fn mini(banks: Dtype, kv: Dtype) -> Model {
        Model::new(banks, banks, kv, Dims::mini())
    }

    fn new(banks: Dtype, expert_banks: Dtype, kv: Dtype, d: Dims) -> Model {
        assert!(
            d.head_dim.is_multiple_of(4),
            "the 2-D rope splits a head into two even blocks; {} is not a multiple of 4",
            d.head_dim
        );
        assert!(
            d.q_heads.is_multiple_of(d.kv_heads),
            "GQA: {} query heads do not group into {} kv heads",
            d.q_heads,
            d.kv_heads
        );
        assert!(
            d.head_hidden.is_multiple_of(GN_GROUPS) && d.hidden.is_multiple_of(GN_GROUPS),
            "the image head group-norms {GN_GROUPS} ways"
        );
        let q_heads = d.q_heads;
        let kv_heads = d.kv_heads;
        let moe_inter = d.moe_inter;
        let shared_inter = d.shared_inter;

        let dense = crate::dense(banks);
        let hidden = u64::from(d.hidden);
        let hd = u64::from(d.head_dim);
        let q_w = u64::from(q_heads) * hd;
        let kv_w = u64::from(kv_heads) * hd;
        let iw = u64::from(moe_inter);
        let sw = u64::from(shared_inter);
        let n_experts = u64::from(d.experts);

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                Layer {
                    attn_norm: Weight::sym(n("attn_norm"), [hidden], dense),
                    qkv: Weight::sym(n("qkv"), [q_w + 2 * kv_w, hidden], banks)
                        .packed([q_w, kv_w, kv_w]),
                    q_norm: Weight::sym(n("q_norm"), [hd], dense),
                    k_norm: Weight::sym(n("k_norm"), [hd], dense),
                    o_proj: Weight::sym(n("o_proj"), [hidden, q_w], banks).rows(),
                    kv: format!("kv.{l}"),
                    mlp_norm: Weight::sym(n("mlp_norm"), [hidden], dense),
                    router: Weight::sym(n("router"), [n_experts, hidden], dense),
                    experts_gate_up: Weight::sym(
                        n("experts_gate_up"),
                        [n_experts, 2 * iw, hidden],
                        expert_banks,
                    )
                    .bank([iw, iw]),
                    experts_down: Weight::sym(
                        n("experts_down"),
                        [n_experts, hidden, iw],
                        expert_banks,
                    )
                    .rows(),
                    shared_gate_up: Weight::sym(n("shared_gate_up"), [2 * sw, hidden], banks)
                        .packed([sw, sw]),
                    shared_down: Weight::sym(n("shared_down"), [hidden, sw], banks).rows(),
                }
            })
            .collect();

        let hw = d.head_hidden;
        Model {
            banks,
            expert_banks,
            kv_dtype: kv,
            dims: d,
            q_heads,
            kv_heads,
            moe_inter,
            shared_inter,
            embed: Weight::sym("wte", [u64::from(d.vocab), hidden], banks),
            head: Weight::sym("lm_head", [u64::from(d.vocab), hidden], banks),
            final_norm: Weight::sym("ln_f", [hidden], dense),
            layers,
            timestep_emb: Embedder {
                mlp_in: Linear::at("timestep_emb.in", d.hidden, T_FREQ_DIM, banks),
                mlp_out: Linear::at("timestep_emb.out", 2 * d.hidden, d.hidden, banks),
            },
            time_embed: Embedder::at("time_embed", d.hidden, d.hidden, banks),
            time_embed_2: Embedder::at("time_embed_2", d.hidden, d.hidden, banks),
            patch_embed: UNetDown {
                conv_in: Conv::at("patch_embed.conv", hw, LATENT_CHANNELS, CONV3, banks),
                res: ResBlock::at("patch_embed.res", hw, d.hidden, d.hidden, banks),
            },
            final_layer: UNetUp {
                res: ResBlock::at("final_layer.res", d.hidden, hw, d.hidden, banks),
                norm_out: GroupNorm::at("final_layer.norm_out", hw),
                conv_out: Conv::at("final_layer.conv", LATENT_CHANNELS, hw, CONV3, banks),
            },
            ones: Weight::sym("special.ones", [2 * hidden, 1], banks),
        }
    }

    #[must_use]
    pub fn q_width(&self) -> u32 {
        self.q_heads * self.dims.head_dim
    }

    #[must_use]
    pub fn kv_width(&self) -> u32 {
        self.kv_heads * self.dims.head_dim
    }
}
