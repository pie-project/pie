use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerAttention {

    pub head_dim: u32,

    pub window: i32,

    pub kv_source: u32,

    pub sm_scale: f32,

    pub rope_theta: f32,

    pub rotary_dim: u32,

    pub kv_heads: u32,

    pub q_gate: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MlpGate {

    Silu,

    GeluTanh,

    SiluClamped {

        limit: f32,

        alpha: f32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvStyle {

    Paged,

    Mla {

        kv_lora_rank: u32,

        qk_rope_head_dim: u32,
    },

    CompressedPlane {

        ratios: Vec<i32>,
    },
}

impl KvStyle {

    #[must_use]
    pub fn has_a_store_in_this_build(&self) -> bool {
        match self {
            Self::Paged => true,
            Self::Mla { .. } | Self::CompressedPlane { .. } => false,
        }
    }

    #[must_use]
    pub fn store_refusal(&self) -> Option<Refusal> {
        match self {
            Self::Paged => None,
            Self::Mla { .. } => Some(Refusal::Unsupported(
                "this build provisions no MLA latent store; a compressed KV \
                 plane and a positional one do not fit the k/v pair the pager \
                 allocates",
            )),
            Self::CompressedPlane { .. } => Some(Refusal::Unsupported(
                "this build provisions no compressed KV plane store; the row's \
                 per-layer compressed entries have nowhere to live",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentShape {

    pub linear_layers: Vec<u32>,

    pub conv_stride: usize,

    pub state_stride: usize,

    pub state_elem: usize,

    pub k_h: i32,

    pub v_h: i32,

    pub k_d: i32,

    pub v_d: i32,

    pub conv_dim: i32,

    pub conv_k: i32,

    pub n_groups: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillStyle {

    Planned,

    Planless,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormPlacement {

    Pre,

    Post,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnOutput {

    DriverPinned,

    StatedArgs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Geometry {

    pub hidden: u32,

    pub q_heads: u32,

    pub kv_heads: u32,

    pub head_dim: u32,

    pub head_dim_kernel: u32,

    pub intermediate: u32,

    pub moe_intermediate: u32,

    pub experts_per_token: u32,

    pub shared_intermediate: u32,

    pub vocab: u32,
}

pub const ATTN_HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

#[must_use]
pub fn round_up_attn_head_dim(head_dim: u32) -> u32 {
    ATTN_HEAD_DIMS
        .iter()
        .copied()
        .filter(|&d| d >= head_dim)
        .min()
        .unwrap_or(head_dim)
}

impl Geometry {

    pub const EMPTY: Self = Self {
        hidden: 0,
        q_heads: 0,
        kv_heads: 0,
        head_dim: 0,
        head_dim_kernel: 0,
        intermediate: 0,
        moe_intermediate: 0,
        experts_per_token: 0,
        shared_intermediate: 0,
        vocab: 0,
    };

    #[must_use]
    pub const fn gqa_group(&self) -> u32 {

        match self.q_heads.checked_div(self.kv_heads) {
            Some(group) => group,
            None => 0,
        }
    }

    #[must_use]
    pub const fn head_dim_alloc(&self) -> u32 {
        if self.head_dim_kernel > self.head_dim {
            self.head_dim_kernel
        } else {
            self.head_dim
        }
    }

    #[must_use]
    pub const fn widest_mlp(&self) -> u32 {
        if self.moe_intermediate > self.intermediate {
            self.moe_intermediate
        } else {
            self.intermediate
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScaling {

    Piecewise {

        factor: f32,

        low_freq_factor: f32,

        high_freq_factor: f32,

        original_max_position: u32,
    },

    Yarn {

        factor: f32,

        beta_fast: f32,

        beta_slow: f32,

        attention_factor: f32,

        original_max_position: u32,

        truncate: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Deployment {

    pub layers: u32,

    pub norm_eps: f32,

    pub shape: Geometry,

    pub attention: Vec<LayerAttention>,

    pub kv: KvStyle,

    pub recurrent: Option<RecurrentShape>,

    pub prefill: PrefillStyle,

    pub attn_output: AttnOutput,

    pub logit_softcap: f32,

    pub attn_logit_softcap: f32,

    pub ple_dim: u32,

    pub norm: NormPlacement,

    pub norm_unit_offset: bool,

    pub v_norm: bool,

    pub mlp_gate: MlpGate,

    pub norm_topk_prob: bool,

    pub routed_scaling: f32,

    pub scales: BTreeMap<String, f32>,

    pub advertised: Advertised,

    pub rope_scaling: Option<RopeScaling>,

    pub towers: Towers,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Towers {

    pub audio: Option<AudioTower>,

    pub vision: Option<VisionTower>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioTower {

    pub layers: u32,

    pub hidden: u32,

    pub heads: u32,

    pub conv_kernel: u32,

    pub feature_size: u32,

    pub subsample_channels_0: u32,

    pub subsample_channels_1: u32,

    pub output_dims: u32,

    pub chunk_size: u32,

    pub context_left: u32,

    pub context_right: u32,

    pub logit_cap: f32,

    pub residual_weight: f32,

    pub norm_eps: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisionTower {

    pub layers: u32,

    pub hidden: u32,

    pub heads: u32,

    pub intermediate: u32,

    pub pooling_kernel: u32,

    pub norm_eps: f32,

    pub rope_theta: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Advertised {

    pub arch: &'static str,

    pub max_model_len: u32,

    pub media_encode: bool,
}

impl Deployment {

    #[must_use = "the refusal is the whole result; dropping it deploys a \
                  row whose store does not exist"]
    pub fn provisioned(self) -> Result<Self, Refusal> {
        match self.kv.store_refusal() {
            Some(no_store) => Err(no_store),
            None => Ok(self),
        }
    }

    #[must_use]
    pub fn empty() -> Self {
        Self {
            layers: 0,
            norm_eps: 0.0,

            norm_topk_prob: false,
            routed_scaling: 1.0,
            shape: Geometry::EMPTY,
            attention: Vec::new(),
            kv: KvStyle::Paged,
            recurrent: None,
            prefill: PrefillStyle::Planned,
            attn_output: AttnOutput::StatedArgs,
            logit_softcap: 0.0,

            attn_logit_softcap: 0.0,
            ple_dim: 0,
            norm: NormPlacement::Pre,
            norm_unit_offset: false,
            v_norm: false,
            mlp_gate: MlpGate::Silu,
            scales: BTreeMap::new(),
            advertised: Advertised::default(),
            rope_scaling: None,
            towers: Towers::default(),
        }
    }

    #[must_use]
    pub fn decode_head_dims(&self) -> Option<(u32, u32)> {
        let first = self.attention.first()?.head_dim;
        let other = self
            .attention
            .iter()
            .find(|a| a.head_dim != first)?
            .head_dim;
        Some((first, other))
    }

    #[must_use]
    pub fn full_attention_shape(&self) -> Option<(u32, u32, u32)> {
        let first = self.attention.first()?;
        let full = self.attention.iter().find(|a| a.window < 0)?;
        if full.head_dim == first.head_dim && full.kv_heads == first.kv_heads {
            return None;
        }
        Some((full.head_dim, full.kv_heads, full.rotary_dim))
    }

    #[must_use]
    pub fn shares_kv(&self) -> bool {
        self.attention
            .iter()
            .enumerate()
            .any(|(l, a)| a.kv_source as usize != l)
    }

    pub fn servable_by(&self, groups: &[u32]) -> Result<(), Refusal> {
        let (q, kv) = (self.shape.q_heads, self.shape.kv_heads);
        if kv == 0 || q % kv != 0 {
            return Err(Refusal::Unsupported(
                "the query heads do not divide the kv heads, so this stack \
                 asks for a fractional GQA group no build instantiates",
            ));
        }
        if groups.contains(&self.shape.gqa_group()) {
            Ok(())
        } else {
            Err(Refusal::Unsupported(
                "this build's decode does not instantiate the GQA group size \
                 this stack asks for",
            ))
        }
    }

    #[must_use]
    pub fn windows(&self) -> Vec<i32> {
        self.attention.iter().map(|a| a.window).collect()
    }

    #[must_use]
    pub fn theta_by_layer(&self) -> Vec<f32> {
        let first = self.attention.first().map_or(0.0, |a| a.rope_theta);
        if self.attention.iter().all(|a| a.rope_theta == first) {
            return Vec::new();
        }
        self.attention.iter().map(|a| a.rope_theta).collect()
    }

    #[must_use]
    pub fn rotary_by_layer(&self) -> Vec<u32> {
        if self.attention.iter().all(|a| a.rotary_dim == 0) {
            return Vec::new();
        }
        self.attention.iter().map(|a| a.rotary_dim).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {

    Unsupported(&'static str),

    Malformed(&'static str),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "this build cannot serve it: {what}"),
            Self::Malformed(why) => write!(f, "the checkpoint contradicts its own type: {why}"),
        }
    }
}

impl std::error::Error for Refusal {}
