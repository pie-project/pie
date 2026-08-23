use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormPlacement {
    #[default]
    Pre,
    Post,
    Sandwich,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum QkNorm {
    #[default]
    Off,
    PerHead,
    Global,
}

pub fn window_left_at(list: &[i32], l: u32) -> i32 {
    match list.len() {
        0 => -1,
        n => list[(l as usize).min(n - 1)],
    }
}

pub fn rope_theta_at(list: &[f32], l: u32) -> f32 {
    match list.len() {
        0 => 0.0,
        n => list[(l as usize).min(n - 1)],
    }
}

#[must_use]
pub fn full_attn_at(interval: u32, l: u32) -> bool {
    interval > 0 && (l + 1).is_multiple_of(interval)
}

#[must_use]
pub fn after_dense_prefix(dense_layers: u32, l: u32) -> bool {
    l >= dense_layers
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MlaFacts {
    pub hidden: u32,
    pub heads: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,

    #[serde(default)]
    pub output_gate: bool,
}

impl MlaFacts {
    #[must_use]
    pub const fn qk_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    #[must_use]
    pub const fn q_b_width(&self) -> u32 {
        self.heads * self.qk_head_dim()
    }

    #[must_use]
    pub const fn kv_a_width(&self) -> u32 {
        self.kv_lora_rank + self.qk_rope_head_dim
    }

    #[must_use]
    pub const fn v_width(&self) -> u32 {
        self.heads * self.v_head_dim
    }

    #[must_use]
    pub const fn q_kv_a_width(&self) -> u32 {
        self.q_lora_rank + self.kv_a_width()
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MoeFacts {
    pub num_experts: u32,
    pub top_k: u32,

    pub norm_topk_prob: bool,

    pub routed_scaling: f32,
    pub moe_intermediate: u32,
    pub shared_intermediate: u32,
}

impl MoeFacts {
    #[must_use]
    pub const fn has_shared_expert(&self) -> bool {
        self.shared_intermediate > 0
    }

    #[must_use]
    pub const fn routes(&self, tokens: u32) -> u32 {
        tokens * self.top_k
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GqaFacts {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
}

impl GqaFacts {
    #[must_use]
    pub const fn q_width(&self) -> u32 {
        self.heads * self.head_dim
    }

    #[must_use]
    pub const fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }

    #[must_use]
    pub const fn group_size(&self) -> u32 {
        match self.heads.checked_div(self.kv_heads) {
            Some(group) => group,
            None => 0,
        }
    }
}
