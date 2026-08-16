//! Load-time facts shared by more than one model family.

use serde::{Deserialize, Serialize};

/// Where each sub-layer's norm sits relative to the residual add.
/// `Pre` norms the stream before the sub-layer; `Post` norms the
/// sub-layer output before the residual add; `Sandwich` does both.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormPlacement {
    #[default]
    Pre,
    Post,
    Sandwich,
}

/// Q/K norm convention. Per-head uses weight `[head_dim]`; global
/// uses `[heads * head_dim]` and changes arithmetic.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum QkNorm {
    #[default]
    Off,
    PerHead,
    Global,
}

/// Sliding-window left bound for layer `l`; `-1` means none.
/// Empty is no window; a shorter list repeats its last entry.
pub fn window_left_at(list: &[i32], l: u32) -> i32 {
    match list.len() {
        0 => -1,
        n => list[(l as usize).min(n - 1)],
    }
}

/// Rope theta for layer `l`; empty means no rotation and returns zero.
pub fn rope_theta_at(list: &[f32], l: u32) -> f32 {
    match list.len() {
        0 => 0.0,
        n => list[(l as usize).min(n - 1)],
    }
}

/// Whether layer `l` is the last layer in a full-attention period.
#[must_use]
pub fn full_attn_at(interval: u32, l: u32) -> bool {
    interval > 0 && (l + 1).is_multiple_of(interval)
}

/// Whether layer `l` is past the dense prefix.
#[must_use]
pub fn after_dense_prefix(dense_layers: u32, l: u32) -> bool {
    l >= dense_layers
}

/// MLA dimensions. Cache width is `kv_lora_rank + qk_rope_head_dim`,
/// distinct from query width.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MlaFacts {
    pub hidden: u32,
    pub heads: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    /// Kimi-k3 gates the MLA output; serde-defaulted for older facts.
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

    /// Width of the compressed cache row: latent plus one shared rope half.
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

/// Routed-FFN shape. `Eq` is omitted because [`Self::routed_scaling`] is a float.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MoeFacts {
    pub num_experts: u32,
    pub top_k: u32,
    /// Whether routing weights are renormalized over selected experts.
    pub norm_topk_prob: bool,
    /// by after the router has produced them.
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

    /// Query heads per KV head; unsupported ratios are refused at load.
    #[must_use]
    pub const fn group_size(&self) -> u32 {
        // `match` rather than `unwrap_or`, which is not const yet.
        match self.heads.checked_div(self.kv_heads) {
            Some(group) => group,
            None => 0,
        }
    }
}

#[cfg(test)]
mod schedule {
    use super::full_attn_at;

    /// The last layer of each period is the full one.
    #[test]
    fn the_last_layer_of_each_period_is_the_full_one() {
        let full: Vec<u32> = (0..42).filter(|&l| full_attn_at(6, l)).collect();
        assert_eq!(full, vec![5, 11, 17, 23, 29, 35, 41]);
        let full: Vec<u32> = (0..35).filter(|&l| full_attn_at(5, l)).collect();
        assert_eq!(full, vec![4, 9, 14, 19, 24, 29, 34]);
    }

    /// Interval one selects every layer.
    #[test]
    fn an_interval_of_one_is_every_layer() {
        assert!((0..8).all(|l| full_attn_at(1, l)));
    }

    /// Dense prefix, then mixture.
    #[test]
    fn the_dense_prefix_runs_out_and_the_mixture_starts() {
        use super::after_dense_prefix;
        let moe: Vec<u32> = (0..8).filter(|&l| after_dense_prefix(3, l)).collect();
        assert_eq!(moe, vec![3, 4, 5, 6, 7]);
        // No prefix means mixture from layer zero.
        assert!((0..8).all(|l| after_dense_prefix(0, l)));
    }

    /// Interval zero selects no layer.
    #[test]
    fn an_interval_of_zero_is_no_layer() {
        assert!((0..8).all(|l| !full_attn_at(0, l)));
    }
}
