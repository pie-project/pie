use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Gemma4Facts {
    pub hidden: u32,
    pub layers: u32,

    pub full_attn_interval: u32,
    pub q_heads: u32,
    pub kv_heads: u32,

    pub head_dim: u32,

    pub global_head_dim: u32,

    pub global_kv_heads: u32,

    pub global_rotary_dim: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub tied_embeddings: bool,

    pub kv_shared_layers: u32,

    pub ple_dim: u32,

    pub ple_vocab: u32,

    pub double_wide_shared: bool,

    pub logit_softcap: f32,
}

impl Gemma4Facts {
    #[must_use]
    pub fn is_full_attn(&self, l: u32) -> bool {
        model_ir::facts::full_attn_at(self.full_attn_interval, l)
    }

    #[must_use]
    pub fn is_kv_shared(&self, l: u32) -> bool {
        l >= self.layers.saturating_sub(self.kv_shared_layers)
    }

    #[must_use]
    pub fn intermediate_of(&self, l: u32) -> u32 {
        if self.double_wide_shared && self.is_kv_shared(l) {
            self.intermediate * 2
        } else {
            self.intermediate
        }
    }

    #[must_use]
    pub fn kv_source(&self, l: u32) -> Option<u32> {
        if !self.is_kv_shared(l) {
            return None;
        }
        let first_shared = self.layers.saturating_sub(self.kv_shared_layers);
        (0..first_shared)
            .rev()
            .find(|&j| self.is_full_attn(j) == self.is_full_attn(l))
    }

    #[must_use]
    pub fn head_dim_of(&self, l: u32) -> u32 {
        if self.is_full_attn(l) {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }

    #[must_use]
    pub fn kv_heads_of(&self, l: u32) -> u32 {
        if self.is_full_attn(l) {
            self.global_kv_heads
        } else {
            self.kv_heads
        }
    }

    #[must_use]
    pub const fn gemma_4_e4b() -> Self {
        Self {
            hidden: 2560,
            layers: 42,
            full_attn_interval: 6,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            global_head_dim: 512,

            global_kv_heads: 2,

            global_rotary_dim: 128,
            intermediate: 10_240,
            vocab: 262_144,
            tied_embeddings: true,
            kv_shared_layers: 18,
            ple_dim: 256,
            ple_vocab: 262_144,
            double_wide_shared: false,
            logit_softcap: 30.0,
        }
    }

    #[must_use]
    pub const fn gemma_4_e2b() -> Self {
        Self {
            hidden: 1536,
            layers: 35,
            full_attn_interval: 5,
            q_heads: 8,
            kv_heads: 1,
            head_dim: 256,
            global_head_dim: 512,

            global_kv_heads: 1,

            global_rotary_dim: 128,
            intermediate: 6144,
            vocab: 262_144,
            tied_embeddings: true,
            kv_shared_layers: 20,
            ple_dim: 256,
            ple_vocab: 262_144,
            double_wide_shared: true,
            logit_softcap: 30.0,
        }
    }

    #[must_use]
    pub const fn gemma_4_31b() -> Self {
        Self {
            hidden: 5376,
            layers: 60,
            full_attn_interval: 6,
            q_heads: 32,
            kv_heads: 16,
            head_dim: 256,
            global_head_dim: 512,
            global_kv_heads: 4,

            global_rotary_dim: 128,
            intermediate: 21_504,
            vocab: 262_144,
            tied_embeddings: true,
            kv_shared_layers: 0,
            ple_dim: 0,
            ple_vocab: 0,
            double_wide_shared: false,
            logit_softcap: 30.0,
        }
    }

    #[must_use]
    pub const fn gemma_4_26b_a4b() -> Self {
        Self {
            hidden: 2816,
            layers: 30,
            full_attn_interval: 6,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 256,
            global_head_dim: 512,
            global_kv_heads: 2,
            global_rotary_dim: 128,
            intermediate: 2112,
            vocab: 262_144,
            tied_embeddings: true,
            kv_shared_layers: 0,
            ple_dim: 0,
            ple_vocab: 0,
            double_wide_shared: false,
            logit_softcap: 30.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Gemma4Mixture {
    pub num_experts: u32,

    pub experts_per_token: u32,

    pub moe_intermediate: u32,
}

impl Gemma4Mixture {
    #[must_use]
    pub const fn gemma_4_26b_a4b() -> Self {
        Self {
            num_experts: 128,
            experts_per_token: 8,
            moe_intermediate: 704,
        }
    }
}
