use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Spec {
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
    pub double_wide_shared: bool,
    pub logit_softcap: f32,
}


pub const fn gemma_4_e4b() -> Spec {
    Spec {
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
        double_wide_shared: false,
        logit_softcap: 30.0,
    }
}

pub const fn gemma_4_e2b() -> Spec {
    Spec {
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
        double_wide_shared: true,
        logit_softcap: 30.0,
    }
}

pub const fn gemma_4_31b() -> Spec {
    Spec {
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
        double_wide_shared: false,
        logit_softcap: 30.0,
    }
}

pub const fn gemma_4_26b_a4b() -> Spec {
    Spec {
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
        double_wide_shared: false,
        logit_softcap: 30.0,
    }
}
