use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma2AttnFacts {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,

    pub attn_logit_softcap: bool,
}

impl Gemma2AttnFacts {
    pub const fn q_width(&self) -> u32 {
        self.heads * self.head_dim
    }
    pub const fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma2Facts {
    pub layers: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub intermediate: u32,
    pub tied_embeddings: bool,

    pub final_logit_softcap: bool,

    pub sliding_window: i32,

    pub full_attn_interval: u32,
    pub attn: Gemma2AttnFacts,
}

impl Gemma2Facts {

    #[must_use]
    pub fn is_global(&self, l: u32) -> bool {
        model_ir::facts::full_attn_at(self.full_attn_interval, l)
    }

    #[must_use]
    pub fn window_left_at(&self, l: u32) -> i32 {
        if self.is_global(l) {
            -1
        } else {
            self.sliding_window
        }
    }

    #[must_use]
    pub fn window_by_layer(&self) -> Vec<i32> {
        (0..self.layers).map(|l| self.window_left_at(l)).collect()
    }

    pub fn gemma_2_9b() -> Self {
        Gemma2Facts {
            layers: 42,
            vocab: 256_000,
            hidden: 3584,
            intermediate: 14336,
            tied_embeddings: true,
            final_logit_softcap: true,

            sliding_window: 4096,
            full_attn_interval: 2,
            attn: Gemma2AttnFacts {
                heads: 16,
                kv_heads: 8,
                head_dim: 256,
                attn_logit_softcap: true,
            },
        }
    }
}
