use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub struct Gemma3nAltUpFacts {

    pub num_streams: u32,
    pub active: u32,
}

pub type Gemma3nAttnFacts = model_ir::facts::GqaFacts;

#[must_use]
pub const fn window_schedule<const N: usize>(
    full_attn_interval: u32,
    sliding_window: i32,
) -> [i32; N] {
    let mut out = [sliding_window; N];
    let mut l = 0;
    while l < N {
        if full_attn_interval > 0 && (l as u32 + 1).is_multiple_of(full_attn_interval) {
            out[l] = -1;
        }
        l += 1;
    }
    out
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Gemma3nFacts {
    pub vocab: u32,
    pub hidden: u32,

    pub per_layer_intermediate: &'static [u32],

    pub laurel_rank: u32,

    pub ple_width: u32,

    pub ple_vocab: u32,

    pub sparsity_layers: u32,
    pub altup: Gemma3nAltUpFacts,
    pub attn: Gemma3nAttnFacts,

    pub window_left: &'static [i32],
}

impl Gemma3nFacts {
    pub fn layers(&self) -> u32 {
        self.per_layer_intermediate.len() as u32
    }
    pub fn intermediate(&self, l: u32) -> u32 {
        self.per_layer_intermediate[l as usize]
    }

    pub fn is_sparse(&self, l: u32) -> bool {
        l < self.sparsity_layers
    }

    pub fn sparsity_std_mult(&self) -> f32 {
        1.644_853_6
    }

    pub fn gemma3n_synthetic() -> Self {
        Gemma3nFacts {

            window_left: &[],
            vocab: 262_144,
            hidden: 2048,
            per_layer_intermediate: &[8192; 6],
            laurel_rank: 64,
            ple_width: 256,
            ple_vocab: 262_144,
            sparsity_layers: 3,
            altup: Gemma3nAltUpFacts {
                num_streams: 4,
                active: 0,
            },
            attn: Gemma3nAttnFacts {
                heads: 8,
                kv_heads: 2,
                head_dim: 256,
            },
        }
    }
}
