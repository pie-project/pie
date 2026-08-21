use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NemotronLayerKind {

    Mamba,

    Attention,

    Mlp,
}

impl NemotronLayerKind {

    #[must_use]
    pub const fn glyph(self) -> char {
        match self {
            Self::Mamba => 'M',
            Self::Attention => '*',
            Self::Mlp => '-',
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct NemotronMambaFacts {
    pub num_heads: u32,
    pub head_dim: u32,
    pub state_size: u32,

    pub n_groups: u32,
    pub conv_kernel: u32,
}

impl NemotronMambaFacts {

    #[must_use]
    pub const fn intermediate(&self) -> u32 {
        self.num_heads * self.head_dim
    }

    #[must_use]
    pub const fn conv_dim(&self) -> u32 {
        self.intermediate() + 2 * self.n_groups * self.state_size
    }

    #[must_use]
    pub const fn in_proj_width(&self) -> u32 {
        self.intermediate() + self.conv_dim() + self.num_heads
    }
}

pub type NemotronAttnFacts = model_ir::facts::GqaFacts;

pub type NemotronMoeFacts = model_ir::facts::MoeFacts;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NemotronHFacts {
    pub vocab: u32,
    pub hidden: u32,

    pub layer_types: &'static [NemotronLayerKind],
    pub mamba: NemotronMambaFacts,
    pub attn: NemotronAttnFacts,

    pub moe: NemotronMoeFacts,

    pub tied_embeddings: bool,

    pub window_left: &'static [i32],
}

pub const SCHEDULE_52: &[NemotronLayerKind] = &{
    use NemotronLayerKind::{Attention as A, Mamba as M, Mlp as F};
    [
        M, F, M, F, M, F, M, A, F, M, F, M, F, M, F, M, F, M, A, F, M, F, M, F, M, F, M, F, M, A,
        F, M, F, M, F, M, F, M, F, M, A, F, M, F, M, F, M, F, M, F, M, F,
    ]
};

pub const SCHEDULE_98: &[NemotronLayerKind] = &{
    use NemotronLayerKind::{Attention as A, Mamba as M, Mlp as F};
    [
        M, F, M, F, M, F, M, F, M, F, M, F, M, F, M, F, M, A, F, M, F, M, F, M, F, M, F, M, F, M,
        F, M, F, M, F, M, F, M, A, F, M, F, M, F, M, F, M, F, M, A, F, M, F, M, F, M, F, M, F, M,
        A, F, M, F, M, F, M, F, M, F, M, F, M, F, M, F, F, F, M, M, F, F, F, M, F, M, A, F, M, F,
        M, F, M, F, M, F, M, F,
    ]
};

impl NemotronHFacts {
    #[must_use]
    pub fn layers(&self) -> u32 {
        self.layer_types.len() as u32
    }

    #[must_use]
    pub fn kind(&self, l: u32) -> NemotronLayerKind {
        self.layer_types[l as usize]
    }

    #[must_use]
    pub fn mamba_layers(&self) -> Vec<u32> {
        (0..self.layers())
            .filter(|&l| self.kind(l) == NemotronLayerKind::Mamba)
            .collect()
    }

    #[must_use]
    pub const fn is_mixture(&self) -> bool {
        self.moe.num_experts > 0
    }

    #[must_use]
    pub const fn nemotron_h_8b() -> Self {
        Self {
            vocab: 131_072,
            hidden: 4096,
            layer_types: SCHEDULE_52,
            mamba: NemotronMambaFacts {
                num_heads: 128,
                head_dim: 64,
                state_size: 128,
                n_groups: 8,
                conv_kernel: 4,
            },

            attn: NemotronAttnFacts {
                heads: 32,
                kv_heads: 8,
                head_dim: 128,
            },
            moe: dense(21_504),
            tied_embeddings: false,
            window_left: &[],
        }
    }

    #[must_use]
    pub const fn nemotron_h_4b() -> Self {
        Self {
            vocab: 131_072,
            hidden: 3072,
            layer_types: SCHEDULE_52,
            mamba: NemotronMambaFacts {
                num_heads: 112,
                head_dim: 64,
                state_size: 128,
                n_groups: 8,
                conv_kernel: 4,
            },
            attn: NemotronAttnFacts {
                heads: 32,
                kv_heads: 8,
                head_dim: 128,
            },
            moe: dense(12_288),
            tied_embeddings: false,
            window_left: &[],
        }
    }

    #[must_use]
    pub const fn nemotron_h_47b() -> Self {
        Self {
            vocab: 131_072,
            hidden: 8192,
            layer_types: SCHEDULE_98,
            mamba: NemotronMambaFacts {
                num_heads: 256,
                head_dim: 64,

                state_size: 256,
                n_groups: 8,
                conv_kernel: 4,
            },
            attn: NemotronAttnFacts {
                heads: 64,
                kv_heads: 8,
                head_dim: 128,
            },
            moe: dense(30_720),
            tied_embeddings: false,
            window_left: &[],
        }
    }

    #[must_use]
    pub const fn nemotron_h_synthetic() -> Self {
        use NemotronLayerKind::{Attention, Mamba, Mlp};
        Self {
            vocab: 131_072,
            hidden: 2048,
            layer_types: &[Mamba, Mlp, Mamba, Attention, Mamba, Mlp],
            mamba: NemotronMambaFacts {
                num_heads: 16,
                head_dim: 64,
                state_size: 128,
                n_groups: 8,
                conv_kernel: 4,
            },
            attn: NemotronAttnFacts {
                heads: 16,
                kv_heads: 4,
                head_dim: 128,
            },
            moe: NemotronMoeFacts {
                num_experts: 32,
                top_k: 4,

                norm_topk_prob: true,

                routed_scaling: 1.0,
                moe_intermediate: 1024,
                shared_intermediate: 1024,
            },
            tied_embeddings: false,

            window_left: &[],
        }
    }
}

const fn dense(intermediate: u32) -> NemotronMoeFacts {
    NemotronMoeFacts {
        num_experts: 0,
        top_k: 0,

        norm_topk_prob: true,
        routed_scaling: 1.0,
        moe_intermediate: intermediate,
        shared_intermediate: 0,
    }
}
