#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsmBackboneFacts {
    pub hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,

    pub head_dim: u32,
    pub intermediate: u32,

    pub text_vocab: u32,

    pub audio_vocab: u32,

    pub codebooks: u32,
}

impl CsmBackboneFacts {

    #[must_use]
    pub const fn q_width(&self) -> u32 {
        self.q_heads * self.head_dim
    }

    #[must_use]
    pub const fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsmDepthFacts {
    pub hidden: u32,

    pub backbone_hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,

    pub head_dim: u32,
    pub intermediate: u32,

    pub vocab: u32,

    pub codebooks: u32,
}

impl CsmDepthFacts {

    #[must_use]
    pub const fn code_table_rows(&self) -> u32 {
        self.codebooks * self.vocab
    }

    #[must_use]
    pub const fn head_slices(&self) -> u32 {
        self.codebooks - 1
    }

    #[must_use]
    pub const fn q_width(&self) -> u32 {
        self.q_heads * self.head_dim
    }

    #[must_use]
    pub const fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsmCodecFacts {

    pub hidden: u32,

    pub codebook_dim: u32,

    pub codebook_size: u32,

    pub quantizers: u32,

    pub semantic_quantizers: u32,

    pub filters: u32,
}

impl CsmCodecFacts {

    #[must_use]
    pub const fn acoustic_quantizers(&self) -> u32 {
        self.quantizers - self.semantic_quantizers
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsmFacts {
    pub backbone: CsmBackboneFacts,
    pub depth: CsmDepthFacts,
    pub codec: CsmCodecFacts,

    pub tied_codebooks: bool,

    pub tied_embeddings: bool,
}

impl CsmFacts {

    #[must_use]
    pub const fn csm_1b() -> Self {
        Self {
            backbone: CsmBackboneFacts {
                hidden: 2048,
                layers: 16,
                q_heads: 32,
                kv_heads: 8,
                head_dim: 64,
                intermediate: 8192,
                text_vocab: 128_256,
                audio_vocab: 2051,
                codebooks: 32,
            },
            depth: CsmDepthFacts {
                hidden: 1024,
                backbone_hidden: 2048,
                layers: 4,
                q_heads: 8,
                kv_heads: 2,
                head_dim: 128,
                intermediate: 8192,
                vocab: 2051,
                codebooks: 32,
            },
            codec: CsmCodecFacts {
                hidden: 512,
                codebook_dim: 256,
                codebook_size: 2048,
                quantizers: 32,
                semantic_quantizers: 1,
                filters: 64,
            },
            tied_codebooks: true,
            tied_embeddings: false,
        }
    }

    #[must_use]
    pub const fn csm_synthetic() -> Self {
        Self {
            backbone: CsmBackboneFacts {
                hidden: 128,
                layers: 4,
                q_heads: 8,
                kv_heads: 8,
                head_dim: 16,
                intermediate: 128,
                text_vocab: 1000,
                audio_vocab: 1000,
                codebooks: 8,
            },
            depth: CsmDepthFacts {
                hidden: 64,
                backbone_hidden: 128,
                layers: 2,
                q_heads: 4,
                kv_heads: 2,
                head_dim: 16,
                intermediate: 128,
                vocab: 2048,
                codebooks: 8,
            },
            codec: CsmCodecFacts {
                hidden: 64,
                codebook_dim: 32,
                codebook_size: 1024,
                quantizers: 8,
                semantic_quantizers: 1,
                filters: 16,
            },
            tied_codebooks: true,
            tied_embeddings: false,
        }
    }
}
