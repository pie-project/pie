#[derive(Debug, Clone, PartialEq)]
pub struct GptOssFacts {
    pub hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,

    pub intermediate: u32,
    pub experts: u32,
    pub top_k: u32,
    pub vocab: u32,
    pub tied_embeddings: bool,

    pub swiglu_limit: f32,
}

impl GptOssFacts {

    pub fn is_sliding(&self, l: u32) -> bool {
        l.is_multiple_of(2)
    }

    pub fn gpt_oss_20b() -> Self {
        Self {
            hidden: 2880,
            layers: 24,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            intermediate: 2880,
            experts: 32,
            top_k: 4,
            vocab: 201088,
            tied_embeddings: false,
            swiglu_limit: 7.0,
        }
    }
}
