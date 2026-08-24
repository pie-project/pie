use kernels::shader::{Tensor, Usize, bf16};

#[derive(Debug, Clone, Copy)]
pub struct PagedKvView {
    pub keys: Tensor<bf16>,

    pub values: Tensor<bf16>,

    pub page_indices: Tensor<u32>,

    pub page_indptr: Tensor<u32>,

    pub write_page: Tensor<u32>,

    pub write_offset: Tensor<u32>,

    pub page_size: i32,

    pub seq_stride: Usize,

    pub head_stride: Usize,
}

#[derive(Debug, Clone, Copy)]
pub struct RecurrentView {
    pub state: Tensor<f32>,

    pub slots: Tensor<u32>,

    pub conv_state: Tensor<f32>,

    pub new_conv_state: Tensor<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct MaskView {
    pub mask: Tensor<u8>,

    pub enabled: Tensor<u8>,

    pub stride: u32,
}

kernels::raise!(

    KvCache = "kv_cache" => PagedKvView
);
kernels::raise!(

    RecurrentState = "recurrent_state" => RecurrentView
);
kernels::raise!(

    AttnMask = "attention_mask" => MaskView
);

#[derive(Debug, Clone, Copy)]
pub struct SplitView {
    pub partials: Tensor<f32>,

    pub splits: i32,
}

kernels::raise!(

    AttnSplit = "attn.split_policy" => SplitView
);
