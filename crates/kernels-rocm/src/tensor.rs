use dtype::Dtype;

use crate::jit::ArgValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub ptr: u64,

    pub rows: u32,

    pub width: u32,

    pub dtype: Dtype,
}

impl Tensor {
    #[must_use]
    pub const fn new(ptr: u64, rows: u32, width: u32, dtype: Dtype) -> Self {
        Self {
            ptr,
            rows,
            width,
            dtype,
        }
    }

    #[must_use]
    pub const fn arg(&self) -> ArgValue {
        ArgValue::Ptr(self.ptr)
    }

    #[must_use]
    pub const fn elements(&self) -> u64 {
        self.rows as u64 * self.width as u64
    }

    pub const ABSENT: Tensor = Tensor::new(0, 0, 0, Dtype::U8);

    #[must_use]
    pub const fn is_absent(&self) -> bool {
        self.ptr == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaggedTensor {
    pub data: Tensor,

    pub indptr: Tensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPool {
    pub keys: Tensor,

    pub values: Tensor,

    pub bf16_keys: Tensor,

    pub bf16_values: Tensor,

    pub key_scales: Tensor,

    pub value_scales: Tensor,

    pub page_indices: Tensor,

    pub page_indptr: Tensor,

    pub last_page_lens: Tensor,

    pub row_valid: Tensor,

    pub env_min: Tensor,

    pub env_max: Tensor,

    pub has_envelopes: bool,

    pub page_size: i32,

    pub seq_stride: i64,

    pub head_stride: i64,

    pub layout: i32,

    pub scheme_byte: i32,

    pub block_size: i32,

    pub max_pages_per_request: i32,

    pub pages_in_batch: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrentPool {
    pub slab: Tensor,

    pub slot_ids: Tensor,

    pub slot_stride_elems: i64,

    pub conv_slab: Tensor,

    pub conv_stride: i64,

    pub write_state: bool,

    pub write_state_mask: Tensor,

    pub commit_len: Tensor,

    pub begin_at: Tensor,

    pub fused_decay: bool,
}
