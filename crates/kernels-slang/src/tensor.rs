use dtype::Dtype;

use crate::encode::ArgValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tensor {
    pub buf: u32,

    pub rows: u32,

    pub width: u32,

    pub dtype: Dtype,
}

impl Tensor {
    #[must_use]
    pub const fn new(buf: u32, rows: u32, width: u32, dtype: Dtype) -> Self {
        Self {
            buf,
            rows,
            width,
            dtype,
        }
    }

    #[must_use]
    pub const fn arg(self) -> ArgValue {
        ArgValue::Buffer(self.buf)
    }

    #[must_use]
    pub const fn arg_mut(self) -> ArgValue {
        ArgValue::BufferMut(self.buf)
    }
}

/// A quantized weight bank: codes, a scale per group, and a bias per group
/// when the bank is affine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bank {
    pub codes: Tensor,

    pub scales: Tensor,

    pub biases: Option<Tensor>,

    pub group: u32,

    pub bits: u32,
}

/// A ragged batch: `data`'s rows split by the i32 CSR `indptr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RaggedTensor {
    pub data: Tensor,

    pub indptr: Tensor,
}

/// A paged KV cache: key and value pages, the pages each request reads
/// (`page_indices` split by `page_indptr`), and the strides of a page row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvPool {
    pub keys: Tensor,

    pub values: Tensor,

    pub page_indices: Tensor,

    pub page_indptr: Tensor,

    pub page_size: i32,

    pub seq_stride: u64,

    pub head_stride: u64,
}
