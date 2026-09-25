use dtype::Dtype;

use crate::encode::ArgValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaggedTensor {
    pub data: Tensor,

    pub indptr: Tensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPool {
    pub keys: Tensor,

    pub values: Tensor,

    pub page_indices: Tensor,

    pub page_indptr: Tensor,

    pub page_size: i32,

    pub seq_stride: u64,

    pub head_stride: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrentPool {
    pub state: Tensor,

    pub slots: Tensor,

    pub conv_state: Tensor,

    pub new_conv_state: Tensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bank {
    pub codes: Tensor,

    pub scales: Tensor,

    pub biases: Option<Tensor>,

    pub group: u32,

    pub bits: u32,
}

impl Bank {
    #[must_use]
    pub const fn affine(&self) -> bool {
        self.biases.is_some()
    }
}
