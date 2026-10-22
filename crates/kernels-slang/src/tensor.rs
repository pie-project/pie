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
