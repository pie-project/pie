//! The handle vocabulary: runtime-typed, `Copy`, and never dereferenced.
//! Regions are always dense (`stride == width`); no stride or offset is
//! carried. Outputs are passed by value too: a handle names GPU memory the
//! host never dereferences, so read/write intent is recorded in the argument
//! marshalling instead ([`Tensor::arg`] vs [`Tensor::arg_mut`]).

use dtype::Dtype;

use crate::encode::ArgValue;

/// A dtype-erased view of one device buffer: `rows x width` dense elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor {
    /// The driver-scoped buffer handle an [`Encode`](crate::encode::Encode)
    /// sink resolves at fire time.
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

    /// Marshal as a read binding.
    #[must_use]
    pub const fn arg(self) -> ArgValue {
        ArgValue::Buffer(self.buf)
    }

    /// Marshal as a write binding — the only place write intent lives.
    #[must_use]
    pub const fn arg_mut(self) -> ArgValue {
        ArgValue::BufferMut(self.buf)
    }
}

/// A fire-aligned value paired with the fire's shared boundaries. `indptr` is
/// the one indptr of the whole fire (`i32`, `[lanes + 1]`). Boundary-oblivious
/// kernels take `data` alone; boundary-aware entries take the pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaggedTensor {
    pub data: Tensor,

    /// `i32`, `[lanes + 1]`, shared across the fire.
    pub indptr: Tensor,
}

/// The paged kv cache a fire reads and appends into. Storage only: all
/// geometry, including write tables, arrives as declared inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPool {
    /// Key pages (the kv dtype rides on the handle).
    pub keys: Tensor,

    /// Value pages.
    pub values: Tensor,

    /// `u32` page table: which page holds each logical block.
    pub page_indices: Tensor,

    /// `u32`, `[lanes + 1]`: each request's span of `page_indices`.
    pub page_indptr: Tensor,

    /// Tokens per page.
    pub page_size: i32,

    /// Elements from one in-page token row to the next.
    pub seq_stride: u64,

    /// Elements from one head's plane to the next.
    pub head_stride: u64,
}

/// The recurrent-state pool (ssm/linear-attention). Slot-addressed, not paged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrentPool {
    /// `f32` recurrent state, one bank per slot.
    pub state: Tensor,

    /// `u32` per lane: which state slot each request owns.
    pub slots: Tensor,

    /// `f32` rolling convolution state, read side.
    pub conv_state: Tensor,

    /// `f32` rolling convolution state, write side.
    pub new_conv_state: Tensor,
}

/// A quantized weight as the shaders read it: packed codes, per-group scale,
/// and — for an affine scheme — per-group offset.
///
/// `biases` is `Option` and not a zero handle: MXFP4's e8m0 exponent is a
/// pure scale (`w = code * 2^(e-127)`), nothing to offset by, while MLX's
/// affine u4 stores `w = code * scale + bias` and a bank read without its
/// biases is a wrong weight, not just a coarser one.
///
/// `group` and `bits` travel with the planes since a checkpoint is not
/// uniform in them (e.g. a 4-bit stack whose router gate is 8-bit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bank {
    /// The packed codes, `bits` apiece.
    pub codes: Tensor,

    /// One scale per `group` codes: bf16 factors for an affine bank, e8m0
    /// exponent bytes for an mxfp4 one.
    pub scales: Tensor,

    /// One zero point per `group` codes, in the activation's dtype — `None`
    /// for a symmetric scheme.
    pub biases: Option<Tensor>,

    /// Codes per scale entry.
    pub group: u32,

    /// Bits per code.
    pub bits: u32,
}

impl Bank {
    /// Whether this bank's groups are offset as well as scaled — the one
    /// question that picks between the affine and the mxfp4 point families.
    #[must_use]
    pub const fn affine(&self) -> bool {
        self.biases.is_some()
    }
}
