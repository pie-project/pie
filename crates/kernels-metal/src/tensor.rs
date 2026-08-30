//! The handle vocabulary: runtime-typed, `Copy`, and never dereferenced.
//!
//! The old plane spoke `Tensor<T>` — a `u32` buffer handle behind a phantom
//! scalar, with rows/width riding on the `In`/`Out`/`InOut` marks around it.
//! Here the type parameter is erased into a [`Dtype`] field and the marks'
//! extents fold onto the handle itself; nothing else is added. The old
//! handle carried no stride or offset (its regions were always dense,
//! `stride == width`), so none are carried here either.
//!
//! One convention, stated once: **outputs are passed by value too.** A handle
//! names GPU memory the host never dereferences, so `&`/`&mut` would be
//! intent the borrow checker cannot enforce — read/write intent is recorded
//! where it is real, in the argument marshalling ([`Tensor::arg`] vs
//! [`Tensor::arg_mut`]).

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

/// A fire-aligned value paired with the fire's shared boundaries (design §5).
///
/// `indptr` is the one indptr of the whole fire — `i32`, `[lanes + 1]` — and
/// every fire-aligned value can be viewed through it. Boundary-oblivious
/// kernels take `data` alone; boundary-aware entries (`prefill`,
/// `*_chunked`) take the pair, already built by the driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaggedTensor {
    pub data: Tensor,

    /// `i32`, `[lanes + 1]`, shared across the fire.
    pub indptr: Tensor,
}

/// The paged kv cache a fire reads and appends into — the old `PagedKvView`,
/// dtype-erased. Storage only: all geometry arrives as declared inputs
/// (design §7) — the write tables included, since the `kv_append` ops state
/// `write_page`/`write_offset` themselves; this is what the pool pointer
/// resolves to.
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

    /// Elements from one in-page token row to the next (was `Usize`).
    pub seq_stride: u64,

    /// Elements from one head's plane to the next (was `Usize`).
    pub head_stride: u64,
}

/// The recurrent-state pool (ssm/linear-attention) — the old `RecurrentView`,
/// dtype-erased. Slot-addressed, not paged.
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

/// A quantized weight as the shaders read it: the packed codes, the scale
/// each group is multiplied by, and — for an affine scheme — the offset each
/// group is shifted by.
///
/// **THE THIRD PLANE IS AN `Option` AND NOT A ZERO HANDLE.** Whether a bank
/// carries zero points is what separates the two four-bit formats this plane
/// serves: MXFP4's e8m0 exponent is a pure scale, so `w = code * 2^(e-127)`
/// and there is nothing to offset by, while MLX's affine u4 stores
/// `w = code * scale + bias` and a bank read without its biases is not a
/// coarser weight but a wrong one. Both entry families hold the biases seat —
/// the mxfp4 points bind a null there — so the `Option` is what the driver
/// says and the null is what the encoder does with it.
///
/// `group` and `bits` travel WITH the planes because they are the point
/// selection: `affine_qmv_fast_bfloat16_gs_64_b_4` and its neighbours differ
/// in nothing else, and a checkpoint is not uniform in them (`mlx_lm`
/// publishes a 4-bit stack whose router gate is 8-bit). A shell holding one
/// pair for the whole model reads the odd tensor at the wrong point and
/// nothing anywhere says so.
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
