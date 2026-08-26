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

use new_model_ir::Dtype;

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
/// (design §7); this is what the pool pointer resolves to.
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

    /// `u32` per lane: the page this fire appends into.
    pub write_page: Tensor,

    /// `u32` per lane: the in-page slot the append starts at.
    pub write_offset: Tensor,

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
