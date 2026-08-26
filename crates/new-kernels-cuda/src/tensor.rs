//! The handle vocabulary: runtime-typed, `Copy`, and never dereferenced.
//!
//! The old plane spoke `Tensor<T>` — a device pointer behind a phantom
//! scalar, with rows/width riding on the `In`/`Out`/`InOut` marks around it.
//! Here the type parameter is erased into a [`Dtype`] field and the marks'
//! extents fold onto the handle itself; nothing else is added. The old
//! handle carried no stride or offset (its regions were always dense,
//! `stride == width`), so none are carried here either.
//!
//! One rule, stated once: **`&`/`&mut` is intent, not borrow discipline.**
//! A handle names GPU memory the host never dereferences, so the borrow
//! checker cannot enforce anything about the bytes behind it — an entry
//! takes its outputs as `&mut Tensor` to *say* what it writes, and that
//! signature is the whole record of write intent (design §5).

use new_model_ir::Dtype;

use crate::jit::ArgValue;

/// A dtype-erased view of one device buffer: `rows x width` dense elements
/// behind a `CUdeviceptr`-shaped address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor {
    /// The device address, resolved by the driver (arena offset, weight
    /// table, or pool) before this handle reaches an entry.
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

    /// Marshal as one launch argument — read/write intent already lives in
    /// the entry's `&`/`&mut` signature, so there is only one spelling.
    #[must_use]
    pub const fn arg(&self) -> ArgValue {
        ArgValue::Ptr(self.ptr)
    }

    /// Total elements, as the `usize` the elementwise kernels count in.
    #[must_use]
    pub const fn elements(&self) -> u64 {
        self.rows as u64 * self.width as u64
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
/// dtype-erased. Storage plus the page geometry the driver resolved from
/// declared inputs (design §7); this is what the pool pointer resolves to.
///
/// Erasure retired two fields outright: `storage_dtype` and `native_bf16`
/// restated what `keys.dtype` now says, and `qo_indptr` is the fire's shared
/// indptr, which boundary-aware entries receive on their [`RaggedTensor`]
/// instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPool {
    /// Key pages (the storage dtype rides on the handle — `U8` for fp8
    /// schemes, `Bf16` for native storage).
    pub keys: Tensor,

    /// Value pages.
    pub values: Tensor,

    /// The bf16 shadow pages a mixed scheme keeps beside quantized storage.
    pub bf16_keys: Tensor,

    /// Shadow value pages.
    pub bf16_values: Tensor,

    /// Per-group key scales for quantized schemes.
    pub key_scales: Tensor,

    /// Per-group value scales.
    pub value_scales: Tensor,

    /// `i32` page table: which page holds each logical block.
    pub page_indices: Tensor,

    /// `i32`, `[lanes + 1]`: each request's span of `page_indices`.
    pub page_indptr: Tensor,

    /// `i32` per lane: how full each request's last page is.
    pub last_page_lens: Tensor,

    /// `i32` per lane: the page this fire appends into.
    pub write_page: Tensor,

    /// `i32` per lane: the in-page slot the append starts at.
    pub write_offset: Tensor,

    /// `u8` per row: the CUDA-graph padding mask — rows past the live count
    /// replay with this cleared.
    pub row_valid: Tensor,

    /// `bf16` per-page key envelopes (min side), when the scheme keeps them.
    pub env_min: Tensor,

    /// Envelope max side.
    pub env_max: Tensor,

    pub has_envelopes: bool,

    /// Tokens per page.
    pub page_size: i32,

    /// Elements from one in-page token row to the next.
    pub seq_stride: i64,

    /// Elements from one head's plane to the next.
    pub head_stride: i64,

    /// The page layout enumerator the device text reads.
    pub layout: i32,

    /// The quantization scheme enumerator the device text reads.
    pub scheme_byte: i32,

    /// Elements per quantization block.
    pub block_size: i32,

    pub max_pages_per_request: i32,

    pub pages_in_batch: i32,
}

/// The recurrent-state pool (ssm/linear-attention) — the old
/// `RecurrentView`, dtype-erased and trimmed to the fields this plane's
/// launches actually bind. Slot-addressed, not paged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrentPool {
    /// The recurrent-state slab, one bank per slot.
    pub slab: Tensor,

    /// `i32` per lane: which state slot each request owns.
    pub slot_ids: Tensor,

    /// Elements from one slot's bank to the next.
    pub slot_stride_elems: i64,

    /// The rolling convolution-state slab.
    pub conv_slab: Tensor,

    /// Elements from one slot's convolution state to the next.
    pub conv_stride: i64,
}
