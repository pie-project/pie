//! Handle vocabulary: runtime-typed, `Copy`, never dereferenced by the host.
//! `&`/`&mut` on a `Tensor` is write intent, not borrow discipline.

use dtype::Dtype;

use crate::jit::ArgValue;

/// A dtype-erased view of one device buffer: `rows x width` dense elements
/// behind a `CUdeviceptr`-shaped address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor {
    /// Device address, resolved by the engine before this handle reaches
    /// an entry.
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

    /// An unbound seat, as a handle rather than an `Option`: a null address
    /// is what device text reads as `nullptr`.
    pub const ABSENT: Tensor = Tensor::new(0, 0, 0, Dtype::U8);

    /// Is this the seat nobody filled?
    #[must_use]
    pub const fn is_absent(&self) -> bool {
        self.ptr == 0
    }
}

/// A fire-aligned value paired with the fire's shared boundaries.
/// Boundary-oblivious kernels take `data` alone; boundary-aware entries take
/// the pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaggedTensor {
    pub data: Tensor,

    /// `i32`, `[lanes + 1]`, shared across the fire.
    pub indptr: Tensor,
}

/// The paged kv cache a fire reads and appends into: storage plus the page
/// geometry the engine resolved from declared inputs.
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

/// The recurrent-state pool (ssm/linear-attention). Slot-addressed, not
/// paged.
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

    /// Whether this launch folds its boundary into the state. `true`: the
    /// end-of-sequence writeback is the fold. `false`: the pass computes
    /// outputs and leaves the folded state untouched (a rejected draft).
    pub write_state: bool,

    /// Per-request fold predicate — `u8`, `[requests]`, or the zero tensor
    /// for "every request folds". A refused pass's byte is zero, so its
    /// fold is a no-op.
    pub write_state_mask: Tensor,

    /// Where the accepted prefix ends — `i32`, `[requests]`, or the zero
    /// tensor for "the whole request". Truncates request `r`'s replay at
    /// `commit_len[r]` tokens; a length and nothing else (decay rounding is
    /// [`RecurrentPool::fused_decay`]).
    pub commit_len: Tensor,

    /// Where the segment this launch owns begins — `i32`, `[requests]`, or
    /// zero for "at the row's own first token". Mirrors `commit_len`
    /// (which cuts at the back, this at the front); a row whose fold
    /// boundary falls inside its own tokens runs as two ordered launches,
    /// never both seats on one launch.
    pub begin_at: Tensor,

    /// How the decay is rounded. `false` (default): the decayed state is
    /// rounded to bf16 before the update reads it. `true`: folds the decay
    /// into the update instead — same arithmetic, different bits.
    pub fused_decay: bool,
}


