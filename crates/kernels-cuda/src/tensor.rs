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

use dtype::Dtype;

use crate::jit::ArgValue;

/// A dtype-erased view of one device buffer: `rows x width` dense elements
/// behind a `CUdeviceptr`-shaped address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor {
    /// The device address, resolved by the engine (arena offset, weight
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

    /// **The seat nobody filled**, as a handle rather than an `Option`.
    ///
    /// A null address is what a device text reads as `nullptr`, and every
    /// optional pointer seat in `attn/ssm.cuh` already spells its absence that
    /// way (`commit_len != nullptr`, `row_persists`'s `mask == nullptr`). So
    /// an unbound seat is this handle and the marshalling is unchanged —
    /// wrapping it in an `Option` would put a branch in every entry to
    /// re-derive the zero the kernel already tests for.
    pub const ABSENT: Tensor = Tensor::new(0, 0, 0, Dtype::U8);

    /// Is this the seat nobody filled?
    #[must_use]
    pub const fn is_absent(&self) -> bool {
        self.ptr == 0
    }
}

/// A fire-aligned value paired with the fire's shared boundaries (design §5).
///
/// `indptr` is the one indptr of the whole fire — `i32`, `[lanes + 1]` — and
/// every fire-aligned value can be viewed through it. Boundary-oblivious
/// kernels take `data` alone; boundary-aware entries (`prefill`,
/// `*_chunked`) take the pair, already built by the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaggedTensor {
    pub data: Tensor,

    /// `i32`, `[lanes + 1]`, shared across the fire.
    pub indptr: Tensor,
}

/// The paged kv cache a fire reads and appends into — the old `PagedKvView`,
/// dtype-erased. Storage plus the page geometry the engine resolved from
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

    /// **Does this launch fold its boundary into the state at all?** (alto
    /// design §6.)
    ///
    /// `true` is the plain path: the recurrence's end-of-sequence writeback
    /// IS the fold. `false` is the buffered scatter — the pass computes its
    /// outputs and leaves the folded state exactly where it was, which is
    /// what makes a rejected draft pure host bookkeeping.
    pub write_state: bool,

    /// **The per-REQUEST fold predicate** — `u8`, `[requests]`, or the zero
    /// tensor for "every request folds".
    ///
    /// `attn/ssm.cuh`'s `row_persists(mask, r)` takes `r = blockIdx` over
    /// REQUESTS, not over token rows, so this is one byte per lane of the
    /// window the launch runs in — sliced by `lane_offset` exactly as
    /// [`RecurrentPool::slot_ids`] is. A refused pass's byte is zero and its
    /// fold is a no-op, which is what makes the fold a predicated commit like
    /// every other durable advance (design §1 article 3).
    pub write_state_mask: Tensor,

    /// **Where the accepted prefix ends** — `i32`, `[requests]`, or the zero
    /// tensor for "the whole request".
    ///
    /// The chunked scans truncate request `r` at `commit_len[r]` tokens
    /// (`ssm.cuh:127-128, 197-198, 1631-1632`), which is what turns a replay
    /// of a buffered window into a fold of the accepted prefix of it. Sliced
    /// per window beside the mask.
    ///
    /// **A LENGTH AND NOTHING ELSE, SINCE F3b.** The fla scan used to read
    /// `commit_len != nullptr` as a second thing — a different bf16 rounding
    /// of the decay — so binding a seat that truncated nothing still moved
    /// bytes. The rounding is [`RecurrentPool::fused_decay`] now, and this
    /// seat means only what it says.
    pub commit_len: Tensor,

    /// **Where the segment this launch owns BEGINS** — `i32`, `[requests]`,
    /// or the zero tensor for "at the row's own first token" (alto design
    /// §6's 2R interior split, wave F3b).
    ///
    /// [`RecurrentPool::commit_len`]'s mirror: that one cuts a request at the
    /// back, this one cuts it at the front, and a row whose fold boundary
    /// falls INSIDE its own tokens is run as two ordered launches over the
    /// same rows — the head `[0, n)` with the length seat and the fold, the
    /// tail `[n, rows)` with this one and no fold. Never bound together with
    /// `commit_len` on one launch; the front cut is applied first, so what
    /// `commit_len` would count is the segment and not the row.
    ///
    /// Only the arms that can carry an interior boundary have a seat for it
    /// (the chunked conv and the fla delta scan); every other arm refuses a
    /// bound one by name rather than folding a segment it cannot address.
    pub begin_at: Tensor,

    /// **How the decay is rounded** — dev's `single_round` (`ssm.cuh:1660`),
    /// promoted out of `commit_len`'s null test into an argument of its own.
    ///
    /// `false` is the fold's own policy and the default everywhere in this
    /// tree: the decayed state is rounded to bf16 before the update reads it,
    /// which is what the plain fold-per-token forward does. `true` folds the
    /// decay into the update instead — the same arithmetic, different bits.
    ///
    /// **IT IS A POLICY AND NOT A CONSEQUENCE**, which is the whole of the
    /// F3b finding: while it rode `commit_len`'s presence, a replay that
    /// accepted its whole window and a fold over the same tokens differed by
    /// 3,115,437 of 10,321,884 state bytes for no reason but the rounding,
    /// and a truncated fold could not be compared against a shorter buffer at
    /// all.
    pub fused_decay: bool,
}


