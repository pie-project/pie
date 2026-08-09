//! The recurrent state cache: allocation, slot resets and slot copies.
//!
//! Port of `driver-cuda/csrc/src/store/recurrent_state_cache.{hpp,cpp}`.
//! [`crate::layout::recurrent_layout`] already holds the stride and offset
//! arithmetic; this module is the object that owns the four buffers and turns
//! runtime requests into stream operations.
//!
//! Structured the same way as [`crate::pools::kv_cache`]: every device
//! operation is produced as a **value** first -- a [`StateOp`] -- and executed
//! second. A `cudaMemset2DAsync` with the wrong pitch does not fail, it zeroes
//! another layer's state and shows up thousands of tokens later as a quality
//! regression, so the descriptors are what the differential oracle checks.
//!
//! Four buffers, all pooled and flat:
//!
//! | buffer | shape | dtype |
//! |---|---|---|
//! | conv | `[linear_layers, slots, conv_kernel * conv_dim]` | `u16` |
//! | recurrent | `[linear_layers, slots, v_heads * k_dim * v_dim]` | `bf16`/`f32` |
//! | mtp pending hidden | `[slots, hidden_size]` | `u16` |
//! | verify stash | `[linear_layers, max_tokens, hidden]` | `u16` |
//! | rs buffer pool | `[linear_layers, slots, page_tokens, hidden]` | `u16` |
//!
//! The first two are `[layer, slot, state]` and not `[slot, layer, state]`
//! precisely so a whole-slot reset is one strided 2D operation per buffer
//! rather than one call per layer: slot `s` appears at a constant offset with
//! a constant pitch, which is exactly `cudaMemset2DAsync`'s argument shape.

use crate::error::{Error, Result};
use crate::layout::recurrent_layout::{RecurrentShape, RecurrentStateLayout};

/// Which pooled buffer an operation addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Buffer {
    /// The causal-conv window, `[linear_layers, slots, conv_kernel*conv_dim]`.
    Conv,
    /// The linear-attention running state.
    Recurrent,
    /// The MTP pending-hidden row, `[slots, hidden_size]`.
    MtpHidden,
}

impl Buffer {
    /// The short tag used in transcripts.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Conv => "conv",
            Self::Recurrent => "rec",
            Self::MtpHidden => "mtp",
        }
    }
}

/// One device operation the cache wants performed.
///
/// An enum rather than an immediate call because these are the whole
/// observable behaviour of `reset`, `reset_slot` and the two copies. Nothing
/// about a `cudaMemset2DAsync` is checkable after the fact, so it is checked
/// before the fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateOp {
    /// Contiguous zero fill: `cudaMemsetAsync(base + offset, 0, len)`.
    Memset {
        /// Which buffer.
        buffer: Buffer,
        /// Byte offset from that buffer's base.
        offset: u64,
        /// Bytes to zero.
        len: u64,
    },
    /// Strided zero fill: `cudaMemset2DAsync(base + offset, pitch, 0, width,
    /// rows)`. One row per linear layer.
    Memset2D {
        /// Which buffer.
        buffer: Buffer,
        /// Byte offset of the first row.
        offset: u64,
        /// Bytes between consecutive rows.
        pitch: u64,
        /// Bytes zeroed within each row.
        width: u64,
        /// Row count.
        rows: u64,
    },
    /// Contiguous device-to-device copy.
    Memcpy {
        /// Which buffer.
        buffer: Buffer,
        /// Destination byte offset.
        dst: u64,
        /// Source byte offset.
        src: u64,
        /// Bytes copied.
        len: u64,
    },
    /// Strided device-to-device copy, one row per linear layer.
    Memcpy2D {
        /// Which buffer.
        buffer: Buffer,
        /// Destination byte offset of the first row.
        dst: u64,
        /// Source byte offset of the first row.
        src: u64,
        /// Bytes between consecutive rows, both sides.
        pitch: u64,
        /// Bytes copied within each row.
        width: u64,
        /// Row count.
        rows: u64,
    },
    /// The `zero_slots_if_fresh` kernel: a device-predicated scatter reset.
    ///
    /// Distinct from [`Self::Memset2D`] because the slot ids live in device
    /// memory -- the host cannot know which rows will be touched, so it cannot
    /// be expressed as a fixed offset.
    ZeroSlotsIfFresh {
        /// Which buffer.
        buffer: Buffer,
        /// Bytes per slot.
        slot_bytes: u64,
        /// Bytes between consecutive rows.
        row_pitch: u64,
        /// Row count.
        rows: u64,
        /// Number of candidate requests read from the device arrays.
        request_count: u64,
    },
}

/// A recurrent state cache: geometry, buffer sizes and the optional tiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentStateCache {
    layout: RecurrentStateLayout,
    verify_stash: Option<StashDims>,
    rs_buffer_pool: Option<PoolDims>,
    verify_frozen: bool,
}

/// Dimensions of the frozen-verify hidden stash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StashDims {
    /// Token capacity per linear layer, after the `PIE_RS_STASH_TOKENS` cap.
    pub max_tokens: u32,
    /// Row width in elements.
    pub hidden: u32,
}

/// Dimensions of the persistent buffered-activation pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolDims {
    /// Tokens per buffered slab.
    pub page_tokens: u32,
    /// Row width in elements.
    pub hidden: u32,
    /// Buffered slabs.
    pub num_slots: u32,
}

/// Bytes per `u16` element -- conv, MTP hidden, stash and pool are all `u16`.
const U16: u64 = 2;

/// The `PIE_RS_STASH_TOKENS` override, parsed the way `std::atoi` parses.
///
/// `atoi` returns 0 for anything it cannot read, including an empty string and
/// trailing garbage after a valid prefix -- and 0 is exactly the value
/// `configure_verify_hidden_stash` ignores, so a typo silently leaves the
/// stash at its full prefill width rather than failing.
#[must_use]
pub fn stash_tokens_cap() -> Option<i32> {
    let raw = std::env::var("PIE_RS_STASH_TOKENS").ok()?;
    let t = raw.trim_start();
    let (sign, digits) = match t.strip_prefix('-') {
        Some(rest) => (-1i64, rest),
        None => (1i64, t.strip_prefix(0x2b as char).unwrap_or(t)),
    };
    let mut v: i64 = 0;
    for c in digits.chars() {
        let Some(d) = c.to_digit(10) else { break };
        v = (v * 10 + i64::from(d)).min(i64::from(i32::MAX) + 1);
    }
    Some(i32::try_from(sign * v).unwrap_or(i32::MAX))
}

/// Whether recurrent state is stored as `bf16`.
///
/// **Always true.** The C++ reads
///
/// ```text
/// const char* v = nullptr;
/// if (v == nullptr) return true;
/// ```
///
/// -- the `std::getenv("PIE_QWEN35_RS_STATE_DTYPE")` that used to initialise
/// `v` has been deleted, so the whole spelling table below it (`fp32`,
/// `float32`, `bf16`, `bfloat16`, leading `0`/`1`) is unreachable and the
/// switch documented in the header no longer exists.
///
/// Ported as a constant rather than as a re-implemented env lookup, because
/// re-adding the lookup would change behaviour: several downstream things --
/// the fp32 `recurrent_states_` buffer, the `recurrent_state()` accessor, and
/// `allocate_bf16_recurrent`'s re-allocation path -- have been dead long
/// enough that nothing has been exercising them. This is the one place that
/// says so.
#[must_use]
pub const fn recurrent_state_bf16_default() -> bool {
    true
}

impl RecurrentStateCache {
    /// Plan a cache.
    ///
    /// `hidden_size` clamps up at 0 and `max_slots` clamps up at 1, both
    /// matching the C++. A zero `hidden_size` means "no MTP tier" rather than
    /// an error, because the non-MTP models simply do not pass one.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        layer_is_linear: &[bool],
        conv_dim: u32,
        conv_kernel: u32,
        v_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
        hidden_size: i32,
        max_slots: i32,
    ) -> Self {
        let hidden_size = hidden_size.max(0).unsigned_abs();
        // Not clamped up to 1 here: `RecurrentStateLayout::new` owns that, and
        // clamping in both places would leave neither one load-bearing.
        let max_slots = max_slots.max(0).unsigned_abs();
        Self {
            layout: RecurrentStateLayout::new(
                layer_is_linear,
                RecurrentShape {
                    conv_dim,
                    conv_kernel,
                    v_heads,
                    head_k_dim,
                    head_v_dim,
                    hidden_size,
                    max_slots,
                    recurrent_is_bf16: recurrent_state_bf16_default(),
                },
            ),
            verify_stash: None,
            rs_buffer_pool: None,
            verify_frozen: false,
        }
    }

    /// Plan a cache whose recurrent state is `bf16` regardless of the default.
    ///
    /// Nemotron-H's Mamba2 state is defined in activation dtype, and fp32
    /// storage is too large at serving request counts.
    ///
    /// The C++ implements this by calling `allocate` and then *re-allocating*
    /// the recurrent slab if the default came back fp32 -- which, since
    /// [`recurrent_state_bf16_default`] is now a constant `true`, never
    /// happens. Here the flag is simply forced, so no allocation is made and
    /// discarded. The observable result is identical; what differs is that the
    /// C++ would issue a second `cudaMalloc` and a second full `reset` if the
    /// default ever changed back.
    #[must_use]
    pub fn allocate_bf16_recurrent(
        layer_is_linear: &[bool],
        conv_dim: u32,
        conv_kernel: u32,
        v_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
        max_slots: i32,
    ) -> Self {
        let mut c = Self::allocate(
            layer_is_linear,
            conv_dim,
            conv_kernel,
            v_heads,
            head_k_dim,
            head_v_dim,
            0,
            max_slots,
        );
        c.layout.force_recurrent_bf16();
        c
    }

    /// The underlying stride and offset arithmetic.
    #[must_use]
    pub const fn layout(&self) -> &RecurrentStateLayout {
        &self.layout
    }

    /// Whether the verify forward is currently persisting nothing.
    #[must_use]
    pub const fn verify_frozen(&self) -> bool {
        self.verify_frozen
    }

    /// Set frozen-verify mode.
    pub const fn set_verify_frozen(&mut self, frozen: bool) {
        self.verify_frozen = frozen;
    }

    /// Whether the MTP pending-hidden buffer exists.
    ///
    /// The C++ tests `mtp_pending_hidden_.data() != nullptr && hidden_size_ >
    /// 0`. Both halves are the same condition -- a `DeviceBuffer` of zero
    /// elements never allocates -- but the redundancy is preserved in one
    /// place here so the call sites read as one question.
    #[must_use]
    pub const fn has_mtp_hidden(&self) -> bool {
        self.layout.hidden_size() > 0
    }

    /// Whether the frozen-verify hidden stash is configured.
    #[must_use]
    pub const fn verify_hidden_stash_enabled(&self) -> bool {
        self.verify_stash.is_some()
    }

    /// Channels in the causal-conv state — `KvCache`-style forwarding so
    /// the generated bodies' `state_cache.conv_dim()` reads translate
    /// one-to-one; the layout remains the single statement of the shape.
    #[must_use]
    pub const fn conv_dim(&self) -> u32 {
        self.layout.conv_dim()
    }

    /// Taps in the causal-conv kernel.
    #[must_use]
    pub const fn conv_kernel(&self) -> u32 {
        self.layout.conv_kernel()
    }

    /// Value heads in the linear-attention state.
    #[must_use]
    pub const fn v_heads(&self) -> u32 {
        self.layout.v_heads()
    }

    /// Key dimension per head.
    #[must_use]
    pub const fn head_k_dim(&self) -> u32 {
        self.layout.head_k_dim()
    }

    /// Value dimension per head.
    #[must_use]
    pub const fn head_v_dim(&self) -> u32 {
        self.layout.head_v_dim()
    }

    /// Width of the MTP pending-hidden row, or 0 when there is none.
    #[must_use]
    pub const fn hidden_size(&self) -> u32 {
        self.layout.hidden_size()
    }

    /// Total layers, linear and full-attention alike.
    #[must_use]
    pub fn num_layers(&self) -> u32 {
        self.layout.num_layers()
    }

    /// Committed slots per linear layer.
    #[must_use]
    pub const fn max_slots(&self) -> u32 {
        self.layout.max_slots()
    }

    /// Is the recurrent state stored as bf16 rather than f32?
    #[must_use]
    pub const fn recurrent_state_bf16(&self) -> bool {
        self.layout.recurrent_is_bf16()
    }

    /// Bytes between consecutive slots of `conv_state`.
    #[must_use]
    pub const fn conv_slot_stride_bytes(&self) -> u64 {
        self.layout.conv_slot_stride_bytes()
    }

    /// Elements between consecutive slots of `recurrent_state` — the C++
    /// spells this `recurrent_slot_stride_floats`, a name that predates the
    /// bf16 storage; the elements have not been floats since.
    #[must_use]
    pub const fn recurrent_slot_stride_floats(&self) -> u64 {
        self.layout.recurrent_slot_stride_elems()
    }

    /// Bytes between consecutive slots of `recurrent_state`.
    #[must_use]
    pub const fn recurrent_slot_stride_bytes(&self) -> u64 {
        self.layout.recurrent_slot_stride_bytes()
    }

    /// The verify stash's token capacity, `0` when unconfigured — the C++
    /// field's resting value.
    #[must_use]
    pub const fn verify_stash_max_tokens(&self) -> u32 {
        match self.verify_stash {
            Some(d) => d.max_tokens,
            None => 0,
        }
    }

    /// The verify stash's row width, `0` when unconfigured.
    #[must_use]
    pub const fn verify_stash_hidden(&self) -> u32 {
        match self.verify_stash {
            Some(d) => d.hidden,
            None => 0,
        }
    }

    /// The stash dimensions, if configured.
    #[must_use]
    pub const fn verify_stash(&self) -> Option<StashDims> {
        self.verify_stash
    }

    /// Whether the buffered-activation pool is configured.
    #[must_use]
    pub const fn rs_buffer_pool_enabled(&self) -> bool {
        self.rs_buffer_pool.is_some()
    }

    /// The pool dimensions, if configured.
    #[must_use]
    pub const fn rs_buffer_pool(&self) -> Option<PoolDims> {
        self.rs_buffer_pool
    }

    /// Zero every slot of every linear layer, plus the MTP tier.
    ///
    /// Issued at the start of each fresh prefill: this driver's batching model
    /// carries no recurrent state across prefills.
    #[must_use]
    pub fn reset(&self) -> Vec<StateOp> {
        let mut ops = Vec::new();
        let slots = u64::from(self.layout.max_slots());
        let layers = u64::from(self.layout.num_linear_layers());
        if layers > 0 {
            ops.push(StateOp::Memset {
                buffer: Buffer::Conv,
                offset: 0,
                len: self.layout.conv_slot_stride_bytes() * slots * layers,
            });
            ops.push(StateOp::Memset {
                buffer: Buffer::Recurrent,
                offset: 0,
                len: self.layout.recurrent_slot_stride_bytes() * slots * layers,
            });
        }
        if self.has_mtp_hidden() {
            ops.push(StateOp::Memset {
                buffer: Buffer::MtpHidden,
                offset: 0,
                len: u64::from(self.layout.hidden_size()) * slots * U16,
            });
        }
        ops
    }

    /// Zero one slot across every linear layer, plus its MTP row.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] when `slot` is not a valid slot id.
    pub fn reset_slot(&self, slot: i32) -> Result<Vec<StateOp>> {
        self.check_slot(slot, "RecurrentStateCache::reset_slot")?;
        let mut ops = Vec::new();
        let slots = u64::from(self.layout.max_slots());
        let layers = u64::from(self.layout.num_linear_layers());
        if layers > 0 {
            for (buffer, stride) in self.state_strides() {
                ops.push(StateOp::Memset2D {
                    buffer,
                    offset: u64::from(slot.unsigned_abs()) * stride,
                    pitch: stride * slots,
                    width: stride,
                    rows: layers,
                });
            }
        }
        if self.has_mtp_hidden() {
            let hidden = u64::from(self.layout.hidden_size());
            ops.push(StateOp::Memset {
                buffer: Buffer::MtpHidden,
                offset: u64::from(slot.unsigned_abs()) * hidden * U16,
                len: hidden * U16,
            });
        }
        Ok(ops)
    }

    /// Device-predicated reset for the rows a fixed envelope marks fresh.
    ///
    /// Returns no operations at all when `request_count == 0`, matching the
    /// C++'s early return. The kernel is not launched with a zero grid; it is
    /// not launched.
    #[must_use]
    pub fn reset_slots_if_fresh(
        &self,
        slot_ids: Option<&[i32]>,
        is_fresh: Option<&[u8]>,
        request_count: i32,
    ) -> Vec<StateOp> {
        if slot_ids.is_none() || is_fresh.is_none() || request_count <= 0 {
            return Vec::new();
        }
        let request_count = request_count.unsigned_abs();
        let mut ops = Vec::new();
        let slots = u64::from(self.layout.max_slots());
        let layers = u64::from(self.layout.num_linear_layers());
        if layers > 0 {
            for (buffer, stride) in self.state_strides() {
                ops.push(StateOp::ZeroSlotsIfFresh {
                    buffer,
                    slot_bytes: stride,
                    row_pitch: stride * slots,
                    rows: layers,
                    request_count: u64::from(request_count),
                });
            }
        }
        if self.has_mtp_hidden() {
            let hidden_bytes = u64::from(self.layout.hidden_size()) * U16;
            // One row, not `layers`: the MTP tier is `[slots, hidden]` with no
            // layer axis at all.
            ops.push(StateOp::ZeroSlotsIfFresh {
                buffer: Buffer::MtpHidden,
                slot_bytes: hidden_bytes,
                row_pitch: hidden_bytes * slots,
                rows: 1,
                request_count: u64::from(request_count),
            });
        }
        ops
    }

    /// Copy one slot to another across every linear layer, plus the MTP row.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] when either slot id is invalid.
    pub fn copy_slot_d2d(&self, src_slot: i32, dst_slot: i32) -> Result<Vec<StateOp>> {
        self.check_slot(src_slot, "RecurrentStateCache::copy_slot_d2d")?;
        self.check_slot(dst_slot, "RecurrentStateCache::copy_slot_d2d")?;
        let mut ops = self.slot_copy_ops(src_slot, dst_slot);
        if src_slot != dst_slot && self.has_mtp_hidden() {
            let hidden = u64::from(self.layout.hidden_size());
            ops.push(StateOp::Memcpy {
                buffer: Buffer::MtpHidden,
                dst: u64::from(dst_slot.unsigned_abs()) * hidden * U16,
                src: u64::from(src_slot.unsigned_abs()) * hidden * U16,
                len: hidden * U16,
            });
        }
        Ok(ops)
    }

    /// Copy only the conv and recurrent slabs, leaving MTP pending-hidden
    /// alone.
    ///
    /// The asymmetry is deliberate in the C++ and preserved here: a
    /// speculative verifier rollback restores recurrent state to the accepted
    /// prefix, but the MTP state was already rebuilt from exactly those
    /// accepted tokens, so copying it would overwrite the newer value with an
    /// older one.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] when either slot id is invalid.
    pub fn copy_linear_state_slot_d2d(&self, src_slot: i32, dst_slot: i32) -> Result<Vec<StateOp>> {
        self.check_slot(src_slot, "RecurrentStateCache::copy_linear_state_slot_d2d")?;
        self.check_slot(dst_slot, "RecurrentStateCache::copy_linear_state_slot_d2d")?;
        Ok(self.slot_copy_ops(src_slot, dst_slot))
    }

    fn slot_copy_ops(&self, src_slot: i32, dst_slot: i32) -> Vec<StateOp> {
        let slots = u64::from(self.layout.max_slots());
        let layers = u64::from(self.layout.num_linear_layers());
        if src_slot == dst_slot || layers == 0 {
            return Vec::new();
        }
        self.state_strides()
            .into_iter()
            .map(|(buffer, stride)| StateOp::Memcpy2D {
                buffer,
                dst: u64::from(dst_slot.unsigned_abs()) * stride,
                src: u64::from(src_slot.unsigned_abs()) * stride,
                pitch: stride * slots,
                width: stride,
                rows: layers,
            })
            .collect()
    }

    /// Configure the frozen-verify hidden stash.
    ///
    /// A no-op when any dimension is zero or the model has no linear layers,
    /// matching the C++'s early return -- so a caller cannot tell a rejected
    /// configuration from one it never made.
    ///
    /// `stash_tokens_cap` is the `PIE_RS_STASH_TOKENS` override, which only
    /// ever lowers `max_tokens`. It exists because `max_tokens` arrives as a
    /// *prefill* width (8192) while the stash is only written on a
    /// frozen-verify fire, whose width is draft-tokens x requests (~256). At
    /// 8192 x 10336 x 48 layers x 2 B the untrimmed stash is 7752 MiB, the
    /// largest single item in the state arena -- and the state arena is
    /// charged in full at every frame commit, even though prefill never
    /// touches this buffer.
    pub fn configure_verify_hidden_stash(
        &mut self,
        max_tokens: u32,
        hidden: u32,
        stash_tokens_cap: Option<i32>,
    ) {
        if max_tokens == 0 || hidden == 0 || self.layout.num_linear_layers() == 0 {
            return;
        }
        let mut max_tokens = max_tokens;
        if let Some(want) = stash_tokens_cap
            && want > 0
            && want.unsigned_abs() < max_tokens
        {
            max_tokens = want.unsigned_abs();
        }
        self.verify_stash = Some(StashDims { max_tokens, hidden });
    }

    /// Byte offset of one linear layer's stash region.
    ///
    /// `None` when the stash is not configured or `linear_idx` is out of
    /// range. Indexed by the **dense** linear index, not the transformer layer
    /// index -- the caller has already compacted.
    #[must_use]
    pub fn verify_hidden_stash_layer(&self, linear_idx: u32) -> Option<u64> {
        let dims = self.verify_stash?;
        if linear_idx >= self.layout.num_linear_layers() {
            return None;
        }
        Some(u64::from(linear_idx) * u64::from(dims.max_tokens) * u64::from(dims.hidden) * U16)
    }

    /// Total bytes of the verify hidden stash.
    #[must_use]
    pub fn verify_hidden_stash_bytes(&self) -> u64 {
        self.verify_stash.map_or(0, |d| {
            u64::from(d.max_tokens)
                * u64::from(d.hidden)
                * U16
                * u64::from(self.layout.num_linear_layers())
        })
    }

    /// Configure the persistent buffered-activation pool.
    ///
    /// A no-op when any dimension is zero or the model has no linear layers.
    pub fn configure_rs_buffer_pool(&mut self, page_tokens: u32, hidden: u32, num_slots: u32) {
        if page_tokens == 0 || hidden == 0 || num_slots == 0 || self.layout.num_linear_layers() == 0
        {
            return;
        }
        self.rs_buffer_pool = Some(PoolDims {
            page_tokens,
            hidden,
            num_slots,
        });
    }

    /// Byte offset of one buffered slab.
    ///
    /// `None` when the pool is not configured or either index is out of range.
    /// `slot` indexes the **pool's own** slot count, which is independent of
    /// `max_slots`: the runtime's arena hands out `RsSlab` object ids, and
    /// there are as many of those as there are buffered slabs, not as there
    /// are concurrent requests.
    #[must_use]
    pub fn rs_buffer_slab(&self, linear_idx: u32, slot: u32) -> Option<u64> {
        let dims = self.rs_buffer_pool?;
        if linear_idx >= self.layout.num_linear_layers() || slot >= dims.num_slots {
            return None;
        }
        let per_slot = u64::from(dims.page_tokens) * u64::from(dims.hidden) * U16;
        Some(
            u64::from(linear_idx) * per_slot * u64::from(dims.num_slots)
                + u64::from(slot) * per_slot,
        )
    }

    /// Total bytes of the buffered-activation pool.
    #[must_use]
    pub fn rs_buffer_pool_bytes(&self) -> u64 {
        self.rs_buffer_pool.map_or(0, |d| {
            u64::from(d.page_tokens)
                * u64::from(d.hidden)
                * U16
                * u64::from(d.num_slots)
                * u64::from(self.layout.num_linear_layers())
        })
    }

    /// Byte offset of one layer/slot's conv state within the conv buffer.
    ///
    /// `Ok(None)` for a full-attention layer, which has no recurrent state and
    /// whose C++ counterpart returns a null pointer -- a legitimate answer
    /// that callers branch on, not a failure.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] when `slot` or `layer` is out of range. The **slot**
    /// is checked first; asking for layer -1 of slot -1 reports the slot.
    pub fn conv_state(&self, layer: i32, slot: i32) -> Result<Option<u64>> {
        self.checked_index(layer, slot, "RecurrentStateCache::conv_state")
            .map(|idx| {
                idx.map(|(linear_idx, slot)| {
                    let stride = self.layout.conv_slot_stride_bytes();
                    (u64::from(linear_idx) * u64::from(self.layout.max_slots()) + u64::from(slot))
                        * stride
                })
            })
    }

    /// Byte offset of one layer/slot's recurrent state within its buffer.
    ///
    /// Named for the C++ `recurrent_state_raw`, but the C++ reports itself as
    /// `recurrent_state` in both of its exceptions -- the `_raw` suffix was
    /// added later and the messages were not updated. Preserved verbatim.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] when `slot` or `layer` is out of range.
    pub fn recurrent_state_raw(&self, layer: i32, slot: i32) -> Result<Option<u64>> {
        self.checked_index(layer, slot, "RecurrentStateCache::recurrent_state")
            .map(|idx| {
                idx.map(|(linear_idx, slot)| {
                    let stride = self.layout.recurrent_slot_stride_bytes();
                    (u64::from(linear_idx) * u64::from(self.layout.max_slots()) + u64::from(slot))
                        * stride
                })
            })
    }

    /// The `f32` view of the recurrent state.
    ///
    /// **Always an error.** The C++ checks `recurrent_state_bf16_` before it
    /// checks anything else, and that flag is now unconditionally true (see
    /// [`recurrent_state_bf16_default`]) -- so this accessor cannot succeed
    /// for any argument, valid or not. Kept, and kept failing, because callers
    /// that still reach for a `float*` should say so loudly rather than get a
    /// pointer to `bf16` and read garbage.
    ///
    /// # Errors
    ///
    /// Always, while the recurrent state is `bf16`; otherwise as
    /// [`Self::recurrent_state_raw`].
    pub fn recurrent_state_f32(&self, layer: i32, slot: i32) -> Result<Option<u64>> {
        if self.layout.recurrent_is_bf16() {
            return Err(Error::invalid(
                "RecurrentStateCache::recurrent_state",
                "recurrent state is bf16",
            ));
        }
        self.recurrent_state_raw(layer, slot)
    }

    /// Byte offset of one slot's MTP pending-hidden row.
    ///
    /// `Ok(None)` when the model has no MTP tier.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] when `slot` is out of range.
    pub fn mtp_pending_hidden(&self, slot: i32) -> Result<Option<u64>> {
        if slot < 0 || slot.unsigned_abs() >= self.layout.max_slots() {
            return Err(Error::invalid(
                "RecurrentStateCache::mtp_pending_hidden",
                "slot out of range",
            ));
        }
        if !self.has_mtp_hidden() {
            return Ok(None);
        }
        Ok(Some(
            u64::from(slot.unsigned_abs()) * u64::from(self.layout.hidden_size()) * U16,
        ))
    }

    /// Shared prologue of the two state accessors: validate, then compact.
    fn checked_index(
        &self,
        layer: i32,
        slot: i32,
        who: &'static str,
    ) -> Result<Option<(u32, u32)>> {
        if slot < 0 || slot.unsigned_abs() >= self.layout.max_slots() {
            return Err(Error::invalid(who, "slot out of range"));
        }
        if layer < 0 || layer.unsigned_abs() >= self.layout.num_layers() {
            return Err(Error::invalid(who, "layer out of range"));
        }
        Ok(self
            .layout
            .linear_index(layer.unsigned_abs())
            .map(|idx| (idx, slot.unsigned_abs())))
    }

    /// The two per-slot state strides, in the order the C++ issues them.
    fn state_strides(&self) -> [(Buffer, u64); 2] {
        [
            (Buffer::Conv, self.layout.conv_slot_stride_bytes()),
            (Buffer::Recurrent, self.layout.recurrent_slot_stride_bytes()),
        ]
    }

    /// `who` is the name the C++ puts in the `std::out_of_range`, which is
    /// not always the name of the method that throws: `recurrent_state_raw`
    /// reports itself as `recurrent_state`, and `copy_slot_d2d` and
    /// `copy_linear_state_slot_d2d` report separately. Passing it in keeps
    /// those quirks at the call sites where they are visible.
    fn check_slot(&self, slot: i32, who: &'static str) -> Result<()> {
        if slot < 0 || slot.unsigned_abs() >= self.layout.max_slots() {
            return Err(Error::invalid(who, "slot out of range"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> RecurrentStateCache {
        // 6 layers, 3 of them linear.
        RecurrentStateCache::allocate(
            &[true, false, true, false, true, false],
            128,
            4,
            2,
            8,
            16,
            64,
            4,
        )
    }

    #[test]
    fn the_env_switch_documented_in_the_header_no_longer_exists() {
        assert!(recurrent_state_bf16_default());
        assert!(cache().layout().recurrent_is_bf16());
    }

    #[test]
    fn a_full_reset_is_three_contiguous_memsets() {
        let c = cache();
        let ops = c.reset();
        assert_eq!(ops.len(), 3);
        // conv: 4 taps * 128 ch * 2 B = 1024 per slot, 4 slots, 3 layers.
        assert_eq!(
            ops[0],
            StateOp::Memset {
                buffer: Buffer::Conv,
                offset: 0,
                len: 1024 * 4 * 3,
            }
        );
        // recurrent: 2*8*16 = 256 elems * 2 B (bf16) = 512 per slot.
        assert_eq!(
            ops[1],
            StateOp::Memset {
                buffer: Buffer::Recurrent,
                offset: 0,
                len: 512 * 4 * 3,
            }
        );
        assert_eq!(
            ops[2],
            StateOp::Memset {
                buffer: Buffer::MtpHidden,
                offset: 0,
                len: 64 * 4 * 2,
            }
        );
    }

    #[test]
    fn a_slot_reset_strides_by_the_whole_slot_run_of_one_layer() {
        let ops = cache().reset_slot(2).unwrap();
        assert_eq!(
            ops[0],
            StateOp::Memset2D {
                buffer: Buffer::Conv,
                offset: 2 * 1024,
                pitch: 1024 * 4,
                width: 1024,
                rows: 3,
            }
        );
    }

    #[test]
    fn an_out_of_range_slot_is_rejected_rather_than_clamped() {
        let e = cache().reset_slot(4).unwrap_err().to_string();
        assert!(e.contains("slot out of range"), "{e}");
        assert!(cache().reset_slot(3).is_ok());
    }

    #[test]
    fn copying_a_slot_onto_itself_issues_nothing() {
        assert!(cache().copy_slot_d2d(1, 1).unwrap().is_empty());
    }

    #[test]
    fn the_linear_only_copy_leaves_the_mtp_row_alone() {
        let c = cache();
        assert_eq!(c.copy_slot_d2d(0, 1).unwrap().len(), 3);
        assert_eq!(c.copy_linear_state_slot_d2d(0, 1).unwrap().len(), 2);
    }

    #[test]
    fn a_stack_with_no_linear_layers_only_resets_the_mtp_tier() {
        let c = RecurrentStateCache::allocate(&[false, false], 128, 4, 2, 8, 16, 64, 2);
        assert_eq!(c.reset().len(), 1);
        assert_eq!(c.reset_slot(0).unwrap().len(), 1);
        assert!(c.copy_linear_state_slot_d2d(0, 1).unwrap().is_empty());
    }

    #[test]
    fn the_stash_cap_only_ever_lowers_the_token_count() {
        let mut c = cache();
        c.configure_verify_hidden_stash(8192, 512, Some(256));
        assert_eq!(c.verify_stash().unwrap().max_tokens, 256);

        let mut c = cache();
        c.configure_verify_hidden_stash(8192, 512, Some(99_999));
        assert_eq!(c.verify_stash().unwrap().max_tokens, 8192);

        let mut c = cache();
        c.configure_verify_hidden_stash(8192, 512, Some(0));
        assert_eq!(c.verify_stash().unwrap().max_tokens, 8192);
    }

    #[test]
    fn a_zero_dimension_leaves_the_optional_tiers_unconfigured() {
        let mut c = cache();
        c.configure_verify_hidden_stash(0, 512, None);
        c.configure_rs_buffer_pool(16, 0, 4);
        assert!(!c.verify_hidden_stash_enabled());
        assert!(!c.rs_buffer_pool_enabled());
        assert_eq!(c.verify_hidden_stash_layer(0), None);
        assert_eq!(c.rs_buffer_slab(0, 0), None);
    }

    #[test]
    fn the_buffer_pool_slot_axis_is_its_own_not_max_slots() {
        let mut c = cache();
        c.configure_rs_buffer_pool(16, 32, 9);
        // per slab: 16*32*2 = 1024 B. Layer 1 starts after 9 slabs.
        assert_eq!(c.rs_buffer_slab(1, 0), Some(9 * 1024));
        assert_eq!(c.rs_buffer_slab(1, 8), Some(9 * 1024 + 8 * 1024));
        assert_eq!(c.rs_buffer_slab(1, 9), None);
        assert_eq!(c.rs_buffer_slab(3, 0), None);
        assert_eq!(c.rs_buffer_pool_bytes(), 1024 * 9 * 3);
    }

    #[test]
    fn a_zero_request_count_launches_no_predicated_kernel() {
        let ids = [0i32, 1];
        let fresh = [1u8, 0];
        assert!(
            cache()
                .reset_slots_if_fresh(Some(&ids), Some(&fresh), 0)
                .is_empty()
        );
        assert!(
            cache()
                .reset_slots_if_fresh(None, Some(&fresh), 5)
                .is_empty()
        );
        assert!(cache().reset_slots_if_fresh(Some(&ids), None, 5).is_empty());
        assert_eq!(
            cache()
                .reset_slots_if_fresh(Some(&ids), Some(&fresh), 5)
                .len(),
            3
        );
    }
}
