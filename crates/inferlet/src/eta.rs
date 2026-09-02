//! `inferlet::eta` — the author-facing ETA bridge over the WIT forward surface.
//!
//! `ForwardPass` lives in one of three modules ([`self::attention`],
//! [`self::recurrent`], [`self::hybrid`]), selected by `model::pass_kind()`.
//! It wraps the WIT resources and drives the neutral [`Builder`](eta_dsl::Builder),
//! lowering author stage closures to the ETA container. A [`Channel`] owns
//! both the trace declaration and the WIT resource.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Bound, RangeBounds};
use std::rc::Rc;

use eta_dsl::builder::Builder;
use eta_dsl::channel::PutValue;
use eta_dsl::value::ConstData;
use eta_dsl::{Channel as DslChannel, IntoConst, IntoPut, IntoShape, Port, Shape, Stage, Tensor};

use crate::pie::inferlet::channel as wit_channel;
use crate::pie::inferlet::forward as wit_attention;
use crate::pie::inferlet::forward_hybrid as wit_hybrid;
use crate::pie::inferlet::forward_recurrent as wit_recurrent;
use crate::pie::inferlet::pipeline as wit_pipeline;
use crate::pie::inferlet::types::Dtype as WitDtype;
use crate::working_set::{KvWorkingSet, PageRange, PageSpan};

pub use eta_dsl::intrinsics;

// Re-export the eDSL vocabulary so an author writes stage closures with a
// single `use inferlet::eta::<kind>::prelude::*;`.
pub use eta_dsl::Dtype;
pub use eta_dsl::{
    abs, add, and, broadcast, cast, causal_mask, cummass_le, cumprod, cumsum, div, dtype, entropy,
    entropy_from_logprobs, eq, exp, gather, gather_row, ge, gt, gumbel, gumbel_max, indptr, iota,
    l2norm, le, log, log_softmax, lt, mask_apply, masked_argmax, matmul, max_elem, min_elem, mul,
    ne, neg, not, nucleus_sample, or, pivot_threshold, prob_ge, rank_le, recip, reduce_argmax,
    reduce_max, reduce_min, reduce_sum, rem, reshape, rng, row_membership, scalar_gather,
    scatter_add, scatter_set, select, sign, sink_window_mask, sliding_window_mask, softmax,
    sort_desc, sub, top_k, transpose,
};

// ---------------------------------------------------------------------------
// gid -> WIT channel registry
// ---------------------------------------------------------------------------
// Interns channels by dsl gid; thread-local since inferlets are single-threaded (wasm).
// The WIT resource is created on first use so `.capacity(n)` can still edit it beforehand.

thread_local! {
    static WIT_CHANNELS: RefCell<HashMap<u64, Rc<wit_channel::Channel>>> = RefCell::new(HashMap::new());
    static DECLARED: RefCell<HashMap<u64, ChannelSpec>> = RefCell::new(HashMap::new());
}

#[derive(Clone)]
struct ChannelSpec {
    dims: Vec<u32>,
    dtype: WitDtype,
    capacity: u32,
}

fn declare_channel(gid: u64, spec: ChannelSpec) {
    DECLARED.with(|m| {
        m.borrow_mut().insert(gid, spec);
    });
}

/// Whether `gid`'s WIT resource exists yet (a declared-but-unused channel has none).
fn channel_exists(gid: u64) -> bool {
    WIT_CHANNELS.with(|m| m.borrow().contains_key(&gid))
}

fn set_declared_capacity(gid: u64, capacity: u32) {
    DECLARED.with(|m| {
        if let Some(spec) = m.borrow_mut().get_mut(&gid) {
            spec.capacity = capacity;
        }
    });
}

/// The WIT handle for `gid`, creating it from the declaration on first ask.
fn lookup_channel(gid: u64) -> Option<Rc<wit_channel::Channel>> {
    if let Some(wit) = WIT_CHANNELS.with(|m| m.borrow().get(&gid).cloned()) {
        return Some(wit);
    }
    let spec = DECLARED.with(|m| m.borrow().get(&gid).cloned())?;
    let wit = Rc::new(wit_channel::Channel::new(
        &spec.dims,
        spec.dtype,
        spec.capacity,
    ));
    WIT_CHANNELS.with(|m| {
        m.borrow_mut().insert(gid, Rc::clone(&wit));
    });
    Some(wit)
}

fn to_wit_dtype(d: Dtype) -> WitDtype {
    match d {
        Dtype::F32 => WitDtype::F32,
        Dtype::I32 => WitDtype::I32,
        Dtype::U32 => WitDtype::U32,
        Dtype::Bool => WitDtype::Bool,
        // The WIT enum is four wide; `Dtype` is the tree's, seventeen wide.
        // There is no tag to hand the host for anything else.
        other => panic!("{other:?} is not a dtype ETA computes in; it has no WIT tag"),
    }
}

fn dims_of(shape: Shape) -> Vec<u32> {
    shape.dims().to_vec()
}

// ---------------------------------------------------------------------------
// Channel
// ---------------------------------------------------------------------------

/// Records the port's endpoint claim at pass construction (not first-submit
/// build), so a later-constructed sibling pass sees it too.
fn claim_port(port: Port, ch: &Channel) -> DslChannel {
    let dsl = ch.dsl();
    dsl.note_desc_claim(port.consumes());
    dsl
}

/// A GPU-resident bounded queue, backing both the `eta-dsl` trace and the
/// WIT `channel` resource. Cheap `Copy` token over gid-keyed registry state.
#[derive(Clone, Copy)]
pub struct Channel {
    gid: u64,
    shape: Shape,
    dtype: Dtype,
}

/// In-band validity sentinel: `-1` marks a token slot as not existing — it
/// embeds nothing, appends no KV, and advances no position.
pub const TOKEN_PAD: i32 = -1;

/// Pad a token window to `envelope` slots with [`TOKEN_PAD`]; panics if too long.
pub fn pad_tokens(tokens: &[u32], envelope: usize) -> Vec<i32> {
    assert!(
        tokens.len() <= envelope,
        "window of {} tokens exceeds its envelope of {envelope}",
        tokens.len(),
    );
    tokens
        .iter()
        .map(|&token| token as i32)
        .chain(std::iter::repeat(TOKEN_PAD))
        .take(envelope)
        .collect()
}

/// Recover the live tokens from a device envelope, dropping every [`TOKEN_PAD`] slot.
pub fn unpad_tokens(window: &[i32]) -> Vec<u32> {
    window
        .iter()
        .filter(|&&token| token != TOKEN_PAD)
        .map(|&token| token as u32)
        .collect()
}

impl Channel {
    /// `Channel::new([shape], dtype)` at capacity 1.
    pub fn new(shape: impl IntoShape, dtype: Dtype) -> Channel {
        Channel::build(shape.into_shape(), dtype, false)
    }

    /// An initially empty channel whose producer is the host, so a consuming
    /// pass may be submitted run-ahead and receive the value later.
    pub fn writer(shape: impl IntoShape, dtype: Dtype) -> Channel {
        let channel = Channel::build(shape.into_shape(), dtype, false);
        channel.dsl().note_host_put();
        channel
    }

    /// Registry-resolved DSL trace state; panics if `gid` isn't registered.
    fn dsl(&self) -> DslChannel {
        DslChannel::by_gid(self.gid).expect("channel token resolves in the DSL registry")
    }

    /// The registry-resolved WIT handle.
    fn wit(&self) -> Rc<wit_channel::Channel> {
        lookup_channel(self.gid).expect("channel token resolves in the WIT registry")
    }

    /// Widen the ring to `n` cells (deeper run-ahead). Must be called before
    /// the channel is first used — the WIT resource takes capacity at construction.
    pub fn capacity(self, n: u32) -> Channel {
        assert!(
            !channel_exists(self.gid),
            "capacity must be set before the channel is used"
        );
        self.dsl().capacity(n);
        set_declared_capacity(self.gid, n);
        self
    }

    /// Name the channel (improves trace-error messages).
    pub fn named(self, name: &str) -> Channel {
        let _ = self.dsl().named(name);
        self
    }

    /// `Channel::from(v)` — a channel seeded full with the per-instance value
    /// `v`, rides as a pre-submit `put`, never the container.
    pub fn from(v: impl IntoConst) -> Channel {
        let data: ConstData = v.into_const();
        let ch = Channel::build(data.shape, data.dtype, true);
        ch.wit()
            .put(&data.bytes)
            .expect("stage seed on a fresh channel");
        ch
    }

    /// A seeded channel of a given shape whose seed value is supplied at
    /// instantiation (device loop-carried multi-dim channels).
    pub fn seeded(shape: impl IntoShape, dtype: Dtype) -> Channel {
        Channel::build(shape.into_shape(), dtype, true)
    }

    /// Like [`from`], but reinterprets flat seed `v` under an explicit multi-dim
    /// `shape` (element counts must match) — e.g. a `[B, POOL]` bool mask.
    pub fn from_shaped(shape: impl IntoShape, v: impl IntoConst) -> Channel {
        let data: ConstData = v.into_const();
        let shape = shape.into_shape();
        assert_eq!(
            shape.numel(),
            data.shape.numel(),
            "from_shaped: element count mismatch"
        );
        let ch = Channel::build(shape, data.dtype, true);
        ch.wit()
            .put(&data.bytes)
            .expect("stage seed on a fresh channel");
        ch
    }

    fn build(shape: Shape, dtype: Dtype, seeded: bool) -> Channel {
        let dsl = if seeded {
            DslChannel::seeded(shape, dtype)
        } else {
            DslChannel::new(shape, dtype)
        };
        let gid = dsl.gid();
        declare_channel(
            gid,
            ChannelSpec {
                dims: dims_of(shape),
                dtype: to_wit_dtype(dtype),
                capacity: 1,
            },
        );
        Channel { gid, shape, dtype }
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Consume a cell, inside a stage closure — records a `ChanTake`. Host
    /// counterpart is [`take_host`](Self::take_host).
    pub fn take(&self) -> Tensor {
        self.dsl().take()
    }

    /// `read()` — peek a cell (leaves it full), inside a stage closure. Device
    /// counterpart of [`read_host`](Self::read_host).
    pub fn read(&self) -> Tensor {
        self.dsl().read()
    }

    /// Consume a cell on the host, decoded as `T`. Awaits in-flight fires; a
    /// poisoned channel returns `Err`. `T`'s element type must match the
    /// channel's own dtype — decoding across dtypes reinterprets bytes.
    pub async fn take_host<T: FromChannel>(&self) -> Result<T, String> {
        self.check_host::<T>("take")?;
        self.dsl().note_host_take();
        let raw = self.wit().take().await;
        self.decode_host::<T>(raw, "take")
    }

    /// Peek a cell on the host (leaves it full). Same as
    /// [`take_host`](Self::take_host) otherwise.
    pub async fn read_host<T: FromChannel>(&self) -> Result<T, String> {
        self.check_host::<T>("read")?;
        self.dsl().note_host_read();
        let raw = self.wit().read().await;
        self.decode_host::<T>(raw, "read")
    }

    /// Prefix for host readback errors ("{channel} take"/"{channel} read").
    fn host_label(&self, verb: &str) -> String {
        format!("{} {verb}", self.dsl().name())
    }

    fn check_host<T: FromChannel>(&self, verb: &str) -> Result<(), String> {
        if T::DTYPE != self.dtype {
            return Err(format!(
                "{}: channel holds {:?}, decoded as {:?}",
                self.host_label(verb),
                self.dtype,
                T::DTYPE
            ));
        }
        Ok(())
    }

    fn decode_host<T: FromChannel>(
        &self,
        raw: Result<Vec<u8>, String>,
        verb: &str,
    ) -> Result<T, String> {
        let label = self.host_label(verb);
        let raw = raw.map_err(|e| format!("{label}: {e}"))?;
        T::from_bytes(&raw).map_err(|e| format!("{label}: {e}"))
    }

    /// In a stage closure, records a `ChanPut` device-side; on the host, stages
    /// `v` for the next submit (fire-and-forget — failures surface via
    /// [`take_host`](Self::take_host)).
    pub fn put(&self, v: impl IntoPut) {
        match v.into_put() {
            PutValue::Tensor(t) => {
                self.dsl().put(t);
            }
            PutValue::Data(data) => {
                self.dsl().note_host_put();
                let _ = self.wit().put(&data.bytes);
            }
        }
    }

    /// Atomically replace the committed front cell without changing queue
    /// occupancy. A host operation; unlike a stage `put`, it records no ETA op.
    pub fn set(&self, v: impl IntoConst) -> Result<(), String> {
        let data: ConstData = v.into_const();
        self.wit().set(&data.bytes)
    }
}

/// Seed a channel from an iterator without materializing a `Vec` first.
macro_rules! channel_from_iter {
    ($t:ty) => {
        impl FromIterator<$t> for Channel {
            fn from_iter<I: IntoIterator<Item = $t>>(iter: I) -> Channel {
                Channel::from(iter.into_iter().collect::<Vec<$t>>())
            }
        }
    };
}
channel_from_iter!(u32);
channel_from_iter!(i32);
channel_from_iter!(f32);
channel_from_iter!(bool);

/// A host-readable element type (little-endian, 4 bytes/elem; `bool` is 1 byte).
pub trait HostElem: Copy {
    const DTYPE: Dtype;
    fn decode(raw: &[u8]) -> Vec<Self>;
}
impl HostElem for i32 {
    const DTYPE: Dtype = Dtype::I32;
    fn decode(raw: &[u8]) -> Vec<i32> {
        raw.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
impl HostElem for u32 {
    const DTYPE: Dtype = Dtype::U32;
    fn decode(raw: &[u8]) -> Vec<u32> {
        raw.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
impl HostElem for f32 {
    const DTYPE: Dtype = Dtype::F32;
    fn decode(raw: &[u8]) -> Vec<f32> {
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
impl HostElem for bool {
    const DTYPE: Dtype = Dtype::Bool;
    fn decode(raw: &[u8]) -> Vec<bool> {
        raw.iter().map(|&byte| byte != 0).collect()
    }
}

/// A type [`Channel::take_host`] can decode into: a whole `Vec<T>`, or one bare `T`.
pub trait FromChannel: Sized {
    /// The dtype the channel must hold.
    const DTYPE: Dtype;
    fn from_bytes(raw: &[u8]) -> Result<Self, String>;
}

macro_rules! from_channel {
    ($t:ty) => {
        impl FromChannel for Vec<$t> {
            const DTYPE: Dtype = <$t as HostElem>::DTYPE;
            fn from_bytes(raw: &[u8]) -> Result<Self, String> {
                Ok(<$t as HostElem>::decode(raw))
            }
        }
        impl FromChannel for $t {
            const DTYPE: Dtype = <$t as HostElem>::DTYPE;
            fn from_bytes(raw: &[u8]) -> Result<Self, String> {
                <$t as HostElem>::decode(raw)
                    .into_iter()
                    .next()
                    .ok_or_else(|| format!("channel is empty, expected one {}", stringify!($t)))
            }
        }
    };
}
from_channel!(i32);
from_channel!(u32);
from_channel!(f32);
from_channel!(bool);

// ---------------------------------------------------------------------------
// WorkingSet
// ---------------------------------------------------------------------------

/// The attention working set — a logical page address space over the KV
/// mapping trie. Every page reference is working-set-relative, never a
/// physical page id; `reserve` is purely logical until a forward writes.
pub struct WorkingSet {
    kv: Rc<KvWorkingSet>,
}

impl WorkingSet {
    pub fn new() -> WorkingSet {
        WorkingSet {
            kv: Rc::new(KvWorkingSet::new()),
        }
    }

    /// Current logical extent in pages, including reserved-but-unwritten space.
    pub fn page_len(&self) -> u32 {
        self.kv.page_len()
    }

    /// Extend the logical address space by `pages`; returns the granted index
    /// range. Purely logical — physical pages allocate only when a forward
    /// writes them.
    pub fn reserve(&self, pages: u32) -> Result<PageGrant, String> {
        let range = self.kv.reserve(pages)?;
        Ok(PageGrant {
            start: range.start,
            ids: (range.start..range.start + range.len).collect(),
        })
    }

    /// Insert or atomically replace an opaque, model-scoped index entry for
    /// this fully mapped and settled working set.
    pub fn update_index(&self, key: &[u8]) -> Result<(), String> {
        self.kv.update_index(key)
    }

    /// Exact best-effort lookup of an opaque, model-scoped working-set index.
    pub fn from_index(key: &[u8]) -> Result<Option<WorkingSet>, String> {
        Ok(KvWorkingSet::from_index(key)?.map(|kv| WorkingSet { kv: Rc::new(kv) }))
    }

    /// Remove only an index root. Working sets returned by an earlier lookup
    /// remain valid.
    pub fn remove_index(key: &[u8]) -> Result<bool, String> {
        KvWorkingSet::remove_index(key)
    }

    /// Remove `ranges` (pre-discard indexes, applied atomically), ordered on
    /// `on`. Suffix indexes shift down — publish new geometry after. A
    /// shared-path interior range errs.
    pub fn discard(&self, on: &Pipeline, ranges: &[PageRange]) -> Result<(), String> {
        self.kv.discard(&on.wit, ranges)
    }

    /// O(1) copy-on-write child over the complete logical address space,
    /// ordered on `on` — the branching primitive (beam/MCTS/self-correct).
    pub fn fork(&self, on: &Pipeline) -> Result<WorkingSet, String> {
        Ok(WorkingSet {
            kv: Rc::new(self.kv.fork(&on.wit)?),
        })
    }

    /// Structurally shared child over `[start, start+len)`, rebased to page
    /// zero in the child, ordered on `on`.
    pub fn slice(&self, on: &Pipeline, start: u32, len: u32) -> Result<WorkingSet, String> {
        let child = self.kv.slice(&on.wit, PageRange { start, len })?;
        Ok(WorkingSet { kv: Rc::new(child) })
    }

    /// Move KV cells across all layers from (`src_page_ids[i]`,
    /// `src_tok_idx[i]`) to (`dst_page_ids[i]`, `dst_tok_idx[i]`); the four
    /// lists are parallel. Caller guarantees disjoint src/dst spans.
    pub fn copy_into(
        &self,
        on: &Pipeline,
        dst_page_ids: &[u32],
        dst_tok_idx: &[u32],
        src_page_ids: &[u32],
        src_tok_idx: &[u32],
    ) -> Result<(), String> {
        self.kv.copy_into(
            &on.wit,
            dst_page_ids,
            dst_tok_idx,
            src_page_ids,
            src_tok_idx,
        )
    }
}

impl Default for WorkingSet {
    fn default() -> Self {
        WorkingSet::new()
    }
}

/// A grant of fresh logical page indexes from [`WorkingSet::reserve`] —
/// per-instance data. Puttable into a channel.
pub struct PageGrant {
    start: u32,
    ids: Vec<u32>,
}

impl PageGrant {
    /// The granted WorkingSet-relative page indexes (contiguous).
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// The grant as a WIT `page-range` (e.g. to `discard` it later).
    pub fn range(&self) -> PageRange {
        PageRange {
            start: self.start,
            len: self.ids.len() as u32,
        }
    }
}

impl IntoPut for PageGrant {
    fn into_put(self) -> PutValue {
        PutValue::Data(self.ids.into_const())
    }
}

// ---------------------------------------------------------------------------
// RsWorkingSet
// ---------------------------------------------------------------------------

/// Runtime recurrent-state slots for hybrid / linear-attention models (GDN,
/// Mamba2), one per request in resolved order.
pub struct RsWorkingSet {
    rs: Rc<crate::working_set::RsWorkingSet>,
}

impl RsWorkingSet {
    pub fn new() -> RsWorkingSet {
        RsWorkingSet {
            rs: Rc::new(crate::working_set::RsWorkingSet::new()),
        }
    }

    /// Size in bytes of one folded recurrent-state object for this model.
    /// A cached [`crate::model::rs_state_size`]; see [`WorkingSet::page_size`].
    pub fn state_size(&self) -> u64 {
        thread_local! {
            static SIZE: std::cell::OnceCell<u64> = const { std::cell::OnceCell::new() };
        }
        SIZE.with(|c| *c.get_or_init(crate::model::rs_state_size))
    }

    /// Current number of buffered page slots.
    pub fn buffer_size(&self) -> u32 {
        self.rs.buffer_size()
    }

    /// Tokens per buffered RS page for this model/engine. A cached
    /// [`crate::model::rs_buffer_page_size`]; see [`WorkingSet::page_size`].
    pub fn buffer_page_size(&self) -> u32 {
        thread_local! {
            static SIZE: std::cell::OnceCell<u32> = const { std::cell::OnceCell::new() };
        }
        SIZE.with(|c| *c.get_or_init(crate::model::rs_buffer_page_size))
    }

    /// Append `n` reserved buffered page slots; returns the contiguous
    /// range. Purely logical — a slot is materialized by the first fire whose
    /// `fold-len` leaves tokens in the buffer.
    pub fn alloc_buffer(&self, n: u32) -> Result<crate::working_set::PageRange, String> {
        self.rs.alloc_buffer(n)
    }

    /// Drop the buffered slots at `indices` and densely compact — the
    /// reject half of fold-commit: a speculative tail that was buffered but
    /// never folded is abandoned, and no folded state was ever perturbed by it.
    pub fn free_buffer(&self, indices: &[u32]) -> Result<(), String> {
        self.rs.free_buffer(indices)
    }

    /// Forget the last `count` buffered tokens — free, since the slots it
    /// releases are overwritten by the next append. Twin of `fold-len`, which
    /// moves the folded boundary right and cannot be undone.
    pub fn discard_buffered(&self, count: u32) -> Result<(), String> {
        self.rs.discard_buffered(count)
    }

    /// Reorder the buffered slots by the full bijection `perm`.
    pub fn reorder_buffer(&self, perm: &[u32]) -> Result<(), String> {
        self.rs.reorder_buffer(perm)
    }

    /// Copy-on-write child sharing the current folded state and buffered
    /// suffix, ordered on `on`.
    pub fn fork(&self, on: &Pipeline) -> Result<RsWorkingSet, String> {
        Ok(RsWorkingSet {
            rs: Rc::new(self.rs.fork(&on.wit)?),
        })
    }
}

impl Default for RsWorkingSet {
    fn default() -> Self {
        RsWorkingSet::new()
    }
}

// ---------------------------------------------------------------------------
// ForwardPass
// ---------------------------------------------------------------------------

type StageClosure = Box<dyn Fn()>;

/// The forward-pass resource of one `pie:inferlet` forward interface: one of
/// `forward`, `forward-recurrent`, `forward-hybrid`.
pub trait PassWit: Sized + 'static {
    fn new() -> Self;

    fn embed(
        &self,
        tokens: &wit_channel::Channel,
        indptr: &wit_channel::Channel,
    ) -> Result<(), String>;

    fn readout(&self, indices: &wit_channel::Channel) -> Result<(), String>;

    /// The pass's layer truncation.
    fn set_max_layers(&self, max_layers: u32) -> Result<(), String>;

    fn program(&self, bytes: &[u8], channels: &[&wit_channel::Channel]) -> Result<(), String>;

    fn submit(on: &wit_pipeline::Pipeline, slots: &[Option<&Self>]) -> Result<(), String>;
}

impl PassWit for wit_attention::ForwardPass {
    fn new() -> Self {
        wit_attention::ForwardPass::new()
    }
    fn embed(
        &self,
        tokens: &wit_channel::Channel,
        indptr: &wit_channel::Channel,
    ) -> Result<(), String> {
        wit_attention::ForwardPass::embed(self, tokens, indptr)
    }
    fn readout(&self, indices: &wit_channel::Channel) -> Result<(), String> {
        wit_attention::ForwardPass::readout(self, indices)
    }
    fn set_max_layers(&self, max_layers: u32) -> Result<(), String> {
        wit_attention::ForwardPass::set_max_layers(self, max_layers).map_err(|e| e.to_string())
    }
    fn program(&self, bytes: &[u8], channels: &[&wit_channel::Channel]) -> Result<(), String> {
        wit_attention::ForwardPass::program(self, bytes, channels)
    }
    fn submit(on: &wit_pipeline::Pipeline, slots: &[Option<&Self>]) -> Result<(), String> {
        wit_attention::submit(on, slots)
    }
}

impl PassWit for wit_recurrent::ForwardPass {
    fn new() -> Self {
        wit_recurrent::ForwardPass::new()
    }
    fn embed(
        &self,
        tokens: &wit_channel::Channel,
        indptr: &wit_channel::Channel,
    ) -> Result<(), String> {
        wit_recurrent::ForwardPass::embed(self, tokens, indptr)
    }
    fn readout(&self, indices: &wit_channel::Channel) -> Result<(), String> {
        wit_recurrent::ForwardPass::readout(self, indices)
    }
    fn set_max_layers(&self, max_layers: u32) -> Result<(), String> {
        wit_recurrent::ForwardPass::set_max_layers(self, max_layers).map_err(|e| e.to_string())
    }
    fn program(&self, bytes: &[u8], channels: &[&wit_channel::Channel]) -> Result<(), String> {
        wit_recurrent::ForwardPass::program(self, bytes, channels)
    }
    fn submit(on: &wit_pipeline::Pipeline, slots: &[Option<&Self>]) -> Result<(), String> {
        wit_recurrent::submit(on, slots)
    }
}

impl PassWit for wit_hybrid::ForwardPass {
    fn new() -> Self {
        wit_hybrid::ForwardPass::new()
    }
    fn embed(
        &self,
        tokens: &wit_channel::Channel,
        indptr: &wit_channel::Channel,
    ) -> Result<(), String> {
        wit_hybrid::ForwardPass::embed(self, tokens, indptr)
    }
    fn readout(&self, indices: &wit_channel::Channel) -> Result<(), String> {
        wit_hybrid::ForwardPass::readout(self, indices)
    }
    fn set_max_layers(&self, max_layers: u32) -> Result<(), String> {
        wit_hybrid::ForwardPass::set_max_layers(self, max_layers).map_err(|e| e.to_string())
    }
    fn program(&self, bytes: &[u8], channels: &[&wit_channel::Channel]) -> Result<(), String> {
        wit_hybrid::ForwardPass::program(self, bytes, channels)
    }
    fn submit(on: &wit_pipeline::Pipeline, slots: &[Option<&Self>]) -> Result<(), String> {
        wit_hybrid::submit(on, slots)
    }
}

/// The PEFT adapter surface's expression vocabulary, used by [`Pass::adapter`].
pub mod adapter {
    use super::Channel;

    /// Model projection sites; the engine refuses unconsumed sites loudly.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Site {
        Q,
        K,
        V,
        O,
        GateUp,
        Down,
    }

    impl Site {
        pub fn bit(self) -> u32 {
            match self {
                Site::Q => 1 << 0,
                Site::K => 1 << 1,
                Site::V => 1 << 2,
                Site::O => 1 << 3,
                Site::GateUp => 1 << 4,
                Site::Down => 1 << 5,
            }
        }
    }

    /// One adapter expression node, consumed by the classifier — never executed directly.
    pub struct Expr {
        pub(crate) kind: ExprKind,
    }

    pub(crate) enum ExprKind {
        X,
        Y,
        Mm(Channel, Box<Expr>),
        Add(Box<Expr>, Box<Expr>),
        Scale(Channel, Box<Expr>),
    }

    impl Expr {
        pub(crate) fn x() -> Self {
            Expr { kind: ExprKind::X }
        }
        pub(crate) fn y() -> Self {
            Expr { kind: ExprKind::Y }
        }
    }

    /// `mm(w, e)` — multiply by the channel-borne weight (leading dim is
    /// the layer axis).
    pub fn mm(w: &Channel, e: Expr) -> Expr {
        Expr {
            kind: ExprKind::Mm(*w, Box::new(e)),
        }
    }

    /// `scale(e, l)` — elementwise multiply by the channel-borne vector
    /// `l: [num_layers, d_out]` (IA3's form).
    pub fn scale(e: Expr, l: &Channel) -> Expr {
        Expr {
            kind: ExprKind::Scale(*l, Box::new(e)),
        }
    }

    impl std::ops::Add for Expr {
        type Output = Expr;
        fn add(self, rhs: Expr) -> Expr {
            Expr {
                kind: ExprKind::Add(Box::new(self), Box::new(rhs)),
            }
        }
    }
}

/// The forward-pass builder over the `W` interface (see [`attention`] / [`recurrent`] / [`hybrid`]).
pub struct Pass<W: PassWit> {
    wit: W,
    inner: RefCell<ForwardInner>,
}

struct ForwardInner {
    ports: Vec<(Port, DslChannel)>,
    stages: Vec<(Stage, StageClosure)>,
    vocab: u32,
    page_size: u32,
    attention_ws: Option<Rc<KvWorkingSet>>,
    rs_working_sets: Vec<Rc<crate::working_set::RsWorkingSet>>,
    program_attached: bool,
    adapter_lowrank_sites: u32,
    adapter_scale_sites: u32,
}

/// A [`KvGeometry`] with claimed ports and resolved WIT handles, held until state-binding.
struct StagedKv {
    ws: Rc<KvWorkingSet>,
    readable: PageDeclaration,
    writable: PageDeclaration,
    kv_len: Rc<wit_channel::Channel>,
    pages: Rc<wit_channel::Channel>,
    page_indptr: Rc<wit_channel::Channel>,
    w_slot: Rc<wit_channel::Channel>,
    w_off: Rc<wit_channel::Channel>,
    positions: Rc<wit_channel::Channel>,
    mask: Option<Rc<wit_channel::Channel>>,
}

/// The attention geometry of one fire — mirrors WIT `kv-geometry` field-for-field.
pub struct KvGeometry<'a, R, W> {
    pub readable_pages: R,
    pub writable_pages: W,
    pub kv_len: &'a Channel,
    pub pages: &'a Channel,
    pub page_indptr: &'a Channel,
    pub w_slot: &'a Channel,
    pub w_off: &'a Channel,
    pub positions: &'a Channel,
    /// `None` omits ETA's AttnMask port; `Some` binds that channel to it.
    pub mask: Option<&'a Channel>,
}

/// Where the bound recurrent state's folded boundary lands — mirrors WIT `rs-geometry`.
pub struct RsGeometry<'a, B> {
    /// Per-request advance of the folded boundary, clamped to `[buffer |
    /// this fire's tokens]`. `None` means fold-everything.
    pub fold_len: Option<&'a Channel>,
    /// Capacity grant, not an address — a guest copy could only agree or be refused.
    pub buffer: B,
}

/// A KV working set and the geometry it's read/written through — mirrors WIT `kv-binding`.
pub struct KvBinding<'a, R, W> {
    pub working_set: &'a WorkingSet,
    pub geometry: KvGeometry<'a, R, W>,
}

thread_local! {
    /// The `fold-len` of a pass that folds unconditionally: `u32::MAX`.
    static FOLD_ALL: Channel = Channel::from(vec![u32::MAX]);
}

/// [`RsGeometry`] with its port claimed and channel resolved. See [`StagedKv`].
struct StagedRs {
    working_sets: Vec<Rc<crate::working_set::RsWorkingSet>>,
    fold_len: Rc<wit_channel::Channel>,
    buffer: PageDeclaration,
}

#[derive(Clone, Copy)]
struct PageDeclaration {
    start: u32,
    end: Option<u32>,
}

impl PageDeclaration {
    fn from_range(range: impl RangeBounds<u32>) -> Result<Self, String> {
        let start = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start
                .checked_add(1)
                .ok_or_else(|| "attention page-span start overflows u32".to_string())?,
        };
        let end = match range.end_bound() {
            Bound::Unbounded => None,
            Bound::Excluded(&end) => Some(end),
            Bound::Included(&end) => Some(
                end.checked_add(1)
                    .ok_or_else(|| "attention page-span end overflows u32".to_string())?,
            ),
        };
        if end.is_some_and(|end| start > end) {
            return Err(format!(
                "attention page-span start {start} exceeds end {}",
                end.unwrap()
            ));
        }
        Ok(Self { start, end })
    }

    fn wit(self) -> PageSpan {
        PageSpan {
            start: self.start,
            end: self.end,
        }
    }
}

#[cfg(test)]
mod page_declaration_tests {
    use super::*;

    #[test]
    fn normalizes_inclusive_and_exclusive_bounds() {
        let closed = PageDeclaration::from_range(2..5).unwrap();
        assert_eq!((closed.start, closed.end), (2, Some(5)));

        let inclusive = PageDeclaration::from_range(2..=5).unwrap();
        assert_eq!((inclusive.start, inclusive.end), (2, Some(6)));
    }

}

impl<W: PassWit> Pass<W> {
    /// Does this pass bind a dense `AttnMask` channel? Kept here (not
    /// [`live_slots`]) since that call site has no pass in hand.
    pub fn binds_device_mask(&self) -> bool {
        self.inner
            .borrow()
            .ports
            .iter()
            .any(|(port, _)| *port == Port::AttnMask)
    }

    pub fn new() -> Pass<W> {
        let vocab = crate::model::output_vocab_size();
        let page_size = kv_page_size();
        Pass {
            wit: W::new(),
            inner: RefCell::new(ForwardInner {
                ports: Vec::new(),
                stages: Vec::new(),
                vocab,
                page_size,
                attention_ws: None,
                rs_working_sets: Vec::new(),
                program_attached: false,
                adapter_lowrank_sites: 0,
                adapter_scale_sites: 0,
            }),
        }
    }

    fn ensure_ports_available(&self, ports: &[Port]) -> Result<(), String> {
        let inner = self.inner.borrow();
        if inner.program_attached {
            return Err("forward pass program is already attached".to_string());
        }
        if let Some(port) = ports
            .iter()
            .find(|port| inner.ports.iter().any(|(bound, _)| bound == *port))
        {
            return Err(format!(
                "forward pass port {} is already bound",
                port.name()
            ));
        }
        Ok(())
    }

    /// Bind token ids and CSR row indptr. Both descriptor inputs are channels.
    pub fn embed(&self, tokens: &Channel, indptr: &Channel) -> Result<(), String> {
        self.ensure_ports_available(&[Port::EmbedTokens, Port::EmbedIndptr])?;
        let token_wit = tokens.wit();
        let indptr_wit = indptr.wit();
        self.wit.embed(token_wit.as_ref(), indptr_wit.as_ref())?;
        self.inner.borrow_mut().ports.extend([
            (Port::EmbedTokens, claim_port(Port::EmbedTokens, tokens)),
            (Port::EmbedIndptr, claim_port(Port::EmbedIndptr, indptr)),
        ]);
        Ok(())
    }

    /// Claim the KV geometry ports and resolve its channels. After the
    /// program is attached this is a rebind — only WIT-side handles refresh.
    fn stage_kv<R, Wr>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, Wr>,
    ) -> Result<StagedKv, String>
    where
        R: RangeBounds<u32>,
        Wr: RangeBounds<u32>,
    {
        let KvGeometry {
            readable_pages,
            writable_pages,
            kv_len,
            pages,
            page_indptr,
            w_slot,
            w_off,
            positions,
            mask,
        } = geom;
        let rebind = self.inner.borrow().program_attached;
        if !rebind {
            let mut ports = vec![
                Port::KvLen,
                Port::Pages,
                Port::PageIndptr,
                Port::WSlot,
                Port::WOff,
                Port::Positions,
            ];
            if mask.is_some() {
                ports.push(Port::AttnMask);
            }
            self.ensure_ports_available(&ports)?;
        }
        let staged = StagedKv {
            ws: ws.kv.clone(),
            readable: PageDeclaration::from_range(readable_pages)?,
            writable: PageDeclaration::from_range(writable_pages)?,
            kv_len: kv_len.wit(),
            pages: pages.wit(),
            page_indptr: page_indptr.wit(),
            w_slot: w_slot.wit(),
            w_off: w_off.wit(),
            positions: positions.wit(),
            mask: mask.map(Channel::wit),
        };
        let mut inner = self.inner.borrow_mut();
        inner.attention_ws = Some(ws.kv.clone());
        if !rebind {
            inner.ports.extend([
                (Port::KvLen, claim_port(Port::KvLen, kv_len)),
                (Port::Pages, claim_port(Port::Pages, pages)),
                (Port::PageIndptr, claim_port(Port::PageIndptr, page_indptr)),
                (Port::WSlot, claim_port(Port::WSlot, w_slot)),
                (Port::WOff, claim_port(Port::WOff, w_off)),
                (Port::Positions, claim_port(Port::Positions, positions)),
            ]);
            if let Some(mask) = mask {
                inner
                    .ports
                    .push((Port::AttnMask, claim_port(Port::AttnMask, mask)));
            }
        }
        Ok(staged)
    }

    /// Claim the fold-len port and resolve the recurrent working sets, in
    /// resolved request order. See [`Pass::stage_kv`] for the rebind rule.
    fn stage_rs<B>(
        &self,
        working_sets: &[RsWorkingSet],
        geom: RsGeometry<'_, B>,
    ) -> Result<StagedRs, String>
    where
        B: RangeBounds<u32>,
    {
        if working_sets.is_empty() {
            return Err(
                "forward pass needs one recurrent-state working set per request".to_string(),
            );
        }
        let buffer = PageDeclaration::from_range(geom.buffer)?;
        let staged = match geom.fold_len {
            Some(fold_len) => {
                if !self.inner.borrow().program_attached {
                    self.ensure_ports_available(&[Port::RsFoldLen])?;
                    let mut inner = self.inner.borrow_mut();
                    inner
                        .ports
                        .push((Port::RsFoldLen, claim_port(Port::RsFoldLen, fold_len)));
                }
                fold_len.wit()
            }
            // Minted once per guest thread so a rebind doesn't leak a
            // channel; no port claimed since folding everything computes nothing.
            None => FOLD_ALL.with(Channel::wit),
        };
        let working_sets: Vec<Rc<crate::working_set::RsWorkingSet>> =
            working_sets.iter().map(|rs| rs.rs.clone()).collect();
        self.inner.borrow_mut().rs_working_sets = working_sets.clone();
        Ok(StagedRs {
            working_sets,
            fold_len: staged,
            buffer,
        })
    }

    /// Bind readout indexes through a channel, separately from embedding.
    pub fn readout(&self, indices: &Channel) -> Result<(), String> {
        self.ensure_ports_available(&[Port::Readout])?;
        let indices_wit = indices.wit();
        self.wit.readout(indices_wit.as_ref())?;
        self.inner
            .borrow_mut()
            .ports
            .push((Port::Readout, claim_port(Port::Readout, indices)));
        Ok(())
    }

    /// Run only the first `max_layers` transformer layers for this pass's
    /// fires and take the head there (the layerskip-draft / logit-lens
    /// class). Call before `program`.
    pub fn set_max_layers(&self, max_layers: u32) -> Result<(), String> {
        self.wit.set_max_layers(max_layers)
    }

    /// Attach a PEFT adapter at `site`: `f` receives input `x` and base
    /// output `y`, returns the corrected [`adapter`] expression. Lowers
    /// LoRA, IA3, and DoRA forms into per-layer prologue sinks. One adapter
    /// per site per pass.
    pub fn adapter(
        &self,
        site: adapter::Site,
        f: impl FnOnce(adapter::Expr, adapter::Expr) -> adapter::Expr,
    ) -> Result<(), String> {
        use adapter::ExprKind as K;
        let expr = f(adapter::Expr::x(), adapter::Expr::y());
        // DoRA lowers to the low-rank sink then the scale sink on the same site.
        if let K::Scale(l, inner) = &expr.kind
            && let K::Add(lhs, rhs) = &inner.kind
        {
            let delta = match (&lhs.kind, &rhs.kind) {
                (K::Y, _) => &rhs.kind,
                (_, K::Y) => &lhs.kind,
                _ => &inner.kind, // falls to the refusal below
            };
            if let K::Mm(b, mid) = delta
                && let K::Mm(a, x) = &mid.kind
                && matches!(x.kind, K::X)
            {
                let (a, b, l) = (*a, *b, *l);
                {
                    let mut st = self.inner.borrow_mut();
                    if (st.adapter_lowrank_sites | st.adapter_scale_sites) & site.bit() != 0 {
                        return Err(format!(
                            "adapter: site {site:?} already carries an \
                                         adapter on this pass"
                        ));
                    }
                    st.adapter_lowrank_sites |= site.bit();
                    st.adapter_scale_sites |= site.bit();
                }
                self.prologue(move || {
                    intrinsics::kernel::lora(a.read(), b.read(), Tensor::constant(site.bit()));
                    intrinsics::kernel::adapter_scale(l.read(), Tensor::constant(site.bit()));
                });
                return Ok(());
            }
        }
        // The scale form (IA3): scale(y, l).
        if let K::Scale(l, inner) = &expr.kind
            && matches!(inner.kind, K::Y)
        {
            let l = *l;
            {
                let mut inner_state = self.inner.borrow_mut();
                if inner_state.adapter_scale_sites & site.bit() != 0 {
                    return Err(format!(
                        "adapter: site {site:?} already carries a scale on \
                             this pass"
                    ));
                }
                inner_state.adapter_scale_sites |= site.bit();
            }
            self.prologue(move || {
                intrinsics::kernel::adapter_scale(l.read(), Tensor::constant(site.bit()));
            });
            return Ok(());
        }
        let (lhs, rhs) = match expr.kind {
            K::Add(l, r) => (*l, *r),
            _ => {
                return Err("adapter: form not lowerable (v0 lowers the low-rank \
                     form `y + mm(b, mm(a, x))` only)"
                    .to_string());
            }
        };
        let delta = match (&lhs.kind, &rhs.kind) {
            (K::Y, _) => rhs.kind,
            (_, K::Y) => lhs.kind,
            _ => return Err("adapter: the base output `y` must be one addend".to_string()),
        };
        let (b, a) = match delta {
            K::Mm(b, inner) => match inner.kind {
                K::Mm(a, x) if matches!(x.kind, K::X) => (b, a),
                _ => return Err("adapter: the delta must be mm(b, mm(a, x))".to_string()),
            },
            _ => return Err("adapter: the delta must be mm(b, mm(a, x))".to_string()),
        };
        {
            // One pair per site: each call emits its own lora sink.
            let mut inner = self.inner.borrow_mut();
            if inner.adapter_lowrank_sites & site.bit() != 0 {
                return Err(format!(
                    "adapter: site {site:?} already carries an adapter on this pass"
                ));
            }
            inner.adapter_lowrank_sites |= site.bit();
        }
        self.prologue(move || {
            intrinsics::kernel::lora(a.read(), b.read(), Tensor::constant(site.bit()));
        });
        Ok(())
    }

    pub fn prologue(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::Prologue, body);
    }
    /// Attach the `epilogue` stage (sampling programs; after the forward).
    pub fn epilogue(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::Epilogue, body);
    }

    fn set_stage(&self, stage: Stage, body: impl Fn() + 'static) {
        let mut inner = self.inner.borrow_mut();
        assert!(
            !inner.program_attached,
            "stage attachment is construction-only"
        );
        if let Some(slot) = inner.stages.iter_mut().find(|(s, _)| *s == stage) {
            slot.1 = Box::new(body);
        } else {
            inner.stages.push((stage, Box::new(body)));
        }
    }

    /// Enqueue this pass as a single-slot frame on `on`: slot 0 is this
    /// pass, the rest pad to no-ops. For a one-shot fire only — a decode
    /// loop should use [`run_ahead`] or [`submit_frame`] instead.
    pub fn submit(&self, on: &Pipeline) -> Result<(), String> {
        submit_frame(on, &[Some(self)])
    }

    fn attach_program(&self) -> Result<(), String> {
        if self.inner.borrow().program_attached {
            return Ok(());
        }
        let inner = self.inner.borrow();
        let mut builder = Builder::new(inner.vocab, inner.page_size);
        for (port, channel) in &inner.ports {
            builder.bind_port_recorded(*port, channel.clone());
        }
        for (stage, body) in &inner.stages {
            builder.stage(*stage, body);
        }
        let traced = builder.build().map_err(|error| error.to_string())?;
        drop(builder);
        let handles: Vec<Rc<wit_channel::Channel>> = traced
            .channel_order()
            .iter()
            .map(|gid| lookup_channel(*gid).expect("channel registered before submit"))
            .collect();
        let borrows: Vec<&wit_channel::Channel> = handles.iter().map(Rc::as_ref).collect();
        let bytes = traced.encode();
        self.wit.program(&bytes, &borrows)?;
        drop(inner);
        self.inner.borrow_mut().program_attached = true;
        Ok(())
    }
}

impl<W: PassWit> Default for Pass<W> {
    fn default() -> Self {
        Pass::new()
    }
}

/// Waves per frame (k) for this deployment (cached; fixed at runtime
/// start). Guests must be output-correct for any k.
pub fn frame_size() -> usize {
    thread_local! {
        static FRAME_SIZE: std::cell::OnceCell<usize> = const { std::cell::OnceCell::new() };
    }
    FRAME_SIZE.with(|k| *k.get_or_init(|| crate::model::frame_size().max(1) as usize))
}

/// How long a pipeline may hold a frame's wait-set before the runtime stops
/// waiting (cached); not fatal to overrun, but call `Pipeline::park` around long work.
pub fn submit_deadline() -> std::time::Duration {
    thread_local! {
        static DEADLINE: std::cell::OnceCell<u64> = const { std::cell::OnceCell::new() };
    }
    std::time::Duration::from_micros(
        DEADLINE.with(|d| *d.get_or_init(crate::model::submit_deadline_us)),
    )
}

/// Host-reader channel capacity, in cells, that sustains the runtime's
/// run-ahead for one lane; not cached, unlike [`frame_size`].
pub fn channel_capacity() -> usize {
    (crate::model::channel_capacity() as usize).max(2)
}

/// Live slots per frame for the bound model: k for dense, 1 for recurrent
/// (linear/hybrid) — conservative, not a hard constraint.
pub fn live_slots() -> usize {
    thread_local! {
        static LIVE: std::cell::OnceCell<usize> = const { std::cell::OnceCell::new() };
    }
    LIVE.with(|live| {
        *live.get_or_init(|| {
            if crate::model::pass_kind() != crate::model::ForwardKind::Attention {
                1
            } else {
                frame_size()
            }
        })
    })
}

/// Tokens per KV page (cached); prefer [`WorkingSet::page_size`] when a
/// working set is in hand.
pub fn kv_page_size() -> u32 {
    thread_local! {
        static PAGE: std::cell::OnceCell<u32> = const { std::cell::OnceCell::new() };
    }
    PAGE.with(|c| *c.get_or_init(crate::model::kv_page_size))
}

/// Max embed tokens in a single pass (cached, guest-side prefill chunk
/// budget); split a longer prompt with [`prefill_chunks`].
pub fn max_embed_length() -> usize {
    thread_local! {
        static MAX_EMBED: std::cell::OnceCell<usize> = const { std::cell::OnceCell::new() };
    }
    MAX_EMBED.with(|c| *c.get_or_init(|| crate::model::max_embed_length().max(1) as usize))
}

/// The `[start, end)` spans a prompt of `n` tokens must be prefilled in,
/// respecting [`max_embed_length`]. `cap` overrides the limit, `None` for default.
pub fn prefill_chunks(n: u32, cap: Option<u32>) -> Vec<(u32, u32)> {
    let cap = cap
        .unwrap_or(u32::MAX)
        .min(max_embed_length().max(1) as u32);
    even_spans(n, cap)
}

/// Arithmetic of [`prefill_chunks`], split out to test off-device.
fn even_spans(n: u32, cap: u32) -> Vec<(u32, u32)> {
    if n == 0 {
        return Vec::new();
    }
    let cap = cap.min(n).max(1);
    let k = n.div_ceil(cap).max(1);
    let (q, r) = (n / k, n % k);
    let mut out = Vec::with_capacity(k as usize);
    let mut base = 0u32;
    for i in 0..k {
        let end = base + q + u32::from(i < r);
        out.push((base, end));
        base = end;
    }
    debug_assert_eq!(base, n);
    out
}

/// Submit ONE FRAME on `on`: up to `frame_size()` slots, slot i executing
/// in wave i; trailing slots pad with no-ops. First submit attaches the program.
pub fn submit_frame<W: PassWit>(on: &Pipeline, slots: &[Option<&Pass<W>>]) -> Result<(), String> {
    let k = frame_size();
    if slots.len() > k {
        return Err(format!(
            "frame holds {} slot(s); model.frame-size() is {k}",
            slots.len()
        ));
    }
    for pass in slots.iter().flatten() {
        pass.attach_program()?;
    }
    if slots.iter().flatten().next().is_none() {
        return Ok(());
    }
    let mut borrows: Vec<Option<&W>> = slots.iter().map(|slot| slot.map(|p| &p.wit)).collect();
    borrows.resize(k, None);
    W::submit(&on.wit, &borrows)
}

/// Keeps the runtime's run-ahead window full while `on_token` consumes
/// results, until `budget` fires submit or `on_token` breaks. Returns the run count.
pub async fn run_ahead<W: PassWit>(
    on: &Pipeline,
    pass: &Pass<W>,
    budget: usize,
    mut on_token: impl AsyncFnMut() -> Result<std::ops::ControlFlow<()>, String>,
) -> Result<usize, String> {
    use std::ops::ControlFlow;

    if budget == 0 {
        return Ok(0);
    }
    // A pass that binds a dense device mask takes one slot per frame — see
    // `Pass::binds_device_mask`.
    let r = if pass.binds_device_mask() {
        1
    } else {
        live_slots()
    };
    // `channel_capacity()` carries the staging margin; the window is what
    // remains, in frames of `r` live slots.
    let window_frames = ((channel_capacity() - 1) / r.max(1)).max(1);

    let mut submitted = 0usize;
    let mut consumed = 0usize;

    // One frame of up to `r` live slots, never past `budget`.
    let submit_one_frame = |submitted: &mut usize| -> Result<(), String> {
        let live = r.min(budget - *submitted);
        if live == 0 {
            return Ok(());
        }
        let slots: Vec<Option<&Pass<W>>> = vec![Some(pass); live];
        submit_frame(on, &slots)?;
        *submitted += live;
        Ok(())
    };

    for _ in 0..window_frames {
        if submitted >= budget {
            break;
        }
        submit_one_frame(&mut submitted)?;
    }

    // Close must fire the instant this lane stops submitting, or other lanes
    // hold their seal waiting on it. Safe with fires in flight — close never waits.
    let mut ended = false;

    // (a) priming loop already spent the budget; close before the first take.
    if submitted >= budget && !ended {
        on.close();
        ended = true;
    }
    while consumed < submitted {
        if on_token().await? == ControlFlow::Break(()) {
            // (b) an early stop also ends the stream; close reclaims the rest.
            if !ended {
                on.close();
            }
            return Ok(consumed + 1);
        }
        consumed += 1;
        // Refill a whole frame at a time — a partial frame can't be topped up later.
        if submitted < budget && submitted - consumed <= (window_frames - 1) * r {
            submit_one_frame(&mut submitted)?;
        }
        if submitted >= budget && !ended {
            on.close();
            ended = true;
        }
    }
    // (c) a zero-width lane never submits, so the loop can exit with
    //     `submitted < budget` without (a) or the in-loop check firing.
    if !ended {
        on.close();
    }
    Ok(consumed)
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// A run-ahead ordering domain — every command on it linearizes in
/// submission order. Concurrent streams need separate pipelines.
pub struct Pipeline {
    wit: wit_pipeline::Pipeline,
}

impl Pipeline {
    pub fn new() -> Pipeline {
        Pipeline {
            wit: wit_pipeline::Pipeline::new(),
        }
    }

    /// End the stream and release its scheduler wait-set immediately.
    /// Already-submitted fires still drain and remain take-able.
    pub fn close(&self) {
        self.wit.close();
    }

    /// Leave the frame wait-set until this pipeline submits again — the way
    /// to go idle without running down `submit_deadline`. Unlike `close`,
    /// the pipeline stays usable; the next `submit` rejoins automatically.
    pub fn park(&self) {
        // Every forward interface declares the same `park`, one runtime function.
        wit_attention::park(&self.wit);
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Pipeline::new()
    }
}

/// The kind-independent half of every pass prelude. No top-level
/// `eta::prelude`: importing a pass type requires naming its kind.
pub mod shared_prelude {
    pub use super::{
        Channel, KvBinding, KvGeometry, PageGrant, Pipeline, RsGeometry, RsWorkingSet, TOKEN_PAD,
        WorkingSet, channel_capacity, frame_size, kv_page_size, live_slots, max_embed_length,
        pad_tokens, prefill_chunks, unpad_tokens,
    };
    /// Every inferlet returns `inferlet::Result` and uses `model`, so both ride the prelude.
    pub use crate::{Context, Result, model};
    pub use std::ops::ControlFlow;
    /// Only `Stage`; dtypes are spelled `dtype::f32` and friends.
    pub use eta_dsl::Stage;
    pub use eta_dsl::dtype;
    pub use eta_dsl::intrinsics;
    /// Arithmetic intrinsics are absent — `+ - * / %` and unary `-` are their spelling.
    pub use eta_dsl::value::{
        Tensor, abs, and, broadcast, cast, causal_mask, cummass_le, cumprod, cumsum, entropy,
        entropy_from_logprobs, eq, exp, gather, gather_row, ge, gt, gumbel, gumbel_max, indptr,
        iota, l2norm, le, log, log_softmax, lt, mask_apply, masked_argmax, matmul, max_elem,
        min_elem, ne, not, nucleus_sample, or, pivot_threshold, prob_ge, rank_le, recip,
        reduce_argmax, reduce_max, reduce_min, reduce_sum, reshape, rng, row_membership,
        scalar_gather, scatter_add, scatter_set, select, sign, sink_window_mask,
        sliding_window_mask, softmax, sort_desc, top_k, transpose,
    };
}

// ---------------------------------------------------------------------------
// The three author-facing pass modules
// ---------------------------------------------------------------------------

/// Attention taps, legal only where attention layers exist.
impl Pass<wit_attention::ForwardPass> {
    /// Attach the `on_attn_proj` stage (per layer, before attention).
    pub fn on_attn_proj(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::OnAttnProj, body);
    }
    /// Attach the `on_attn` stage (per layer, after attention).
    pub fn on_attn(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::OnAttn, body);
    }

    /// `pie:inferlet/forward.attention` — bind the KV working set and all of
    /// its geometry channels. See [`KvGeometry`]. REQUIRED.
    pub fn attention<R, W>(&self, ws: &WorkingSet, geom: KvGeometry<'_, R, W>) -> Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        let kv = self.stage_kv(ws, geom)?;
        wit_attention::ForwardPass::attention(
            &self.wit,
            kv.ws.as_ref(),
            &wit_attention::KvGeometry {
                readable_pages: kv.readable.wit(),
                writable_pages: kv.writable.wit(),
                kv_len: kv.kv_len.as_ref(),
                pages: kv.pages.as_ref(),
                page_indptr: kv.page_indptr.as_ref(),
                w_slot: kv.w_slot.as_ref(),
                w_off: kv.w_off.as_ref(),
                positions: kv.positions.as_ref(),
                mask: kv.mask.as_deref(),
            },
        )
    }

    /// `pie:inferlet/forward.media` — carries a media span's payload,
    /// order-matched to the placeholder token runs already in the sequence
    /// (see `img.tokens()`). Attention interface only.
    pub fn media(&self, spans: &[wit_attention::MediaSpan<'_>]) -> Result<(), String> {
        wit_attention::ForwardPass::media(&self.wit, spans)
    }
}

impl Pass<wit_hybrid::ForwardPass> {
    /// Attach the `on_attn_proj` stage (per attention layer, before attention).
    pub fn on_attn_proj(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::OnAttnProj, body);
    }
    /// Attach the `on_attn` stage (per attention layer, after attention).
    pub fn on_attn(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::OnAttn, body);
    }

    /// `pie:inferlet/forward-hybrid.attention` — bind both halves of state in
    /// one call; `kv` is [`Option`] for a recurrent-only fire. Rebinds on recall.
    pub fn attention<R, W, B>(
        &self,
        kv: Option<KvBinding<'_, R, W>>,
        rs: &[RsWorkingSet],
        rs_geom: RsGeometry<'_, B>,
    ) -> Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
        B: RangeBounds<u32>,
    {
        let kv = kv
            .map(|kv| self.stage_kv(kv.working_set, kv.geometry))
            .transpose()?;
        let rs = self.stage_rs(rs, rs_geom)?;
        let binding = kv.as_ref().map(|kv| wit_hybrid::KvBinding {
            working_set: kv.ws.as_ref(),
            geometry: wit_hybrid::KvGeometry {
                readable_pages: kv.readable.wit(),
                writable_pages: kv.writable.wit(),
                kv_len: kv.kv_len.as_ref(),
                pages: kv.pages.as_ref(),
                page_indptr: kv.page_indptr.as_ref(),
                w_slot: kv.w_slot.as_ref(),
                w_off: kv.w_off.as_ref(),
                positions: kv.positions.as_ref(),
                mask: kv.mask.as_deref(),
            },
        });
        let borrows: Vec<&crate::working_set::RsWorkingSet> =
            rs.working_sets.iter().map(Rc::as_ref).collect();
        wit_hybrid::ForwardPass::attention(
            &self.wit,
            binding.as_ref(),
            &borrows,
            &wit_hybrid::RsGeometry {
                fold_len: rs.fold_len.as_ref(),
                buffer: rs.buffer.wit(),
            },
        )
    }
}

impl Pass<wit_recurrent::ForwardPass> {
    /// `pie:inferlet/forward-recurrent.attention` — bind the recurrent state:
    /// one working set per request, plus where its folded boundary lands.
    pub fn attention<B>(&self, rs: &[RsWorkingSet], geom: RsGeometry<'_, B>) -> Result<(), String>
    where
        B: RangeBounds<u32>,
    {
        let rs = self.stage_rs(rs, geom)?;
        let borrows: Vec<&crate::working_set::RsWorkingSet> =
            rs.working_sets.iter().map(Rc::as_ref).collect();
        wit_recurrent::ForwardPass::attention(
            &self.wit,
            &borrows,
            &wit_recurrent::RsGeometry {
                fold_len: rs.fold_len.as_ref(),
                buffer: rs.buffer.wit(),
            },
        )
    }
}

// ---------------------------------------------------------------------------
// The three author-facing pass aliases
// ---------------------------------------------------------------------------
// One alias per `pie:inferlet` forward interface; distinct types sharing no impl.

/// `pie:inferlet/forward` — paged, per-token, reversibly discardable KV only.
/// Valid when `model.pass_kind()` is `ForwardKind::Attention`.
pub mod attention {
    /// An attention-only forward pass.
    pub type ForwardPass = super::Pass<super::wit_attention::ForwardPass>;
    pub use super::{run_ahead, submit_frame};

    /// Glob-import surface for attention-only inferlet authors.
    pub mod prelude {
        pub use super::{ForwardPass, run_ahead, submit_frame};
        pub use crate::eta::shared_prelude::*;
    }
}

/// `pie:inferlet/forward-recurrent` — irreversibly folded recurrent state only.
/// `on_attn_proj` / `on_attn` don't exist here — no attention layer to fire on.
pub mod recurrent {
    /// A recurrent-only forward pass.
    pub type ForwardPass = super::Pass<super::wit_recurrent::ForwardPass>;
    pub use super::{run_ahead, submit_frame};

    /// Glob-import surface for recurrent-only inferlet authors.
    pub mod prelude {
        pub use super::{ForwardPass, run_ahead, submit_frame};
        pub use crate::eta::shared_prelude::*;
    }
}

/// `pie:inferlet/forward-hybrid` — attention layers and recurrent layers in
/// one forward (Qwen3.5 GDN, Nemotron-H Mamba2).
pub mod hybrid {
    /// A hybrid forward pass.
    pub type ForwardPass = super::Pass<super::wit_hybrid::ForwardPass>;
    pub use super::{run_ahead, submit_frame};

    /// Glob-import surface for hybrid inferlet authors.
    pub mod prelude {
        pub use super::{ForwardPass, run_ahead, submit_frame};
        pub use crate::eta::shared_prelude::*;
    }
}

