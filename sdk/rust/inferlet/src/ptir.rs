//! `inferlet::ptir` — the author-facing PTIR bridge over the WIT `ptir` surface.
//!
//! This is the only home of the overview §3/§5 author surface
//! (`ForwardPass`/`Pipeline`/`WorkingSet`/`Channel`). `ForwardPass` lives in
//! one of three modules -- [`self::attention`], [`self::recurrent`], [`self::hybrid`] -- one per
//! `pie:inferlet` forward interface, selected by `model::pass_kind()`. It
//! wraps the WIT resources (`channel`, `forward-pass`, `kv-working-set`,
//! `pipeline`) and
//! drives the neutral [`Builder`](pie_dsl::Builder) from the `pie-dsl`
//! crate: the author writes stage closures + port bindings, the bridge lowers
//! them to the canonical PTIR container, orders the WIT channel handles by the
//! builder↔bridge contract
//! ([`Traced::channel_order`](pie_dsl::Traced::channel_order)), and calls
//! the empty `forward-pass` builder and attaches the traced program (which
//! binds against the model — the guest does not bind, D6). Program identity,
//! dedup, and validation happen host-side at program attachment.
//!
//! A [`Channel`] owns BOTH sides: the `pie-dsl` trace declaration (its `take`/
//! `put`/`read` record ops inside a stage closure, and host `put`s record the
//! host-role endpoint) and the WIT `channel` resource (the host transport). The
//! two are constructed from the same `(shape, dtype, capacity)` so the decl
//! validates against the container by construction.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Bound, RangeBounds};
use std::rc::Rc;

use pie_dsl::builder::Builder;
use pie_dsl::channel::PutValue;
use pie_dsl::value::ConstData;
use pie_dsl::{
    Channel as DslChannel, DType, IntoConst, IntoPut, IntoShape, Port, Shape, Stage, Tensor,
};

use crate::pie::inferlet::channel as wit_channel;
use crate::pie::inferlet::forward as wit_attention;
use crate::pie::inferlet::forward_hybrid as wit_hybrid;
use crate::pie::inferlet::forward_recurrent as wit_recurrent;
use crate::pie::inferlet::pipeline as wit_pipeline;
use crate::pie::inferlet::types::Dtype as WitDtype;
use crate::working_set::{KvWorkingSet, PageRange, PageSpan};

pub use pie_dsl::intrinsics;

// Re-export the eDSL vocabulary so an author writes stage closures with a
// single `use inferlet::ptir::<kind>::prelude::*;`.
pub use pie_dsl::DType as Dtype;
pub use pie_dsl::{
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
//
// A stage trace interns channels and yields dense channel ids keyed by the
// dsl channel's gid; `forward-pass.program` wants the WIT handles in that dense
// order. Every channel the author can reference is created via `Channel::new`/
// `from`/`seeded`, so keying the WIT side by gid lets a forward pass resolve
// each `Traced.channel_order` entry. Inferlets are single-threaded (wasm), so
// thread-local registries are sound.
//
// The WIT resource is created on FIRST USE, not at declaration, because its
// capacity is a constructor argument: `Channel::new(..).capacity(n)` would
// otherwise have to throw away the resource it just made -- which is exactly
// what it used to do, wasting one host channel per call and silently dropping
// the staged seed of a `Channel::from(v).capacity(n)`.

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

/// Whether `gid`'s WIT resource exists yet. A declared-but-unused channel has
/// none, which is what makes its capacity still editable.
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

fn to_wit_dtype(d: DType) -> WitDtype {
    match d {
        DType::F32 => WitDtype::F32,
        DType::I32 => WitDtype::I32,
        DType::U32 => WitDtype::U32,
        DType::Bool => WitDtype::Bool,
    }
}

fn dims_of(shape: Shape) -> Vec<u32> {
    shape.dims().to_vec()
}

// ---------------------------------------------------------------------------
// Channel
// ---------------------------------------------------------------------------

/// F8 eager descriptor claim: record the port's endpoint claim on the shared
/// channel state AT PASS CONSTRUCTION (not at first-submit build), so a
/// channel consumed by a later-constructed sibling pass is visible to every
/// pass's host-role derivation. Cross-pass handoffs therefore need only
/// construction order — build every pass sharing a channel before the first
/// submit that touches it — and no annotation. `bound()` binds with
/// `Builder::bind_port_recorded` so the claim is not double-counted.
fn claim_port(port: Port, ch: &Channel) -> DslChannel {
    let dsl = ch.dsl();
    dsl.note_desc_claim(port.consumes());
    dsl
}

/// A GPU-resident bounded queue (overview §1). Owns the `pie-dsl` trace
/// declaration and the WIT `channel` resource. In a stage closure `take`/`read`/
/// `put` record IR ops; on the host `put` stages a value (seed / host-writer
/// cell) and `take_host`/`read_host` await a committed value.
/// A registry-backed COPY TOKEN (F9): the channel's shared state (DSL trace
/// state + WIT handle) lives in thread-local registries keyed by gid, and
/// this token holds only the gid plus immutable metadata. Stage closures
/// capture tokens by value, so closures are `'static`, a `ForwardPass` has
/// no lifetime parameter, and inferlets need no `Box::leak` to satisfy it.
/// Handle lifetime is owned by the registries — which is what makes an
/// explicit endpoint release at finish/close-settle possible later (the W2
/// endpoint-release follow-up; flagged, not implemented).
#[derive(Clone, Copy)]
pub struct Channel {
    gid: u64,
    shape: Shape,
    dtype: DType,
}

/// In-band validity sentinel: a token slot holding `-1` does not exist —
/// it embeds nothing, appends no KV, and advances no position. Envelope
/// shapes stay fixed while `-1` decides which slots are real (shape decides
/// slots, `-1` decides existence, loop-carry decides position).
pub const TOKEN_PAD: i32 = -1;

/// Pad a token window to its fixed envelope with [`TOKEN_PAD`] sentinels.
///
/// Every fire of an envelope-shaped pass must supply exactly the envelope's
/// slot count; the sentinel slots ride along as non-existent. Panics if the
/// window is larger than the envelope — that is a programming error, not a
/// runtime condition.
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

/// Recover the live tokens from an envelope read back from the device,
/// dropping every [`TOKEN_PAD`] slot (interior or trailing).
pub fn unpad_tokens(window: &[i32]) -> Vec<u32> {
    window
        .iter()
        .filter(|&&token| token != TOKEN_PAD)
        .map(|&token| token as u32)
        .collect()
}

impl Channel {
    /// `Channel::new([shape], dtype)` at capacity 1 (overview §1).
    pub fn new(shape: impl IntoShape, dtype: DType) -> Channel {
        Channel::build(shape.into_shape(), dtype, false)
    }

    /// An initially empty channel whose producer is the host.
    ///
    /// Unlike [`Channel::new`], this declares the host-writer endpoint before
    /// the first value is available, so a consuming pass may be submitted
    /// run-ahead and receive the value later.
    pub fn writer(shape: impl IntoShape, dtype: DType) -> Channel {
        let channel = Channel::build(shape.into_shape(), dtype, false);
        channel.dsl().note_host_put();
        channel
    }

    /// The registry-resolved DSL trace state (panics on an unregistered
    /// token — construction always registers, so that is a frontend bug).
    fn dsl(&self) -> DslChannel {
        DslChannel::by_gid(self.gid).expect("channel token resolves in the DSL registry")
    }

    /// The registry-resolved WIT handle.
    fn wit(&self) -> Rc<wit_channel::Channel> {
        lookup_channel(self.gid).expect("channel token resolves in the WIT registry")
    }

    /// Widen the ring to `n` cells (deeper run-ahead).
    ///
    /// The WIT resource takes its capacity at construction, so this must come
    /// before the channel is first used -- which for a seeded channel means
    /// before the seed, since staging it is a use.
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
    /// `v` (overview §1). The seed is instance data (D2): it rides the WIT
    /// channel as a pre-submit `put`, never the container.
    pub fn from(v: impl IntoConst) -> Channel {
        let data: ConstData = v.into_const();
        let ch = Channel::build(data.shape, data.dtype, true);
        ch.wit()
            .put(&data.bytes)
            .expect("stage seed on a fresh channel");
        ch
    }

    /// A seeded channel of a given shape whose seed value is supplied at
    /// instantiation (device loop-carried multi-dim channels, D2).
    pub fn seeded(shape: impl IntoShape, dtype: DType) -> Channel {
        Channel::build(shape.into_shape(), dtype, true)
    }

    /// `Channel::from_shaped([shape], v)` — like [`from`], but reinterprets the
    /// flat seed `v` with the explicit multi-dim `shape` (element counts must
    /// match). `IntoConst` only produces flat 1-D seeds, so use this for a
    /// concrete multi-dim seed (e.g. a `[B, POOL]` bool attention mask) that
    /// downstream ops type against as rank-2.
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

    fn build(shape: Shape, dtype: DType, seeded: bool) -> Channel {
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

    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// `take()` — consume a cell, inside a stage closure. Records a `ChanTake`
    /// and yields the in-program value. The host counterpart is
    /// [`take_host`](Self::take_host); asking a host channel for an in-program
    /// value is an authoring error and is reported as one.
    pub fn take(&self) -> Tensor {
        self.dsl().take()
    }

    /// `read()` — peek a cell (leaves it full), inside a stage closure. Device
    /// counterpart of [`read_host`](Self::read_host).
    pub fn read(&self) -> Tensor {
        self.dsl().read()
    }

    /// Consume a cell on the host, decoded as `T` — a whole `Vec<f32>`, or a
    /// bare `f32` for a channel that holds one value. Awaits in-flight fires;
    /// a poisoned channel returns `Err`.
    ///
    /// `T`'s element type must be the channel's own dtype. The bytes are a raw
    /// little-endian window, so decoding an `f32` channel as `i32` would
    /// reinterpret the bit pattern and return plausible garbage.
    ///
    /// Errors already name the channel, so inferlets do not repeat it in a
    /// [`context`](crate::Context) call.
    ///
    /// ```ignore
    /// let logits = out.take_host::<Vec<f32>>().await?;
    /// let count = n_ch.take_host::<i32>().await?;
    /// ```
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

    /// `"{channel} take"` / `"{channel} read"` — the prefix on every host
    /// readback error, so the message names the channel without the inferlet
    /// spelling it out again.
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

    /// `put(v)` — in a stage closure `v` is an in-program [`Tensor`] (device
    /// side, records a `ChanPut`); on the host `v` is data (staged on the WIT
    /// channel as the next cell / a seed, and the host-writer endpoint is
    /// recorded on the trace side for host-role derivation). Fire-and-forget
    /// (D1: staged puts coalesce into the next submit); a fire that fails
    /// surfaces downstream as an `Err` from [`take_host`](Self::take_host).
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
    /// occupancy. A later value already queued by [`put`](Self::put) is left
    /// untouched. This is a host operation; unlike a stage `put`, it records no
    /// PTIR op.
    pub fn set(&self, v: impl IntoConst) -> Result<(), String> {
        let data: ConstData = v.into_const();
        self.wit().set(&data.bytes)
    }
}

/// Seed a channel straight from an iterator, without materializing a `Vec`
/// first: `Channel::from_iter(0..n)` rather than
/// `Channel::from((0..n).collect::<Vec<_>>())`. Page tables, position vectors
/// and write-slot maps are all built this way.
///
/// `IntoConst` cannot accept iterators — a blanket impl over `IntoIterator`
/// collides with the scalar impls — so the element types are spelled out here
/// instead, which is also what makes `.collect::<Channel>()` infer.
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
///
/// [`DTYPE`](Self::DTYPE) is what makes the raw byte window self-describing:
/// [`Channel::take_host`] refuses a `T` the channel does not hold.
pub trait HostElem: Copy {
    const DTYPE: DType;
    fn decode(raw: &[u8]) -> Vec<Self>;
}
impl HostElem for i32 {
    const DTYPE: DType = DType::I32;
    fn decode(raw: &[u8]) -> Vec<i32> {
        raw.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
impl HostElem for u32 {
    const DTYPE: DType = DType::U32;
    fn decode(raw: &[u8]) -> Vec<u32> {
        raw.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
impl HostElem for f32 {
    const DTYPE: DType = DType::F32;
    fn decode(raw: &[u8]) -> Vec<f32> {
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
impl HostElem for bool {
    const DTYPE: DType = DType::Bool;
    fn decode(raw: &[u8]) -> Vec<bool> {
        raw.iter().map(|&byte| byte != 0).collect()
    }
}

/// A type [`Channel::take_host`] can materialize a channel into: a whole
/// `Vec<T>`, or a bare `T` for a channel that holds a single value.
///
/// The impls are spelled out rather than blanket over [`HostElem`] because
/// `impl<T: HostElem> FromChannel for T` and `for Vec<T>` overlap under
/// coherence.
pub trait FromChannel: Sized {
    /// The dtype the channel must hold.
    const DTYPE: DType;
    fn from_bytes(raw: &[u8]) -> Result<Self, String>;
}

macro_rules! from_channel {
    ($t:ty) => {
        impl FromChannel for Vec<$t> {
            const DTYPE: DType = <$t as HostElem>::DTYPE;
            fn from_bytes(raw: &[u8]) -> Result<Self, String> {
                Ok(<$t as HostElem>::decode(raw))
            }
        }
        impl FromChannel for $t {
            const DTYPE: DType = <$t as HostElem>::DTYPE;
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

/// The attention working set (overview §5.2) — a logical page address space
/// over the runtime's KV mapping trie (kv_refact.md). Wraps the WIT
/// `kv-working-set`. Every page reference on this surface is a
/// WorkingSet-RELATIVE index (never a physical page id); the runtime
/// translates at the kernel through the working set's flattened table.
/// `reserve` is purely logical — no memory is held until a forward writes.
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
    /// range. Purely logical (physical pages are allocated only when a
    /// forward writes them). The grant is per-instance data that flows
    /// through a channel (`fresh.put(ws.reserve(B)?)`), never a trace
    /// constant (D2).
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
    /// `on`. Suffix indexes shift down — publish new PTIR geometry after. A
    /// shared-path interior range errs (growth-boundary invariant).
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

    /// Ordered KV cell move within this working set (Design-B lazy KV
    /// compaction): move `n` token cells, for ALL layers, from
    /// (`src_page_ids[i]`, `src_tok_idx[i]`) to (`dst_page_ids[i]`,
    /// `dst_tok_idx[i]`); the four lists are parallel. Page ids are
    /// WorkingSet-relative indexes; token indices are in-page offsets. Rides
    /// the same run-ahead FIFO as submits on `on` (ordered after prior fires'
    /// writes, before later fires' reads — no barrier). The caller guarantees
    /// disjoint src/dst spans and computes the post-move layout itself.
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
/// per-instance data (D2). Puttable into a channel.
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

/// The runtime's recurrent-state slots for hybrid / linear-attention models
/// (GDN, Mamba2). Wraps the WIT `rs-working-set`. Bind via
/// [`self::recurrent::ForwardPass::attention`] or
/// [`self::hybrid::ForwardPass::recurrent`]
/// -- one per request, in resolved request order. A pure-attention model has
/// no recurrent state, and its pass type has no method that accepts one.
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

    /// Tokens per buffered RS page for this model/driver. A cached
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

    /// Drop the buffered slots at `indices` and densely compact. This is the
    /// REJECT half of fold-commit: a speculative tail that was buffered but
    /// never folded is abandoned by dropping its slots, and no folded state
    /// was ever perturbed by it.
    pub fn free_buffer(&self, indices: &[u32]) -> Result<(), String> {
        self.rs.free_buffer(indices)
    }

    /// Forget the last `count` buffered tokens: they never happened.
    ///
    /// The twin of `fold-len` on the other end of the buffer. A fold moves the
    /// folded boundary RIGHT and cannot be undone; this moves the live end
    /// LEFT and is free, because the slots it releases are overwritten by the
    /// next append. `free_buffer` is the CAPACITY operation; this is the
    /// CONTENT one.
    ///
    /// It is what collapses a speculative window to one fire. The rejected
    /// tail is discarded here, and the NEXT window's fire folds the accepted
    /// prefix while writing its own tokens — instead of a separate commit
    /// fire whose only job was to empty the buffer so `free_buffer` could
    /// take the tail with it.
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

/// The forward-pass resource of one `pie:inferlet` forward interface.
///
/// `pie:inferlet` exposes three of them (`forward`, `forward-recurrent`,
/// `forward-hybrid`) so that attention-only state algorithms cannot be named
/// against a folded recurrent state, and wit-bindgen generates three unrelated
/// Rust resource types to match. This trait is the shape those three share, so
/// [`Pass`]'s tracing and bookkeeping is written once WITHOUT collapsing them
/// into a single type: `Pass<W>` stays as distinct per interface as `W` is.
pub trait PassWit: Sized + 'static {
    fn new() -> Self;

    fn embed(
        &self,
        tokens: &wit_channel::Channel,
        indptr: &wit_channel::Channel,
    ) -> Result<(), String>;

    fn readout(&self, indices: &wit_channel::Channel) -> Result<(), String>;

    /// tart (0.3 re-port): the pass's layer truncation.
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
        wit_attention::ForwardPass::set_max_layers(self, max_layers)
            .map_err(|e| e.to_string())
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
        wit_recurrent::ForwardPass::set_max_layers(self, max_layers)
            .map_err(|e| e.to_string())
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
        wit_hybrid::ForwardPass::set_max_layers(self, max_layers)
            .map_err(|e| e.to_string())
    }
    fn program(&self, bytes: &[u8], channels: &[&wit_channel::Channel]) -> Result<(), String> {
        wit_hybrid::ForwardPass::program(self, bytes, channels)
    }
    fn submit(on: &wit_pipeline::Pipeline, slots: &[Option<&Self>]) -> Result<(), String> {
        wit_hybrid::submit(on, slots)
    }
}

/// The PEFT adapter surface's expression vocabulary
/// ([`Pass::adapter`]): a tiny CLOSED language over the site's input `x`
/// and base output `y`, classified — never interpreted — into the
/// driver's CORRECTION lowerings. `mm` multiplies by a channel-borne
/// weight; `+` composes. Scale/bias join with their forms. (0.3 re-port
/// of the 0.2 surface; lowering is guest-side into prologue sinks, so no
/// WIT change rides along.)
pub mod adapter {
    use super::Channel;

    /// Model projection sites, the llama-like bit vocabulary
    /// (driver/cuda/src/model/lora.hpp `LoraSite` — placement is
    /// structure; the driver refuses unconsumed sites loudly).
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

    /// One adapter expression node. Built by the closure, consumed by
    /// the classifier; never executed directly.
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

    /// `mm(w, e)` — multiply the expression by the channel-borne weight
    /// (the channel's leading dim is the layer axis; rank rides its
    /// shape).
    pub fn mm(w: &Channel, e: Expr) -> Expr {
        Expr {
            kind: ExprKind::Mm(*w, Box::new(e)),
        }
    }

    /// `scale(e, l)` — elementwise multiply by the channel-borne vector
    /// `l: [num_layers, d_out]` (IA3's form; any static scale folds into
    /// `l`'s contents).
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

/// The forward-pass builder over the `W` interface. Guests name it through the
/// per-interface aliases in [`attention`] / [`recurrent`] / [`hybrid`], which
/// is what makes a cross-kind helper a COMPILE error rather than a silent
/// semantic bug.
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

/// A [`KvGeometry`] with its ports claimed and its channels resolved to WIT
/// handles, held only long enough for the caller to issue its own interface's
/// state-binding call. The handles are `Rc` because [`Channel::wit`] hands
/// out shared ownership, and the WIT record borrows them.
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

/// The attention geometry of one fire — a field-for-field mirror of the WIT
/// `kv-geometry` record (`forward.wit`, `forward-hybrid.wit`).
///
/// The record exists in WIT so the whole group can be handled as a unit; it
/// exists here for the same reason, and because six consecutive `&Channel`
/// arguments carrying five different `u32` meanings is a swap waiting to
/// happen. Nothing is hidden: every field is exactly the channel the host
/// binds to that PTIR port.
///
/// The two page spans take any `u32` range, which is what `page-span`'s
/// `{ start, end: option<u32> }` means: `..` is the whole working set,
/// `first..` is a suffix, `a..b` a window.
///
/// ```ignore
/// fwd.attention(&ws, KvGeometry {
///     readable_pages: ..,
///     writable_pages: (n / page_size)..,
///     kv_len: &kv_len,
///     pages: &pages,
///     page_indptr: &page_indptr,
///     w_slot: &w_slot,
///     w_off: &w_off,
///     positions: &positions,
///     mask: None,
/// })?;
/// ```
pub struct KvGeometry<'a, R, W> {
    pub readable_pages: R,
    pub writable_pages: W,
    pub kv_len: &'a Channel,
    pub pages: &'a Channel,
    pub page_indptr: &'a Channel,
    pub w_slot: &'a Channel,
    pub w_off: &'a Channel,
    pub positions: &'a Channel,
    /// `None` omits PTIR's AttnMask port; `Some` binds that channel to it.
    pub mask: Option<&'a Channel>,
}

/// Where the bound recurrent state's folded boundary lands for one fire — a
/// field-for-field mirror of the WIT `rs-geometry` record
/// (`forward-recurrent.wit`, `forward-hybrid.wit`).
///
/// ```ignore
/// // Fold everything: the ordinary decode.
/// RsGeometry { fold_len: None, buffer: .. }
/// // Fold nothing: buffer this fire's tokens, which is what makes a linear
/// // model speculatable -- unfolded tokens cost nothing to abandon.
/// RsGeometry { fold_len: Some(&zero), buffer: ..n }
/// ```
pub struct RsGeometry<'a, B> {
    /// A channel carrying, per request, how far the folded boundary advances
    /// over `[buffer | this fire's tokens]`, clamped to that tail. `None` is
    /// the fold-everything channel of `u32::MAX` the SDK would otherwise make
    /// every caller build by hand.
    ///
    /// A channel rather than a scalar so that a device-computed accepted count
    /// can drive the fold directly, fusing verify and commit into one fire:
    /// the driver resolves it as a descriptor port, so the count never
    /// round-trips through the host.
    pub fold_len: Option<&'a Channel>,
    /// Capacity grant, not an address. The buffer's addressing is derived by
    /// the runtime from its own occupancy; a guest copy of it could only
    /// agree or be refused.
    pub buffer: B,
}

/// A KV working set together with the geometry it is read and written
/// through — a field-for-field mirror of the WIT `kv-binding` record
/// (`forward-hybrid.wit`), which exists so that the pair can be made optional
/// as a unit.
pub struct KvBinding<'a, R, W> {
    pub working_set: &'a WorkingSet,
    pub geometry: KvGeometry<'a, R, W>,
}

thread_local! {
    /// The `fold-len` of a pass that folds unconditionally: `u32::MAX` clamps
    /// to "every token of every fire". Degenerate but MEANINGFUL -- the fast
    /// path stated explicitly, not the dummy geometry this surface avoids.
    static FOLD_ALL: Channel = Channel::from(vec![u32::MAX]);
}

/// [`RsGeometry`] with its port claimed and its channel resolved, alongside the
/// working sets it applies to. See [`StagedKv`].
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
    fn preserves_open_ends() {
        let all = PageDeclaration::from_range(..).unwrap();
        assert_eq!((all.start, all.end), (0, None));

        let tail = PageDeclaration::from_range(7..).unwrap();
        assert_eq!((tail.start, tail.end), (7, None));
    }

    #[test]
    fn normalizes_inclusive_and_exclusive_bounds() {
        let closed = PageDeclaration::from_range(2..5).unwrap();
        assert_eq!((closed.start, closed.end), (2, Some(5)));

        let inclusive = PageDeclaration::from_range(2..=5).unwrap();
        assert_eq!((inclusive.start, inclusive.end), (2, Some(6)));
    }

    #[test]
    fn rejects_reversed_closed_spans() {
        assert!(PageDeclaration::from_range(5..4).is_err());
    }

    #[test]
    fn rejects_bound_overflow() {
        assert!(
            PageDeclaration::from_range((Bound::Excluded(u32::MAX), Bound::Unbounded)).is_err()
        );
        assert!(PageDeclaration::from_range(..=u32::MAX).is_err());
    }
}

impl<W: PassWit> Pass<W> {
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

    /// Claim the KV geometry ports and resolve its channels. The caller issues
    /// its own interface's state-binding call with the result: `forward` takes
    /// the working set and geometry directly, `forward-hybrid` wraps them in a
    /// `kv-binding` alongside the recurrent half.
    ///
    /// After the program is attached this is a REBIND -- a request-set or
    /// geometry change between fires -- and the ports are already traced into
    /// the program, so only the WIT-side handles are refreshed.
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
            // `u32::MAX` per request clamps to "the whole tail" host-side, so
            // one broadcast element says fold everything for every request.
            // Minted once per guest thread: a rebind must not leak a fresh
            // channel every time the request set changes. No port is claimed
            // either -- a pass that folds everything unconditionally has
            // nothing to compute, and a dead port would land in every plain
            // recurrent trace.
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

    /// tart (0.3 re-port): run only the first `max_layers` transformer
    /// layers for this pass's fires and take the head there (the
    /// layerskip-draft / logit-lens class). Call before `program`.
    pub fn set_max_layers(&self, max_layers: u32) -> Result<(), String> {
        self.wit.set_max_layers(max_layers)
    }

    /// Attach the `prologue` stage (overview §5.3).
    /// Attach a PEFT adapter at `site`: `f` receives the site's input `x`
    /// and base output `y` and returns the corrected output as an
    /// [`adapter`] expression. v0 lowers three forms — LoRA
    /// `y + mm(b, mm(a, x))`, IA3 `scale(y, l)`, and the DoRA composite
    /// `scale(y + mm(b, mm(a, x)), s)` — into prologue sinks the driver
    /// resolves per layer. One adapter per site per pass.
    pub fn adapter(
        &self,
        site: adapter::Site,
        f: impl FnOnce(adapter::Expr, adapter::Expr) -> adapter::Expr,
    ) -> Result<(), String> {
        use adapter::ExprKind as K;
        let expr = f(adapter::Expr::x(), adapter::Expr::y());
        // DoRA: scale(y + mm(b, mm(a, x)), s) — the composite lowers to
        // the low-rank sink THEN the scale sink on the same site (the
        // driver applies every scale after every delta).
        if let K::Scale(l, inner) = &expr.kind {
            if let K::Add(lhs, rhs) = &inner.kind {
                let delta = match (&lhs.kind, &rhs.kind) {
                    (K::Y, _) => &rhs.kind,
                    (_, K::Y) => &lhs.kind,
                    _ => &inner.kind, // falls to the refusal below
                };
                if let K::Mm(b, mid) = delta {
                    if let K::Mm(a, x) = &mid.kind {
                        if matches!(x.kind, K::X) {
                            let (a, b, l) = (*a, *b, *l);
                            {
                                let mut st = self.inner.borrow_mut();
                                if (st.adapter_lowrank_sites | st.adapter_scale_sites)
                                    & site.bit()
                                    != 0
                                {
                                    return Err(format!(
                                        "adapter: site {site:?} already carries an \
                                         adapter on this pass"
                                    ));
                                }
                                st.adapter_lowrank_sites |= site.bit();
                                st.adapter_scale_sites |= site.bit();
                            }
                            self.prologue(move || {
                                intrinsics::kernel::lora(
                                    a.read(),
                                    b.read(),
                                    Tensor::constant(site.bit()),
                                );
                                intrinsics::kernel::adapter_scale(
                                    l.read(),
                                    Tensor::constant(site.bit()),
                                );
                            });
                            return Ok(());
                        }
                    }
                }
            }
        }
        // The SCALE form (IA3): scale(y, l) — lowered to the 2-argument
        // adapter sink.
        if let K::Scale(l, inner) = &expr.kind {
            if matches!(inner.kind, K::Y) {
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
                    intrinsics::kernel::adapter_scale(
                        l.read(),
                        Tensor::constant(site.bit()),
                    );
                });
                return Ok(());
            }
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
            // One pair per SITE (the per-site rung): each call emits its
            // own lora sink; the driver enforces the same disjointness at
            // resolution.
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

    /// Enqueue this pass as a SINGLE-SLOT FRAME on `on`: slot 0 is this pass,
    /// the other k−1 slots pad to no-ops.
    ///
    /// This is the ONE-SHOT path — a prefill chunk, a partial trailing frame,
    /// or a fire the runtime submits solo (a recurrent pass whose `fold-len`
    /// differs from the fast path). The padding
    /// is unconditional, not a fallback when slots run out, so at k > 1 this
    /// spends a whole frame on a single pass: exactly as many frame boundaries
    /// per token as k = 1, with none of k's batching benefit.
    ///
    /// A decode loop should NOT call this. Use [`run_ahead`], which fills
    /// [`live_slots`] per frame and sizes its window from
    /// [`channel_capacity`], or [`submit_frame`] to drive frames by hand.
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

/// Waves per frame (k) for this deployment — the static constant
/// `forward.submit` sizes its slot list to (cached; fixed at engine start,
/// exactly like the KV page size). Guests must be output-correct for any k.
pub fn frame_size() -> usize {
    thread_local! {
        static FRAME_SIZE: std::cell::OnceCell<usize> = const { std::cell::OnceCell::new() };
    }
    FRAME_SIZE.with(|k| *k.get_or_init(|| crate::model::frame_size().max(1) as usize))
}

/// How long a pipeline may hold a frame's wait-set without submitting before
/// the engine stops waiting for it (cached; static, like `frame_size`).
///
/// Measures consecutive silence while actually blocking a seal, and stops
/// while the engine is what you are waiting on — so a pipeline that keeps
/// submitting can never trip it. To stop legitimately, call `Pipeline::park`.
///
/// Overrunning it is not fatal: the slot leaves the frame, work already
/// submitted still runs, and the next submit rejoins. It costs a boundary,
/// not the instance. (Going silent for orders of magnitude longer without
/// parking is read as an abandoned pipeline and is terminated.)
///
/// Note that work between receiving a result and submitting the next fire is
/// on this clock: the fleet is stalled on you and nothing is owed back. Park
/// around anything that can outlast the bound — tokenizing a large document,
/// compiling a grammar, an RPC, waiting on a user turn — and simply submit
/// again to rejoin. Parking is cheap and unparking is implicit.
pub fn submit_deadline() -> std::time::Duration {
    thread_local! {
        static DEADLINE: std::cell::OnceCell<u64> = const { std::cell::OnceCell::new() };
    }
    std::time::Duration::from_micros(
        DEADLINE.with(|d| *d.get_or_init(crate::model::submit_deadline_us)),
    )
}

/// Host-reader channel capacity, in cells, that sustains the engine's
/// run-ahead for one lane. Size every host-reader channel to at least this.
///
/// Deliberately NOT cached, unlike [`frame_size`]: `frame-size` is promised to
/// be a static deployment constant, this one is not — it derives from the host
/// resubmit turnaround, which the runtime may later adapt.
pub fn channel_capacity() -> usize {
    (crate::model::channel_capacity() as usize).max(2)
}

/// Live slots per frame for the bound model: how many of the k slots a lane
/// can actually fill with work.
///
/// k for a dense model, 1 for a recurrent (linear/hybrid) one. That 1 is now
/// CONSERVATISM, not a constraint — see the body.
///
/// Kept as its own query rather than folded into [`frame_size`]: it answers
/// "how much can this lane submit per frame", which is a model property that
/// has diverged from k before and may again. It is NOT the place to encode
/// per-PASS restrictions — a fire that buffers or commits is submitted solo by
/// the runtime because its `fold-len` picks the RS execution mode for the whole
/// composed batch, and that is a property of the fire, not of the model.
pub fn live_slots() -> usize {
    // k for a dense model, 1 for a recurrent (linear/hybrid) one.
    //
    // The reason this was ever 1 is GONE. A run-ahead lane chains slot i+1's
    // token off slot i; an RS fire used to resolve its descriptor ports on the
    // HOST at frame entry, before any slot had run, because the device-composed
    // template refused every fire carrying RS rows. `299b76320` lifted that:
    // an unbuffered fold-all recurrent decode — the ordinary shape — takes the
    // template and resolves at KERNEL time, so the chained value is there. The
    // engine's refusal was narrowed to match (`validate_frame` in
    // `runtime/engine/src/pipeline/fire.rs`), and `cuda_rs_buffer_bench`'s
    // `coframe` arm proves the shape runs on device with an identical
    // trajectory — and 31.7% faster than the 1-wide arm.
    //
    // It stays 1 only because raising it is a scheduling change that has to be
    // benchmarked as one, not smuggled in with a correctness fix. A lane that
    // wants the intra-frame run-ahead can already take it with `submit_frame`.
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

/// Tokens per KV page (cached) — the divisor in every page-geometry
/// expression. Also reachable as [`WorkingSet::page_size`], which is the
/// spelling to prefer when a working set is in hand.
pub fn kv_page_size() -> u32 {
    thread_local! {
        static PAGE: std::cell::OnceCell<u32> = const { std::cell::OnceCell::new() };
    }
    PAGE.with(|c| *c.get_or_init(crate::model::kv_page_size))
}

/// Max embed tokens in a single pass (C) — the guest-side prefill chunk
/// budget (cached). Split a prompt of L tokens into `ceil(L / C)` chunks, or
/// let [`prefill_chunks`] do it, which is what you want.
pub fn max_embed_length() -> usize {
    thread_local! {
        static MAX_EMBED: std::cell::OnceCell<usize> = const { std::cell::OnceCell::new() };
    }
    MAX_EMBED.with(|c| *c.get_or_init(|| crate::model::max_embed_length().max(1) as usize))
}

/// The `[start, end)` spans a prompt of `n` tokens must be prefilled in.
///
/// The driver's per-launch token capacity ([`max_embed_length`]) is a hard
/// structural limit, so any prompt longer than it has to be split. This is the
/// split to use, and the reason it is here rather than in each guest is that
/// the obvious version is subtly wrong.
///
/// Chunking "C tokens at a time until the remainder" puts the entire remainder
/// on the LAST chunk, which can leave it a single token. That is harmless for a
/// policy that only writes KV, but it is not harmless for a policy that
/// OBSERVES the prefill: an attention-score capture records the last `window`
/// query rows of the fire it is attached to, and a final chunk shorter than
/// `window` silently truncates the observation to whatever was left over. The
/// resulting ranking is plausible and wrong, which is the worst combination.
///
/// So the remainder is spread over the FIRST chunks instead: with
/// `k = ceil(n / C)`, chunk `i` gets `n/k + (i < n mod k)` tokens. The chunk
/// count is identical, every chunk is within one token of every other, and the
/// final chunk is `floor(n / k)` -- the largest a last chunk can be.
///
/// `cap` overrides the driver limit (clamped to it); pass `None` for the
/// default. A smaller cap is useful in tests, where forcing the multi-chunk
/// path on a short prompt is the only practical way to check that concatenating
/// the chunks reproduces the one-shot fire.
///
/// Returns an empty vector for `n == 0`.
pub fn prefill_chunks(n: u32, cap: Option<u32>) -> Vec<(u32, u32)> {
    let cap = cap
        .unwrap_or(u32::MAX)
        .min(max_embed_length().max(1) as u32);
    even_spans(n, cap)
}

/// The arithmetic of [`prefill_chunks`], with the driver limit already applied.
/// Split out so it is testable off-device: `max_embed_length` reaches the host.
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

/// Submit ONE FRAME on `on`: up to `frame_size()` ordered slots, slot i
/// executing in wave i; missing trailing slots are padded with no-ops. The
/// same pass may repeat across slots (a plain decode frame is the same pass
/// in every slot) and slots may be heterogeneous (prefill chunks first, then
/// decode). First submit of a pass traces and attaches its program;
/// attachment, bind, and frame-validation errors surface here.
///
/// A frame is monomorphic in its slot type by construction: `W` is one
/// interface, so a mixed frame does not typecheck.
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

/// Keeps the engine's run-ahead window full while `on_token` consumes results,
/// submitting `pass` on `on` until `budget` fires have been submitted or
/// `on_token` returns [`ControlFlow::Break`].
///
/// Nothing here is hidden — it is the loop you would otherwise hand-write:
///
/// ```ignore
/// let r      = ptir::live_slots();
/// let frames = (ptir::channel_capacity() - 1) / r;
/// // prime `frames` frames of `r` slots; refill one frame per `r` results
/// ```
///
/// with the two mistakes that loop invites removed:
///
/// - the window is counted in FRAMES, not fires. A fire-counted window
///   overshoots by a whole frame, and — worse — shrinks in real work as k
///   shrinks, which is the unit error that collapsed k = 1 throughput.
/// - `budget` bounds submission, so stopping early does not strand a full
///   window of already-submitted fires.
///
/// `on_token` is called once per completed fire, in submission order; it is
/// where the guest takes its channels and does its host-side work. Returning
/// `Break` stops submission immediately. Up to one window of fires may still
/// be in flight at that point — their cells are simply never taken, and
/// [`Pipeline::close`] reclaims them.
///
/// Returns the number of times `on_token` ran.
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
    let r = live_slots();
    // `channel_capacity()` carries the staging margin, so the window is what
    // remains once that margin is set aside — in FRAMES of `r` live slots.
    // Dividing by `r` rather than assuming k keeps the ring just as full for
    // any lane whose live width has been narrowed below k.
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

    while consumed < submitted {
        if on_token().await? == ControlFlow::Break(()) {
            return Ok(consumed + 1);
        }
        consumed += 1;
        // Refill a whole frame at a time: `submit_frame` is the only way to
        // publish, and a partial frame cannot be topped up after the fact.
        if submitted < budget && submitted - consumed <= (window_frames - 1) * r {
            submit_one_frame(&mut submitted)?;
        }
    }
    Ok(consumed)
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// A run-ahead ordering domain (overview §3, pipeline.wit) — every command on
/// it linearizes in submission order through the per-driver sequencer.
/// Ordering across fires is carried by the channels' full/empty bits, not
/// host code. Working-set mutators ([`WorkingSet::fork`]/`slice`/`discard`/
/// `copy_into`) and every pass `submit` take `&Pipeline`.
///
/// # Canonical usage (one pipeline per sequential stream)
///
/// A `Pipeline` is an ordering domain, not a program: heterogeneous passes
/// (an N-wide prefill, then a loop-carried decode) are ONE sequential
/// stream and belong on ONE pipeline — never split phases of the same
/// stream across pipelines. Call [`Pipeline::close`] right after the last
/// submit; already-submitted run-ahead fires settle normally and remain
/// take-able. Separate pipelines are for
/// genuinely CONCURRENT streams only (draft vs target model in speculative
/// decoding, parallel beam branches, independent requests) — close each
/// stream when it will accept no more submissions.
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
    /// Already-submitted fires drain to settlement in FIFO order and remain
    /// take-able; later submissions fail. Dropping a pipeline is identical.
    pub fn close(&self) {
        self.wit.close();
    }

    /// Leave the frame wait-set until this pipeline submits again — the way
    /// to go idle, block on a user turn, or wait on a peer without holding
    /// the frame, without running down `submit_deadline`, and without ever
    /// looking abandoned however long you stay away.
    ///
    /// Ordered against this pipeline's own submits, so outstanding fires are
    /// fine: their results still arrive, and the exit follows them. The next
    /// `submit` rejoins automatically; there is no explicit rejoin. Unlike
    /// `close`, the pipeline stays usable.
    pub fn park(&self) {
        // Every forward interface declares the same `park`; the engine
        // implements all three with one function, so the pass kind (which we
        // do not have here — park carries no pass) is irrelevant.
        wit_attention::park(&self.wit);
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Pipeline::new()
    }
}

// ---------------------------------------------------------------------------
// prelude
// ---------------------------------------------------------------------------

/// The kind-independent half of every pass prelude: the eDSL vocabulary plus
/// the wrapper types that mean the same thing in all three interfaces.
///
/// There is deliberately NO top-level `ptir::prelude`. Importing a pass type
/// now requires naming its kind, which is what stops an attention-only
/// algorithm from silently compiling against a hybrid model.
pub mod shared_prelude {
    pub use super::{
        Channel, KvBinding, KvGeometry, PageGrant, Pipeline, RsGeometry, RsWorkingSet, TOKEN_PAD,
        WorkingSet, channel_capacity, frame_size, kv_page_size, live_slots, max_embed_length,
        pad_tokens, prefill_chunks, unpad_tokens,
    };
    /// Every inferlet returns `inferlet::Result` from `#[inferlet::main]` and
    /// asks the model for its tokenizer and vocabulary, so both come with the
    /// prelude rather than being imported by hand 34 times over.
    pub use crate::{Context, Result, model};
    /// Only `Stage`: dtypes are spelled `dtype::f32` and friends, one
    /// lowercase spelling for the one thing they name.
    pub use pie_dsl::Stage;
    pub use pie_dsl::dtype;
    pub use pie_dsl::intrinsics;
    /// The arithmetic intrinsics are deliberately absent: `+ - * / %` and
    /// unary `-` are their spelling, and one spelling is the point.
    pub use pie_dsl::value::{
        Tensor, abs, and, broadcast, cast, causal_mask, cummass_le, cumprod, cumsum, entropy,
        entropy_from_logprobs, eq, exp, gather, gather_row, ge, gt, gumbel, gumbel_max, indptr,
        iota, l2norm, le, log, log_softmax, lt, mask_apply, masked_argmax, matmul, max_elem,
        min_elem, ne, not, nucleus_sample, or, pivot_threshold, prob_ge, rank_le, recip,
        reduce_argmax, reduce_max, reduce_min, reduce_sum, reshape, rng, row_membership,
        scalar_gather, scatter_add, scatter_set, select, sign, sink_window_mask,
        sliding_window_mask, softmax, sort_desc, top_k, transpose,
    };
    pub use std::ops::ControlFlow;
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

    /// `pie:inferlet/forward-hybrid.attention` — bind BOTH halves of the state
    /// this pass fires over. REQUIRED.
    ///
    /// One call because they are one binding: the same forward reads paged KV
    /// in its attention layers and a folded state in its recurrent ones, and
    /// the runtime has to see them together to place the fire. `kv` is an
    /// [`Option`] so a recurrent-only commit fire is expressible without
    /// inventing dummy attention geometry.
    ///
    /// Calling it again after the first submit REBINDS -- the request set or
    /// the fold boundary changed between fires -- which is how a beam fork or
    /// a speculative commit re-aims an already-traced pass.
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
    /// one working set per request, in resolved request order, plus where its
    /// folded boundary lands. REQUIRED.
    ///
    /// The mechanism is a recurrence, but the SLOT NAME is `attention` in all
    /// three interfaces: the role inside a pass is the same regardless of
    /// mechanism, and only the signature varies.
    ///
    /// Calling it again after the first submit REBINDS.
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
//
// One alias per `pie:inferlet` forward interface. They are DISTINCT types --
// `Pass<A>` and `Pass<H>` share no impl -- which is the correctness gate this
// split exists for: a helper written for attention-only KV eviction (H2O,
// SnapKV, TOVA, Quest) cannot even name a hybrid model's pass.

/// `pie:inferlet/forward` — paged, per-token, reversibly discardable KV only.
///
/// Valid when `model.pass_kind()` is [`ForwardKind::Attention`]. KV eviction
/// and reordering algorithms belong HERE and nowhere else: evicting a KV page
/// is reversible in a way that unfolding a recurrent state is not.
///
/// [`ForwardKind::Attention`]: crate::model::ForwardKind
pub mod attention {
    /// An attention-only forward pass.
    pub type ForwardPass = super::Pass<super::wit_attention::ForwardPass>;
    pub use super::{run_ahead, submit_frame};

    /// Glob-import surface for attention-only inferlet authors.
    pub mod prelude {
        pub use super::{ForwardPass, run_ahead, submit_frame};
        pub use crate::ptir::shared_prelude::*;
    }
}

/// `pie:inferlet/forward-recurrent` — irreversibly folded recurrent state only.
///
/// Valid when `model.pass_kind()` is [`ForwardKind::Recurrent`]: a model with
/// no attention layers anywhere. `on_attn_proj` / `on_attn` do not exist here
/// because there is no attention layer for them to fire on.
///
/// [`ForwardKind::Recurrent`]: crate::model::ForwardKind
pub mod recurrent {
    /// A recurrent-only forward pass.
    pub type ForwardPass = super::Pass<super::wit_recurrent::ForwardPass>;
    pub use super::{run_ahead, submit_frame};

    /// Glob-import surface for recurrent-only inferlet authors.
    pub mod prelude {
        pub use super::{ForwardPass, run_ahead, submit_frame};
        pub use crate::ptir::shared_prelude::*;
    }
}

/// `pie:inferlet/forward-hybrid` — attention layers and recurrent layers in
/// ONE forward (Qwen3.5 GDN, Nemotron-H Mamba2).
///
/// Valid when `model.pass_kind()` is [`ForwardKind::Hybrid`]. All five stages
/// are legal. KV eviction algorithms are NOT valid here: dropping a KV page
/// does not undo the fold that already consumed those tokens.
///
/// [`ForwardKind::Hybrid`]: crate::model::ForwardKind
pub mod hybrid {
    /// A hybrid forward pass.
    pub type ForwardPass = super::Pass<super::wit_hybrid::ForwardPass>;
    pub use super::{run_ahead, submit_frame};

    /// Glob-import surface for hybrid inferlet authors.
    pub mod prelude {
        pub use super::{ForwardPass, run_ahead, submit_frame};
        pub use crate::ptir::shared_prelude::*;
    }
}

#[cfg(test)]
mod prefill_chunk_tests {
    use super::even_spans;

    fn lens(n: u32, cap: u32) -> Vec<u32> {
        even_spans(n, cap).into_iter().map(|(a, b)| b - a).collect()
    }

    #[test]
    fn covers_the_prompt_exactly_and_contiguously() {
        for n in [1u32, 2, 7, 16, 37, 1302, 8192, 8193, 15032, 16385] {
            for cap in [1u32, 3, 16, 37, 128, 999, 8192, u32::MAX] {
                let spans = even_spans(n, cap);
                assert_eq!(spans[0].0, 0, "n={n} cap={cap}");
                assert_eq!(spans[spans.len() - 1].1, n, "n={n} cap={cap}");
                for w in spans.windows(2) {
                    assert_eq!(w[0].1, w[1].0, "gap/overlap at n={n} cap={cap}");
                }
                for &(a, b) in &spans {
                    assert!(a < b, "empty chunk at n={n} cap={cap}");
                    assert!(b - a <= cap.max(1), "over cap at n={n} cap={cap}");
                }
            }
        }
    }

    #[test]
    fn uses_the_fewest_chunks_the_cap_allows() {
        for n in [1u32, 37, 1302, 8192, 8193, 15032, 16385] {
            for cap in [1u32, 16, 37, 8192] {
                assert_eq!(
                    even_spans(n, cap).len() as u32,
                    n.div_ceil(cap.min(n).max(1)),
                    "n={n} cap={cap}"
                );
            }
        }
    }

    #[test]
    fn every_chunk_is_within_one_token_of_every_other() {
        // The property SnapKV depends on: the last chunk is never a sliver,
        // because a final chunk shorter than the capture window silently
        // truncates the observation. Greedy chunking fails this -- n=1302,
        // cap=37 gives 35 chunks of 37 and a final chunk of 7.
        for n in [1u32, 7, 37, 1302, 8193, 15032, 16385, 65537] {
            for cap in [1u32, 3, 16, 37, 128, 999, 8192] {
                let l = lens(n, cap);
                let (lo, hi) = (*l.iter().min().unwrap(), *l.iter().max().unwrap());
                assert!(hi - lo <= 1, "n={n} cap={cap}: lengths span {lo}..={hi}");
                assert_eq!(
                    *l.last().unwrap(),
                    lo,
                    "n={n} cap={cap}: tail is not the min"
                );
            }
        }
    }

    #[test]
    fn the_greedy_sliver_is_the_case_this_exists_for() {
        assert_eq!(*lens(1302, 37).last().unwrap(), 36);
        assert_eq!(lens(1302, 37).len(), 36);
        // Greedy would have been 35 chunks of 37 plus a final chunk of 7.
        assert_eq!(1302 - 35 * 37, 7);
    }

    #[test]
    fn a_prompt_within_the_cap_is_a_single_chunk() {
        assert_eq!(even_spans(8192, 8192), vec![(0, 8192)]);
        assert_eq!(even_spans(1, 8192), vec![(0, 1)]);
        assert_eq!(even_spans(0, 8192), vec![]);
    }
}
