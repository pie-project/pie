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
use pie_dsl::value::{Arg, ConstData};
use pie_dsl::{
    AsTensor, Channel as DslChannel, DType, IntoConst, IntoPut, IntoShape, Port, Shape, Stage,
    Tensor,
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
    entropy_from_logprobs, eq, exp, gather, gather_row, ge, gt, gumbel, gumbel_max, iota, l2norm,
    le, log, log_softmax, lt, mask_apply, masked_argmax, matmul, max_elem, min_elem, mul, ne, neg,
    not, nucleus_sample, or, pivot_threshold, prob_ge, rank_le, recip, reduce_argmax, reduce_max,
    reduce_min, reduce_sum, rem, reshape, rng, row_membership, scalar_gather, scatter_add,
    scatter_set, select, sign, sink_window_mask, sliding_window_mask, softmax, sort_desc, sub,
    top_k, transpose,
};

// ---------------------------------------------------------------------------
// gid -> WIT channel registry
// ---------------------------------------------------------------------------
//
// A stage trace interns channels and yields dense channel ids keyed by the
// dsl channel's gid; `forward-pass.program` wants the WIT handles in that dense
// order. Every channel the author can reference is created via `Channel::new`/
// `from`/`seeded`, so registering (gid -> Rc<wit_channel::Channel>) at construction
// lets a forward pass resolve each `Traced.channel_order` entry. Inferlets are
// single-threaded (wasm), so a thread-local registry is sound.

thread_local! {
    static WIT_CHANNELS: RefCell<HashMap<u64, Rc<wit_channel::Channel>>> = RefCell::new(HashMap::new());
}

fn register_channel(gid: u64, wit: Rc<wit_channel::Channel>) {
    WIT_CHANNELS.with(|m| {
        m.borrow_mut().insert(gid, wit);
    });
}

fn lookup_channel(gid: u64) -> Option<Rc<wit_channel::Channel>> {
    WIT_CHANNELS.with(|m| m.borrow().get(&gid).cloned())
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
/// cell) and `Taken::get().await`/`Taken::bytes().await` materialize a committed value.
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
        Channel::build(shape.into_shape(), dtype, 1, false)
    }

    /// An initially empty channel whose producer is the host.
    ///
    /// Unlike [`Channel::new`], this declares the host-writer endpoint before
    /// the first value is available, so a consuming pass may be submitted
    /// run-ahead and receive the value later.
    pub fn writer(shape: impl IntoShape, dtype: DType) -> Channel {
        let channel = Channel::build(shape.into_shape(), dtype, 1, false);
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
    pub fn capacity(self, n: u32) -> Channel {
        let dsl = self.dsl().capacity(n);
        let wit = Rc::new(wit_channel::Channel::new(
            &dims_of(self.shape),
            to_wit_dtype(self.dtype),
            n,
        ));
        register_channel(dsl.gid(), wit);
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
        let ch = Channel::build(data.shape, data.dtype, 1, true);
        ch.wit()
            .put(&data.bytes)
            .expect("stage seed on a fresh channel");
        ch
    }

    /// A seeded channel of a given shape whose seed value is supplied at
    /// instantiation (device loop-carried multi-dim channels, D2).
    pub fn seeded(shape: impl IntoShape, dtype: DType) -> Channel {
        Channel::build(shape.into_shape(), dtype, 1, true)
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
        let ch = Channel::build(shape, data.dtype, 1, true);
        ch.wit()
            .put(&data.bytes)
            .expect("stage seed on a fresh channel");
        ch
    }

    fn build(shape: Shape, dtype: DType, capacity: u32, seeded: bool) -> Channel {
        let dsl = if seeded {
            DslChannel::seeded(shape, dtype)
        } else {
            DslChannel::new(shape, dtype)
        };
        let dsl = if capacity != 1 {
            dsl.capacity(capacity)
        } else {
            dsl
        };
        let wit = Rc::new(wit_channel::Channel::new(
            &dims_of(shape),
            to_wit_dtype(dtype),
            capacity,
        ));
        let gid = dsl.gid();
        register_channel(gid, wit);
        Channel { gid, shape, dtype }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// `take()` — consume a cell. In a stage closure: records a `ChanTake` and
    /// yields an in-program value ([`AsTensor`]). On the host: [`Taken::get`]
    /// awaits the committed value (awaits until a fire fills it; poison ⇒
    /// `Err`).
    pub fn take(&self) -> Taken {
        Taken {
            dsl: self.dsl().take(),
            wit: self.wit(),
            mode: TakenMode::Take,
            dtype: self.dtype,
        }
    }

    /// `read()` — peek a cell (leaves it full). Same dual as [`take`](Self::take).
    pub fn read(&self) -> Taken {
        Taken {
            dsl: self.dsl().read(),
            wit: self.wit(),
            mode: TakenMode::Read,
            dtype: self.dtype,
        }
    }

    /// `put(v)` — in a stage closure `v` is an in-program [`Tensor`] (device
    /// side, records a `ChanPut`); on the host `v` is data (staged on the WIT
    /// channel as the next cell / a seed, and the host-writer endpoint is
    /// recorded on the trace side for host-role derivation). Fire-and-forget
    /// (D1: staged puts coalesce into the next submit); a fire that fails
    /// surfaces downstream as poison at [`Taken::get`]/[`Taken::bytes`].
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

/// The result of [`Channel::take`]/[`Channel::read`]. In a stage closure it is
/// an in-program value (via [`AsTensor`]); on the host [`get`](Self::get) /
/// [`bytes`](Self::bytes) await the committed value.
pub struct Taken {
    dsl: pie_dsl::Taken,
    wit: Rc<wit_channel::Channel>,
    mode: TakenMode,
    dtype: DType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TakenMode {
    Take,
    Read,
}

impl Taken {
    /// The in-program [`Tensor`] (panics on a host take — a frontend bug).
    pub fn tensor(self) -> Tensor {
        self.dsl.tensor()
    }

    /// Materialize the committed value to the host as raw little-endian bytes.
    /// Awaits in-flight fires; a poisoned channel returns `Err`.
    pub async fn bytes(self) -> Result<Vec<u8>, String> {
        match self.mode {
            TakenMode::Take => self.wit.take().await,
            TakenMode::Read => self.wit.read().await,
        }
    }

    /// Materialize the committed value to the host, decoded to `T`.
    pub async fn get<T: HostElem>(self) -> Result<Vec<T>, String> {
        let raw = self.bytes().await?;
        let _ = self.dtype;
        Ok(T::decode(&raw))
    }
}

impl AsTensor for Taken {
    fn to_arg(&self) -> Arg {
        self.dsl.to_arg()
    }
}
impl AsTensor for &Taken {
    fn to_arg(&self) -> Arg {
        (*self).to_arg()
    }
}

/// A host-readable element type (little-endian, 4 bytes/elem; `bool` is 1 byte).
pub trait HostElem: Copy {
    fn decode(raw: &[u8]) -> Vec<Self>;
}
impl HostElem for i32 {
    fn decode(raw: &[u8]) -> Vec<i32> {
        raw.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
impl HostElem for u32 {
    fn decode(raw: &[u8]) -> Vec<u32> {
        raw.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
impl HostElem for f32 {
    fn decode(raw: &[u8]) -> Vec<f32> {
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

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

    /// Tokens per KV page for this working set's model.
    pub fn page_size(&self) -> u32 {
        self.kv.page_size()
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
    /// First granted index.
    pub fn start(&self) -> u32 {
        self.start
    }

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
    pub fn state_size(&self) -> u64 {
        self.rs.state_size()
    }

    /// Current number of buffered page slots.
    pub fn buffer_size(&self) -> u32 {
        self.rs.buffer_size()
    }

    /// Tokens per buffered RS page for this working set's model/driver.
    pub fn buffer_page_size(&self) -> u32 {
        self.rs.buffer_page_size()
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

/// The one WIT forward-pass resource, whichever interface it came from.
///
/// `pie:inferlet` exposes three forward interfaces (`forward`,
/// `forward-recurrent`, `forward-hybrid`) so that attention-only state
/// algorithms cannot be named against a folded recurrent state. wit-bindgen
/// generates three unrelated Rust resource types for them; this enum is where
/// that three-way split stops, so the tracing/bookkeeping below exists once.
enum PassWit {
    Attention(wit_attention::ForwardPass),
    Recurrent(wit_recurrent::ForwardPass),
    Hybrid(wit_hybrid::ForwardPass),
}

/// The shared forward-pass builder. Never public: guests always hold one of
/// the three newtypes in [`attention`] / [`recurrent`] / [`hybrid`], which is
/// what makes a cross-kind helper a COMPILE error rather than a silent
/// semantic bug.
struct PassCore {
    wit: PassWit,
    inner: RefCell<ForwardInner>,
}

struct ForwardInner {
    ports: Vec<(Port, DslChannel)>,
    stages: Vec<(Stage, StageClosure)>,
    vocab: u32,
    page_size: u32,
    attention_ws: Option<Rc<KvWorkingSet>>,
    rs_working_sets: Vec<Rc<crate::working_set::RsWorkingSet>>,
    /// The KV half of the state binding, recorded but not yet sent.
    kv_bind: Option<KvBind>,
    rs_bind: Option<RsGeometryBind>,
    program_attached: bool,
}

/// A recorded KV binding. The WIT call is DEFERRED to `attach_program`
/// because `forward-hybrid.attention` binds KV and RS in one call -- they are
/// one set of geometry over different layers of the same forward -- while the
/// guest-facing SDK keeps them as two independent, order-free calls.
struct KvBind {
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

/// A recorded `rs-geometry`: where the bound recurrent state lives for this
/// fire and where its folded boundary lands.
struct RsGeometryBind {
    /// The fold-len channel as a descriptor port, or `None` for the
    /// synthesized fold-everything geometry. Claiming it is what lets a stage
    /// COMPUTE the fold length: the driver resolves the port on device, so a
    /// speculative decode's accepted count never round-trips through the host.
    port: Option<(Port, Channel)>,
    fold_len: Rc<wit_channel::Channel>,
    /// Capacity grant, not an address. The buffer's addressing is derived by
    /// the runtime from its own occupancy; a guest copy of it could only
    /// agree or be refused.
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

impl PassCore {
    fn new(wit: PassWit) -> PassCore {
        let vocab = crate::model::output_vocab_size();
        let page_size = crate::model::kv_page_size();
        PassCore {
            wit,
            inner: RefCell::new(ForwardInner {
                rs_bind: None,
                ports: Vec::new(),
                stages: Vec::new(),
                vocab,
                page_size,
                attention_ws: None,
                rs_working_sets: Vec::new(),
                kv_bind: None,
                program_attached: false,
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
        match &self.wit {
            PassWit::Attention(w) => w.embed(token_wit.as_ref(), indptr_wit.as_ref()),
            PassWit::Recurrent(w) => w.embed(token_wit.as_ref(), indptr_wit.as_ref()),
            PassWit::Hybrid(w) => w.embed(token_wit.as_ref(), indptr_wit.as_ref()),
        }?;
        self.inner.borrow_mut().ports.extend([
            (Port::EmbedTokens, claim_port(Port::EmbedTokens, tokens)),
            (Port::EmbedIndptr, claim_port(Port::EmbedIndptr, indptr)),
        ]);
        Ok(())
    }

    /// Bind attention and all of its geometry channels. This is the only
    /// attention binding surface; `mask: None` omits PTIR's existing AttnMask
    /// port, while `Some` binds that channel.
    #[allow(clippy::too_many_arguments)]
    pub fn attention<R, W>(
        &self,
        ws: &WorkingSet,
        readable: R,
        writable: W,
        kv_len: &Channel,
        pages: &Channel,
        page_indptr: &Channel,
        w_slot: &Channel,
        w_off: &Channel,
        positions: &Channel,
        mask: Option<&Channel>,
    ) -> Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
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
        let readable = PageDeclaration::from_range(readable)?;
        let writable = PageDeclaration::from_range(writable)?;
        let kv_len_wit = kv_len.wit();
        let pages_wit = pages.wit();
        let page_indptr_wit = page_indptr.wit();
        let w_slot_wit = w_slot.wit();
        let w_off_wit = w_off.wit();
        let positions_wit = positions.wit();
        let mask_wit = mask.map(Channel::wit);
        let mut inner = self.inner.borrow_mut();
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
        inner.attention_ws = Some(ws.kv.clone());
        inner.kv_bind = Some(KvBind {
            ws: ws.kv.clone(),
            readable,
            writable,
            kv_len: kv_len_wit,
            pages: pages_wit,
            page_indptr: page_indptr_wit,
            w_slot: w_slot_wit,
            w_off: w_off_wit,
            positions: positions_wit,
            mask: mask_wit,
        });
        Ok(())
    }

    /// Record the recurrent-state working sets (resolved request order) and,
    /// optionally, the buffered geometry. Like the KV half, the WIT call is
    /// deferred to `attach_program`.
    fn bind_recurrent(
        &self,
        working_sets: &[RsWorkingSet],
        geom: RsGeometryBind,
    ) -> Result<(), String> {
        if working_sets.is_empty() {
            return Err(
                "forward pass needs one recurrent-state working set per request".to_string(),
            );
        }
        if self.inner.borrow().program_attached {
            // Post-attachment this is a REBIND (a request-set change between
            // fires), which goes straight out over WIT instead of being
            // recorded for `flush_state_binding`.
            return self.rebind_recurrent(working_sets, geom);
        }
        let mut inner = self.inner.borrow_mut();
        inner.rs_working_sets = working_sets.iter().map(|rs| rs.rs.clone()).collect();
        if let Some((port, channel)) = geom.port {
            inner.ports.push((port, claim_port(port, &channel)));
        }
        inner.rs_bind = Some(geom);
        Ok(())
    }

    /// Rebind the recurrent-state working sets of an ALREADY-BOUND pass (a
    /// request-set change between fires). Only legal after the program is
    /// attached; before that, `bind_recurrent` records instead.
    fn rebind_recurrent(
        &self,
        working_sets: &[RsWorkingSet],
        geom: RsGeometryBind,
    ) -> Result<(), String> {
        let replacement: Vec<Rc<crate::working_set::RsWorkingSet>> =
            working_sets.iter().map(|rs| rs.rs.clone()).collect();
        let borrows: Vec<&crate::working_set::RsWorkingSet> =
            replacement.iter().map(Rc::as_ref).collect();
        match &self.wit {
            PassWit::Recurrent(w) => w.attention(&borrows, &recurrent_rs_geometry(&geom)),
            PassWit::Hybrid(w) => w.attention(None, &borrows, &hybrid_rs_geometry(&geom)),
            PassWit::Attention(_) => {
                Err("an attention-only forward pass has no recurrent state".to_string())
            }
        }?;
        let mut inner = self.inner.borrow_mut();
        inner.rs_working_sets = replacement;
        inner.rs_bind = Some(geom);
        Ok(())
    }

    /// Bind readout indexes through a channel, separately from embedding.
    pub fn readout(&self, indices: &Channel) -> Result<(), String> {
        self.ensure_ports_available(&[Port::Readout])?;
        let indices_wit = indices.wit();
        match &self.wit {
            PassWit::Attention(w) => w.readout(indices_wit.as_ref()),
            PassWit::Recurrent(w) => w.readout(indices_wit.as_ref()),
            PassWit::Hybrid(w) => w.readout(indices_wit.as_ref()),
        }?;
        self.inner
            .borrow_mut()
            .ports
            .push((Port::Readout, claim_port(Port::Readout, indices)));
        Ok(())
    }

    /// Attach the `prologue` stage (overview §5.3).
    pub fn prologue(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::Prologue, body);
    }
    /// Attach the `on_attn_proj` stage (per layer, before attention).
    pub fn on_attn_proj(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::OnAttnProj, body);
    }
    /// Attach the `on_attn` stage (per layer, after attention).
    pub fn on_attn(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::OnAttn, body);
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
    fn submit(&self, on: &Pipeline) -> Result<(), String> {
        submit_core_frame(on, &[Some(self)])
    }

    /// Issue the ONE deferred state-binding WIT call. Called once, at the top
    /// of `attach_program`, so that a hybrid pass -- whose interface binds KV
    /// and RS together -- still lets the guest record the two halves through
    /// independent, order-free SDK calls.
    fn flush_state_binding(&self) -> Result<(), String> {
        let inner = self.inner.borrow();
        let rs: Vec<&crate::working_set::RsWorkingSet> =
            inner.rs_working_sets.iter().map(Rc::as_ref).collect();
        match &self.wit {
            PassWit::Attention(w) => {
                let Some(kv) = inner.kv_bind.as_ref() else {
                    return Err("attention must be bound before submit".to_string());
                };
                let geom = wit_attention::KvGeometry {
                        readable_pages: kv.readable.wit(),
                        writable_pages: kv.writable.wit(),
                        kv_len: kv.kv_len.as_ref(),
                        pages: kv.pages.as_ref(),
                        page_indptr: kv.page_indptr.as_ref(),
                        w_slot: kv.w_slot.as_ref(),
                        w_off: kv.w_off.as_ref(),
                        positions: kv.positions.as_ref(),
                    mask: kv.mask.as_deref(),
                };
                w.attention(kv.ws.as_ref(), &geom)
            }
            PassWit::Recurrent(w) => {
                let Some(geom) = inner.rs_bind.as_ref() else {
                    return Err("recurrent state must be bound before submit".to_string());
                };
                w.attention(&rs, &recurrent_rs_geometry(geom))
            }
            PassWit::Hybrid(w) => {
                let Some(kv) = inner.kv_bind.as_ref() else {
                    return Err("attention must be bound before submit".to_string());
                };
                let binding = wit_hybrid::KvBinding {
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
                    };
                let Some(geom) = inner.rs_bind.as_ref() else {
                    return Err("recurrent state must be bound before submit".to_string());
                };
                w.attention(Some(&binding), &rs, &hybrid_rs_geometry(geom))
            }
        }
    }

    fn attach_program(&self) -> Result<(), String> {
        if self.inner.borrow().program_attached {
            return Ok(());
        }
        self.flush_state_binding()?;

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
        match &self.wit {
            PassWit::Attention(w) => w.program(&bytes, &borrows),
            PassWit::Recurrent(w) => w.program(&bytes, &borrows),
            PassWit::Hybrid(w) => w.program(&bytes, &borrows),
        }?;
        drop(inner);
        self.inner.borrow_mut().program_attached = true;
        Ok(())
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
/// Always k. A recurrent-state (linear) model used to get 1 whatever k was,
/// because its mapping published at FINALIZE: slot i+1's prepare read a stale
/// mapping unless slot i had already settled, which a frame can never reach.
/// RS now publishes at prepare, in slot order, so a linear lane fills a frame
/// exactly like a dense one.
///
/// Kept as its own query rather than folded into [`frame_size`]: it answers
/// "how much can this lane submit per frame", which is a model property that
/// has diverged from k before and may again. It is NOT the place to encode
/// per-PASS restrictions — a fire that buffers or commits is submitted solo by
/// the runtime because its `fold-len` picks the RS execution mode for the whole
/// composed batch, and that is a property of the fire, not of the model.
pub fn live_slots() -> usize {
    frame_size()
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
fn submit_core_frame(on: &Pipeline, slots: &[Option<&PassCore>]) -> Result<(), String> {
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
    // A frame is monomorphic in its slot type. The public entry points take
    // `&[Option<&ForwardPass>]` of ONE module, so a mixed frame is already
    // unrepresentable; this match just picks the matching WIT `submit`.
    let kind = slots.iter().flatten().next();
    let Some(kind) = kind else {
        return Ok(());
    };
    macro_rules! dispatch {
        ($variant:ident, $wit:path) => {{
            let mut borrows: Vec<Option<&_>> = slots
                .iter()
                .map(|slot| match slot.map(|pass| &pass.wit) {
                    Some(PassWit::$variant(w)) => Some(w),
                    Some(_) => unreachable!("frames are monomorphic in their pass kind"),
                    None => None,
                })
                .collect();
            borrows.resize(k, None);
            $wit(&on.wit, &borrows)
        }};
    }
    match &kind.wit {
        PassWit::Attention(_) => dispatch!(Attention, wit_attention::submit),
        PassWit::Recurrent(_) => dispatch!(Recurrent, wit_recurrent::submit),
        PassWit::Hybrid(_) => dispatch!(Hybrid, wit_hybrid::submit),
    }
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
async fn run_ahead_core(
    on: &Pipeline,
    pass: &PassCore,
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
        let slots: Vec<Option<&PassCore>> = vec![Some(pass); live];
        submit_core_frame(on, &slots)?;
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
        Channel, PageGrant, Pipeline, RsWorkingSet, TOKEN_PAD, WorkingSet, channel_capacity,
        frame_size, live_slots, max_embed_length, pad_tokens, prefill_chunks, unpad_tokens,
    };
    pub use std::ops::ControlFlow;
    pub use pie_dsl::dtype;
    pub use pie_dsl::intrinsics;
    pub use pie_dsl::value::{
        AsTensor, Tensor, abs, add, and, broadcast, cast, causal_mask, cummass_le, cumprod, cumsum,
        div, entropy, entropy_from_logprobs, eq, exp, gather, gather_row, ge, gt, gumbel,
        gumbel_max, iota, l2norm, le, log, log_softmax, lt, mask_apply, masked_argmax, matmul,
        max_elem, min_elem, mul, ne, neg, not, nucleus_sample, or, pivot_threshold, prob_ge,
        rank_le, recip, reduce_argmax, reduce_max, reduce_min, reduce_sum, rem, reshape, rng,
        row_membership, scalar_gather, scatter_add, scatter_set, select, sign, sink_window_mask,
        sliding_window_mask, softmax, sort_desc, sub, top_k, transpose,
    };
    pub use pie_dsl::{DType, Stage};
}

// ---------------------------------------------------------------------------
// The three author-facing pass modules
// ---------------------------------------------------------------------------

/// Everything the three pass modules share, generated once. The DUPLICATION is
/// the product: three unrelated `ForwardPass` types mean a helper written for
/// attention-only KV eviction (H2O, SnapKV, TOVA, Quest) cannot even name a
/// hybrid model's pass, which is the correctness gate this split exists for.
macro_rules! pass_module {
    ($variant:ident, $wit:path, $doc:expr) => {
        #[doc = $doc]
        pub struct ForwardPass(::std::rc::Rc<$crate::ptir::PassCore>);

        impl ForwardPass {
            pub fn new() -> Self {
                Self(::std::rc::Rc::new($crate::ptir::PassCore::new(
                    $crate::ptir::PassWit::$variant(<$wit>::new()),
                )))
            }

            /// Bind token ids and CSR row indptr. Both are channels.
            pub fn embed(
                &self,
                tokens: &$crate::ptir::Channel,
                indptr: &$crate::ptir::Channel,
            ) -> Result<(), String> {
                self.0.embed(tokens, indptr)
            }

            /// Bind readout indexes through a channel, separately from embedding.
            pub fn readout(&self, indices: &$crate::ptir::Channel) -> Result<(), String> {
                self.0.readout(indices)
            }

            /// Attach the `prologue` stage.
            pub fn prologue(&self, body: impl Fn() + 'static) {
                self.0.prologue(body)
            }
            /// Attach the `epilogue` stage (sampling programs; after the forward).
            pub fn epilogue(&self, body: impl Fn() + 'static) {
                self.0.epilogue(body)
            }

            /// Enqueue this pass as a SINGLE-SLOT FRAME on `on`.
            pub fn submit(&self, on: &$crate::ptir::Pipeline) -> Result<(), String> {
                self.0.submit(on)
            }
        }

        impl Default for ForwardPass {
            fn default() -> Self {
                Self::new()
            }
        }

        /// Submit ONE FRAME on `on`: up to `frame_size()` ordered slots, slot i
        /// executing in wave i; missing trailing slots are padded with no-ops.
        pub fn submit_frame(
            on: &$crate::ptir::Pipeline,
            slots: &[Option<&ForwardPass>],
        ) -> Result<(), String> {
            let cores: Vec<Option<&$crate::ptir::PassCore>> = slots
                .iter()
                .map(|slot| slot.map(|pass| pass.0.as_ref()))
                .collect();
            $crate::ptir::submit_core_frame(on, &cores)
        }

        /// Keeps the engine's run-ahead window full while `on_token` consumes
        /// results. See the crate-level `run_ahead` documentation.
        pub async fn run_ahead(
            on: &$crate::ptir::Pipeline,
            pass: &ForwardPass,
            budget: usize,
            on_token: impl AsyncFnMut() -> Result<::std::ops::ControlFlow<()>, String>,
        ) -> Result<usize, String> {
            $crate::ptir::run_ahead_core(on, pass.0.as_ref(), budget, on_token).await
        }
    };
}

/// Attention taps, legal only where attention layers exist.
macro_rules! pass_module_attention_stages {
    () => {
        impl ForwardPass {
            /// Attach the `on_attn_proj` stage (per layer, before attention).
            pub fn on_attn_proj(&self, body: impl Fn() + 'static) {
                self.0.on_attn_proj(body)
            }
            /// Attach the `on_attn` stage (per layer, after attention).
            pub fn on_attn(&self, body: impl Fn() + 'static) {
                self.0.on_attn(body)
            }
        }
    };
}

/// `pie:inferlet/forward` — paged, per-token, reversibly discardable KV only.
///
/// Valid when `model.pass_kind()` is [`ForwardKind::Attention`]. KV eviction
/// and reordering algorithms belong HERE and nowhere else: evicting a KV page
/// is reversible in a way that unfolding a recurrent state is not.
///
/// [`ForwardKind::Attention`]: crate::model::ForwardKind
pub mod attention {
    use super::*;

    pass_module!(
        Attention,
        wit_attention::ForwardPass,
        "An attention-only forward pass."
    );
    pass_module_attention_stages!();

    impl ForwardPass {
        /// Bind attention and all of its geometry channels. `mask: None` omits
        /// PTIR's AttnMask port; `Some` binds that channel to it.
        #[allow(clippy::too_many_arguments)]
        pub fn attention<R, W>(
            &self,
            ws: &WorkingSet,
            readable: R,
            writable: W,
            kv_len: &Channel,
            pages: &Channel,
            page_indptr: &Channel,
            w_slot: &Channel,
            w_off: &Channel,
            positions: &Channel,
            mask: Option<&Channel>,
        ) -> Result<(), String>
        where
            R: ::std::ops::RangeBounds<u32>,
            W: ::std::ops::RangeBounds<u32>,
        {
            self.0.attention(
                ws,
                readable,
                writable,
                kv_len,
                pages,
                page_indptr,
                w_slot,
                w_off,
                positions,
                mask,
            )
        }
    }

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
    use super::*;

    pass_module!(
        Recurrent,
        wit_recurrent::ForwardPass,
        "A recurrent-only forward pass."
    );

    impl ForwardPass {
        /// Bind the recurrent-state working sets in resolved request order.
        /// The mechanism is a recurrence, but the SLOT is the same one
        /// `attention` names in the other two interfaces.
        pub fn attention(&self, working_sets: &[RsWorkingSet]) -> Result<(), String> {
            self.0
                .bind_recurrent(working_sets, rs_geometry_fold_all()?)
        }

        /// Bind the recurrent state AND state where its folded boundary
        /// lands. See [`hybrid::ForwardPass::recurrent_with`] -- the surface
        /// and the semantics are identical.
        ///
        /// [`hybrid::ForwardPass::recurrent_with`]: crate::ptir::hybrid::ForwardPass::recurrent_with
        pub fn attention_with<B>(
            &self,
            working_sets: &[RsWorkingSet],
            fold_len: &Channel,
            buffer: B,
        ) -> Result<(), String>
        where
            B: RangeBounds<u32>,
        {
            let geom = rs_geometry_bind(fold_len, buffer)?;
            self.0.bind_recurrent(working_sets, geom)
        }
    }

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
    use super::*;

    pass_module!(Hybrid, wit_hybrid::ForwardPass, "A hybrid forward pass.");
    pass_module_attention_stages!();

    impl ForwardPass {
        /// Bind the ATTENTION half. The two halves are recorded independently
        /// and issued as the single `forward-hybrid.attention` call on first
        /// submit, so their order here does not matter -- but both are
        /// required.
        #[allow(clippy::too_many_arguments)]
        pub fn attention<R, W>(
            &self,
            ws: &WorkingSet,
            readable: R,
            writable: W,
            kv_len: &Channel,
            pages: &Channel,
            page_indptr: &Channel,
            w_slot: &Channel,
            w_off: &Channel,
            positions: &Channel,
            mask: Option<&Channel>,
        ) -> Result<(), String>
        where
            R: ::std::ops::RangeBounds<u32>,
            W: ::std::ops::RangeBounds<u32>,
        {
            self.0.attention(
                ws,
                readable,
                writable,
                kv_len,
                pages,
                page_indptr,
                w_slot,
                w_off,
                positions,
                mask,
            )
        }

        /// Bind the RECURRENT half: one working set per request, in resolved
        /// request order.
        pub fn recurrent(&self, working_sets: &[RsWorkingSet]) -> Result<(), String> {
            self.0
                .bind_recurrent(working_sets, rs_geometry_fold_all()?)
        }

        /// Bind the RECURRENT half AND state where its folded boundary lands.
        ///
        /// `fold_len` is a channel carrying, per request, how far the folded
        /// boundary advances over `[buffer | this fire's tokens]`, clamped to
        /// that tail:
        ///
        /// - `u32::MAX` -- fold everything. What [`recurrent`] supplies.
        /// - `0` -- fold nothing; append every token of this fire to the
        ///   buffer. This is what makes a linear model speculatable: buffered
        ///   tokens that are never folded cost nothing to abandon.
        /// - `n <= buffer_len` -- fold the first `n` buffered tokens and drop
        ///   the covered head pages. The COMMIT half of speculation.
        ///
        /// A channel rather than a scalar so that a device-computed accepted
        /// count can drive the fold directly, fusing verify and commit into
        /// one fire.
        ///
        /// [`recurrent`]: ForwardPass::recurrent
        pub fn recurrent_with<B>(
            &self,
            working_sets: &[RsWorkingSet],
            fold_len: &Channel,
            buffer: B,
        ) -> Result<(), String>
        where
            B: RangeBounds<u32>,
        {
            let geom = rs_geometry_bind(fold_len, buffer)?;
            self.0.bind_recurrent(working_sets, geom)
        }
    }

    /// Glob-import surface for hybrid inferlet authors.
    pub mod prelude {
        pub use super::{ForwardPass, run_ahead, submit_frame};
        pub use crate::ptir::shared_prelude::*;
    }
}

fn recurrent_rs_geometry(geom: &RsGeometryBind) -> wit_recurrent::RsGeometry<'_> {
    wit_recurrent::RsGeometry {
        fold_len: geom.fold_len.as_ref(),
        buffer: geom.buffer.wit(),
    }
}

fn hybrid_rs_geometry(geom: &RsGeometryBind) -> wit_hybrid::RsGeometry<'_> {
    wit_hybrid::RsGeometry {
        fold_len: geom.fold_len.as_ref(),
        buffer: geom.buffer.wit(),
    }
}

fn rs_geometry_bind<B>(fold_len: &Channel, buffer: B) -> Result<RsGeometryBind, String>
where
    B: RangeBounds<u32>,
{
    Ok(RsGeometryBind {
        port: Some((Port::RsFoldLen, *fold_len)),
        fold_len: fold_len.wit(),
        buffer: PageDeclaration::from_range(buffer)?,
    })
}

/// The geometry of a pass that never buffers: an empty buffer and a `fold-len`
/// past the tail, which clamps to "fold every token of every fire".
///
/// This is degenerate but MEANINGFUL -- it is the fast path stated explicitly,
/// not the dummy geometry this surface exists to avoid. The buffer channels are
/// never dereferenced because `buffer-len` is zero.
fn rs_geometry_fold_all() -> Result<RsGeometryBind, String> {
    thread_local! {
        /// Minted once per guest thread: a rebind must not leak a fresh
        /// channel every time the request set changes.
        static FOLD_ALL: Channel = Channel::from(vec![u32::MAX]);
    }
    FOLD_ALL.with(|fold_len| {
        let mut geom = rs_geometry_bind(fold_len, 0..0)?;
        // A pass that folds everything unconditionally has nothing to compute,
        // so claiming the port would put a dead one in every plain recurrent
        // trace.
        geom.port = None;
        Ok(geom)
    })
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
