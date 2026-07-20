//! `inferlet::ptir` — the author-facing PTIR bridge over the WIT `ptir` surface.
//!
//! This is the only home of the overview §3/§5 author surface
//! (`ForwardPass`/`Pipeline`/`WorkingSet`/`Channel`). It wraps the WIT
//! resources (`channel`, `forward-pass`, `kv-working-set`, `pipeline`) and
//! drives the neutral [`Builder`](ptir_dsl::Builder) from the `ptir-dsl`
//! crate: the author writes stage closures + port bindings, the bridge lowers
//! them to the canonical PTIR container, orders the WIT channel handles by the
//! builder↔bridge contract
//! ([`Traced::channel_order`](ptir_dsl::Traced::channel_order)), and calls
//! `forward-pass.new` (which binds against the model — the guest does not bind,
//! D6). Program identity, dedup, and validation happen host-side inside
//! `forward-pass.new`/`forward-pass.submit`.
//!
//! A [`Channel`] owns BOTH sides: the `ptir-dsl` trace declaration (its `take`/
//! `put`/`read` record ops inside a stage closure, and host `put`s record the
//! host-role endpoint) and the WIT `channel` resource (the host transport). The
//! two are constructed from the same `(shape, dtype, capacity)` so the decl
//! validates against the container by construction.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Bound, RangeBounds};
use std::rc::Rc;

use ptir_dsl::builder::{Builder, PortInput};
use ptir_dsl::channel::PutValue;
use ptir_dsl::value::{Arg, ConstData};
use ptir_dsl::{
    AsTensor, Channel as DslChannel, DType, IntoConst, IntoPut, IntoShape, Port, Shape, Stage,
    Tensor,
};

use crate::pie::inferlet::forward as wit;
use crate::pie::inferlet::pipeline as wit_pipeline;
use crate::pie::inferlet::types::Dtype as WitDtype;
use crate::working_set::{KvWorkingSet, PageRange, PageSpan};

pub use ptir_dsl::intrinsics;

// Re-export the eDSL vocabulary so an author writes stage closures with a single
// `use inferlet::ptir::prelude::*;` (mirrors the old `ptir::prelude`).
pub use ptir_dsl::{DType as Dtype, model};
pub use ptir_dsl::{
    add, and, broadcast, cast, causal_mask, cummass_le, cumprod, cumsum, div, dtype, entropy,
    entropy_from_logprobs, eq, exp, gather, gather_row, ge, gt, gumbel, gumbel_max, iota, l2norm,
    le, log, log_softmax, lt, mask_apply, masked_argmax, matmul, max_elem, min_elem, mul, ne, neg,
    not, nucleus_sample, or, pivot_threshold, prob_ge, rank_le, reduce_argmax, reduce_max,
    reduce_min, reduce_sum, rem, reshape, rng, row_membership, scalar_gather, scatter_add,
    scatter_set, select, sink_window_mask, sliding_window_mask, softmax, sub, top_k, transpose,
};

// ---------------------------------------------------------------------------
// gid -> WIT channel registry
// ---------------------------------------------------------------------------
//
// A stage trace interns channels and yields dense channel ids keyed by the
// dsl channel's gid; `forward-pass.new` wants the WIT handles in that dense
// order. Every channel the author can reference is created via `Channel::new`/
// `from`/`seeded`, so registering (gid -> Rc<wit::Channel>) at construction
// lets a `ForwardPass` resolve each `Traced.channel_order` entry. Inferlets are
// single-threaded (wasm), so a thread-local registry is sound.

thread_local! {
    static WIT_CHANNELS: RefCell<HashMap<u64, Rc<wit::Channel>>> = RefCell::new(HashMap::new());
}

fn register_channel(gid: u64, wit: Rc<wit::Channel>) {
    WIT_CHANNELS.with(|m| {
        m.borrow_mut().insert(gid, wit);
    });
}

fn lookup_channel(gid: u64) -> Option<Rc<wit::Channel>> {
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

/// A GPU-resident bounded queue (overview §1). Owns the `ptir-dsl` trace
/// declaration and the WIT `channel` resource. In a stage closure `take`/`read`/
/// `put` record IR ops; on the host `put` stages a value (seed / host-writer
/// cell) and `Taken::get().await`/`Taken::bytes().await` materialize a committed value.
/// A registry-backed COPY TOKEN (F9): the channel's shared state (DSL trace
/// state + WIT handle) lives in thread-local registries keyed by gid, and
/// this token holds only the gid plus immutable metadata. Stage closures
/// capture tokens by value, so closures are `'static`, [`ForwardPass`] has
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

/// Default number of decode fires kept submitted ahead of host consumption.
/// Matches the engine scheduler's run-ahead depth (`MAX_IN_FLIGHT` in
/// quorum.rs) so a single pipeline can keep every scheduler wave slot fed.
pub const DEFAULT_RUNAHEAD_DEPTH: usize = 2;

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
    fn wit(&self) -> Rc<wit::Channel> {
        lookup_channel(self.gid).expect("channel token resolves in the WIT registry")
    }

    /// Widen the ring to `n` cells (deeper run-ahead).
    pub fn capacity(self, n: u32) -> Channel {
        let dsl = self.dsl().capacity(n);
        let wit = Rc::new(wit::Channel::new(
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
        let wit = Rc::new(wit::Channel::new(
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
}

/// The result of [`Channel::take`]/[`Channel::read`]. In a stage closure it is
/// an in-program value (via [`AsTensor`]); on the host [`get`](Self::get) /
/// [`bytes`](Self::bytes) await the committed value.
pub struct Taken {
    dsl: ptir_dsl::Taken,
    wit: Rc<wit::Channel>,
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
/// [`ForwardPass::set_rs_working_sets`] for models whose
/// `model::rs_state_size()` is nonzero; pure-attention models bind none.
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

/// A descriptor-port binding staged on the [`ForwardPass`] until submit.
enum PortSpec {
    Channel(Port, DslChannel),
    Const(Port, Tensor),
}

type StageClosure = Box<dyn Fn()>;

/// The forward pass (overview §5). Attach descriptor ports + stage closures,
/// submit through a [`Pipeline`]. On first submit the bridge drives the builder,
/// lowers to the container, and calls `forward-pass.new`; the bound WIT resource
/// is memoized. The lifetime lets stage closures borrow the channels they touch.
pub struct ForwardPass {
    inner: RefCell<ForwardInner>,
}

struct ForwardInner {
    ports: Vec<PortSpec>,
    stages: Vec<(Stage, StageClosure)>,
    attn: Option<AttnWorkingSet>,
    derive_dense_geometry: bool,
    dense_page_capacity: Option<u32>,
    auto_geometry: Option<AutoGeometry>,
    geometry_materialized: bool,
    rs_working_sets: Vec<Rc<crate::working_set::RsWorkingSet>>,
    bound: Option<Rc<wit::ForwardPass>>,
}

struct AttnWorkingSet {
    ws: Rc<KvWorkingSet>,
    readable: PageDeclaration,
    writable: PageDeclaration,
}

#[derive(Clone)]
enum GeometryInput {
    Channel(DslChannel),
    Const(Tensor),
}

impl GeometryInput {
    fn shape(&self) -> Shape {
        match self {
            GeometryInput::Channel(channel) => channel.shape(),
            GeometryInput::Const(value) => value.shape(),
        }
    }

    fn tensor(&self) -> Tensor {
        match self {
            GeometryInput::Channel(channel) => channel.read().tensor(),
            GeometryInput::Const(value) => value.clone(),
        }
    }
}

struct AutoGeometry {
    tokens: DslChannel,
    indptr: GeometryInput,
    kv_len: GeometryInput,
    positions: Option<DslChannel>,
    pages: Option<DslChannel>,
    page_indptr: Option<DslChannel>,
    w_slot: Option<DslChannel>,
    w_off: Option<DslChannel>,
    token_count: u32,
    lane_count: u32,
    page_count: u32,
    page_size: u32,
    token_dtype: DType,
}

impl AutoGeometry {
    fn trace(&self) {
        for output in [
            self.positions.as_ref(),
            self.pages.as_ref(),
            self.page_indptr.as_ref(),
            self.w_slot.as_ref(),
            self.w_off.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            output.take();
        }

        let needs_token_geometry =
            self.positions.is_some() || self.w_slot.is_some() || self.w_off.is_some();
        let needs_kv_len = needs_token_geometry || self.page_indptr.is_some();
        let kv_len = needs_kv_len.then(|| self.kv_len.tensor());

        let positions = needs_token_geometry.then(|| {
            let tokens = self.tokens.read().tensor();
            let indptr = self.indptr.tensor();
            let kv_len = kv_len.as_ref().expect("KvLen read for token geometry");
            let sentinel = match self.token_dtype {
                DType::I32 => Tensor::constant(-1i32),
                DType::U32 => Tensor::constant(u32::MAX),
                _ => unreachable!("embed tokens are integer"),
            };
            let valid = ne(&tokens, &sentinel);

            let rows = broadcast(
                reshape(iota(self.token_count), [self.token_count, 1]),
                [self.token_count, self.lane_count],
            );
            let lane_indices = iota(self.lane_count);
            let starts = broadcast(
                reshape(gather(&indptr, &lane_indices), [1, self.lane_count]),
                [self.token_count, self.lane_count],
            );
            let ends = broadcast(
                reshape(
                    gather(&indptr, add(&lane_indices, 1u32)),
                    [1, self.lane_count],
                ),
                [self.token_count, self.lane_count],
            );
            let membership = and(ge(&rows, &starts), lt(&rows, &ends));
            let live = and(
                &membership,
                broadcast(
                    reshape(&valid, [self.token_count, 1]),
                    [self.token_count, self.lane_count],
                ),
            );

            // Scan each CSR lane independently. F32 is exact for every
            // practical token envelope (all live counts remain below 2^24).
            let live_by_lane = transpose(cast(&live, DType::F32));
            let exclusive_by_lane = sub(cumsum(&live_by_lane), &live_by_lane);
            let rank = cast(
                reduce_sum(mul(
                    transpose(exclusive_by_lane),
                    cast(&membership, DType::F32),
                )),
                DType::U32,
            );
            let live_rows = cast(reduce_sum(&live_by_lane), DType::U32);
            let base_by_lane = sub(kv_len, &live_rows);
            let base = reduce_sum(mul(
                broadcast(
                    reshape(base_by_lane, [1, self.lane_count]),
                    [self.token_count, self.lane_count],
                ),
                cast(membership, DType::U32),
            ));
            add(base, rank)
        });

        if let Some(output) = &self.positions {
            output.put(positions.as_ref().expect("generated Positions"));
        }
        if let Some(output) = &self.pages {
            output.put(broadcast(
                reshape(iota(self.page_count), [1, self.page_count]),
                [self.lane_count, self.page_count],
            ));
        }
        if let Some(output) = &self.page_indptr {
            let page_counts = div(
                add(
                    kv_len.as_ref().expect("KvLen read for PageIndptr"),
                    self.page_size - 1,
                ),
                self.page_size,
            );
            let cumulative = cast(cumsum(cast(page_counts, DType::F32)), DType::U32);
            output.put(scatter_set(
                Tensor::constant(vec![0u32; self.lane_count as usize + 1]),
                add(iota(self.lane_count), 1u32),
                cumulative,
            ));
        }
        if let Some(output) = &self.w_slot {
            output.put(div(
                positions.as_ref().expect("generated WSlot"),
                self.page_size,
            ));
        }
        if let Some(output) = &self.w_off {
            output.put(rem(
                positions.as_ref().expect("generated WOff"),
                self.page_size,
            ));
        }
    }
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

impl ForwardInner {
    fn input(&self, port: Port) -> Option<GeometryInput> {
        self.ports.iter().find_map(|spec| match spec {
            PortSpec::Channel(bound, channel) if *bound == port => {
                Some(GeometryInput::Channel(channel.clone()))
            }
            PortSpec::Const(bound, value) if *bound == port => {
                Some(GeometryInput::Const(value.clone()))
            }
            _ => None,
        })
    }

    fn has_port(&self, port: Port) -> bool {
        self.ports.iter().any(|spec| match spec {
            PortSpec::Channel(bound, _) | PortSpec::Const(bound, _) => *bound == port,
        })
    }

    fn materialize_geometry(&mut self) {
        if self.geometry_materialized {
            return;
        }

        let missing_positions = !self.has_port(Port::Positions);
        let missing_pages = !self.has_port(Port::Pages);
        let missing_page_indptr = !self.has_port(Port::PageIndptr);
        let missing_w_slot = !self.has_port(Port::WSlot);
        let missing_w_off = !self.has_port(Port::WOff);
        if ![
            missing_positions,
            missing_pages,
            missing_page_indptr,
            missing_w_slot,
            missing_w_off,
        ]
        .into_iter()
        .any(|missing| missing)
        {
            self.geometry_materialized = true;
            return;
        }

        let Some(tokens) = self.ports.iter().find_map(|spec| match spec {
            PortSpec::Channel(Port::EmbedTokens, channel) => Some(channel.clone()),
            _ => None,
        }) else {
            return;
        };
        let Some(indptr) = self.input(Port::EmbedIndptr) else {
            return;
        };
        let Some(kv_len) = self.input(Port::KvLen) else {
            // The PTIR validator reports the missing author-bound KvLen.
            return;
        };
        let Some(attn) = self.attn.as_ref() else {
            return;
        };

        let token_count =
            u32::try_from(tokens.shape().numel()).expect("embed token envelope fits in u32");
        let indptr_shape = indptr.shape();
        assert_eq!(indptr_shape.dims().len(), 1, "EmbedIndptr must be a vector");
        let lane_count = u32::try_from(indptr_shape.numel().saturating_sub(1))
            .expect("EmbedIndptr lane count fits in u32");
        assert!(token_count > 0, "embed token envelope must be non-empty");
        assert!(
            lane_count > 0,
            "EmbedIndptr must describe at least one lane"
        );
        assert_eq!(
            kv_len.shape().dims(),
            [lane_count],
            "KvLen must have one post-write extent per EmbedIndptr lane"
        );

        let page_size = attn.ws.page_size();
        let page_count = if missing_pages {
            let page_count = self
                .dense_page_capacity
                .unwrap_or_else(|| attn.ws.page_len());
            assert!(
                page_count > 0,
                "dense geometry must declare at least one page of capacity"
            );
            assert!(
                page_count >= attn.ws.page_len(),
                "dense geometry page capacity is smaller than the current working-set extent"
            );
            page_count
        } else {
            0
        };
        let token_dtype = tokens.dtype();

        let positions = missing_positions
            .then(|| Channel::from(vec![0u32; token_count as usize]).named("__geometry_positions"));
        let pages = missing_pages.then(|| {
            Channel::from_shaped(
                [lane_count, page_count],
                vec![0u32; lane_count as usize * page_count as usize],
            )
            .named("__geometry_pages")
        });
        let page_indptr = missing_page_indptr.then(|| {
            Channel::from(vec![0u32; lane_count as usize + 1]).named("__geometry_page_indptr")
        });
        let w_slot = missing_w_slot
            .then(|| Channel::from(vec![0u32; token_count as usize]).named("__geometry_w_slot"));
        let w_off = missing_w_off
            .then(|| Channel::from(vec![0u32; token_count as usize]).named("__geometry_w_off"));

        if let Some(channel) = &positions {
            self.ports.push(PortSpec::Channel(
                Port::Positions,
                claim_port(Port::Positions, channel),
            ));
        }
        if let Some(channel) = &pages {
            self.ports.push(PortSpec::Channel(
                Port::Pages,
                claim_port(Port::Pages, channel),
            ));
        }
        if let Some(channel) = &page_indptr {
            self.ports.push(PortSpec::Channel(
                Port::PageIndptr,
                claim_port(Port::PageIndptr, channel),
            ));
        }
        if let Some(channel) = &w_slot {
            self.ports.push(PortSpec::Channel(
                Port::WSlot,
                claim_port(Port::WSlot, channel),
            ));
        }
        if let Some(channel) = &w_off {
            self.ports.push(PortSpec::Channel(
                Port::WOff,
                claim_port(Port::WOff, channel),
            ));
        }

        self.auto_geometry = Some(AutoGeometry {
            tokens,
            indptr,
            kv_len,
            positions: positions.map(|channel| channel.dsl()),
            pages: pages.map(|channel| channel.dsl()),
            page_indptr: page_indptr.map(|channel| channel.dsl()),
            w_slot: w_slot.map(|channel| channel.dsl()),
            w_off: w_off.map(|channel| channel.dsl()),
            token_count,
            lane_count,
            page_count,
            page_size,
            token_dtype,
        });
        self.geometry_materialized = true;
    }
}

#[cfg(test)]
mod auto_geometry_tests {
    use super::*;
    use ptir_dsl::ptir::container::{PortSource, TraceContainer};
    use ptir_dsl::ptir::op::Op;
    use ptir_dsl::ptir::registry::ModelProfile;
    use ptir_dsl::ptir::validate::bind;

    struct GeometryTrace {
        container: TraceContainer,
        tokens: u32,
        indptr: Option<u32>,
        kv_len: u32,
        positions: u32,
        pages: u32,
        page_indptr: u32,
        w_slot: u32,
        w_off: u32,
    }

    fn channel_index(traced: &ptir_dsl::Traced, channel: &DslChannel) -> u32 {
        traced
            .channel_order()
            .iter()
            .position(|gid| *gid == channel.gid())
            .expect("test channel is traced") as u32
    }

    fn trace_geometry(
        token_count: u32,
        lane_count: u32,
        channel_indptr: bool,
        page_count: u32,
        page_size: u32,
        generated: [bool; 5],
    ) -> GeometryTrace {
        let tokens = DslChannel::seeded([token_count], DType::I32);
        let indptr_channel = DslChannel::seeded([lane_count + 1], DType::U32);
        let kv_len = DslChannel::seeded([lane_count], DType::U32);
        let positions = DslChannel::seeded([token_count], DType::U32);
        let pages = DslChannel::seeded([lane_count, page_count], DType::U32);
        let page_indptr = DslChannel::seeded([lane_count + 1], DType::U32);
        let w_slot = DslChannel::seeded([token_count], DType::U32);
        let w_off = DslChannel::seeded([token_count], DType::U32);

        let indptr = if channel_indptr {
            GeometryInput::Channel(indptr_channel.clone())
        } else {
            assert_eq!(lane_count, 1);
            GeometryInput::Const(Tensor::constant(vec![0u32, token_count]))
        };
        let geometry = AutoGeometry {
            tokens: tokens.clone(),
            indptr: indptr.clone(),
            kv_len: GeometryInput::Channel(kv_len.clone()),
            positions: generated[0].then(|| positions.clone()),
            pages: generated[1].then(|| pages.clone()),
            page_indptr: generated[2].then(|| page_indptr.clone()),
            w_slot: generated[3].then(|| w_slot.clone()),
            w_off: generated[4].then(|| w_off.clone()),
            token_count,
            lane_count,
            page_count,
            page_size,
            token_dtype: DType::I32,
        };

        let mut builder = Builder::new();
        builder.bind_port(Port::EmbedTokens, PortInput::Channel(tokens.clone()));
        match indptr {
            GeometryInput::Channel(channel) => {
                builder.bind_port(Port::EmbedIndptr, PortInput::Channel(channel))
            }
            GeometryInput::Const(value) => {
                builder.bind_port(Port::EmbedIndptr, PortInput::Const(value))
            }
        }
        builder.bind_port(Port::KvLen, PortInput::Channel(kv_len.clone()));
        builder.bind_port(Port::Positions, PortInput::Channel(positions.clone()));
        builder.bind_port(Port::Pages, PortInput::Channel(pages.clone()));
        builder.bind_port(Port::PageIndptr, PortInput::Channel(page_indptr.clone()));
        builder.bind_port(Port::WSlot, PortInput::Channel(w_slot.clone()));
        builder.bind_port(Port::WOff, PortInput::Channel(w_off.clone()));
        builder.stage(Stage::Prologue, || geometry.trace());
        let traced = builder.build().expect("geometry trace builds");
        let container = traced.container().clone();
        bind(container.clone(), ModelProfile::dummy()).expect("generated geometry validates");
        GeometryTrace {
            container,
            tokens: channel_index(&traced, &tokens),
            indptr: channel_indptr.then(|| channel_index(&traced, &indptr_channel)),
            kv_len: channel_index(&traced, &kv_len),
            positions: channel_index(&traced, &positions),
            pages: channel_index(&traced, &pages),
            page_indptr: channel_index(&traced, &page_indptr),
            w_slot: channel_index(&traced, &w_slot),
            w_off: channel_index(&traced, &w_off),
        }
    }

    fn assert_read_only(container: &TraceContainer, channel: u32) {
        let ops = &container.stages[0].ops;
        assert!(
            ops.iter()
                .any(|op| matches!(op, Op::ChanRead(bound) if *bound == channel))
        );
        assert!(!ops.iter().any(|op| {
            matches!(op, Op::ChanTake(bound) if *bound == channel)
                || matches!(op, Op::ChanPut { chan, .. } if *chan == channel)
        }));
    }

    #[test]
    fn const_one_lane_prefill_preserves_csr_and_channel_shapes() {
        let trace = trace_geometry(4, 1, false, 3, 4, [true; 5]);

        assert_eq!(
            trace.container.channels[trace.positions as usize].shape,
            Shape::vector(4)
        );
        assert_eq!(
            trace.container.channels[trace.pages as usize].shape,
            Shape::matrix(1, 3)
        );
        assert_eq!(
            trace.container.channels[trace.page_indptr as usize].shape,
            Shape::vector(2)
        );
        assert_eq!(
            trace.container.channels[trace.w_slot as usize].shape,
            Shape::vector(4)
        );
        assert_eq!(
            trace.container.channels[trace.w_off as usize].shape,
            Shape::vector(4)
        );
        let embed_indptr = trace
            .container
            .ports
            .iter()
            .find(|binding| binding.port == Port::EmbedIndptr)
            .expect("EmbedIndptr remains bound");
        let PortSource::Const { data, .. } = &embed_indptr.source else {
            panic!("one-lane EmbedIndptr remains the caller's constant");
        };
        let values: Vec<u32> = data
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("u32 word")))
            .collect();
        assert_eq!(values, [0, 4]);
        assert_read_only(&trace.container, trace.kv_len);
    }

    #[test]
    fn channel_csr_is_read_only_and_segments_live_ranks() {
        let trace = trace_geometry(5, 2, true, 4, 4, [true; 5]);

        assert_eq!(
            trace.container.channels[trace.pages as usize].shape,
            Shape::matrix(2, 4)
        );
        assert_read_only(&trace.container, trace.kv_len);
        assert_read_only(
            &trace.container,
            trace.indptr.expect("channel EmbedIndptr is retained"),
        );
        let ops = &trace.container.stages[0].ops;
        assert!(ops.iter().any(|op| matches!(op, Op::Transpose(_))));
        assert!(ops.iter().filter(|op| matches!(op, Op::CumSum(_))).count() >= 2);
    }

    #[test]
    fn partial_override_generates_only_the_missing_port() {
        let trace = trace_geometry(3, 1, false, 2, 4, [false, false, true, false, false]);
        let ops = &trace.container.stages[0].ops;

        assert!(
            ops.iter()
                .any(|op| matches!(op, Op::ChanTake(bound) if *bound == trace.page_indptr))
        );
        for untouched in [trace.positions, trace.pages, trace.w_slot, trace.w_off] {
            assert!(!ops.iter().any(|op| {
                matches!(op, Op::ChanTake(bound) if *bound == untouched)
                    || matches!(op, Op::ChanPut { chan, .. } if *chan == untouched)
            }));
        }
        assert_read_only(&trace.container, trace.kv_len);
        assert!(
            !ops.iter()
                .any(|op| matches!(op, Op::ChanRead(bound) if *bound == trace.tokens))
        );
    }
}

impl ForwardPass {
    pub fn new() -> ForwardPass {
        ForwardPass {
            inner: RefCell::new(ForwardInner {
                ports: Vec::new(),
                stages: Vec::new(),
                attn: None,
                derive_dense_geometry: false,
                dense_page_capacity: None,
                auto_geometry: None,
                geometry_materialized: false,
                rs_working_sets: Vec::new(),
                bound: None,
            }),
        }
    }

    fn invalidate(&self) {
        self.inner.borrow_mut().bound = None;
    }

    /// `embed(&toks, indptr)` — token ids per lane; consumes (take). `indptr`
    /// is a trace-known constant for rectangular batches, or a channel.
    pub fn embed(&self, toks: &Channel, indptr: impl Indptr) {
        {
            let mut inner = self.inner.borrow_mut();
            inner.ports.push(PortSpec::Channel(
                Port::EmbedTokens,
                claim_port(Port::EmbedTokens, toks),
            ));
            match indptr.resolve() {
                IndptrSpec::Const(t) => inner.ports.push(PortSpec::Const(Port::EmbedIndptr, t)),
                IndptrSpec::Channel(c) => inner.ports.push(PortSpec::Channel(Port::EmbedIndptr, c)),
            }
        }
        self.invalidate();
    }

    /// `positions(&pos)` — explicit RoPE positions; consumes (take).
    pub fn positions(&self, pos: &Channel) {
        self.inner.borrow_mut().ports.push(PortSpec::Channel(
            Port::Positions,
            claim_port(Port::Positions, pos),
        ));
        self.invalidate();
    }

    /// Declare this pass's attention working set and readable/writable page
    /// spans. Open-ended ranges (for example `..` or `start..`) follow later
    /// working-set growth. These are per-fire access declarations, not page
    /// inventories: the runtime checks each fire's resolved dense read/write
    /// pages against them.
    ///
    /// Every pass with [`Port::EmbedTokens`] must author-bind [`Port::KvLen`],
    /// whose values are each lane's post-write readable token extent. Geometry
    /// is never inferred from this declaration: bind every geometry port
    /// explicitly, or opt into the SDK's named dense lowering with
    /// [`derive_dense_geometry`](Self::derive_dense_geometry).
    ///
    /// Before first submit the declaration is passed to `forward-pass.new`.
    /// Re-calling this method after bind atomically updates the native pass in
    /// place; it never recreates the pass or resets its channels.
    pub fn attn_working_set<R, W>(
        &self,
        ws: &WorkingSet,
        readable: R,
        writable: W,
    ) -> Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        let readable = PageDeclaration::from_range(readable)?;
        let writable = PageDeclaration::from_range(writable)?;
        let (bound, generated_page_limit) = {
            let inner = self.inner.borrow();
            (
                inner.bound.clone(),
                inner
                    .auto_geometry
                    .as_ref()
                    .and_then(|geometry| geometry.pages.as_ref().map(|_| geometry.page_count)),
            )
        };
        if let Some(bound) = bound {
            if generated_page_limit.is_some_and(|limit| ws.page_len() > limit) {
                return Err(
                    "attention lease growth exceeds this pass's generated page envelope; \
                     construct a new pass"
                        .to_string(),
                );
            }
            bound.set_attn_working_set(ws.kv.as_ref(), readable.wit(), writable.wit())?;
        }
        self.inner.borrow_mut().attn = Some(AttnWorkingSet {
            ws: ws.kv.clone(),
            readable,
            writable,
        });
        Ok(())
    }

    /// Explicitly select the SDK's ordinary dense layout lowering.
    ///
    /// On first bind, any missing `Positions`/`Pages`/`PageIndptr`/`WSlot`/
    /// `WOff` ports are traced from author-bound `EmbedTokens`, `EmbedIndptr`,
    /// and `KvLen`. Explicitly bound ports remain authoritative. Without this
    /// opt-in, every geometry port must be bound by the inferlet.
    pub fn derive_dense_geometry(&self) {
        let mut inner = self.inner.borrow_mut();
        assert!(
            inner.bound.is_none(),
            "derive_dense_geometry is construction-only"
        );
        inner.derive_dense_geometry = true;
    }

    /// Select dense lowering with an explicit static page-channel capacity,
    /// decoupled from the WorkingSet's current logical reserve frontier.
    pub fn derive_dense_geometry_with_page_capacity(&self, page_capacity: u32) {
        assert!(page_capacity > 0, "dense page capacity must be nonzero");
        let mut inner = self.inner.borrow_mut();
        assert!(
            inner.bound.is_none(),
            "derive_dense_geometry_with_page_capacity is construction-only"
        );
        inner.derive_dense_geometry = true;
        inner.dense_page_capacity = Some(page_capacity);
    }

    /// Add one recurrent-state working set in resolved request order.
    ///
    /// This additive convenience is for initial construction. Use
    /// [`set_rs_working_sets`](Self::set_rs_working_sets) after the pass has
    /// been bound.
    pub fn rs_working_set(&self, rs: &RsWorkingSet) {
        let mut inner = self.inner.borrow_mut();
        assert!(
            inner.bound.is_none(),
            "rs_working_set is construction-only after binding; use set_rs_working_sets"
        );
        inner.rs_working_sets.push(rs.rs.clone());
    }

    /// Replace the recurrent-state bindings in resolved request order.
    ///
    /// Hybrid / linear-attention forwards bind one working set per request
    /// row; pure-attention forwards pass an empty slice. Replacing the list
    /// updates the already-bound host pass in-place; it never recreates the
    /// native instance or resets seeds/KV/device geometry.
    pub fn set_rs_working_sets(&self, working_sets: &[RsWorkingSet]) -> Result<(), String> {
        let replacement: Vec<Rc<crate::working_set::RsWorkingSet>> =
            working_sets.iter().map(|rs| rs.rs.clone()).collect();
        let mut inner = self.inner.borrow_mut();
        if let Some(bound) = inner.bound.as_ref() {
            let borrows: Vec<&crate::working_set::RsWorkingSet> =
                replacement.iter().map(Rc::as_ref).collect();
            bound.set_rs_working_sets(&borrows)?;
        }
        inner.rs_working_sets = replacement;
        Ok(())
    }

    /// `attn_mask(&m)` — masks this pass's queries over the KV axis (peeked).
    pub fn attn_mask(&self, m: &Channel) {
        self.inner.borrow_mut().ports.push(PortSpec::Channel(
            Port::AttnMask,
            claim_port(Port::AttnMask, m),
        ));
        self.invalidate();
    }

    /// Bind a descriptor [`Port`] directly to a channel — the escape hatch for
    /// device-geometry ports (e.g. `PageIndptr`/`Pages` fed by device-computed
    /// wire-form channels). Records the port's endpoint claim per its
    /// consumption discipline, exactly like the other port setters.
    pub fn port_channel(&self, port: Port, ch: &Channel) {
        self.inner
            .borrow_mut()
            .ports
            .push(PortSpec::Channel(port, claim_port(port, ch)));
        self.invalidate();
    }

    /// Bind a descriptor [`Port`] to a trace-known constant — the const
    /// companion of [`port_channel`](Self::port_channel) for explicit
    /// geometry (e.g. a fixed `PageIndptr` alongside channel-fed `Pages`).
    pub fn port_const(&self, port: Port, value: &Tensor) {
        self.inner
            .borrow_mut()
            .ports
            .push(PortSpec::Const(port, value.clone()));
        self.invalidate();
    }

    /// `readout(&out_idx)` — which positions are read out. A constant `out_idx`
    /// fixes the read-out row count.
    pub fn readout(&self, out_idx: &Tensor) {
        self.port_const(Port::Readout, out_idx);
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
        {
            let mut inner = self.inner.borrow_mut();
            if let Some(slot) = inner.stages.iter_mut().find(|(s, _)| *s == stage) {
                slot.1 = Box::new(body);
            } else {
                inner.stages.push((stage, Box::new(body)));
            }
        }
        self.invalidate();
    }

    /// `submit(&pipeline)` — enqueue this pass run-ahead on `on`. RS-bound
    /// passes serialize preparation behind prior FIFO operations; pure
    /// attention remains non-blocking. The first submit lowers + binds
    /// (`forward-pass.new`); bind errors surface here.
    pub fn submit(&self, on: &Pipeline) -> Result<(), String> {
        let bound = self.bound()?;
        bound.submit(&on.wit)
    }

    /// Lower + bind, memoized. Traces the stage closures once into the canonical
    /// container, orders the WIT channel handles by the builder↔bridge contract,
    /// and calls `forward-pass.new` (bind errors surface here as the validator's
    /// message).
    fn bound(&self) -> Result<Rc<wit::ForwardPass>, String> {
        if let Some(fp) = &self.inner.borrow().bound {
            return Ok(fp.clone());
        }
        {
            let mut inner = self.inner.borrow_mut();
            if inner.derive_dense_geometry {
                inner.materialize_geometry();
            }
            let missing = [
                Port::Positions,
                Port::Pages,
                Port::PageIndptr,
                Port::WSlot,
                Port::WOff,
            ]
            .into_iter()
            .filter(|&port| !inner.has_port(port))
            .map(|port| port.name())
            .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(format!(
                    "forward pass is missing explicit geometry ports: {}; \
                     bind them or call derive_dense_geometry()",
                    missing.join(", ")
                ));
            }
        }

        let fp = {
            let inner = self.inner.borrow();
            let mut builder = Builder::new();
            for spec in &inner.ports {
                match spec {
                    PortSpec::Channel(port, ch) => {
                        builder.bind_port_recorded(*port, PortInput::Channel(ch.clone()))
                    }
                    PortSpec::Const(port, t) => {
                        builder.bind_port(*port, PortInput::Const(t.clone()))
                    }
                }
            }
            for (stage, body) in &inner.stages {
                if *stage == Stage::Prologue && inner.auto_geometry.is_some() {
                    continue;
                }
                builder.stage(*stage, body);
            }
            if let Some(geometry) = inner.auto_geometry.as_ref() {
                builder.stage(Stage::Prologue, || {
                    if let Some((_, body)) = inner
                        .stages
                        .iter()
                        .find(|(stage, _)| *stage == Stage::Prologue)
                    {
                        body();
                    }
                    geometry.trace();
                });
            }
            let traced = builder.build().map_err(|e| e.to_string())?;
            drop(builder);

            let handles: Vec<Rc<wit::Channel>> = traced
                .channel_order()
                .iter()
                .map(|gid| lookup_channel(*gid).expect("channel registered before submit"))
                .collect();
            let borrows: Vec<&wit::Channel> = handles.iter().map(|rc| rc.as_ref()).collect();
            let attn = inner.attn.as_ref().ok_or_else(|| {
                "attention working set must be declared before submit".to_string()
            })?;
            let rs_borrows: Vec<&crate::working_set::RsWorkingSet> =
                inner.rs_working_sets.iter().map(|rc| rc.as_ref()).collect();
            let bytes = traced.encode();
            Rc::new(wit::ForwardPass::new(
                &bytes,
                &borrows,
                attn.ws.as_ref(),
                attn.readable.wit(),
                attn.writable.wit(),
                &rs_borrows,
            )?)
        };

        self.inner.borrow_mut().bound = Some(fp.clone());
        Ok(fp)
    }
}

impl Default for ForwardPass {
    fn default() -> Self {
        ForwardPass::new()
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// A run-ahead ordering domain (overview §3, pipeline.wit) — every command on
/// it linearizes in submission order through the per-driver sequencer.
/// Ordering across fires is carried by the channels' full/empty bits, not
/// host code. Working-set mutators ([`WorkingSet::fork`]/`slice`/`discard`/
/// `copy_into`) and [`ForwardPass::submit`] take `&Pipeline`.
///
/// # Canonical usage (one pipeline per sequential stream)
///
/// A `Pipeline` is an ordering domain, not a program: heterogeneous passes
/// (an N-wide prefill, then a loop-carried decode) are ONE sequential
/// stream and belong on ONE pipeline — never split phases of the same
/// stream across pipelines. Call [`Pipeline::finish`] right after the last
/// submit; on an early stop (stop token), call [`Pipeline::close`], which
/// CANCELS everything still unexecuted — takes of cancelled fires error,
/// so a drain loop must tolerate that. Separate pipelines are for
/// genuinely CONCURRENT streams only (draft vs target model in speculative
/// decoding, parallel beam branches, independent requests) — each such
/// stream still ends with its own `finish()`.
pub struct Pipeline {
    wit: wit_pipeline::Pipeline,
}

impl Pipeline {
    pub fn new() -> Pipeline {
        Pipeline {
            wit: wit_pipeline::Pipeline::new(),
        }
    }

    /// `finish()` — graceful END OF STREAM: no further submissions; queued
    /// fires drain normally. Sequenced with the submissions (same FIFO),
    /// so calling it right after the last submit is exact — the engine
    /// stops awaiting the pipeline the moment its last fire dispatches,
    /// and the drain tail never holds the wave barrier. Later submits
    /// error; `close()` stays legal afterwards.
    pub fn finish(&self) {
        self.wit.finish();
    }

    /// `close()` — CANCEL the pipeline (implied by drop; close and drop
    /// mean the same thing): queued/preparing fires are cancelled and
    /// discarded, already-dispatched fires run to settlement. This is the
    /// early-stop/abort path — the normal end of a stream is
    /// [`Pipeline::finish`] after its last submission.
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
// Port argument traits (mirrors the ptir-dsl author surface)
// ---------------------------------------------------------------------------

/// An `embed` indptr: a trace-known constant, or a channel.
pub enum IndptrSpec {
    Const(Tensor),
    Channel(DslChannel),
}

/// Anything usable as an `embed` indptr.
pub trait Indptr {
    fn resolve(self) -> IndptrSpec;
}
impl Indptr for Tensor {
    fn resolve(self) -> IndptrSpec {
        IndptrSpec::Const(self)
    }
}
impl Indptr for &Channel {
    fn resolve(self) -> IndptrSpec {
        IndptrSpec::Channel(self.dsl())
    }
}

// ---------------------------------------------------------------------------
// prelude
// ---------------------------------------------------------------------------

/// Glob-import surface for PTIR inferlet authors: the eDSL vocabulary plus the
/// four author-facing wrapper types.
pub mod prelude {
    pub use super::{
        Channel, DEFAULT_RUNAHEAD_DEPTH, ForwardPass, PageGrant, Pipeline, RsWorkingSet, TOKEN_PAD,
        WorkingSet, pad_tokens, unpad_tokens,
    };
    pub use ptir_dsl::dtype;
    pub use ptir_dsl::intrinsics;
    pub use ptir_dsl::model;
    pub use ptir_dsl::value::{
        AsTensor, Tensor, add, and, broadcast, cast, causal_mask, cummass_le, cumprod, cumsum, div,
        entropy, entropy_from_logprobs, eq, exp, gather, gather_row, ge, gt, gumbel, gumbel_max,
        iota, l2norm, le, log, log_softmax, lt, mask_apply, masked_argmax, matmul, max_elem,
        min_elem, mul, ne, neg, not, nucleus_sample, or, pivot_threshold, prob_ge, rank_le,
        reduce_argmax, reduce_max, reduce_min, reduce_sum, rem, reshape, rng, row_membership,
        scalar_gather, scatter_add, scatter_set, select, sink_window_mask, sliding_window_mask,
        softmax, sub, top_k, transpose,
    };
    pub use ptir_dsl::{DType, Port, Stage};
}
