//! `inferlet::ptir` — the author-facing PTIR bridge over the WIT `ptir` surface.
//!
//! This is the only home of the overview §3/§5 author surface
//! (`ForwardPass`/`Pipeline`/`WorkingSet`/`Channel`). It wraps the WIT `ptir`
//! resources (`channel`, `forward-pass`, `pipeline`) and drives the neutral
//! [`Builder`](ptir_dsl::Builder) from the `ptir-dsl` crate: the author writes
//! stage closures + port bindings, the bridge lowers them to the canonical PTIR
//! container, orders the WIT channel handles by the builder↔bridge contract
//! ([`Traced::channel_order`](ptir_dsl::Traced::channel_order)), and calls
//! `forward-pass.new` (which binds against the model — the guest does not bind,
//! D6). Program identity, dedup, and validation happen host-side inside
//! `forward-pass.new`/`pipeline.submit`.
//!
//! A [`Channel`] owns BOTH sides: the `ptir-dsl` trace declaration (its `take`/
//! `put`/`read` record ops inside a stage closure, and host `put`s record the
//! host-role endpoint) and the WIT `channel` resource (the host transport). The
//! two are constructed from the same `(shape, dtype, capacity)` so the decl
//! validates against the container by construction.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use ptir_dsl::builder::{Builder, PortInput};
use ptir_dsl::channel::PutValue;
use ptir_dsl::value::{Arg, ConstData};
use ptir_dsl::{
    AsTensor, Channel as DslChannel, DType, IntoConst, IntoPut, IntoShape, Port, Shape, Stage,
    Tensor,
};

use crate::pie::core::ptir as wit;
use crate::pie::core::tensor::Dtype as WitDtype;
use crate::working_set::KvWorkingSet;

pub use ptir_dsl::intrinsics;

// Re-export the eDSL vocabulary so an author writes stage closures with a single
// `use inferlet::ptir::prelude::*;` (mirrors the old `ptir::prelude`).
pub use ptir_dsl::{
    add, and, broadcast, cast, cumprod, cummass_le, cumsum, div, dtype, eq, exp, gather,
    gather_row, ge, gt, gumbel, iota, l2norm, le, log, log_softmax, lt, mask_apply, matmul,
    max_elem, min_elem, mul, ne, neg, not, or, pivot_threshold, prob_ge, rank_le, reduce_argmax,
    reduce_max, reduce_min, reduce_sum, rem, reshape, rng, scatter_add, scatter_set, select,
    softmax, sub, top_k, transpose,
};
pub use ptir_dsl::{model, DType as Dtype};

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

/// A GPU-resident bounded queue (overview §1). Owns the `ptir-dsl` trace
/// declaration and the WIT `channel` resource. In a stage closure `take`/`read`/
/// `put` record IR ops; on the host `put` stages a value (seed / host-writer
/// cell) and `Taken::get` moves a committed value out.
pub struct Channel {
    dsl: DslChannel,
    wit: Rc<wit::Channel>,
}

impl Channel {
    /// `Channel::new([shape], dtype)` at capacity 1 (overview §1).
    pub fn new(shape: impl IntoShape, dtype: DType) -> Channel {
        Channel::build(shape.into_shape(), dtype, 1, false)
    }

    /// Widen the ring to `n` cells (deeper run-ahead).
    pub fn capacity(self, n: u32) -> Channel {
        let shape = self.dsl.shape();
        let dtype = self.dsl.dtype();
        let dsl = self.dsl.capacity(n);
        let wit = Rc::new(wit::Channel::new(&dims_of(shape), to_wit_dtype(dtype), n));
        register_channel(dsl.gid(), wit.clone());
        Channel { dsl, wit }
    }

    /// Name the channel (improves trace-error messages).
    pub fn named(mut self, name: &str) -> Channel {
        self.dsl = self.dsl.named(name);
        self
    }

    /// `Channel::from(v)` — a channel seeded full with the per-instance value
    /// `v` (overview §1). The seed is instance data (D2): it rides the WIT
    /// channel as a pre-submit `put`, never the container.
    pub fn from(v: impl IntoConst) -> Channel {
        let data: ConstData = v.into_const();
        let ch = Channel::build(data.shape, data.dtype, 1, true);
        ch.wit
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
        ch.wit
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
        let dsl = if capacity != 1 { dsl.capacity(capacity) } else { dsl };
        let wit = Rc::new(wit::Channel::new(&dims_of(shape), to_wit_dtype(dtype), capacity));
        register_channel(dsl.gid(), wit.clone());
        Channel { dsl, wit }
    }

    pub fn dtype(&self) -> DType {
        self.dsl.dtype()
    }
    pub fn shape(&self) -> Shape {
        self.dsl.shape()
    }

    /// `take()` — consume a cell. In a stage closure: records a `ChanTake` and
    /// yields an in-program value ([`AsTensor`]). On the host: [`Taken::get`]
    /// moves the committed value out (blocks until a fire fills it; poison ⇒
    /// `Err`).
    pub fn take(&self) -> Taken {
        let dsl = self.dsl.take();
        Taken { dsl, wit: self.wit.clone(), dtype: self.dsl.dtype() }
    }

    /// `read()` — peek a cell (leaves it full). Same dual as [`take`](Self::take).
    pub fn read(&self) -> Taken {
        let dsl = self.dsl.read();
        Taken { dsl, wit: self.wit.clone(), dtype: self.dsl.dtype() }
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
                self.dsl.put(t);
            }
            PutValue::Data(data) => {
                self.dsl.note_host_put();
                let _ = self.wit.put(&data.bytes);
            }
        }
    }
}

/// The result of [`Channel::take`]/[`Channel::read`]. In a stage closure it is
/// an in-program value (via [`AsTensor`]); on the host [`get`](Self::get) /
/// [`bytes`](Self::bytes) move the committed value out.
pub struct Taken {
    dsl: ptir_dsl::Taken,
    wit: Rc<wit::Channel>,
    dtype: DType,
}

impl Taken {
    /// The in-program [`Tensor`] (panics on a host take — a frontend bug).
    pub fn tensor(self) -> Tensor {
        self.dsl.tensor()
    }

    /// Move the committed value out to the host as raw little-endian bytes.
    /// Blocks by awaiting in-flight fires; a poisoned channel returns `Err`.
    pub fn bytes(self) -> Result<Vec<u8>, String> {
        self.wit.take()
    }

    /// Move the committed value out to the host, decoded to `T`.
    pub fn get<T: HostElem>(self) -> Result<Vec<T>, String> {
        let raw = self.wit.take()?;
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
        raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
}
impl HostElem for u32 {
    fn decode(raw: &[u8]) -> Vec<u32> {
        raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
}
impl HostElem for f32 {
    fn decode(raw: &[u8]) -> Vec<f32> {
        raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
}

// ---------------------------------------------------------------------------
// WorkingSet
// ---------------------------------------------------------------------------

/// The attention working set (overview §5.2) — a flat pool of KV page slots.
/// Wraps the WIT `kv-working-set`; the host owns its shape (`alloc`/`free`).
pub struct WorkingSet {
    kv: Rc<KvWorkingSet>,
}

impl WorkingSet {
    pub fn new() -> WorkingSet {
        WorkingSet { kv: Rc::new(KvWorkingSet::new()) }
    }

    /// Tokens per KV page for this working set's model.
    pub fn page_size(&self) -> u32 {
        self.kv.page_size()
    }

    /// Current number of page slots.
    pub fn size(&self) -> u32 {
        self.kv.size()
    }

    /// Grant `n` fresh (or recycled) stable slot ids — per-instance data that
    /// flows through a channel (`fresh.put(ws.alloc(B)?)`), never a trace
    /// constant (D2).
    pub fn alloc(&self, n: u32) -> Result<SlotGrant, String> {
        let ids = self.kv.alloc_slots(n)?;
        Ok(SlotGrant { ids })
    }

    /// Tombstone the slots at `ids` (non-compacting).
    pub fn free(&self, ids: Vec<u32>) -> Result<(), String> {
        self.kv.free_slots(&ids)
    }
}

impl Default for WorkingSet {
    fn default() -> Self {
        WorkingSet::new()
    }
}

/// A grant of slot ids — per-instance data (D2). Puttable into a channel.
pub struct SlotGrant {
    ids: Vec<u32>,
}

impl SlotGrant {
    /// The granted physical slot ids (stable page ids owned by the working set).
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
}

impl IntoPut for SlotGrant {
    fn into_put(self) -> PutValue {
        PutValue::Data(self.ids.into_const())
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

type StageClosure<'a> = Box<dyn Fn() + 'a>;

/// The forward pass (overview §5). Attach descriptor ports + stage closures,
/// submit through a [`Pipeline`]. On first submit the bridge drives the builder,
/// lowers to the container, and calls `forward-pass.new`; the bound WIT resource
/// is memoized. The lifetime lets stage closures borrow the channels they touch.
pub struct ForwardPass<'a> {
    inner: RefCell<ForwardInner<'a>>,
}

struct ForwardInner<'a> {
    ports: Vec<PortSpec>,
    stages: Vec<(Stage, StageClosure<'a>)>,
    working_sets: Vec<Rc<KvWorkingSet>>,
    bound: Option<Rc<wit::ForwardPass>>,
}

impl<'a> ForwardPass<'a> {
    pub fn new() -> ForwardPass<'a> {
        ForwardPass {
            inner: RefCell::new(ForwardInner {
                ports: Vec::new(),
                stages: Vec::new(),
                working_sets: Vec::new(),
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
            inner.ports.push(PortSpec::Channel(Port::EmbedTokens, toks.dsl.clone()));
            match indptr.resolve() {
                IndptrSpec::Const(t) => inner.ports.push(PortSpec::Const(Port::EmbedIndptr, t)),
                IndptrSpec::Channel(c) => {
                    inner.ports.push(PortSpec::Channel(Port::EmbedIndptr, c))
                }
            }
        }
        self.invalidate();
    }

    /// `positions(&pos)` — explicit RoPE positions; consumes (take).
    pub fn positions(&self, pos: &Channel) {
        self.inner.borrow_mut().ports.push(PortSpec::Channel(Port::Positions, pos.dsl.clone()));
        self.invalidate();
    }

    /// `attn_working_set(&ws, ..)` — binds attention's memory in one call.
    pub fn attn_working_set(&self, ws: &WorkingSet, args: impl AttnWsArgs) {
        let r = args.resolve();
        {
            let mut inner = self.inner.borrow_mut();
            inner.ports.push(PortSpec::Channel(Port::KvLen, r.kv_len));
            if let Some(p) = r.pages {
                inner.ports.push(PortSpec::Channel(Port::Pages, p));
            }
            if let Some(pi) = r.page_indptr {
                inner.ports.push(PortSpec::Const(Port::PageIndptr, pi));
            }
            if let Some(w) = r.w_slot {
                inner.ports.push(PortSpec::Channel(Port::WSlot, w));
            }
            if let Some(w) = r.w_off {
                inner.ports.push(PortSpec::Channel(Port::WOff, w));
            }
            inner.working_sets.push(ws.kv.clone());
        }
        self.invalidate();
    }

    /// `attn_mask(&m)` — masks this pass's queries over the KV axis (peeked).
    pub fn attn_mask(&self, m: &Channel) {
        self.inner.borrow_mut().ports.push(PortSpec::Channel(Port::AttnMask, m.dsl.clone()));
        self.invalidate();
    }

    /// Bind a descriptor [`Port`] directly to a channel — the escape hatch for
    /// device-geometry ports (e.g. `PageIndptr`/`Pages` fed by device-computed
    /// wire-form channels) that the `attn_working_set` sugar (which takes a const
    /// `page_indptr`) cannot express. Records the port's endpoint claim per its
    /// consumption discipline, exactly like the other port setters.
    pub fn port_channel(&self, port: Port, ch: &Channel) {
        self.inner.borrow_mut().ports.push(PortSpec::Channel(port, ch.dsl.clone()));
        self.invalidate();
    }

    /// `readout(&out_idx)` — which positions are read out. A constant `out_idx`
    /// fixes the read-out row count.
    pub fn readout(&self, out_idx: &Tensor) {
        self.inner.borrow_mut().ports.push(PortSpec::Const(Port::Readout, out_idx.clone()));
        self.invalidate();
    }

    /// Attach the `prologue` stage (overview §5.3).
    pub fn prologue(&self, body: impl Fn() + 'a) {
        self.set_stage(Stage::Prologue, body);
    }
    /// Attach the `on_attn_proj` stage (per layer, before attention).
    pub fn on_attn_proj(&self, body: impl Fn() + 'a) {
        self.set_stage(Stage::OnAttnProj, body);
    }
    /// Attach the `on_attn` stage (per layer, after attention).
    pub fn on_attn(&self, body: impl Fn() + 'a) {
        self.set_stage(Stage::OnAttn, body);
    }
    /// Attach the `epilogue` stage (sampling programs; after the forward).
    pub fn epilogue(&self, body: impl Fn() + 'a) {
        self.set_stage(Stage::Epilogue, body);
    }

    fn set_stage(&self, stage: Stage, body: impl Fn() + 'a) {
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

    /// Lower + bind, memoized. Traces the stage closures once into the canonical
    /// container, orders the WIT channel handles by the builder↔bridge contract,
    /// and calls `forward-pass.new` (bind errors surface here as the validator's
    /// message).
    fn bound(&self) -> Result<Rc<wit::ForwardPass>, String> {
        if let Some(fp) = &self.inner.borrow().bound {
            return Ok(fp.clone());
        }

        let fp = {
            let inner = self.inner.borrow();
            let mut builder = Builder::new();
            for spec in &inner.ports {
                match spec {
                    PortSpec::Channel(port, ch) => {
                        builder.bind_port(*port, PortInput::Channel(ch.clone()))
                    }
                    PortSpec::Const(port, t) => builder.bind_port(*port, PortInput::Const(t.clone())),
                }
            }
            for (stage, body) in &inner.stages {
                builder.stage(*stage, body);
            }
            let traced = builder.build().map_err(|e| e.to_string())?;
            drop(builder);

            let handles: Vec<Rc<wit::Channel>> = traced
                .channel_order()
                .iter()
                .map(|gid| lookup_channel(*gid).expect("channel registered before submit"))
                .collect();
            let borrows: Vec<&wit::Channel> = handles.iter().map(|rc| rc.as_ref()).collect();
            let kv_borrows: Vec<&KvWorkingSet> =
                inner.working_sets.iter().map(|rc| rc.as_ref()).collect();
            let bytes = traced.encode();
            Rc::new(wit::ForwardPass::new(&bytes, &borrows, &kv_borrows, &[])?)
        };

        self.inner.borrow_mut().bound = Some(fp.clone());
        Ok(fp)
    }
}

impl<'a> Default for ForwardPass<'a> {
    fn default() -> Self {
        ForwardPass::new()
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// A run-ahead submission pipeline (overview §3). Ordering is carried by the
/// channels' full/empty bits, not host code.
pub struct Pipeline {
    wit: wit::Pipeline,
}

impl Pipeline {
    pub fn new() -> Pipeline {
        Pipeline { wit: wit::Pipeline::new() }
    }

    /// `submit(&fwd)` — enqueue a pass run-ahead. Ensures the bound pass exists
    /// (first submit lowers + binds), then delegates; never blocks.
    pub fn submit(&self, fwd: &ForwardPass<'_>) -> Result<(), String> {
        let bound = fwd.bound()?;
        self.wit.submit(bound.as_ref())
    }

    /// `close()` — signal no further submissions (implied by drop).
    pub fn close(&self) {
        self.wit.close();
    }

    /// `copy_into(&ws, dst_pages, dst_toks, src_pages, src_toks)` — Design-B
    /// lazy KV compaction: move `n` token KV cells within `ws` (all layers)
    /// from (`src_page_ids[i]`, `src_tok_idx[i]`) to (`dst_page_ids[i]`,
    /// `dst_tok_idx[i]`). Rides the SAME run-ahead FIFO as `submit` (ordered
    /// after prior fires' writes, before later fires' reads — no barrier).
    /// Correct because KV is stored post-RoPE (a slot is pure storage). Page ids
    /// are the PHYSICAL KV page ids (the same ids bound to the `Pages`/`WSlot`
    /// ports — Design B works in physical page ids directly); token indices are
    /// in-page offsets. The four lists are parallel; the caller guarantees
    /// disjoint src/dst spans and computes the post-move layout itself (a move,
    /// not a fire — no result).
    pub fn copy_into(
        &self,
        ws: &WorkingSet,
        dst_page_ids: &[u32],
        dst_tok_idx: &[u32],
        src_page_ids: &[u32],
        src_tok_idx: &[u32],
    ) -> Result<(), String> {
        self.wit.copy_into(
            ws.kv.as_ref(),
            dst_page_ids,
            dst_tok_idx,
            src_page_ids,
            src_tok_idx,
        )
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
        IndptrSpec::Channel(self.dsl.clone())
    }
}

/// The resolved trailing args of `attn_working_set`.
pub struct AttnWsResolved {
    kv_len: DslChannel,
    pages: Option<DslChannel>,
    page_indptr: Option<Tensor>,
    w_slot: Option<DslChannel>,
    w_off: Option<DslChannel>,
}

/// The `attn_working_set` trailing-argument arities (overview §5.1).
pub trait AttnWsArgs {
    fn resolve(self) -> AttnWsResolved;
}
/// Sugar arity: `attn_working_set(&ws, &len)`.
impl AttnWsArgs for &Channel {
    fn resolve(self) -> AttnWsResolved {
        AttnWsResolved {
            kv_len: self.dsl.clone(),
            pages: None,
            page_indptr: None,
            w_slot: None,
            w_off: None,
        }
    }
}
/// Full arity: `(&pages, page_indptr, &klen, &w_slot, &w_off)`.
impl AttnWsArgs for (&Channel, Tensor, &Channel, &Channel, &Channel) {
    fn resolve(self) -> AttnWsResolved {
        let (pages, page_indptr, klen, w_slot, w_off) = self;
        AttnWsResolved {
            kv_len: klen.dsl.clone(),
            pages: Some(pages.dsl.clone()),
            page_indptr: Some(page_indptr),
            w_slot: Some(w_slot.dsl.clone()),
            w_off: Some(w_off.dsl.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// prelude
// ---------------------------------------------------------------------------

/// Glob-import surface for PTIR inferlet authors: the eDSL vocabulary plus the
/// four author-facing wrapper types.
pub mod prelude {
    pub use super::{Channel, ForwardPass, Pipeline, WorkingSet};
    pub use ptir_dsl::dtype;
    pub use ptir_dsl::intrinsics;
    pub use ptir_dsl::value::{
        add, and, broadcast, cast, cumprod, cummass_le, cumsum, div, eq, exp, gather, gather_row,
        ge, gt, gumbel, iota, l2norm, le, log, log_softmax, lt, mask_apply, matmul, max_elem,
        min_elem, mul, ne, neg, not, or, pivot_threshold, prob_ge, rank_le, reduce_argmax,
        reduce_max, reduce_min, reduce_sum, rem, reshape, rng, scatter_add, scatter_set, select,
        softmax, sub, top_k, transpose, Tensor,
    };
    pub use ptir_dsl::{DType, Port, Stage};
    pub use ptir_dsl::model;
}
