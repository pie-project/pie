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
//! the empty `forward-pass` builder and attaches the traced program (which
//! binds against the model — the guest does not bind, D6). Program identity,
//! dedup, and validation happen host-side at program attachment.
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

use ptir_dsl::builder::Builder;
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
pub use ptir_dsl::DType as Dtype;
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
// dsl channel's gid; `forward-pass.program` wants the WIT handles in that dense
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
/// [`ForwardPass::rs_working_sets`] for models whose
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

type StageClosure = Box<dyn Fn()>;

/// A forward-pass builder. Its WIT resource is constructed empty, descriptor
/// resources are attached through typed methods, and the traced program is
/// attached once on first submit.
pub struct ForwardPass {
    wit: Rc<wit::ForwardPass>,
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

impl ForwardPass {
    pub fn new() -> ForwardPass {
        let vocab = crate::model::output_vocab_size();
        let page_size = crate::model::kv_page_size();
        ForwardPass {
            wit: Rc::new(wit::ForwardPass::new()),
            inner: RefCell::new(ForwardInner {
                ports: Vec::new(),
                stages: Vec::new(),
                vocab,
                page_size,
                attention_ws: None,
                rs_working_sets: Vec::new(),
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
        self.wit.embed(token_wit.as_ref(), indptr_wit.as_ref())?;
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
        self.wit.attention(
            ws.kv.as_ref(),
            readable.wit(),
            writable.wit(),
            kv_len_wit.as_ref(),
            pages_wit.as_ref(),
            page_indptr_wit.as_ref(),
            w_slot_wit.as_ref(),
            w_off_wit.as_ref(),
            positions_wit.as_ref(),
            mask_wit.as_deref(),
        )?;

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
        Ok(())
    }

    /// Bind recurrent-state working sets in resolved request order.
    pub fn rs_working_sets(&self, working_sets: &[RsWorkingSet]) -> Result<(), String> {
        let replacement: Vec<Rc<crate::working_set::RsWorkingSet>> =
            working_sets.iter().map(|rs| rs.rs.clone()).collect();
        let borrows: Vec<&crate::working_set::RsWorkingSet> =
            replacement.iter().map(Rc::as_ref).collect();
        self.wit.set_rs_working_sets(&borrows)?;
        self.inner.borrow_mut().rs_working_sets = replacement;
        Ok(())
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

    /// Enqueue this pass as a SINGLE-SLOT FRAME on `on` (slot 0; the other
    /// k−1 slots are no-ops). At the default deployment (`frame_size() == 1`)
    /// this is exactly the classic per-pass run-ahead submit. At k > 1 it
    /// costs one whole frame per pass — k times fewer tokens per boundary —
    /// so hot loops should fill all k slots via [`submit_frame`].
    pub fn submit(&self, on: &Pipeline) -> Result<(), String> {
        submit_frame(on, &[Some(self)])
    }

    fn attach_program(&self) -> Result<(), String> {
        if self.inner.borrow().program_attached {
            return Ok(());
        }

        let inner = self.inner.borrow();
        let required = [
            Port::EmbedTokens,
            Port::EmbedIndptr,
            Port::KvLen,
            Port::Pages,
            Port::PageIndptr,
            Port::WSlot,
            Port::WOff,
            Port::Positions,
        ];
        let missing = required
            .into_iter()
            .filter(|port| !inner.ports.iter().any(|(bound, _)| bound == port))
            .map(Port::name)
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(format!(
                "forward pass is missing descriptor channels: {}",
                missing.join(", ")
            ));
        }
        if inner.attention_ws.is_none() {
            return Err("attention must be bound before submit".to_string());
        }

        let mut builder = Builder::new(inner.vocab, inner.page_size);
        for (port, channel) in &inner.ports {
            builder.bind_port_recorded(*port, channel.clone());
        }
        for (stage, body) in &inner.stages {
            builder.stage(*stage, body);
        }
        let traced = builder.build().map_err(|error| error.to_string())?;
        drop(builder);
        let handles: Vec<Rc<wit::Channel>> = traced
            .channel_order()
            .iter()
            .map(|gid| lookup_channel(*gid).expect("channel registered before submit"))
            .collect();
        let borrows: Vec<&wit::Channel> = handles.iter().map(Rc::as_ref).collect();
        let bytes = traced.encode();
        self.wit.program(&bytes, &borrows)?;
        drop(inner);
        self.inner.borrow_mut().program_attached = true;
        Ok(())
    }
}

impl Default for ForwardPass {
    fn default() -> Self {
        ForwardPass::new()
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

/// Max embed tokens in a single pass (C) — the guest-side prefill chunk
/// budget (cached). Split a prompt of L tokens into `ceil(L / C)` chunks.
pub fn max_embed_length() -> usize {
    thread_local! {
        static MAX_EMBED: std::cell::OnceCell<usize> = const { std::cell::OnceCell::new() };
    }
    MAX_EMBED.with(|c| *c.get_or_init(|| crate::model::max_embed_length().max(1) as usize))
}

/// Submit ONE FRAME on `on`: up to `frame_size()` ordered slots, slot i
/// executing in wave i; missing trailing slots are padded with no-ops. The
/// same pass may repeat across slots (a plain decode frame is the same pass
/// in every slot) and slots may be heterogeneous (prefill chunks first, then
/// decode). First submit of a pass traces and attaches its program;
/// attachment, bind, and frame-validation errors surface here.
pub fn submit_frame(on: &Pipeline, slots: &[Option<&ForwardPass>]) -> Result<(), String> {
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
    let wits: Vec<Option<Rc<wit::ForwardPass>>> = slots
        .iter()
        .map(|slot| slot.map(|pass| pass.wit.clone()))
        .collect();
    let mut borrows: Vec<Option<&wit::ForwardPass>> = wits
        .iter()
        .map(|slot| slot.as_deref())
        .collect();
    borrows.resize(k, None);
    wit::submit(&on.wit, &borrows)
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

/// Glob-import surface for PTIR inferlet authors: the eDSL vocabulary plus the
/// four author-facing wrapper types.
pub mod prelude {
    pub use super::{
        Channel, DEFAULT_RUNAHEAD_DEPTH, ForwardPass, PageGrant, Pipeline, RsWorkingSet, TOKEN_PAD,
        WorkingSet, frame_size, max_embed_length, pad_tokens, submit_frame, unpad_tokens,
    };
    pub use ptir_dsl::dtype;
    pub use ptir_dsl::intrinsics;
    pub use ptir_dsl::value::{
        AsTensor, Tensor, add, and, broadcast, cast, causal_mask, cummass_le, cumprod, cumsum, div,
        entropy, entropy_from_logprobs, eq, exp, gather, gather_row, ge, gt, gumbel, gumbel_max,
        iota, l2norm, le, log, log_softmax, lt, mask_apply, masked_argmax, matmul, max_elem,
        min_elem, mul, ne, neg, not, nucleus_sample, or, pivot_threshold, prob_ge, rank_le,
        reduce_argmax, reduce_max, reduce_min, reduce_sum, rem, reshape, rng, row_membership,
        scalar_gather, scatter_add, scatter_set, select, sink_window_mask, sliding_window_mask,
        softmax, sub, top_k, transpose,
    };
    pub use ptir_dsl::{DType, Stage};
}
