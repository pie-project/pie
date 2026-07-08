//! `ForwardPass` and `WorkingSet` — the forward contract (overview §5) and the
//! stage attachment points (§5.3). `ForwardPass::trace()` runs the stage closures
//! once, assembles echo's canonical [`TraceContainer`], runs the SDK span lints
//! (P1.3), and binds it against the model profile (echo's authoritative
//! validator), returning a [`TracedForward`].

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use pie_ptir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use pie_ptir::op::Op;
use pie_ptir::registry::{Port, Stage};
use pie_ptir::validate::{bind, BoundTrace};

use crate::channel::Channel;
use crate::context::{self, ChannelRef, SinkCall};
use crate::error::{Span, TraceError, TraceErrors};
use crate::model;
use crate::value::{ConstData, Tensor};

// ---------------------------------------------------------------------------
// WorkingSet
// ---------------------------------------------------------------------------

/// The attention working set — one flat pool of page slots (overview §5.2). The
/// host owns its shape (`alloc`/`free`); programs only inherit slot ids.
pub struct WorkingSet {
    inner: Rc<RefCell<WsInner>>,
}

struct WsInner {
    page_size: u32,
    size: u32,
    next_slot: u32,
    gid: u64,
}

impl WorkingSet {
    pub fn new() -> WorkingSet {
        static NEXT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);
        WorkingSet {
            inner: Rc::new(RefCell::new(WsInner {
                page_size: model::page_size(),
                size: 0,
                next_slot: 0,
                gid: NEXT.fetch_add(1, core::sync::atomic::Ordering::Relaxed),
            })),
        }
    }

    pub fn page_size(&self) -> u32 {
        self.inner.borrow().page_size
    }
    pub fn size(&self) -> u32 {
        self.inner.borrow().size
    }

    /// Grant `n` fresh (or recycled) slot ids — per-instance data that must flow
    /// through a channel (`fresh.put(ws.alloc(B))`), never a trace constant (D2).
    pub fn alloc(&self, n: u32) -> SlotGrant {
        let mut w = self.inner.borrow_mut();
        let base = w.next_slot;
        w.next_slot += n;
        w.size += n;
        SlotGrant { ids: (base..base + n).collect() }
    }

    /// Non-compacting `free` (tombstone + free-list; overview §5.2). Host-only.
    pub fn free(&self, ids: impl Into<Vec<u32>>) {
        let ids = ids.into();
        self.inner.borrow_mut().size = self.inner.borrow().size.saturating_sub(ids.len() as u32);
    }

    /// Token-space `compact` (overview §5.2). Stub for P1 (`gather_tokens` lands
    /// with tier-0, P4).
    pub fn compact(&self, _live_runs: &[(u32, u32)]) -> Remap {
        Remap { map: Vec::new() }
    }

    pub(crate) fn gid(&self) -> u64 {
        self.inner.borrow().gid
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

impl crate::channel::IntoPut for SlotGrant {
    fn into_put(self) -> crate::channel::PutValue {
        use crate::value::IntoConst;
        crate::channel::PutValue::Data(self.ids.into_const())
    }
}

/// The old→new slot remap returned by [`WorkingSet::compact`].
pub struct Remap {
    pub map: Vec<(u32, u32)>,
}

// ---------------------------------------------------------------------------
// ForwardPass
// ---------------------------------------------------------------------------

/// An `embed` indptr: a trace-known constant (`LANE_1`) or a channel.
pub enum IndptrSpec {
    Const(ConstData),
    Channel(ChannelRef),
}

/// The attention-working-set binding (sugar `&len`, or the full ragged arity).
struct AttnWs {
    kv_len: ChannelRef,
    pages: Option<ChannelRef>,
    page_indptr: Option<ConstData>,
    w_slot: Option<ChannelRef>,
    w_off: Option<ChannelRef>,
}

type StageClosure<'a> = Box<dyn Fn() + 'a>;

struct ForwardInner<'a> {
    embed: Option<(ChannelRef, IndptrSpec)>,
    positions: Option<ChannelRef>,
    attn_ws: Option<AttnWs>,
    attn_mask: Option<ChannelRef>,
    readout: Option<PortSource>,
    readout_rows: Option<u32>,
    prologue: Option<StageClosure<'a>>,
    on_attn_proj: Option<StageClosure<'a>>,
    on_attn: Option<StageClosure<'a>>,
    epilogue: Option<StageClosure<'a>>,
    cached: Option<Result<Rc<TracedForward>, TraceErrors>>,
}

/// The forward pass (overview §5). Build with [`ForwardPass::new`], attach
/// ports/stages, submit through a [`Pipeline`](crate::Pipeline). The lifetime
/// lets stage closures borrow the channels they read/write (no `move`).
#[derive(Clone)]
pub struct ForwardPass<'a> {
    inner: Rc<RefCell<ForwardInner<'a>>>,
}

impl<'a> ForwardPass<'a> {
    pub fn new() -> ForwardPass<'a> {
        ForwardPass {
            inner: Rc::new(RefCell::new(ForwardInner {
                embed: None,
                positions: None,
                attn_ws: None,
                attn_mask: None,
                readout: None,
                readout_rows: None,
                prologue: None,
                on_attn_proj: None,
                on_attn: None,
                epilogue: None,
                cached: None,
            })),
        }
    }

    /// `embed(&toks, indptr)` — token ids per lane; consumes (take). `indptr`
    /// folds to a constant for rectangular batches (`LANE_1`), or a channel.
    #[track_caller]
    pub fn embed(&self, toks: &Channel, indptr: impl Indptr) {
        let span = Span::here();
        toks.claim_desc_take(span);
        let spec = indptr.resolve();
        if let IndptrSpec::Channel(ch) = &spec {
            ch.borrow_mut().desc_reads.push(span);
        }
        self.inner.borrow_mut().embed = Some((toks.state().clone(), spec));
        self.invalidate();
    }

    /// `positions(&pos)` — explicit RoPE positions; consumes (take; §5.1).
    #[track_caller]
    pub fn positions(&self, pos: &Channel) {
        pos.claim_desc_take(Span::here());
        self.inner.borrow_mut().positions = Some(pos.state().clone());
        self.invalidate();
    }

    /// `attn_working_set(&ws, ..)` — binds attention's memory in one call (§5.1).
    #[track_caller]
    pub fn attn_working_set(&self, ws: &WorkingSet, args: impl AttnWsArgs) {
        let span = Span::here();
        let r = args.resolve();
        for ch in [&r.kv_len].into_iter().chain(r.pages.iter()) {
            ch.borrow_mut().desc_reads.push(span);
        }
        // w_slot / w_off are token-family (consume).
        for ch in r.w_slot.iter().chain(r.w_off.iter()) {
            ch.borrow_mut().desc_takes.push(span);
        }
        let _ = ws.gid();
        self.inner.borrow_mut().attn_ws = Some(AttnWs {
            kv_len: r.kv_len,
            pages: r.pages,
            page_indptr: r.page_indptr,
            w_slot: r.w_slot,
            w_off: r.w_off,
        });
        self.invalidate();
    }

    /// `attn_mask(&m)` — masks this pass's queries over the KV axis (§5.3; peeked).
    #[track_caller]
    pub fn attn_mask(&self, m: &Channel) {
        m.claim_desc_read(Span::here());
        self.inner.borrow_mut().attn_mask = Some(m.state().clone());
        self.invalidate();
    }

    /// `readout(&out_idx)` — which positions are read out (§5.1). A constant
    /// `out_idx` fixes the read-out row count.
    #[track_caller]
    pub fn readout(&self, out_idx: &Tensor) {
        if let Some(cd) = out_idx.as_const_data() {
            self.inner.borrow_mut().readout_rows = Some(cd.shape.numel() as u32);
            self.inner.borrow_mut().readout =
                Some(PortSource::Const { dtype: cd.dtype, shape: cd.shape, data: cd.bytes });
        }
        self.invalidate();
    }

    // -- stage attachments (overview §5.3) --
    #[track_caller]
    pub fn prologue(&self, body: impl Fn() + 'a) {
        self.inner.borrow_mut().prologue = Some(Box::new(body));
        self.invalidate();
    }
    #[track_caller]
    pub fn on_attn_proj(&self, body: impl Fn() + 'a) {
        self.inner.borrow_mut().on_attn_proj = Some(Box::new(body));
        self.invalidate();
    }
    #[track_caller]
    pub fn on_attn(&self, body: impl Fn() + 'a) {
        self.inner.borrow_mut().on_attn = Some(Box::new(body));
        self.invalidate();
    }
    #[track_caller]
    pub fn epilogue(&self, body: impl Fn() + 'a) {
        self.inner.borrow_mut().epilogue = Some(Box::new(body));
        self.invalidate();
    }

    fn invalidate(&self) {
        self.inner.borrow_mut().cached = None;
    }

    /// Read-out rows for `intrinsics::logits()`: an explicit `readout` count,
    /// else the number of `embed` lanes (rectangular indptr = `numel - 1`).
    fn rows(&self) -> u32 {
        let inner = self.inner.borrow();
        if let Some(r) = inner.readout_rows {
            return r.max(1);
        }
        match &inner.embed {
            Some((_, IndptrSpec::Const(cd))) => (cd.shape.numel() as u32).saturating_sub(1).max(1),
            _ => 1,
        }
    }

    /// Trace + lint + bind, memoized (overview P1.2).
    pub fn trace(&self) -> Result<Rc<TracedForward>, TraceErrors> {
        if let Some(cached) = &self.inner.borrow().cached {
            return cached.clone();
        }
        let result = self.assemble();
        self.inner.borrow_mut().cached = Some(result.clone());
        result
    }

    fn assemble(&self) -> Result<Rc<TracedForward>, TraceErrors> {
        let rows = self.rows();
        let (result, channels) = context::with_session(|| self.record(rows));
        let (stage_results, ports) = result;

        // The recorder interns channels in first-REFERENCE order (the order they
        // appear in the traced body, e.g. `embed(toks,…)` interns `toks` first).
        // But an inferlet DECLARES + indexes channels — seeds, host endpoints — in
        // DECLARATION order. Re-key the container to gid (declaration) order so the
        // two agree, remapping every channel reference (ChanTake / ChanRead /
        // ChanPut ops + descriptor `PortSource::Channel`). Without this, a
        // channel-0 [B,P] seed (e.g. beam `pages`) validates against whatever
        // channel happened to be referenced first (a [B] channel) → numel mismatch.
        let mut order: Vec<usize> = (0..channels.len()).collect();
        order.sort_by_key(|&i| channels[i].borrow().gid);
        let mut remap = vec![0u32; channels.len()];
        for (new_idx, &old_idx) in order.iter().enumerate() {
            remap[old_idx] = new_idx as u32;
        }
        let channels: Vec<ChannelRef> = order.iter().map(|&i| channels[i].clone()).collect();
        let stage_results: Vec<_> = stage_results
            .into_iter()
            .map(|mut r| {
                for op in &mut r.ops {
                    match op {
                        Op::ChanTake(c) | Op::ChanRead(c) => *c = remap[*c as usize],
                        Op::ChanPut { chan, .. } => *chan = remap[*chan as usize],
                        _ => {}
                    }
                }
                r
            })
            .collect();
        let ports: Vec<PortBinding> = ports
            .into_iter()
            .map(|mut p| {
                if let PortSource::Channel(ci) = &mut p.source {
                    *ci = remap[*ci as usize];
                }
                p
            })
            .collect();

        // Sink lint input (stage, sink).
        let sinks: Vec<(Stage, SinkCall)> = stage_results
            .iter()
            .flat_map(|r| r.sinks.iter().map(move |s| (r.stage, s.clone())))
            .collect();

        // Build echo's channel declarations with derived HostRole + seeded.
        let channel_decls: Vec<ChannelDecl> = channels
            .iter()
            .map(|c| {
                let st = c.borrow();
                let has_prog_put = !st.prog_puts.is_empty();
                let has_prog_consume = !st.prog_takes.is_empty() || !st.prog_reads.is_empty();
                let has_desc_use = !st.desc_takes.is_empty() || !st.desc_reads.is_empty();
                let has_host_put = !st.host_puts.is_empty();
                let host_consumes = !st.host_takes.is_empty() || !st.host_reads.is_empty();
                // A program-PRODUCED channel with NO program consumer (take/read),
                // NO descriptor binding, and NO host writer is a terminal OUTPUT the
                // host reads (e.g. beam `out`/`out_par`/`out_scr`): the guest's `take`
                // at runtime isn't visible at trace time, so infer Reader here.
                let is_terminal_output = has_prog_put
                    && !has_prog_consume
                    && !has_desc_use
                    && !has_host_put
                    && !st.seeded
                    && st.seed.is_none();
                let host_role = if has_host_put && !has_prog_put {
                    HostRole::Writer
                } else if host_consumes && (!st.prog_takes.is_empty() || has_prog_put) {
                    // A host-consumed, pass-produced/loop-carried channel.
                    HostRole::Reader
                } else if is_terminal_output {
                    HostRole::Reader
                } else {
                    HostRole::None
                };
                let seeded = st.seeded || (has_host_put && has_prog_put);
                ChannelDecl {
                    shape: st.shape,
                    dtype: ChanDType::Concrete(st.dtype),
                    capacity: st.capacity,
                    host_role,
                    seeded,
                }
            })
            .collect();

        // A host-Reader is the SPSC sole consumer (host-drained). The tracer's
        // `record_channel_put` auto-drain emits a device `ChanTake` for a channel
        // it sees as device-private at record time — but a terminal OUTPUT is
        // host-read, so that drain must be dropped (else `validate::bind` flags it
        // SecondConsumer: a stage consumes a host-Reader).
        let reader_ch: Vec<bool> = channel_decls
            .iter()
            .map(|d| d.host_role == HostRole::Reader)
            .collect();
        let stage_results: Vec<_> = stage_results
            .into_iter()
            .map(|mut r| {
                r.ops.retain(|op| match op {
                    Op::ChanTake(c) | Op::ChanRead(c) => !reader_ch[*c as usize],
                    _ => true,
                });
                r
            })
            .collect();

        let stages: Vec<StageProgram> = stage_results
            .into_iter()
            .map(|r| StageProgram { stage: r.stage, ops: r.ops })
            .collect();

        let mut ports = ports;
        ports.sort_by_key(|p| p.port as u8);

        let container = TraceContainer { externs: Vec::new(), names: Vec::new(), channels: channel_decls, ports, stages };

        // SDK span lints first (friendly, spans); then echo's authoritative bind.
        let mut errs: Vec<TraceError> = Vec::new();
        crate::lint::lint(&channels, &sinks, &mut errs);
        if !errs.is_empty() {
            return Err(TraceErrors(errs));
        }

        let names = channels.iter().map(|c| c.borrow().name.clone()).collect();
        match bind(container, model::profile()) {
            Ok(bound) => Ok(Rc::new(TracedForward { bound, channel_names: names })),
            Err(e) => Err(TraceErrors(alloc::vec![TraceError::Bind(e)])),
        }
    }

    /// Assemble the raw container WITHOUT lint/bind — for debugging emission.
    #[doc(hidden)]
    pub fn debug_container(&self) -> TraceContainer {
        let rows = self.rows();
        let (result, channels) = context::with_session(|| self.record(rows));
        let (stage_results, ports) = result;
        let channel_decls: Vec<ChannelDecl> = channels
            .iter()
            .map(|c| {
                let st = c.borrow();
                ChannelDecl {
                    shape: st.shape,
                    dtype: ChanDType::Concrete(st.dtype),
                    capacity: st.capacity,
                    host_role: HostRole::None,
                    seeded: st.seeded,
                }
            })
            .collect();
        let stages = stage_results.into_iter().map(|r| StageProgram { stage: r.stage, ops: r.ops }).collect();
        let mut ports = ports;
        ports.sort_by_key(|p| p.port as u8);
        TraceContainer { externs: Vec::new(), names: Vec::new(), channels: channel_decls, ports, stages }
    }

    /// Intern descriptor-port channels + trace each present stage (inside a session).
    fn record(&self, rows: u32) -> (Vec<context::StageResult>, Vec<PortBinding>) {
        let mut ports: Vec<PortBinding> = Vec::new();
        {
            let inner = self.inner.borrow();
            if let Some((toks, spec)) = &inner.embed {
                ports.push(PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(context::intern_channel(toks)),
                });
                match spec {
                    IndptrSpec::Const(cd) => ports.push(PortBinding {
                        port: Port::EmbedIndptr,
                        source: PortSource::Const { dtype: cd.dtype, shape: cd.shape, data: cd.bytes.clone() },
                    }),
                    IndptrSpec::Channel(ch) => ports.push(PortBinding {
                        port: Port::EmbedIndptr,
                        source: PortSource::Channel(context::intern_channel(ch)),
                    }),
                }
            }
            if let Some(pos) = &inner.positions {
                ports.push(PortBinding {
                    port: Port::Positions,
                    source: PortSource::Channel(context::intern_channel(pos)),
                });
            }
            if let Some(ws) = &inner.attn_ws {
                ports.push(PortBinding {
                    port: Port::KvLen,
                    source: PortSource::Channel(context::intern_channel(&ws.kv_len)),
                });
                if let Some(pages) = &ws.pages {
                    ports.push(PortBinding {
                        port: Port::Pages,
                        source: PortSource::Channel(context::intern_channel(pages)),
                    });
                }
                if let Some(cd) = &ws.page_indptr {
                    ports.push(PortBinding {
                        port: Port::PageIndptr,
                        source: PortSource::Const { dtype: cd.dtype, shape: cd.shape, data: cd.bytes.clone() },
                    });
                }
                if let Some(w) = &ws.w_slot {
                    ports.push(PortBinding {
                        port: Port::WSlot,
                        source: PortSource::Channel(context::intern_channel(w)),
                    });
                }
                if let Some(w) = &ws.w_off {
                    ports.push(PortBinding {
                        port: Port::WOff,
                        source: PortSource::Channel(context::intern_channel(w)),
                    });
                }
            }
            if let Some(m) = &inner.attn_mask {
                ports.push(PortBinding {
                    port: Port::AttnMask,
                    source: PortSource::Channel(context::intern_channel(m)),
                });
            }
            if let Some(src) = &inner.readout {
                ports.push(PortBinding { port: Port::Readout, source: src.clone() });
            }
        }

        let mut results = Vec::new();
        for stage in [Stage::Prologue, Stage::OnAttnProj, Stage::OnAttn, Stage::Epilogue] {
            let present = {
                let inner = self.inner.borrow();
                self.stage_slot(&inner, stage).is_some()
            };
            if !present {
                continue;
            }
            let res = context::trace_stage(stage, rows, || {
                let inner = self.inner.borrow();
                let f = self.stage_slot(&inner, stage).expect("stage present");
                f();
            });
            results.push(res);
        }
        (results, ports)
    }

    fn stage_slot<'b>(&self, inner: &'b ForwardInner<'a>, stage: Stage) -> Option<&'b StageClosure<'a>> {
        match stage {
            Stage::Prologue => inner.prologue.as_ref(),
            Stage::OnAttnProj => inner.on_attn_proj.as_ref(),
            Stage::OnAttn => inner.on_attn.as_ref(),
            Stage::Epilogue => inner.epilogue.as_ref(),
        }
    }
}

impl<'a> Default for ForwardPass<'a> {
    fn default() -> Self {
        ForwardPass::new()
    }
}

/// A traced + validated forward pass: echo's bound trace (container + C3 hash +
/// types + readiness + §7.1 classes) plus the SDK channel names.
#[derive(Debug)]
pub struct TracedForward {
    bound: BoundTrace,
    channel_names: Vec<String>,
}

impl TracedForward {
    pub fn container(&self) -> &TraceContainer {
        &self.bound.container
    }
    /// The validated, typed artifact (echo's [`BoundTrace`]).
    pub fn bound(&self) -> &BoundTrace {
        &self.bound
    }
    /// Program-set identity hash (FNV-1a over the canonical container bytes, C3).
    pub fn identity_hash(&self) -> u64 {
        self.bound.hash
    }
    /// The canonical trace-container bytes.
    pub fn encode(&self) -> Vec<u8> {
        self.bound.container.encode()
    }
    /// SDK channel names by dense index (debug).
    pub fn channel_names(&self) -> &[String] {
        &self.channel_names
    }
}

// ---------------------------------------------------------------------------
// Port argument traits
// ---------------------------------------------------------------------------

/// Anything usable as an `embed` indptr.
pub trait Indptr {
    fn resolve(self) -> IndptrSpec;
}
impl Indptr for Tensor {
    fn resolve(self) -> IndptrSpec {
        match self.as_const_data() {
            Some(cd) => IndptrSpec::Const(cd),
            None => panic!("embed indptr must be a Tensor::constant or a Channel"),
        }
    }
}
impl Indptr for &Channel {
    fn resolve(self) -> IndptrSpec {
        IndptrSpec::Channel(self.state().clone())
    }
}

/// The resolved trailing args of `attn_working_set`.
#[doc(hidden)]
pub struct AttnWsResolved {
    kv_len: ChannelRef,
    pages: Option<ChannelRef>,
    page_indptr: Option<ConstData>,
    w_slot: Option<ChannelRef>,
    w_off: Option<ChannelRef>,
}

/// The `attn_working_set` trailing-argument arities (overview §5.1). Rust has no
/// variadics, so the multi-arg forms are passed as tuples (deviation flagged).
pub trait AttnWsArgs {
    fn resolve(self) -> AttnWsResolved;
}
/// Sugar arity: `attn_working_set(&ws, &len)`.
impl AttnWsArgs for &Channel {
    fn resolve(self) -> AttnWsResolved {
        AttnWsResolved {
            kv_len: self.state().clone(),
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
            kv_len: klen.state().clone(),
            pages: Some(pages.state().clone()),
            page_indptr: page_indptr.as_const_data(),
            w_slot: Some(w_slot.state().clone()),
            w_off: Some(w_off.state().clone()),
        }
    }
}
