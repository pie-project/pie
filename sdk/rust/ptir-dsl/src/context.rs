//! The trace-recording context: a thread-local **session** holding the stage
//! currently being traced plus the channel registry. Channel/Tensor methods
//! consult it — inside a traced stage closure they record echo's canonical
//! [`ptir::op::Op`](pie_ptir::op::Op); on the host they take the
//! async path.
//!
//! Single-threaded by construction (wasm inferlets; host tests run each trace on
//! one thread). `std` provides the `thread_local!`.

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use pie_ptir::op::{ChannelIndex, Op};
use pie_ptir::types::{DType, Shape, ValueType};

use crate::error::Span;
use crate::value::ConstData;

/// Attachment stage — re-export of echo's canonical [`Stage`](pie_ptir::registry::Stage).
pub use pie_ptir::registry::Stage;

/// A channel's mutable shared state (behind `Rc<RefCell<..>>`; a `Channel` is a
/// handle to it). Carries the trace decl, the per-instance seed flag, and the
/// endpoint claims the SPSC/host-role derivation + span lints read.
#[doc(hidden)]
#[derive(Debug)]
pub struct ChannelState {
    pub gid: u64,
    pub name: String,
    pub shape: Shape,
    pub dtype: DType,
    pub capacity: u32,
    /// Per-instance seed value (from `Channel::from` / a pre-submit host `put`);
    /// its bytes are instance data, never in the container (D2).
    pub seed: Option<ConstData>,
    pub seeded: bool,

    // -- endpoint claims (host-role derivation + span lints) --
    pub prog_puts: Vec<(Stage, Span)>,
    pub prog_takes: Vec<(Stage, Span)>,
    pub prog_reads: Vec<(Stage, Span)>,
    pub host_puts: Vec<Span>,
    pub host_takes: Vec<Span>,
    pub host_reads: Vec<Span>,
    /// Descriptor-port claims: `embed`/`positions`/`w_slot`/`w_off` consume
    /// (take), geometry/masks peek (read).
    pub desc_takes: Vec<Span>,
    pub desc_reads: Vec<Span>,
}

impl ChannelState {
    pub fn elem_ty(&self) -> ValueType {
        ValueType::new(self.shape, self.dtype)
    }
}

pub type ChannelRef = Rc<RefCell<ChannelState>>;

/// A sink call recorded in a stage (for the T11 span pre-lint; echo's validator
/// is the authoritative gate).
#[derive(Clone, Debug)]
pub(crate) struct SinkCall {
    pub name: String,
    pub span: Span,
    pub scope: pie_ptir::registry::SinkScope,
}

/// The stage currently being traced.
pub(crate) struct Recorder {
    pub stage: Stage,
    /// Read-out rows for `intrinsics::logits()` shape.
    pub rows: u32,
    pub ops: Vec<Op>,
    /// Light per-value types (author ergonomics; echo's `infer` is authoritative).
    pub types: Vec<ValueType>,
    pub next_id: u32,
    pub sinks: Vec<SinkCall>,
    /// Dense ids consumed (taken) so far in this stage — drives the auto-drain
    /// of pure-derivative device channels (put-without-take on a capacity-1
    /// full cell; overview §6.2 elides the drain, the trace can't).
    pub consumed: alloc::collections::BTreeSet<ChannelIndex>,
    /// Positions (op indices) of the SYNTHESIZED drain `ChanTake`s this stage —
    /// the only ops the builder may drop when a channel's derived role turns
    /// out host-Reader (their result values are never exposed to the author).
    pub drains: Vec<usize>,
}

impl Recorder {
    fn new(stage: Stage, rows: u32) -> Self {
        Recorder {
            stage,
            rows,
            ops: Vec::new(),
            types: Vec::new(),
            next_id: 0,
            sinks: Vec::new(),
            consumed: alloc::collections::BTreeSet::new(),
            drains: Vec::new(),
        }
    }

    fn push(&mut self, op: Op, result_tys: &[ValueType]) -> u32 {
        let base = self.next_id;
        debug_assert_eq!(
            op.result_count(),
            result_tys.len() as u32,
            "result arity mismatch: {op:?}"
        );
        for ty in result_tys {
            self.types.push(*ty);
        }
        self.next_id += result_tys.len() as u32;
        self.ops.push(op);
        base
    }
}

/// The trace session accumulating one forward's channels + stage programs.
pub(crate) struct Session {
    chan_by_gid: alloc::collections::BTreeMap<u64, ChannelIndex>,
    pub channels: Vec<ChannelRef>,
    pub current: Option<Recorder>,
}

impl Session {
    fn new() -> Self {
        Session {
            chan_by_gid: alloc::collections::BTreeMap::new(),
            channels: Vec::new(),
            current: None,
        }
    }

    fn intern(&mut self, ch: &ChannelRef) -> ChannelIndex {
        let gid = ch.borrow().gid;
        if let Some(&id) = self.chan_by_gid.get(&gid) {
            return id;
        }
        let id = self.channels.len() as ChannelIndex;
        self.chan_by_gid.insert(gid, id);
        self.channels.push(ch.clone());
        id
    }
}

thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

/// Are we currently tracing a stage closure?
pub(crate) fn is_tracing() -> bool {
    SESSION.with_borrow(|s| s.as_ref().map(|s| s.current.is_some()).unwrap_or(false))
}

pub(crate) fn intern_channel(ch: &ChannelRef) -> ChannelIndex {
    SESSION.with_borrow_mut(|s| s.as_mut().expect("session active").intern(ch))
}

/// Run `f` with a fresh session active; return `f`'s result + the interned channels.
pub(crate) fn with_session<R>(f: impl FnOnce() -> R) -> (R, Vec<ChannelRef>) {
    SESSION.with_borrow_mut(|s| {
        debug_assert!(s.is_none(), "nested trace session");
        *s = Some(Session::new());
    });
    let r = f();
    let channels = SESSION.with_borrow_mut(|s| s.take().expect("session present").channels);
    (r, channels)
}

/// Trace one stage closure into a completed [`StageResult`]. `rows` = the pass's
/// read-out row count.
pub(crate) fn trace_stage(stage: Stage, rows: u32, body: impl FnOnce()) -> StageResult {
    SESSION.with_borrow_mut(|s| {
        let sess = s.as_mut().expect("session active");
        debug_assert!(sess.current.is_none(), "nested stage");
        sess.current = Some(Recorder::new(stage, rows));
    });
    body();
    SESSION.with_borrow_mut(|s| {
        let rec = s.as_mut().expect("session active").current.take().expect("stage recorder");
        StageResult { stage: rec.stage, ops: rec.ops, sinks: rec.sinks, drains: rec.drains }
    })
}

pub(crate) struct StageResult {
    pub stage: Stage,
    pub ops: Vec<Op>,
    pub sinks: Vec<SinkCall>,
    /// Synthesized auto-drain positions in `ops` (see [`Recorder::drains`]).
    pub drains: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Recording primitives called by Tensor / Channel / intrinsics.
// ---------------------------------------------------------------------------

pub(crate) fn current_rows() -> u32 {
    SESSION.with_borrow(|s| {
        s.as_ref().and_then(|s| s.current.as_ref().map(|r| r.rows)).unwrap_or(1)
    })
}

/// Emit an op into the current stage; returns its first result id.
pub(crate) fn emit(op: Op, result_tys: &[ValueType]) -> u32 {
    SESSION.with_borrow_mut(|s| {
        s.as_mut()
            .and_then(|s| s.current.as_mut())
            .expect("emit outside a traced stage")
            .push(op, result_tys)
    })
}

/// The recorded type of an already-defined value id (light inference). Retained
/// for shape-derived ops; harmless if a given trace doesn't need it.
#[allow(dead_code)]
pub(crate) fn type_of(id: u32) -> ValueType {
    SESSION.with_borrow(|s| {
        s.as_ref()
            .and_then(|s| s.current.as_ref())
            .and_then(|r| r.types.get(id as usize).copied())
            .expect("value id has no recorded type")
    })
}

/// Record a channel `take`/`read` inside a stage: intern, push the op, register
/// the endpoint claim; return the produced value id + type.
pub(crate) fn record_channel_read(ch: &ChannelRef, consume: bool, span: Span) -> (u32, ValueType) {
    SESSION.with_borrow_mut(|s| {
        let sess = s.as_mut().expect("session active");
        let dense = sess.intern(ch);
        let elem = ch.borrow().elem_ty();
        {
            let stage = sess.current.as_ref().expect("stage active").stage;
            let mut st = ch.borrow_mut();
            if consume {
                st.prog_takes.push((stage, span));
            } else {
                st.prog_reads.push((stage, span));
            }
        }
        let rec = sess.current.as_mut().expect("stage active");
        let op = if consume { Op::ChanTake(dense) } else { Op::ChanRead(dense) };
        if consume {
            rec.consumed.insert(dense);
        }
        let id = rec.push(op, &[elem]);
        (id, elem)
    })
}

/// Record a channel `put` inside a stage (the value id must already match the
/// channel's shape+dtype — the caller reshapes as needed). Auto-injects a drain
/// `take` first when a **pure-derivative device channel** is put without being
/// consumed this pass (else the capacity-1 full cell back-pressures forever;
/// overview §6.2's klen/kvm — the doc elides the drain, the trace can't).
pub(crate) fn record_channel_put(ch: &ChannelRef, value: u32, span: Span) {
    SESSION.with_borrow_mut(|s| {
        let sess = s.as_mut().expect("session active");
        let dense = sess.intern(ch);
        let stage = sess.current.as_ref().expect("stage active").stage;

        // Auto-drain: device-private (no host role, not descriptor-consumed) and
        // not yet taken this pass ⇒ the committed cell is full at entry, so the
        // put needs a preceding drain.
        let needs_drain = {
            let st = ch.borrow();
            let device_private = st.host_puts.is_empty()
                && st.host_takes.is_empty()
                && st.host_reads.is_empty()
                && st.desc_takes.is_empty();
            device_private && !sess.current.as_ref().unwrap().consumed.contains(&dense)
        };
        if needs_drain {
            let elem = ch.borrow().elem_ty();
            let rec = sess.current.as_mut().expect("stage active");
            rec.consumed.insert(dense);
            rec.drains.push(rec.ops.len());
            rec.push(Op::ChanTake(dense), &[elem]);
        }

        {
            ch.borrow_mut().prog_puts.push((stage, span));
        }
        let rec = sess.current.as_mut().expect("stage active");
        rec.push(Op::ChanPut { chan: dense, value }, &[]);
    })
}

pub(crate) fn record_sink(
    name: String,
    span: Span,
    scope: pie_ptir::registry::SinkScope,
) {
    SESSION.with_borrow_mut(|s| {
        s.as_mut()
            .and_then(|s| s.current.as_mut())
            .expect("sink outside a traced stage")
            .sinks
            .push(SinkCall { name, span, scope });
    })
}
