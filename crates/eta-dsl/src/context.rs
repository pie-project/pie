//! The trace-recording context: a thread-local session holding the stage
//! currently being traced plus the channel registry. Single-threaded by
//! construction (wasm inferlets; host tests run each trace on one thread).

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use eta_ir::op::{ChannelIndex, Op};
use eta_ir::types::{Dtype, Shape, ValueType};

use crate::error::{Span, TraceError};
use crate::value::ConstData;

/// Attachment stage — re-export of the IR's canonical [`Stage`](eta_ir::registry::Stage).
pub use eta_ir::registry::Stage;

/// A channel's mutable shared state (behind `Rc<RefCell<..>>`; a `Channel` is a
/// handle to it). Carries the trace decl, the per-instance seed flag, and the
/// endpoint claims the SPSC/host-role derivation + span lints read.
#[doc(hidden)]
#[derive(Debug)]
pub struct ChannelState {
    pub gid: u64,
    pub name: String,
    pub shape: Shape,
    pub dtype: Dtype,
    pub capacity: u32,
    /// Per-instance seed value (from `Channel::from` / a pre-submit host `put`);
    /// its bytes are instance data, never in the container.
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

/// A sink call recorded in a stage (for the T11 span pre-lint; the IR's validator
/// is the authoritative gate).
#[derive(Clone, Debug)]
pub(crate) struct SinkCall {
    pub name: String,
    pub span: Span,
    pub scope: eta_ir::registry::SinkScope,
}

/// The stage currently being traced.
pub(crate) struct Recorder {
    pub stage: Stage,
    /// Read-out rows for `intrinsics::logits()` shape.
    pub rows: u32,
    pub ops: Vec<Op>,
    /// Light per-value types (author ergonomics; the IR's `infer` is authoritative).
    pub types: Vec<ValueType>,
    pub sinks: Vec<SinkCall>,
}

impl Recorder {
    fn new(stage: Stage, rows: u32) -> Self {
        Recorder {
            stage,
            rows,
            ops: Vec::new(),
            types: Vec::new(),
            sinks: Vec::new(),
        }
    }

    /// Records `op` and returns the id of its first result. `result_tys` is
    /// checked against `op.result_count()` (a real assert, not
    /// debug-only, since debug-asserts are compiled out of release guest
    /// traces) — a mismatch would silently shift every later value id.
    fn push(&mut self, op: Op, result_tys: &[ValueType]) -> u32 {
        let base = self.types.len() as u32;
        assert_eq!(
            op.result_count() as usize,
            result_tys.len(),
            "result arity mismatch for {op:?}: recording {} types against \
             {} results would shift every later value id",
            result_tys.len(),
            op.result_count()
        );
        self.types.extend_from_slice(result_tys);
        self.ops.push(op);
        base
    }
}

/// The trace session accumulating one forward's channels + stage programs.
pub(crate) struct Session {
    chan_by_gid: alloc::collections::BTreeMap<u64, ChannelIndex>,
    pub channels: Vec<ChannelRef>,
    pub current: Option<Recorder>,
    /// Second-party names, in first-use order. The container's name table is
    /// SHARED across stages (a `NameIndex` in one stage's op means the same
    /// thing in another), so it is interned at session scope, not stage scope.
    pub names: Vec<String>,
    /// Authoring mistakes found while recording. Collected instead of
    /// panicked so one `build()` reports all of them; see
    /// [`TraceError::Authoring`](crate::error::TraceError::Authoring).
    pub errors: Vec<TraceError>,
}

impl Session {
    fn new() -> Self {
        Session {
            chan_by_gid: alloc::collections::BTreeMap::new(),
            channels: Vec::new(),
            current: None,
            names: Vec::new(),
            errors: Vec::new(),
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
    /// Authoring mistakes recorded before a session opened — from a `Channel`
    /// constructor, which runs ahead of the trace body. Drained by
    /// [`with_session`] into the session it belongs to.
    static PENDING: RefCell<Vec<TraceError>> = const { RefCell::new(Vec::new()) };
    /// gid -> channel state. The guest-facing `Channel` is a `Copy` token
    /// holding only its gid; every op resolves shared state through this
    /// registry, which owns entries until [`release_channel_state`] is
    /// called (unbounded retention otherwise — acceptable since channels are
    /// declared once at model setup, not per request).
    static CHANNELS_BY_GID: RefCell<alloc::collections::BTreeMap<u64, ChannelRef>> =
        const { RefCell::new(alloc::collections::BTreeMap::new()) };
}

pub(crate) fn register_channel_state(gid: u64, state: ChannelRef) {
    CHANNELS_BY_GID.with_borrow_mut(|map| {
        map.insert(gid, state);
    });
}

pub(crate) fn channel_state_by_gid(gid: u64) -> Option<ChannelRef> {
    CHANNELS_BY_GID.with_borrow(|map| map.get(&gid).cloned())
}

pub(crate) fn release_channel_state(gid: u64) -> bool {
    CHANNELS_BY_GID.with_borrow_mut(|map| map.remove(&gid).is_some())
}

/// How many channels the registry is holding. Surfaced as
/// [`Channel::registered_count`](crate::channel::Channel::registered_count).
pub(crate) fn registered_channel_count() -> usize {
    CHANNELS_BY_GID.with_borrow(|map| map.len())
}

/// Are we currently tracing a stage closure?
pub(crate) fn is_tracing() -> bool {
    SESSION.with_borrow(|s| s.as_ref().map(|s| s.current.is_some()).unwrap_or(false))
}

pub(crate) fn intern_channel(ch: &ChannelRef) -> ChannelIndex {
    SESSION.with_borrow_mut(|s| s.as_mut().expect("session active").intern(ch))
}

/// Run `f` with a fresh session active; return `f`'s result + the interned channels.
pub(crate) fn with_session<R>(
    f: impl FnOnce() -> R,
) -> (R, Vec<ChannelRef>, Vec<String>, Vec<TraceError>) {
    let carried = PENDING.with_borrow_mut(core::mem::take);
    SESSION.with_borrow_mut(|s| {
        debug_assert!(s.is_none(), "nested trace session");
        let mut session = Session::new();
        session.errors = carried;
        *s = Some(session);
    });
    let r = f();
    let (channels, names, errors) = SESSION.with_borrow_mut(|s| {
        let session = s.take().expect("session present");
        (session.channels, session.names, session.errors)
    });
    (r, channels, names, errors)
}

/// Record an authoring mistake for the next [`Builder::build`] to report.
/// Works with no session active (channels are declared before the trace
/// runs): errors land in [`PENDING`] and [`with_session`] adopts them.
///
/// [`Builder::build`]: crate::builder::Builder::build
pub(crate) fn record_error(detail: String, span: Span) {
    let error = TraceError::Authoring { detail, span };
    SESSION.with_borrow_mut(|s| match s.as_mut() {
        Some(session) => session.errors.push(error),
        None => PENDING.with_borrow_mut(|p| p.push(error)),
    });
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
        let rec = s
            .as_mut()
            .expect("session active")
            .current
            .take()
            .expect("stage recorder");
        StageResult {
            stage: rec.stage,
            ops: rec.ops,
            sinks: rec.sinks,
        }
    })
}

pub(crate) struct StageResult {
    pub stage: Stage,
    pub ops: Vec<Op>,
    pub sinks: Vec<SinkCall>,
}

// ---------------------------------------------------------------------------
// Recording primitives called by Tensor / Channel / intrinsics.
// ---------------------------------------------------------------------------

pub(crate) fn current_rows() -> u32 {
    SESSION.with_borrow(|s| {
        s.as_ref()
            .and_then(|s| s.current.as_ref().map(|r| r.rows))
            .unwrap_or(1)
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
        let op = if consume {
            Op::ChanTake(dense)
        } else {
            Op::ChanRead(dense)
        };
        let id = rec.push(op, &[elem]);
        (id, elem)
    })
}

/// Record a channel `put` inside a stage (the value id must already match the
/// channel's shape+dtype — the caller reshapes as needed).
///
/// A channel bound to a peeked descriptor port ([`eta_ir::registry::Port::consumes`]
/// false — geometry and masks) is drained first: the descriptor phase reads
/// its front without draining, so a bare re-put would grow the ring forever.
/// Safe even if the guest already took explicitly, since take is a per-pass
/// flag and not a counter.
pub(crate) fn record_channel_put(ch: &ChannelRef, value: u32, span: Span) {
    SESSION.with_borrow_mut(|s| {
        let sess = s.as_mut().expect("session active");
        let dense = sess.intern(ch);
        let stage = sess.current.as_ref().expect("stage active").stage;
        let (drain, elem) = {
            let st = ch.borrow();
            let peeked_port = !st.desc_reads.is_empty() && st.desc_takes.is_empty();
            (peeked_port && st.prog_takes.is_empty(), st.elem_ty())
        };
        if drain {
            ch.borrow_mut().prog_takes.push((stage, span));
            let rec = sess.current.as_mut().expect("stage active");
            rec.push(Op::ChanTake(dense), &[elem]);
        }
        {
            ch.borrow_mut().prog_puts.push((stage, span));
        }
        let rec = sess.current.as_mut().expect("stage active");
        rec.push(Op::ChanPut { chan: dense, value }, &[]);
    })
}

/// Intern a second-party name into the session's shared name table, returning
/// its `NameIndex`. First use wins, so the table is deterministic in trace
/// order and the container bytes stay byte-stable across runs.
pub(crate) fn intern_name(name: &str) -> u16 {
    SESSION.with_borrow_mut(|s| {
        let sess = s.as_mut().expect("name interned outside a session");
        if let Some(index) = sess.names.iter().position(|n| n == name) {
            return index as u16;
        }
        sess.names.push(String::from(name));
        (sess.names.len() - 1) as u16
    })
}

pub(crate) fn record_sink(name: String, span: Span, scope: eta_ir::registry::SinkScope) {
    SESSION.with_borrow_mut(|s| {
        s.as_mut()
            .and_then(|s| s.current.as_mut())
            .expect("sink outside a traced stage")
            .sinks
            .push(SinkCall { name, span, scope });
    })
}
