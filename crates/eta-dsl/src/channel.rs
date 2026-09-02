//! `Channel`: GPU-resident bounded queue of cells with full/empty bits.
//! Inside a traced stage, `take`/`read`/`put` record the IR's
//! `ChanTake`/`ChanRead`/`ChanPut` ops; on the host they take the async path.

use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::sync::atomic::{AtomicU64, Ordering};

use eta_ir::types::{Dtype, Shape, ValueType};

use crate::context::{self, ChannelRef, ChannelState};
use crate::error::Span;
use crate::value::{AsTensor, ConstData, IntoConst, IntoShape, Tensor, reshape_id_to};

static NEXT_GID: AtomicU64 = AtomicU64::new(1);

/// A handle to a channel's shared state. Cheap to clone; captured
/// by both host code and the stage closures that read/write it.
#[derive(Clone)]
pub struct Channel {
    state: ChannelRef,
}

impl Channel {
    /// `Channel::new([shape], dtype)` — a capacity-1 channel.
    pub fn new(shape: impl IntoShape, dtype: Dtype) -> Channel {
        Channel::build(shape.into_shape(), dtype, 1, None)
    }

    /// `Channel::from(v)` — sugar for `new` + `put`: a channel seeded full with
    /// `v`. `v` may be per-instance *data* (a request seed); the
    /// seed is instance state, never in the container.
    pub fn from(v: impl IntoConst) -> Channel {
        let data = v.into_const();
        Channel::build(data.shape, data.dtype, 1, Some(data))
    }

    /// `Channel::from_shaped([shape], v)` — like `from`, but reinterprets the
    /// flat seed data `v` under an explicit multi-dim `shape` (element counts
    /// must match); use for a concrete multi-dim seed a downstream op types
    /// against as rank-2+.
    #[track_caller]
    pub fn from_shaped(shape: impl IntoShape, v: impl IntoConst) -> Channel {
        let mut data = v.into_const();
        let shape = shape.into_shape();
        if shape.numel() == data.shape.numel() {
            data.shape = shape;
        } else {
            // Element counts differ: keep the seed's own shape instead.
            context::record_error(
                alloc::format!(
                    "Channel::from_shaped: {:?} holds {} elements but the seed holds {}",
                    shape,
                    shape.numel(),
                    data.shape.numel()
                ),
                Span::here(),
            );
        }
        Channel::build(data.shape, data.dtype, 1, Some(data))
    }

    /// A seeded channel whose initial value is per-instance data supplied at
    /// instantiation rather than a trace constant.
    pub fn seeded(shape: impl IntoShape, dtype: Dtype) -> Channel {
        let ch = Channel::build(shape.into_shape(), dtype, 1, None);
        ch.state.borrow_mut().seeded = true;
        ch
    }

    /// Resolve a channel handle from its global id. A guest-facing Copy
    /// token stores only the gid; every op resolves through here.
    pub fn by_gid(gid: u64) -> Option<Channel> {
        context::channel_state_by_gid(gid).map(|state| Channel { state })
    }

    /// Drop this channel's state from the registry, returning whether an
    /// entry was there to drop. Since the guest-facing handle is a `Copy`
    /// token, the registry has no `Drop` to learn from, so a frontend that
    /// creates channels dynamically must call this explicitly. Do not call
    /// for a channel still reachable from a stage closure or a pending host
    /// `put`/`take`.
    pub fn release(gid: u64) -> bool {
        context::release_channel_state(gid)
    }

    /// How many channels the registry currently holds — lets a frontend
    /// watch its own release discipline instead of discovering leaks as
    /// memory growth.
    pub fn registered_count() -> usize {
        context::registered_channel_count()
    }

    /// Whether the channel starts full with a seed value (built via
    /// [`Channel::from`] or [`Channel::seeded`]), so its first `take` needs
    /// no producer.
    pub fn is_seeded(&self) -> bool {
        self.state.borrow().seeded
    }

    fn build(shape: Shape, dtype: Dtype, capacity: u32, seed: Option<ConstData>) -> Channel {
        let gid = NEXT_GID.fetch_add(1, Ordering::Relaxed);
        let seeded = seed.is_some();
        let state = Rc::new(RefCell::new(ChannelState {
            gid,
            name: format!("ch{gid}"),
            shape,
            dtype,
            capacity,
            seed,
            seeded,
            prog_puts: Vec::new(),
            prog_takes: Vec::new(),
            prog_reads: Vec::new(),
            host_puts: Vec::new(),
            host_takes: Vec::new(),
            host_reads: Vec::new(),
            desc_takes: Vec::new(),
            desc_reads: Vec::new(),
        }));
        context::register_channel_state(gid, state.clone());
        Channel { state }
    }

    /// Widen the ring to `n` cells (deeper run-ahead).
    pub fn capacity(self, n: u32) -> Channel {
        self.state.borrow_mut().capacity = n;
        self
    }

    /// Give the channel a name (improves trace-error messages).
    pub fn named(self, name: &str) -> Channel {
        self.state.borrow_mut().name = String::from(name);
        self
    }

    pub(crate) fn state(&self) -> &ChannelRef {
        &self.state
    }
    /// The scalar [`Dtype`] of the channel's cells.
    pub fn dtype(&self) -> Dtype {
        self.state.borrow().dtype
    }

    /// The channel's name, as set by [`named`](Self::named) — `chN` if it was
    /// never named. Frontends use it to label host-readback errors.
    pub fn name(&self) -> String {
        self.state.borrow().name.clone()
    }
    /// The [`Shape`] of one cell (a single queue slot's tensor).
    pub fn shape(&self) -> Shape {
        self.state.borrow().shape
    }
    /// The channel's global identity (declaration order); matched by
    /// [`Traced::channel_order`](crate::Traced::channel_order) and the
    /// `inferlet` WIT bridge's handle list.
    pub fn gid(&self) -> u64 {
        self.state.borrow().gid
    }

    /// Record a host-writer/seed endpoint span on the trace side without
    /// staging data (the WIT bridge stages bytes separately and calls this
    /// to keep host-role derivation correct).
    #[track_caller]
    pub fn note_host_put(&self) {
        self.state.borrow_mut().host_puts.push(Span::here());
    }

    /// Record a host-consumer endpoint span for a `take` — the readback
    /// counterpart of [`note_host_put`](Self::note_host_put). The bytes cross
    /// the engine boundary, so there is no in-program value to hand back.
    #[track_caller]
    pub fn note_host_take(&self) {
        self.state.borrow_mut().host_takes.push(Span::here());
    }

    /// Record a host-consumer endpoint span for a `read` (a peek). Same as
    /// [`note_host_take`](Self::note_host_take) otherwise.
    #[track_caller]
    pub fn note_host_read(&self) {
        self.state.borrow_mut().host_reads.push(Span::here());
    }

    /// Record a descriptor endpoint claim (take vs read per the port's
    /// consumption discipline) without binding a port. The bridge claims
    /// eagerly at pass construction, then binds with
    /// [`crate::builder::Builder::bind_port_recorded`] so it isn't double-counted.
    #[track_caller]
    pub fn note_desc_claim(&self, consumes: bool) {
        let span = Span::here();
        let mut st = self.state.borrow_mut();
        if consumes {
            st.desc_takes.push(span);
        } else {
            st.desc_reads.push(span);
        }
    }

    /// `take()` — full ⇒ value + empty; empty ⇒ block. The in-program taken
    /// [`Tensor`], for use inside a stage closure. A host channel has no
    /// in-program value: use [`note_host_take`](Self::note_host_take) and read
    /// the bytes across the engine instead.
    #[track_caller]
    pub fn take(&self) -> Tensor {
        let span = Span::here();
        if context::is_tracing() {
            let (id, ty) = context::record_channel_read(&self.state, true, span);
            Tensor::node(id, ty)
        } else {
            host_take_poison(&self.state)
        }
    }

    /// `read()` — full ⇒ copy, stays full; empty ⇒ block. A peek (does not claim
    /// the consumer endpoint). Same in-program/host rule as [`take`](Self::take).
    #[track_caller]
    pub fn read(&self) -> Tensor {
        let span = Span::here();
        if context::is_tracing() {
            let (id, ty) = context::record_channel_read(&self.state, false, span);
            Tensor::node(id, ty)
        } else {
            host_take_poison(&self.state)
        }
    }

    /// `put(v)` — empty ⇒ fill + full; full ⇒ block (back-pressure). In-program
    /// `v` is a `Tensor` (reshaped to fit the cell); on the host `v` is data.
    /// On a channel bound to a peeked descriptor port, drains the stale value
    /// first (an explicit `take` in the same trace is honoured, not repeated).
    #[track_caller]
    pub fn put(&self, v: impl IntoPut) -> Put {
        let span = Span::here();
        match v.into_put() {
            PutValue::Tensor(t) => {
                debug_assert!(context::is_tracing(), "put(Tensor) outside a traced stage");
                let (id, ty) = t.to_arg().materialize();
                let chan_shape = self.state.borrow().shape;
                let fitted = reshape_id_to(id, ty, chan_shape);
                context::record_channel_put(&self.state, fitted, span);
                Put::done()
            }
            PutValue::Data(data) => {
                // seed vs host-Writer is decided at assembly; values aren't needed here.
                let mut st = self.state.borrow_mut();
                st.host_puts.push(span);
                let _ = data;
                Put::done()
            }
        }
    }
}

/// Stand-in for a host take asked to act as an in-program value, typed as the
/// channel it came from so the rest of the trace still checks.
#[track_caller]
fn host_take_poison(chan: &ChannelRef) -> Tensor {
    let st = chan.borrow();
    crate::value::poison_const(
        alloc::format!(
            "channel {} is a host channel: its take crosses the engine boundary and has no \
             in-program value",
            st.name
        ),
        ValueType::new(st.shape, st.dtype),
    )
}

/// The (fire-and-forget) result of a `put`. Host puts coalesce before the next
/// submit; the handle exists so back-pressure can be awaited.
pub struct Put(());
impl Put {
    fn done() -> Put {
        Put(())
    }
}

// ---------------------------------------------------------------------------
// put value coercion
// ---------------------------------------------------------------------------

/// A value handed to `Channel::put`.
pub enum PutValue {
    /// An in-program value: a [`Tensor`] recorded into the trace.
    Tensor(Tensor),
    /// Host bytes: a [`ConstData`] seed or host-writer payload staged across
    /// the engine.
    Data(ConstData),
}

/// Anything puttable: a `Tensor` (in-program) or host data (arrays / vecs / scalars).
pub trait IntoPut {
    /// Coerces `self` into a [`PutValue`] — an in-program [`Tensor`] or host
    /// [`ConstData`].
    fn into_put(self) -> PutValue;
}

impl IntoPut for Tensor {
    fn into_put(self) -> PutValue {
        PutValue::Tensor(self)
    }
}
impl IntoPut for &Tensor {
    fn into_put(self) -> PutValue {
        PutValue::Tensor(self.clone())
    }
}

macro_rules! into_put_data {
    ($($t:ty),*) => { $(
        impl IntoPut for $t {
            fn into_put(self) -> PutValue { PutValue::Data(self.into_const()) }
        }
    )* };
}
into_put_data!(i32, u32, f32, bool);
into_put_data!(Vec<i32>, Vec<u32>, Vec<f32>, Vec<bool>);
impl<const N: usize> IntoPut for [i32; N] {
    fn into_put(self) -> PutValue {
        PutValue::Data(self.into_const())
    }
}
impl<const N: usize> IntoPut for [u32; N] {
    fn into_put(self) -> PutValue {
        PutValue::Data(self.into_const())
    }
}
impl<const N: usize> IntoPut for [f32; N] {
    fn into_put(self) -> PutValue {
        PutValue::Data(self.into_const())
    }
}
impl<const N: usize> IntoPut for [bool; N] {
    fn into_put(self) -> PutValue {
        PutValue::Data(self.into_const())
    }
}

