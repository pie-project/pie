//! `Channel` — GPU-resident ordered memory: a bounded queue of
//! cells with full/empty bits. Inside a traced stage, `take`/`read`/`put` record
//! the IR's `ChanTake`/`ChanRead`/`ChanPut` ops; on the host they take the async
//! path.

use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::sync::atomic::{AtomicU64, Ordering};

use pie_ir::types::{DType, Shape, ValueType};

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
    pub fn new(shape: impl IntoShape, dtype: DType) -> Channel {
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
    /// flat seed data `v` with the explicit multi-dim `shape` (element counts
    /// must match). `IntoConst` only produces flat 1-D seeds, so use this for a
    /// concrete multi-dim seed (e.g. a `[B, POOL]` bool attention mask) that
    /// downstream ops (`gather`/`or`) type against as rank-2.
    #[track_caller]
    pub fn from_shaped(shape: impl IntoShape, v: impl IntoConst) -> Channel {
        let mut data = v.into_const();
        let shape = shape.into_shape();
        if shape.numel() == data.shape.numel() {
            data.shape = shape;
        } else {
            // Keep the seed's own shape. Reinterpreting under a shape of a
            // different extent would either read past the seed or leave a tail
            // of it unaddressable, and the resulting channel would then type
            // downstream ops against a length its data cannot supply.
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

    /// A seeded channel of a given shape (`Channel::from` where the initial
    /// value is per-instance data supplied at instantiation, D2). Use for
    /// device loop-carried multi-dim channels (`pages [B,P]`, `kvm [B, P*page]`)
    /// whose seed value is not a trace constant.
    pub fn seeded(shape: impl IntoShape, dtype: DType) -> Channel {
        let ch = Channel::build(shape.into_shape(), dtype, 1, None);
        ch.state.borrow_mut().seeded = true;
        ch
    }

    /// Resolve a channel handle from its global id (the registry is
    /// thread-local, like the trace session). A guest-facing Copy token
    /// stores only the gid; every op resolves through here.
    pub fn by_gid(gid: u64) -> Option<Channel> {
        context::channel_state_by_gid(gid).map(|state| Channel { state })
    }

    /// Drop this channel's state from the registry, returning whether an
    /// entry was there to drop.
    ///
    /// The counterpart to construction. Because the guest-facing handle is a
    /// `Copy` token holding only a gid, the registry cannot learn from a
    /// `Drop` that a channel is finished — so a frontend that creates
    /// channels dynamically must say so here, at the point where it knows no
    /// token survives. Everything a released gid names is gone: a later
    /// [`Channel::by_gid`] returns `None`, and the SDK's resolve — which
    /// `expect`s a hit, correctly, since an unregistered token is a frontend
    /// bug — would panic.
    ///
    /// Do not call this for a channel still reachable from a stage closure or
    /// a pending host `put`/`take`. Releasing mid-trace is the one way to
    /// turn this registry's design into the failure a `Weak` would have made
    /// routine.
    ///
    /// Inferlets as written today declare their channels once at setup and
    /// never need this; it exists so that a guest with per-request channels
    /// has a bounded option, and so that the retention is a stated policy
    /// rather than an accident.
    pub fn release(gid: u64) -> bool {
        context::release_channel_state(gid)
    }

    /// How many channels the registry currently holds.
    ///
    /// The retention documented on [`Channel::release`] is otherwise
    /// invisible: nothing observable changes as the map grows. A frontend
    /// that creates channels dynamically can watch this to check its own
    /// release discipline instead of discovering the growth as memory.
    pub fn registered_count() -> usize {
        context::registered_channel_count()
    }

    /// Returns whether the channel starts full with a seed value, so its
    /// first `take` needs no producer.
    ///
    /// True for channels built by [`Channel::from`] or [`Channel::seeded`];
    /// the seed is per-instance state supplied at instantiation, never part
    /// of the container.
    pub fn is_seeded(&self) -> bool {
        self.state.borrow().seeded
    }

    fn build(shape: Shape, dtype: DType, capacity: u32, seed: Option<ConstData>) -> Channel {
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
    /// The scalar [`DType`] of the channel's cells.
    pub fn dtype(&self) -> DType {
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
    /// The channel's global identity (declaration order). The
    /// builder↔bridge contract: [`Traced::channel_order`](crate::Traced::channel_order)
    /// lists these gids in dense declaration order, and the `inferlet` WIT bridge
    /// orders its `forward-pass.program` handle list to match.
    pub fn gid(&self) -> u64 {
        self.state.borrow().gid
    }

    /// Record a host-writer/seed endpoint span on the trace side without staging
    /// any data (the `inferlet` WIT bridge stages the bytes on the WIT channel
    /// and calls this to keep the trace-side host-role derivation correct).
    #[track_caller]
    pub fn note_host_put(&self) {
        self.state.borrow_mut().host_puts.push(Span::here());
    }

    /// Record a host-consumer endpoint span for a `take` — the readback
    /// counterpart of [`note_host_put`](Self::note_host_put). The bytes cross
    /// the driver boundary, so there is no in-program value to hand back.
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
    /// consumption discipline) WITHOUT binding a port. The `inferlet` bridge
    /// claims EAGERLY at pass construction: with claims on the shared
    /// state before any pass's build, a channel consumed by a
    /// later-constructed sibling pass never misinfers as a terminal output —
    /// cross-pass handoffs need construction order, not an annotation. The
    /// bridge's build then binds with [`crate::builder::Builder::bind_port_recorded`]
    /// so the claim is not double-counted.
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
    /// the bytes across the driver instead.
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
    ///
    /// On a channel bound to a *peeked* descriptor port (geometry, masks — the
    /// ports whose discipline is read, not take) the put drains the stale value
    /// first, so a loop-carried update is one call whichever side of
    /// [`pie_ir::registry::Port::consumes`] the port falls on. An explicit
    /// `take` in the same trace is honoured and not repeated.
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
                // Record every host data put; whether it is a *seed* (a device
                // loop-carried channel the host fills once) or a host-Writer
                // edge is decided at assembly (seed ⇔ a stage also produces it).
                let mut st = self.state.borrow_mut();
                st.host_puts.push(span);
                let _ = data; // seed *values* are instance data, not needed here
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
            "channel {} is a host channel: its take crosses the driver boundary and has no \
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
    /// the driver.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Construction registers and [`Channel::release`] unregisters — the
    /// round trip the registry's ownership rests on.
    #[test]
    fn a_released_channel_stops_resolving() {
        let before = Channel::registered_count();
        let channel = Channel::new([4], DType::F32);
        let gid = channel.gid();
        assert_eq!(Channel::registered_count(), before + 1);
        assert!(Channel::by_gid(gid).is_some());

        assert!(Channel::release(gid), "the first release finds the entry");
        assert_eq!(Channel::registered_count(), before);
        assert!(
            Channel::by_gid(gid).is_none(),
            "a released gid no longer resolves"
        );
        assert!(
            !Channel::release(gid),
            "releasing twice is a no-op, not a panic"
        );
    }

    /// The retention this documents: a channel outlives every handle to it,
    /// because the handle is not what owns it. Dropping the DSL-side
    /// `Channel` must leave the state resolvable, or the SDK's gid tokens
    /// would dangle.
    #[test]
    fn dropping_a_handle_does_not_release_the_state() {
        let gid = {
            let channel = Channel::new([2], DType::F32);
            channel.gid()
        };
        assert!(
            Channel::by_gid(gid).is_some(),
            "the registry owns the state, not the handle"
        );
        Channel::release(gid);
    }
}
