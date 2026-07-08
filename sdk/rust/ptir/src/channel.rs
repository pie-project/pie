//! `Channel` — GPU-resident ordered memory (overview §1): a bounded queue of
//! cells with full/empty bits. Inside a traced stage, `take`/`read`/`put` record
//! echo's `ChanTake`/`ChanRead`/`ChanPut` ops; on the host they take the async
//! path.

use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::sync::atomic::{AtomicU64, Ordering};

use pie_ptir::types::{DType, Shape};

use crate::context::{self, ChannelRef, ChannelState};
use crate::error::Span;
use crate::value::{reshape_id_to, AsTensor, ConstData, IntoConst, IntoShape, Tensor};

static NEXT_GID: AtomicU64 = AtomicU64::new(1);

/// A handle to a channel's shared state (overview §1). Cheap to clone; captured
/// by both host code and the stage closures that read/write it.
#[derive(Clone)]
pub struct Channel {
    state: ChannelRef,
}

impl Channel {
    /// `Channel::new([shape], dtype)` — a capacity-1 channel (overview §1).
    pub fn new(shape: impl IntoShape, dtype: DType) -> Channel {
        Channel::build(shape.into_shape(), dtype, 1, None)
    }

    /// `Channel::from(v)` — sugar for `new` + `put`: a channel seeded full with
    /// `v` (overview §1). `v` may be per-instance *data* (a request seed); the
    /// seed is instance state, never in the container (D2).
    pub fn from(v: impl IntoConst) -> Channel {
        let data = v.into_const();
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
        Channel { state }
    }

    /// Widen the ring to `n` cells (deeper run-ahead; overview §1/§3).
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
    pub fn dtype(&self) -> DType {
        self.state.borrow().dtype
    }
    pub fn shape(&self) -> Shape {
        self.state.borrow().shape
    }

    /// Record a descriptor-port token consume (`embed`/`positions`/`w_slot`/
    /// `w_off`); the forward's consumer endpoint (overview §5.1).
    pub(crate) fn claim_desc_take(&self, span: Span) {
        self.state.borrow_mut().desc_takes.push(span);
    }
    /// Record a descriptor-port geometry/mask peek (exempt from the consumer count).
    pub(crate) fn claim_desc_read(&self, span: Span) {
        self.state.borrow_mut().desc_reads.push(span);
    }

    /// `take()` — full ⇒ value + empty; empty ⇒ block. In-program: returns the
    /// taken `Tensor`. On the host: an awaitable yielding a wasm `Vec`.
    #[track_caller]
    pub fn take(&self) -> Taken {
        let span = Span::here();
        if context::is_tracing() {
            let (id, ty) = context::record_channel_read(&self.state, true, span);
            Taken::in_program(Tensor::node(id, ty))
        } else {
            self.state.borrow_mut().host_takes.push(span);
            Taken::host(self.state.clone(), true)
        }
    }

    /// `read()` — full ⇒ copy, stays full; empty ⇒ block. A peek (does not claim
    /// the consumer endpoint; overview §1/§3 `len`).
    #[track_caller]
    pub fn read(&self) -> Taken {
        let span = Span::here();
        if context::is_tracing() {
            let (id, ty) = context::record_channel_read(&self.state, false, span);
            Taken::in_program(Tensor::node(id, ty))
        } else {
            self.state.borrow_mut().host_reads.push(span);
            Taken::host(self.state.clone(), false)
        }
    }

    /// `put(v)` — empty ⇒ fill + full; full ⇒ block (back-pressure). In-program
    /// `v` is a `Tensor` (reshaped to fit the cell); on the host `v` is data.
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
                let _ = data; // seed *values* are instance data (D2), not needed here
                Put::done()
            }
        }
    }
}

/// The result of `channel.take()` / `channel.read()`. In a program it is a
/// `Tensor` (via [`AsTensor`]); on the host it is `.await`-able.
pub struct Taken {
    inner: TakenInner,
}

enum TakenInner {
    InProgram(Tensor),
    Host { chan: ChannelRef, consume: bool },
}

impl Taken {
    fn in_program(t: Tensor) -> Taken {
        Taken { inner: TakenInner::InProgram(t) }
    }
    fn host(chan: ChannelRef, consume: bool) -> Taken {
        Taken { inner: TakenInner::Host { chan, consume } }
    }
    /// The in-program `Tensor` (panics if this is a host take — a frontend bug).
    pub fn tensor(self) -> Tensor {
        match self.inner {
            TakenInner::InProgram(t) => t,
            TakenInner::Host { .. } => panic!("host channel take used as an in-program Tensor"),
        }
    }
}

impl AsTensor for Taken {
    fn to_arg(&self) -> crate::value::Arg {
        match &self.inner {
            TakenInner::InProgram(t) => t.to_arg(),
            TakenInner::Host { .. } => panic!("host channel take used as an in-program Tensor"),
        }
    }
}
impl AsTensor for &Taken {
    fn to_arg(&self) -> crate::value::Arg {
        (*self).to_arg()
    }
}

// Host-side await surface (P3 wires real async over the driver channel).
impl Taken {
    /// Await the host value (P3). Consumes the cell for `take`, copies for `read`.
    pub async fn get<T: HostElem>(self) -> Result<Vec<T>, HostError> {
        match self.inner {
            TakenInner::Host { chan, consume } => {
                let _ = (&chan, consume);
                Err(HostError::NotBound)
            }
            TakenInner::InProgram(_) => Err(HostError::InProgram),
        }
    }
}

/// Host errors surfaced by a blocked `take`/`read` (poison; overview §1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostError {
    Poisoned,
    NotBound,
    InProgram,
}

/// A host-readable element type.
pub trait HostElem: Copy {}
impl HostElem for i32 {}
impl HostElem for u32 {}
impl HostElem for f32 {}

/// The (fire-and-forget) result of a `put`. Host puts coalesce before the next
/// submit (D1); the handle exists so back-pressure can be awaited in P3.
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
    Tensor(Tensor),
    Data(ConstData),
}

/// Anything puttable: a `Tensor` (in-program) or host data (arrays / vecs / scalars).
pub trait IntoPut {
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
