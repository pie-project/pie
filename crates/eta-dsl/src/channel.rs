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

#[derive(Clone)]
pub struct Channel {
    state: ChannelRef,
}

impl Channel {
    pub fn new(shape: impl IntoShape, dtype: Dtype) -> Channel {
        Channel::build(shape.into_shape(), dtype, 1, None)
    }

    pub fn from(v: impl IntoConst) -> Channel {
        let data = v.into_const();
        Channel::build(data.shape, data.dtype, 1, Some(data))
    }

    #[track_caller]
    pub fn from_shaped(shape: impl IntoShape, v: impl IntoConst) -> Channel {
        let mut data = v.into_const();
        let shape = shape.into_shape();
        if shape.numel() == data.shape.numel() {
            data.shape = shape;
        } else {
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

    pub fn seeded(shape: impl IntoShape, dtype: Dtype) -> Channel {
        let ch = Channel::build(shape.into_shape(), dtype, 1, None);
        ch.state.borrow_mut().seeded = true;
        ch
    }

    pub fn by_gid(gid: u64) -> Option<Channel> {
        context::channel_state_by_gid(gid).map(|state| Channel { state })
    }

    pub fn release(gid: u64) -> bool {
        context::release_channel_state(gid)
    }

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

    pub fn capacity(self, n: u32) -> Channel {
        self.state.borrow_mut().capacity = n;
        self
    }

    pub fn named(self, name: &str) -> Channel {
        self.state.borrow_mut().name = String::from(name);
        self
    }

    pub(crate) fn state(&self) -> &ChannelRef {
        &self.state
    }
    pub fn dtype(&self) -> Dtype {
        self.state.borrow().dtype
    }

    pub fn name(&self) -> String {
        self.state.borrow().name.clone()
    }
    pub fn shape(&self) -> Shape {
        self.state.borrow().shape
    }
    pub fn gid(&self) -> u64 {
        self.state.borrow().gid
    }

    #[track_caller]
    pub fn note_host_put(&self) {
        self.state.borrow_mut().host_puts.push(Span::here());
    }

    #[track_caller]
    pub fn note_host_take(&self) {
        self.state.borrow_mut().host_takes.push(Span::here());
    }

    #[track_caller]
    pub fn note_host_read(&self) {
        self.state.borrow_mut().host_reads.push(Span::here());
    }

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
                let mut st = self.state.borrow_mut();
                st.host_puts.push(span);
                let _ = data;
                Put::done()
            }
        }
    }
}

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

pub struct Put(());
impl Put {
    fn done() -> Put {
        Put(())
    }
}

pub enum PutValue {
    Tensor(Tensor),
    Data(ConstData),
}

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
