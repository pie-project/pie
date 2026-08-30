use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use eta_ir::container::{ChanDType, HostRole};

use super::plan::ExecPlan;
use super::value::{
    Value, concrete_dtype, decode_wire, encode_wire, value_matches, wire_cell_bytes,
};
use crate::shape_numel;

const POISONED: &str = "a channel's cells were left locked by a panic";

pub struct ChannelState {
    dtype: eta_ir::Dtype,
    numel: usize,
    capacity: usize,
    cell_bytes: usize,
    cap1: usize,
    cells: Mutex<Vec<u8>>,
    words: [AtomicU64; 4],
}

impl std::fmt::Debug for ChannelState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelState")
            .field("dtype", &self.dtype)
            .field("numel", &self.numel)
            .field("capacity", &self.capacity)
            .field("head", &self.head())
            .field("tail", &self.tail())
            .field("closed", &self.closed())
            .finish()
    }
}

impl ChannelState {
    #[must_use]
    pub fn host(dtype: eta_ir::Dtype, numel: usize, capacity: usize) -> ChannelState {
        let numel = numel.max(1);
        let capacity = capacity.max(1);
        let cell_bytes = wire_cell_bytes(dtype, numel);
        let cap1 = capacity + 1;
        ChannelState {
            dtype,
            numel,
            capacity,
            cell_bytes,
            cap1,
            cells: Mutex::new(vec![0u8; cell_bytes * cap1]),
            words: Default::default(),
        }
    }

    fn load_word(&self, index: usize) -> u64 {
        self.words[index].load(Ordering::Acquire)
    }

    fn store_word(&self, index: usize, value: u64) {
        self.words[index].store(value, Ordering::Release);
    }

    #[must_use]
    pub fn head(&self) -> u64 {
        self.load_word(0)
    }

    #[must_use]
    pub fn tail(&self) -> u64 {
        self.load_word(1)
    }

    #[must_use]
    pub fn poison(&self) -> u64 {
        self.load_word(2)
    }

    #[must_use]
    pub fn closed(&self) -> u64 {
        self.load_word(3)
    }

    #[must_use]
    pub fn size(&self) -> usize {
        let h = self.head();
        let t = self.tail();
        if t >= h { (t - h) as usize } else { 0 }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    #[must_use]
    pub fn is_full(&self) -> bool {
        self.size() >= self.capacity
    }

    #[must_use]
    pub fn cells_len(&self) -> usize {
        self.cells.lock().expect(POISONED).len()
    }

    #[must_use]
    pub fn words_len(&self) -> usize {
        self.words.len() * size_of::<u64>()
    }

    pub fn close(&self) {
        self.store_word(3, 1);
    }

    pub fn fault(&self) {
        self.store_word(2, 1);
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn slot_range(&self, sequence: u64) -> std::ops::Range<usize> {
        let base = (sequence % self.cap1 as u64) as usize * self.cell_bytes;
        base..base + self.cell_bytes
    }

    #[must_use]
    pub fn decode_sequence(&self, sequence: u64) -> Value {
        let cells = self.cells.lock().expect(POISONED);
        decode_wire(&cells[self.slot_range(sequence)], self.dtype, self.numel)
            .unwrap_or_else(|| Value::zeros(self.dtype, self.numel))
    }

    pub fn encode_sequence(&self, sequence: u64, value: &Value) {
        let range = self.slot_range(sequence);
        let mut cells = self.cells.lock().expect(POISONED);
        encode_wire(value, &mut cells[range]);
    }

    #[must_use]
    pub fn front(&self) -> Value {
        self.decode_sequence(self.head())
    }

    #[must_use]
    pub fn current(&self) -> Value {
        let h = self.head();
        if self.tail() > h {
            return self.decode_sequence(h);
        }
        let last_slot = (h + self.cap1 as u64 - 1) % self.cap1 as u64;
        self.decode_sequence(last_slot)
    }

    pub fn push(&self, value: &Value) -> bool {
        if value.dtype() != self.dtype || value.len() != self.numel || self.is_full() {
            return false;
        }
        let t = self.tail();
        self.encode_sequence(t, value);
        self.store_word(1, t + 1);
        true
    }

    pub fn store_head(&self, sequence: u64) {
        self.store_word(0, sequence);
    }

    pub fn store_tail(&self, sequence: u64) {
        self.store_word(1, sequence);
    }

    #[must_use]
    pub fn pop(&self) -> Option<Value> {
        if self.is_empty() {
            return None;
        }
        let h = self.head();
        let value = self.decode_sequence(h);
        self.store_word(0, h + 1);
        Some(value)
    }
}

#[must_use]
pub fn make_host_channel_state(dtype: ChanDType, dims: &[u32], capacity: u32) -> Arc<ChannelState> {
    Arc::new(ChannelState::host(
        concrete_dtype(dtype),
        shape_numel(dims) as usize,
        capacity as usize,
    ))
}

#[derive(Clone, Debug, Default)]
pub struct InterpInstance {
    pub channels: Vec<Arc<ChannelState>>,

    pub poisoned: bool,
}

#[must_use]
pub fn make_instance(plan: &ExecPlan, channels: Vec<Arc<ChannelState>>) -> InterpInstance {
    let mut inst = InterpInstance::default();
    if channels.len() == plan.package.channels.len() {
        inst.channels = channels;
    }
    inst
}

#[must_use]
pub fn make_host_instance(
    plan: &ExecPlan,
    externs: &BTreeMap<u32, Arc<ChannelState>>,
    seeds: &BTreeMap<u32, Value>,
) -> InterpInstance {
    let mut inst = InterpInstance::default();
    for (ci, decl) in plan.package.channels.iter().enumerate() {
        let ci = ci as u32;
        let ring = externs
            .get(&ci)
            .cloned()
            .unwrap_or_else(|| make_host_channel_state(decl.dtype, &decl.shape, decl.capacity));
        if let Some(seed) = seeds.get(&ci)
            && ring.is_empty()
        {
            let _ = ring.push(seed);
        }
        inst.channels.push(ring);
    }
    inst
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostOp {
    Ok,

    WouldBlock,

    Poisoned,

    WrongRole,

    TypeMismatch,
}

#[must_use]
pub fn host_put(inst: &InterpInstance, plan: &ExecPlan, chan: u32, value: &Value) -> HostOp {
    if inst.poisoned {
        return HostOp::Poisoned;
    }
    let decl = &plan.package.channels[chan as usize];
    // Was `host_visible(flags) && !host_reader(flags)` over two bits. The
    // three states those two bits encoded are exactly `HostRole`'s variants,
    // and the fourth bit pattern — reader without visible — was reachable and
    // meant nothing.
    if decl.host_role != HostRole::Writer {
        return HostOp::WrongRole;
    }
    if !value_matches(value, decl.dtype, &decl.shape) {
        return HostOp::TypeMismatch;
    }
    if inst.channels[chan as usize].push(value) {
        HostOp::Ok
    } else {
        HostOp::WouldBlock
    }
}

#[must_use]
pub fn host_take(inst: &InterpInstance, plan: &ExecPlan, chan: u32) -> (HostOp, Option<Value>) {
    if inst.poisoned {
        return (HostOp::Poisoned, None);
    }
    let decl = &plan.package.channels[chan as usize];
    if decl.host_role != HostRole::Reader {
        return (HostOp::WrongRole, None);
    }
    match inst.channels[chan as usize].pop() {
        Some(value) => (HostOp::Ok, Some(value)),
        None => (HostOp::WouldBlock, None),
    }
}
