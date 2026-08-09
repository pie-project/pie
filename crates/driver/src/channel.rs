//! The channel ring: the one authoritative queue a program communicates
//! through, and the per-instance state that owns them.
//!
//! A channel is a bounded SPSC queue of fixed-shape cells. Its bytes are not an
//! implementation detail: on Apple hardware the same allocation is
//! `MTLStorageModeShared`, so the head/tail words and cell bytes this module
//! reads and writes are exactly the bytes a generated kernel would bind. That
//! is why the head/tail words are published with release ordering even though
//! this CPU interpreter is single-threaded per instance — the ordering is the
//! cross-agent contract, not a local convenience.
//!
//! # Sequence addressing, not modular indices
//!
//! Head and tail are **monotonic sequence numbers**, mapped to one of
//! `capacity + 1` physical slots by `sequence % (capacity + 1)`. The extra slot
//! is what distinguishes full from empty without a separate count: `tail - head`
//! is the live size directly, and it never aliases because the sequence numbers
//! never wrap in practice. A modular index scheme would need a full/empty bit
//! that a concurrent reader could observe mid-update.
//!
//! # Host storage only
//!
//! This module allocates plain host memory. The platform (Metal-shared)
//! allocation the C++ `make_platform_channel_state` returns is a GPU concern and
//! belongs behind the Apple `cfg` with the rest of the Metal shell; a future
//! binding supplies such a ring through [`make_instance`]. Keeping this file
//! host-only is what makes the ring unit-testable without a GPU.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use driver_api::local::{PIE_CHANNEL_HOST_READER, PIE_CHANNEL_HOST_VISIBLE};

use super::plan::ExecPlan;
use super::shape_numel;
use super::value::{
    Value, concrete_dtype, decode_wire, encode_wire, value_matches, wire_cell_bytes,
};

/// One authoritative channel ring.
///
/// Interior mutability (a [`RefCell`] for the cell bytes, atomics for the
/// head/tail/poison/closed words) rather than `&mut self`, because a channel is
/// shared: an imported/exported channel is one ring referenced by two
/// instances, and an [`ExecPlan`]'s instance holds each ring behind an [`Rc`].
/// The `&self` methods keep that sharing expressible without unsafe pointer
/// aliasing, which is exactly the class of bug the port exists to remove.
pub struct ChannelState {
    dtype: tensor_ir::DType,
    numel: usize,
    capacity: usize,
    cell_bytes: usize,
    cap1: usize,
    cells: RefCell<Vec<u8>>,
    words: [AtomicU64; 4],
}

impl std::fmt::Debug for ChannelState {
    /// The ring's shape and where its head and tail are, not its cells.
    ///
    /// Printing the cells would be a megabyte for a logits ring, and the
    /// question anyone debugging a channel is asking is how full it is.
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
    /// Allocate a host-backed ring for `capacity` live cells of `numel` lanes.
    ///
    /// `dtype` is folded through [`concrete_dtype`] first, so an `Act`
    /// declaration becomes an `f32` ring; every lane count and capacity is
    /// floored at one, matching the reference interpreter (a zero-capacity ring
    /// could never hold the seed cell a channel is declared with).
    #[must_use]
    pub fn host(dtype: tensor_ir::DType, numel: usize, capacity: usize) -> ChannelState {
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
            cells: RefCell::new(vec![0u8; cell_bytes * cap1]),
            words: Default::default(),
        }
    }

    /// The address of the ring's cell storage.
    ///
    /// For the ABI's `PieChannelEndpointBinding`, which hands a host a base and
    /// a length so it can read the ring without calling back. Stable for the
    /// life of the state: [`ChannelState::host`] allocates once and nothing
    /// resizes the vector, which is what makes an address worth handing out at
    /// all.
    #[must_use]
    pub fn mirror_base(&self) -> u64 {
        self.cells.borrow().as_ptr() as u64
    }

    /// The address of the four control words.
    ///
    /// Their ORDER is the ABI's: head, tail, poison, closed at indices 0..3.
    /// `load_word`/`store_word` below use the same indices, so the two cannot
    /// drift without this comment being wrong.
    #[must_use]
    pub fn word_base(&self) -> u64 {
        self.words.as_ptr() as u64
    }

    fn load_word(&self, index: usize) -> u64 {
        self.words[index].load(Ordering::Acquire)
    }

    /// Publish `value` into word `index` with release ordering.
    ///
    /// Release, not relaxed, because a cell's bytes must be visible before the
    /// tail word that exposes them: a consumer that sees the new tail must also
    /// see the cell it points at. `step` relies on this to keep a fire's
    /// channel effects atomic.
    fn store_word(&self, index: usize, value: u64) {
        self.words[index].store(value, Ordering::Release);
    }

    /// The head sequence number — the next cell a consumer will take.
    #[must_use]
    pub fn head(&self) -> u64 {
        self.load_word(0)
    }
    /// The tail sequence number — where the next produced cell lands.
    #[must_use]
    pub fn tail(&self) -> u64 {
        self.load_word(1)
    }
    /// The poison word: non-zero once a producer has faulted the channel.
    #[must_use]
    pub fn poison(&self) -> u64 {
        self.load_word(2)
    }
    /// The closed word: non-zero once a producer has closed the channel.
    #[must_use]
    pub fn closed(&self) -> u64 {
        self.load_word(3)
    }

    /// The number of live cells (`tail - head`, clamped at zero).
    #[must_use]
    pub fn size(&self) -> usize {
        let h = self.head();
        let t = self.tail();
        if t >= h { (t - h) as usize } else { 0 }
    }
    /// Whether the ring holds no live cells.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }
    /// Whether the ring is at its logical capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.size() >= self.capacity
    }

    /// Bytes of cell storage, including the extra slot that distinguishes
    /// full from empty.
    #[must_use]
    pub fn cells_len(&self) -> usize {
        self.cells.borrow().len()
    }

    /// Bytes of control words: head, tail, poison, closed.
    #[must_use]
    pub fn words_len(&self) -> usize {
        self.words.len() * size_of::<u64>()
    }

    /// Latch the closed word, so anyone still holding the ring can see that
    /// nothing more will arrive.
    ///
    /// The value it holds is left alone. A reader that already has a cell is
    /// entitled to finish reading it; closed means no more, not gone.
    pub fn close(&self) {
        self.store_word(3, 1);
    }

    /// How many live cells the ring holds.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn slot_range(&self, sequence: u64) -> std::ops::Range<usize> {
        let base = (sequence % self.cap1 as u64) as usize * self.cell_bytes;
        base..base + self.cell_bytes
    }

    /// Decode the cell at `sequence`, falling back to zeros on a codec
    /// mismatch. A mismatch cannot arise from bytes this module wrote; the
    /// fallback keeps decode total for a ring an external agent may have
    /// touched.
    #[must_use]
    pub fn decode_sequence(&self, sequence: u64) -> Value {
        let cells = self.cells.borrow();
        decode_wire(&cells[self.slot_range(sequence)], self.dtype, self.numel)
            .unwrap_or_else(|| Value::zeros(self.dtype, self.numel))
    }

    /// Encode `value` into the cell at `sequence`. The caller must publish the
    /// tail word afterward for the write to become visible.
    pub fn encode_sequence(&self, sequence: u64, value: &Value) {
        let range = self.slot_range(sequence);
        let mut cells = self.cells.borrow_mut();
        encode_wire(value, &mut cells[range]);
    }

    /// The cell at the head sequence, whether or not it is live.
    #[must_use]
    pub fn front(&self) -> Value {
        self.decode_sequence(self.head())
    }

    /// The most recently readable cell: the head cell when the ring is non-
    /// empty, otherwise the last slot written.
    ///
    /// This is what a `chan_read` resolves to. An empty ring still answers with
    /// its last-written cell rather than zeros, because a peek of a
    /// once-written channel should see the value that was there, matching the
    /// register semantics the trace model gives channels.
    #[must_use]
    pub fn current(&self) -> Value {
        let h = self.head();
        if self.tail() > h {
            return self.decode_sequence(h);
        }
        let last_slot = (h + self.cap1 as u64 - 1) % self.cap1 as u64;
        self.decode_sequence(last_slot)
    }

    /// Push `value` as a new tail cell, or `false` if it does not match the
    /// ring's type or the ring is full.
    ///
    /// Returns `bool` rather than a `Result` because the two failure modes are
    /// one bit each and the caller ([`host_put`], [`crate::step`])
    /// already knows which it is testing — a mismatched value cannot occur once
    /// `value_matches` has passed, and a full ring is the ordinary back-pressure
    /// signal, not an error.
    pub fn push(&self, value: &Value) -> bool {
        if value.dtype() != self.dtype || value.len() != self.numel || self.is_full() {
            return false;
        }
        let t = self.tail();
        self.encode_sequence(t, value);
        self.store_word(1, t + 1);
        true
    }

    /// Publish the head sequence word (release ordering).
    ///
    /// Separate from [`ChannelState::pop`] because the pass-atomic commit in
    /// [`crate::step`] computes every ring's next head *before* it
    /// writes any of them, so it must advance the head word without also
    /// decoding a cell the way `pop` does.
    pub fn store_head(&self, sequence: u64) {
        self.store_word(0, sequence);
    }

    /// Publish the tail sequence word (release ordering).
    ///
    /// The commit encodes the pending cell with [`ChannelState::encode_sequence`]
    /// and only then releases the tail here, so a consumer that observes the new
    /// tail is guaranteed to observe the cell it exposes.
    pub fn store_tail(&self, sequence: u64) {
        self.store_word(1, sequence);
    }

    /// Pop the head cell, or `None` if the ring is empty.
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

/// Allocate a host-backed ring for a channel declaration.
///
/// A thin wrapper over [`ChannelState::host`] that reads the lane count and
/// capacity straight off the declaration's shape, returning an [`Rc`] because a
/// channel is shared by construction.
#[must_use]
pub fn make_host_channel_state(dtype_byte: u8, dims: &[u32], capacity: u32) -> Rc<ChannelState> {
    Rc::new(ChannelState::host(
        concrete_dtype(dtype_byte),
        shape_numel(dims) as usize,
        capacity as usize,
    ))
}

/// One running program instance: its channel rings and a poison latch.
///
/// The poison latch is a hard-fault flag: once a fire faults mid-commit, the
/// instance is dead and every later `step`/`host_*` call short-circuits. This
/// is a status bit about the *instance*, kept separate from the per-channel
/// poison word (which is about one ring an external agent can observe).
#[derive(Clone, Debug, Default)]
pub struct InterpInstance {
    /// One ring per declared channel, in channel-id order.
    pub channels: Vec<Rc<ChannelState>>,
    /// Whether a hard fault has killed this instance.
    pub poisoned: bool,
}

/// Bind an instance to an externally supplied set of channel rings.
///
/// Used by the production binding path, where the registry owns each channel's
/// platform-shared ring. The rings are adopted only if their count matches the
/// plan's channel count; a mismatch yields an instance with no channels, which
/// every `step` then reports as un-ready rather than indexing out of bounds.
#[must_use]
pub fn make_instance(plan: &ExecPlan, channels: Vec<Rc<ChannelState>>) -> InterpInstance {
    let mut inst = InterpInstance::default();
    if channels.len() == plan.package.channels.len() {
        inst.channels = channels;
    }
    inst
}

/// Bind a pure-host instance, allocating any channel not supplied in `externs`
/// and seeding empty channels from `seeds`.
///
/// The test/reference counterpart of [`make_instance`]: production bindings
/// supply platform-shared rings, but a host test wants every channel allocated
/// for it, with optional seed cells pushed into channels declared
/// `Channel::from(seed)`. A seed is only pushed into an empty ring, so re-
/// binding an already-seeded shared ring does not double it.
#[must_use]
pub fn make_host_instance(
    plan: &ExecPlan,
    externs: &BTreeMap<u32, Rc<ChannelState>>,
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

/// The outcome of a host channel operation.
///
/// A dedicated enum rather than a `Result<(), _>` because `WouldBlock` is not an
/// error: it is the ordinary "ring full/empty, try again" signal a host loop
/// polls on. Collapsing it into an `Err` would make every caller re-classify it
/// back out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostOp {
    /// The value was pushed / a value was taken.
    Ok,
    /// The ring was full (put) or empty (take); retry later.
    WouldBlock,
    /// The instance has hard-faulted.
    Poisoned,
    /// The channel is not host-writable (put) or not host-readable (take).
    WrongRole,
    /// The value does not match the channel's declared type (put only).
    TypeMismatch,
}

fn host_visible(flags: u8) -> bool {
    flags & PIE_CHANNEL_HOST_VISIBLE != 0
}
fn host_reader(flags: u8) -> bool {
    flags & PIE_CHANNEL_HOST_READER != 0
}

/// Push a value the host produced into a host-visible input channel.
///
/// Refuses a channel that is not host-visible, or that the host reads rather
/// than writes ([`HostOp::WrongRole`]), and a value that does not match the
/// declared type ([`HostOp::TypeMismatch`]) — the type gate is what keeps a
/// mis-shaped host value from ever reaching a ring cell.
#[must_use]
pub fn host_put(inst: &InterpInstance, plan: &ExecPlan, chan: u32, value: &Value) -> HostOp {
    if inst.poisoned {
        return HostOp::Poisoned;
    }
    let decl = &plan.package.channels[chan as usize];
    if !host_visible(decl.flags) || host_reader(decl.flags) {
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

/// Take a value the program produced out of a host-readable output channel.
///
/// Returns the taken cell alongside the outcome: data rides in the tuple,
/// status in [`HostOp`], so a `WouldBlock` cannot be mistaken for a real cell.
/// Refuses a channel the host does not harvest ([`HostOp::WrongRole`]).
#[must_use]
pub fn host_take(inst: &InterpInstance, plan: &ExecPlan, chan: u32) -> (HostOp, Option<Value>) {
    if inst.poisoned {
        return (HostOp::Poisoned, None);
    }
    let decl = &plan.package.channels[chan as usize];
    if !host_reader(decl.flags) {
        return (HostOp::WrongRole, None);
    }
    match inst.channels[chan as usize].pop() {
        Some(value) => (HostOp::Ok, Some(value)),
        None => (HostOp::WouldBlock, None),
    }
}

#[cfg(test)]
mod tests {
    use tensor_ir::DType;

    use super::*;

    #[test]
    fn size_distinguishes_full_from_empty_via_the_extra_slot() {
        let ring = ChannelState::host(DType::F32, 1, 2);
        assert!(ring.is_empty(), "a fresh ring holds nothing");
        assert!(
            ring.push(&Value::F32(vec![1.0])),
            "first push into a capacity-2 ring"
        );
        assert!(ring.push(&Value::F32(vec![2.0])), "second push fills it");
        assert!(ring.is_full(), "two live cells is the logical capacity");
        assert!(
            !ring.push(&Value::F32(vec![3.0])),
            "a full ring must refuse a third push"
        );
        assert_eq!(
            ring.size(),
            2,
            "size is tail - head, independent of the +1 slot"
        );
    }

    #[test]
    fn push_pop_round_trips_the_cell_bytes_through_the_ring() {
        let ring = ChannelState::host(DType::I32, 3, 4);
        let cell = Value::I32(vec![7, -8, 9]);
        assert!(ring.push(&cell));
        assert_eq!(
            ring.pop(),
            Some(cell),
            "a popped cell must equal the pushed one"
        );
        assert_eq!(
            ring.pop(),
            None,
            "an emptied ring yields None, never a stale cell"
        );
    }

    #[test]
    fn push_refuses_a_value_whose_type_disagrees_with_the_ring() {
        let ring = ChannelState::host(DType::F32, 2, 1);
        assert!(
            !ring.push(&Value::I32(vec![1, 2])),
            "wrong dtype must not be encoded"
        );
        assert!(
            !ring.push(&Value::F32(vec![1.0])),
            "wrong lane count must not be encoded"
        );
        assert!(
            ring.push(&Value::F32(vec![1.0, 2.0])),
            "the matching type is accepted"
        );
    }

    #[test]
    fn a_read_of_an_emptied_channel_sees_the_last_written_cell() {
        let ring = ChannelState::host(DType::F32, 1, 2);
        ring.push(&Value::F32(vec![42.0]));
        assert_eq!(ring.pop(), Some(Value::F32(vec![42.0])));
        assert!(ring.is_empty());
        assert_eq!(
            ring.current(),
            Value::F32(vec![42.0]),
            "an empty channel peeks its last value (register semantics), not zeros"
        );
    }
}
