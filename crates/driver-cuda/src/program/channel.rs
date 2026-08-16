//! The device-side channel ring: cells the kernels read/write and the four
//! cursors the control kernels advance. A cell is native on the device (a byte
//! per bool lane, four per anything else) and wire on the host (bool packed
//! eight to a byte). The ring is `capacity + 1`, the extra a sentinel; the
//! `full` array's row pitch is [`MAX_RING`] whatever the capacity, else it reads
//! the next channel's flags.

use driver::tensor_ir::DType;

use crate::device::{Allocator, DeviceBuffer, StreamRef};
use crate::error::{Error, Result};

use super::run::MAX_RING;

/// Native device bytes for a cell of `numel` lanes of `dtype`: one byte per
/// bool lane (a kernel indexes lane by lane), four per anything else.
#[must_use]
pub fn native_cell_bytes(dtype: DType, numel: usize) -> usize {
    if dtype == DType::Bool {
        numel
    } else {
        numel * 4
    }
}

/// One channel's geometry, as the ring needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelShape {
    /// Lanes in one cell.
    pub numel: usize,
    /// The cell's element type.
    pub dtype: DType,
    /// How many unconsumed items the channel holds. The ring is one longer.
    pub capacity: u32,
}

impl ChannelShape {
    /// The ring length: `capacity + 1`, for the sentinel.
    ///
    /// # Errors
    ///
    /// If the ring exceeds [`MAX_RING`], the pitch the `full` array is indexed
    /// with — a longer ring would write into the next channel's flags.
    pub fn ring(&self) -> Result<u32> {
        let ring = self.capacity.saturating_add(1);
        if ring > MAX_RING {
            return Err(Error::invalid(
                "program::channel",
                format!(
                    "a ring of {ring} exceeds the {MAX_RING}-slot pitch the full \
                     array is indexed with"
                ),
            ));
        }
        Ok(ring)
    }

    /// Native bytes in one cell.
    #[must_use]
    pub fn cell_bytes(&self) -> usize {
        native_cell_bytes(self.dtype, self.numel)
    }
}

/// Every channel the driver holds: the cells and the four cursor arrays,
/// indexed by a global channel slot.
///
/// One registry, not one per instance: a channel has one ring wherever it is
/// named from. The four cursor arrays stay flat and contiguous because the
/// control kernels take them as four pointers indexed by global slot; cells are
/// one allocation per slot, so growth reallocates only the cursor rows.
#[derive(Debug)]
pub struct Rings {
    cells: Vec<DeviceBuffer>,
    full: DeviceBuffer,
    head: DeviceBuffer,
    tail: DeviceBuffer,
    cap1: DeviceBuffer,
    shapes: Vec<ChannelShape>,
    /// Slots the four cursor arrays have room for; never below `shapes.len()`.
    reserved: usize,
}

impl Rings {
    /// Allocate and zero a registry holding `shapes`, at slots `0..shapes.len()`.
    pub fn new(alloc: &Allocator, shapes: &[ChannelShape], stream: StreamRef<'_>) -> Result<Self> {
        // Zeroed: a fresh allocation is not promised zero, and a garbage cursor
        // is a ring already full or mid-sequence.
        let zero = |bytes: usize| -> Result<DeviceBuffer> {
            let mut buffer = alloc.alloc(bytes.max(1))?;
            buffer.memset(0, stream)?;
            Ok(buffer)
        };
        let mut rings = Self {
            cells: Vec::new(),
            full: zero(0)?,
            head: zero(0)?,
            tail: zero(0)?,
            cap1: zero(0)?,
            shapes: Vec::new(),
            reserved: 0,
        };
        for shape in shapes {
            rings.register(alloc, *shape, stream)?;
        }
        Ok(rings)
    }

    /// Give a channel a ring of its own, and answer the slot it landed at.
    ///
    /// Slots are handed out in order and never reused: a slot is the identity a
    /// `Session`'s dense map points at, so reuse would hand a new channel the old
    /// one's cursors. A closed slot is not reclaimed — no free list yet.
    ///
    /// # Errors
    ///
    /// If the ring is longer than [`MAX_RING`], if the cell is empty, or if the
    /// device refuses the allocation.
    pub fn register(
        &mut self,
        alloc: &Allocator,
        shape: ChannelShape,
        stream: StreamRef<'_>,
    ) -> Result<u32> {
        let ring = shape.ring()?;
        let cell = shape.cell_bytes();
        if cell == 0 {
            return Err(Error::invalid(
                "program::channel",
                "a channel whose cell is zero bytes holds nothing and can \
                 never be ready",
            ));
        }
        let slot = self.shapes.len();
        self.reserve(alloc, slot + 1, stream)?;
        let mut cells = alloc.alloc(cell * ring as usize)?;
        cells.memset(0, stream)?;
        self.cells.push(cells);
        self.shapes.push(shape);
        // `cap1` alone is written rather than zeroed: it is the modulus every
        // cursor advance divides by, and a zero there faults the commit kernel.
        self.cap1
            .write_at(slot * size_of::<u32>(), &ring.to_le_bytes(), stream)?;
        u32::try_from(slot)
            .map_err(|_| Error::invalid("program::channel", "more channels than a u32 counts"))
    }

    /// Grow the four cursor arrays to at least `want` slots (doubling, so not
    /// every registration reallocates), preserving what the live ones say.
    fn reserve(&mut self, alloc: &Allocator, want: usize, stream: StreamRef<'_>) -> Result<()> {
        if want <= self.reserved {
            return Ok(());
        }
        let grown = want.max(self.reserved * 2).max(8);
        // Read the live prefix back before the swap: the cursors live on the
        // device, so a host-side copy would go stale on the first commit.
        let live = self.shapes.len();
        let mut old_full = vec![0u8; live * MAX_RING as usize];
        let mut old_head = vec![0u8; live * size_of::<u32>()];
        let mut old_tail = vec![0u8; live * size_of::<u32>()];
        let mut old_cap1 = vec![0u8; live * size_of::<u32>()];
        if live != 0 {
            self.full.copy_to_host(&mut old_full, stream)?;
            self.head.copy_to_host(&mut old_head, stream)?;
            self.tail.copy_to_host(&mut old_tail, stream)?;
            self.cap1.copy_to_host(&mut old_cap1, stream)?;
            stream.synchronize()?;
        }
        let fresh = |bytes: usize, seed: &[u8]| -> Result<DeviceBuffer> {
            let mut buffer = alloc.alloc(bytes)?;
            buffer.memset(0, stream)?;
            if !seed.is_empty() {
                buffer.write_at(0, seed, stream)?;
            }
            Ok(buffer)
        };
        self.full = fresh(grown * MAX_RING as usize, &old_full)?;
        self.head = fresh(grown * size_of::<u32>(), &old_head)?;
        self.tail = fresh(grown * size_of::<u32>(), &old_tail)?;
        self.cap1 = fresh(grown * size_of::<u32>(), &old_cap1)?;
        stream.synchronize()?;
        self.reserved = grown;
        Ok(())
    }

    /// How many channels this registry holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    /// Whether the registry holds no channels yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// Channel `c`'s geometry.
    #[must_use]
    pub fn shape(&self, c: usize) -> Option<ChannelShape> {
        self.shapes.get(c).copied()
    }

    /// The device address of channel `c`'s cell at ring slot `slot` — what a
    /// [`LaneChannelSlot`](driver::LaneChannelSlot)'s `committed_cell`/`pending_cell`
    /// hold, resolved on the host because the kernel does no ring arithmetic.
    ///
    /// # Errors
    ///
    /// If `c` is not a channel. `slot` is reduced modulo the ring, not refused:
    /// a cursor is monotonic and the ring position is its residue.
    pub fn cell_address(&self, c: usize, slot: u32) -> Result<u64> {
        let shape = self
            .shapes
            .get(c)
            .ok_or_else(|| Error::invalid("program::channel", format!("no channel {c}")))?;
        let ring = shape.ring()?;
        let at = (slot % ring) as usize * shape.cell_bytes();
        Ok(self.cells[c].as_ptr() as u64 + at as u64)
    }

    /// The `full` flag array, for the control kernels.
    #[must_use]
    pub fn full_ptr(&self) -> *mut std::ffi::c_void {
        self.full.as_ptr()
    }

    /// The `head` cursor array.
    #[must_use]
    pub fn head_ptr(&self) -> *mut std::ffi::c_void {
        self.head.as_ptr()
    }

    /// The `tail` cursor array.
    #[must_use]
    pub fn tail_ptr(&self) -> *mut std::ffi::c_void {
        self.tail.as_ptr()
    }

    /// The `cap1` modulus array.
    #[must_use]
    pub fn cap1_ptr(&self) -> *mut std::ffi::c_void {
        self.cap1.as_ptr()
    }

    /// Write `bytes` into channel `c`'s cell at `slot` and mark it full. Writes
    /// the native form, so a wire cell must be unpacked first.
    ///
    /// # Errors
    ///
    /// If the channel does not exist or `bytes` is not exactly one native cell;
    /// a short write would leave real-looking garbage in the cell's tail.
    pub fn seed(&mut self, c: usize, slot: u32, bytes: &[u8], stream: StreamRef<'_>) -> Result<()> {
        let shape = self
            .shapes
            .get(c)
            .copied()
            .ok_or_else(|| Error::invalid("program::channel", format!("no channel {c}")))?;
        if bytes.len() != shape.cell_bytes() {
            return Err(Error::invalid(
                "program::channel",
                format!(
                    "channel {c}'s native cell is {} bytes and {} were offered",
                    shape.cell_bytes(),
                    bytes.len()
                ),
            ));
        }
        let ring = shape.ring()?;
        let at = (slot % ring) as usize * shape.cell_bytes();
        self.cells[c].write_at(at, bytes, stream)?;

        // The flag, then the cursor: a full cell whose tail has not advanced is
        // merely unpublished, but a tail past a clear flag blocks a reader forever.
        self.full.write_at(
            c * MAX_RING as usize + (slot % ring) as usize,
            &[1u8],
            stream,
        )?;
        let next = ((slot % ring) + 1) % ring;
        self.tail
            .write_at(c * size_of::<u32>(), &next.to_le_bytes(), stream)?;
        Ok(())
    }

    /// Read channel `c`'s cell at `slot` back to the host, in native form.
    pub fn read_cell(&self, c: usize, slot: u32, stream: StreamRef<'_>) -> Result<Vec<u8>> {
        let shape = self
            .shapes
            .get(c)
            .copied()
            .ok_or_else(|| Error::invalid("program::channel", format!("no channel {c}")))?;
        let ring = shape.ring()?;
        let at = (slot % ring) as usize * shape.cell_bytes();
        let mut out = vec![0u8; shape.cell_bytes()];
        self.cells[c].read_at(at, &mut out, stream)?;
        Ok(out)
    }

    /// Consume channel `c`'s committed cell: clear its full bit and advance
    /// `head`, as the commit kernel's `taken` loop does.
    ///
    /// The host does this for descriptor ports (`EmbedTokens`, `Positions`,
    /// `WSlot`, `WOff`): no `LaunchOp` names them, so left unconsumed their tails
    /// outrun their heads until the ring fills and the decode stalls. Safe on the
    /// host because the previous step's `fire` ended on `stream.synchronize()`.
    pub fn consume_front(&mut self, c: usize, stream: StreamRef<'_>) -> Result<()> {
        let shape = self
            .shapes
            .get(c)
            .copied()
            .ok_or_else(|| Error::invalid("program::channel", format!("no channel {c}")))?;
        let ring = shape.ring()?;
        let mut head = [0u8; 4];
        self.head.read_at(c * size_of::<u32>(), &mut head, stream)?;
        stream.synchronize()?;
        let head = u32::from_le_bytes(head) % ring;
        // The flag, then the cursor — the commit kernel's order, so a reader
        // catching the pair mid-update sees unconsumed, not consumed twice.
        self.full
            .write_at(c * MAX_RING as usize + head as usize, &[0u8], stream)?;
        let next = (head + 1) % ring;
        self.head
            .write_at(c * size_of::<u32>(), &next.to_le_bytes(), stream)?;
        stream.synchronize()?;
        Ok(())
    }

    /// The four cursors of every channel, read back from the device rather than
    /// tracked (a host copy would go stale on the first commit).
    pub fn cursors(&self, stream: StreamRef<'_>) -> Result<Vec<Cursors>> {
        let count = self.shapes.len();
        let mut head = vec![0u8; count * size_of::<u32>()];
        let mut tail = vec![0u8; count * size_of::<u32>()];
        let mut full = vec![0u8; count * MAX_RING as usize];
        self.head.copy_to_host(&mut head, stream)?;
        self.tail.copy_to_host(&mut tail, stream)?;
        self.full.copy_to_host(&mut full, stream)?;
        stream.synchronize()?;

        let word = |bytes: &[u8], i: usize| {
            u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("four bytes"))
        };
        Ok((0..count)
            .map(|c| Cursors {
                head: word(&head, c),
                tail: word(&tail, c),
                full: full[c * MAX_RING as usize..(c + 1) * MAX_RING as usize].to_vec(),
            })
            .collect())
    }
}

/// One channel's cursors, read back from the device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cursors {
    /// The consumer's position.
    pub head: u32,
    /// The producer's position.
    pub tail: u32,
    /// One flag per ring slot, [`MAX_RING`] long whatever the capacity is.
    pub full: Vec<u8>,
}

impl Cursors {
    /// Whether the cell at `head` holds a published value.
    #[must_use]
    pub fn is_readable(&self) -> bool {
        self.full
            .get(self.head as usize)
            .is_some_and(|&flag| flag != 0)
    }

    /// How many published items are unconsumed, given the ring length.
    #[must_use]
    pub fn depth(&self, ring: u32) -> u32 {
        (self.tail + ring - self.head) % ring
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bool cell is a byte per lane on the device, a bit per lane on the wire.
    #[test]
    fn a_bool_cell_is_bytes_on_the_device_and_bits_on_the_wire() {
        assert_eq!(native_cell_bytes(DType::Bool, 128), 128);
        assert_eq!(driver::wire_cell_bytes(DType::Bool, 128), 16);
        assert_eq!(native_cell_bytes(DType::F32, 128), 512);
        assert_eq!(driver::wire_cell_bytes(DType::F32, 128), 512);
    }

    /// The ring is one longer than the capacity — the sentinel slot.
    #[test]
    fn the_ring_carries_a_sentinel_slot_beyond_the_capacity() {
        let shape = ChannelShape {
            numel: 4,
            dtype: DType::F32,
            capacity: 1,
        };
        assert_eq!(shape.ring().expect("fits"), 2);
        assert_eq!(shape.cell_bytes(), 16);
    }

    /// A ring longer than the `full` pitch is refused at the geometry, not the launch.
    #[test]
    fn a_ring_longer_than_the_flag_pitch_is_refused() {
        let shape = ChannelShape {
            numel: 1,
            dtype: DType::F32,
            capacity: MAX_RING,
        };
        let error = shape.ring().expect_err("a ring of 65 must not be accepted");
        assert!(error.to_string().contains("pitch"), "{error}");

        let largest = ChannelShape {
            numel: 1,
            dtype: DType::F32,
            capacity: MAX_RING - 1,
        };
        assert_eq!(largest.ring().expect("63 + 1 fits"), MAX_RING);
    }

    /// Depth is the cursor difference mod the ring: a full ring reads as `capacity`.
    #[test]
    fn depth_reads_a_full_ring_as_its_capacity_and_not_as_empty() {
        let empty = Cursors {
            head: 3,
            tail: 3,
            full: vec![0; MAX_RING as usize],
        };
        assert_eq!(empty.depth(4), 0);
        let full = Cursors {
            head: 0,
            tail: 3,
            full: vec![1; MAX_RING as usize],
        };
        assert_eq!(full.depth(4), 3, "a ring of 4 holds at most 3 items");
        let wrapped = Cursors {
            head: 3,
            tail: 1,
            full: vec![1; MAX_RING as usize],
        };
        assert_eq!(wrapped.depth(4), 2, "the difference wraps with the ring");
    }
}

// ── The host plane the device rings above are bridged to ──

// A channel exists twice: `pie_cuda_register_channel` allocates the pinned host
// mirror the engine polls, and [`Rings`] the device rings the kernels read.
// Nothing joins them, so the bridge is a copy: a fire pulls inputs across before
// its stages and pushes outputs back after. The bool encoding is the subtlety —
// native bytes on the device, bit-packed LSB-first on the wire — so a byte copy
// would be right for every dtype but bool.

/// Bytes one cell occupies in the host mirror, for `numel` lanes — the
/// counterpart of [`native_cell_bytes`], and the only difference between the planes.
#[must_use]
pub fn wire_cell_bytes(dtype: DType, numel: usize) -> usize {
    if dtype == DType::Bool {
        numel.div_ceil(8)
    } else {
        numel * 4
    }
}

/// A wire cell, as the device wants it (native form).
///
/// # Errors
///
/// If `wire` is not exactly one wire cell; a short cell reads real-looking
/// garbage past its end.
pub fn wire_to_native(
    dtype: DType,
    numel: usize,
    wire: &[u8],
) -> std::result::Result<Vec<u8>, String> {
    let want = wire_cell_bytes(dtype, numel);
    if wire.len() != want {
        return Err(format!(
            "a {} wire cell of {numel} lanes is {want} bytes and {} were offered",
            dtype.name(),
            wire.len()
        ));
    }
    if dtype != DType::Bool {
        return Ok(wire.to_vec());
    }
    Ok((0..numel)
        .map(|i| u8::from(wire[i / 8] >> (i % 8) & 1 == 1))
        .collect())
}

/// A native cell, as the wire wants it (packed form).
///
/// Any nonzero byte is `true`: the device promises only nonzero-means-set, so
/// reading `== 1` would drop a `0xff` mask byte.
pub fn native_to_wire(
    dtype: DType,
    numel: usize,
    native: &[u8],
) -> std::result::Result<Vec<u8>, String> {
    let want = native_cell_bytes(dtype, numel);
    if native.len() != want {
        return Err(format!(
            "a {} native cell of {numel} lanes is {want} bytes and {} were offered",
            dtype.name(),
            native.len()
        ));
    }
    if dtype != DType::Bool {
        return Ok(native.to_vec());
    }
    let mut out = vec![0u8; wire_cell_bytes(dtype, numel)];
    for (i, &b) in native.iter().enumerate().take(numel) {
        if b != 0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    Ok(out)
}

/// One channel's host plane: the pinned mirror the engine polls and the four
/// control words `[head, tail, poison, closed]` beside it. A view, not an owner
/// — `register_channel` allocates both and this borrows them; its layout must
/// agree with that entry point byte for byte.
///
/// Both cursors free-run and the slot is `cursor % ring`, so empty and full are
/// distinguishable — wrapping the cursors would make `head == tail` mean both.
pub struct HostChannel {
    mirror: *mut u8,
    words: *mut u64,
    /// Wire bytes per cell.
    pub cell_bytes: usize,
    /// `capacity + 1`.
    pub ring: u32,
    /// `PIE_CHANNEL_HOST_ROLE_*`: which side of this mirror the engine is on.
    ///
    /// One head/tail pair serves both directions, so the role decides direction:
    /// the driver takes only from a plane the engine writes and publishes only
    /// into one it reads, or a loop-carried channel re-injects its own output.
    /// `NONE` is device-only.
    pub role: u8,
}

impl HostChannel {
    /// Borrow the host plane of a registered channel.
    ///
    /// # Safety
    ///
    /// `mirror` must point at `cell_bytes * ring` writable bytes and `words` at
    /// four `u64`s, both live for `'_` — what `register_channel` allocates.
    #[must_use]
    pub const unsafe fn new(
        mirror: *mut std::ffi::c_void,
        words: *mut std::ffi::c_void,
        cell_bytes: usize,
        ring: u32,
        role: u8,
    ) -> Self {
        Self {
            mirror: mirror.cast(),
            words: words.cast(),
            cell_bytes,
            ring,
            role,
        }
    }

    /// Does the engine write this plane? Only then may the driver take from it.
    /// See [`HostChannel::role`].
    #[must_use]
    pub const fn engine_writes(&self) -> bool {
        self.role == driver::driver_api::local::PIE_CHANNEL_HOST_ROLE_WRITER
    }

    /// Does the engine read this plane? Only then may the driver publish into
    /// it. See [`HostChannel::role`].
    #[must_use]
    pub const fn engine_reads(&self) -> bool {
        self.role == driver::driver_api::local::PIE_CHANNEL_HOST_ROLE_READER
    }

    fn word(&self, i: usize) -> u64 {
        unsafe { self.words.add(i).read_volatile() }
    }

    /// How many published items the consumer has not taken.
    #[must_use]
    pub fn depth(&self) -> u64 {
        self.word(1).wrapping_sub(self.word(0))
    }

    /// Take the oldest published cell, advancing `head` — the driver reading
    /// the engine's side of an input channel.
    #[must_use]
    pub fn take(&mut self) -> Option<Vec<u8>> {
        let (head, tail) = (self.word(0), self.word(1));
        if head == tail {
            return None;
        }
        let slot = (head % u64::from(self.ring)) as usize;
        let mut cell = vec![0u8; self.cell_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.mirror.add(slot * self.cell_bytes),
                cell.as_mut_ptr(),
                self.cell_bytes,
            );
        }
        // The read, then the cursor: this acquire fence pairs with `publish`'s
        // release and orders the copy above.
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        unsafe { self.words.add(0).write_volatile(head + 1) };
        Some(cell)
    }

    /// Publish a wire cell, advancing `tail`. `false` when the ring is full,
    /// which the caller retries rather than blocking or dropping.
    #[must_use]
    pub fn publish(&mut self, cell: &[u8]) -> bool {
        if cell.len() != self.cell_bytes {
            return false;
        }
        let (head, tail) = (self.word(0), self.word(1));
        if tail.wrapping_sub(head) >= u64::from(self.ring) {
            return false;
        }
        let slot = (tail % u64::from(self.ring)) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                cell.as_ptr(),
                self.mirror.add(slot * self.cell_bytes),
                cell.len(),
            );
        }
        // The cell, then the cursor: a consumer seeing the tail advance must
        // see the bytes. Release pairs with the engine's acquire.
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        unsafe { self.words.add(1).write_volatile(tail + 1) };
        true
    }
}

/// Which channels a stage reads and which it puts, as global indices: the four
/// arguments the control kernels take, derived from the program's ops.
/// `LaunchOp::channel` is a local slot; `channel_bindings` maps it to the
/// program-global dense index the rings use.
///
/// A `chan_take` consumes (full before, head advances after); a `chan_read`
/// peeks (full before, no advance); a `chan_put` produces (room before, tail
/// advances after). Conflating read with take drops a value per fire.
///
/// Readiness is first touch: a channel answers one of `need_full`/`need_empty`
/// however often the stage touches it. A loop-carried counter in both sets would
/// demand a ring at once non-empty and non-full, unsatisfiable at capacity 1.
/// `taken` and `put` are unaffected: the commit advances both.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct StageChannels {
    /// Channels that must hold a value: everything read or taken.
    pub need_full: Vec<u32>,
    /// Channels that must have room: everything put.
    pub need_empty: Vec<u32>,
    /// Channels whose head advances on commit: only what was TAKEN.
    pub taken: Vec<u32>,
    /// Channels whose tail advances on commit: everything put.
    pub put: Vec<u32>,
}

/// See [`StageChannels`].
///
/// # Errors
///
/// If an op names a local slot the bindings do not cover — a plan whose ops and
/// bindings disagree, refused rather than skipped.
pub fn stage_channels(
    plan: &driver::driver_api::plan::LaunchStagePlan,
) -> std::result::Result<StageChannels, String> {
    use driver::tensor_ir::op::tags;

    let mut out = StageChannels::default();
    let push = |v: &mut Vec<u32>, c: u32| {
        if !v.contains(&c) {
            v.push(c);
        }
    };
    // Channels that have already answered the readiness question (first touch).
    let mut gated: Vec<u32> = Vec::new();
    for op in &plan.ops {
        let local = op.channel;
        if local == u32::MAX {
            continue;
        }
        let Some(&global) = plan.channel_bindings.get(local as usize) else {
            return Err(format!(
                "op {:#04x} names local channel slot {local} and the plan binds {}",
                op.code,
                plan.channel_bindings.len()
            ));
        };
        // `tags::*` are `u8` and `code` is a `u16`; widen the tag rather than
        // narrow `code` to keep the comparison total.
        let first_touch = !gated.contains(&global);
        match op.code {
            c if c == u16::from(tags::CHAN_TAKE) => {
                if first_touch {
                    push(&mut out.need_full, global);
                }
                push(&mut out.taken, global);
            }
            c if c == u16::from(tags::CHAN_READ) => {
                if first_touch {
                    push(&mut out.need_full, global);
                }
            }
            c if c == u16::from(tags::CHAN_PUT) => {
                if first_touch {
                    push(&mut out.need_empty, global);
                }
                push(&mut out.put, global);
            }
            // A non-channel op touches nothing, so it claims no first touch.
            _ => continue,
        }
        gated.push(global);
    }
    Ok(out)
}

#[cfg(test)]
mod tests_2 {
    use super::*;

    use driver::driver_api::plan::{LaunchOp, LaunchStagePlan};
    use driver::tensor_ir::op::tags;

    fn op(code: u8, channel: u32) -> LaunchOp {
        LaunchOp {
            code: u16::from(code),
            channel,
            ..LaunchOp::default()
        }
    }

    /// A take consumes and a read does not; conflating them drops a value per fire.
    #[test]
    fn a_read_needs_its_value_and_does_not_consume_it() {
        let plan = LaunchStagePlan {
            ops: vec![op(tags::CHAN_READ, 0), op(tags::CHAN_PUT, 1)],
            channel_bindings: vec![0, 1],
            ..LaunchStagePlan::default()
        };
        let got = stage_channels(&plan).expect("bindings cover the ops");
        assert_eq!(got.need_full, vec![0], "the read needs a value");
        assert_eq!(got.taken, Vec::<u32>::new(), "and does not consume it");
        assert_eq!(got.need_empty, vec![1], "the put needs room");
        assert_eq!(got.put, vec![1], "and advances a tail");

        let plan = LaunchStagePlan {
            ops: vec![op(tags::CHAN_TAKE, 0)],
            channel_bindings: vec![0],
            ..LaunchStagePlan::default()
        };
        let got = stage_channels(&plan).expect("bindings cover the ops");
        assert_eq!(got.need_full, vec![0]);
        assert_eq!(got.taken, vec![0], "a take DOES consume");
    }

    /// The op's channel is a local slot, not the global index the rings use.
    #[test]
    fn a_local_slot_is_not_the_global_channel_index() {
        let plan = LaunchStagePlan {
            ops: vec![op(tags::CHAN_TAKE, 0), op(tags::CHAN_PUT, 1)],
            // This stage's two slots are the program's channels 5 and 2.
            channel_bindings: vec![5, 2],
            ..LaunchStagePlan::default()
        };
        let got = stage_channels(&plan).expect("bindings cover the ops");
        assert_eq!(got.need_full, vec![5], "local 0 is global 5");
        assert_eq!(got.taken, vec![5]);
        assert_eq!(got.need_empty, vec![2], "local 1 is global 2");
        assert_eq!(got.put, vec![2]);
    }

    /// A channel a stage both takes and puts gates on its first touch only: both
    /// sets would demand a ring at once non-empty and non-full. Commit advances both.
    #[test]
    fn a_channel_taken_and_put_gates_on_its_first_touch_only() {
        let plan = LaunchStagePlan {
            ops: vec![op(tags::CHAN_TAKE, 0), op(tags::CHAN_PUT, 0)],
            channel_bindings: vec![4],
            ..LaunchStagePlan::default()
        };
        let got = stage_channels(&plan).expect("bindings cover the ops");
        assert_eq!(got.need_full, vec![4], "the take is the first touch");
        assert_eq!(
            got.need_empty,
            Vec::<u32>::new(),
            "and the put does not also demand room: no ring is both"
        );
        assert_eq!(got.taken, vec![4], "the commit still consumes");
        assert_eq!(got.put, vec![4], "and still publishes");

        // The other order gives the other answer — a rule about the ops.
        let plan = LaunchStagePlan {
            ops: vec![op(tags::CHAN_PUT, 0), op(tags::CHAN_TAKE, 0)],
            channel_bindings: vec![4],
            ..LaunchStagePlan::default()
        };
        let got = stage_channels(&plan).expect("bindings cover the ops");
        assert_eq!(got.need_empty, vec![4], "the put is the first touch");
        assert_eq!(got.need_full, Vec::<u32>::new());
    }

    /// One channel named twice appears once; a repeat would double-advance a cursor.
    #[test]
    fn a_channel_touched_twice_is_listed_once() {
        let plan = LaunchStagePlan {
            ops: vec![
                op(tags::CHAN_READ, 0),
                op(tags::CHAN_READ, 0),
                op(tags::CHAN_PUT, 1),
                op(tags::CHAN_PUT, 1),
            ],
            channel_bindings: vec![0, 1],
            ..LaunchStagePlan::default()
        };
        let got = stage_channels(&plan).expect("bindings cover the ops");
        assert_eq!(got.need_full, vec![0]);
        assert_eq!(got.put, vec![1]);
    }

    /// An op touching no channel says so with `u32::MAX` and is not looked up.
    #[test]
    fn a_channelless_op_is_skipped_rather_than_looked_up() {
        let plan = LaunchStagePlan {
            ops: vec![op(0x10, u32::MAX), op(tags::CHAN_PUT, 0)],
            channel_bindings: vec![7],
            ..LaunchStagePlan::default()
        };
        let got = stage_channels(&plan).expect("the arithmetic op names no channel");
        assert_eq!(got.put, vec![7]);
    }

    /// A plan whose ops and bindings disagree is refused, not skipped.
    #[test]
    fn an_op_naming_an_unbound_slot_refuses() {
        let plan = LaunchStagePlan {
            ops: vec![op(tags::CHAN_PUT, 3)],
            channel_bindings: vec![0, 1],
            ..LaunchStagePlan::default()
        };
        assert!(stage_channels(&plan).is_err());
    }

    /// A host stand-in for the pinned mirror, to test ring semantics without a GPU.
    struct Plane {
        mirror: Vec<u8>,
        words: Vec<u64>,
        cell_bytes: usize,
        ring: u32,
    }

    impl Plane {
        fn new(cell_bytes: usize, ring: u32) -> Self {
            Self {
                mirror: vec![0u8; cell_bytes * ring as usize],
                words: vec![0u64; 4],
                cell_bytes,
                ring,
            }
        }
        /// The plane as a channel the engine both writes and reads — not a real
        /// role, but these cases exercise the ring, not the role.
        fn chan(&mut self) -> HostChannel {
            unsafe {
                HostChannel::new(
                    self.mirror.as_mut_ptr().cast(),
                    self.words.as_mut_ptr().cast(),
                    self.cell_bytes,
                    self.ring,
                    driver::driver_api::local::PIE_CHANNEL_HOST_ROLE_WRITER,
                )
            }
        }
    }

    /// The ring wraps: the third publish reuses slot 0 but is not the first.
    #[test]
    fn the_ring_wraps_without_confusing_a_reused_slot() {
        let mut plane = Plane::new(4, 3);
        let mut c = plane.chan();
        for i in 0..3u32 {
            assert!(c.publish(&i.to_le_bytes()), "publish {i}");
        }
        assert!(!c.publish(&3u32.to_le_bytes()), "a full ring refuses");
        for i in 0..3u32 {
            let got = c.take().expect("published");
            assert_eq!(got, i.to_le_bytes(), "item {i} in order");
        }
        assert!(c.take().is_none(), "an empty ring yields nothing");
        // Accepts again once drained: the refusal above was fullness, not poison.
        assert!(c.publish(&9u32.to_le_bytes()));
        assert_eq!(c.take().expect("published"), 9u32.to_le_bytes());
    }

    /// Both cursors free-run, so an empty ring (`head == tail`) is not a full one.
    #[test]
    fn an_empty_ring_and_a_full_one_are_distinguishable() {
        let mut plane = Plane::new(4, 3);
        let mut c = plane.chan();
        assert_eq!(c.depth(), 0);
        for i in 0..3u32 {
            assert!(c.publish(&i.to_le_bytes()));
        }
        assert_eq!(c.depth(), 3, "full");
        assert_eq!(plane.words[0], 0, "head has not moved");
        assert_eq!(plane.words[1], 3, "tail counted every publish");
    }

    /// A cell of the wrong width is refused rather than truncated.
    #[test]
    fn a_publish_of_the_wrong_width_is_refused() {
        let mut plane = Plane::new(4, 2);
        let mut c = plane.chan();
        assert!(!c.publish(&[0u8; 3]));
        assert!(!c.publish(&[0u8; 5]));
        assert_eq!(c.depth(), 0, "a refused publish advances nothing");
    }

    /// A bool cell survives the wire and the ring — the fire's round trip, minus the device.
    #[test]
    fn a_bool_cell_survives_the_wire_and_the_ring_together() {
        let numel = 17;
        let native: Vec<u8> = (0..numel).map(|i| u8::from(i % 3 == 0)).collect();
        let mut plane = Plane::new(wire_cell_bytes(DType::Bool, numel), 2);
        let mut c = plane.chan();

        let wire = native_to_wire(DType::Bool, numel, &native).expect("to wire");
        assert!(c.publish(&wire), "the wire cell fits the mirror");

        let back_wire = c.take().expect("published");
        let back = wire_to_native(DType::Bool, numel, &back_wire).expect("to native");
        assert_eq!(back, native, "seventeen lanes through both planes");
    }

    /// The four dtypes at a ragged 17-lane width: the last wire byte has one live bit.
    #[test]
    fn a_cell_survives_the_round_trip_at_a_ragged_width() {
        for dtype in [DType::F32, DType::I32, DType::U32, DType::Bool] {
            let numel = 17;
            let native: Vec<u8> = (0..native_cell_bytes(dtype, numel))
                .map(|i| {
                    if dtype == DType::Bool {
                        u8::from(i % 3 == 0)
                    } else {
                        (i % 251) as u8
                    }
                })
                .collect();
            let wire = native_to_wire(dtype, numel, &native).expect("to wire");
            assert_eq!(
                wire.len(),
                wire_cell_bytes(dtype, numel),
                "{} wire width",
                dtype.name()
            );
            let back = wire_to_native(dtype, numel, &wire).expect("back");
            assert_eq!(back, native, "{} round trip", dtype.name());
        }
    }

    /// Only bools differ between the planes; every other dtype matches.
    #[test]
    fn the_two_planes_disagree_only_about_bools() {
        for numel in [1usize, 7, 8, 9, 64, 65] {
            for dtype in [DType::F32, DType::I32, DType::U32] {
                assert_eq!(
                    wire_cell_bytes(dtype, numel),
                    native_cell_bytes(dtype, numel),
                    "{} at {numel} lanes must be the same on both planes",
                    dtype.name()
                );
            }
            assert_eq!(native_cell_bytes(DType::Bool, numel), numel);
            assert_eq!(wire_cell_bytes(DType::Bool, numel), numel.div_ceil(8));
        }
    }

    /// Padding bits past `numel` stay clear, or the engine reports a lane never written.
    #[test]
    fn packing_leaves_no_bit_set_past_the_last_lane() {
        let numel = 9;
        let native = vec![1u8; numel];
        let wire = native_to_wire(DType::Bool, numel, &native).expect("to wire");
        assert_eq!(wire.len(), 2);
        assert_eq!(wire[0], 0xff);
        assert_eq!(wire[1], 0b0000_0001, "lanes 9..16 are not lanes");
    }

    /// Any nonzero native byte is a set lane.
    #[test]
    fn a_mask_byte_is_true_and_not_merely_one() {
        let native = [0u8, 1, 0xff, 0x80, 0, 0, 0, 0];
        let wire = native_to_wire(DType::Bool, 8, &native).expect("to wire");
        assert_eq!(wire[0], 0b0000_1110);
    }

    /// A short cell is refused rather than padded.
    #[test]
    fn a_cell_of_the_wrong_width_is_refused() {
        assert!(wire_to_native(DType::F32, 4, &[0u8; 12]).is_err());
        assert!(native_to_wire(DType::F32, 4, &[0u8; 12]).is_err());
        assert!(wire_to_native(DType::Bool, 17, &[0u8; 2]).is_err());
        assert!(native_to_wire(DType::Bool, 17, &[0u8; 16]).is_err());
    }
}
