//! The device-side channel ring: cells the kernels read and write, and the
//! four cursors the control kernels advance.
//!
//! # Two shapes for one channel, and why
//!
//! A channel's cell exists twice, in two different encodings, and confusing
//! them is the defect this module is written against:
//!
//! * **native**, on the device: one byte per bool lane, four per anything
//!   else. A kernel indexes a bool cell lane by lane, so it cannot be packed.
//! * **wire**, on the host: bool packs eight lanes to a byte
//!   ([`wire_cell_bytes`](driver::wire_cell_bytes)); everything else
//!   is still four bytes per lane.
//!
//! Every non-bool channel therefore has the same size in both, which is
//! exactly why a port can carry the confusion for a long time and only meet it
//! on the first bool channel — where a 128-lane mask is 128 bytes on the
//! device and 16 on the wire, and reading one as the other yields a mask that
//! is one eighth right.
//!
//! # The ring, and the sentinel cell
//!
//! `capacity + 1` slots, and the extra one is load-bearing: full and empty are
//! otherwise the same state (`head == tail`) and cannot be told apart without
//! a separate count. With the sentinel, empty is `head == tail` and full is
//! `(tail + 1) % cap1 == head`, so a capacity-N channel holds at most N
//! unconsumed items and the cursors alone say which.
//!
//! # What lives where
//!
//! | array | element | length | indexed |
//! |---|---|---|---|
//! | cells | native bytes | `cap1 * native_bytes` per channel | `base + slot * native_bytes` |
//! | `full` | `u8` | `channels * MAX_RING` | `full[c * MAX_RING + slot]` |
//! | `head` | `u32` | `channels` | `head[c]` |
//! | `tail` | `u32` | `channels` | `tail[c]` |
//! | `cap1` | `u32` | `channels` | `cap1[c]` |
//!
//! `full` is indexed with [`MAX_RING`] as its row pitch **whatever a channel's
//! capacity is**. Using `cap1` as the pitch would be the natural-looking
//! mistake and would make every channel past the first read its neighbour's
//! flags.
//!
//! # Everything is zeroed, deliberately
//!
//! A fresh CUDA allocation is not promised zero, and garbage cursors mean a
//! ring that is already mid-sequence, already full, or already poisoned. The
//! Metal port's `Ring::new` records the same requirement for the same reason.

use driver::tensor_ir::DType;

use crate::device::{Allocator, DeviceBuffer, StreamRef};
use crate::error::{Error, Result};

use super::run::MAX_RING;

/// Native device bytes for a cell of `numel` lanes of `dtype`.
///
/// One byte per bool lane rather than one bit: a kernel indexes a bool cell
/// lane by lane. The wire form packs, and
/// [`wire_cell_bytes`](driver::wire_cell_bytes) is that one.
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
    /// If the ring would exceed [`MAX_RING`], which is the pitch the `full`
    /// array is indexed with — a longer ring would write into the next
    /// channel's flags.
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

/// Every channel THE DRIVER HOLDS: the cells and the four cursor arrays,
/// indexed by a global channel SLOT.
///
/// # Why this is one registry and not one per instance
///
/// It used to be per instance, sized from that instance's own `channel_ids`,
/// and [`ChannelState::is_extern`](crate::serve::state::ChannelState::is_extern)
/// recorded what that cost: two instances naming one channel got two rings, so
/// the exporter filled its own and the importer read a zeroed cell and called
/// it a value. That is exactly the shape of the bench's decode loop — the
/// prefill's epilogue puts the sampled token on a channel the DECODE instance
/// binds to `EmbedTokens` — so a chained frame could not be served at all.
///
/// So a channel has ONE ring, wherever it is named from, and an instance is a
/// MAP from its dense channel index to the slot that ring lives at
/// ([`Session`](super::session::Session)). The four cursor arrays stay flat and
/// contiguous, because the control kernels take them as four pointers and index
/// them by the channel numbers they are handed — which are now global slots.
///
/// # The cells are per slot, and that is what makes growth cheap
///
/// Channels are registered one at a time over a serving day, so the registry
/// grows. One allocation per slot's cells means growth adds an allocation and
/// touches nothing that already exists; a single block with offsets would have
/// to move every live cell to make room. The four cursor arrays DO have to be
/// contiguous, so those are reallocated and copied — they are
/// `MAX_RING + 12` bytes per slot, which is why that is affordable and the
/// cells are not.
#[derive(Debug)]
pub struct Rings {
    cells: Vec<DeviceBuffer>,
    full: DeviceBuffer,
    head: DeviceBuffer,
    tail: DeviceBuffer,
    cap1: DeviceBuffer,
    shapes: Vec<ChannelShape>,
    /// Slots the four cursor arrays have room for. Never below
    /// `shapes.len()`.
    reserved: usize,
}

impl Rings {
    /// Allocate and zero a registry holding `shapes`, at slots `0..shapes.len()`.
    ///
    /// # Errors
    ///
    /// If a ring is longer than [`MAX_RING`], if a cell is empty, or if the
    /// device refuses the allocation.
    pub fn new(alloc: &Allocator, shapes: &[ChannelShape], stream: StreamRef<'_>) -> Result<Self> {
        // Zeroed, all four. A fresh allocation is not promised zero, and a
        // garbage cursor is a ring that is already full, already mid-sequence,
        // or already holding a value nothing wrote.
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
    /// Slots are handed out in registration order and NEVER REUSED, because a
    /// slot is the identity a `Session`'s dense map points at and a reused one
    /// would hand a new channel the old channel's cursors while some session
    /// still named it.
    ///
    /// # What that costs, and it is not yet paid
    ///
    /// A closed channel's slot is not reclaimed, so a serving day of N
    /// requests leaves N × (its channels) rings allocated — the cells, plus
    /// `MAX_RING + 12` bytes of cursor array each. Nothing frees them.
    /// Reclaiming needs a free list AND a way to know no live session still
    /// holds the slot, which is the same liveness question `close_instance`
    /// answers for sessions and nothing answers for channels. Measured on the
    /// bench: 18 channels per request, so 16 requests reach slot 288 — small,
    /// and linear in requests served, which is the shape that eventually is
    /// not.
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
        // `cap1` is the one array with contents rather than zeros: it is the
        // modulus every cursor advance divides by, and a zero there is a
        // division by zero inside the commit kernel.
        self.cap1
            .write_at(slot * size_of::<u32>(), &ring.to_le_bytes(), stream)?;
        u32::try_from(slot)
            .map_err(|_| Error::invalid("program::channel", "more channels than a u32 counts"))
    }

    /// Make the four cursor arrays hold at least `want` slots, preserving what
    /// the live ones already say.
    ///
    /// Doubling rather than exact, because every registration would otherwise
    /// reallocate and copy all four.
    fn reserve(&mut self, alloc: &Allocator, want: usize, stream: StreamRef<'_>) -> Result<()> {
        if want <= self.reserved {
            return Ok(());
        }
        let grown = want.max(self.reserved * 2).max(8);
        // Read the live prefix back before the swap. The cursors live on the
        // DEVICE — the control kernels advance them — so a host-side copy kept
        // alongside would be a guess that is right until the first fire that
        // commits.
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

    /// Whether the registry holds nothing yet, which is its state before the
    /// first channel is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// Channel `c`'s geometry.
    #[must_use]
    pub fn shape(&self, c: usize) -> Option<ChannelShape> {
        self.shapes.get(c).copied()
    }

    /// The device address of channel `c`'s cell at ring slot `slot`.
    ///
    /// This is what a [`LaneChannelSlot`](driver::LaneChannelSlot)'s
    /// `committed_cell` and `pending_cell` are: absolute device addresses,
    /// resolved on the host, because the kernel is handed pointers and does no
    /// ring arithmetic of its own.
    ///
    /// # Errors
    ///
    /// If `c` is not a channel. `slot` is reduced modulo the ring rather than
    /// refused — a cursor is monotonic and the ring position is its residue,
    /// which is the same arithmetic the C++ does at the same point.
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

    /// Write `bytes` into channel `c`'s cell at `slot` and mark it full.
    ///
    /// This is how a seed reaches the device — the value an instance is bound
    /// with, which every later fire reads. It writes the NATIVE form, so a
    /// caller holding a wire cell must unpack first.
    ///
    /// # Errors
    ///
    /// If the channel does not exist or `bytes` is not exactly one native
    /// cell. A short write would leave the tail of the cell holding whatever
    /// the zeroing left, which is a value the program would read as real.
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

        // The flag, then the cursor. A cell that is full but whose tail has
        // not advanced is merely not yet published; a tail that has advanced
        // past a cell whose flag is clear is a reader blocked forever on a
        // value that is already there.
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
    ///
    /// # Errors
    ///
    /// If the channel does not exist, or the copy fails.
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
    /// `head`, exactly as the commit kernel's `taken` loop does.
    ///
    /// # Why the host does this and not the commit kernel
    ///
    /// A DESCRIPTOR port is not a stage op. `Port::consumes` names the four
    /// that take — `EmbedTokens`, `Positions`, `WSlot`, `WOff` — and no
    /// `LaunchOp` in any stage mentions them, so `stage_channels` cannot see
    /// them and the commit kernel is never handed them. Left unconsumed their
    /// heads never move while their tails do, and after `capacity` fires the
    /// ring is full: the epilogue's own `chan_put` then fails readiness and the
    /// decode loop stops producing, silently, some tens of tokens in.
    ///
    /// It is a host write because the driver has already synchronized by the
    /// time it resolves a step's descriptors — the previous step's
    /// `Session::fire` ends on `stream.synchronize()` — so there is no kernel
    /// ordering left to respect, and a launch would buy nothing.
    ///
    /// # Errors
    ///
    /// If the channel does not exist, or a copy fails.
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
        // The flag, THEN the cursor — the same order the commit kernel writes
        // them in, so a reader that catches the pair mid-update sees a cell
        // that is not yet consumed rather than one that is consumed twice.
        self.full
            .write_at(c * MAX_RING as usize + head as usize, &[0u8], stream)?;
        let next = (head + 1) % ring;
        self.head
            .write_at(c * size_of::<u32>(), &next.to_le_bytes(), stream)?;
        stream.synchronize()?;
        Ok(())
    }

    /// The four cursors of every channel, as the host sees them.
    ///
    /// Read back rather than tracked, because the control kernels advance them
    /// on the device: a host-side copy would be a guess that is right until
    /// the first fire that commits.
    ///
    /// # Errors
    ///
    /// If a copy fails.
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

    /// A bool cell is one byte per lane on the device and one BIT per lane on
    /// the wire. Every other dtype is the same size in both, which is why the
    /// two can be confused for a long time and then be wrong by a factor of
    /// eight on the first mask.
    #[test]
    fn a_bool_cell_is_bytes_on_the_device_and_bits_on_the_wire() {
        assert_eq!(native_cell_bytes(DType::Bool, 128), 128);
        assert_eq!(driver::wire_cell_bytes(DType::Bool, 128), 16);
        assert_eq!(native_cell_bytes(DType::F32, 128), 512);
        assert_eq!(driver::wire_cell_bytes(DType::F32, 128), 512);
    }

    /// The ring is one longer than the capacity, and the extra slot is what
    /// makes full and empty distinguishable without a count.
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

    /// A ring longer than the `full` array's pitch would write into the next
    /// channel's flags, so it is refused at the geometry rather than at the
    /// launch, where the symptom would be another channel's readiness.
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

    /// Depth is the cursor difference modulo the ring, which is what makes the
    /// sentinel work: a full ring reads as `capacity`, never as zero.
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

// The two channel planes, and the copy between them.
//
// A channel exists twice. `pie_cuda_register_channel` allocates PINNED
// HOST rings — a mirror of `capacity + 1` cells and four `u64` control
// words — and hands their addresses to the engine, which polls them
// directly. [`crate::program::channel::Rings`] allocates DEVICE rings, which is
// what a compiled program's kernels read and write. Nothing joins them,
// which is why `ptir_programs` has a writer and no reader: a program
// could be compiled and could not be handed its inputs.
//
// This is the join, and it is a COPY rather than a mapping. The two
// planes are different memory by construction — the engine must be able
// to poll without a device round-trip, and a kernel must be able to read
// without a host one — so a fire pulls its inputs across before the
// stages and pushes its outputs back after.
//
// # The bool encoding is the whole subtlety
//
// `ptir/mod.rs` names it and this is where it lands: *"Native bytes on
// the device, bit-packed on the wire, and the difference is invisible
// until the first bool channel."* A `[64]` bool cell is 64 bytes in a
// device ring and 8 bytes in the host mirror. Every other dtype is four
// bytes per lane on both sides, so a bridge that forgot this would be
// right for f32, i32 and u32 — which is to say, right until the first
// program that compares anything.
//
// The packing is LSB-first within each byte, which is what
// `register_channel` sizes for (`numel.div_ceil(8)`) and what the engine
// reads. Neither side stores anything in the padding bits of the last
// byte, so the round trip is exact rather than merely lossless.

/// Bytes one cell occupies in the HOST mirror, for `numel` lanes.
///
/// The counterpart of [`native_cell_bytes`], and deliberately in the same
/// vocabulary: given a shape, the two functions are the only difference
/// between the planes.
#[must_use]
pub fn wire_cell_bytes(dtype: DType, numel: usize) -> usize {
    if dtype == DType::Bool {
        numel.div_ceil(8)
    } else {
        numel * 4
    }
}

/// A wire cell, as the device wants it.
///
/// # Errors
///
/// If `wire` is not exactly one wire cell. A short cell is not a smaller
/// value — it is a value whose tail the caller would read out of whatever
/// the allocation last held.
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

/// A native cell, as the wire wants it.
///
/// Any nonzero byte is `true`, because the device writes whatever its
/// comparison produced and only promises nonzero-means-set. Reading it as
/// `== 1` would drop a `0xff`, which is what a lane that stored a mask
/// byte rather than a flag would hold.
///
/// # Errors
///
/// If `native` is not exactly one native cell.
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

/// One channel's HOST plane: the pinned mirror the engine polls, and the
/// four control words beside it.
///
/// A view, not an owner. `register_channel` allocates both and hands
/// their addresses to the engine; this borrows them for the length of a
/// fire. The layout is that entry point's and is restated here because
/// the two have to agree byte for byte and only one of them can be read
/// by the compiler:
///
/// | word | meaning |
/// |---|---|
/// | 0 | `head` — what the consumer has taken |
/// | 1 | `tail` — what the producer has published |
/// | 2 | `poison` |
/// | 3 | `closed` |
///
/// Both cursors are FREE-RUNNING `u64`s and the slot is `cursor % ring`,
/// which is why an empty channel and a full one are distinguishable at
/// all — the alternative, wrapping the cursors themselves, makes
/// `head == tail` mean both.
pub struct HostChannel {
    mirror: *mut u8,
    words: *mut u64,
    /// Wire bytes per cell.
    pub cell_bytes: usize,
    /// `capacity + 1`.
    pub ring: u32,
    /// `PIE_CHANNEL_HOST_ROLE_*`: which side of this mirror the ENGINE is on.
    ///
    /// The bridge is DIRECTIONAL and the mirror is not: one head/tail pair
    /// serves both, so a driver that both publishes into a plane and takes
    /// from it reads back its own writes. On a loop-carried channel — the
    /// epilogue puts the next token, `push` mirrors it, `pull` takes it — that
    /// is one extra cell in the device ring per fire, which fills the ring and
    /// then reads the fire-after-next's value as this fire's.
    ///
    /// So the role decides: the driver TAKES only from a plane the engine
    /// writes, and PUBLISHES only into one the engine reads. A device-only
    /// channel (`NONE`) is bridged in neither direction, which is exactly what
    /// it means — both ends of it are programs.
    pub role: u8,
}

impl HostChannel {
    /// Borrow the host plane of a registered channel.
    ///
    /// # Safety
    ///
    /// `mirror` must point at `cell_bytes * ring` writable bytes and
    /// `words` at four `u64`s, both live for `'_`. Those are exactly what
    /// `pie_cuda_register_channel` allocates and never resizes.
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

    /// Does the ENGINE write this plane? Only then may the driver take from it.
    /// See [`HostChannel::role`].
    #[must_use]
    pub const fn engine_writes(&self) -> bool {
        self.role == driver::driver_api::local::PIE_CHANNEL_HOST_ROLE_WRITER
    }

    /// Does the ENGINE read this plane? Only then may the driver publish into
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

    /// Take the oldest published cell, advancing `head`.
    ///
    /// This is the ENGINE's side of the ring being read by the driver,
    /// which is the right way round for an input channel: the engine
    /// published a value and the program is the consumer.
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
        // The read, THEN the cursor. An acquire fence would be the
        // symmetric partner of `publish`'s release; the copy above is
        // what it orders.
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        unsafe { self.words.add(0).write_volatile(head + 1) };
        Some(cell)
    }

    /// Publish a wire cell, advancing `tail`.
    ///
    /// `false` when the ring is full — the engine has not consumed — which
    /// is a dropped output rather than an error, because the alternative
    /// is a fire that blocks on a reader.
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
        // The cell, THEN the cursor: a consumer that saw the tail advance
        // must see the bytes. Release pairs with the engine's acquire.
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        unsafe { self.words.add(1).write_volatile(tail + 1) };
        true
    }
}

/// Which channels a stage READS and which it PUTS, as global indices.
///
/// The four arguments the control kernels take —
/// `launch_control::readiness`'s `need_full` and `need_empty`, and
/// `commit`'s `taken` and `put` — and the reason they are derived rather
/// than configured: a program states them in its ops, so a driver that
/// asked its caller would be asking about something it can already read.
///
/// # The two vocabularies
///
/// `LaunchOp::channel` is a LOCAL slot; `LaunchStagePlan::channel_bindings`
/// maps it to the program-global dense index the rings are laid out by.
/// The gap between them is the one thing this function exists to close,
/// and it is exactly the gap that would be invisible in a single-channel
/// program — every local slot is its own global index when there is one
/// of each.
///
/// # What each set means
///
/// A `chan_take` CONSUMES, so its channel must hold a value before the
/// fire and its head advances after. A `chan_read` peeks without
/// consuming: it needs the value present and must NOT advance. A
/// `chan_put` produces, so its channel needs room before and its tail
/// advances after. Conflating read with take is a head that runs away
/// from a reader that never asked to consume — which is a value dropped
/// on the floor, one per fire, and looks like a slow leak rather than a
/// bug.
///
/// # Readiness is FIRST TOUCH, and the loop-carried channel is why
///
/// `need_full` and `need_empty` are the readiness question, and a channel
/// answers ONE of them however many times the stage touches it. The rule is
/// `driver::channel_effects`'s, said there in as many words: *"First touch,
/// as shipped: not `take || read` against `put`. An in-place channel is in
/// both sets, and gating on both would ask for a ring that is at once
/// non-empty and non-full."*
///
/// A loop-carried counter is exactly that channel — `let n = kv_len.take();
/// kv_len.put(&(n + 1))`, which is the shape every decode epilogue in the
/// tree has — and this used to list it in both. On a capacity-1 ring the two
/// demands are unsatisfiable at once: holding the one value it was given
/// makes it non-empty, which is the take's requirement and the put's
/// refusal. So EVERY decode epilogue answered `NotReady`, forever, and the
/// fire fell through to raw logits with nothing to say it had.
///
/// `taken` and `put` are unaffected: those are what the commit advances, and
/// a channel both taken and put advances BOTH — which is why the commit
/// kernel publishes before it consumes.
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
/// If an op names a local slot the plan's bindings do not cover, which is
/// a plan whose ops and bindings disagree — refused rather than skipped,
/// because a missing binding is not a channel the fire may ignore.
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
    // Which channels have already answered the readiness question. See the
    // FIRST TOUCH paragraph above: a channel answers it once.
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
        // `tags::*` are `u8` and `LaunchOp::code` is a `u16`, because the
        // plan's field is sized for the whole opcode space and a tag is a
        // byte of it. Widening here rather than narrowing `code` keeps the
        // comparison total.
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
            // An op that is not a channel op touches no channel, whatever it
            // names, so it must not claim the first touch either.
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

    /// A take consumes and a read does not, which is the distinction the
    /// commit acts on.
    ///
    /// Both need their channel full, and only the take advances its head.
    /// Conflating them advances a head no reader asked to move — one value
    /// dropped per fire, which reads as a slow leak rather than a bug.
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

    /// The op's channel is a LOCAL slot and the rings are laid out by the
    /// GLOBAL index.
    ///
    /// Invisible in a one-channel program — every local slot is its own
    /// global index when there is one of each — which is exactly why it is
    /// worth a case where the two disagree. A fire that passed the local
    /// slot would ask the control kernels about someone else's channel.
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

    /// A channel a stage BOTH takes and puts answers ONE readiness question:
    /// the direction of its first touch.
    ///
    /// `let n = kv_len.take(); kv_len.put(&(n + 1))` is every decode
    /// epilogue in the tree, and listing it in both sets asks for a ring that
    /// is at once non-empty and non-full. On the capacity-1 ring the engine
    /// declares for it, no state satisfies both — so the fire answered
    /// `NotReady` forever and the decode loop produced nothing, with the
    /// driver reporting only that it had waited.
    ///
    /// `taken` and `put` are unaffected: the commit advances BOTH, which is
    /// why the commit kernel publishes before it consumes.
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

        // The other order is the other answer, which is what makes this a
        // rule about the ops rather than a preference for takes.
        let plan = LaunchStagePlan {
            ops: vec![op(tags::CHAN_PUT, 0), op(tags::CHAN_TAKE, 0)],
            channel_bindings: vec![4],
            ..LaunchStagePlan::default()
        };
        let got = stage_channels(&plan).expect("bindings cover the ops");
        assert_eq!(got.need_empty, vec![4], "the put is the first touch");
        assert_eq!(got.need_full, Vec::<u32>::new());
    }

    /// One channel named twice appears once.
    ///
    /// The control kernels take a set; a repeated index would advance a
    /// cursor twice for one fire.
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

    /// An op that touches no channel says so with `u32::MAX`, and is not a
    /// binding lookup.
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

    /// A plan whose ops and bindings disagree is refused.
    ///
    /// Not skipped: a slot with no binding is not a channel the fire may
    /// ignore, it is a plan that cannot be laid out.
    #[test]
    fn an_op_naming_an_unbound_slot_refuses() {
        let plan = LaunchStagePlan {
            ops: vec![op(tags::CHAN_PUT, 3)],
            channel_bindings: vec![0, 1],
            ..LaunchStagePlan::default()
        };
        assert!(stage_channels(&plan).is_err());
    }

    /// A host-allocated stand-in for the pinned mirror, so the ring
    /// semantics can be tested without a GPU or a registered channel.
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
        /// The plane as a channel the engine both writes and reads.
        ///
        /// Not a role any real registration carries — a channel is one or the
        /// other — but these cases are about the RING, and a stand-in that
        /// refused half its own operations could not exercise it.
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

    /// Published items come back in order and the ring wraps.
    ///
    /// Three cells through a two-capacity ring, which is the case that
    /// distinguishes a free-running cursor from a wrapped one: the third
    /// publish reuses slot 0 and must not be confused with the first.
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
        // And it accepts again once drained, which is what says the
        // refusal above was fullness and not a poisoned cursor.
        assert!(c.publish(&9u32.to_le_bytes()));
        assert_eq!(c.take().expect("published"), 9u32.to_le_bytes());
    }

    /// `depth` is the unconsumed count, and an empty ring is not a full
    /// one.
    ///
    /// The reason both cursors free-run: wrapping them makes
    /// `head == tail` mean empty AND full, which is the classic ring bug
    /// and is unobservable until a fire fills one exactly.
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

    /// The two halves compose: a bool cell published on the wire comes
    /// back as the native bytes the device wrote.
    ///
    /// This is the round trip the fire performs, minus the device — and
    /// the one that would have been silently wrong for every bool channel
    /// had the bridge copied bytes straight through.
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

    /// The four dtypes, at a width that is not a multiple of eight.
    ///
    /// Seventeen lanes rather than sixteen on purpose: the last wire byte
    /// then holds one live bit and seven that must stay clear, which is
    /// the case a `numel / 8` would round away and a `div_ceil` gets
    /// right only if nothing writes past lane 16.
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

    /// A bool cell is EIGHT times narrower on the wire, which is the
    /// difference the rest of this module exists for.
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

    /// The padding bits of the last wire byte stay clear.
    ///
    /// Not cosmetic: the engine reads the mirror with its own unpacker,
    /// and a set bit past `numel` is a lane it would report that the
    /// program never wrote.
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
