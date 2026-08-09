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

use crate::cuda::{Allocator, DeviceBuffer, StreamRef};
use crate::error::{Error, Result};

use super::control::MAX_RING;

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
                "ptir::ring",
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

/// Every channel of one instance: the cells and the four cursor arrays.
///
/// One allocation per array rather than per channel. The C++ does the same,
/// and the reason is not tidiness: the four cursor arrays are passed to the
/// control kernels as four pointers, so they have to be contiguous per
/// channel index whatever a channel's own size is.
#[derive(Debug)]
pub struct Rings {
    cells: DeviceBuffer,
    full: DeviceBuffer,
    head: DeviceBuffer,
    tail: DeviceBuffer,
    cap1: DeviceBuffer,
    /// Byte offset of each channel's ring within `cells`.
    offsets: Vec<usize>,
    shapes: Vec<ChannelShape>,
}

impl Rings {
    /// Allocate and zero the rings for `shapes`, in channel order.
    ///
    /// # Errors
    ///
    /// If a ring is longer than [`MAX_RING`], if a cell is empty, or if the
    /// device refuses the allocation.
    pub fn new(alloc: &Allocator, shapes: &[ChannelShape], stream: StreamRef<'_>) -> Result<Self> {
        let count = shapes.len();
        if count == 0 {
            return Err(Error::invalid("ptir::ring", "an instance with no channels"));
        }

        let mut offsets = Vec::with_capacity(count);
        let mut total = 0usize;
        for shape in shapes {
            let ring = shape.ring()?;
            let cell = shape.cell_bytes();
            if cell == 0 {
                return Err(Error::invalid(
                    "ptir::ring",
                    "a channel whose cell is zero bytes holds nothing and can \
                     never be ready",
                ));
            }
            offsets.push(total);
            total = total
                .checked_add(cell * ring as usize)
                .ok_or_else(|| Error::invalid("ptir::ring", "the rings do not fit in memory"))?;
        }

        // Zeroed, all five. A fresh allocation is not promised zero, and a
        // garbage cursor is a ring that is already full, already mid-sequence,
        // or already holding a value nothing wrote.
        let zero = |bytes: usize| -> Result<DeviceBuffer> {
            let mut buffer = alloc.alloc(bytes)?;
            buffer.memset(0, stream)?;
            Ok(buffer)
        };
        let cells = zero(total)?;
        let full = zero(count * MAX_RING as usize)?;
        let head = zero(count * size_of::<u32>())?;
        let tail = zero(count * size_of::<u32>())?;
        let mut cap1 = zero(count * size_of::<u32>())?;

        // `cap1` is the one array with contents rather than zeros: it is the
        // modulus every cursor advance divides by, and a zero there is a
        // division by zero inside the commit kernel.
        let cap1_bytes: Vec<u8> = shapes
            .iter()
            .map(|shape| shape.ring().expect("checked above"))
            .flat_map(u32::to_le_bytes)
            .collect();
        cap1.copy_from_host(&cap1_bytes, stream)?;

        Ok(Self {
            cells,
            full,
            head,
            tail,
            cap1,
            offsets,
            shapes: shapes.to_vec(),
        })
    }

    /// How many channels this instance carries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    /// Whether there are no channels. There never are none — [`Rings::new`]
    /// refuses that — but clippy asks and the answer is honest.
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
            .ok_or_else(|| Error::invalid("ptir::ring", format!("no channel {c}")))?;
        let ring = shape.ring()?;
        let at = self.offsets[c] + (slot % ring) as usize * shape.cell_bytes();
        Ok(self.cells.as_ptr() as u64 + at as u64)
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
            .ok_or_else(|| Error::invalid("ptir::ring", format!("no channel {c}")))?;
        if bytes.len() != shape.cell_bytes() {
            return Err(Error::invalid(
                "ptir::ring",
                format!(
                    "channel {c}'s native cell is {} bytes and {} were offered",
                    shape.cell_bytes(),
                    bytes.len()
                ),
            ));
        }
        let ring = shape.ring()?;
        let at = self.offsets[c] + (slot % ring) as usize * shape.cell_bytes();
        self.cells.write_at(at, bytes, stream)?;

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
            .ok_or_else(|| Error::invalid("ptir::ring", format!("no channel {c}")))?;
        let ring = shape.ring()?;
        let at = self.offsets[c] + (slot % ring) as usize * shape.cell_bytes();
        let mut out = vec![0u8; shape.cell_bytes()];
        self.cells.read_at(at, &mut out, stream)?;
        Ok(out)
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
