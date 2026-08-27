//! One stage, encoded: the device rings a guest's channels live in, the
//! buffers one fire binds, and the dispatch that runs a region.
//!
//! **THIS FILE IS SHORTER THAN ITS CUDA TWIN BY HALF, AND ALMOST ALL OF THE
//! DIFFERENCE IS THE M2 ABI RATHER THAN THE PLATFORM.** The CUDA emitter
//! hands its kernel seven pointers, three scalars and six side tables, and
//! the driver builds a LANE TABLE — a header, a record per lane, a flat
//! array of channel slots holding the committed and pending CELL ADDRESSES —
//! because a CUDA kernel argument is a pointer and a cell is an address
//! inside a ring. The Metal M2 kernel takes the two cells as BUFFER
//! BINDINGS (`committed_{k}` at index `7 + 2k`, `pending_{k}` at `8 + 2k`),
//! and Metal binds a buffer plus an offset — so the cell address IS the
//! binding, and the whole lane table, its host mirror, and the sixteen-byte
//! per-fire patch that keeps them in step all disappear.
//!
//! Three more things go with it. There is no `CudaOpParams`: the emitted
//! `struct M1OpParams` is exactly `driver::OpParams`, 64 bytes, so the
//! shared record is uploaded as it stands and the widening plus its
//! twenty-one `offset_of!` assertions are not written. There is no pending
//! -flag buffer: the M2 kernel keeps `current_{k}` in a register and
//! reassigns it after a put. And there is no bool conversion at the
//! boundary: the runtime packs and unpacks bits on the device (tags
//! `0x90`/`0x91`/`0x92`), so **a Metal channel cell IS a wire cell for every
//! dtype** and `native_cell_bytes`/`wire_to_native`/`native_to_wire` have no
//! counterpart here. That is a real ABI difference, not a simplification: on
//! the CUDA plane a bool ring holds one byte per lane and the host converts;
//! here the ring holds the packed bits the interpreter would.
//!
//! **The verdict is richer, and that is the one place this plane gains.**
//! The CUDA kernel writes one `u32`, started at one and CLEARED to refuse,
//! so a declined fire is a boolean. The M2 kernel writes a 16-byte
//! `driver::Status` — a state, a fault word, two reserved words the fault
//! sites use — so a refusal can be named with `driver::describe_fault`
//! instead of merely reported.

use std::mem::size_of;

use driver::driver_api::program::{LaunchChannel, LaunchStagePlan};
use driver::{Extents, OpParams, OpRuntime, SCRATCH_ALIGN, Status, ValueDesc, describe, layout};
use driver::tensor_ir::DType;
use driver::tensor_ir::op::{IntrinsicId, tags};

use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};

use super::compile::Region;

/// The buffer index the first channel's committed cell binds at. Everything
/// below it is fixed: status, descriptors, params, offsets, scratch,
/// temporary, logits.
///
/// Read off `tensor_compiler::codegen::metal::fused::emit_fused_region`,
/// which writes `7 + channel * 2` and `8 + channel * 2` into the emitted
/// signature. A hand-kept copy is exactly what the emitter's own doc warns
/// about, so this constant exists to be named once and asserted against the
/// emitter in `program_parity` rather than trusted.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
const FIRST_CHANNEL_BUFFER: usize = 7;

/// What a ring cell's offset is rounded up to.
///
/// Sixteen because that is the widest alignment any scalar a cell can hold
/// asks for, and because a bound buffer offset is the one number in this
/// file that a validation layer refuses rather than mis-reads. See
/// [`ChannelShape::cell_stride`].
const CELL_ALIGN: usize = 16;

/// One channel's ring geometry, as the launch package declares it.
///
/// **NO `native` SIDE.** The CUDA twin carries a cell width for the ring and
/// a second one for the wire, because its bool rings are one byte per lane
/// while a wire cell is packed bits. The Metal runtime packs on the device,
/// so there is one width and it is the wire's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelShape {
    /// How many cells the ring holds, not counting the spare.
    pub capacity: u32,
    /// Lanes in one cell.
    pub numel: usize,
    /// The cell's element type.
    pub dtype: DType,
}

impl ChannelShape {
    /// The shape one declared channel states.
    #[must_use]
    pub fn of(declared: &LaunchChannel) -> ChannelShape {
        ChannelShape {
            capacity: declared.capacity.max(1),
            numel: declared
                .shape
                .iter()
                .map(|&dim| dim as usize)
                .product::<usize>()
                .max(1),
            dtype: driver::concrete_dtype(declared.dtype),
        }
    }

    /// Bytes in one cell — the wire cell, which on this plane is the only
    /// cell there is.
    #[must_use]
    pub fn cell_bytes(&self) -> usize {
        driver::wire_cell_bytes(self.dtype, self.numel)
    }

    /// Bytes from one cell to the next inside the ring.
    ///
    /// **NOT `cell_bytes`, AND THE DIFFERENCE IS A BINDING RULE RATHER THAN
    /// A LAYOUT CHOICE.** A cell reaches the kernel as a BUFFER BINDING —
    /// `setBuffer:offset:atIndex:` at the cell's own offset — and a bound
    /// offset must be aligned. A packed `Bool` cell is
    /// `numel.div_ceil(8)` bytes, which is 1 for eight lanes, so a ring laid
    /// out at its wire width would put every second cell at an offset Metal
    /// will not bind. The stride is therefore rounded up to
    /// [`CELL_ALIGN`], and nothing else changes: the kernel is handed ONE
    /// cell at one offset and writes `sink_bytes` of it, so the padding is
    /// invisible to it, and the host's own ring is a separate array whose
    /// slot-for-slot diff is against the CONTENTS rather than the addresses.
    #[must_use]
    pub fn cell_stride(&self) -> usize {
        self.cell_bytes().next_multiple_of(CELL_ALIGN).max(CELL_ALIGN)
    }
}

/// The device rings, one buffer per channel.
///
/// **ONE BUFFER PER CHANNEL RATHER THAN ONE SLAB**, where the CUDA twin
/// carves one allocation: a ring's cell has to be BOUND at its own buffer
/// index, and a binding names a buffer and an offset, so a per-channel
/// buffer keeps every offset inside the channel it belongs to and makes a
/// cell that runs off the end a bounds check rather than a read of the next
/// channel's ring.
#[derive(Debug)]
pub struct Rings {
    slabs: Vec<Buffer>,
    shapes: Vec<ChannelShape>,
}

impl Rings {
    /// Reserve one ring per shape: `capacity + 1` cells, the spare included.
    ///
    /// The spare is what makes "full" distinguishable from "empty" with two
    /// monotone cursors, which is the shared ring arithmetic's own rule
    /// (`driver::program::channel`) and not this file's.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] when the device declined a reservation,
    /// [`Fault::Program`] for a shape whose ring will not fit a `u64`.
    pub fn allocate(device: &Context, shapes: &[ChannelShape]) -> Result<Rings> {
        let mut slabs = Vec::with_capacity(shapes.len());
        for shape in shapes {
            let cells = u64::from(shape.capacity) + 1;
            let bytes = cells
                .checked_mul(shape.cell_stride() as u64)
                .ok_or_else(|| Fault::program("program::launch", "a ring past what a u64 counts"))?;
            slabs.push(Buffer::zeroed(device, bytes.max(1))?);
        }
        Ok(Rings {
            slabs,
            shapes: shapes.to_vec(),
        })
    }

    /// How many channels this instance carries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slabs.len()
    }

    /// Whether it carries none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slabs.is_empty()
    }

    /// One channel's shape.
    #[must_use]
    pub fn shape(&self, channel: usize) -> Option<ChannelShape> {
        self.shapes.get(channel).copied()
    }

    /// Where one cell begins inside its ring, in bytes.
    ///
    /// The sequence is monotone and the ring is modular: `sequence %
    /// (capacity + 1)` is the slot, which is what makes a cursor a count
    /// rather than an index.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a channel this instance does not carry.
    pub fn cell_offset(&self, channel: usize, sequence: u64) -> Result<u64> {
        let shape = self.shape(channel).ok_or_else(|| {
            Fault::program(
                "program::launch",
                format!("channel {channel} is not one this instance carries"),
            )
        })?;
        let cells = u64::from(shape.capacity) + 1;
        Ok((sequence % cells) * shape.cell_stride() as u64)
    }

    /// The buffer one channel's ring lives in.
    pub(crate) fn slab(&self, channel: usize) -> Result<&Buffer> {
        self.slabs.get(channel).ok_or_else(|| {
            Fault::program(
                "program::launch",
                format!("channel {channel} is not one this instance carries"),
            )
        })
    }

    /// Write one cell's wire bytes.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a channel this instance does not carry or a
    /// payload that is not one cell, and whatever the copy said.
    pub fn write_cell(&mut self, channel: usize, sequence: u64, bytes: &[u8]) -> Result<()> {
        let at = self.cell_offset(channel, sequence)?;
        let width = self
            .shape(channel)
            .map_or(0, |shape| shape.cell_bytes());
        if bytes.len() != width {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "channel {channel}'s cell is {width} bytes and this write carries {}",
                    bytes.len()
                ),
            ));
        }
        self.slabs[channel].write(at, bytes)
    }

    /// Read one cell's wire bytes.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a channel this instance does not carry, and
    /// whatever the copy said.
    pub fn read_cell(&self, channel: usize, sequence: u64) -> Result<Vec<u8>> {
        let at = self.cell_offset(channel, sequence)?;
        let width = self
            .shape(channel)
            .map_or(0, |shape| shape.cell_bytes());
        let mut cell = vec![0u8; width];
        self.slabs[channel].read(at, &mut cell)?;
        Ok(cell)
    }
}

/// Where one channel stands this fire: the committed front and the pending
/// back, as sequence numbers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    /// The cell a take reads.
    pub head: u64,
    /// The cell a put writes.
    pub tail: u64,
}

/// One channel's two bound cells, resolved for this fire.
#[derive(Debug, Clone)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Bound {
    /// The ring the two cells live in — retained, so the binding outlives
    /// the borrow that resolved it.
    slab: Buffer,
    committed: u64,
    pending: u64,
}

/// Everything one stage binds for one fire.
#[derive(Debug)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Prepared {
    /// `M1Status`, 16 bytes at buffer 0. Started at `state = 1` every fire;
    /// the kernel raises it to 3 with a fault word, or to 4 on commit.
    status: Buffer,
    descriptors: Buffer,
    params: Buffer,
    offsets: Buffer,
    /// The values' scratch AND the temporary that follows it: the emitted
    /// kernel takes them as two bindings (`scratch` at 4, `temporary` at 5)
    /// and they are one allocation, bound twice at two offsets.
    scratch: Buffer,
    /// The intrinsic buffer bound at index 6, and where in it this fire's
    /// rows begin. `None` binds nil, which the kernel may not dereference —
    /// a stage with no `INTRINSIC_VAL` op never reads it.
    logits: Option<(Buffer, u64)>,
    channel_count: u32,
    value_count: u32,
    scratch_stride: u32,
    temporary_offset: u32,
    /// This stage's local channel slot → the instance's dense channel index.
    bindings: Vec<u32>,
    /// The resolved cells, in stage-local slot order. Filled by
    /// [`Prepared::refresh`] and read by [`Prepared::launch_region`].
    bound: Vec<Bound>,
}

impl Prepared {
    /// Carve every buffer one stage needs, for a single-lane fire.
    ///
    /// **ONE LANE.** The M2 kernel is `if (gid != 0 …) return;` — it is one
    /// thread over one lane by construction, where the CUDA kernel reads
    /// `blockIdx.x`. A grouped fire is the M3 path, which binds raw device
    /// addresses (`lane.logits_base`, `channel.committed_cell`) and needs
    /// `MTLBuffer.gpuAddress` plumbed through `device::alloc` first; it is
    /// named here and not built.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when a value's shape does not resolve against
    /// `extents`, when the scratch exceeds what `driver::layout` permits, or
    /// when a put names a channel the plan does not bind; whatever the
    /// reservations said otherwise.
    pub fn build(
        device: &Context,
        plan: &LaunchStagePlan,
        shapes: &[ChannelShape],
        extents: Extents,
    ) -> Result<Prepared> {
        let channel_count = u32::try_from(plan.channel_bindings.len())
            .map_err(|_| Fault::program("program::launch", "more channels than a u32 can count"))?;
        let value_count = u32::try_from(plan.value_types.len())
            .map_err(|_| Fault::program("program::launch", "more values than a u32 can count"))?;

        // ── The value descriptors, and the scratch they size. ──
        let descriptors: Vec<ValueDesc> = plan
            .value_types
            .iter()
            .map(|value| {
                describe(value, &extents).map_err(|why| {
                    Fault::program(
                        "program::launch",
                        format!("a value's shape does not resolve against this fire: {why:?}"),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let scratch_layout = layout(&descriptors).map_err(|why| {
            Fault::program(
                "program::launch",
                format!("this fire's scratch does not fit: {why:?}"),
            )
        })?;
        let scratch_stride = u32::try_from(scratch_layout.total)
            .map_err(|_| Fault::program("program::launch", "a scratch stride past a u32"))?;
        let temporary_offset = u32::try_from(scratch_layout.temporary)
            .map_err(|_| Fault::program("program::launch", "a temporary offset past a u32"))?;

        // ── Op params. The SHARED record, uploaded as it stands: the
        //    emitted `struct M1OpParams` is `driver::OpParams`, 64 bytes,
        //    field for field, which is why this plane has no widening step
        //    and no offset assertions.
        let mut records = Vec::with_capacity(plan.ops.len());
        let mut result_base = 0u32;
        for op in &plan.ops {
            let mut record = OpParams::of(op, result_base, OpRuntime::default());
            if let (true, Some(channel)) = (op.tag == tags::CHAN_PUT, op.channel) {
                // `sink_bytes` IS THE CELL, exactly: the emitted put writes
                // `0..sink_bytes` of the pending cell (zero-filling the tail
                // past the value's own bytes) and faults when the value is
                // wider. Zero refuses every put silently; anything larger
                // runs off the end of the ring slot.
                let dense = plan
                    .channel_bindings
                    .get(channel as usize)
                    .copied()
                    .ok_or_else(|| {
                        Fault::program(
                            "program::launch",
                            format!(
                                "a put names stage-local channel {channel}, which the plan \
                                 does not bind"
                            ),
                        )
                    })?;
                let shape = shapes.get(dense as usize).copied().ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!(
                            "a put targets channel {dense}, which this instance does not carry"
                        ),
                    )
                })?;
                record.sink_bytes = u32::try_from(shape.cell_bytes()).map_err(|_| {
                    Fault::program("program::launch", "a channel cell past what a u32 counts")
                })?;
            }
            records.push(record);
            result_base += u32::from(op.result_count);
        }

        let mut params = Buffer::zeroed(
            device,
            (records.len() * size_of::<OpParams>()).max(size_of::<OpParams>()) as u64,
        )?;
        params.write(0, &records_bytes(&records))?;

        let descriptor_bytes: Vec<u8> = descriptors.iter().flat_map(record_bytes).collect();
        let mut descriptor_buffer =
            Buffer::zeroed(device, descriptor_bytes.len().max(1) as u64)?;
        descriptor_buffer.write(0, &descriptor_bytes)?;

        let offset_bytes: Vec<u8> = scratch_layout
            .values
            .iter()
            .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut offsets =
            Buffer::zeroed(device, offset_bytes.len().max(size_of::<u32>()) as u64)?;
        offsets.write(0, &offset_bytes)?;

        let scratch_bytes = u64::from(scratch_stride).max(SCRATCH_ALIGN);
        let scratch = Buffer::zeroed(device, scratch_bytes)?;
        let status = Buffer::zeroed(device, driver::STATUS_BYTES as u64)?;

        Ok(Prepared {
            status,
            descriptors: descriptor_buffer,
            params,
            offsets,
            scratch,
            logits: None,
            channel_count,
            value_count,
            scratch_stride,
            temporary_offset,
            bindings: plan.channel_bindings.clone(),
            bound: Vec::new(),
        })
    }

    /// Resolve this fire's cells and reset everything a fire starts from.
    ///
    /// The status word starts at ONE, not zero: the kernel's first line is
    /// `if (gid != 0 || status->state != 1) return;`, so a zero start would
    /// enter every region and do nothing.
    ///
    /// **THERE IS NO STAGING COPY AND NO STREAM.** A reservation here is
    /// `StorageModeShared`, so a host write IS the device write; what
    /// survives is the ORDER, and it is this call standing before
    /// [`Prepared::launch_region`] opens its command buffer.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when a stage-local slot names a channel this
    /// instance does not carry, and whatever the copies said.
    pub fn refresh(&mut self, rings: &Rings, cursors: &[Cursor]) -> Result<()> {
        self.bound.clear();
        self.bound.reserve(self.bindings.len());
        for (local, &dense) in self.bindings.iter().enumerate() {
            let channel = dense as usize;
            let cursor = cursors.get(channel).copied().ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    format!(
                        "stage-local channel {local} binds channel {dense}, which this \
                         instance does not carry"
                    ),
                )
            })?;
            self.bound.push(Bound {
                slab: rings.slab(channel)?.clone(),
                committed: rings.cell_offset(channel, cursor.head)?,
                pending: rings.cell_offset(channel, cursor.tail)?,
            });
        }
        // Zeroed every fire, not once: a value slot no op writes reads back
        // as whatever the LAST fire left there, and zeros are the state the
        // emitted kernels — and the host interpreter they are diffed against
        // — both assume.
        let bytes = self.scratch.bytes();
        self.scratch.zero_span(0, bytes)?;
        let ready = Status {
            state: 1,
            fault: 0,
            reserved0: 0,
            reserved1: 0,
        };
        self.status.write(0, &record_bytes(&ready))?;
        Ok(())
    }

    /// Point the one intrinsic buffer at what a model fire produced.
    ///
    /// **THERE IS ONE INTRINSIC BUFFER ON THIS PLANE AND IT IS ALWAYS
    /// bf16.** The CUDA emitter carries a per-intrinsic slot table — five
    /// side arrays of eight words each — and the driver stages a base, a
    /// storage mode, a width, a stride and an offset per intrinsic. The
    /// Metal M2 emitter binds `const device uchar* logits [[buffer(6)]]` and
    /// sets it as the first argument of EVERY `INTRINSIC_VAL` op, and the
    /// runtime reads it as `bfloat`. So there is no storage mode to state
    /// (there is only one), and `IntrinsicId::Logits` and
    /// `IntrinsicId::MtpLogits` cannot be pointed at two buffers in one
    /// stage — which is a limit of the emitted ABI and is refused here
    /// rather than silently satisfied by the last binding to arrive.
    ///
    /// **AND THERE IS NO WIDTH, NO STRIDE AND NO STORAGE MODE TO STATE.**
    /// The CUDA twin takes five words per intrinsic because its kernel reads
    /// five side tables; the M2 runtime takes the ROW OFFSET out of the op
    /// record it was planned with (`logits + p.imm2 * p.imm`) and the
    /// element type out of the source (`bfloat`, always). So what a caller
    /// used to pass as `row_offset` it passes here as BYTES in `offset`,
    /// which is the buffer binding's own offset and therefore the one number
    /// that cannot disagree with what the encoder did.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a second intrinsic bound at a different
    /// rectangle in one stage — the emitter gives them all one buffer, so
    /// the second binding would silently move the first.
    pub fn bind_intrinsic(
        &mut self,
        intrinsic: IntrinsicId,
        base: &Buffer,
        offset: u64,
    ) -> Result<()> {
        if let Some((_, at)) = &self.logits
            && *at != offset
        {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "this stage already bound its one intrinsic buffer at {at}; the M2 \
                     emitter gives every intrinsic `logits [[buffer(6)]]`, so \
                     {intrinsic:?} cannot have a second rectangle"
                ),
            ));
        }
        self.logits = Some((base.clone(), offset));
        Ok(())
    }

    /// Encode and run one generated region, and wait for it.
    ///
    /// **THE ONE SYNCHRONIZATION A GUEST PASS HAS**, and it is here because
    /// the host reads the verdict immediately afterwards: the status word,
    /// the pending cells, the scratch. Everything before it is encode-only,
    /// as decision #15 requires of every dispatch on this plane.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] when the command buffer would not open or the GPU
    /// refused the work, [`Fault::Deviceless`] off Apple.
    pub fn launch_region(
        &self,
        device: &Context,
        pipelines: &crate::device::Pipelines,
        rings: &Rings,
        region: &Region,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

            let _ = (pipelines, rings);
            let frame = device.frame()?;
            let encoder = frame.encoder();
            encoder.setComputePipelineState(region.pipeline());
            // SAFETY: every buffer below is retained by `self` or by
            // `region` for the length of this call, and every offset was
            // bounds-checked when it was computed.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.status.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.descriptors.raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(self.params.raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(self.offsets.raw()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(self.scratch.raw()), 0, 4);
                encoder.setBuffer_offset_atIndex(
                    Some(self.scratch.raw()),
                    self.temporary_offset as usize,
                    5,
                );
                match &self.logits {
                    Some((buffer, at)) => encoder.setBuffer_offset_atIndex(
                        Some(buffer.raw()),
                        usize::try_from(*at).unwrap_or(0),
                        6,
                    ),
                    None => encoder.setBuffer_offset_atIndex(None, 0, 6),
                }
                for (local, bound) in self.bound.iter().enumerate() {
                    let at = FIRST_CHANNEL_BUFFER + local * 2;
                    encoder.setBuffer_offset_atIndex(
                        Some(bound.slab.raw()),
                        usize::try_from(bound.committed).unwrap_or(0),
                        at,
                    );
                    encoder.setBuffer_offset_atIndex(
                        Some(bound.slab.raw()),
                        usize::try_from(bound.pending).unwrap_or(0),
                        at + 1,
                    );
                }
            }
            let one = MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreads_threadsPerThreadgroup(one, one);
            frame.commit()
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, pipelines, rings, region);
            Err(Fault::Deviceless)
        }
    }

    /// The kernel's verdict, whole.
    ///
    /// Where the CUDA twin answers a boolean, this answers `driver::Status`
    /// — the state, the fault word and the two site words — so a refusal can
    /// be NAMED (`driver::describe_fault`) instead of merely counted.
    ///
    /// # Errors
    ///
    /// Whatever the readback said.
    pub fn status(&self) -> Result<Status> {
        let mut bytes = [0u8; driver::STATUS_BYTES];
        self.status.read(0, &mut bytes)?;
        Status::read(&bytes).ok_or_else(|| {
            Fault::program(
                "program::launch",
                "the status word read back short, which is a reservation this plane carved",
            )
        })
    }

    /// How many channel slots this stage binds.
    #[must_use]
    pub const fn channel_count(&self) -> u32 {
        self.channel_count
    }

    /// How many values this stage's scratch carries.
    #[must_use]
    pub const fn value_count(&self) -> u32 {
        self.value_count
    }

    /// How wide one lane's scratch is.
    #[must_use]
    pub const fn scratch_stride(&self) -> u32 {
        self.scratch_stride
    }
}

/// One `#[repr(C)]` record's bytes.
///
/// # Safety
///
/// `T` must be `#[repr(C)]` with no padding holes carrying meaning and no
/// invalid bit patterns — the three records this file uploads (`OpParams`,
/// `ValueDesc`, `Status`) are all flat `u32` structs.
fn record_bytes<T: Copy>(record: &T) -> Vec<u8> {
    // SAFETY: as stated above; the slice's life is this expression's.
    let bytes =
        unsafe { std::slice::from_raw_parts((record as *const T).cast::<u8>(), size_of::<T>()) };
    bytes.to_vec()
}

/// A slice of records, end to end.
fn records_bytes<T: Copy>(records: &[T]) -> Vec<u8> {
    records.iter().flat_map(record_bytes).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cell_offset_wraps_at_capacity_plus_the_spare() {
        let shape = ChannelShape {
            capacity: 3,
            numel: 1,
            dtype: DType::I32,
        };
        let rings = Rings {
            slabs: Vec::new(),
            shapes: vec![shape],
        };
        let width = shape.cell_stride() as u64;
        assert_eq!(rings.cell_offset(0, 0).unwrap(), 0);
        assert_eq!(rings.cell_offset(0, 3).unwrap(), 3 * width);
        // Four cells for a capacity of three: the spare is what makes full
        // and empty two different cursor pairs.
        assert_eq!(rings.cell_offset(0, 4).unwrap(), 0);
        assert_eq!(rings.cell_offset(0, 9).unwrap(), width);
    }

    #[test]
    fn a_channel_this_instance_does_not_carry_is_refused_by_number() {
        let rings = Rings {
            slabs: Vec::new(),
            shapes: Vec::new(),
        };
        let said = rings.cell_offset(2, 0).expect_err("no such channel").to_string();
        assert!(said.contains('2'), "the refusal names the channel: {said}");
    }

    #[test]
    fn a_packed_bool_cell_is_padded_to_a_bindable_stride() {
        // Eight bool lanes are ONE byte on the wire, and a ring at that
        // width puts cell 1 at offset 1 — an offset Metal will not bind.
        let shape = ChannelShape {
            capacity: 4,
            numel: 8,
            dtype: DType::Bool,
        };
        assert_eq!(shape.cell_bytes(), 1);
        assert_eq!(shape.cell_stride(), CELL_ALIGN);
        // And a wide cell is not padded past its own alignment.
        let wide = ChannelShape {
            capacity: 1,
            numel: 8,
            dtype: DType::I32,
        };
        assert_eq!(wide.cell_bytes(), 32);
        assert_eq!(wide.cell_stride(), 32);
    }

    #[test]
    fn the_shared_op_record_is_the_emitted_one() {
        // `struct M1OpParams` in `ptir_m1_runtime.metal` is sixteen `uint`s.
        // If this ever stops being true the params buffer is misread field
        // for field and every guest program computes garbage silently.
        assert_eq!(size_of::<OpParams>(), 64);
        assert_eq!(size_of::<Status>(), 16);
    }

    #[test]
    fn the_first_channel_binds_where_the_emitter_writes_it() {
        // `emit_fused_region` writes `7 + channel * 2` / `8 + channel * 2`.
        assert_eq!(FIRST_CHANNEL_BUFFER, 7);
    }
}
