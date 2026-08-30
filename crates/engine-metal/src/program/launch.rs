//! One stage, encoded: the device rings a guest's channels live in, the
//! buffers one fire binds, and the dispatch that runs a region.
//!
//! **THIS FILE IS SHORTER THAN ITS CUDA TWIN BY HALF, AND ALMOST ALL OF THE
//! DIFFERENCE IS THE M2 ABI RATHER THAN THE PLATFORM.** The CUDA emitter
//! hands its kernel seven pointers, three scalars and six side tables, and
//! the engine builds a LANE TABLE — a header, a record per lane, a flat
//! array of channel slots holding the committed and pending CELL ADDRESSES —
//! because a CUDA kernel argument is a pointer and a cell is an address
//! inside a ring. The Metal M2 kernel takes the two cells as BUFFER
//! BINDINGS (`committed_{k}` at index `7 + 2k`, `pending_{k}` at `8 + 2k`),
//! and Metal binds a buffer plus an offset — so the cell address IS the
//! binding, and the whole lane table, its host mirror, and the sixteen-byte
//! per-fire patch that keeps them in step all disappear.
//!
//! Three more things go with it. There is no `CudaOpParams`: the emitted
//! `struct M1OpParams` is exactly `eta_exec::OpParams`, 64 bytes, so the
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
//! `eta_exec::Status` — a state, a fault word, two reserved words the fault
//! sites use — so a refusal can be named with `eta_exec::describe_fault`
//! instead of merely reported.

use std::mem::size_of;

use eta_compiler::codegen::launch::{LaunchChannel, LaunchStagePlan};
use eta_exec::{Extents, OpParams, OpRuntime, SCRATCH_ALIGN, Status, ValueDesc, describe, layout};
use eta_ir::Dtype;
use eta_ir::op::{IntrinsicId, tags};

use crate::device::ctx::Frame;
use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};

use super::compile::Region;

/// The buffer index the first channel's committed cell binds at. Everything
/// below it is fixed: status, descriptors, params, offsets, scratch,
/// temporary, the trunk's logits.
///
/// **AND THE CHANNELS NO LONGER RUN TO THE END OF THE SPACE.** The slot table
/// (`eta_compiler::codegen::metal::intrinsics`) puts every intrinsic other
/// than the trunk's at the TOP of Metal's argument indices, growing down from
/// 30, so a stage that reads a second rectangle meets the channels in the
/// middle at eleven rather than twelve. The emitter is what enforces that —
/// `fused_channel_ceiling` — and this side never has to know, because it
/// binds a slot only where one is bound and the emitter refused the stage
/// where the two would collide.
///
/// Read off `eta_compiler::codegen::metal::fused::emit_fused_region`,
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
    pub dtype: Dtype,
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
            dtype: eta_exec::concrete_dtype(declared.dtype),
        }
    }

    /// Bytes in one cell — the wire cell, which on this plane is the only
    /// cell there is.
    #[must_use]
    pub fn cell_bytes(&self) -> usize {
        eta_exec::wire_cell_bytes(self.dtype, self.numel)
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
    /// (`eta_exec::channel`) and not this file's.
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

/// How many rows the per-intrinsic tables carry.
///
/// **PROJECTED FROM THE WIRE, NOT COUNTED.** `IntrinsicId::SLOTS` is one past
/// the largest id rather than the number of ids, which is the same stride the
/// CUDA side indexes its five side arrays with — an id that overflowed it
/// would not fault, it would read the next intrinsic's slot.
const INTRINSIC_SLOTS: usize = IntrinsicId::SLOTS as usize;

/// One intrinsic's rectangle, as this plane binds one.
///
/// **THE CUDA TWIN'S FIVE NUMBERS, AND ONLY TWO OF THEM SURVIVE AS DATA.**
/// That side writes `(base, storage, width, row_stride, row_offset)` into five
/// side arrays because a CUDA kernel argument is a raw device address it must
/// be told how to walk. Metal binds an OBJECT at an OFFSET, so the base, the
/// stride and the row offset ARE `setBuffer:offset:atIndex:`, and the other
/// two never reach the kernel at all: the emitted `0xA0` handler reads
/// `out0.len` consecutive elements from whatever argument it was handed, at
/// an element type it picks off the INTRINSIC ID rather than off any word the
/// host sets, so the width and the element type are things the HOST has to
/// check rather than things the device can be told.
/// [`Prepared::bind_intrinsic`] checks them and keeps neither — a stored copy
/// of a number nothing reads back is a side table with no kernel behind it.
#[derive(Debug, Clone)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Slot {
    /// The allocation the rectangle lives in, retained so the binding
    /// outlives the borrow that resolved it.
    base: Buffer,
    /// Where in it this fire's rows begin, in bytes. This is the CUDA side's
    /// `row_offset` already multiplied out, because a bound offset is the one
    /// number that cannot disagree with what the encoder did.
    offset: u64,
}

/// What a stage's own ops say about an intrinsic's rectangle.
///
/// **THE READER'S CLAIM, TAKEN OFF THE PLAN AND KEPT TO ARGUE WITH.** A guest
/// declares `logits` as `[n_out, vocab]`; the shell hands over a rectangle it
/// resolved from the bake. Those two numbers have never been compared on this
/// plane, and while there was one rectangle the failure was invisible. With a
/// slot table there are two, and a draft column whose width is not the trunk's
/// would be read at the trunk's stride — rows sliding by a few thousand
/// elements each, which is a plausible-looking answer and a wrong one.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Declared {
    /// The row width the reader's output resolved to — the CUDA handler's
    /// `logical_width`, which is a CEILING on the offered rectangle's width
    /// and not an equality (`fused_block0.cuh`: the only refusal there is
    /// `stride < logical_width`).
    width: u32,
    /// How many rows of that width the reader gathers — the CUDA handler's
    /// `out0.len / logical_width`. Load-bearing here because this plane has
    /// no `intrinsic_row_stride`: see [`Prepared::bind_intrinsic`].
    rows: u32,
    /// The most elements any one reader of this intrinsic gathers — the
    /// emitted handler's `out0.len`, which is what it walks off the binding.
    /// In ELEMENTS and not bytes, because how wide one is depends on the
    /// intrinsic (`m2_intrinsic_element_bytes`) and this claim is the
    /// program's.
    elements: u64,
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
    /// **THE SLOT TABLE**: one rectangle per intrinsic id, indexed by the
    /// ordinal the wire gives it. `None` at the trunk's slot binds nil at
    /// index 6, which the kernel may not dereference — a stage with no
    /// `INTRINSIC_VAL` op never reads it; `None` anywhere else binds nothing
    /// at all, because those indices come out of the TOP of the argument
    /// space and a nil written there would clobber a channel cell in a stage
    /// that never asked for a second rectangle.
    ///
    /// This was `Option<(IntrinsicId, Buffer, u64)>` — one rectangle, and the
    /// id carried only so the second binding could be REFUSED. See
    /// [`Prepared::bind_intrinsic`].
    intrinsics: [Option<Slot>; INTRINSIC_SLOTS],
    /// What this stage's own ops declared about each intrinsic they read,
    /// read off the plan once at [`Prepared::build`] and argued with at every
    /// bind. `None` for an intrinsic no op reads, and for one whose reader's
    /// output is not the gather itself (`mtp_drafts` answers `[k]` I32 token
    /// ids, which say nothing about the rectangle they were argmaxed out of).
    declared: [Option<Declared>; INTRINSIC_SLOTS],
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
    /// `extents`, when the scratch exceeds what `eta_exec::layout` permits, or
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
        //    emitted `struct M1OpParams` is `eta_exec::OpParams`, 64 bytes,
        //    field for field, which is why this plane has no widening step
        //    and no offset assertions.
        let mut records = Vec::with_capacity(plan.ops.len());
        let mut declared: [Option<Declared>; INTRINSIC_SLOTS] = [None; INTRINSIC_SLOTS];
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
            // ── **WHAT THIS STAGE'S READERS CLAIM ABOUT EACH RECTANGLE.**
            //    The `0xA0` handler walks `out0.len` consecutive elements off
            //    whatever buffer it was handed, `out0.last` to a row, so the
            //    reader's own output descriptor states the geometry the
            //    binding has to match. Taken here because this is where the
            //    records and the descriptors are both in hand.
            // Rank 1 is `mtp_drafts`: the output is `[k]` token ids and the
            // rows they were chosen from are not in it. Nothing to claim, so
            // nothing is claimed — an empty seat refuses a bind by name
            // rather than by a number nobody computed.
            if let Some(intrinsic) = op.intrinsic
                && let Some(seat) = declared.get_mut(intrinsic as usize)
                && let Some(out) = descriptors.get(record.o0 as usize)
                && out.rank >= 2
            {
                let claim = Declared {
                    width: out.last,
                    rows: out.rows.max(1),
                    elements: u64::from(out.len),
                };
                *seat = Some(match *seat {
                    Some(prior) => Declared {
                        width: prior.width.max(claim.width),
                        rows: prior.rows.max(claim.rows),
                        elements: prior.elements.max(claim.elements),
                    },
                    None => claim,
                });
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
        let status = Buffer::zeroed(device, eta_exec::STATUS_BYTES as u64)?;

        Ok(Prepared {
            status,
            descriptors: descriptor_buffer,
            params,
            offsets,
            scratch,
            intrinsics: [const { None }; INTRINSIC_SLOTS],
            declared,
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

    /// Point one intrinsic's slot at the rectangle a model fire produced.
    ///
    /// **THERE IS A SLOT PER INTRINSIC NOW, AND THERE USED TO BE ONE FOR ALL
    /// OF THEM.** The M2 emitter wrote `const device uchar* logits
    /// [[buffer(6)]]` and made it the first argument of EVERY `INTRINSIC_VAL`
    /// op, so `logits` and `mtp_logits` in one stage were two names for one
    /// rectangle — and this function's whole body was the refusal that said
    /// so. `eta_compiler::codegen::metal::intrinsics` gives each id an
    /// argument index of its own (the trunk keeps 6; the rest come down from
    /// 30), so the second rectangle exists and this points at it.
    ///
    /// **THE CUDA TWIN'S FIVE NUMBERS ARE STILL FIVE, IN THREE SPELLINGS.**
    /// That side writes `(base, storage, width, row_stride, row_offset)` into
    /// side arrays its kernel walks. Here the base and the row offset ARE the
    /// binding — `setBuffer:offset:atIndex:` — and the storage mode is a
    /// CHECK rather than a selector, because the emitted `0xA0` handler picks
    /// its element type off the intrinsic id and cannot be told a different
    /// one. What is
    /// left as data is `width`, and it is kept to be ARGUED WITH: the reading
    /// program declared a row width when its shapes resolved, the shell
    /// resolved another off the bake, and until this wave nothing compared
    /// them. One rectangle made that invisible; two make it a draft column
    /// read at the trunk's stride. What is compared is the CUDA handler's
    /// own relation and not a stricter one — see
    /// [`Prepared::bind_intrinsic`], where the declared width is a CEILING.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an intrinsic past the table's pitch, for one
    /// this plane's emitter cannot bind at all, for a rectangle whose element
    /// type is not the one the handler reads for that id, for one NARROWER
    /// than the width the stage's own readers
    /// declared, and for a multi-row read whose declared width is not the
    /// rectangle's — the one shape the CUDA twin's stride expresses and this
    /// plane's consecutive gather cannot. [`Fault::Ceiling`] for a rectangle
    /// the reader would walk off the end of.
    pub fn bind_intrinsic(
        &mut self,
        intrinsic: IntrinsicId,
        base: &Buffer,
        offset: u64,
        width: u32,
        dtype: Dtype,
    ) -> Result<()> {
        if eta_compiler::codegen::metal::m2_intrinsic_buffer(intrinsic as u16).is_none() {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "{intrinsic:?} has no argument index in the M2 slot table, so a \
                     rectangle bound for it would be read as another intrinsic's"
                ),
            ));
        }
        let declared = *self
            .declared
            .get(intrinsic as usize)
            .ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    format!(
                        "{intrinsic:?} is past the pitch the slot table is indexed with"
                    ),
                )
            })?;
        // **THE ELEMENT TYPE IS THE INTRINSIC'S, AND THIS IS THE HOST HALF OF
        // AN ARM THE RUNTIME TAKES.** This was one type and one refusal: the
        // `0xA0` handler read `bfloat` for everything, so an F32 rectangle —
        // which is what an attention-score plane is — was named here and
        // refused rather than bound and misread. The runtime grew the arm
        // (`ptir_m1_runtime.metal` branches on `p.intr` and gathers `float`
        // for `AttnScore`), and because Metal binds an OBJECT rather than an
        // address there is still no storage word to set: the element type is a
        // function of the ID, published once as
        // `m2_intrinsic_element_bytes` and read here so this side cannot
        // disagree with the kernel about how far a reader reaches.
        //
        // The CUDA side still selects between `RawBf16`, `RowPointers` and
        // `F32` off a per-binding side array, and that difference is the
        // platform rather than the idea.
        let element = eta_compiler::codegen::metal::m2_intrinsic_element_bytes(intrinsic as u16)
            .map(u64::from)
            .ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    format!("{intrinsic:?} has no element width in the M2 slot table"),
                )
            })?;
        let wanted = if element == 4 { Dtype::F32 } else { Dtype::Bf16 };
        if dtype != wanted {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "a rectangle for {intrinsic:?} landed as {dtype:?}; the emitted \
                     `0xA0` handler reads it as {wanted:?} and has no other element type \
                     for this intrinsic"
                ),
            ));
        }
        if let Some(declared) = declared {
            // ── **THE DECLARED WIDTH IS A CEILING, NOT AN EQUALITY**, and
            //    the authority is the CUDA twin's handler rather than this
            //    plane's convenience. `runtime/cuda/ptir_m1_runtime_body.cuh`
            //    (tag `0xA0`) reads `logical_width = out0.last`, takes the
            //    STRIDE from `intrinsic_row_stride`, and faults on exactly
            //    one relation — `stride < logical_width`. A guest declaring
            //    `[1, 8]` against a 248320-wide out seam is therefore SERVED
            //    there, with the first eight columns of the row it was
            //    pointed at, which is the written rule in `attn-score.md`
            //    §6.2 ("the returned extent is a DECLARED CEILING zero-padded
            //    to"). So a reader asking for MORE than the rectangle holds
            //    is the refusal, and it is the same refusal on both planes.
            if declared.width > width {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "this stage reads {intrinsic:?} as rows of {} elements and the \
                         rectangle offered is only {width} wide; a declared extent is a \
                         ceiling on the row it is pointed at, so serving this one would \
                         read past the end of every row",
                        declared.width
                    ),
                ));
            }
            // ── **AND THE MULTI-ROW READ IS WHERE THE TWO PLANES PART.**
            //    CUDA carries a stride per binding, so it walks
            //    `first_row + i / logical_width` at `stride`; the emitted
            //    `0xA0` handler here reads `out0.len` CONSECUTIVE elements
            //    off `setBuffer:offset:` and has no stride to be told. When
            //    the two widths agree, consecutive IS the stride and every
            //    shape is expressible. When they do not, only a ONE-ROW read
            //    lands on the same bytes the CUDA twin would return; a
            //    `k`-row verifier would have row 1 start `declared.width`
            //    elements in rather than `width`, sliding every row after the
            //    first. That is a missing runtime argument, not a guest
            //    error, so it is named here rather than bound and misread.
            if declared.width < width && declared.rows > 1 {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "this stage reads {intrinsic:?} as {} rows of {} elements out of a \
                         rectangle {width} wide, and the emitted gather walks its rows \
                         consecutively: it has no row stride to be told, so every row \
                         after the first would land {} elements short. A narrower read \
                         of ONE row is served; this one needs the runtime to carry a \
                         stride the way the CUDA handler's `intrinsic_row_stride` does",
                        declared.rows,
                        declared.width,
                        width - declared.width
                    ),
                ));
            }
            // The reader walks `out0.len` consecutive elements off the
            // binding — two bytes each where the handler reads `bfloat`, four
            // where it reads `float` — and the binding's own offset is where
            // it starts. So this is the whole of what it can touch, and it is
            // the one bound the kernel cannot check for itself. The width
            // comes off the same table the emitter routed the argument
            // through, because a bounds check computed against the wrong
            // element is no bounds check.
            let reach = offset.saturating_add(declared.elements.saturating_mul(element));
            if reach > base.bytes() {
                return Err(Fault::Ceiling {
                    what: "bytes in the rectangle an intrinsic is pointed at",
                    need: reach,
                    have: base.bytes(),
                });
            }
        }
        // **RE-POINTING THE SAME INTRINSIC IS THE NORMAL CASE**, and the
        // refusal that used to stand here had to be careful not to catch it:
        // an attached fire binds lane 3's last row this step and lane 3's
        // last row one token later, a different byte in the same arena
        // rectangle each time. With a slot per intrinsic there is nothing
        // left to collide, so the write is unconditional.
        self.intrinsics[intrinsic as usize] = Some(Slot {
            base: base.clone(),
            offset,
        });
        Ok(())
    }

    /// Encode one generated region into a pass someone else opened, and do
    /// not commit.
    ///
    /// **THIS IS THE HALF THAT CAN RIDE IN A MODEL FIRE'S COMMAND BUFFER**,
    /// and separating it out is what let the attachment path exist:
    /// [`Prepared::launch_region`] below is this plus a command buffer of its
    /// own plus a wait, which is right for a pass fired on its own and
    /// impossible inside `serve::enqueue`, where nothing may block.
    ///
    /// The caller owns the ordering. A Metal compute pass is
    /// `MTLDispatchTypeSerial`, so regions encoded into one pass observe each
    /// other's writes in encode order with the barriers Metal inserts — which
    /// is exactly the guarantee the stage loop used to get from waiting on a
    /// command buffer per region, and it is why a whole program's stages can
    /// be encoded up front.
    ///
    /// # Errors
    ///
    /// [`Fault::Deviceless`] off Apple.
    pub fn encode_into(&self, frame: &Frame, region: &Region) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

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
                // ── **THE SLOT TABLE, ONE `setBuffer` PER BOUND
                //    RECTANGLE.** The trunk's index is written whatever
                //    happens, nil included: it sits BELOW the channels, a
                //    stage with no `INTRINSIC_VAL` op never dereferences it,
                //    and leaving it holding whatever the last region bound is
                //    how a stale rectangle survives a rebind. Every other
                //    index comes out of the TOP of the argument space, where
                //    a channel cell may already be bound — so an unbound slot
                //    there is written with NOTHING rather than with nil.
                for (slot, held) in self.intrinsics.iter().enumerate() {
                    let Some(at) = u16::try_from(slot)
                        .ok()
                        .and_then(eta_compiler::codegen::metal::m2_intrinsic_buffer)
                    else {
                        continue;
                    };
                    match held {
                        Some(bound) => encoder.setBuffer_offset_atIndex(
                            Some(bound.base.raw()),
                            usize::try_from(bound.offset).unwrap_or(0),
                            at,
                        ),
                        None if at == eta_compiler::codegen::metal::M2_LOGITS_BUFFER => {
                            encoder.setBuffer_offset_atIndex(None, 0, at);
                        }
                        None => {}
                    }
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
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (frame, region);
            Err(Fault::Deviceless)
        }
    }

    /// Encode and run one generated region, and wait for it.
    ///
    /// **THE ONE SYNCHRONIZATION A GUEST PASS FIRED ON ITS OWN HAS**, and it
    /// is here because the host reads the verdict immediately afterwards: the
    /// status word, the pending cells, the scratch. Everything before it is
    /// encode-only, as decision #15 requires of every dispatch on this plane.
    ///
    /// **IT IS NO LONGER THE ONLY WAY A REGION REACHES THE DEVICE.** A pass
    /// attached to a model fire is encoded by [`Prepared::encode_into`] into
    /// that fire's own command buffer and never waits at all; this door is
    /// what [`super::Session::fire`] — the standalone verb — still takes, and
    /// the wait is honest there because the caller is standing at it for the
    /// verdict.
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
            let _ = (pipelines, rings);
            let frame = device.frame()?;
            self.encode_into(&frame, region)?;
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
    /// Where the CUDA twin answers a boolean, this answers `eta_exec::Status`
    /// — the state, the fault word and the two site words — so a refusal can
    /// be NAMED (`eta_exec::describe_fault`) instead of merely counted.
    ///
    /// # Errors
    ///
    /// Whatever the readback said.
    pub fn status(&self) -> Result<Status> {
        let mut bytes = [0u8; eta_exec::STATUS_BYTES];
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
            dtype: Dtype::I32,
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
            dtype: Dtype::Bool,
        };
        assert_eq!(shape.cell_bytes(), 1);
        assert_eq!(shape.cell_stride(), CELL_ALIGN);
        // And a wide cell is not padded past its own alignment.
        let wide = ChannelShape {
            capacity: 1,
            numel: 8,
            dtype: Dtype::I32,
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
