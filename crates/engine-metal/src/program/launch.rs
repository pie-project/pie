//! One stage, encoded: the device rings a guest's channels live in, the
//! buffers one fire binds, and the dispatch that runs a region.
//!
//! **THERE ARE TWO SEATS HERE, AND THE SECOND ONE IS THE LANE TABLE THE
//! PARAGRAPH BELOW SAYS DISAPPEARS.** That paragraph is still true of the M2
//! fused form, which is what this file carried alone; [`Grouped`] is the
//! other, and it exists because the argument-slot ABI has a ceiling the
//! corpus reaches (twelve channels) and a width it cannot exceed (one
//! thread). The two are carved side by side for every stage and chosen per
//! region by [`super::compile`], because the emitter emits per region and a
//! stage may legitimately mix them.
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
use std::sync::Arc;

use eta_compiler::codegen::launch::{LaunchChannel, LaunchStagePlan};
use eta_exec::{
    Extents, LANE_ABI_VERSION, LaneChannelSlot, LaneHeader, LaneRecord, LaneShape, NO_TICKET,
    OpParams, OpRuntime, SCRATCH_ALIGN, Status, ValueDesc, describe, layout,
};
use eta_ir::Dtype;
use eta_ir::op::{IntrinsicId, tags};

use crate::device::ctx::Frame;
use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};

use super::compile::{Form, Region};
use super::shared::SharedRing;

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

/// How wide one element of the rectangle `lane.logits_base` points at is.
///
/// **TWO, AND UNCONDITIONALLY SO, BECAUSE THIS IS THE READOUT'S ELEMENT AND
/// NOT AN INTRINSIC'S.** The M2 form asks the slot table
/// (`m2_intrinsic_element_bytes`) because one argument index serves whichever
/// id was routed to it; the grouped form has one address per RECTANGLE, and
/// the rectangle `logits_base` names is the bf16 readout for both the trunk
/// and the draft column. The score plane is F32 and rides
/// `LaneRecord::attn_score_base` with a pitch of its own, so it is not
/// counted here and never was.
///
/// Named because it is the row stride draft columns are counted in — see
/// [`Prepared::regroup`], which converts a byte displacement between two
/// rectangles into whole rows of this element.
const INTRINSIC_ELEMENT_BYTES: u64 = 2;

/// Threads a grouped LIBRARY sampler's threadgroup must have.
///
/// The emitted nucleus and top-k kernels open with
/// `if (threads != 256u || layout->reserved1 == 0u) return;` — a width they
/// check rather than adapt to, because their radix passes size threadgroup
/// arrays as `256 * 16`. A dispatch at any other width runs no rows and
/// leaves the output at whatever the fire memset, which is a wrong answer
/// rather than a refusal. `the_library_samplers_still_demand_this_width`
/// holds this against the emitted source, since the emitter states it as a
/// literal inside an MSL string and has no constant to publish.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub(super) const LIBRARY_SAMPLER_THREADS: usize = 256;

/// Threads a grouped fused region's threadgroup gets, at most.
///
/// Read off the emitter rather than transcribed: the kernel sizes its argmax
/// reduction buffer to exactly this and faults `0xB3` on a wider launch. The
/// dispatch narrows it by the pipeline's own limit, which is the only
/// direction that is safe — and it IS safe, because the emitted argmax and
/// gather both stride by `m3_threads`.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub(super) const REGION_THREADS: u32 =
    eta_compiler::codegen::metal::fused::METAL_M3_REGION_THREADS;

/// One reservation's address at `offset`, or the refusal that says why the
/// grouped form cannot be handed one.
///
/// # Errors
///
/// [`Fault::Program`] for an offset with no GPU address — off Apple, where
/// there are no addresses at all, and for one outside the reservation.
fn address_of(buffer: &Buffer, offset: u64) -> Result<u64> {
    buffer.address_at(offset).ok_or_else(|| {
        Fault::program(
            "program::launch",
            format!(
                "offset {offset} of a {}-byte reservation has no GPU address, and the \
                 grouped form binds addresses rather than buffers",
                buffer.bytes()
            ),
        )
    })
}

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
    /// **The channels whose ring is not this instance's** — one entry per
    /// dense slot, `Some` for a device-only channel the engine registered and
    /// two passes share (design §5).
    ///
    /// The slab at that slot is a CLONE of the shared one, so every read and
    /// write below is already against the right bytes and needs no arm; what
    /// this list carries is the CURSOR, which is the other half of a ring and
    /// the half a `Vec<Cursor>` inside one session cannot be.
    shared: Vec<Option<Arc<SharedRing>>>,
}

impl Rings {
    /// Reserve one ring per shape: `capacity + 1` cells, the spare included.
    ///
    /// The spare is what makes "full" distinguishable from "empty" with two
    /// monotone cursors, which is the shared ring arithmetic's own rule
    /// (`eta_exec::channel`) and not this file's.
    ///
    /// **`adopted` IS WHAT MAKES A RING BELONG TO ITS CHANNEL** (design §5,
    /// and [`SharedRing`]'s own header). A dense slot with an adopted ring
    /// reserves nothing: it takes a retain of the one slab the channel owns,
    /// so the prefill's put and the decode's take address the same cells.
    /// Every other slot is cut here, per instance, exactly as before — which
    /// is right for the rings one pass owns, and that is most of them.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] when the device declined a reservation,
    /// [`Fault::Program`] for a shape whose ring will not fit a `u64`, and
    /// for an adopted ring cut at a geometry this instance does not declare.
    pub fn allocate(
        device: &Context,
        shapes: &[ChannelShape],
        adopted: &[Option<Arc<SharedRing>>],
    ) -> Result<Rings> {
        let mut slabs = Vec::with_capacity(shapes.len());
        let mut shared = Vec::with_capacity(shapes.len());
        for (channel, shape) in shapes.iter().enumerate() {
            if let Some(ring) = adopted.get(channel).and_then(Option::as_ref) {
                // **THE TWO DECLARATIONS HAVE TO AGREE, AND A DISAGREEMENT IS
                // A WRONG TOKEN RATHER THAN A FAULT.** The ring was cut at
                // the geometry the REGISTRATION stated and this instance
                // addresses it at the geometry its own package declares; if
                // those differ the cells are at one stride and the reads at
                // another, which no launch faults on.
                if ring.shape() != *shape {
                    return Err(Fault::program(
                        "program::launch",
                        format!(
                            "channel {channel}'s shared ring was cut for {} cell(s) of \
                             {:?} at capacity {} and this instance declares {} of {:?} \
                             at capacity {}: one ring addressed at two strides is a \
                             wrong cell and never a fault",
                            ring.shape().numel,
                            ring.shape().dtype,
                            ring.shape().capacity,
                            shape.numel,
                            shape.dtype,
                            shape.capacity
                        ),
                    ));
                }
                slabs.push(ring.slab());
                shared.push(Some(Arc::clone(ring)));
                continue;
            }
            let cells = u64::from(shape.capacity) + 1;
            let bytes = cells
                .checked_mul(shape.cell_stride() as u64)
                .ok_or_else(|| Fault::program("program::launch", "a ring past what a u64 counts"))?;
            slabs.push(Buffer::zeroed(device, bytes.max(1))?);
            shared.push(None);
        }
        Ok(Rings {
            slabs,
            shapes: shapes.to_vec(),
            shared,
        })
    }

    /// The ring channel `channel` shares with the other instances bound to
    /// it, or `None` for one this instance owns alone.
    #[must_use]
    pub fn shared(&self, channel: usize) -> Option<&Arc<SharedRing>> {
        self.shared.get(channel).and_then(Option::as_ref)
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
/// stride and the row offset ARE `setBuffer:offset:atIndex:`. The element
/// type never reaches the kernel at all — the emitted `0xA0` handler picks it
/// off the INTRINSIC ID rather than off any word the host sets — so it is a
/// thing the HOST checks rather than a thing the device is told, and
/// [`Prepared::bind_intrinsic`] checks it and keeps nothing.
///
/// **THE WIDTH IS THE ONE THAT STOPPED BEING A CHECK AND BECAME AN
/// ARGUMENT.** It is still argued with at bind, but the grouped form also
/// STRIDES by it: `M3GroupLayout::vocab` is the row pitch its gather
/// multiplies a row index by, which is this plane's answer to
/// `intrinsic_row_stride` and the reason a narrow read of more than one row
/// is expressible here at all. So it is kept.
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
    /// The rectangle's own row width, in elements.
    ///
    /// **KEPT NOW, AND IT USED TO BE ARGUED WITH AND DISCARDED.** The M2 form
    /// needs it only for the bounds argument at bind, because a bound offset
    /// carries the stride implicitly for the one row it can serve; the
    /// grouped form STRIDES by it — `GroupLayout::vocab`, written from HERE
    /// and no longer from what the reader declared — and reaches a second
    /// rectangle by counting rows of it. So it stops being a number nothing
    /// reads back.
    width: u32,
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
    /// `out0.len / logical_width`. Load-bearing because the two FORMS answer
    /// it differently: the grouped gather is told a row pitch and walks the
    /// rectangle's rows, the single-lane one walks `out0.len` consecutive
    /// elements and can only be right for one. See
    /// [`Prepared::bind_intrinsic`] and [`Prepared::no_stride_owed`].
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

// ─────────────────────────────────────────────────────────────────────────
// The grouped (M3) seat
// ─────────────────────────────────────────────────────────────────────────

/// `M3GroupLayout` — the grouped kernel's scalar arguments, in one record.
///
/// **THE HOST HALF OF A STRUCT THAT IS TEXT ON THE OTHER SIDE.** The three
/// lane-table structs are declared once in `eta_compiler::plan::lane_table`
/// and printed into MSL from `codegen::layout`, so neither side can drift.
/// These three words-and-a-half are not: `codegen::metal::preamble` emits
/// them as a literal, because the CUDA twin passes the same numbers as
/// separate kernel arguments and there was no host struct to pin. So this is
/// a second spelling, and the tie is
/// [`tests::the_group_layout_matches_the_emitted_struct`] — it reads the
/// emitted preamble and fails if a field is added, removed or reordered.
///
/// The three `reserved` words carry meaning the emitted kernels read, and
/// their names are the preamble's rather than their jobs'. Renaming them
/// would be a preamble change, which is a whole-corpus golden diff to buy
/// three better words.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct GroupLayout {
    /// Lanes this launch covers; a threadgroup past it returns.
    lane_count: u32,
    /// The per-lane stride of `all_descriptors`.
    value_count: u32,
    /// The per-lane stride of `all_scratch`.
    scratch_stride: u32,
    /// Where the temporary begins inside one lane's scratch.
    temporary_offset: u32,
    /// The row width of the rectangle `lane.logits_base` points at — the
    /// grouped gather's stride, which is the argument the M2 form does not
    /// have and the reason a narrow multi-row read is expressible here.
    vocab: u32,
    /// The per-lane stride of `channel_bindings`.
    reserved0: u32,
    /// Rows per lane the library samplers decompose their grid by:
    /// `dispatch_lane = threadgroup / reserved1`, `row = threadgroup %
    /// reserved1`. Zero makes those kernels return without running, so a
    /// region that is one of them must state it.
    reserved1: u32,
    /// The per-lane stride of `params`.
    reserved2: u32,
}

/// `M3RowMeta` — where one lane's rows live in `row_indices`.
///
/// Pinned the same way [`GroupLayout`] is, and for the same reason.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct RowMeta {
    /// This lane's first entry in `row_indices`.
    offset: u32,
    /// How many entries are the lane's.
    count: u32,
    /// Where the DRAFT rows begin, counted from `offset`. The trunk's rows
    /// are `[offset, offset + mtp_offset)` and the draft column's are the
    /// rest — one rectangle, two blocks, which is how this plane serves two
    /// intrinsics through the single `lane.logits_base`.
    mtp_offset: u32,
    /// Padding. Zero.
    reserved: u32,
}

/// Everything the grouped form binds that the single-lane form does not.
///
/// **ONE LANE, AND THE TABLE IS SHAPED FOR MANY ANYWAY.** A [`Session`] is
/// one instance and [`Prepared`] is one instance's stage, so every table here
/// is built at `lane_count = 1`. That is not the grouped form's ceiling — it
/// is where the SHELL's is: co-batching two instances into one launch needs a
/// frame admission this plane does not have, and inventing one to make the
/// lane count two would be a batching change wearing this one's clothes. What
/// the grouped form buys at one lane is already the whole of what it was
/// wanted for: the channels move out of the argument slots (so the
/// twelve-channel ceiling stops applying) and the region gets a threadgroup
/// instead of a thread (so a vocabulary-wide gather is 512-way rather than
/// serial). Every offset below still goes through [`LaneShape`], so the day
/// the admission lands the lane count is the only number that changes.
///
/// [`Session`]: super::Session
#[derive(Debug)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Grouped {
    /// The lane table: `LaneHeader`, then one `LaneRecord` per lane, then the
    /// flat `LaneChannelSlot` array. Bound whole at buffer 0; the kernel does
    /// its own casting off `sizeof` the same structs.
    table: Buffer,
    /// The table's geometry, so an offset is asked for rather than computed.
    shape: LaneShape,
    /// One [`GroupLayout`] per region of the stage, identical but for
    /// `reserved1`.
    ///
    /// **PER REGION BECAUSE `reserved1` IS**, and one shared record would be
    /// a host write racing a dispatch: every region of a stage is encoded
    /// into one command buffer before any of them runs, so a word rewritten
    /// between two encodes is a word rewritten under the first kernel. A
    /// thirty-two byte reservation per region is the cheaper half of that
    /// trade by a wide margin.
    layouts: Vec<Buffer>,
    /// Stage-local channel slot → the instance's dense channel index, which
    /// is what indexes the lane's slot window. The CUDA twin folds this into
    /// the emitted kernel as a constant; the Metal grouped kernel reads it,
    /// so it is a table.
    bindings: Buffer,
    /// One byte per (lane, dense channel): whether this fire has already put
    /// to that channel, which is how a grouped kernel keeps `current_k`
    /// across a threadgroup where the single-lane form keeps it in a
    /// register. Zeroed every fire.
    pending_flags: Buffer,
    /// Dispatch lane → lane-record index. Identity here; the indirection is
    /// the emitted kernel's, for a group whose members are not contiguous.
    lane_indices: Buffer,
    /// One [`RowMeta`] per lane.
    row_meta: Buffer,
    /// The rows of `lane.logits_base` this fire reads, trunk block first.
    row_indices: Buffer,
    /// Rows in the trunk block, which is also where the draft block begins.
    trunk_rows: u32,
    /// Rows in the draft block. Zero for a stage that reads no draft column.
    draft_rows: u32,
    /// The [`GroupLayout`] each entry of `layouts` holds, kept so a word can
    /// be changed without reading a device reservation back to find the
    /// other seven.
    layout_words: Vec<GroupLayout>,
    /// The one lane's record, kept for the same reason: a field that moves
    /// per fire is changed here and the whole record is written back, rather
    /// than patched at an offset this file would have to spell.
    record: LaneRecord,
}

/// Lanes one grouped launch of this shell covers. One — see [`Grouped`].
const GROUPED_LANES: u32 = 1;

/// The lane a single-instance grouped launch is.
const THE_LANE: u32 = 0;

impl Grouped {
    /// Carve and fill everything the grouped form binds that does not change
    /// per fire, and leave the per-fire words at the value a fire that never
    /// ran would have written.
    ///
    /// Answers `None` for a stage the planner itself says the grouped path
    /// cannot cover (`StageNeeds::grouped_valid`) — the emitter will have
    /// declined those regions too, so carving tables for them would be seven
    /// reservations nothing can ever bind.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the status word has no device address to hand
    /// the kernel, and whatever the reservations said.
    fn build(
        device: &Context,
        plan: &LaunchStagePlan,
        descriptors: &[ValueDesc],
        shape: LaneShape,
        status: &Buffer,
        extents: Extents,
        trunk_rows: u32,
        draft_rows: u32,
    ) -> Result<Option<Grouped>> {
        if !plan.needs.grouped_valid {
            return Ok(None);
        }
        let Some(bytes) = shape.bytes() else {
            return Err(Fault::program(
                "program::launch",
                "a lane table whose size does not fit a u64: the channel count and the \
                 lane count multiply past what an allocation can be",
            ));
        };
        let mut table = Buffer::zeroed(device, bytes)?;
        table.write(
            0,
            &record_bytes(&LaneHeader {
                abi_version: LANE_ABI_VERSION,
                lane_count: shape.lanes,
                channel_slots_per_lane: shape.channel_slots_per_lane,
                flags: 0,
            }),
        )?;
        // **THE COMMIT SLOT IS THE STATUS WORD, AS AN ADDRESS.** The M2
        // kernel takes it as `[[buffer(0)]]`; the grouped kernel's first act
        // is `reinterpret_cast<device M1Status*>(lane.commit_slot)`, so if
        // this is zero the very first load is a null dereference rather than
        // a refusal. It is resolved once here because the reservation lives
        // as long as the `Prepared` does.
        let commit_slot = status.address_at(0).ok_or_else(|| {
            Fault::program(
                "program::launch",
                "the status word has no GPU address, so a grouped kernel would have \
                 nothing to write its verdict through",
            )
        })?;
        let record = LaneRecord {
            commit_slot,
            channel_slot_offset: shape.slot_index(THE_LANE).unwrap_or(0),
            kv_len: extents.kv_len,
            page_count: extents.page_count,
            row_count: extents.row_count,
            token_count: extents.token_count,
            sampled_rows: extents.sampled_rows,
            query_len: extents.query_len,
            key_len: extents.key_len,
            ..LaneRecord::default()
        };
        let at = shape.record_offset(THE_LANE).ok_or_else(|| {
            Fault::program("program::launch", "the one lane is outside the lane table")
        })?;
        table.write(at, &record_bytes(&record))?;

        // The per-lane stride of `channel_bindings`, and the row the one lane
        // reads out of it. Stage-local slot `k` answers the dense channel the
        // instance carries it as, which is what indexes the slot window.
        let stride = u32::try_from(plan.channel_bindings.len()).map_err(|_| {
            Fault::program("program::launch", "more channels than a u32 can count")
        })?;
        let binding_bytes: Vec<u8> = (0..GROUPED_LANES)
            .flat_map(|_| plan.channel_bindings.iter().copied())
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut bindings = Buffer::zeroed(device, binding_bytes.len().max(4) as u64)?;
        bindings.write(0, &binding_bytes)?;

        let flags = u64::from(GROUPED_LANES) * u64::from(shape.channel_slots_per_lane);
        let pending_flags = Buffer::zeroed(device, flags.max(1))?;

        let index_bytes: Vec<u8> = (0..GROUPED_LANES).flat_map(u32::to_le_bytes).collect();
        let mut lane_indices = Buffer::zeroed(device, index_bytes.len() as u64)?;
        lane_indices.write(0, &index_bytes)?;

        let row_meta = Buffer::zeroed(
            device,
            (GROUPED_LANES as u64) * size_of::<RowMeta>() as u64,
        )?;
        let rows = u64::from(trunk_rows) + u64::from(draft_rows);
        let row_indices = Buffer::zeroed(device, rows.max(1) * size_of::<u32>() as u64)?;

        // ── One layout per region. Everything but `reserved1` is the
        //    stage's; `reserved1` is how many rows a library sampler's grid
        //    decomposes by, and it is an UPPER BOUND taken off the region's
        //    own inputs rather than an exact count — those kernels guard
        //    `row >= input_desc.rows` themselves, so an over-dispatched
        //    threadgroup returns at its first branch. The bound is the
        //    region's rather than the stage's so a sampler beside a wide
        //    matrix does not dispatch the matrix's row count in empty
        //    threadgroups.
        let scratch_layout = layout(descriptors).map_err(|why| {
            Fault::program(
                "program::launch",
                format!("this fire's scratch does not fit: {why:?}"),
            )
        })?;
        let base = GroupLayout {
            lane_count: GROUPED_LANES,
            value_count: u32::try_from(plan.value_types.len()).unwrap_or(u32::MAX),
            scratch_stride: u32::try_from(scratch_layout.total).unwrap_or(u32::MAX),
            temporary_offset: u32::try_from(scratch_layout.temporary).unwrap_or(u32::MAX),
            vocab: 0,
            reserved0: stride,
            reserved1: 1,
            reserved2: u32::try_from(plan.ops.len()).unwrap_or(u32::MAX),
        };
        let mut layout_words = Vec::with_capacity(plan.fused.len());
        let mut layouts = Vec::with_capacity(plan.fused.len());
        for region in &plan.fused {
            let rows = region
                .inputs
                .iter()
                .filter_map(|&value| descriptors.get(value as usize))
                .map(|desc| desc.rows)
                .max()
                .unwrap_or(1)
                .max(1);
            let words = GroupLayout {
                reserved1: rows,
                ..base
            };
            let mut buffer = Buffer::zeroed(device, size_of::<GroupLayout>() as u64)?;
            buffer.write(0, &record_bytes(&words))?;
            layout_words.push(words);
            layouts.push(buffer);
        }

        Ok(Some(Grouped {
            table,
            shape,
            layouts,
            bindings,
            pending_flags,
            lane_indices,
            row_meta,
            row_indices,
            trunk_rows,
            draft_rows,
            layout_words,
            record,
        }))
    }

    /// Point every region's layout at a rectangle `vocab` elements wide.
    ///
    /// **THE STRIDE THE M2 FORM HAS NO WORD FOR.** The single-lane gather
    /// walks `out0.len` CONSECUTIVE elements off a bound offset, so a reader
    /// narrower than the rectangle can only be served for one row; the
    /// grouped gather takes the row width as data and indexes
    /// `row_indices[…] * vocab`, which is the same freedom the CUDA twin gets
    /// from `intrinsic_row_stride`.
    ///
    /// # Errors
    ///
    /// Whatever the writes said.
    fn set_vocab(&mut self, vocab: u32) -> Result<()> {
        for (words, buffer) in self.layout_words.iter_mut().zip(self.layouts.iter_mut()) {
            words.vocab = vocab;
            buffer.write(0, &record_bytes(words))?;
        }
        Ok(())
    }

    /// Write the one lane's `RowMeta` and the rows it names.
    ///
    /// **ONE RECTANGLE, TWO BLOCKS.** The trunk's rows come first, starting
    /// at row zero of `lane.logits_base`, and the draft column's follow,
    /// starting at `draft_base`; `mtp_offset` is where the second block
    /// begins, which is how a plane with ONE readout address serves two
    /// intrinsics. The M2 form spends a second argument index instead, and
    /// that is the whole difference between the two.
    ///
    /// # Errors
    ///
    /// Whatever the writes said.
    fn set_rows(&mut self, draft_base: u32) -> Result<()> {
        let bytes: Vec<u8> = (0..self.trunk_rows)
            .chain((0..self.draft_rows).map(|row| draft_base.saturating_add(row)))
            .flat_map(u32::to_le_bytes)
            .collect();
        if !bytes.is_empty() {
            self.row_indices.write(0, &bytes)?;
        }
        let meta = RowMeta {
            offset: 0,
            count: self.trunk_rows.saturating_add(self.draft_rows),
            mtp_offset: self.trunk_rows,
            reserved: 0,
        };
        self.row_meta.write(0, &record_bytes(&meta))
    }

    /// Point the lane at the rectangle its readout lives in.
    ///
    /// # Errors
    ///
    /// Whatever the write said.
    fn set_logits(&mut self, base: u64, row_offset: u32, row_count: u32) -> Result<()> {
        self.record.logits_base = base;
        self.record.logits_row_offset = row_offset;
        self.record.logits_row_count = row_count;
        self.write_record()
    }

    /// Point the lane at its block of the observability slab.
    ///
    /// **THE SECOND RECTANGLE, AND THE WHOLE OF WHAT THIS WAVE ADDED TO THE
    /// SHARED RECORD.** The draft column is reached by counting rows off
    /// `logits_base` because it IS that rectangle in a second row block; the
    /// score slab is `crate::scores`'s own reservation, so no displacement
    /// off the readout lands in it and the grouped form had nothing to
    /// dereference. Now it has an address and a pitch — the CUDA twin's
    /// `(intrinsic_base, intrinsic_row_stride)` pair, said in the one place a
    /// kernel with no per-intrinsic argument index can be told it.
    ///
    /// `base` is the LANE's block rather than the slab's origin: the offset
    /// `crate::scores::Scores::lane_base` answers is folded in before the
    /// address is taken, exactly as the M2 form folds it into
    /// `setBuffer:offset:atIndex:`. So the two forms are pointed at the same
    /// first byte by construction rather than by two agreeing computations.
    ///
    /// Zero for a lane with no score rectangle bound, which is the value a
    /// fresh record already holds; the emitted gather faults on it rather
    /// than dereferencing, because a lane that did not capture has no block
    /// and the previous fire's mass is a wrong answer.
    ///
    /// # Errors
    ///
    /// Whatever the write said.
    fn set_scores(&mut self, base: u64, row_stride: u32) -> Result<()> {
        self.record.attn_score_base = base;
        self.record.attn_score_row_stride = row_stride;
        self.write_record()
    }

    /// Write the one lane's record back whole.
    ///
    /// Whole rather than patched at a field offset: the offsets belong to
    /// `eta_compiler::codegen::layout`, and a byte range spelled here would be
    /// a seventh copy of the lane table exactly where six were just collapsed
    /// into one.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the lane is outside its own table, and whatever
    /// the write said.
    fn write_record(&mut self) -> Result<()> {
        let at = self.shape.record_offset(THE_LANE).ok_or_else(|| {
            Fault::program("program::launch", "the one lane is outside the lane table")
        })?;
        self.table.write(at, &record_bytes(&self.record))
    }

    /// Resolve this fire's channel cells into the lane's slot window, and
    /// clear the put flags the last fire left.
    ///
    /// **THE CELLS ARE ADDRESSES HERE AND BINDINGS IN THE OTHER FORM**, which
    /// is the whole of why the twelve-channel ceiling does not apply: a slot
    /// is a row of this table rather than a `[[buffer(n)]]`, and Metal's
    /// argument space stops being the thing that runs out.
    ///
    /// The two ticket words stay [`NO_TICKET`]. They are what a DEVICE
    /// readiness kernel would check the host's observation against, and this
    /// plane gates on the host — the module header's argument — so a ticket
    /// here would be a number written for a kernel nobody dispatches.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when a cell has no device address, and whatever the
    /// writes said.
    fn refresh(&mut self, bindings: &[u32], bound: &[Bound]) -> Result<()> {
        let bytes = self.pending_flags.bytes();
        self.pending_flags.zero_span(0, bytes)?;
        for (&dense, cell) in bindings.iter().zip(bound) {
            let slot = LaneChannelSlot {
                committed_cell: address_of(&cell.slab, cell.committed)?,
                pending_cell: address_of(&cell.slab, cell.pending)?,
                expected_head: NO_TICKET,
                expected_tail: NO_TICKET,
            };
            let at = self.shape.slot_offset(THE_LANE, dense).ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    format!(
                        "channel {dense} is outside the lane table's slot window, which \
                         was carved for the channels this instance carries"
                    ),
                )
            })?;
            self.table.write(at, &record_bytes(&slot))?;
        }
        Ok(())
    }
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
    /// The grouped form's tables. Built whenever the stage has a chance of
    /// taking that path, which [`super::compile`] decides per region — this
    /// side carries the tables and the encoder picks.
    grouped: Option<Grouped>,
    /// Which intrinsics each FUSED region reads, as a bitmask over
    /// `IntrinsicId` ordinals, indexed by region.
    ///
    /// **BECAUSE A REFUSAL THAT NAMES A FORM HAS TO KNOW WHICH REGION TOOK
    /// IT.** Which form a region runs in is [`super::compile`]'s answer and
    /// it is not known at bind time — it depends on what the emitter emitted
    /// and on what the pipeline's register pressure allowed. So the shape
    /// only a strided form can serve is admitted at bind and held against
    /// the regions at ENCODE, where `Region::form` is in hand.
    region_intrinsics: Vec<u64>,
    /// Which intrinsics are bound to a rectangle WIDER than their readers'
    /// declared row, across more than one row — the one shape that needs a
    /// row stride and the M2 gather has none.
    strided: u64,
}

impl Prepared {
    /// Carve every buffer one stage needs, for a single-lane fire.
    ///
    /// **ONE LANE, IN BOTH FORMS NOW.** The M2 kernel is `if (gid != 0 …)
    /// return;` — one thread over one lane by construction, where the CUDA
    /// kernel reads `blockIdx.x`. The M3 grouped kernel reads
    /// `threadgroup_position_in_grid` and could serve many, and this shell
    /// still gives it one, because a `Session` is one instance: what changed
    /// is that the channels and the readout are now RAW ADDRESSES in a lane
    /// table rather than argument slots, which is what lifts the twelve-
    /// channel ceiling and hands the region a threadgroup. See [`Grouped`].
    ///
    /// The grouped tables are built for every stage that could take that
    /// path; which regions actually do is [`super::compile`]'s to say, and it
    /// says it per region because the emitter emits per region.
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

        // ── **THE GROUPED TABLES**, carved beside the argument-slot seats
        //    rather than instead of them. Which of the two a region takes is
        //    decided per region at compile time and read at encode; both are
        //    resident because a stage may mix them — a library sampler on the
        //    grouped path beside a region whose intrinsic only the M2 slot
        //    table can bind.
        let lanes = LaneShape::of(
            GROUPED_LANES,
            u32::try_from(shapes.len()).map_err(|_| {
                Fault::program("program::launch", "more channels than a u32 can count")
            })?,
        );
        // The two row blocks `row_indices` carries. The trunk's is what this
        // stage's own readers declared; the draft column's is the larger of
        // what they declared and what the plan says the stage reads, because
        // `mtp_drafts` answers `[k]` token ids and declares nothing about the
        // rows it argmaxed them out of.
        // A reader whose output is RANK ONE declares nothing — a `[vocab]`
        // gather says how many elements it wants and not how many rows they
        // came from — so a stage that has such a reader and no other gets a
        // block of one row rather than of none. Zero rows would make the
        // grouped gather refuse geometry the single-lane one serves.
        let reads = |wanted: IntrinsicId| {
            plan.ops
                .iter()
                .any(|op| op.intrinsic.is_some_and(|id| id as usize == wanted as usize))
        };
        let trunk_rows = declared[IntrinsicId::Logits as usize]
            .map_or(0, |it| it.rows)
            .max(u32::from(reads(IntrinsicId::Logits)));
        let draft_rows = declared[IntrinsicId::MtpLogits as usize]
            .map_or(0, |it| it.rows)
            .max(plan.mtp_rows)
            .max(u32::from(
                reads(IntrinsicId::MtpLogits) || reads(IntrinsicId::MtpDrafts),
            ));
        let grouped = Grouped::build(
            device,
            plan,
            &descriptors,
            lanes,
            &status,
            extents,
            trunk_rows,
            draft_rows,
        )?;

        // Which intrinsics each fused region reads. `region.nodes` are op
        // indices, and `super::compile::grouped_region` asks this same
        // question of the same two tables to decide the form — so the two
        // agree by construction rather than by a comment.
        let region_intrinsics = plan
            .fused
            .iter()
            .map(|region| {
                region.nodes.iter().fold(0u64, |mask, &node| {
                    match plan.ops.get(node as usize).and_then(|op| op.intrinsic) {
                        Some(id) if (id as u32) < u64::BITS => mask | (1u64 << (id as u32)),
                        _ => mask,
                    }
                })
            })
            .collect();

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
            grouped,
            region_intrinsics,
            strided: 0,
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
        // The same cells the argument slots above hold, as addresses in the
        // lane table. Written whether or not any region of this stage takes
        // the grouped path: a stage that mixes the two forms would otherwise
        // hand the grouped one a table describing the fire before last.
        if let Some(grouped) = self.grouped.as_mut() {
            grouped.refresh(&self.bindings, &self.bound)?;
        }
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
    /// than the width the stage's own readers declared, and for a multi-row
    /// read whose declared width is not the rectangle's WHERE NO FORM ON
    /// THIS STAGE CAN STRIDE — the grouped one can and serves it; the
    /// single-lane one cannot and refuses it at encode
    /// ([`Prepared::no_stride_owed`]), where which form won is known.
    /// [`Fault::Ceiling`] for a rectangle the reader would walk off the end
    /// of.
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
            // ── **AND THE MULTI-ROW READ IS WHERE THE TWO FORMS PART**,
            //    which is a narrower statement than the one that stood here
            //    and had to say "the two PLANES". CUDA carries a stride per
            //    binding, so it walks `first_row + i / logical_width` at
            //    `stride`. The M2 `0xA0` handler still reads `out0.len`
            //    CONSECUTIVE elements off `setBuffer:offset:` and has no
            //    stride to be told, so for it only a ONE-ROW narrow read
            //    lands on the bytes the CUDA twin would return — a `k`-row
            //    verifier would have row 1 start `declared.width` elements
            //    in rather than `width`, sliding every row after the first.
            //
            //    The GROUPED form does carry one. `M3GroupLayout::vocab` is
            //    a per-region row pitch the kernel already multiplies a row
            //    index by, and the only reason it could not express this was
            //    that the host wrote the READER's width into it and the
            //    emitted gather spent that one number on the pitch and the
            //    row width both. Those are now two numbers — see
            //    [`Prepared::regroup`] and `codegen::metal::fused` — so the
            //    shape is expressible wherever a region takes that form, and
            //    `super::compile::grouped_region` sends every region that
            //    reads a gatherable intrinsic there.
            //
            //    Which form a region actually took is not known HERE: it
            //    depends on what the emitter emitted and what the pipeline
            //    admitted. So what is refused at bind is only what no form
            //    could serve — an intrinsic the grouped emitter declines by
            //    name, or a stage with no grouped seat at all. The first of
            //    those is empty today: `AttnScore` was the whole of it, and
            //    it has `lane.attn_score_base` now. The rest is recorded and
            //    held against the regions at encode, where `Region::form` is
            //    in hand.
            if declared.width < width && declared.rows > 1 && !self.strideable(intrinsic) {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "this stage reads {intrinsic:?} as {} rows of {} elements out of a \
                         rectangle {width} wide, and every form that can run it walks its \
                         rows consecutively: it has no row stride to be told, so every row \
                         after the first would land {} elements short. A narrower read \
                         of ONE row is served; this one needs the grouped form, which \
                         carries a row pitch the way the CUDA handler's \
                         `intrinsic_row_stride` does — and this stage has no grouped seat \
                         for it, either because the plan said that path cannot cover the \
                         stage or because the emitter binds no address for {intrinsic:?}",
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
        // **WHAT THE ENCODER WILL NEED TO KNOW.** A rebind of the same
        // intrinsic to a rectangle of a different width has to CLEAR this as
        // readily as it sets it, which is why it is recomputed rather than
        // or-ed in: an attached fire rebinds every step.
        let bit = 1u64 << (intrinsic as u32);
        if declared.is_some_and(|it| it.width < width && it.rows > 1) {
            self.strided |= bit;
        } else {
            self.strided &= !bit;
        }
        self.intrinsics[intrinsic as usize] = Some(Slot {
            base: base.clone(),
            offset,
            width,
        });
        self.regroup()
    }

    /// Whether a narrow multi-row read of `intrinsic` has a form that serves
    /// it — a grouped seat on this stage, and an id the grouped emitter will
    /// bind.
    ///
    /// **BOTH HALVES, BECAUSE EITHER ONE MISSING PUTS EVERY READER BACK ON
    /// THE SINGLE-LANE FORM.** `m3_intrinsic_bindable` is the emitter's own
    /// rule and it declines nothing now that the score rectangle has a
    /// lane-record address of its own — but it is still ASKED rather than
    /// assumed, because the day a new id gains an M2 argument index and no
    /// grouped route, the shape only a strided form can serve has to be
    /// refused at bind rather than encoded. The other half stands unchanged:
    /// a stage the planner said the grouped path cannot cover has no tables
    /// carved at all (`StageNeeds::grouped_valid`, [`Grouped::build`]).
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    fn strideable(&self, intrinsic: IntrinsicId) -> bool {
        self.grouped.is_some()
            && eta_compiler::codegen::metal::m3_intrinsic_bindable(intrinsic as u16)
    }

    /// Refuse a SINGLE-LANE region that reads an intrinsic bound at a shape
    /// only a strided form can serve.
    ///
    /// **THE HALF OF `bind_intrinsic`'S REFUSAL THAT KNOWS WHICH FORM WON.**
    /// A region whose intrinsic the grouped emitter declined, or whose
    /// grouped pipeline the device would not build, falls back here — and
    /// its gather is the consecutive one, so the read has to be named rather
    /// than encoded. A region index this stage cannot place is refused too:
    /// not knowing which intrinsics a region reads is not a reason to
    /// believe it reads none.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] naming the intrinsic and the form.
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    fn no_stride_owed(&self, region: &Region) -> Result<()> {
        let reads = self
            .region_intrinsics
            .get(region.region_index as usize)
            .copied()
            .unwrap_or(u64::MAX);
        if reads & self.strided == 0 {
            return Ok(());
        }
        Err(Fault::program(
            "program::launch",
            format!(
                "region {} of this stage reads a rectangle wider than the rows its \
                 own readers declared, across more than one row, and it compiled to \
                 the SINGLE-LANE form, whose gather walks `out0.len` consecutive \
                 elements off the binding and has no row pitch to be told. The \
                 grouped form carries one; this region did not get it",
                region.region_index
            ),
        ))
    }

    /// Re-derive the grouped form's rectangle words from the slot table.
    ///
    /// **THE TWO FORMS DISAGREE ABOUT WHAT A SECOND RECTANGLE IS**, and this
    /// is where the disagreement is resolved rather than papered over. The M2
    /// form gives each intrinsic its own argument index, so any two rectangles
    /// may be any two unrelated allocations. The grouped kernels are handed
    /// ADDRESSES, and the lane record carries two of them — so which
    /// disagreement applies depends on which pair is asked about:
    ///
    /// * `logits` and `mtp_logits` share `lane.logits_base` and the draft
    ///   column is reached by counting rows off it, so those two have to be
    ///   one reservation at a row-aligned displacement. When they are, this
    ///   states the displacement and the grouped path serves both columns;
    ///   when they are not, it leaves the draft block pointing at the trunk's
    ///   first row and [`super::compile`]'s choice keeps a region reading the
    ///   draft column on the form that can bind it.
    /// * `attn_score` has `lane.attn_score_base` to itself, so there is no
    ///   displacement to compute and nothing to agree with. The slab and the
    ///   readout are unrelated allocations by construction (`crate::scores`
    ///   owns one, the arena the other) and the record now says so instead of
    ///   the emitter refusing to.
    ///
    /// A stage with no trunk rectangle writes no readout words: `vocab` stays
    /// zero, and every emitted grouped gather refuses a zero vocabulary with
    /// `FUSED_GEOMETRY_MISMATCH` rather than striding by it. The score words
    /// are written regardless, for the reason spelled at the head of the body.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the readout or the score slab has no GPU
    /// address; whatever the writes said.
    fn regroup(&mut self) -> Result<()> {
        if self.grouped.is_none() {
            return Ok(());
        }
        // ── **THE SCORE RECTANGLE IS DERIVED FIRST, AND ON ITS OWN.** It is a
        //    separate reservation with a separate pitch, so nothing about it
        //    is a function of the readout — and an epilogue that reads scores
        //    and no logits is a legal program (`attn-score.md` §4: only
        //    decisions cross to the host, and a policy that only ranks keys
        //    never touches the vocabulary). Deriving it below the trunk's
        //    early return would leave exactly that program with a zero base
        //    and a device-side `FUSED_GEOMETRY_MISMATCH` for a rectangle the
        //    shell had already bound.
        //
        //    Zero and zero for an unbound rectangle, and written rather than
        //    skipped: a rebind is the normal case on an attached fire, so a
        //    slot that stops holding a rectangle has to CLEAR the record's
        //    words as readily as a bind sets them.
        let (score_base, score_stride) =
            match self.intrinsics[IntrinsicId::AttnScore as usize].as_ref() {
                Some(slot) => (address_of(&slot.base, slot.offset)?, slot.width),
                None => (0, 0),
            };
        self.grouped
            .as_mut()
            .expect("the seat was there one statement ago")
            .set_scores(score_base, score_stride)?;
        let Some(trunk) = self.intrinsics[IntrinsicId::Logits as usize].as_ref() else {
            return Ok(());
        };
        // ── **THE STRIDE IS THE RECTANGLE'S ROW, AND IT USED TO BE THE
        //    READER'S.** It was the reader's for parity: the single-lane
        //    `0xA0` handler walks `out0.len` CONSECUTIVE elements off the
        //    bound offset, so its row stride IS `out0.last` whatever the
        //    rectangle underneath is that wide, and writing the declared
        //    width here made the grouped gather land on the same bytes.
        //
        //    That bought byte-identity by spending the one argument this
        //    form has and the other does not. `layout->vocab` is the M3
        //    kernel's `intrinsic_row_stride`; the emitted gather now takes
        //    its ROW WIDTH from `intrinsic_desc.last` — the reader's own
        //    claim, which is where the M2 form reads it too — and its PITCH
        //    from here, so the two numbers stop being one. Where they agree
        //    nothing moves and the two forms are byte-identical exactly as
        //    before; where they differ this plane can finally express what
        //    `intrinsic_row_stride` expresses, which is a narrow read of
        //    more than one row.
        let width = trunk.width;
        let logits_base = address_of(&trunk.base, trunk.offset)?;
        // Where the draft block starts, as a row of the trunk's rectangle.
        // Zero — the trunk's own first row — whenever the two are not one
        // reservation at a whole number of rows apart, which is the case a
        // grouped kernel cannot express and is refused at emit rather than
        // bound wrong here.
        let stride = u64::from(trunk.width) * INTRINSIC_ELEMENT_BYTES;
        let draft_base = self.intrinsics[IntrinsicId::MtpLogits as usize]
            .as_ref()
            .filter(|drafts| drafts.width == trunk.width && stride != 0)
            .and_then(|drafts| {
                let there = address_of(&drafts.base, drafts.offset).ok()?;
                let apart = there.checked_sub(logits_base)?;
                (apart % stride == 0)
                    .then(|| u32::try_from(apart / stride).ok())
                    .flatten()
            })
            .unwrap_or(0);
        let grouped = self
            .grouped
            .as_mut()
            .expect("the seat was there one statement ago");
        let rows = grouped.trunk_rows.saturating_add(grouped.draft_rows);
        grouped.set_vocab(width)?;
        grouped.set_logits(logits_base, 0, rows)?;
        grouped.set_rows(draft_base)
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

            if region.form != Form::Fused {
                return self.encode_grouped(frame, region);
            }
            self.no_stride_owed(region)?;
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

    /// Encode one GROUPED region: eleven bindings, a residency declaration
    /// per reservation an address reaches, and a threadgroup per lane.
    ///
    /// **THE ELEVEN ARE FIXED AND THE CHANNELS ARE NOT AMONG THEM.** Where
    /// the M2 form spends two argument indices per channel and runs out at
    /// twelve, every channel here is a row of the lane table at buffer 0 —
    /// so the signature of a sixteen-channel region is the signature of a
    /// one-channel region, and the ceiling that shaped this plane's corpus
    /// stops applying. What replaces it is Metal's own limit on the table
    /// (`METAL_M1_MAX_CHANNELS`, twenty-nine), which the single-lane
    /// readiness and commit kernels hit first and which no guest in the
    /// corpus approaches.
    ///
    /// **AND RESIDENCY IS NOW THE SHELL'S JOB.** `setBuffer:` tells Metal a
    /// reservation is used; a `ulong` inside a table tells it nothing. So
    /// every buffer whose ADDRESS escaped into the lane record — the status
    /// word the verdict is written through, each ring the cells live in, the
    /// readout rectangle, and the observability slab the score base points
    /// into — is declared with `useResource:usage:` on this encoder, which is
    /// the same bookkeeping `icb.rs` does for its slabs and `rebind.rs` for
    /// the reservations behind its argument buffer. A missing declaration is
    /// not a validation error; it is a page that may not be resident when the
    /// kernel reads it.
    ///
    /// The slab needs no line of its own: the loop over `self.intrinsics`
    /// below walks every rectangle this stage has been pointed at, and the
    /// score slab is one of them — it is bound through the same slot table
    /// whichever form ends up reading it, which is precisely why the two
    /// forms cannot be pointed at different bytes.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the stage carries no grouped tables — which
    /// means [`super::compile`] chose a form [`Prepared::build`] did not
    /// carve for — and [`Fault::Deviceless`] off Apple.
    #[cfg(target_vendor = "apple")]
    fn encode_grouped(&self, frame: &Frame, region: &Region) -> Result<()> {
        use objc2::runtime::ProtocolObject;
        use objc2_metal::{
            MTLComputeCommandEncoder, MTLComputePipelineState, MTLResource, MTLResourceUsage,
            MTLSize,
        };

        let grouped = self.grouped.as_ref().ok_or_else(|| {
            Fault::program(
                "program::launch",
                "a region was compiled for the grouped form and this stage carries no \
                 lane table; the plan said the grouped path could not cover it",
            )
        })?;
        let layout = grouped
            .layouts
            .get(region.region_index as usize)
            .ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    format!(
                        "region {} has no group layout, so its library sampler would \
                         decompose its grid by a row count nobody stated",
                        region.region_index
                    ),
                )
            })?;

        let encoder = frame.encoder();
        encoder.setComputePipelineState(region.pipeline());
        // SAFETY: every reservation below is retained by `self` for the length
        // of this call, and every offset is zero — the grouped kernel strides
        // its own tables off `layout` rather than off a bound offset.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(grouped.table.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(self.descriptors.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(self.params.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(self.offsets.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(self.scratch.raw()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(layout.raw()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(grouped.bindings.raw()), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(grouped.pending_flags.raw()), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(grouped.lane_indices.raw()), 0, 8);
            encoder.setBuffer_offset_atIndex(Some(grouped.row_meta.raw()), 0, 9);
            encoder.setBuffer_offset_atIndex(Some(grouped.row_indices.raw()), 0, 10);
        }

        let resident = |buffer: &Buffer, usage: MTLResourceUsage| {
            let resource: &ProtocolObject<dyn MTLResource> =
                ProtocolObject::from_ref(&**buffer.slab());
            encoder.useResource_usage(resource, usage);
        };
        resident(
            &self.status,
            MTLResourceUsage::Read | MTLResourceUsage::Write,
        );
        for cell in &self.bound {
            resident(
                &cell.slab,
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
        }
        for held in self.intrinsics.iter().flatten() {
            resident(&held.base, MTLResourceUsage::Read);
        }

        // **ONE FORM'S WIDTH IS EXACT AND THE OTHER'S IS A CEILING.** A
        // library sampler opens with `if (threads != 256u …) return;` — it
        // does not adapt, it declines, and `compile::grouped_region` is where
        // a pipeline that cannot give it 256 is sent back to the M2 form. A
        // grouped fused region strides its argmax by `m3_threads` and its
        // gather by the same, so it is correct at any width up to the
        // reduction buffer's; the buffer is sized for
        // `METAL_M3_REGION_THREADS` and the kernel faults `0xB3` above it, so
        // the launch takes the NARROWER of that and what this pipeline will
        // accept. A large region measures well under 512 — the register
        // pressure is the kernel's own — and running it 384 threads wide is
        // the whole point of asking.
        let rows = grouped
            .layout_words
            .get(region.region_index as usize)
            .map_or(1, |words| words.reserved1 as usize);
        let (groups, threads) = match region.form {
            Form::Fused => unreachable!("`encode_into` routes the single-lane form"),
            Form::GroupedLibrary => ((GROUPED_LANES as usize) * rows, LIBRARY_SAMPLER_THREADS),
            Form::Grouped => (
                GROUPED_LANES as usize,
                region
                    .pipeline()
                    .maxTotalThreadsPerThreadgroup()
                    .clamp(1, REGION_THREADS as usize),
            ),
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups.max(1),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
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
            shared: vec![None],
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
            shared: Vec::new(),
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

    /// The one field list in the emitted preamble that this file declares a
    /// second time.
    ///
    /// The three lane-table structs are printed from
    /// `eta_compiler::codegen::layout` and pinned there with `offset_of!`, so
    /// they cannot drift. `M3GroupLayout` and `M3RowMeta` are text in
    /// `codegen::metal::preamble` — the CUDA twin passes the same numbers as
    /// separate kernel arguments, so there was no host struct to pin them to
    /// — and [`GroupLayout`] / [`RowMeta`] here are the host halves. A field
    /// added on one side and not the other would shift every field after it:
    /// the kernel would read `vocab` where the host wrote `temporary_offset`,
    /// with no error anywhere. So the two are compared as text.
    #[test]
    fn the_group_layout_matches_the_emitted_struct() {
        let preamble = eta_compiler::codegen::metal::preamble::grouped_preamble();
        let fields = |name: &str| -> Vec<String> {
            let open = format!("struct {name} {{");
            let at = preamble
                .find(&open)
                .unwrap_or_else(|| panic!("the grouped preamble declares no `{name}`"))
                + open.len();
            let body = &preamble[at..at + preamble[at..].find("};").expect("a closed struct")];
            body.split(';')
                .filter_map(|field| {
                    let field = field.trim();
                    (!field.is_empty()).then(|| field.to_string())
                })
                .collect()
        };
        assert_eq!(
            fields("M3GroupLayout"),
            [
                "uint lane_count",
                "uint value_count",
                "uint scratch_stride",
                "uint temporary_offset",
                "uint vocab",
                "uint reserved0",
                "uint reserved1",
                "uint reserved2",
            ],
            "`GroupLayout` in this file is the host half of the emitted \
             `M3GroupLayout`, and the two have parted"
        );
        assert_eq!(size_of::<GroupLayout>(), 8 * size_of::<u32>());
        assert_eq!(
            fields("M3RowMeta"),
            ["uint offset", "uint count", "uint mtp_offset", "uint reserved"],
            "`RowMeta` in this file is the host half of the emitted `M3RowMeta`, \
             and the two have parted"
        );
        assert_eq!(size_of::<RowMeta>(), 4 * size_of::<u32>());
    }

    /// The eleven bindings every grouped kernel takes, and the one width two
    /// of them refuse to run at any other value of.
    ///
    /// [`Prepared::encode_grouped`] binds ONE signature for all three forms
    /// and dispatches two widths. Both halves of that are read off the
    /// emitter's own text here rather than trusted: a sampler that grew a
    /// twelfth argument would be handed eleven and read whatever the last
    /// dispatch left at the missing index, and one dispatched at the wrong
    /// width does not fault — it returns, leaves the output at whatever the
    /// fire memset, and answers a confident zero.
    #[test]
    fn the_grouped_samplers_take_the_bindings_this_file_writes() {
        // In binding order, so a reader sees the ABI rather than a count.
        const BOUND: [&str; 11] = [
            "lane_bytes",
            "all_descriptors",
            "params",
            "offsets",
            "all_scratch",
            "layout",
            "channel_bindings",
            "pending_flags",
            "lane_indices",
            "all_row_meta",
            "row_indices",
        ];
        let demanded = format!("threads != {LIBRARY_SAMPLER_THREADS}u");
        for signature in [
            eta_compiler::codegen::metal::nucleus::SIGNATURE,
            eta_compiler::codegen::metal::topk::SIGNATURE,
        ] {
            for (index, name) in BOUND.iter().enumerate() {
                assert!(
                    signature.contains(&format!("{name} [[buffer({index})]]")),
                    "a grouped library sampler does not take `{name}` at buffer \
                     {index}, and `encode_grouped` binds it there"
                );
            }
            assert!(
                signature.contains(&demanded),
                "a grouped library sampler no longer refuses every width but \
                 {LIBRARY_SAMPLER_THREADS}, so this dispatch's width is a guess"
            );
        }
    }
}
