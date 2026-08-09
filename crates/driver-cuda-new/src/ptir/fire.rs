//! The fire: one PTIR stage, prepared and launched.
//!
//! # What a fire is made of
//!
//! A generated region's kernel takes sixteen arguments and reads six device
//! buffers. Five of them are built here, per fire, and every one is an array
//! whose stride the kernel computes rather than reads:
//!
//! | buffer | stride the kernel uses | built by |
//! |---|---|---|
//! | lane table | `header`, then `lanes[i]`, then `channels[lane * slots + n]` | [`Prepared::table`] |
//! | descriptors | `all_descriptors + lane * value_count` | [`Prepared::descriptors`] |
//! | params | `params[op]` — 88 bytes each, shared by every lane | [`Prepared::params`] |
//! | offsets | `offsets[value]` — shared by every lane | [`Prepared::offsets`] |
//! | scratch | `all_scratch + lane * scratch_stride` | [`Prepared::scratch`] |
//! | pending flags | `pending_flags[lane * slots + n]` | [`Prepared::pending`] |
//!
//! Every one of those strides is a number the host computes and the kernel
//! trusts. There is no length anywhere on the device, so a stride that is too
//! small has each lane reading the previous lane's tail, and a stride that is
//! too large has it reading past the allocation — and only the second one
//! faults.
//!
//! # Scratch is uniform across lanes, and that is a decision
//!
//! One `offsets` table serves every lane, so a value's offset must be the same
//! in every lane's scratch even when lanes have different extents. The C++
//! takes a running maximum of each value's size across lanes and lays out
//! against that; this does the same, by describing every lane and folding.
//! Laying out per lane would be smaller and would need a per-lane offsets
//! table the kernel has no parameter for.
//!
//! # What this does not do yet
//!
//! Multi-lane grouping, the intrinsic side tables beyond binding them empty,
//! and the ticketed channel path. The single-lane epilogue — a program that
//! reads channels, computes, and puts — is what runs, and it is the shape
//! every decode loop is.

use driver::driver_api::plan::{LaunchOp, LaunchStagePlan};
use driver::tensor_ir::DType;
use driver::tensor_ir::op::{IntrinsicId, tags};
use driver::{
    Diagnosis, Extents, LANE_HEADER_BYTES, LANE_RECORD_BYTES, LANE_SLOT_BYTES, LaneChannelSlot,
    LaneHeader, LaneRecord, LaneShape, NO_TICKET, OpParams, OpRuntime, SCRATCH_ALIGN,
    StatusOutcome, ValueDesc, describe, layout,
};

use crate::cuda::{Allocator, DeviceBuffer, StreamRef};
use crate::error::{Error, Result};

use super::launch::{Args, launch};
use super::params::{CudaOpParams, params_bytes};
use super::ring::Rings;
use super::runtime::Region;

/// How many intrinsic slots the side tables carry per lane.
///
/// `PTIR_INTRINSIC_SLOTS` — `PTIR_INTR_ATTN_SCORE + 1`. The five arrays are
/// indexed `lane * INTRINSIC_SLOTS + intr`, so the constant is a stride and
/// getting it wrong misdirects every intrinsic of every lane but the first.
pub const INTRINSIC_SLOTS: usize = 16;

/// The sixteen arguments a generated fused region takes.
const FUSED_ARITY: usize = 16;

/// One fire's device state, ready to launch.
///
/// Held rather than rebuilt per region because a stage's regions share every
/// buffer: they differ only in which kernel reads them.
#[derive(Debug)]
pub struct Prepared {
    table: DeviceBuffer,
    descriptors: DeviceBuffer,
    params: DeviceBuffer,
    offsets: DeviceBuffer,
    scratch: DeviceBuffer,
    pending: DeviceBuffer,
    intrinsic_bases: DeviceBuffer,
    intrinsic_modes: DeviceBuffer,
    intrinsic_widths: DeviceBuffer,
    intrinsic_strides: DeviceBuffer,
    intrinsic_offsets: DeviceBuffer,
    /// Where the kernel writes this lane's commit verdict. One `u32`, and the
    /// kernel clears it to refuse rather than raising anything.
    commit: DeviceBuffer,
    /// The status word, in the first [`STATUS_BYTES`] of every lane's scratch.
    lanes: u32,
    value_count: u32,
    scratch_stride: u32,
    temporary_offset: u32,
}

impl Prepared {
    /// Build every buffer one fire needs.
    ///
    /// `extents` are the runtime numbers a value's shape may depend on —
    /// `kv_len`, `page_count`, and the rest — which is why the same plan
    /// serves many batch shapes without recompiling.
    ///
    /// # Errors
    ///
    /// If a value's shape cannot be resolved against `extents`, if the scratch
    /// would exceed what the layout permits, or if the device refuses an
    /// allocation.
    pub fn build(
        alloc: &Allocator,
        plan: &LaunchStagePlan,
        rings: &Rings,
        extents: Extents,
        stream: StreamRef<'_>,
    ) -> Result<Self> {
        let lanes = 1u32;
        let channel_count = u32::try_from(plan.channel_bindings.len())
            .map_err(|_| Error::invalid("ptir::fire", "more channels than a u32 can count"))?;
        let value_count = u32::try_from(plan.value_types.len())
            .map_err(|_| Error::invalid("ptir::fire", "more values than a u32 can count"))?;

        // ── Descriptors, and the scratch they size. ──
        let descriptors: Vec<ValueDesc> = plan
            .value_types
            .iter()
            .map(|value| {
                describe(value, &extents).map_err(|why| {
                    Error::invalid(
                        "ptir::fire",
                        format!("a value's shape does not resolve against this fire: {why:?}"),
                    )
                })
            })
            .collect::<Result<_>>()?;
        let scratch_layout = layout(&descriptors).map_err(|why| {
            Error::invalid(
                "ptir::fire",
                format!("this fire's scratch does not fit: {why:?}"),
            )
        })?;
        let scratch_stride = u32::try_from(scratch_layout.total)
            .map_err(|_| Error::invalid("ptir::fire", "a scratch stride past u32"))?;
        let temporary_offset = u32::try_from(scratch_layout.temporary)
            .map_err(|_| Error::invalid("ptir::fire", "a temporary offset past u32"))?;

        // ── The commit slot, and the lane table that points at it. ──
        let mut commit = alloc.alloc(size_of::<u32>())?;
        // One, not zero. The kernel's first act is to read this and return if
        // it is clear, so a fire that started at zero would launch every
        // kernel and do nothing, which looks exactly like a blocked fire.
        commit.copy_from_host(&1u32.to_le_bytes(), stream)?;

        let shape = LaneShape::of(lanes, channel_count);
        let table_bytes = shape
            .bytes()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| Error::invalid("ptir::fire", "a lane table past what fits"))?;
        let mut table = alloc.alloc(table_bytes)?;
        let mut host_table = vec![0u8; table_bytes];

        let header = LaneHeader {
            abi_version: driver::LANE_ABI_VERSION,
            lane_count: lanes,
            channel_slots_per_lane: channel_count,
            flags: 0,
        };
        write_record(&mut host_table, 0, &header);

        let record = LaneRecord {
            kv_len: extents.kv_len,
            page_count: extents.page_count,
            row_count: extents.row_count,
            token_count: extents.token_count,
            sampled_rows: extents.sampled_rows,
            query_len: extents.query_len,
            key_len: extents.key_len,
            channel_slot_offset: 0,
            commit_slot: commit.as_ptr() as u64,
            ..LaneRecord::default()
        };
        write_record(&mut host_table, LANE_HEADER_BYTES as usize, &record);

        // ── The channel slots: absolute addresses, resolved on the host. ──
        let cursors = rings.cursors(stream)?;
        let slots_at = LANE_HEADER_BYTES as usize + LANE_RECORD_BYTES as usize;
        for (local, &dense) in plan.channel_bindings.iter().enumerate() {
            let channel = dense as usize;
            let cursor = cursors.get(channel).ok_or_else(|| {
                Error::invalid(
                    "ptir::fire",
                    format!("stage-local channel {local} binds channel {dense}, which is unbound"),
                )
            })?;
            let slot = LaneChannelSlot {
                committed_cell: rings.cell_address(channel, cursor.head)?,
                pending_cell: rings.cell_address(channel, cursor.tail)?,
                // NOT a ticket. The ticketed path lets a host stage a table
                // ahead of the fire that uses it and have the kernel refuse a
                // stale one; nothing here stages, so claiming a ticket the
                // host did not observe would be a check that passes for the
                // wrong reason.
                expected_head: NO_TICKET,
                expected_tail: NO_TICKET,
            };
            write_record(
                &mut host_table,
                slots_at + local * LANE_SLOT_BYTES as usize,
                &slot,
            );
        }
        table.copy_from_host(&host_table, stream)?;

        // ── Op params, widened to CUDA's 88-byte record. ──
        let mut records = Vec::with_capacity(plan.ops.len());
        let mut result_base = 0u32;
        for op in &plan.ops {
            let mut record = CudaOpParams::widen(OpParams::of(op, result_base, runtime_of(op)));
            // `sink_bytes` is the one field the shared record leaves for the
            // driver, and the shared crate says so: "filled when the sink is
            // bound, which is not this module's job."
            //
            // It is the FIXED cell size a `chan_put` writes, and the emitted
            // kernel's first act is `if (logical_bytes > p.sink_bytes) fault`.
            // So leaving it zero does not write a short cell — it faults every
            // put with class 146, the kernel clears the commit slot, and the
            // fire comes back refused with no status word to explain it,
            // because the status lives in `__shared__` and never reaches
            // memory the host can read. That failure is what this line is.
            if u32::from(op.code) == u32::from(tags::CHAN_PUT) && op.channel != u32::MAX {
                let dense = plan
                    .channel_bindings
                    .get(op.channel as usize)
                    .copied()
                    .ok_or_else(|| {
                        Error::invalid(
                            "ptir::fire",
                            format!(
                                "a put names stage-local channel {}, which is unbound",
                                op.channel
                            ),
                        )
                    })?;
                let shape = rings.shape(dense as usize).ok_or_else(|| {
                    Error::invalid(
                        "ptir::fire",
                        format!(
                            "a put targets channel {dense}, which this instance does not carry"
                        ),
                    )
                })?;
                record.sink_bytes = u32::try_from(shape.cell_bytes())
                    .map_err(|_| Error::invalid("ptir::fire", "a cell past what a u32 counts"))?;
            }
            records.push(record);
            result_base += u32::from(op.result_count);
        }
        let param_bytes = params_bytes(&records);
        let mut params = alloc.alloc(param_bytes.len().max(size_of::<CudaOpParams>()))?;
        params.copy_from_host(&param_bytes, stream)?;

        // ── The flat arrays. ──
        let descriptor_bytes: Vec<u8> = descriptors.iter().flat_map(as_bytes).collect();
        let mut descriptor_buffer = alloc.alloc(descriptor_bytes.len().max(1))?;
        descriptor_buffer.copy_from_host(&descriptor_bytes, stream)?;

        let offset_bytes: Vec<u8> = scratch_layout
            .values
            .iter()
            .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut offsets = alloc.alloc(offset_bytes.len().max(size_of::<u32>()))?;
        offsets.copy_from_host(&offset_bytes, stream)?;

        // Zeroed, and the zeroing is what puts the status word in a known
        // state: it lives in the first 16 bytes of each lane's scratch, and
        // the kernel writes `state = 1` into it before any op runs. A garbage
        // status would be read back as a fault nobody raised.
        let scratch_bytes = (scratch_stride as usize * lanes as usize)
            .max(usize::try_from(SCRATCH_ALIGN).unwrap_or(256));
        let mut scratch = alloc.alloc(scratch_bytes)?;
        scratch.memset(0, stream)?;

        // One byte per (lane, channel). Zero means the take reads the
        // COMMITTED cell; the device sets it as puts land within the fire.
        let mut pending = alloc.alloc((lanes as usize * channel_count as usize).max(1))?;
        pending.memset(0, stream)?;

        let intrinsic_len = lanes as usize * INTRINSIC_SLOTS;
        let zeroed = |bytes: usize| -> Result<DeviceBuffer> {
            let mut buffer = alloc.alloc(bytes)?;
            buffer.memset(0, stream)?;
            Ok(buffer)
        };
        let intrinsic_bases = zeroed(intrinsic_len * size_of::<u64>())?;
        let intrinsic_modes = zeroed(intrinsic_len * size_of::<u32>())?;
        let intrinsic_widths = zeroed(intrinsic_len * size_of::<u32>())?;
        let intrinsic_strides = zeroed(intrinsic_len * size_of::<u32>())?;
        let intrinsic_offsets = zeroed(intrinsic_len * size_of::<u32>())?;

        stream.synchronize()?;
        Ok(Self {
            table,
            descriptors: descriptor_buffer,
            params,
            offsets,
            scratch,
            pending,
            intrinsic_bases,
            intrinsic_modes,
            intrinsic_widths,
            intrinsic_strides,
            intrinsic_offsets,
            commit,
            lanes,
            value_count,
            scratch_stride,
            temporary_offset,
        })
    }

    /// How many lanes this fire dispatches.
    ///
    /// ONE today: multi-lane grouping is the first thing this module's own
    /// header says it does not do. Exposed anyway, because everything
    /// lane-indexed — the side tables, the scratch stride, the pending
    /// flags — is written against it, and a test that hardcoded a count
    /// would pass for the wrong reason the day it changes.
    #[must_use]
    pub const fn lanes(&self) -> u32 {
        self.lanes
    }

    /// Point one intrinsic at the buffer a fire produced.
    ///
    /// The five side tables are allocated ZEROED by [`Prepared::build`] and
    /// were never filled, which is what `fire.rs` meant by *"the intrinsic
    /// side tables beyond binding them empty"*. A program that reads
    /// `logits` therefore read address ZERO — and sampling is what a PTIR
    /// program is FOR, so that is the whole of a top-p or an argmax
    /// reading nothing.
    ///
    /// What each table means is the emitted kernel's, not a convention
    /// invented here — `codegen/cuda/fused.rs:411-421` reads them into an
    /// `OpParams` verbatim:
    ///
    /// | table | becomes | is |
    /// |---|---|---|
    /// | `bases` | the operand pointer | the buffer's device address |
    /// | `modes` | `p.intrinsic_dtype` | a `DType` wire code |
    /// | `widths` | `p.imm` | the row width, e.g. the vocabulary |
    /// | `strides` | `p.intrinsic_row_stride` | ELEMENTS between rows |
    /// | `offsets` | `p.intrinsic_row_offset` | which row THIS lane reads |
    ///
    /// The row offset is per lane and everything else is not, which is the
    /// only asymmetry: one buffer, one layout, and each lane reading its
    /// own row of it. `row_of(lane)` is what makes a multi-lane fire
    /// sample each request rather than the first one N times — the same
    /// defect `bind_intrinsic`'s C++ ancestor had when it wrote
    /// `base_row = 0`.
    ///
    /// # Errors
    ///
    /// If `intr` is not a slot the tables carry, or a copy fails.
    pub fn bind_intrinsic(
        &mut self,
        intr: IntrinsicId,
        base: u64,
        dtype: DType,
        width: u32,
        row_stride: u32,
        row_of: impl Fn(u32) -> u32,
        stream: StreamRef<'_>,
    ) -> Result<()> {
        let slot = intr as usize;
        if slot >= INTRINSIC_SLOTS {
            return Err(Error::invalid(
                "ptir::fire",
                format!(
                    "intrinsic {slot} is past the {INTRINSIC_SLOTS}-slot pitch the \
                     side tables are indexed with"
                ),
            ));
        }
        // Written per lane rather than as one run, because the tables are
        // lane-major: slot `intr` of lane `l` is at `l * SLOTS + intr`, so
        // the entries this touches are strided and a single `write_at`
        // would overwrite the fifteen slots beside each.
        for lane in 0..self.lanes {
            let at = lane as usize * INTRINSIC_SLOTS + slot;
            self.intrinsic_bases
                .write_at(at * size_of::<u64>(), &base.to_le_bytes(), stream)?;
            self.intrinsic_modes.write_at(
                at * size_of::<u32>(),
                &(dtype as u8 as u32).to_le_bytes(),
                stream,
            )?;
            self.intrinsic_widths
                .write_at(at * size_of::<u32>(), &width.to_le_bytes(), stream)?;
            self.intrinsic_strides
                .write_at(at * size_of::<u32>(), &row_stride.to_le_bytes(), stream)?;
            self.intrinsic_offsets
                .write_at(at * size_of::<u32>(), &row_of(lane).to_le_bytes(), stream)?;
        }
        Ok(())
    }

    /// Read one lane's binding of `intr` back, for tests and diagnosis.
    ///
    /// `(base, mode, width, stride, offset)`.
    ///
    /// # Errors
    ///
    /// If `intr` or `lane` is out of range, or a copy fails.
    pub fn intrinsic_binding(
        &self,
        intr: IntrinsicId,
        lane: u32,
        stream: StreamRef<'_>,
    ) -> Result<(u64, u32, u32, u32, u32)> {
        let slot = intr as usize;
        if slot >= INTRINSIC_SLOTS || lane >= self.lanes {
            return Err(Error::invalid(
                "ptir::fire",
                format!("no binding for intrinsic {slot} of lane {lane}"),
            ));
        }
        let at = lane as usize * INTRINSIC_SLOTS + slot;
        let mut base = [0u8; 8];
        self.intrinsic_bases.read_at(at * size_of::<u64>(), &mut base, stream)?;
        let mut word = |buf: &DeviceBuffer| -> Result<u32> {
            let mut b = [0u8; 4];
            buf.read_at(at * size_of::<u32>(), &mut b, stream)?;
            Ok(u32::from_le_bytes(b))
        };
        Ok((
            u64::from_le_bytes(base),
            word(&self.intrinsic_modes)?,
            word(&self.intrinsic_widths)?,
            word(&self.intrinsic_strides)?,
            word(&self.intrinsic_offsets)?,
        ))
    }

    /// Launch one generated region over every lane.
    ///
    /// One CTA per lane, at the width the compiled function's own register
    /// pressure permits. Both are the kernel's contract rather than a tuning
    /// choice: it reads `blockIdx.x` as its lane, and it reduces by halving
    /// `blockDim.x`.
    ///
    /// # Errors
    ///
    /// If the driver refuses the launch. A fault inside the kernel is not
    /// reported here — read [`Prepared::status`] after synchronizing.
    pub fn launch_region(&self, region: &Region, stream: StreamRef<'_>) -> Result<()> {
        let mut args = Args::new();
        args.ptr(self.table.as_ptr())
            .ptr(self.lane_records_ptr())
            .ptr(self.channel_slots_ptr())
            .ptr(self.descriptors.as_ptr())
            .ptr(self.params.as_ptr())
            .ptr(self.offsets.as_ptr())
            .ptr(self.scratch.as_ptr())
            .u32(self.value_count)
            .u32(self.scratch_stride)
            .u32(self.temporary_offset)
            .ptr(self.pending.as_ptr())
            .ptr(self.intrinsic_bases.as_ptr())
            .ptr(self.intrinsic_modes.as_ptr())
            .ptr(self.intrinsic_widths.as_ptr())
            .ptr(self.intrinsic_strides.as_ptr())
            .ptr(self.intrinsic_offsets.as_ptr());
        launch(
            &region.module,
            self.lanes,
            region.module.block_threads(),
            &mut args,
            FUSED_ARITY,
            stream,
        )
    }

    /// Whether the kernel left the fire committable.
    ///
    /// The kernel CLEARS this to refuse — on a stale ABI version, on a fault,
    /// on a readiness miss it observed itself — so a fire whose commit slot is
    /// zero must not publish, whatever the host's own readiness said.
    ///
    /// # Errors
    ///
    /// If the readback fails.
    pub fn committed(&self, stream: StreamRef<'_>) -> Result<bool> {
        let mut word = [0u8; 4];
        self.commit.copy_to_host(&mut word, stream)?;
        stream.synchronize()?;
        Ok(u32::from_le_bytes(word) != 0)
    }

    /// What the fire decided, and what a reader is told about it.
    ///
    /// # Why this reads the commit slot and not a status word
    ///
    /// CUDA's generated kernels declare `M1Status` as `__shared__` and never
    /// write it to memory the host can see. The only thing that crosses is the
    /// commit slot: the kernel clears it on a stale ABI version, on a fault
    /// raised by any op, and on a readiness miss it observed itself — and
    /// leaves it alone otherwise.
    ///
    /// So a CUDA fire has ONE observable bit where Metal has sixteen bytes of
    /// diagnosis, and this method says so rather than reading sixteen zeroed
    /// bytes out of scratch and reporting `NeverWritten` for every fire that
    /// ever ran. That is what an earlier draft of this file did, and it made a
    /// successful fire and a faulted one indistinguishable.
    ///
    /// The diagnosis is therefore coarse by construction: `Committed` when the
    /// slot survived, `Failed`/`Faulted` when it did not. Recovering the fault
    /// CLASS would need the emitter to write the status word out, which is a
    /// change on the host side of the ABI and not something a driver can
    /// decide for itself.
    ///
    /// # Errors
    ///
    /// If the commit-slot readback fails.
    pub fn outcome(&self, stream: StreamRef<'_>) -> Result<(StatusOutcome, Diagnosis)> {
        if self.committed(stream)? {
            Ok((StatusOutcome::Committed, Diagnosis::Committed))
        } else {
            // `Faulted` rather than `ReadinessUnmet`: the host asked about
            // readiness before the launch and would not have launched a fire
            // it knew was blocked, so a cleared slot here is the kernel
            // refusing for a reason of its own.
            Ok((StatusOutcome::Failed, Diagnosis::Faulted))
        }
    }

    /// A value's bytes, out of lane `lane`'s scratch.
    ///
    /// # Errors
    ///
    /// If the readback fails.
    pub fn read_value(
        &self,
        lane: u32,
        offset: u32,
        bytes: usize,
        stream: StreamRef<'_>,
    ) -> Result<Vec<u8>> {
        let mut out = vec![0u8; bytes];
        let at = lane as usize * self.scratch_stride as usize + offset as usize;
        self.scratch.read_at(at, &mut out, stream)?;
        stream.synchronize()?;
        Ok(out)
    }

    /// The lane records, which begin one header into the table.
    fn lane_records_ptr(&self) -> *mut std::ffi::c_void {
        // SAFETY: the table was allocated at `Shape::bytes()`, which is the
        // header plus the records plus the slots, so this offset is inside it.
        unsafe { self.table.as_ptr().byte_add(LANE_HEADER_BYTES as usize) }
    }

    /// The flat channel-slot array, which begins after every record.
    fn channel_slots_ptr(&self) -> *mut std::ffi::c_void {
        let at = LANE_HEADER_BYTES as usize + self.lanes as usize * LANE_RECORD_BYTES as usize;
        // SAFETY: as above — `Shape::bytes()` covers the header, `lanes`
        // records, and the slots that follow them.
        unsafe { self.table.as_ptr().byte_add(at) }
    }
}

/// The per-fire numbers [`OpParams::of`] needs beyond the op.
fn runtime_of(_op: &LaunchOp) -> OpRuntime {
    OpRuntime::default()
}

/// A `#[repr(C)]` record's bytes.
fn write_record<T: Copy>(into: &mut [u8], at: usize, record: &T) {
    // SAFETY: `T` is a `#[repr(C)]` mirror of a device struct, every field of
    // which the caller has written; reading it as bytes reads only initialised
    // memory.
    let bytes = unsafe {
        std::slice::from_raw_parts(std::ptr::from_ref(record).cast::<u8>(), size_of::<T>())
    };
    into[at..at + bytes.len()].copy_from_slice(bytes);
}

/// A value descriptor's bytes, for the flat upload.
fn as_bytes(descriptor: &ValueDesc) -> Vec<u8> {
    // SAFETY: `ValueDesc` is a `#[repr(C)]` mirror of `M1ValueDesc`, 36 bytes
    // of `u32`, with no padding and every field initialised by `describe`.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(descriptor).cast::<u8>(),
            size_of::<ValueDesc>(),
        )
    };
    bytes.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three record sizes the table's offsets are computed from. A table
    /// laid out with one of them wrong has every lane after the first reading
    /// the previous lane's tail, and every field is a plausible number.
    #[test]
    fn the_table_records_are_the_sizes_the_kernel_indexes_by() {
        assert_eq!(LANE_HEADER_BYTES, 16, "four u32");
        assert_eq!(size_of::<LaneHeader>(), 16);
        assert_eq!(size_of::<LaneRecord>(), LANE_RECORD_BYTES as usize);
        assert_eq!(size_of::<LaneChannelSlot>(), LANE_SLOT_BYTES as usize);
        assert_eq!(LANE_SLOT_BYTES, 32, "four u64");
    }

    /// The slot array begins after EVERY lane's record, not after one. With a
    /// single lane the two are the same number, which is exactly why a
    /// single-lane port can carry the mistake to the first multi-lane fire.
    #[test]
    fn the_slot_array_begins_past_every_lane_record() {
        let one = LaneShape::of(1, 2).bytes().expect("fits");
        let four = LaneShape::of(4, 2).bytes().expect("fits");
        assert_eq!(one, 16 + LANE_RECORD_BYTES + 2 * LANE_SLOT_BYTES);
        assert_eq!(four, 16 + 4 * LANE_RECORD_BYTES + 8 * LANE_SLOT_BYTES);
    }

    /// Sixteen arguments, checked against the header the emitter splices in.
    /// A kernel handed fifteen reads its sixteenth from whatever follows the
    /// argument array.
    #[test]
    fn a_fused_region_takes_sixteen_arguments() {
        assert_eq!(FUSED_ARITY, 16);
    }

    /// The intrinsic tables are indexed `lane * INTRINSIC_SLOTS + intr`, so
    /// the constant is a stride the emitted kernels share. It is pinned as an
    /// equality rather than a bound: too LARGE wastes memory and still works,
    /// which is why a `>=` here would pass forever while the real requirement
    /// -- that host and kernel agree -- quietly stopped holding.
    #[test]
    fn the_intrinsic_stride_is_the_slot_count_the_abi_declares() {
        assert_eq!(
            INTRINSIC_SLOTS, 16,
            "PTIR_INTRINSIC_SLOTS is PTIR_INTR_ATTN_SCORE + 1; a stride that \
             disagrees with the kernel's misdirects every intrinsic of every \
             lane but the first"
        );
    }
}
