//! The fire: one PTIR stage, prepared and launched.
//!
//! [`Prepared::build`] fills the per-fire device buffers a region reads, each
//! an array whose stride the kernel trusts. No length lives on the device, so a
//! too-small stride reads the previous lane's tail silently. One `offsets`
//! table serves every lane, laid out against the widest descriptor per value.

use driver::driver_api::plan::{LaunchOp, LaunchStagePlan};
use driver::tensor_ir::op::{IntrinsicId, tags};
use driver::{
    Diagnosis, Extents, LANE_HEADER_BYTES, LANE_RECORD_BYTES, LANE_SLOT_BYTES, LaneChannelSlot,
    LaneHeader, LaneRecord, LaneShape, NO_TICKET, OpParams, OpRuntime, SCRATCH_ALIGN,
    StatusOutcome, ValueDesc, describe, layout,
};

use crate::device::{Allocator, DeviceBuffer, StreamRef};
use crate::error::{Error, Result};

use super::channel::Rings;
use super::params::{CudaOpParams, params_bytes};
use super::runtime::Region;

/// How many intrinsic slots the side tables carry per lane.
///
/// The five side arrays are indexed `lane * INTRINSIC_SLOTS + intr`, so this is
/// a stride: getting it wrong misdirects every intrinsic of every lane but the first.
///
/// PROJECTED FROM THE ABI, not written. It was a literal `16` while the
/// emitted kernel strides by `IntrinsicId::SLOTS` -- `AttnScore + 1`, which is
/// EIGHT -- so host and kernel disagreed by a factor of two. Lane zero
/// coincides at every intrinsic (`0 * 16 + i == 0 * 8 + i`), which is why a
/// single-lane fire never showed it and the test below asserted the wrong
/// number against its own stated rule.
pub const INTRINSIC_SLOTS: usize = driver::tensor_ir::op::IntrinsicId::SLOTS as usize;

/// The sixteen arguments a generated fused region takes.
const FUSED_ARITY: usize = 16;

/// One member of a grouped fire: its rings and its extents.
///
/// A struct, not two parallel slices: zipping them wrongly would have lane 0
/// read lane 1's channels — a wrong answer rather than a fault.
#[derive(Clone, Copy)]
pub struct Lane<'a> {
    /// The channel-ring registry — one for every lane: a channel has one ring
    /// wherever it is named from.
    pub rings: &'a Rings,
    /// This member's dense channel index → registry slot: what makes the lanes
    /// differ, now that the registry does not.
    pub slots: &'a [u32],
    /// How much this member submitted.
    pub extents: Extents,
}

impl Lane<'_> {
    /// The registry slot this member's dense channel `dense` lives at.
    fn slot(&self, dense: u32) -> Result<usize> {
        self.slots
            .get(dense as usize)
            .map(|&g| g as usize)
            .ok_or_else(|| {
                Error::invalid(
                    "program::run",
                    format!("channel {dense} is not one this instance carries"),
                )
            })
    }
}

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
    lanes: u32,
    value_count: u32,
    scratch_stride: u32,
    temporary_offset: u32,
    /// Where each value sits in a lane's scratch, kept host-side so a trace can
    /// read a value back without re-deriving the layout.
    value_offsets: Vec<u32>,
    /// The widest descriptor per value, which is what the offsets were laid
    /// out against.
    value_descriptors: Vec<ValueDesc>,
}

impl Prepared {
    /// Build every buffer one fire needs, one [`Lane`] per member.
    ///
    /// # Errors
    ///
    /// If a value's shape cannot be resolved against a lane's extents, if the
    /// scratch exceeds what the layout permits, or if an allocation fails.
    pub fn build(
        alloc: &Allocator,
        plan: &LaunchStagePlan,
        lanes_in: &[Lane<'_>],
        stream: StreamRef<'_>,
    ) -> Result<Self> {
        if lanes_in.is_empty() {
            return Err(Error::invalid("program::run", "a fire with no lanes"));
        }
        let lane_extents: Vec<Extents> = lanes_in.iter().map(|l| l.extents).collect();
        let lanes = u32::try_from(lanes_in.len())
            .map_err(|_| Error::invalid("program::run", "more lanes than a u32 can count"))?;
        let channel_count = u32::try_from(plan.channel_bindings.len())
            .map_err(|_| Error::invalid("program::run", "more channels than a u32 can count"))?;
        let value_count = u32::try_from(plan.value_types.len())
            .map_err(|_| Error::invalid("program::run", "more values than a u32 can count"))?;

        // ── Descriptors, per lane, and the scratch they size. ──
        //
        // The kernel reads `all_descriptors + lane * value_count` per lane.
        let mut per_lane: Vec<Vec<ValueDesc>> = Vec::with_capacity(lane_extents.len());
        for extents in &lane_extents {
            per_lane.push(
                plan.value_types
                    .iter()
                    .map(|value| {
                        describe(value, extents).map_err(|why| {
                            Error::invalid(
                                "program::run",
                                format!(
                                    "a value's shape does not resolve against this fire: {why:?}"
                                ),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        // Lay out against the widest descriptor per value, so a value's offset
        // matches across lanes.
        let mut descriptors: Vec<ValueDesc> = per_lane[0].clone();
        for lane in &per_lane[1..] {
            for (widest, one) in descriptors.iter_mut().zip(lane) {
                if one.device_bytes() > widest.device_bytes() {
                    *widest = *one;
                }
            }
        }
        // WHAT EACH VALUE'S SHAPE RESOLVED TO. A descriptor whose `last` is
        // zero makes the fused argmax scan no columns and leave its
        // zero-initialised result, which reads back as token 0 forever.
        if std::env::var_os("PIE_TRACE_VALUES").is_some() {
            for (i, d) in descriptors.iter().enumerate() {
                eprintln!("[desc] {i} {d:?}");
            }
        }
        let scratch_layout = layout(&descriptors).map_err(|why| {
            Error::invalid(
                "program::run",
                format!("this fire's scratch does not fit: {why:?}"),
            )
        })?;
        let scratch_stride = u32::try_from(scratch_layout.total)
            .map_err(|_| Error::invalid("program::run", "a scratch stride past u32"))?;
        let temporary_offset = u32::try_from(scratch_layout.temporary)
            .map_err(|_| Error::invalid("program::run", "a temporary offset past u32"))?;

        // ── The commit slot, and the lane table that points at it. ──
        let mut commit = alloc.alloc(size_of::<u32>())?;
        // One, not zero: the kernel reads this first and returns if it is clear,
        // so a zero start would run every kernel and do nothing.
        commit.copy_from_host(&1u32.to_le_bytes(), stream)?;

        let shape = LaneShape::of(lanes, channel_count);
        let table_bytes = shape
            .bytes()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| Error::invalid("program::run", "a lane table past what fits"))?;
        let mut table = alloc.alloc(table_bytes)?;
        let mut host_table = vec![0u8; table_bytes];

        let header = LaneHeader {
            abi_version: driver::LANE_ABI_VERSION,
            lane_count: lanes,
            channel_slots_per_lane: channel_count,
            flags: 0,
        };
        write_record(&mut host_table, 0, &header);

        // ── One record per lane, then that lane's channel slots. ──
        //
        // `channel_slot_offset` is the lane's row in the flat slot array: the
        // kernel indexes `channels[lane * slots + n]`, so it is `lane * channel_count`.
        let slots_at = LANE_HEADER_BYTES as usize + lanes as usize * LANE_RECORD_BYTES as usize;
        for (lane, member) in lanes_in.iter().enumerate() {
            let extents = &member.extents;
            // Each lane gets its own rings: a shared set would have every member read
            // the first's cells — a wrong answer, not a fault.
            let rings = member.rings;
            let cursors = rings.cursors(stream)?;
            let record = LaneRecord {
                kv_len: extents.kv_len,
                page_count: extents.page_count,
                row_count: extents.row_count,
                token_count: extents.token_count,
                sampled_rows: extents.sampled_rows,
                query_len: extents.query_len,
                key_len: extents.key_len,
                channel_slot_offset: u32::try_from(lane)
                    .ok()
                    .and_then(|l| l.checked_mul(channel_count))
                    .ok_or_else(|| {
                        Error::invalid("program::run", "a channel slot offset past u32")
                    })?,
                commit_slot: commit.as_ptr() as u64,
                ..LaneRecord::default()
            };
            write_record(
                &mut host_table,
                LANE_HEADER_BYTES as usize + lane * LANE_RECORD_BYTES as usize,
                &record,
            );

            // Absolute cell addresses, resolved on the host from this lane's rings.
            for (local, &dense) in plan.channel_bindings.iter().enumerate() {
                let channel = member.slot(dense)?;
                let cursor = cursors.get(channel).ok_or_else(|| {
                    Error::invalid(
                        "program::run",
                        format!(
                            "stage-local channel {local} binds channel {dense}, which is unbound"
                        ),
                    )
                })?;
                let slot = LaneChannelSlot {
                    committed_cell: rings.cell_address(channel, cursor.head)?,
                    pending_cell: rings.cell_address(channel, cursor.tail)?,
                    // Not a ticket: nothing stages a table ahead of the fire, so
                    // claiming one would pass a staleness check for the wrong reason.
                    expected_head: NO_TICKET,
                    expected_tail: NO_TICKET,
                };
                write_record(
                    &mut host_table,
                    slots_at
                        + (lane * plan.channel_bindings.len() + local) * LANE_SLOT_BYTES as usize,
                    &slot,
                );
            }
        }
        table.copy_from_host(&host_table, stream)?;

        // ── Op params, widened to CUDA's 88-byte record. ──
        let mut records = Vec::with_capacity(plan.ops.len());
        let mut result_base = 0u32;
        for op in &plan.ops {
            let mut record = CudaOpParams::widen(OpParams::of(op, result_base, runtime_of(op)));
            // `sink_bytes` is the fixed cell size a `chan_put` writes; the kernel
            // faults every put where `logical_bytes > sink_bytes`, so zero refuses.
            if u32::from(op.code) == u32::from(tags::CHAN_PUT) && op.channel != u32::MAX {
                let dense = plan
                    .channel_bindings
                    .get(op.channel as usize)
                    .copied()
                    .ok_or_else(|| {
                        Error::invalid(
                            "program::run",
                            format!(
                                "a put names stage-local channel {}, which is unbound",
                                op.channel
                            ),
                        )
                    })?;
                // The first lane's shapes: every member binds the same plan, so
                // the channel geometry is the plan's — only the cursors differ.
                let slot = lanes_in[0].slot(dense)?;
                let shape = lanes_in[0].rings.shape(slot).ok_or_else(|| {
                    Error::invalid(
                        "program::run",
                        format!(
                            "a put targets channel {dense}, which this instance does not carry"
                        ),
                    )
                })?;
                record.sink_bytes = u32::try_from(shape.cell_bytes())
                    .map_err(|_| Error::invalid("program::run", "a cell past what a u32 counts"))?;
            }
            records.push(record);
            result_base += u32::from(op.result_count);
        }
        let param_bytes = params_bytes(&records);
        let mut params = alloc.alloc(param_bytes.len().max(size_of::<CudaOpParams>()))?;
        params.copy_from_host(&param_bytes, stream)?;

        // ── The flat arrays. ──
        //
        // Descriptors are lane-major: `all_descriptors + lane * value_count`.
        let descriptor_bytes: Vec<u8> = per_lane.iter().flatten().flat_map(as_bytes).collect();
        let mut descriptor_buffer = alloc.alloc(descriptor_bytes.len().max(1))?;
        descriptor_buffer.copy_from_host(&descriptor_bytes, stream)?;

        let offset_bytes: Vec<u8> = scratch_layout
            .values
            .iter()
            .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
            .flat_map(u32::to_le_bytes)
            .collect();
        let value_offsets: Vec<u32> = scratch_layout
            .values
            .iter()
            .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
            .collect();
        let mut offsets = alloc.alloc(offset_bytes.len().max(size_of::<u32>()))?;
        offsets.copy_from_host(&offset_bytes, stream)?;

        // Zeroed: the status word lives in the first 16 bytes of each lane's
        // scratch, and garbage there reads back as a fault nobody raised.
        let scratch_bytes = (scratch_stride as usize * lanes as usize)
            .max(usize::try_from(SCRATCH_ALIGN).unwrap_or(256));
        let mut scratch = alloc.alloc(scratch_bytes)?;
        scratch.memset(0, stream)?;

        // One byte per (lane, channel): zero means the take reads the committed
        // cell; the device sets it as puts land within the fire.
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
            value_offsets,
            value_descriptors: descriptors,
        })
    }

    /// How many lanes this fire dispatches.
    #[must_use]
    pub const fn lanes(&self) -> u32 {
        self.lanes
    }

    /// Point one intrinsic at the buffer a fire produced.
    ///
    /// Side tables are zeroed by [`Prepared::build`], so an unbound intrinsic
    /// reads address zero. Each table, and the emitted-kernel field it becomes:
    ///
    /// | table | becomes | is |
    /// |---|---|---|
    /// | `bases` | the operand pointer | the buffer's device address |
    /// | `modes` | `p.intrinsic_dtype` | an [`INTRINSIC_STORAGE_*`](super::params) code |
    /// | `widths` | `p.imm` | the row width, e.g. the vocabulary |
    /// | `strides` | `p.intrinsic_row_stride` | ELEMENTS between rows |
    /// | `offsets` | `p.intrinsic_row_offset` | which row THIS lane reads |
    ///
    /// `modes` is a storage mode, not a `DType` wire code: they collide only at
    /// `DType::F32 as u8 == 0 == INTRINSIC_STORAGE_F32`, so `DType::F32` for a
    /// BF16 buffer makes the sampler misread every logit, silently.
    pub fn bind_intrinsic(
        &mut self,
        intr: IntrinsicId,
        base: u64,
        storage: u32,
        width: u32,
        row_stride: u32,
        row_of: impl Fn(u32) -> u32,
        stream: StreamRef<'_>,
    ) -> Result<()> {
        let slot = intr as usize;
        if slot >= INTRINSIC_SLOTS {
            return Err(Error::invalid(
                "program::run",
                format!(
                    "intrinsic {slot} is past the {INTRINSIC_SLOTS}-slot pitch the \
                     side tables are indexed with"
                ),
            ));
        }
        // Per lane, not one run: the tables are lane-major (`l * SLOTS + intr`),
        // so the entries this touches are strided.
        for lane in 0..self.lanes {
            let at = lane as usize * INTRINSIC_SLOTS + slot;
            self.intrinsic_bases
                .write_at(at * size_of::<u64>(), &base.to_le_bytes(), stream)?;
            self.intrinsic_modes
                .write_at(at * size_of::<u32>(), &storage.to_le_bytes(), stream)?;
            self.intrinsic_widths
                .write_at(at * size_of::<u32>(), &width.to_le_bytes(), stream)?;
            self.intrinsic_strides.write_at(
                at * size_of::<u32>(),
                &row_stride.to_le_bytes(),
                stream,
            )?;
            let offset = row_of(lane);
            // WHICH ROW the program will scan. Past the end of a gathered
            // buffer reads whatever follows it, and an argmax over that is a
            // token nobody chose.
            if std::env::var_os("PIE_TRACE_VALUES").is_some() {
                eprintln!(
                    "[intr] slot={slot} lane={lane} base={base:#x} \
                     mode={storage} width={width} stride={row_stride} row={offset}"
                );
            }
            self.intrinsic_offsets.write_at(
                at * size_of::<u32>(),
                &offset.to_le_bytes(),
                stream,
            )?;
        }
        Ok(())
    }

    /// Read one lane's binding of `intr` back as `(base, mode, width, stride,
    /// offset)`, for tests and diagnosis.
    pub fn intrinsic_binding(
        &self,
        intr: IntrinsicId,
        lane: u32,
        stream: StreamRef<'_>,
    ) -> Result<(u64, u32, u32, u32, u32)> {
        let slot = intr as usize;
        if slot >= INTRINSIC_SLOTS || lane >= self.lanes {
            return Err(Error::invalid(
                "program::run",
                format!("no binding for intrinsic {slot} of lane {lane}"),
            ));
        }
        let at = lane as usize * INTRINSIC_SLOTS + slot;
        let mut base = [0u8; 8];
        self.intrinsic_bases
            .read_at(at * size_of::<u64>(), &mut base, stream)?;
        let word = |buf: &DeviceBuffer| -> Result<u32> {
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
    /// One CTA per lane at the compiled function's own block width — the
    /// kernel's contract, not a tuning choice: it reads `blockIdx.x` as its lane.
    ///
    /// # Errors
    ///
    /// If the driver refuses the launch. A fault inside the kernel surfaces at
    /// the next synchronize, not here.
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
    /// The kernel clears this to refuse (stale ABI, a fault, a readiness miss it
    /// observed), so a zero commit slot must not publish whatever the host thought.
    pub fn committed(&self, stream: StreamRef<'_>) -> Result<bool> {
        let mut word = [0u8; 4];
        self.commit.copy_to_host(&mut word, stream)?;
        stream.synchronize()?;
        Ok(u32::from_le_bytes(word) != 0)
    }

    /// What the fire decided.
    ///
    /// Only the commit slot crosses to the host — CUDA keeps `M1Status`
    /// `__shared__` and never writes it out — so the diagnosis is coarse:
    /// `Committed` if the slot survived, else `Failed`/`Faulted`.
    pub fn outcome(&self, stream: StreamRef<'_>) -> Result<(StatusOutcome, Diagnosis)> {
        if self.committed(stream)? {
            Ok((StatusOutcome::Committed, Diagnosis::Committed))
        } else {
            // `Faulted`, not `ReadinessUnmet`: the host already checked readiness
            // before launching, so a cleared slot is the kernel's own refusal.
            Ok((StatusOutcome::Failed, Diagnosis::Faulted))
        }
    }

    /// A value's bytes, out of lane `lane`'s scratch.
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

    /// Every value slot's head, after a fire, for one lane.
    ///
    /// WHAT EACH OP LEFT BEHIND. A slot no emitted op writes stays at the
    /// `memset` above, and zeros are a legal float: a softmax over them is
    /// uniform and the draw off it lands on token 0 without faulting. Reading
    /// the slots back is the only way to tell that from a chain that ran.
    pub fn trace_scratch(&self, lane: u32, stream: StreamRef<'_>) -> Result<()> {
        for (value, (&offset, desc)) in self
            .value_offsets
            .iter()
            .zip(self.value_descriptors.iter())
            .enumerate()
        {
            let width = (desc.len as usize).min(8);
            let unit = (desc.device_bytes() as usize / (desc.len as usize).max(1)).max(1);
            let bytes = self.read_value(lane, offset, width * unit, stream)?;
            let head: Vec<String> = bytes
                .chunks_exact(unit)
                .map(|raw| match (unit, desc.dtype) {
                    (4, 0) => format!(
                        "{:.4}",
                        f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
                    ),
                    (4, _) => format!("{}", i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
                    (1, _) => format!("{}", raw[0]),
                    _ => format!("{raw:?}"),
                })
                .collect();
            eprintln!(
                "[val] {value} off={offset} len={} dtype={} {}",
                desc.len,
                desc.dtype,
                head.join(",")
            );
        }
        Ok(())
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
    // SAFETY: `T` is a `#[repr(C)]` mirror of a device struct, every field
    // written by the caller; reading it as bytes reads only initialised memory.
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

    /// One wrong record size has every lane after the first read the last's tail.
    #[test]
    fn the_table_records_are_the_sizes_the_kernel_indexes_by() {
        assert_eq!(LANE_HEADER_BYTES, 16, "four u32");
        assert_eq!(size_of::<LaneHeader>(), 16);
        assert_eq!(size_of::<LaneRecord>(), LANE_RECORD_BYTES as usize);
        assert_eq!(size_of::<LaneChannelSlot>(), LANE_SLOT_BYTES as usize);
        assert_eq!(LANE_SLOT_BYTES, 32, "four u64");
    }

    /// The slot array begins after every lane's record; a single lane hides the bug.
    #[test]
    fn the_slot_array_begins_past_every_lane_record() {
        let one = LaneShape::of(1, 2).bytes().expect("fits");
        let four = LaneShape::of(4, 2).bytes().expect("fits");
        assert_eq!(one, 16 + LANE_RECORD_BYTES + 2 * LANE_SLOT_BYTES);
        assert_eq!(four, 16 + 4 * LANE_RECORD_BYTES + 8 * LANE_SLOT_BYTES);
    }

    /// A kernel handed fifteen of sixteen reads the last past the argument array.
    #[test]
    fn a_fused_region_takes_sixteen_arguments() {
        assert_eq!(FUSED_ARITY, 16);
    }

    /// The intrinsic stride the emitted kernels share, pinned as an equality:
    /// a `>=` would pass forever while host and kernel silently disagreed.
    #[test]
    fn the_intrinsic_stride_is_the_slot_count_the_abi_declares() {
        assert_eq!(
            INTRINSIC_SLOTS,
            tensor_compiler::codegen::cuda::fused::PTIR_INTRINSIC_SLOTS as usize,
            "PTIR_INTRINSIC_SLOTS is PTIR_INTR_ATTN_SCORE + 1; a stride that \
             disagrees with the kernel's misdirects every intrinsic of every \
             lane but the first"
        );
    }
}

// ── Launching the prepared fire above ──

// `cuLaunchKernel` takes `void**` — pointers to each argument's storage, not the
// values — so a scalar must outlive the call and a device pointer is passed by
// the address of its variable. It validates nothing, so marshalling lives behind
// [`Args`]. One CTA per lane; too small a grid drops the tail.

use cudarc::driver::sys as dr;

use super::compile::Module;

/// A kernel's argument list, kept alive for the launch.
///
/// Storage and pointer array are one value: `cuLaunchKernel` dereferences the
/// pointers during the call, so the scalars must outlive it.
#[derive(Default)]
pub struct Args {
    /// Boxed so a later append cannot move an earlier scalar and dangle its
    /// pointer in `slots` (a `Vec<u64>` would reallocate); hence `clippy::vec_box`.
    #[allow(clippy::vec_box)]
    storage: Vec<Box<u64>>,
    slots: Vec<*mut std::ffi::c_void>,
}

impl Args {
    /// An empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a device pointer argument.
    pub fn ptr(&mut self, pointer: *mut std::ffi::c_void) -> &mut Self {
        self.scalar(pointer as u64)
    }

    /// Append a `u32` argument, stored in a `u64` cell and pointed at its first
    /// four bytes — correct on the little-endian hosts CUDA runs on.
    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.scalar(u64::from(value))
    }

    /// Append a raw 64-bit argument.
    fn scalar(&mut self, value: u64) -> &mut Self {
        let mut cell = Box::new(value);
        let at: *mut u64 = &raw mut *cell;
        self.storage.push(cell);
        self.slots.push(at.cast());
        self
    }

    /// How many arguments have been appended.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether nothing has been appended.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The `void**` the driver takes.
    fn as_raw(&mut self) -> *mut *mut std::ffi::c_void {
        self.slots.as_mut_ptr()
    }
}

/// Launch `module`'s entry with `grid` blocks of `block` threads.
///
/// # Errors
///
/// If the driver refuses the launch. A fault inside the kernel is asynchronous
/// and surfaces at the next synchronize, not here.
pub fn launch(
    module: &Module,
    grid: u32,
    block: u32,
    args: &mut Args,
    expected: usize,
    stream: StreamRef<'_>,
) -> Result<()> {
    // The arity check CUDA does not do: too few arguments reads the rest from
    // whatever follows the array, a wrong answer rather than an error.
    if args.len() != expected {
        return Err(Error::invalid(
            "cuLaunchKernel",
            format!(
                "'{}' takes {expected} arguments and {} were bound",
                module.entry_name(),
                args.len()
            ),
        ));
    }
    if grid == 0 {
        // A zero grid launches nothing and returns success, so a fire with no
        // lanes would look like a fire that ran.
        return Err(Error::invalid(
            "cuLaunchKernel",
            format!("'{}' launched with an empty grid", module.entry_name()),
        ));
    }
    // SAFETY: `module.function()` is live for the borrow, and `args` holds every
    // scalar the pointer array points at for this call.
    let code = unsafe {
        dr::cuLaunchKernel(
            module.function(),
            grid,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.as_raw().cast(),
            args.as_raw(),
            std::ptr::null_mut(),
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(Error::Driver {
            call: "cuLaunchKernel",
            code,
        });
    }
    Ok(())
}

/// The two control kernels, launched.
///
/// A module of its own because the pair share a shape no other launch has: the
/// four ring arrays, two uploaded channel-index lists, and single-thread.
pub mod launch_control {
    use super::{Allocator, Args, Control, Error, Result, Rings, StreamRef, launch};

    /// Ask whether a pass may commit: `need_full` are the channels it consumes
    /// (committed cell must hold a value), `need_empty` the ones it produces
    /// (need room). The verdict is read back because the host reports a blocked
    /// fire to the runtime as a retry, which the GPU cannot decide.
    pub fn readiness(
        control: &Control,
        rings: &Rings,
        need_full: &[u32],
        need_empty: &[u32],
        alloc: &Allocator,
        stream: StreamRef<'_>,
    ) -> Result<bool> {
        // The flag starts at 1 and each stage ANDs into it, narrowing one
        // shared verdict without stages knowing about each other.
        let mut pass = alloc.alloc(size_of::<u32>())?;
        pass.copy_from_host(&1u32.to_le_bytes(), stream)?;
        let full_list = upload_indices(alloc, need_full, stream)?;
        let empty_list = upload_indices(alloc, need_empty, stream)?;

        let mut args = Args::new();
        args.ptr(rings.full_ptr())
            .ptr(rings.head_ptr())
            .ptr(rings.tail_ptr())
            .ptr(rings.cap1_ptr())
            .ptr(
                full_list
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |b| b.as_ptr()),
            )
            .u32(u32::try_from(need_full.len()).map_err(too_many)?)
            .ptr(
                empty_list
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |b| b.as_ptr()),
            )
            .u32(u32::try_from(need_empty.len()).map_err(too_many)?)
            .ptr(pass.as_ptr());
        launch(control.readiness(), 1, 1, &mut args, 9, stream)?;

        let mut verdict = [0u8; 4];
        pass.copy_to_host(&mut verdict, stream)?;
        stream.synchronize()?;
        Ok(u32::from_le_bytes(verdict) != 0)
    }

    /// Advance the cursors of a pass that ran.
    ///
    /// When `committed` is false every kernel still launches — the dummy run,
    /// so a blocked fire costs the same as a running one — but nothing moves.
    pub fn commit(
        control: &Control,
        rings: &Rings,
        taken: &[u32],
        put: &[u32],
        committed: bool,
        alloc: &Allocator,
        stream: StreamRef<'_>,
    ) -> Result<()> {
        let mut pass = alloc.alloc(size_of::<u32>())?;
        pass.copy_from_host(&u32::from(committed).to_le_bytes(), stream)?;
        let taken_list = upload_indices(alloc, taken, stream)?;
        let put_list = upload_indices(alloc, put, stream)?;

        let mut args = Args::new();
        args.ptr(rings.full_ptr())
            .ptr(rings.head_ptr())
            .ptr(rings.tail_ptr())
            .ptr(rings.cap1_ptr())
            .ptr(
                taken_list
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |b| b.as_ptr()),
            )
            .u32(u32::try_from(taken.len()).map_err(too_many)?)
            .ptr(
                put_list
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |b| b.as_ptr()),
            )
            .u32(u32::try_from(put.len()).map_err(too_many)?)
            .ptr(pass.as_ptr());
        launch(control.commit(), 1, 1, &mut args, 9, stream)?;
        // The flag buffer must outlive the async launch; synchronizing here is
        // the cheap answer while one fire is in flight.
        stream.synchronize()?;
        Ok(())
    }

    /// Upload a channel-index list, or `None` when it is empty.
    ///
    /// `None`, not a zero-byte allocation: the kernel reads the array only
    /// `count` times, so a null with count zero is exactly correct.
    fn upload_indices(
        alloc: &Allocator,
        indices: &[u32],
        stream: StreamRef<'_>,
    ) -> Result<Option<crate::device::DeviceBuffer>> {
        if indices.is_empty() {
            return Ok(None);
        }
        let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let mut buffer = alloc.alloc(bytes.len())?;
        buffer.copy_from_host(&bytes, stream)?;
        Ok(Some(buffer))
    }

    fn too_many(_: std::num::TryFromIntError) -> Error {
        Error::invalid("program::run", "more channels than a u32 can count")
    }
}

#[cfg(test)]
mod tests_2 {
    use super::*;

    /// A `Vec<u64>` backing would reallocate on append and dangle bound pointers.
    #[test]
    fn appending_an_argument_does_not_move_the_ones_already_bound() {
        let mut args = Args::new();
        for value in 0..64u32 {
            args.u32(value);
        }
        assert_eq!(args.len(), 64);
        for (index, slot) in args.slots.iter().enumerate() {
            // SAFETY: each slot points at a `Box<u64>` this `Args` still owns.
            let seen = unsafe { *slot.cast::<u64>() };
            assert_eq!(
                seen, index as u64,
                "argument {index} moved when later ones were appended"
            );
        }
    }

    /// A pointer argument is bound by the slot's address, not by value.
    #[test]
    fn a_pointer_argument_is_bound_by_address_and_not_by_value() {
        let target = 0xdead_beefu64;
        let mut args = Args::new();
        args.ptr(target as *mut std::ffi::c_void);
        // SAFETY: the slot points at the `Box<u64>` holding the pointer value.
        let stored = unsafe { *args.slots[0].cast::<u64>() };
        assert_eq!(
            stored, target,
            "the slot must hold the pointer, and the slot's own address is \
             what cuLaunchKernel receives"
        );
    }
}

// ── The control kernels that gate and commit it ──

// The two control kernels — readiness before a pass, commit bump after — are
// compiled here through NVRTC (no prebuilt pair), transcribed from C++
// `channels.hpp`. Commit order is puts before takes. Types are bare C: NVRTC
// has no include path, so `<cstdint>` and `std::` are unavailable.

use std::sync::Arc;

use driver::Failure;

use super::cache::{Disk, disk_key};
use super::compile::FailureKind;

/// The ring's fixed row pitch in the `full` array.
///
/// `full` is indexed `full[channel * MAX_RING + slot]` with this stride
/// whatever a channel's capacity is; `cap1` as pitch would index the neighbour's flags.
pub const MAX_RING: u32 = 64;

/// The readiness kernel's entry point.
pub const READINESS_ENTRY: &str = "pie_ptir_stage_readiness";

/// The commit-bump kernel's entry point.
pub const COMMIT_ENTRY: &str = "pie_ptir_commit_bump";

/// Both kernels in one translation unit, compiled and cached together.
///
/// `MAX_RING` is interpolated, not a literal, so the Rust constant is the single
/// definition — a kernel built with a different pitch would misread every channel but 0.
fn source() -> String {
    format!(
        r#"
typedef unsigned char pie_u8;
typedef unsigned int  pie_u32;

#define PIE_MAX_RING {MAX_RING}u

// Stage readiness: AND this stage's requirement into the pass commit flag.
//
// `need_full` names the channels whose first op consumes -- their committed
// cell must be full. `need_empty` names the channels whose first op produces
// -- the ring must have room, which is the standard ring-not-full test
// `(tail + 1) % cap1 != head`, reserving one sentinel cell so a capacity-N
// channel holds at most N unconsumed items.
//
// A miss clears `pass_commit`, which does NOT abort the pass: the region
// kernels still run, over the same cells, and the commit bump then declines to
// publish. That is the dummy run, and it is what makes a blocked fire cost the
// same every time instead of branching on the device.
extern "C" __global__ void {READINESS_ENTRY}(
    const pie_u8*  full,
    const pie_u32* head,
    const pie_u32* tail,
    const pie_u32* cap1,
    const pie_u32* need_full_ch,  pie_u32 n_full,
    const pie_u32* need_empty_ch, pie_u32 n_empty,
    pie_u32* pass_commit) {{
  if (threadIdx.x != 0u || blockIdx.x != 0u) return;
  pie_u32 ok = 1u;
  for (pie_u32 i = 0u; i < n_full; ++i) {{
    const pie_u32 c = need_full_ch[i];
    if (!full[c * PIE_MAX_RING + head[c]]) ok = 0u;
  }}
  for (pie_u32 i = 0u; i < n_empty; ++i) {{
    const pie_u32 c = need_empty_ch[i];
    if (((tail[c] + 1u) % cap1[c]) == head[c]) ok = 0u;
  }}
  *pass_commit &= ok;
}}

// End-of-pass predicated commit bump.
//
// Iff `*pass_commit`: publish every put channel's pending cell by setting its
// full bit and advancing `tail`, then consume every taken channel's committed
// cell by clearing its full bit and advancing `head`.
//
// Puts run BEFORE takes, and the order is the contract rather than a
// preference: a channel that is both taken and put in one pass -- a
// loop-carried counter, the shape every decode loop has -- advances both, and
// publishing first is what leaves the ring holding what the pass produced.
extern "C" __global__ void {COMMIT_ENTRY}(
    pie_u8*  full,
    pie_u32* head,
    pie_u32* tail,
    const pie_u32* cap1,
    const pie_u32* taken_ch, pie_u32 n_taken,
    const pie_u32* put_ch,   pie_u32 n_put,
    const pie_u32* pass_commit) {{
  if (threadIdx.x != 0u || blockIdx.x != 0u) return;
  if (!*pass_commit) return;
  for (pie_u32 i = 0u; i < n_put; ++i) {{
    const pie_u32 c = put_ch[i];
    full[c * PIE_MAX_RING + tail[c]] = 1u;
    tail[c] = (tail[c] + 1u) % cap1[c];
  }}
  for (pie_u32 i = 0u; i < n_taken; ++i) {{
    const pie_u32 c = taken_ch[i];
    full[c * PIE_MAX_RING + head[c]] = 0u;
    head[c] = (head[c] + 1u) % cap1[c];
  }}
}}
"#
    )
}

/// The two control kernels, compiled and loaded.
#[derive(Debug, Clone)]
pub struct Control {
    readiness: Arc<Module>,
    commit: Arc<Module>,
}

impl Control {
    /// Compile both kernels for `architecture`, or take them off `disk`.
    ///
    /// Cached under a key of their own, not a program's: they belong to no
    /// program and are identical for every program on one device.
    ///
    /// # Errors
    ///
    /// [`Failure::Deterministic`] if the source does not compile (this file and
    /// NVRTC disagree about the language); [`Failure::Retryable`] otherwise.
    pub fn compile(
        disk: &Disk,
        architecture: &str,
        identity: &str,
    ) -> std::result::Result<Self, Failure> {
        let source = source();
        let key = disk_key(identity, &source);
        // One cubin with two entry points, so one cache slot holds it. The
        // region index is arbitrary but must not collide with a real program's.
        let cubin = match disk.load(&key, CONTROL_REGION, READINESS_ENTRY) {
            Some(cubin) => cubin,
            None => {
                let cubin =
                    super::compile::compile(&source, architecture).map_err(
                        |error| match error.kind {
                            FailureKind::Deterministic => Failure::Deterministic {
                                reason: error.message,
                            },
                            FailureKind::Retryable => Failure::Retryable {
                                reason: error.message,
                            },
                        },
                    )?;
                disk.store(&key, CONTROL_REGION, READINESS_ENTRY, &cubin);
                cubin
            }
        };

        let load = |entry: &str| -> std::result::Result<Arc<Module>, Failure> {
            Module::load(&cubin, entry)
                .map(Arc::new)
                .map_err(|error| Failure::Retryable {
                    reason: format!("loading control kernel '{entry}': {error}"),
                })
        };
        Ok(Self {
            readiness: load(READINESS_ENTRY)?,
            commit: load(COMMIT_ENTRY)?,
        })
    }

    /// The readiness kernel.
    #[must_use]
    pub fn readiness(&self) -> &Module {
        &self.readiness
    }

    /// The commit-bump kernel.
    #[must_use]
    pub fn commit(&self) -> &Module {
        &self.commit
    }
}

/// The region index the control pair is cached under.
///
/// `u32::MAX`: a real program's regions are dense from zero, so this cannot collide.
const CONTROL_REGION: u32 = u32::MAX;

#[cfg(test)]
mod tests_3 {
    use super::*;

    /// The host and kernel must index `full` with the same stride; this checks
    /// the interpolation happened rather than a literal surviving.
    #[test]
    fn the_ring_pitch_in_the_source_is_the_rust_constant() {
        let source = source();
        assert!(
            source.contains(&format!("#define PIE_MAX_RING {MAX_RING}u")),
            "the kernel must be built with the host's own ring pitch"
        );
        assert_eq!(MAX_RING, 64, "the C++ ring pitch is 64 and this is a port");
    }

    /// `%` is an ordinary character in a Rust `format!` string — only braces are
    /// special — so a doubled `%%` would reach NVRTC verbatim as `a %% b`.
    #[test]
    fn the_modulo_operators_are_single_percent_signs() {
        let source = source();
        assert_eq!(
            source.matches("% cap1[c]").count(),
            3,
            "three modulo operations over cap1: the ring-not-full test in \
             readiness, and the two cursor advances in commit"
        );
        assert!(
            !source.contains("%%"),
            "`%` is not escaped in a Rust format string; a doubled one reaches \
             NVRTC verbatim and `a %% b` is not an expression"
        );
    }

    /// The source must name no header and no `std::`: NVRTC compiles it with no
    /// include path, so an `#include` is a compile failure per process.
    #[test]
    fn the_source_reaches_for_no_header_it_cannot_have() {
        let source = source();
        assert!(!source.contains("#include"), "there is no include path");
        assert!(!source.contains("std::"), "there is no standard library");
    }

    /// Both entry points must be `extern "C"`, or NVRTC mangles the name the
    /// driver looks up and the kernel loads but cannot be found.
    #[test]
    fn both_entry_points_have_c_linkage_so_their_names_survive() {
        let source = source();
        for entry in [READINESS_ENTRY, COMMIT_ENTRY] {
            assert!(
                source.contains(&format!("extern \"C\" __global__ void {entry}(")),
                "{entry} must be declared with C linkage"
            );
        }
    }

    /// Puts before takes: a channel both taken and put in one pass (every decode
    /// loop) keeps a different value under each order.
    #[test]
    fn the_commit_publishes_before_it_consumes() {
        let source = source();
        let commit = source
            .split_once(COMMIT_ENTRY)
            .expect("the commit kernel is in the source")
            .1;
        let put_at = commit.find("put_ch[i]").expect("the put loop");
        let take_at = commit.find("taken_ch[i]").expect("the take loop");
        assert!(
            put_at < take_at,
            "publishing must precede consuming; the other order leaves a \
             loop-carried channel holding what the pass consumed rather than \
             what it produced"
        );
    }
}
