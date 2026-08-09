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
//! # Scratch is uniform across lanes, and so are the offsets
//!
//! One `offsets` table serves every lane, so a value's offset must be the same
//! in every lane even when the lanes differ in extent. [`Prepared::build`]
//! resolves each lane's descriptors separately and then lays out against the
//! WIDEST descriptor for each value — the same running maximum the C++ takes.
//! Each lane still gets its own descriptor ROW, because the kernel reads
//! `all_descriptors + lane * value_count` and a lane must see its own lengths.
//!
//! # Grouping: the policy question is settled
//!
//! The lanes of a fire MAY be different instances, and the fire is built for
//! that: [`Prepared::build`] takes one [`Lane`] per member — its own rings and
//! its own extents — rather than one ring set and a slice of extents. A
//! caller that grouped instances against a single ring set would have every
//! member reading the first one's cells, which is a wrong answer rather than
//! a fault, because the cells are real and just somebody else's.
//!
//! What settled it was LoRA. Each request's adapter is resolved against that
//! request's instance and its own committed cells
//! (`fire::lora::lane_for_instance`), so "one fire, several instances,
//! several channel sets" is a shape this driver already builds — the sampler
//! is simply the last caller that does not group yet.
//!
//! What is still missing is that caller: `serve`'s `run_program` fires once
//! per request, so today every `Prepared` has one lane. The remaining work is
//! a scheduler decision about which requests batch together, not a layout
//! one.
//!
//! Also the intrinsic side tables beyond binding them empty, and the ticketed
//! channel path.

use driver::driver_api::plan::{LaunchOp, LaunchStagePlan};
use driver::tensor_ir::DType;
use driver::tensor_ir::op::{IntrinsicId, tags};
use driver::{
    Diagnosis, Extents, LANE_HEADER_BYTES, LANE_RECORD_BYTES, LANE_SLOT_BYTES, LaneChannelSlot,
    LaneHeader, LaneRecord, LaneShape, NO_TICKET, OpParams, OpRuntime, SCRATCH_ALIGN,
    StatusOutcome, ValueDesc, describe, layout,
};

use crate::gpu::device::{Allocator, DeviceBuffer, StreamRef};
use crate::error::{Error, Result};

use super::params::{CudaOpParams, params_bytes};
use super::channel::Rings;
use super::runtime::Region;

/// How many intrinsic slots the side tables carry per lane.
///
/// `PTIR_INTRINSIC_SLOTS` — `PTIR_INTR_ATTN_SCORE + 1`. The five arrays are
/// indexed `lane * INTRINSIC_SLOTS + intr`, so the constant is a stride and
/// getting it wrong misdirects every intrinsic of every lane but the first.
pub const INTRINSIC_SLOTS: usize = 16;

/// The sixteen arguments a generated fused region takes.
const FUSED_ARITY: usize = 16;

/// One member of a grouped fire: its rings and its extents.
///
/// A STRUCT rather than two parallel slices, because both are per-lane
/// facts about the same member and a caller that zipped them wrongly
/// would build a fire whose lane 0 read lane 1's channels — a wrong
/// answer rather than a fault, because the cells are real and just
/// somebody else's.
#[derive(Clone, Copy)]
pub struct Lane<'a> {
    /// The instance's own channel rings.
    pub rings: &'a Rings,
    /// How much this member submitted.
    pub extents: Extents,
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
        // The kernel reads `all_descriptors + lane * value_count`, so every
        // lane owns a full row — its own extents resolved against the same
        // value types.
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
        // ONE offsets table serves every lane — the kernel has no per-lane
        // offsets parameter — so a value's offset must be the same in every
        // lane even when the lanes differ in extent. Lay out against the
        // WIDEST descriptor for each value, which is what the C++ does by
        // taking a running maximum across lanes.
        let mut descriptors: Vec<ValueDesc> = per_lane[0].clone();
        for lane in &per_lane[1..] {
            for (widest, one) in descriptors.iter_mut().zip(lane) {
                if one.device_bytes() > widest.device_bytes() {
                    *widest = *one;
                }
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
        // One, not zero. The kernel's first act is to read this and return if
        // it is clear, so a fire that started at zero would launch every
        // kernel and do nothing, which looks exactly like a blocked fire.
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
        // `channel_slot_offset` is the lane's own row in the flat slot
        // array: the kernel indexes `channels[lane * slots + n]` and the
        // record has to agree with that arithmetic, so it is
        // `lane * channel_count` and not zero.
        let slots_at = LANE_HEADER_BYTES as usize + lanes as usize * LANE_RECORD_BYTES as usize;
        for (lane, member) in lanes_in.iter().enumerate() {
            let extents = &member.extents;
            // EACH LANE'S OWN RINGS. The members of a grouped fire are
            // different INSTANCES — one per request — and an instance's
            // channels live in its own session. Resolving every lane
            // against one ring set would have every member reading the
            // first one's cells, which is a wrong answer rather than a
            // fault: the cells are real, they are just somebody else's.
            //
            // The lane table's arithmetic always permitted this — every
            // lane owns its own slot row — so what changed is the
            // CALLER, which is what `ptir::run`'s header said was
            // missing.
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

            // Absolute cell addresses, resolved on the host, from THIS
            // session's rings — so the lanes of one fire are members of one
            // instance's batch and share its channels.
            //
            // WHETHER THAT IS THE ONLY GROUPING is not settled here. A fire
            // whose lanes were different INSTANCES would need each lane's
            // slots resolved against that instance's own rings, and this
            // loop would take a per-lane ring source instead of one. The
            // lane table's arithmetic already permits it — every lane owns
            // its own slot row — so that is a change of caller, not of
            // layout. Nothing groups yet, so nothing here has to choose.
            for (local, &dense) in plan.channel_bindings.iter().enumerate() {
                let channel = dense as usize;
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
                    // NOT a ticket. The ticketed path lets a host stage a
                    // table ahead of the fire that uses it and have the
                    // kernel refuse a stale one; nothing here stages, so
                    // claiming a ticket the host did not observe would be a
                    // check that passes for the wrong reason.
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
                            "program::run",
                            format!(
                                "a put names stage-local channel {}, which is unbound",
                                op.channel
                            ),
                        )
                    })?;
                // THE FIRST LANE'S SHAPES. Every member of a grouped
                // fire binds the SAME plan, so the channel geometry is
                // the plan's and not the member's — only the cursors
                // differ, which is why those are read per lane above.
                let shape = lanes_in[0].rings.shape(dense as usize).ok_or_else(|| {
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
        // Descriptors are LANE-MAJOR: the kernel reads
        // `all_descriptors + lane * value_count`.
        let descriptor_bytes: Vec<u8> =
            per_lane.iter().flatten().flat_map(as_bytes).collect();
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
                "program::run",
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
                "program::run",
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

// ── Launching the prepared fire above ──

// `cuLaunchKernel`, and the argument marshalling it needs.
//
// # Why the arguments are the whole content of this file
//
// `cuLaunchKernel` takes `void**` — an array of pointers to each argument's
// *storage*, not the argument values. So passing a device pointer means
// taking the address of the variable that holds it, and passing a `u32` means
// taking the address of a `u32` that must outlive the call. Getting that one
// level of indirection wrong does not fail: the driver reads whatever is at
// the address it was given, and the kernel sees a plausible number.
//
// It is also completely unchecked. There is no arity check, no type check,
// and no diagnostic — a sixteen-parameter kernel handed fifteen arguments
// reads its sixteenth from uninitialised memory. So the marshalling lives
// here, in one place, behind [`Args`], and every launcher builds its list
// through it rather than assembling a `Vec<*mut c_void>` at the call site.
//
// # The one-CTA-per-lane rule
//
// A generated fused region is launched with `grid.x = lane_count` and
// `block.x` from the compiled function's own attribute, because the kernel's
// first line is `dispatch_lane = blockIdx.x`. It is not a tuning choice: two
// lanes per block would have both write the same `commit_slot`, and a grid
// smaller than the lane count silently drops the tail lanes.

use cudarc::driver::sys as dr;


use super::compile::Module;

/// A kernel's argument list, kept alive for the launch.
///
/// The storage and the pointer array are one value on purpose. `cuLaunchKernel`
/// dereferences the pointers *during* the call, so the scalars must outlive
/// it; a builder that returned only the `Vec<*mut c_void>` would compile and
/// would be reading freed stack by the time the driver looked.
#[derive(Default)]
pub struct Args {
    /// Boxed so that pushing another scalar cannot move an earlier one and
    /// invalidate a pointer already recorded in `slots`. A `Vec<u64>` would
    /// reallocate and leave every previous entry dangling — and the launch
    /// would still succeed, with the kernel reading whatever now lives there.
    ///
    /// Clippy calls this an unnecessary box, and it is wrong here for a reason
    /// worth stating: its rule is about the indirection being redundant when
    /// the only thing that matters is the value. What matters here is the
    /// ADDRESS, which is precisely what `Vec`'s reallocation does not preserve
    /// and `Box`'s does.
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

    /// Append a `u32` argument.
    ///
    /// Stored in a `u64` cell and pointed at its first four bytes, which is
    /// correct on every little-endian host — the only kind CUDA runs on, and
    /// the same assumption the ABI's own records make.
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
/// If the driver refuses the launch. Note that a launch is asynchronous, so a
/// fault *inside* the kernel is not reported here — it surfaces at the next
/// synchronization, which is why every caller in this crate synchronizes
/// before it believes a result.
///
/// # Panics
///
/// Never. `expected` is checked against `args.len()` and returns an error.
pub fn launch(
    module: &Module,
    grid: u32,
    block: u32,
    args: &mut Args,
    expected: usize,
    stream: StreamRef<'_>,
) -> Result<()> {
    // The arity check CUDA does not do. A kernel handed too few arguments
    // reads the rest from whatever follows the array, and the failure appears
    // as a wrong answer rather than as an error.
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
    // SAFETY: `module.function()` came from a loaded module and is live for
    // the borrow; `args` holds every scalar the pointer array points at for
    // the duration of this call; no shared memory is requested and no extra
    // block is passed.
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
/// A module of its own rather than free functions because the pair share a
/// shape no other launch has: both take the four ring arrays, both take two
/// channel-index lists that have to be uploaded, and both are single-thread.
pub mod launch_control {
    use super::{Allocator, Args, Control, Error, Result, Rings, StreamRef, launch};

    /// Ask whether a pass may commit.
    ///
    /// `need_full` are the channels the pass consumes — their committed cell
    /// must hold a value — and `need_empty` the ones it produces, which need
    /// room. Returns what the kernel decided.
    ///
    /// The verdict is read back rather than left on the device because the
    /// host has to know: a blocked fire is reported to the runtime as a retry,
    /// and that decision cannot be made on the GPU.
    ///
    /// # Errors
    ///
    /// If an upload, the launch, or the readback fails.
    pub fn readiness(
        control: &Control,
        rings: &Rings,
        need_full: &[u32],
        need_empty: &[u32],
        alloc: &Allocator,
        stream: StreamRef<'_>,
    ) -> Result<bool> {
        // The flag starts at 1 and the kernel ANDs into it, which is what lets
        // several stages narrow one pass's verdict without any of them needing
        // to know about the others.
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
    /// `committed` is the readiness verdict. When it is false every kernel
    /// still launches — that is the dummy run, and it is what makes a blocked
    /// fire cost the same as a running one instead of branching on the device
    /// — and this call moves nothing.
    ///
    /// # Errors
    ///
    /// If an upload or the launch fails.
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
        // The flag buffer must outlive the launch, and the launch is
        // asynchronous. Synchronizing here is the cheap correct answer while
        // there is one fire in flight; a pipelined shell would keep the
        // allocation alive against the stream instead.
        stream.synchronize()?;
        Ok(())
    }

    /// Upload a channel-index list, or `None` when it is empty.
    ///
    /// `None` rather than a zero-byte allocation: the kernel reads the array
    /// only `count` times, so a null with a count of zero is exactly correct
    /// and an empty allocation is a pointer the allocator may or may not
    /// return.
    fn upload_indices(
        alloc: &Allocator,
        indices: &[u32],
        stream: StreamRef<'_>,
    ) -> Result<Option<crate::gpu::device::DeviceBuffer>> {
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

    /// Every scalar must still be where its recorded pointer says it is after
    /// more have been appended. A `Vec<u64>` backing would reallocate and
    /// leave the earlier pointers dangling — and the launch would succeed,
    /// with the kernel reading whatever now lives at those addresses.
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

    /// A pointer argument is the ADDRESS OF the pointer, not the pointer. One
    /// level of indirection either way is a plausible number the kernel reads
    /// without complaint.
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

// The two control kernels: readiness before a pass, the commit bump after it.
//
// # Why these are compiled here rather than linked
//
// The host emitter produces one kind of kernel for CUDA — a fused region per
// generated region — and emits no readiness kernel and no commit kernel,
// because on CUDA those are *prebuilt*. The C++ driver's prebuilt copies live
// in `driver-cuda/csrc/src/pipeline/channels.hpp`, a private header of a
// crate this one is replacing. They are not in `libpie_kernels_cuda.a`: the
// kernel table has no row for them, `kernels-cuda/csrc` includes
// `channels.hpp` nowhere, and `driver-cuda --features bridge` therefore
// links an archive containing neither symbol.
//
// So they have to come from somewhere, and there are two honest options: add
// them to the kernels crate and build them with nvcc, or compile them here
// through the NVRTC path this driver already has for every emitted region.
//
// This file takes the second, and the reason is the toolkit-free build. A
// kernels-crate row would mean `driver-cuda` cannot run a PTIR program
// without `bridge`, which needs nvcc *at build time* — and that build is
// load-bearing for CI. Compiling forty lines through the NVRTC path already
// in `super::nvrtc` costs one compile per process, is cached on disk beside
// every other region, and needs no toolkit at build time at all.
//
// # These are ports and reproduce the C++ exactly
//
// Both bodies are transcriptions of `channels.hpp:93-149`, including the
// `threadIdx.x != 0 || blockIdx.x != 0` guard that makes each a single-thread
// kernel, and including the ORDER of the commit loops — puts before takes.
// The order matters for a channel that is both taken and put in one pass (a
// loop-carried ping-pong): publishing first and consuming second is what
// leaves such a ring with the value the pass produced rather than with the
// one it consumed.
//
// # The types are spelled in bare C
//
// `unsigned char` and `unsigned int`, not `std::uint8_t`/`std::uint32_t`.
// NVRTC compiles these with **no include path at all** — the same condition
// the emitted regions are compiled under — so `<cstdint>` cannot be found and
// `std::` does not exist. The M1 runtime prologue solves the same problem the
// same way, with its `m1_u8`/`m1_u32` typedefs, and says so in its own
// header comment.

use std::sync::Arc;

use driver::Failure;

use super::cache::{Disk, disk_key};
use super::compile::{FailureKind};

/// The ring's fixed row pitch in the `full` array.
///
/// `full` is two-dimensional and indexed `full[channel * MAX_RING + slot]`
/// with **this** stride whatever a channel's actual capacity is. Using `cap1`
/// as the pitch instead would be the natural-looking mistake and would make
/// every channel past the first index into its neighbour's flags.
pub const MAX_RING: u32 = 64;

/// The readiness kernel's entry point.
pub const READINESS_ENTRY: &str = "pie_ptir_stage_readiness";

/// The commit-bump kernel's entry point.
pub const COMMIT_ENTRY: &str = "pie_ptir_commit_bump";

/// Both kernels, in one translation unit.
///
/// One unit rather than two because they are compiled together, cached
/// together, and neither is ever wanted without the other: a pass that can
/// decide it may run and cannot then publish is not a pass.
///
/// `MAX_RING` is interpolated rather than written as a literal, so the Rust
/// constant above is the single definition. A driver whose host arithmetic
/// used 64 and whose kernel used 32 would read the right flags for channel 0
/// and the wrong ones for every channel after it.
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
    /// Cached under a key of their own rather than a program's: they belong to
    /// no program, they are identical for every program on one device, and a
    /// key derived from a program would recompile them per program.
    ///
    /// # Errors
    ///
    /// [`Failure::Deterministic`] if the source above does not compile, which
    /// can only mean this file and the NVRTC in use disagree about the
    /// language; [`Failure::Retryable`] for anything the machine refused.
    pub fn compile(disk: &Disk, architecture: &str, identity: &str) -> std::result::Result<Self, Failure> {
        let source = source();
        let key = disk_key(identity, &source);
        // The pair is ONE cubin with two entry points, so one cache slot holds
        // it and the two modules below are two lookups into one image. The
        // region index is arbitrary and simply must not collide with a real
        // program's -- these share the directory, not the key space.
        let cubin = match disk.load(&key, CONTROL_REGION, READINESS_ENTRY) {
            Some(cubin) => cubin,
            None => {
                let cubin =
                    super::compile::compile(&source, architecture).map_err(|error| match error.kind {
                        FailureKind::Deterministic => Failure::Deterministic {
                            reason: error.message,
                        },
                        FailureKind::Retryable => Failure::Retryable {
                            reason: error.message,
                        },
                    })?;
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
/// A real program's regions are numbered from zero and are dense, so a large
/// number cannot collide with one. It is not a magic constant so much as a
/// statement that these two kernels are not a region of anything.
const CONTROL_REGION: u32 = u32::MAX;

#[cfg(test)]
mod tests_3 {
    use super::*;

    /// The stride the host indexes `full` with must be the one the kernel
    /// indexes it with. Interpolation is what guarantees it; this checks the
    /// interpolation actually happened rather than the literal surviving.
    #[test]
    fn the_ring_pitch_in_the_source_is_the_rust_constant() {
        let source = source();
        assert!(
            source.contains(&format!("#define PIE_MAX_RING {MAX_RING}u")),
            "the kernel must be built with the host's own ring pitch"
        );
        assert_eq!(MAX_RING, 64, "the C++ ring pitch is 64 and this is a port");
    }

    /// `%` is the modulo the ring arithmetic needs, and in a Rust `format!`
    /// string it is an ORDINARY CHARACTER. Only braces are special.
    ///
    /// This test was written expecting the printf rule — that `%%` escapes to
    /// `%` — and it caught the mistake on its first run: the doubled percent
    /// went through verbatim and would have reached NVRTC as `a %% b`, which
    /// is not an expression. It is kept, unchanged in intent, because the
    /// reflex it guards against is the one a reader of a template full of
    /// `{{` and `}}` is most likely to have.
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

    /// The source must name no header and no `std::`, because NVRTC compiles
    /// it with no include path — the same condition the emitted regions are
    /// compiled under. A single `#include` here is a compile failure per
    /// process with a diagnostic three steps from its cause.
    #[test]
    fn the_source_reaches_for_no_header_it_cannot_have() {
        let source = source();
        assert!(!source.contains("#include"), "there is no include path");
        assert!(!source.contains("std::"), "there is no standard library");
    }

    /// Both entry points must be `extern "C"`: NVRTC mangles a C++ name, and
    /// the driver looks each up by the string above. A missing linkage
    /// specifier compiles, loads, and then cannot be found.
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

    /// Puts before takes. A channel that is both taken and put in one pass is
    /// the shape of every decode loop, and the two orders leave different
    /// values in the ring.
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
