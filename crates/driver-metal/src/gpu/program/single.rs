//! Preparing and executing one single-lane fire: the M1 path.
//!
//! `prepare` asks whether the fire may run — the host readiness check — and
//! if so builds its lane table: one lane, one channel slot per ring, the
//! status buffer the kernels commit into. `execute` derives the fire's
//! scratch from the plan and the fire's numbers, fills the descriptor,
//! offset and parameter buffers, binds every region's argument table, and
//! runs readiness → regions → commit as one step.
//!
//! ## What the C++ needed that this does not
//!
//! * **`goto cleanup_failure`.** The C++ acquires four transient buffers and
//!   must recycle them on every one of nine failure exits, so the failure
//!   exits are a label and a `goto` — and the cleanup block is written twice
//!   anyway, once for failure and once inline for success. A [`Transient`]
//!   recycles when dropped; every `?` below is that label.
//! * **`resource_accounted` and `release()`.** The fire registered its
//!   channel storages as external buffers and carried a flag so the release
//!   would not double-count. The rings here are the runtime's own resident
//!   allocations, held alive by the fire's `Rc`s; there is nothing to
//!   register and nothing to release by hand. The global
//!   `m1_prepared_resource_counters` audit dies with the question it
//!   audited.
//! * **The zero-fill readback lie.** The M1 status decode treated "not 4 and
//!   not 2" as an op fault, so a kernel that never wrote its status — or
//!   never finished — was reported as `"generated op fault 0"`.
//!   [`Outcome::of`](crate::channel::StatusOutcome) separates all four; the
//!   report names the fault class instead of printing the code in decimal.
//!
//! ## What stays the C++'s shape
//!
//! The binding layout is the kernels' ABI and is copied exactly: the
//! single-lane effect kernels take `(status, lane_table, words...)`, a
//! singleton region takes `(status, descriptors, a0, a1, a2, o0, o1,
//! temporary, params)`, a fused region takes `(status, descriptors, params,
//! offsets, scratch, temporary, logits, cells...)`. PTIR channels keep
//! register semantics within a pass: a take after a put observes the pending
//! cell, which is the `pending` bookkeeping below.

use std::rc::Rc;

use tensor_ir::op::tags;

use crate::gpu::device::context::Context;
use crate::gpu::device::encoder::{Stepper, Visibility};
use crate::gpu::device::external::Externals;
use crate::gpu::device::handle::Handle;
use crate::gpu::device::allocator::{Pool, Transient};
use crate::gpu::program::executable::ProgramExecutable;
use crate::gpu::device::ring::Ring;
use crate::gpu::program::cache::Runtime;
use crate::gpu::device::argtable::Tables;
use crate::gpu::device::step_cost::Timing;
use crate::channel::{
    DUMMY_BYTES, Extents, LANE_ABI_VERSION, LaneChannelSlot, LaneHeader, LaneRecord, LaneShape,
    NO_TICKET, OpParams, OpRuntime, Readiness, Reason, STATUS_BYTES, Status, StatusOutcome, Ticket,
    ValueDesc, Words, check_words, describe, layout, report_status,
};
use crate::layout::region::Region;
use crate::{Error, Result};

/// `PTIR_OP_CHAN_TAKE`.
const CHAN_TAKE: u16 = tags::CHAN_TAKE as u16;
/// `PTIR_OP_CHAN_READ`.
const CHAN_READ: u16 = tags::CHAN_READ as u16;
/// `PTIR_OP_CHAN_PUT`.
const CHAN_PUT: u16 = tags::CHAN_PUT as u16;
/// `PTIR_OP_INTRINSIC_VAL`.
const INTRINSIC_VAL: u16 = tags::INTRINSIC_VAL as u16;
/// `PTIR_INTR_MTP_LOGITS` and `PTIR_INTR_MTP_DRAFTS`, the row-offset pair.
const MTP_LOGITS: u16 = tensor_ir::op::intrinsic_tags::MTP_LOGITS;
/// See [`MTP_LOGITS`].
const MTP_DRAFTS: u16 = tensor_ir::op::intrinsic_tags::MTP_DRAFTS;

/// The fire's runtime numbers: what the forward produced and how big the
/// batch really is.
///
/// `M1DeviceInputs`, with the two sentinels made absent: no logits is
/// `None`, not an invalid handle, and no MTP draft row is `None`, not `-1`.
#[derive(Clone, Debug, Default)]
pub struct DeviceInputs {
    /// The forward's bf16 logits, when the program reads them.
    pub logits: Option<Handle>,
    /// First row of the logits buffer belonging to this fire.
    pub logits_row_offset: u32,
    /// Rows of the logits buffer belonging to this fire.
    pub logits_row_count: u32,
    /// The explicit logits row map, when the rows are not contiguous.
    ///
    /// Empty means rows `logits_row_offset ..` in order. Only the M3 group
    /// reads it; when present its length must equal
    /// [`logits_row_count`](Self::logits_row_count).
    pub logits_rows: Vec<u32>,
    /// The model's vocabulary width.
    pub vocab: u32,
    /// The MTP draft row this fire runs, if any.
    pub mtp_draft_row: Option<u32>,
    /// The runtime extents symbolic shapes resolve against.
    pub extents: Extents,
}

/// A fire that passed the host readiness check and holds everything it
/// binds.
///
/// `M1PreparedFire`. Owns its status and lane-table buffers (recycled on
/// drop), its cell views, and an `Rc` on every ring — the storage cannot go
/// away under the fire, which is the whole of what `resource_accounted`
/// was balancing by hand.
pub struct PreparedFire {
    pub(super) program: Rc<ProgramExecutable>,
    pub(super) rings: Vec<Rc<Ring>>,
    pub(super) tickets: Vec<Ticket>,
    /// The cell a take reads, per channel, at the prepared head.
    pub(super) committed: Vec<Handle>,
    /// The cell a put writes, per channel, at the prepared tail.
    pub(super) pending: Vec<Handle>,
    /// One [`Status`] record, zeroed at prepare and before every run.
    pub(super) status: Transient,
    /// The single-lane table: header, one record, one slot per channel.
    pub(super) lane_table: Transient,
}

impl PreparedFire {
    /// The program this fire runs.
    #[must_use]
    pub fn program(&self) -> &Rc<ProgramExecutable> {
        &self.program
    }

    /// The tickets the fire was composed against.
    ///
    /// The M3 group builder re-checks readiness per candidate and writes
    /// these into the group's lane table.
    #[must_use]
    pub fn tickets(&self) -> &[Ticket] {
        &self.tickets
    }

    /// The status the kernels last wrote, if it parses.
    #[must_use]
    pub fn status(&self) -> Option<Status> {
        // SAFETY: the buffer is STATUS_BYTES long and the step that wrote it
        // has signalled before anyone asks.
        let bytes = unsafe {
            std::slice::from_raw_parts(self.status.contents().as_ptr().cast::<u8>(), STATUS_BYTES)
        };
        Status::read(bytes)
    }
}

impl std::fmt::Debug for PreparedFire {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedFire")
            .field("program", &format_args!("{:#x}", self.program.program_hash))
            .field("channels", &self.rings.len())
            .finish_non_exhaustive()
    }
}

/// What `prepare` decided.
#[derive(Debug)]
pub enum Prepare {
    /// The fire may run; here is everything it binds.
    ///
    /// Behind an `Rc` because the placed paths share it: the same prepared
    /// fire backs an M1 retry, an M2 command and an M3 lane candidate.
    Ready(Rc<PreparedFire>),
    /// The fire is early — a channel is not in the state it needs, and
    /// waiting is the remedy. Nothing was allocated.
    Retry {
        /// The channel that was not ready.
        channel: usize,
        /// What about it.
        reason: Reason,
    },
}

/// Which dispatch list `execute` runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// One op per dispatch — the path of last resort, always present.
    #[default]
    Singleton,
    /// The host-fused regions, where the stage supports them.
    Fused,
}

/// How one execution ended.
#[derive(Debug)]
pub struct Execution {
    /// What the caller does with the fire.
    pub outcome: StatusOutcome,
    /// The human account, when the outcome is not `Committed`.
    pub report: Option<String>,
    /// The step's measured cost.
    pub timing: Timing,
}

impl Runtime {
    /// Check readiness and build the fire's lane table.
    ///
    /// `rings`, `tickets` and the program's effects are parallel, one entry
    /// per dense channel.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] when the fire can never run — a poisoned, closed
    /// or inconsistent ring, or tables whose lengths disagree. Allocation
    /// failures surface as their own errors. An early fire is not an error;
    /// it is [`Prepare::Retry`].
    pub fn prepare(
        &mut self,
        context: &Context,
        pool: &Pool,
        program: &Rc<ProgramExecutable>,
        rings: &[Rc<Ring>],
        tickets: &[Ticket],
    ) -> Result<Prepare> {
        let words: Vec<Words> = rings.iter().map(|ring| ring.snapshot()).collect();
        match check_words(&words, &program.effects, tickets) {
            Readiness::Ready => {}
            Readiness::Retry { channel, reason } => {
                return Ok(Prepare::Retry { channel, reason });
            }
            Readiness::Failed { channel, reason } => {
                return Err(Error::Program {
                    message: format!(
                        "Metal M1 fire can never run: channel {channel:?}: {reason:?}"
                    ),
                });
            }
            Readiness::Mismatched {
                channels,
                effects,
                tickets,
            } => {
                return Err(Error::Program {
                    message: format!(
                        "Metal M1 fire/channel layout mismatch: {channels} channels, \
                         {effects} effects, {tickets} tickets"
                    ),
                });
            }
        }

        let shape = LaneShape::of(1, rings.len() as u32);
        let table_bytes = shape.bytes().ok_or_else(|| Error::Program {
            message: "Metal M1 lane table exceeds u64".to_owned(),
        })?;
        let status = pool.acquire(context, STATUS_BYTES as u64)?;
        let lane_table = pool.acquire(context, table_bytes)?;
        // SAFETY: both buffers were just acquired; no step names them yet.
        unsafe {
            status.zero(0, status.len())?;
            lane_table.zero(0, lane_table.len())?;
        }

        write_pod(
            &lane_table,
            0,
            &LaneHeader {
                abi_version: LANE_ABI_VERSION,
                lane_count: 1,
                channel_slots_per_lane: rings.len() as u32,
                flags: 0,
            },
        )?;
        write_pod(
            &lane_table,
            offset(shape.record_offset(0), "lane record")?,
            &LaneRecord {
                channel_slot_offset: 0,
                commit_slot: status.gpu_address(),
                ..LaneRecord::default()
            },
        )?;

        let mut committed = Vec::with_capacity(rings.len());
        let mut pending = Vec::with_capacity(rings.len());
        for (channel, (ring, ticket)) in rings.iter().zip(tickets).enumerate() {
            // The prepared cells sit at the *composed* sequences when the
            // ticket pinned them, at the live ones otherwise.
            let head = if ticket.expected_head == NO_TICKET {
                ring.head()
            } else {
                ticket.expected_head
            };
            let tail = if ticket.expected_tail == NO_TICKET {
                ring.tail()
            } else {
                ticket.expected_tail
            };
            let take_cell = ring.committed_cell(head)?;
            let put_cell = ring.pending_cell(tail)?;
            write_pod(
                &lane_table,
                offset(shape.slot_offset(0, channel as u32), "channel slot")?,
                &LaneChannelSlot {
                    committed_cell: take_cell.gpu_address(),
                    pending_cell: put_cell.gpu_address(),
                    expected_head: ticket.expected_head,
                    expected_tail: ticket.expected_tail,
                },
            )?;
            committed.push(take_cell);
            pending.push(put_cell);
        }

        Ok(Prepare::Ready(Rc::new(PreparedFire {
            program: Rc::clone(program),
            rings: rings.to_vec(),
            tickets: tickets.to_vec(),
            committed,
            pending,
            status,
            lane_table,
        })))
    }

    /// Bind, encode and run one fire, and read back what the GPU said.
    ///
    /// `tables` is the ordinal-keyed argument table cache the dispatches
    /// bind through; `pool` supplies the fire's scratch. Both belong to the
    /// caller because they belong to the *context*, not to this runtime's
    /// caches.
    ///
    /// # Errors
    ///
    /// Infrastructure failures — allocation, layout, binding, a wedged
    /// step — are `Err`. A fire the GPU refused is not an error; it is an
    /// [`Execution`] whose outcome says retry or failed, with the report.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        context: &Context,
        stepper: &mut Stepper<'_>,
        tables: &mut Tables,
        pool: &Pool,
        externals: &Externals,
        fire: &PreparedFire,
        inputs: &DeviceInputs,
        mode: Mode,
    ) -> Result<Execution> {
        let program = &fire.program;

        // The single-lane effect kernels: status, lane table, then every
        // ring's words.
        for ordinal in [program.readiness_ordinal, program.commit_ordinal] {
            tables.bind_address(context, ordinal, 0, fire.status.gpu_address())?;
            tables.bind_address(context, ordinal, 1, fire.lane_table.gpu_address())?;
            for (channel, ring) in fire.rings.iter().enumerate() {
                tables.bind_address(context, ordinal, channel + 2, ring.words().gpu_address())?;
            }
        }

        // Resolve every stage's value shapes against the fire's numbers.
        let mut descriptors: Vec<ValueDesc> = Vec::new();
        let mut stage_value_bases = Vec::with_capacity(program.stages.len());
        let mut region_count = 0usize;
        for stage in &program.stages {
            stage_value_bases.push(descriptors.len());
            region_count += stage.executable.regions.len();
            for value in &stage.plan.value_types {
                descriptors.push(describe(value, &inputs.extents).map_err(|why| {
                    Error::Program {
                        message: format!("Metal M1 value shape did not resolve: {why:?}"),
                    }
                })?);
            }
        }
        let scratch_layout = layout(&descriptors).map_err(|why| Error::Program {
            message: format!("Metal M1 fire scratch: {why:?}"),
        })?;

        const DESC_BYTES: u64 = size_of::<ValueDesc>() as u64;
        const PARAM_BYTES: u64 = size_of::<OpParams>() as u64;
        let scratch = pool.acquire(context, scratch_layout.total)?;
        // `max(1)`: Metal rejects a zero-length buffer, and an empty
        // descriptor list is a legal fire.
        let descriptor_buffer =
            pool.acquire(context, (descriptors.len() as u64).max(1) * DESC_BYTES)?;
        let parameter_buffer = pool.acquire(context, (region_count as u64).max(1) * PARAM_BYTES)?;
        let offset_buffer = pool.acquire(context, (descriptors.len() as u64).max(1) * 4)?;

        // SAFETY: `ValueDesc` is `#[repr(C)]` and uploaded as bytes by
        // design (its size is asserted as ABI); the buffers were just
        // acquired and no step names them.
        unsafe {
            descriptor_buffer.write(0, pod_bytes(&descriptors))?;
            let offsets: Vec<u32> = scratch_layout.values.iter().map(|&v| v as u32).collect();
            offset_buffer.write(0, pod_bytes(&offsets))?;
        }

        let scratch_all = Handle::over(scratch.buffer(), scratch.len())?;
        let dummy = scratch_all.slice(0, DUMMY_BYTES)?;
        let temporary =
            scratch_all.slice(scratch_layout.temporary, scratch_layout.temporary_bytes)?;
        let descriptors_all = Handle::over(descriptor_buffer.buffer(), descriptor_buffer.len())?;
        let parameters_all = Handle::over(parameter_buffer.buffer(), parameter_buffer.len())?;
        let offsets_all = Handle::over(offset_buffer.buffer(), offset_buffer.len())?;

        // The forward's logits live in a buffer some other owner allocated;
        // keep it resident for the step's duration.
        let _logits_resident = inputs
            .logits
            .as_ref()
            .map(|logits| externals.insert(context, logits.buffer()));

        // The fire's numbers go into the lane record the effect kernels read.
        let record_offset = offset(
            LaneShape::of(1, fire.rings.len() as u32).record_offset(0),
            "lane record",
        )?;
        let extents = &inputs.extents;
        write_pod(
            &fire.lane_table,
            record_offset,
            &LaneRecord {
                logits_base: inputs.logits.as_ref().map_or(0, Handle::gpu_address),
                logits_row_offset: inputs.logits_row_offset,
                logits_row_count: inputs.logits_row_count,
                kv_len: extents.kv_len,
                page_count: extents.page_count,
                row_count: extents.row_count,
                token_count: extents.token_count,
                sampled_rows: extents.sampled_rows,
                query_len: extents.query_len,
                key_len: extents.key_len,
                channel_slot_offset: 0,
                commit_slot: fire.status.gpu_address(),
                ..LaneRecord::default()
            },
        )?;

        // Bind every singleton region: status, stage descriptors, operands,
        // results, temporary, its own params record.
        let runtime = OpRuntime {
            vocab: inputs.vocab,
            mtp_draft_row: inputs.mtp_draft_row,
        };
        let mut pending_seen = vec![false; fire.rings.len()];
        let mut parameter_index = 0u64;
        for (stage_index, stage) in program.stages.iter().enumerate() {
            let value_base = stage_value_bases[stage_index];
            let stage_descriptors = descriptors_all.slice(
                value_base as u64 * DESC_BYTES,
                (stage.plan.value_types.len() as u64).max(1) * DESC_BYTES,
            )?;
            for region in &stage.executable.regions {
                let op = stage
                    .plan
                    .ops
                    .get(region.operation.node as usize)
                    .ok_or_else(|| Error::Program {
                        message: "Metal M1 region names an op outside its stage".to_owned(),
                    })?;
                let mut params = OpParams::of(op, region.operation.result_base, runtime);

                let value_handle = |index: u32| -> Result<Handle> {
                    let global = value_base + index as usize;
                    let descriptor = descriptors.get(global).ok_or_else(|| Error::Program {
                        message: "Metal M1 op names a value outside its stage".to_owned(),
                    })?;
                    scratch_all.slice(scratch_layout.values[global], descriptor.device_bytes())
                };
                let mut a0 = dummy.clone();
                let mut a1 = dummy.clone();
                let mut a2 = dummy.clone();
                let mut o0 = dummy.clone();
                let mut o1 = dummy.clone();
                if !op.args.is_empty() {
                    a0 = value_handle(params.a0)?;
                }
                if OpParams::binds_second_argument(op) {
                    a1 = value_handle(params.a1)?;
                }
                if op.args.len() > 2 {
                    a2 = value_handle(params.a2)?;
                }
                if op.result_count > 0 {
                    o0 = value_handle(params.o0)?;
                }
                if op.result_count > 1 {
                    o1 = value_handle(params.o1)?;
                }

                if op.code == CHAN_TAKE || op.code == CHAN_READ {
                    let dense = dense_channel(stage, op.channel, fire.rings.len())?;
                    a0 = if pending_seen[dense] {
                        fire.pending[dense].clone()
                    } else {
                        fire.committed[dense].clone()
                    };
                } else if op.code == CHAN_PUT {
                    let dense = dense_channel(stage, op.channel, fire.rings.len())?;
                    let wire = descriptors[value_base + params.a0 as usize].wire_bytes();
                    if wire > fire.pending[dense].len() {
                        return Err(Error::Program {
                            message: "Metal M1 channel sink exceeds fixed cell size".to_owned(),
                        });
                    }
                    params.sink_bytes = fire.pending[dense].len() as u32;
                    o0 = fire.pending[dense].clone();
                    // Register semantics within a pass: a later take observes
                    // the pending last put.
                    pending_seen[dense] = true;
                } else if op.code == INTRINSIC_VAL {
                    a0 = logits_operand(
                        inputs,
                        op.intrinsic,
                        &descriptors[value_base + params.o0 as usize],
                    )?;
                }

                let params_handle =
                    parameters_all.slice(parameter_index * PARAM_BYTES, PARAM_BYTES)?;
                // SAFETY: `OpParams` is `#[repr(C)]`, written for the kernel
                // to read; the step has not been committed.
                unsafe { params_handle.write(0, pod_bytes(std::slice::from_ref(&params)))? };
                parameter_index += 1;

                let ordinal = region.ordinal;
                tables.bind_address(context, ordinal, 0, fire.status.gpu_address())?;
                tables.bind_address(context, ordinal, 1, stage_descriptors.gpu_address())?;
                tables.bind_address(context, ordinal, 2, a0.gpu_address())?;
                tables.bind_address(context, ordinal, 3, a1.gpu_address())?;
                tables.bind_address(context, ordinal, 4, a2.gpu_address())?;
                tables.bind_address(context, ordinal, 5, o0.gpu_address())?;
                tables.bind_address(context, ordinal, 6, o1.gpu_address())?;
                tables.bind_address(context, ordinal, 7, temporary.gpu_address())?;
                tables.bind_address(context, ordinal, 8, params_handle.gpu_address())?;
            }
        }

        // Bind every fused region, where the stage has them: the whole-stage
        // buffers, then a (read, write) cell pair per bound channel.
        let fused_logits = match &inputs.logits {
            Some(logits) if inputs.vocab != 0 => Some(logits.slice(
                u64::from(inputs.logits_row_offset) * u64::from(inputs.vocab) * 2,
                logits.len().saturating_sub(
                    u64::from(inputs.logits_row_offset) * u64::from(inputs.vocab) * 2,
                ),
            )?),
            _ => None,
        };
        let mut fused_pending = vec![false; fire.rings.len()];
        let mut fused_parameter_base = 0u64;
        for (stage_index, stage) in program.stages.iter().enumerate() {
            let value_base = stage_value_bases[stage_index] as u64;
            let parameter_base = fused_parameter_base;
            fused_parameter_base += stage.executable.regions.len() as u64;
            let Ok(fused) = &stage.executable.fused else {
                continue;
            };
            let stage_descriptors = descriptors_all.slice(
                value_base * DESC_BYTES,
                (stage.plan.value_types.len() as u64).max(1) * DESC_BYTES,
            )?;
            let stage_parameters = parameters_all.slice(
                parameter_base * PARAM_BYTES,
                (stage.plan.ops.len() as u64).max(1) * PARAM_BYTES,
            )?;
            let stage_offsets = offsets_all.slice(
                value_base * 4,
                (stage.plan.value_types.len() as u64).max(1) * 4,
            )?;
            for region in fused {
                let ordinal = region.ordinal;
                tables.bind_address(context, ordinal, 0, fire.status.gpu_address())?;
                tables.bind_address(context, ordinal, 1, stage_descriptors.gpu_address())?;
                tables.bind_address(context, ordinal, 2, stage_parameters.gpu_address())?;
                tables.bind_address(context, ordinal, 3, stage_offsets.gpu_address())?;
                tables.bind_address(context, ordinal, 4, scratch_all.gpu_address())?;
                tables.bind_address(context, ordinal, 5, temporary.gpu_address())?;
                tables.bind_address(
                    context,
                    ordinal,
                    6,
                    fused_logits.as_ref().unwrap_or(&dummy).gpu_address(),
                )?;
                for (local, &dense) in stage.plan.channel_bindings.iter().enumerate() {
                    let dense = dense as usize;
                    let read = if fused_pending[dense] {
                        &fire.pending[dense]
                    } else {
                        &fire.committed[dense]
                    };
                    tables.bind_address(context, ordinal, 7 + local * 2, read.gpu_address())?;
                    tables.bind_address(
                        context,
                        ordinal,
                        8 + local * 2,
                        fire.pending[dense].gpu_address(),
                    )?;
                }
                for &node in &region.region.nodes {
                    if let Some(op) = stage.plan.ops.get(node as usize)
                        && op.code == CHAN_PUT
                        && let Some(&dense) = stage.plan.channel_bindings.get(op.channel as usize)
                    {
                        fused_pending[dense as usize] = true;
                    }
                }
            }
        }

        // A retried fire reuses its status buffer; the kernels only ever
        // move it forward, so it starts at zero every run.
        // SAFETY: no step is in flight between prepare/finish and here.
        unsafe { fire.status.zero(0, STATUS_BYTES as u64)? };

        let timing = stepper.run(|step| {
            step.set_pipeline(&program.readiness);
            step.set_argument_table_for(tables, program.readiness_ordinal)?;
            step.dispatch([1, 1, 1], [1, 1, 1])?;
            step.barrier(Visibility::Device);
            for stage in &program.stages {
                match (&stage.executable.fused, mode) {
                    (Ok(fused), Mode::Fused) => {
                        for region in fused {
                            step.set_pipeline(&region.pso);
                            step.set_argument_table_for(tables, region.ordinal)?;
                            step.dispatch([1, 1, 1], [1, 1, 1])?;
                            step.barrier(Visibility::Device);
                        }
                    }
                    _ => {
                        for region in &stage.executable.regions {
                            step.set_pipeline(&region.pso);
                            step.set_argument_table_for(tables, region.ordinal)?;
                            step.dispatch([1, 1, 1], [1, 1, 1])?;
                            step.barrier(Visibility::Device);
                        }
                    }
                }
            }
            step.set_pipeline(&program.commit);
            step.set_argument_table_for(tables, program.commit_ordinal)?;
            step.dispatch([1, 1, 1], [1, 1, 1])
        })?;

        let status = fire.status().ok_or_else(|| Error::Program {
            message: "Metal M1 status buffer shorter than a status".to_owned(),
        })?;
        let (outcome, _diagnosis) = StatusOutcome::of(status, true);
        let report = match outcome {
            StatusOutcome::Committed => None,
            _ => Some(report_status(status, true, fire.rings.len() as u32)),
        };
        Ok(Execution {
            outcome,
            report,
            timing,
        })
    }
}

/// The logits operand of an intrinsic op: the bound rows, checked.
///
/// The C++ writes this twice (M1 and M2) with three hand-rolled byte-range
/// checks; the row checks stay here and the byte check is
/// [`Handle::slice`]'s.
fn logits_operand(inputs: &DeviceInputs, intrinsic: u16, output: &ValueDesc) -> Result<Handle> {
    let Some(logits) = &inputs.logits else {
        return Err(Error::Program {
            message: "Metal M1 logits intrinsic is unbound".to_owned(),
        });
    };
    if inputs.logits_row_count == 0 || inputs.vocab == 0 {
        return Err(Error::Program {
            message: "Metal M1 logits intrinsic is unbound".to_owned(),
        });
    }
    let rows_needed = if intrinsic == MTP_DRAFTS {
        output.len
    } else {
        output.len.div_ceil(inputs.vocab)
    };
    let local_row = match inputs.mtp_draft_row {
        Some(row) if intrinsic == MTP_LOGITS || intrinsic == MTP_DRAFTS => row,
        _ => 0,
    };
    if local_row > inputs.logits_row_count || rows_needed > inputs.logits_row_count - local_row {
        return Err(Error::Program {
            message: "Metal M1 intrinsic row range exceeds bound logits".to_owned(),
        });
    }
    let byte_offset = u64::from(inputs.logits_row_offset) * u64::from(inputs.vocab) * 2;
    if byte_offset > logits.len() {
        return Err(Error::Program {
            message: "Metal M1 intrinsic exceeds logits buffer".to_owned(),
        });
    }
    let required = u64::from(local_row + rows_needed) * u64::from(inputs.vocab) * 2;
    if required > logits.len() - byte_offset {
        return Err(Error::Program {
            message: "Metal M1 intrinsic exceeds logits buffer".to_owned(),
        });
    }
    logits.slice(byte_offset, logits.len() - byte_offset)
}

/// The dense channel a stage-local slot binds, checked.
fn dense_channel(
    stage: &crate::gpu::program::executable::ProgramStage,
    local: u32,
    channels: usize,
) -> Result<usize> {
    let dense = stage
        .plan
        .channel_bindings
        .get(local as usize)
        .copied()
        .ok_or_else(|| Error::Program {
            message: "Metal M1 channel op binding is invalid".to_owned(),
        })?;
    if (dense as usize) >= channels {
        return Err(Error::Program {
            message: "Metal M1 channel binding is out of range".to_owned(),
        });
    }
    Ok(dense as usize)
}

/// A lane-table offset that is `Some` by construction, surfaced as an error
/// rather than trusted.
fn offset(value: Option<u64>, what: &str) -> Result<u64> {
    value.ok_or_else(|| Error::Program {
        message: format!("Metal M1 {what} offset left the lane table"),
    })
}

/// The bytes of a `#[repr(C)]`, padding-free record slice.
///
/// Crate-internal and used only with the lane-table records, `ValueDesc` and
/// `OpParams`, each of which is declared as-uploaded ABI with its size
/// pinned by a test.
pub(super) fn pod_bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: `T` is one of the crate's `#[repr(C)]` ABI records, fully
    // initialised, with no padding bytes; the slice covers exactly the
    // values' own storage.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

/// Write one POD record at `offset` of `region`.
pub(super) fn write_pod<T: Copy>(region: &Transient, at: u64, value: &T) -> Result<()> {
    // SAFETY: the region was just acquired or belongs to a fire no step is
    // running against; `pod_bytes` covers the record exactly.
    unsafe { region.write(at, pod_bytes(std::slice::from_ref(value))) }
}
