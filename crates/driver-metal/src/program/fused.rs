//! The M2 path: a fire's fused regions placed around someone else's forward.
//!
//! The M1 path runs a fire as its own step. The M2 path splits the same fire
//! in two — the prologue regions ride *before* the model forward and the
//! epilogue regions *after* it, inside the forward executor's own command
//! buffer — so the sampling logic runs in the gap the forward would
//! otherwise leave. That is why every function here is handed the *target*:
//! the context, tables and pool the command runs against belong to the
//! forward executor, not to this runtime.
//!
//! ## The lesson M3 learned and M2 did not
//!
//! A command is prepared *before* the forward it rides is encoded, and the
//! forward can still refuse the batch — in which case nothing dispatches and
//! the status buffer keeps its zero fill. The C++ M3 path guards its finish
//! on an `encoded` flag; its M2 path does not, so a refused forward read
//! back as `"Metal M2 fused execution fault 0"` — a GPU fault report for
//! something the GPU was never asked to do. [`M2Command`] carries the flag
//! and [`finish`](M2Command::finish) answers `NeverDispatched` through the
//! same [`Outcome::of`](crate::channel::StatusOutcome) every other path
//! uses.
//!
//! ## What else does not survive
//!
//! * The C++ binds the effect tables twice — once at prepare (lines
//!   2208–2235) and again at every encode (`bind_m2_effect`). The encode is
//!   when the tables must be current, so the prepare-time copy is dropped.
//! * `finish_m2_command` recycles four transients, releases every external
//!   registration and forgets every ordinal by hand, in a specific order.
//!   Here the transients and the [`External`] guards are owned fields —
//!   dropping the command is the release — and only the ordinal forget
//!   remains explicit, because the tables belong to the target.

use std::rc::Rc;

use tensor_ir::op::tags;

use crate::channel::{
    OpParams, OpRuntime, STATUS_BYTES, StatusOutcome, ValueDesc, describe, layout, report_status,
};
use crate::device::allocator::{Pool, Transient};
use crate::device::argtable::Tables;
use crate::device::context::Context;
use crate::device::encoder::{StepEncoder, Visibility};
use crate::device::external::{External, Externals};
use crate::device::handle::Handle;
use crate::layout::region::Region as _;
use crate::program::cache::Runtime;
use crate::program::executable::Pso;
use crate::program::single::{DeviceInputs, PreparedFire};
use crate::{Error, Result};

/// `PTIR_OP_CHAN_PUT`.
const CHAN_PUT: u16 = tags::CHAN_PUT as u16;
/// `PTIR_OP_INTRINSIC_VAL`.
const INTRINSIC_VAL: u16 = tags::INTRINSIC_VAL as u16;
/// `PTIR_INTR_MTP_DRAFTS`.
const MTP_DRAFTS: u16 = tensor_ir::op::intrinsic_tags::MTP_DRAFTS;
/// `PTIR_INTR_MTP_LOGITS`.
const MTP_LOGITS: u16 = tensor_ir::op::intrinsic_tags::MTP_LOGITS;

/// One fused region, bound and ready to encode.
///
/// `M2EncodedRegion`. The seven fixed handles are the fused kernel ABI:
/// status, stage descriptors, stage parameters, stage offsets, scratch,
/// temporary, logits. The channel pairs follow at `7 + 2i`.
struct EncodedRegion {
    pso: Pso,
    ordinal: u32,
    fixed: [Handle; 7],
    channels: Vec<Handle>,
}

/// A fire's regions placed around a forward, with everything they bind.
///
/// `M2CommandPlan`, minus the raw `RawMetalContext* target` — the target's
/// context and tables are arguments to the calls that need them, so a
/// command cannot be encoded against one context and finished against
/// another by holding a stale pointer.
pub struct M2Command {
    fire: Rc<PreparedFire>,
    /// The four transients are held, not read: their handles are already
    /// sliced into the regions' bindings, and holding the owners beside the
    /// views is the crate's discipline for pooled buffers.
    #[allow(dead_code)]
    scratch: Transient,
    #[allow(dead_code)]
    descriptors: Transient,
    #[allow(dead_code)]
    parameters: Transient,
    #[allow(dead_code)]
    offsets: Transient,
    /// Keeps the fire's buffers resident on the *target* context for the
    /// command's life.
    #[allow(dead_code)]
    resident: Vec<External>,
    pre: Vec<EncodedRegion>,
    post: Vec<EncodedRegion>,
    readiness_ordinal: u32,
    commit_ordinal: u32,
    logits_base: Option<Handle>,
    logits_vocab: u32,
    /// Whether any encode ran. See the module docs.
    encoded: bool,
}

impl Runtime {
    /// Build the placed command for `fire` against the forward executor's
    /// `target` context.
    ///
    /// `pool` and `externals` are the target's; the scratch this acquires
    /// and the residency it requests belong to the context the command will
    /// actually run on.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] when the fire cannot be placed — a stage without a
    /// fused executable, a per-layer stage, an unresolvable shape, an
    /// intrinsic without its logits. Allocation failures surface as their
    /// own errors.
    #[allow(clippy::too_many_lines)]
    pub fn prepare_m2(
        &mut self,
        target: &Context,
        pool: &Pool,
        externals: &Externals,
        fire: &Rc<PreparedFire>,
        inputs: &DeviceInputs,
    ) -> Result<M2Command> {
        let program = fire.program();
        for stage in &program.stages {
            if let Err(reason) = &stage.executable.fused {
                return Err(Error::Program {
                    message: reason.clone(),
                });
            }
            if !matches!(
                tensor_ir::registry::Stage::from_u8(stage.kind),
                Some(tensor_ir::registry::Stage::Prologue | tensor_ir::registry::Stage::Epilogue)
            ) {
                return Err(Error::Program {
                    message: "Metal M2 cannot place per-layer stages".to_owned(),
                });
            }
        }

        // Shapes, layout, and the per-op validations the fire's numbers make
        // possible.
        let mut descriptors: Vec<ValueDesc> = Vec::new();
        let mut stage_value_bases = Vec::with_capacity(program.stages.len());
        let mut stage_parameter_bases = Vec::with_capacity(program.stages.len());
        let mut parameter_count = 0usize;
        for stage in &program.stages {
            stage_value_bases.push(descriptors.len());
            stage_parameter_bases.push(parameter_count);
            parameter_count += stage.plan.ops.len();
            for value in &stage.plan.value_types {
                descriptors.push(describe(value, &inputs.extents).map_err(|why| {
                    Error::Program {
                        message: format!("Metal M2 value shape did not resolve: {why:?}"),
                    }
                })?);
            }
        }
        for (stage_index, stage) in program.stages.iter().enumerate() {
            let mut result_base = 0u32;
            for op in stage.plan.ops.iter() {
                if op.code == INTRINSIC_VAL {
                    check_logits(
                        inputs,
                        op.intrinsic,
                        &descriptors[stage_value_bases[stage_index] + result_base as usize],
                    )?;
                } else if op.code == CHAN_PUT {
                    let dense = stage
                        .plan
                        .channel_bindings
                        .get(op.channel as usize)
                        .copied()
                        .ok_or_else(|| Error::Program {
                            message: "Metal M2 channel sink binding is invalid".to_owned(),
                        })? as usize;
                    let arg = op.args.first().copied().unwrap_or_default() as usize;
                    let wire = descriptors[stage_value_bases[stage_index] + arg].wire_bytes();
                    if wire > fire.pending[dense].len() {
                        return Err(Error::Program {
                            message: "Metal M2 channel sink exceeds fixed cell size".to_owned(),
                        });
                    }
                }
                result_base += u32::from(op.result_count);
            }
        }
        let scratch_layout = layout(&descriptors).map_err(|why| Error::Program {
            message: format!("Metal M2 fire scratch: {why:?}"),
        })?;

        const DESC_BYTES: u64 = size_of::<ValueDesc>() as u64;
        const PARAM_BYTES: u64 = size_of::<OpParams>() as u64;
        let scratch = pool.acquire(target, scratch_layout.total)?;
        let descriptor_buffer =
            pool.acquire(target, (descriptors.len() as u64).max(1) * DESC_BYTES)?;
        let parameter_buffer =
            pool.acquire(target, (parameter_count as u64).max(1) * PARAM_BYTES)?;
        let offset_buffer = pool.acquire(target, (descriptors.len() as u64).max(1) * 4)?;

        // SAFETY: `ValueDesc` and `OpParams` are `#[repr(C)]` upload ABI;
        // the buffers were just acquired and no step names them.
        unsafe {
            descriptor_buffer.write(0, pod_bytes(&descriptors))?;
            let offsets: Vec<u32> = scratch_layout.values.iter().map(|&v| v as u32).collect();
            offset_buffer.write(0, pod_bytes(&offsets))?;
        }
        // The parameter fill: one record per op of every stage, by the same
        // `OpParams::of` the M1 path uses — the C++ filled this record in a
        // second 600-line loop that agreed with the first by inspection.
        let runtime = OpRuntime {
            vocab: inputs.vocab,
            mtp_draft_row: inputs.mtp_draft_row,
        };
        let parameters_all = Handle::over(parameter_buffer.buffer(), parameter_buffer.len())?;
        for (stage_index, stage) in program.stages.iter().enumerate() {
            let mut result_base = 0u32;
            for (node, op) in stage.plan.ops.iter().enumerate() {
                let mut params = OpParams::of(op, result_base, runtime);
                if op.code == CHAN_PUT
                    && let Some(&dense) = stage.plan.channel_bindings.get(op.channel as usize)
                {
                    params.sink_bytes = fire.pending[dense as usize].len() as u32;
                }
                let at = (stage_parameter_bases[stage_index] + node) as u64 * PARAM_BYTES;
                let slot = parameters_all.slice(at, PARAM_BYTES)?;
                // SAFETY: as above; the record is the kernels' ABI.
                unsafe { slot.write(0, pod_bytes(std::slice::from_ref(&params)))? };
                result_base += u32::from(op.result_count);
            }
        }

        // The fire's buffers were allocated on the runtime's context; the
        // target must keep them resident for as long as it may dispatch over
        // them.
        let mut resident = Vec::new();
        resident.push(externals.insert(target, fire.status.buffer()));
        resident.push(externals.insert(target, fire.lane_table.buffer()));
        for ring in &fire.rings {
            resident.push(externals.insert(target, ring.cells().buffer()));
            resident.push(externals.insert(target, ring.words().buffer()));
        }
        if let Some(logits) = &inputs.logits {
            resident.push(externals.insert(target, logits.buffer()));
        }

        let scratch_all = Handle::over(scratch.buffer(), scratch.len())?;
        let temporary =
            scratch_all.slice(scratch_layout.temporary, scratch_layout.temporary_bytes)?;
        let descriptors_all = Handle::over(descriptor_buffer.buffer(), descriptor_buffer.len())?;
        let offsets_all = Handle::over(offset_buffer.buffer(), offset_buffer.len())?;

        let mut pre = Vec::new();
        let mut post = Vec::new();
        let mut pending = vec![false; fire.rings.len()];
        for (stage_index, stage) in program.stages.iter().enumerate() {
            let value_base = stage_value_bases[stage_index] as u64;
            let stage_descriptors = descriptors_all.slice(
                value_base * DESC_BYTES,
                (stage.plan.value_types.len() as u64).max(1) * DESC_BYTES,
            )?;
            let stage_parameters = parameters_all.slice(
                stage_parameter_bases[stage_index] as u64 * PARAM_BYTES,
                (stage.plan.ops.len() as u64).max(1) * PARAM_BYTES,
            )?;
            let stage_offsets = offsets_all.slice(
                value_base * 4,
                (stage.plan.value_types.len() as u64).max(1) * 4,
            )?;
            let logits =
                logits_at(inputs, inputs.logits_row_offset)?.unwrap_or_else(|| scratch_all.clone());
            // Every stage was checked fused at entry; a defensive `continue`
            // rather than a panic path.
            let Ok(fused) = &stage.executable.fused else {
                continue;
            };
            for region in fused {
                let mut channels = Vec::with_capacity(stage.plan.channel_bindings.len() * 2);
                for &dense in &stage.plan.channel_bindings {
                    let dense = dense as usize;
                    channels.push(if pending[dense] {
                        fire.pending[dense].clone()
                    } else {
                        fire.committed[dense].clone()
                    });
                    channels.push(fire.pending[dense].clone());
                }
                for &node in &region.region.nodes {
                    if let Some(op) = stage.plan.ops.get(node as usize)
                        && op.code == CHAN_PUT
                        && let Some(&dense) = stage.plan.channel_bindings.get(op.channel as usize)
                    {
                        pending[dense as usize] = true;
                    }
                }
                let encoded = EncodedRegion {
                    pso: region.pso.clone(),
                    ordinal: self.next_ordinal(),
                    fixed: [
                        fire_status(fire)?,
                        stage_descriptors.clone(),
                        stage_parameters.clone(),
                        stage_offsets.clone(),
                        scratch_all.clone(),
                        temporary.clone(),
                        logits.clone(),
                    ],
                    channels,
                };
                if tensor_ir::registry::Stage::from_u8(stage.kind)
                    == Some(tensor_ir::registry::Stage::Prologue)
                {
                    pre.push(encoded);
                } else {
                    post.push(encoded);
                }
            }
        }

        // A reused fire starts every placement from a clean status.
        // SAFETY: no step is in flight against this fire.
        unsafe { fire.status.zero(0, STATUS_BYTES as u64)? };

        Ok(M2Command {
            fire: Rc::clone(fire),
            scratch,
            descriptors: descriptor_buffer,
            parameters: parameter_buffer,
            offsets: offset_buffer,
            resident,
            pre,
            post,
            readiness_ordinal: self.next_ordinal(),
            commit_ordinal: self.next_ordinal(),
            logits_base: inputs.logits.clone(),
            logits_vocab: inputs.vocab,
            encoded: false,
        })
    }
}

impl M2Command {
    /// Point every region's logits binding at `row`, and tell the lane
    /// record.
    ///
    /// The forward decides late which logits row a member owns; this is the
    /// rebind that follows it. A row past the buffer leaves the bindings
    /// unchanged, as in the C++.
    pub fn set_logits_row(&mut self, row: u32) -> Result<()> {
        let (Some(base), true) = (&self.logits_base, self.logits_vocab != 0) else {
            return Ok(());
        };
        let offset = u64::from(row) * u64::from(self.logits_vocab) * 2;
        if offset >= base.len() {
            return Ok(());
        }
        let logits = base.slice(offset, base.len() - offset)?;
        for region in self.pre.iter_mut().chain(self.post.iter_mut()) {
            region.fixed[6] = logits.clone();
        }
        let shape = crate::channel::LaneShape::of(1, self.fire.rings.len() as u32);
        let record = shape.record_offset(0).ok_or_else(|| Error::Program {
            message: "Metal M2 lane record offset left the table".to_owned(),
        })?;
        // `logits_row_offset` is the third word of the record, after the
        // 8-byte base.
        // SAFETY: the lane table is the fire's own buffer, longer than the
        // record, and no step is in flight while the row is being chosen.
        unsafe { self.fire.lane_table.write(record + 8, &row.to_le_bytes()) }
    }

    /// Encode the readiness gate and the prologue regions, before the
    /// forward.
    ///
    /// # Errors
    ///
    /// Binding or dispatch refusals from the target's tables and encoder.
    pub fn encode_pre(
        &mut self,
        target: &Context,
        tables: &mut Tables,
        step: &mut StepEncoder<'_>,
    ) -> Result<()> {
        self.encoded = true;
        self.bind_effect(target, tables, self.readiness_ordinal)?;
        step.set_pipeline(&self.fire.program().readiness);
        step.set_argument_table_for(tables, self.readiness_ordinal)?;
        step.dispatch([1, 1, 1], [1, 1, 1])?;
        step.barrier(Visibility::Device);
        for index in 0..self.pre.len() {
            self.encode_region(target, tables, step, true, index)?;
        }
        Ok(())
    }

    /// Encode the epilogue regions and the commit, after the forward.
    ///
    /// # Errors
    ///
    /// As [`encode_pre`](Self::encode_pre).
    pub fn encode_post(
        &mut self,
        target: &Context,
        tables: &mut Tables,
        step: &mut StepEncoder<'_>,
    ) -> Result<()> {
        self.encoded = true;
        for index in 0..self.post.len() {
            self.encode_region(target, tables, step, false, index)?;
        }
        self.bind_effect(target, tables, self.commit_ordinal)?;
        step.set_pipeline(&self.fire.program().commit);
        step.set_argument_table_for(tables, self.commit_ordinal)?;
        step.dispatch([1, 1, 1], [1, 1, 1])
    }

    /// Read the fire's verdict and give the target its tables back.
    ///
    /// Consumes the command: the transients recycle and the residency
    /// registrations release as it drops. `dispatched` is tracked, not
    /// assumed — see the module docs for the zero-fill lie this prevents.
    #[must_use]
    pub fn finish(self, tables: &mut Tables) -> (StatusOutcome, Option<String>) {
        let status = self.fire.status().unwrap_or_default();
        for region in self.pre.iter().chain(self.post.iter()) {
            tables.forget(region.ordinal);
        }
        tables.forget(self.readiness_ordinal);
        tables.forget(self.commit_ordinal);
        let (outcome, _) = StatusOutcome::of(status, self.encoded);
        let report = match outcome {
            StatusOutcome::Committed => None,
            _ => Some(report_status(
                status,
                self.encoded,
                self.fire.rings.len() as u32,
            )),
        };
        (outcome, report)
    }

    /// The single-lane effect binding: status, lane table, every ring's
    /// words. `bind_m2_effect`.
    fn bind_effect(&self, target: &Context, tables: &mut Tables, ordinal: u32) -> Result<()> {
        tables.bind_address(target, ordinal, 0, self.fire.status.gpu_address())?;
        tables.bind_address(target, ordinal, 1, self.fire.lane_table.gpu_address())?;
        for (channel, ring) in self.fire.rings.iter().enumerate() {
            tables.bind_address(target, ordinal, channel + 2, ring.words().gpu_address())?;
        }
        Ok(())
    }

    /// Bind and dispatch one region. `bind_m2_region` plus the dispatch that
    /// always followed it.
    fn encode_region(
        &self,
        target: &Context,
        tables: &mut Tables,
        step: &mut StepEncoder<'_>,
        pre: bool,
        index: usize,
    ) -> Result<()> {
        let region = if pre {
            &self.pre[index]
        } else {
            &self.post[index]
        };
        for (slot, handle) in region.fixed.iter().enumerate() {
            tables.bind_address(target, region.ordinal, slot, handle.gpu_address())?;
        }
        for (slot, handle) in region.channels.iter().enumerate() {
            tables.bind_address(target, region.ordinal, 7 + slot, handle.gpu_address())?;
        }
        step.set_pipeline(&region.pso);
        step.set_argument_table_for(tables, region.ordinal)?;
        step.dispatch([1, 1, 1], [1, 1, 1])?;
        step.barrier(Visibility::Device);
        Ok(())
    }
}

impl std::fmt::Debug for M2Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("M2Command")
            .field("pre", &self.pre.len())
            .field("post", &self.post.len())
            .field("encoded", &self.encoded)
            .finish_non_exhaustive()
    }
}

/// The whole-buffer view of the fire's status, for a fixed binding.
fn fire_status(fire: &PreparedFire) -> Result<Handle> {
    Handle::over(fire.status.buffer(), fire.status.len())
}

/// The logits view starting at `row`, or `None` when no logits are bound.
fn logits_at(inputs: &DeviceInputs, row: u32) -> Result<Option<Handle>> {
    let (Some(base), true) = (&inputs.logits, inputs.vocab != 0) else {
        return Ok(None);
    };
    let offset = u64::from(row) * u64::from(inputs.vocab) * 2;
    if offset >= base.len() {
        return Ok(None);
    }
    Ok(Some(base.slice(offset, base.len() - offset)?))
}

/// The M2 half of the intrinsic bounds check the M1 path also runs.
fn check_logits(inputs: &DeviceInputs, intrinsic: u16, output: &ValueDesc) -> Result<()> {
    let Some(logits) = &inputs.logits else {
        return Err(Error::Program {
            message: "Metal M2 logits intrinsic is unbound".to_owned(),
        });
    };
    if inputs.logits_row_count == 0 || inputs.vocab == 0 {
        return Err(Error::Program {
            message: "Metal M2 logits intrinsic is unbound".to_owned(),
        });
    }
    let rows_needed = if intrinsic == MTP_DRAFTS {
        output.len
    } else {
        output.len.div_ceil(inputs.vocab)
    };
    let row_offset = match inputs.mtp_draft_row {
        Some(row) if intrinsic == MTP_LOGITS || intrinsic == MTP_DRAFTS => row,
        _ => 0,
    };
    if row_offset > inputs.logits_row_count || rows_needed > inputs.logits_row_count - row_offset {
        return Err(Error::Program {
            message: "Metal M2 intrinsic row range exceeds bound logits".to_owned(),
        });
    }
    let byte_offset = u64::from(inputs.logits_row_offset) * u64::from(inputs.vocab) * 2;
    let required = u64::from(row_offset + rows_needed) * u64::from(inputs.vocab) * 2;
    if byte_offset > logits.len() || required > logits.len() - byte_offset {
        return Err(Error::Program {
            message: "Metal M2 intrinsic exceeds logits buffer".to_owned(),
        });
    }
    Ok(())
}

/// The bytes of `#[repr(C)]`, padding-free upload records.
fn pod_bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: used only with the crate's `#[repr(C)]` ABI records, fully
    // initialised, no padding; the slice covers exactly their storage.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}
