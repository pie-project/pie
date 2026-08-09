//! The M3 path: up to 64 fires dispatched as one group of lanes.
//!
//! Where M2 places one fire around a forward, M3 rides *many*: every lane is
//! a prepared fire, the group shares one lane table, one status array and
//! one set of per-`(lane, channel)` metadata, and each stage's kernels are
//! dispatched once across every lane that agrees on the stage's canonical
//! identity and size bucket ([`GroupKey`]). The grouped effect kernels read
//! their decisions out of the per-channel flag words
//! ([`ChannelMeta::flags`]) instead of having them compiled in, which is
//! what lets one readiness/commit pair serve every program.
//!
//! ## What the C++ got wrong here, and this keeps right
//!
//! The `encoded` guard is the one part of this file the C++ had *learned* —
//! its own comment records why: a group is prepared before the forward it
//! rides is encoded, the forward can refuse the batch, and reading the
//! status buffer's zero fill back lane by lane produces a GPU fault report
//! for work the GPU was never asked to do, hiding the executor's own
//! account. The guard survives the port; what changes is that the M2 path
//! now has it too.
//!
//! ## What does not survive
//!
//! * **The 220-line `release_group` lambda.** Six group transients, up to
//!   seven more per stage, every external registration and the timestamp
//!   heap, recycled by hand on each of fourteen failure exits. Ownership is
//!   the label: every early `?` drops the group and everything it holds.
//! * **The GPU timestamp heap.** `m3_gpu_timestamps_enabled()` returns
//!   `false` at compile time — the C++'s own comment prices it at 5.0ms of a
//!   ~13ms token for one diagnostic stat — and the host-clock fallback
//!   reports the same span for free. The dead branch is not ported; the
//!   fallback is [`GroupStats::post_forward_critical_ns`].
//! * **`kM3RegionThreads` transcription.** The width the region kernels
//!   reduce over is the compiler's `METAL_M3_REGION_THREADS`, whose own doc
//!   says a hand-kept copy with a "must equal" comment has nothing comparing
//!   the two. The mirror here is compared: a dev-dependency test holds it
//!   against the compiler's constant.

use std::collections::{BTreeMap, HashSet};
use std::rc::Rc;
use std::time::Instant;

use objc2_metal::MTLComputePipelineState;
use tensor_ir::op::tags;

use super::context::Context;
use super::encoder::{StepEncoder, Visibility};
use super::external::{External, Externals};
use super::fire::{DeviceInputs, PreparedFire, pod_bytes, write_pod};
use super::pool::{Pool, Transient};
use super::program::{GroupedExecutable, Pso};
use super::runtime::Runtime;
use super::tables::Tables;
use crate::pipeline::{
    ChannelMeta, GroupKey, GroupLayout, LANE_ABI_VERSION, LANE_FLAG_RAGGED, LaneChannelSlot,
    LaneHeader, LaneRecord, LaneShape, MAX_SCRATCH_BYTES, OpParams, OpRuntime, Readiness, RowMeta,
    SCRATCH_ALIGN, STATUS_BYTES, Status, StatusOutcome, ValueDesc, Words, channel_flags,
    check_words, describe, report_status, used_channel_slots,
};
use crate::region::Region as _;
use crate::{Error, Result};

/// `PTIR_OP_CHAN_PUT`.
const CHAN_PUT: u16 = tags::CHAN_PUT as u16;

/// The most lanes one group may carry: the width of
/// [`LaneRecord::active_row_mask`]'s row bitset per lane, and the readiness
/// dispatch's grid.
pub const MAX_LANES: usize = 64;

/// The threadgroup width a generated grouped region is dispatched at,
/// bounded below by a simd and above by the pipeline's own maximum.
///
/// Mirror of `tensor_compiler::codegen::metal::fused::METAL_M3_REGION_THREADS`
/// — the emitted kernels size their threadgroup memory against it and fault
/// `0xB3` on a wider launch — drift-checked by a dev-dependency test.
pub const REGION_THREADS: u32 = 512;

/// One lane of a group: a prepared fire and its runtime numbers.
///
/// `M3LaneCandidate`.
pub struct LaneCandidate {
    /// The fire this lane runs.
    pub fire: Rc<PreparedFire>,
    /// The lane's forward outputs and extents.
    pub inputs: DeviceInputs,
    /// The lane must not be re-fired on a readiness retry — recorded into
    /// [`ChannelMeta::flags`] for the kernels.
    pub retry_ineligible: bool,
}

/// What one group's encodes did.
///
/// `M3GroupStats`, minus nothing: every counter had a reader in the C++'s
/// diagnostics and keeps it here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupStats {
    /// Readiness dispatches encoded.
    pub readiness_launches: u64,
    /// Region dispatches encoded.
    pub body_launches: u64,
    /// Of those, library regions.
    pub library_launches: u64,
    /// Of those, parallel-selection (nucleus/top-k) regions.
    pub parallel_selection_launches: u64,
    /// Of those, regions run on the grouped-singleton fallback.
    pub singleton_fallback_launches: u64,
    /// Commit dispatches encoded.
    pub commit_launches: u64,
    /// Lanes in the group.
    pub lanes: u64,
    /// Host-observed span of the post-forward critical section, from the
    /// first `encode_post` to `finish`.
    pub post_forward_critical_ns: u64,
}

/// One grouped region, bound by address and sized for its dispatch.
///
/// `M3EncodedRegion`. The eleven fixed addresses are the grouped kernel ABI:
/// lane table, descriptors, parameters, offsets, scratch, layout, bindings,
/// pending flags, lane indices, row meta, row indices.
struct EncodedRegion {
    pso: Pso,
    ordinal: u32,
    fixed: [u64; 11],
    grid: [usize; 3],
    threadgroup: [usize; 3],
    library: bool,
    parallel_selection: bool,
}

/// One stage group's buffers and dispatches.
///
/// `M3StageCommand`.
struct StageCommand {
    /// `tensor_ir::registry::Stage` wire byte: prologue rides pre, epilogue
    /// post.
    kind: u8,
    /// The stage ran on the grouped-singleton fallback rather than the
    /// grouped-fused path.
    singleton_fallback: bool,
    #[allow(dead_code)]
    descriptors: Transient,
    #[allow(dead_code)]
    parameters: Transient,
    #[allow(dead_code)]
    offsets: Transient,
    #[allow(dead_code)]
    scratch: Transient,
    #[allow(dead_code)]
    layout: Transient,
    #[allow(dead_code)]
    bindings: Transient,
    #[allow(dead_code)]
    lane_indices: Transient,
    regions: Vec<EncodedRegion>,
}

/// A group of fires, prepared against the forward executor's context.
///
/// `M3GroupCommand`, minus the `target` pointer (an argument, as for M2) and
/// the `timestamp_heap` (see the module docs).
pub struct M3Group {
    candidates: Vec<LaneCandidate>,
    lane_table: Transient,
    statuses: Transient,
    channel_meta: Transient,
    /// Held, not read after prepare: their addresses are in every region's
    /// fixed bindings, and the owners live beside the views.
    #[allow(dead_code)]
    pending_flags: Transient,
    #[allow(dead_code)]
    row_meta: Transient,
    #[allow(dead_code)]
    row_indices: Transient,
    #[allow(dead_code)]
    resident: Vec<External>,
    stages: Vec<StageCommand>,
    readiness: Pso,
    commit: Pso,
    readiness_ordinal: u32,
    commit_ordinal: u32,
    stats: GroupStats,
    encoded: bool,
    post_begin: Option<Instant>,
}

impl Runtime {
    /// Group up to [`MAX_LANES`] prepared fires into one command against the
    /// forward executor's `target`.
    ///
    /// Every candidate is re-checked against the host readiness rules — a
    /// group is composed ahead of time, and a lane that went stale must
    /// abort the group rather than ride it into a device-side fault.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] for a group that cannot be built: no lanes or too
    /// many, a candidate that is not ready, aliased channels between lanes,
    /// a stage without a grouped executable, schema disagreements between
    /// lanes sharing a group key. Allocation failures surface as their own
    /// errors.
    #[allow(clippy::too_many_lines)]
    pub fn prepare_m3(
        &mut self,
        target: &Context,
        pool: &Pool,
        externals: &Externals,
        candidates: Vec<LaneCandidate>,
    ) -> Result<M3Group> {
        if candidates.is_empty() || candidates.len() > MAX_LANES {
            return Err(program_error(format!(
                "Metal M3 group lane count must be in [1,{MAX_LANES}]"
            )));
        }
        let Some((readiness, commit)) = self.grouped_effects() else {
            return Err(program_error(
                "Metal M3 grouped executable is unavailable".to_owned(),
            ));
        };

        let mut aliases = HashSet::new();
        let mut channel_stride = 0usize;
        let mut total_rows = 0u64;
        for candidate in &candidates {
            let fire = &candidate.fire;
            let program = fire.program();
            if program.stages.is_empty() {
                return Err(program_error(
                    "Metal M3 cannot group an empty program".to_owned(),
                ));
            }
            let words: Vec<Words> = fire.rings.iter().map(|ring| ring.snapshot()).collect();
            match check_words(&words, &program.effects, &fire.tickets) {
                Readiness::Ready => {}
                Readiness::Retry { .. } => {
                    return Err(program_error(
                        "Metal M3 group aborted by definitive host readiness".to_owned(),
                    ));
                }
                other => {
                    return Err(program_error(format!(
                        "Metal M3 group failed definitive host readiness: {other:?}"
                    )));
                }
            }
            channel_stride = channel_stride.max(program.effects.len());
            let inputs = &candidate.inputs;
            if !inputs.logits_rows.is_empty()
                && inputs.logits_rows.len() != inputs.logits_row_count as usize
            {
                return Err(program_error(
                    "Metal M3 explicit logits row map size mismatch".to_owned(),
                ));
            }
            if let Some(row) = inputs.mtp_draft_row
                && row > inputs.logits_row_count
            {
                return Err(program_error(
                    "Metal M3 MTP row base exceeds logits row map".to_owned(),
                ));
            }
            total_rows += u64::from(inputs.logits_row_count);
            if total_rows > u64::from(u32::MAX) {
                return Err(program_error(
                    "Metal M3 logits row map exceeds u32".to_owned(),
                ));
            }
            for ring in &fire.rings {
                if !aliases.insert(Rc::as_ptr(ring)) {
                    return Err(program_error(
                        "Metal M3 shared-channel alias requires ordered solo execution".to_owned(),
                    ));
                }
            }
        }

        // The group's shared buffers.
        let lanes = candidates.len();
        let shape = LaneShape::of(lanes as u32, channel_stride as u32);
        let table_bytes = shape
            .bytes()
            .ok_or_else(|| program_error("Metal M3 lane table exceeds u64".to_owned()))?;
        let lane_table = pool.acquire(target, table_bytes)?;
        let statuses = pool.acquire(target, (lanes * STATUS_BYTES) as u64)?;
        let channel_meta = pool.acquire(
            target,
            ((lanes * channel_stride * size_of::<ChannelMeta>()) as u64).max(1),
        )?;
        let pending_flags = pool.acquire(target, ((lanes * channel_stride) as u64).max(1))?;
        let row_meta = pool.acquire(target, (lanes * size_of::<RowMeta>()) as u64)?;
        let row_indices = pool.acquire(target, (total_rows * 4).max(1))?;
        // SAFETY: all six were just acquired; no step names them.
        unsafe {
            lane_table.zero(0, lane_table.len())?;
            statuses.zero(0, statuses.len())?;
            channel_meta.zero(0, channel_meta.len())?;
            pending_flags.zero(0, pending_flags.len())?;
            row_meta.zero(0, row_meta.len())?;
            row_indices.zero(0, row_indices.len())?;
        }

        let mut resident = Vec::new();
        let mut ragged = false;
        let first_rows = candidates[0].inputs.logits_row_count;
        let mut row_cursor = 0u32;
        for (lane, candidate) in candidates.iter().enumerate() {
            let fire = &candidate.fire;
            let inputs = &candidate.inputs;
            ragged |= inputs.logits_row_count != first_rows;
            let mask = match inputs.logits_row_count {
                0 => 0,
                64.. => u64::MAX,
                rows => (1u64 << rows) - 1,
            };
            let extents = &inputs.extents;
            write_pod(
                &lane_table,
                lane_offset(shape.record_offset(lane as u32))?,
                &LaneRecord {
                    logits_base: inputs
                        .logits
                        .as_ref()
                        .map_or(0, super::handle::Handle::gpu_address),
                    logits_row_offset: inputs.logits_row_offset,
                    logits_row_count: inputs.logits_row_count,
                    kv_len: extents.kv_len,
                    page_count: extents.page_count,
                    row_count: extents.row_count,
                    token_count: extents.token_count,
                    sampled_rows: extents.sampled_rows,
                    query_len: extents.query_len,
                    key_len: extents.key_len,
                    channel_slot_offset: shape.slot_index(lane as u32).unwrap_or(0),
                    commit_slot: statuses.gpu_address() + (lane * STATUS_BYTES) as u64,
                    active_row_mask: mask,
                    ..LaneRecord::default()
                },
            )?;
            write_pod(
                &row_meta,
                (lane * size_of::<RowMeta>()) as u64,
                &RowMeta {
                    offset: row_cursor,
                    count: inputs.logits_row_count,
                    mtp_offset: inputs.mtp_draft_row.unwrap_or(0),
                    reserved: 0,
                },
            )?;
            for row in 0..inputs.logits_row_count {
                let index = inputs
                    .logits_rows
                    .get(row as usize)
                    .copied()
                    .unwrap_or(inputs.logits_row_offset + row);
                // SAFETY: `row_cursor` stays under the total the buffer was
                // sized for; no step names the buffer yet.
                unsafe { row_indices.write(u64::from(row_cursor) * 4, &index.to_le_bytes())? };
                row_cursor += 1;
            }
            for (channel, ring) in fire.rings.iter().enumerate() {
                let ticket = fire.tickets[channel];
                write_pod(
                    &lane_table,
                    lane_offset(shape.slot_offset(lane as u32, channel as u32))?,
                    &LaneChannelSlot {
                        committed_cell: fire.committed[channel].gpu_address(),
                        pending_cell: fire.pending[channel].gpu_address(),
                        expected_head: ticket.expected_head,
                        expected_tail: ticket.expected_tail,
                    },
                )?;
                write_pod(
                    &channel_meta,
                    ((lane * channel_stride + channel) * size_of::<ChannelMeta>()) as u64,
                    &ChannelMeta {
                        words: ring.words().gpu_address(),
                        capacity: ring.capacity() as u32,
                        flags: channel_flags(
                            &fire.program().effects[channel],
                            candidate.retry_ineligible,
                        ),
                    },
                )?;
                resident.push(externals.insert(target, ring.cells().buffer()));
                resident.push(externals.insert(target, ring.words().buffer()));
            }
            if let Some(logits) = &inputs.logits {
                resident.push(externals.insert(target, logits.buffer()));
            }
        }
        write_pod(
            &lane_table,
            0,
            &LaneHeader {
                abi_version: LANE_ABI_VERSION,
                lane_count: lanes as u32,
                channel_slots_per_lane: channel_stride as u32,
                flags: if ragged { LANE_FLAG_RAGGED } else { 0 },
            },
        )?;

        // Group the stages by canonical identity and size bucket, prologues
        // apart from epilogues.
        let mut pre: BTreeMap<GroupKey, Vec<(usize, usize)>> = BTreeMap::new();
        let mut post: BTreeMap<GroupKey, Vec<(usize, usize)>> = BTreeMap::new();
        for (lane, candidate) in candidates.iter().enumerate() {
            let mut pre_count = 0;
            let mut post_count = 0;
            for (index, stage) in candidate.fire.program().stages.iter().enumerate() {
                let groups = match tensor_ir::registry::Stage::from_u8(stage.kind) {
                    Some(tensor_ir::registry::Stage::Prologue) => {
                        pre_count += 1;
                        &mut pre
                    }
                    Some(tensor_ir::registry::Stage::Epilogue) => {
                        post_count += 1;
                        &mut post
                    }
                    _ => {
                        return Err(program_error(
                            "Metal M3 cannot place a per-layer stage".to_owned(),
                        ));
                    }
                };
                if pre_count > 1 || post_count > 1 {
                    return Err(program_error(
                        "Metal M3 requires at most one stage per pass boundary".to_owned(),
                    ));
                }
                let key = GroupKey::of(stage.plan.identity, &candidate.inputs.extents).ok_or_else(
                    || program_error("Metal M3 stage has no canonical signature".to_owned()),
                )?;
                groups.entry(key).or_default().push((lane, index));
            }
        }

        let mut stages = Vec::new();
        for refs in pre.into_values().chain(post.into_values()) {
            stages.push(self.build_stage(
                target,
                pool,
                &candidates,
                &lane_table,
                &pending_flags,
                &row_meta,
                &row_indices,
                &refs,
            )?);
        }

        Ok(M3Group {
            stats: GroupStats {
                lanes: lanes as u64,
                ..GroupStats::default()
            },
            candidates,
            lane_table,
            statuses,
            channel_meta,
            pending_flags,
            row_meta,
            row_indices,
            resident,
            stages,
            readiness,
            commit,
            readiness_ordinal: self.next_ordinal(),
            commit_ordinal: self.next_ordinal(),
            encoded: false,
            post_begin: None,
        })
    }

    /// Build one stage group: schema checks, the per-lane buffers, the
    /// layout record, and the sized dispatches.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn build_stage(
        &mut self,
        target: &Context,
        pool: &Pool,
        candidates: &[LaneCandidate],
        lane_table: &Transient,
        pending_flags: &Transient,
        row_meta: &Transient,
        row_indices: &Transient,
        refs: &[(usize, usize)],
    ) -> Result<StageCommand> {
        let (first_lane, first_stage) = refs[0];
        let canon = &candidates[first_lane].fire.program().stages[first_stage];
        let plan = &canon.plan;
        let binding_count = used_channel_slots(&plan.ops).map_err(|why| {
            program_error(format!("Metal M3 stage binds too many channels: {why:?}"))
        })?;
        let use_fused = matches!(&canon.executable.grouped, Ok(regions) if !regions.is_empty());
        let regions: &[GroupedExecutable] = if use_fused {
            canon.executable.grouped.as_ref().map_or(&[], Vec::as_slice)
        } else {
            &canon.executable.grouped_singleton
        };
        if regions.is_empty() {
            let reason = canon
                .executable
                .grouped
                .as_ref()
                .err()
                .cloned()
                .unwrap_or_else(|| "Metal M3 has no grouped singleton fallback".to_owned());
            return Err(program_error(reason));
        }

        // Per-lane shapes against the canonical schema.
        let stage_lanes = refs.len();
        let vocab = candidates[first_lane].inputs.vocab;
        let mut lane_descriptors: Vec<Vec<ValueDesc>> = Vec::with_capacity(stage_lanes);
        let mut max_bytes = vec![4u64; plan.value_types.len()];
        let mut max_value_len = 1u64;
        let mut maximum_rows = 1u32;
        for &(lane, stage_index) in refs {
            let candidate = &candidates[lane];
            let lane_plan = &candidate.fire.program().stages[stage_index].plan;
            if candidate.inputs.vocab != vocab {
                return Err(program_error("Metal M3 stage vocab mismatch".to_owned()));
            }
            if lane_plan.value_types.len() != plan.value_types.len()
                || lane_plan.ops.len() != plan.ops.len()
                || lane_plan.channel_bindings.len() < binding_count
            {
                return Err(program_error(format!(
                    "Metal M3 canonical stage schema mismatch stage={} values={}/{} ops={}/{} \
                     bindings={}/>={binding_count}",
                    canon.kind,
                    lane_plan.value_types.len(),
                    plan.value_types.len(),
                    lane_plan.ops.len(),
                    plan.ops.len(),
                    lane_plan.channel_bindings.len(),
                )));
            }
            let mut descriptors = Vec::with_capacity(lane_plan.value_types.len());
            for (value, value_type) in lane_plan.value_types.iter().enumerate() {
                let descriptor =
                    describe(value_type, &candidate.inputs.extents).map_err(|why| {
                        program_error(format!("Metal M3 value shape did not resolve: {why:?}"))
                    })?;
                max_bytes[value] = max_bytes[value].max(descriptor.device_bytes());
                max_value_len = max_value_len.max(u64::from(descriptor.len));
                maximum_rows = maximum_rows.max(descriptor.rows);
                descriptors.push(descriptor);
            }
            for op in &lane_plan.ops {
                if op.code != CHAN_PUT {
                    continue;
                }
                let Some(&dense) = lane_plan.channel_bindings.get(op.channel as usize) else {
                    continue;
                };
                let arg = op.args.first().copied().unwrap_or_default() as usize;
                let sink = descriptors.get(arg).map_or(0, ValueDesc::wire_bytes);
                let cell = candidate
                    .fire
                    .pending
                    .get(dense as usize)
                    .map_or(0, super::handle::Handle::len);
                if sink > cell {
                    return Err(program_error(
                        "Metal M3 channel sink exceeds fixed cell size".to_owned(),
                    ));
                }
            }
            lane_descriptors.push(descriptors);
        }

        // The per-lane scratch layout: each value at the maximum size any
        // lane needs, so every lane strides identically.
        let mut offsets = vec![0u32; plan.value_types.len()];
        let mut stride = SCRATCH_ALIGN;
        for (value, &bytes) in max_bytes.iter().enumerate() {
            stride = stride
                .checked_next_multiple_of(SCRATCH_ALIGN)
                .ok_or_else(|| program_error("Metal M3 per-lane scratch overflows".to_owned()))?;
            offsets[value] = u32::try_from(stride).map_err(|_| {
                program_error("Metal M3 per-lane scratch exceeds u32 offsets".to_owned())
            })?;
            stride = stride
                .checked_add(bytes.next_multiple_of(SCRATCH_ALIGN))
                .ok_or_else(|| program_error("Metal M3 per-lane scratch overflows".to_owned()))?;
        }
        let temporary_offset = stride
            .checked_next_multiple_of(SCRATCH_ALIGN)
            .ok_or_else(|| program_error("Metal M3 per-lane scratch overflows".to_owned()))?;
        let stride = temporary_offset
            .checked_add((max_value_len * 16).next_multiple_of(SCRATCH_ALIGN))
            .ok_or_else(|| program_error("Metal M3 per-lane scratch overflows".to_owned()))?;
        if stride > MAX_SCRATCH_BYTES {
            return Err(program_error(
                "Metal M3 per-lane scratch exceeds the 512 MiB bound".to_owned(),
            ));
        }

        const DESC_BYTES: u64 = size_of::<ValueDesc>() as u64;
        const PARAM_BYTES: u64 = size_of::<OpParams>() as u64;
        let descriptors_buffer = pool.acquire(
            target,
            ((stage_lanes * plan.value_types.len()) as u64 * DESC_BYTES).max(DESC_BYTES),
        )?;
        let parameters = pool.acquire(
            target,
            ((stage_lanes * plan.ops.len()) as u64).max(1) * PARAM_BYTES,
        )?;
        let offsets_buffer = pool.acquire(target, (offsets.len() as u64).max(1) * 4)?;
        let scratch = pool.acquire(target, (stage_lanes as u64) * stride)?;
        let layout_buffer = pool.acquire(target, size_of::<GroupLayout>() as u64)?;
        let bindings = pool.acquire(target, ((stage_lanes * binding_count) as u64).max(1) * 4)?;
        let lane_indices = pool.acquire(target, (stage_lanes as u64) * 4)?;

        for (slot, descriptors) in lane_descriptors.iter().enumerate() {
            // SAFETY: `ValueDesc` is upload ABI; the buffer was sized for
            // `stage_lanes` blocks and no step names it.
            unsafe {
                descriptors_buffer.write(
                    (slot * plan.value_types.len()) as u64 * DESC_BYTES,
                    pod_bytes(descriptors),
                )?;
            }
        }
        // SAFETY: as above, for the offset table.
        unsafe { offsets_buffer.write(0, pod_bytes(&offsets))? };
        for (slot, &(lane, stage_index)) in refs.iter().enumerate() {
            let lane_plan = &candidates[lane].fire.program().stages[stage_index].plan;
            // SAFETY: `binding_count` entries fit — checked against every
            // lane's binding table above.
            unsafe {
                bindings.write(
                    (slot * binding_count) as u64 * 4,
                    pod_bytes(&lane_plan.channel_bindings[..binding_count]),
                )?;
                lane_indices.write(slot as u64 * 4, &(lane as u32).to_le_bytes())?;
            }
            let runtime = OpRuntime {
                vocab: candidates[lane].inputs.vocab,
                mtp_draft_row: candidates[lane].inputs.mtp_draft_row,
            };
            let mut result_base = 0u32;
            for (node, op) in lane_plan.ops.iter().enumerate() {
                let mut params = OpParams::of(op, result_base, runtime);
                if op.code == CHAN_PUT
                    && let Some(&dense) = lane_plan.channel_bindings.get(op.channel as usize)
                    && let Some(cell) = candidates[lane].fire.pending.get(dense as usize)
                {
                    params.sink_bytes = cell.len() as u32;
                }
                write_pod(
                    &parameters,
                    (slot * lane_plan.ops.len() + node) as u64 * PARAM_BYTES,
                    &params,
                )?;
                result_base += u32::from(op.result_count);
            }
        }
        write_pod(
            &layout_buffer,
            0,
            &GroupLayout {
                lane_count: stage_lanes as u32,
                value_count: plan.value_types.len() as u32,
                scratch_stride: stride as u32,
                temporary_offset: temporary_offset as u32,
                vocab,
                binding_stride: binding_count as u32,
                rows_per_lane: maximum_rows,
                op_stride: plan.ops.len() as u32,
            },
        )?;

        let fixed = [
            lane_table.gpu_address(),
            descriptors_buffer.gpu_address(),
            parameters.gpu_address(),
            offsets_buffer.gpu_address(),
            scratch.gpu_address(),
            layout_buffer.gpu_address(),
            bindings.gpu_address(),
            pending_flags.gpu_address(),
            lane_indices.gpu_address(),
            row_meta.gpu_address(),
            row_indices.gpu_address(),
        ];
        let mut encoded_regions = Vec::with_capacity(regions.len());
        for region in regions {
            let parallel = region.parallel_nucleus || region.parallel_topk;
            let (grid, threadgroup) = if parallel {
                let threads = (stage_lanes as u64) * u64::from(maximum_rows) * 256;
                if threads > u64::from(u32::MAX) {
                    return Err(program_error(
                        "Metal M3 parallel library launch exceeds u32 grid".to_owned(),
                    ));
                }
                ([threads as usize, 1, 1], [256, 1, 1])
            } else {
                // A generated region gets a threadgroup per lane, as wide as
                // the pipeline allows up to the emitter's bound: its
                // vocabulary-wide ops are latency-bound inside one
                // threadgroup, so width is throughput.
                let width = REGION_THREADS
                    .min(region.pso.maxTotalThreadsPerThreadgroup() as u32)
                    .max(32)
                    / 32
                    * 32;
                let threads = (stage_lanes as u64) * u64::from(width);
                if threads > u64::from(u32::MAX) {
                    return Err(program_error(
                        "Metal M3 region launch exceeds u32 grid".to_owned(),
                    ));
                }
                ([threads as usize, 1, 1], [width as usize, 1, 1])
            };
            encoded_regions.push(EncodedRegion {
                pso: region.pso.clone(),
                ordinal: self.next_ordinal(),
                fixed,
                grid,
                threadgroup,
                library: region.region.kind == driver_abi::local::PIE_REGION_LIBRARY,
                parallel_selection: parallel,
            });
        }

        Ok(StageCommand {
            kind: canon.kind,
            singleton_fallback: !use_fused,
            descriptors: descriptors_buffer,
            parameters,
            offsets: offsets_buffer,
            scratch,
            layout: layout_buffer,
            bindings,
            lane_indices,
            regions: encoded_regions,
        })
    }

    /// The grouped effect pair, when a program compile has produced it.
    fn grouped_effects(&self) -> Option<(Pso, Pso)> {
        self.grouped_effects_pair()
    }
}

impl M3Group {
    /// Encode the group readiness and the prologue stages, before the
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
        tables.bind_address(
            target,
            self.readiness_ordinal,
            0,
            self.lane_table.gpu_address(),
        )?;
        tables.bind_address(
            target,
            self.readiness_ordinal,
            1,
            self.channel_meta.gpu_address(),
        )?;
        step.set_pipeline(&self.readiness);
        step.set_argument_table_for(tables, self.readiness_ordinal)?;
        step.dispatch([self.candidates.len(), 1, 1], [1, 1, 1])?;
        self.stats.readiness_launches += 1;
        step.barrier(Visibility::Device);
        self.encode_stages(
            target,
            tables,
            step,
            tensor_ir::registry::Stage::Prologue as u8,
        )
    }

    /// Encode the epilogue stages and the group commit, after the forward.
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
        self.post_begin.get_or_insert_with(Instant::now);
        self.encode_stages(
            target,
            tables,
            step,
            tensor_ir::registry::Stage::Epilogue as u8,
        )?;
        tables.bind_address(
            target,
            self.commit_ordinal,
            0,
            self.lane_table.gpu_address(),
        )?;
        tables.bind_address(
            target,
            self.commit_ordinal,
            1,
            self.channel_meta.gpu_address(),
        )?;
        step.set_pipeline(&self.commit);
        step.set_argument_table_for(tables, self.commit_ordinal)?;
        step.dispatch([self.candidates.len(), 1, 1], [1, 1, 1])?;
        self.stats.commit_launches += 1;
        Ok(())
    }

    /// Read every lane's verdict and give the target its tables back.
    ///
    /// One outcome per candidate, in order. The group-level report covers
    /// the lanes that did not commit; a group never encoded reports that
    /// once, for every lane, rather than reading the zero fill as sixty-four
    /// faults.
    #[must_use]
    pub fn finish(
        mut self,
        tables: &mut Tables,
    ) -> (Vec<StatusOutcome>, Option<String>, GroupStats) {
        if let Some(begin) = self.post_begin {
            self.stats.post_forward_critical_ns = begin.elapsed().as_nanos() as u64;
        }
        for stage in &self.stages {
            for region in &stage.regions {
                tables.forget(region.ordinal);
            }
        }
        tables.forget(self.readiness_ordinal);
        tables.forget(self.commit_ordinal);

        if !self.encoded {
            let outcomes = vec![StatusOutcome::Failed; self.candidates.len()];
            return (
                outcomes,
                Some(
                    "Metal M3 group was prepared but never encoded: the forward it rides was \
                     not run, so no lane dispatched"
                        .to_owned(),
                ),
                self.stats,
            );
        }

        // SAFETY: the statuses buffer holds one record per lane and the
        // step's fence has signalled before finish is called.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.statuses.contents().as_ptr().cast::<u8>(),
                self.candidates.len() * STATUS_BYTES,
            )
        };
        let mut outcomes = Vec::with_capacity(self.candidates.len());
        let mut faults = String::new();
        let mut faulted = 0usize;
        for (lane, candidate) in self.candidates.iter().enumerate() {
            let status = Status::read(&bytes[lane * STATUS_BYTES..(lane + 1) * STATUS_BYTES])
                .unwrap_or_default();
            let (outcome, _) = StatusOutcome::of(status, true);
            if outcome == StatusOutcome::Failed {
                faulted += 1;
                // The lane status is the only account of what happened;
                // bound the report at a handful of lanes so one bad group
                // does not log a novel.
                if faulted <= 4 {
                    use std::fmt::Write as _;
                    let _ = write!(
                        faults,
                        "; lane {lane}: {}",
                        report_status(status, true, candidate.fire.rings.len() as u32)
                    );
                }
            }
            outcomes.push(outcome);
        }
        let report = if faulted == 0 {
            None
        } else {
            let more = faulted.saturating_sub(4);
            let tail = if more > 0 {
                format!("; and {more} more lanes")
            } else {
                String::new()
            };
            Some(format!("Metal M3 group faulted{faults}{tail}"))
        };
        (outcomes, report, self.stats)
    }

    /// What the encodes did.
    #[must_use]
    pub fn stats(&self) -> GroupStats {
        self.stats
    }

    /// Encode every stage of `kind`, with its counters.
    fn encode_stages(
        &mut self,
        target: &Context,
        tables: &mut Tables,
        step: &mut StepEncoder<'_>,
        kind: u8,
    ) -> Result<()> {
        for stage in &self.stages {
            if stage.kind != kind {
                continue;
            }
            for region in &stage.regions {
                for (slot, &address) in region.fixed.iter().enumerate() {
                    tables.bind_address(target, region.ordinal, slot, address)?;
                }
                step.set_pipeline(&region.pso);
                step.set_argument_table_for(tables, region.ordinal)?;
                step.dispatch(region.grid, region.threadgroup)?;
                self.stats.body_launches += 1;
                if region.library {
                    self.stats.library_launches += 1;
                }
                if region.parallel_selection {
                    self.stats.parallel_selection_launches += 1;
                }
                if stage.singleton_fallback {
                    self.stats.singleton_fallback_launches += 1;
                }
                step.barrier(Visibility::Device);
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for M3Group {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("M3Group")
            .field("lanes", &self.candidates.len())
            .field("stages", &self.stages.len())
            .field("encoded", &self.encoded)
            .finish_non_exhaustive()
    }
}

/// An `Error::Program` in one breath.
fn program_error(message: String) -> Error {
    Error::Program { message }
}

/// A lane-table offset that is `Some` by construction.
fn lane_offset(value: Option<u64>) -> Result<u64> {
    value.ok_or_else(|| program_error("Metal M3 lane-table offset left the table".to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The emitter's own doc on `METAL_M3_REGION_THREADS` says a hand-kept
    /// copy with a "must equal" comment has nothing comparing the two. This
    /// is the something.
    #[test]
    fn the_region_thread_mirror_still_matches_the_emitter() {
        assert_eq!(
            REGION_THREADS,
            tensor_compiler::codegen::metal::fused::METAL_M3_REGION_THREADS
        );
    }
}
