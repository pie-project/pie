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

use super::compile::{Form, Region, StreamedStep};
use super::shared::SharedRing;
use eta_compiler::codegen::metal::{StepKind, reduce_dispatch_levels};

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
const FIRST_CHANNEL_BUFFER: usize = 7;

const INTRINSIC_ELEMENT_BYTES: u64 = 2;

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub(super) const LIBRARY_SAMPLER_THREADS: usize = 256;

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub(super) const REGION_THREADS: u32 = eta_compiler::codegen::metal::fused::METAL_M3_REGION_THREADS;

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

const CELL_ALIGN: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelShape {
    pub capacity: u32,
    pub numel: usize,
    pub dtype: Dtype,
}

impl ChannelShape {
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

    #[must_use]
    pub fn cell_bytes(&self) -> usize {
        eta_exec::wire_cell_bytes(self.dtype, self.numel)
    }

    #[must_use]
    pub fn cell_stride(&self) -> usize {
        self.cell_bytes()
            .next_multiple_of(CELL_ALIGN)
            .max(CELL_ALIGN)
    }
}

#[derive(Debug)]
pub struct Rings {
    slabs: Vec<Buffer>,
    shapes: Vec<ChannelShape>,
    shared: Vec<Option<Arc<SharedRing>>>,
}

impl Rings {
    pub fn allocate(
        device: &Context,
        shapes: &[ChannelShape],
        adopted: &[Option<Arc<SharedRing>>],
    ) -> Result<Rings> {
        let mut slabs = Vec::with_capacity(shapes.len());
        let mut shared = Vec::with_capacity(shapes.len());
        for (channel, shape) in shapes.iter().enumerate() {
            if let Some(ring) = adopted.get(channel).and_then(Option::as_ref) {
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
                .ok_or_else(|| {
                    Fault::program("program::launch", "a ring past what a u64 counts")
                })?;
            slabs.push(Buffer::zeroed(device, bytes.max(1))?);
            shared.push(None);
        }
        Ok(Rings {
            slabs,
            shapes: shapes.to_vec(),
            shared,
        })
    }

    #[must_use]
    pub fn shared(&self, channel: usize) -> Option<&Arc<SharedRing>> {
        self.shared.get(channel).and_then(Option::as_ref)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.slabs.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slabs.is_empty()
    }

    #[must_use]
    pub fn shape(&self, channel: usize) -> Option<ChannelShape> {
        self.shapes.get(channel).copied()
    }

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

    pub(crate) fn slab(&self, channel: usize) -> Result<&Buffer> {
        self.slabs.get(channel).ok_or_else(|| {
            Fault::program(
                "program::launch",
                format!("channel {channel} is not one this instance carries"),
            )
        })
    }

    pub fn write_cell(&mut self, channel: usize, sequence: u64, bytes: &[u8]) -> Result<()> {
        let at = self.cell_offset(channel, sequence)?;
        let width = self.shape(channel).map_or(0, |shape| shape.cell_bytes());
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

    pub fn read_cell(&self, channel: usize, sequence: u64) -> Result<Vec<u8>> {
        let at = self.cell_offset(channel, sequence)?;
        let width = self.shape(channel).map_or(0, |shape| shape.cell_bytes());
        let mut cell = vec![0u8; width];
        self.slabs[channel].read(at, &mut cell)?;
        Ok(cell)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    pub head: u64,
    pub tail: u64,
}

const INTRINSIC_SLOTS: usize = IntrinsicId::SLOTS as usize;

#[derive(Debug, Clone)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Slot {
    base: Buffer,
    offset: u64,
    width: u32,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Declared {
    width: u32,
    rows: u32,
    elements: u64,
}

#[derive(Debug, Clone)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Bound {
    slab: Buffer,
    committed: u64,
    pending: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct GroupLayout {
    lane_count: u32,
    value_count: u32,
    scratch_stride: u32,
    temporary_offset: u32,
    vocab: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct RowMeta {
    offset: u32,
    count: u32,
    mtp_offset: u32,
    reserved: u32,
}

#[derive(Debug)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Grouped {
    table: Buffer,
    shape: LaneShape,
    layouts: Vec<Buffer>,
    bindings: Buffer,
    pending_flags: Buffer,
    lane_indices: Buffer,
    row_meta: Buffer,
    row_indices: Buffer,
    trunk_rows: u32,
    draft_rows: u32,
    layout_words: Vec<GroupLayout>,
    record: LaneRecord,
    draft_base: u32,
}

const GROUPED_LANES: u32 = 1;

const THE_LANE: u32 = 0;

impl Grouped {
    #[allow(clippy::too_many_arguments)]
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

        let stride = u32::try_from(plan.channel_bindings.len())
            .map_err(|_| Fault::program("program::launch", "more channels than a u32 can count"))?;
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

        let row_meta =
            Buffer::zeroed(device, (GROUPED_LANES as u64) * size_of::<RowMeta>() as u64)?;
        let rows = u64::from(trunk_rows) + u64::from(draft_rows);
        let row_indices = Buffer::zeroed(device, rows.max(1) * size_of::<u32>() as u64)?;

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
            draft_base: 0,
        }))
    }

    fn set_vocab(&mut self, vocab: u32) -> Result<()> {
        for (words, buffer) in self.layout_words.iter_mut().zip(self.layouts.iter_mut()) {
            words.vocab = vocab;
            buffer.write(0, &record_bytes(words))?;
        }
        Ok(())
    }

    fn set_rows(&mut self, draft_base: u32) -> Result<()> {
        self.draft_base = draft_base;
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

    fn set_logits(&mut self, base: u64, row_offset: u32, row_count: u32) -> Result<()> {
        self.record.logits_base = base;
        self.record.logits_row_offset = row_offset;
        self.record.logits_row_count = row_count;
        self.write_record()
    }

    fn set_scores(&mut self, base: u64, row_stride: u32) -> Result<()> {
        self.record.attn_score_base = base;
        self.record.attn_score_row_stride = row_stride;
        self.write_record()
    }

    fn set_drafts(&mut self, base: u64, depth: u32) -> Result<()> {
        self.record.mtp_drafts_base = base;
        self.record.mtp_drafts_depth = depth;
        self.write_record()
    }

    fn write_record(&mut self) -> Result<()> {
        let at = self.shape.record_offset(THE_LANE).ok_or_else(|| {
            Fault::program("program::launch", "the one lane is outside the lane table")
        })?;
        self.table.write(at, &record_bytes(&self.record))
    }

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

#[derive(Debug)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Prepared {
    status: Buffer,
    descriptors: Buffer,
    params: Buffer,
    offsets: Buffer,
    scratch: Buffer,
    intrinsics: [Option<Slot>; INTRINSIC_SLOTS],
    declared: [Option<Declared>; INTRINSIC_SLOTS],
    channel_count: u32,
    value_count: u32,
    scratch_stride: u32,
    temporary_offset: u32,
    bindings: Vec<u32>,
    bound: Vec<Bound>,
    grouped: Option<Grouped>,
    region_intrinsics: Vec<u64>,
    strided: u64,
    descriptor_bytes: Vec<u8>,
    param_bytes: Vec<u8>,
    offset_bytes: Vec<u8>,
    descriptor_table: Vec<ValueDesc>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct StepWord {
    index: u32,
    level: u32,
    groups: u32,
    reserved: u32,
}

fn scratch_zeroing_skipped() -> bool {
    crate::diag::on().scratch_no_zero
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct StreamedKnobs {
    max_groups: u32,
    repeat: usize,
    repeat_kind: Option<StepKind>,
    limit: usize,
}

const READOUT_INTRINSICS: [IntrinsicId; 3] = [
    IntrinsicId::Logits,
    IntrinsicId::Velocity,
    IntrinsicId::Hidden,
];

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
fn streamed_knobs() -> StreamedKnobs {
    let diag = crate::diag::on();
    StreamedKnobs {
        max_groups: diag.streamed_groups.unwrap_or(STREAMED_MAX_GROUPS),
        repeat: diag.streamed_repeat.max(1),
        repeat_kind: diag.streamed_repeat_kind.map(|kind| match kind {
            crate::diag::StreamedKind::Wide => StepKind::Wide,
            crate::diag::StreamedKind::Partial => StepKind::Partial,
            crate::diag::StreamedKind::Single => StepKind::Single,
            crate::diag::StreamedKind::Reduce => StepKind::Reduce,
            crate::diag::StreamedKind::Argmax => StepKind::Argmax,
        }),
        limit: diag.streamed_limit.unwrap_or(usize::MAX),
    }
}

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
const STREAMED_MAX_GROUPS: u32 = 64;

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
const REDUCE_CHUNKS_PER_GROUP: u32 = 4;

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub(super) fn streamed_threads(max_total: usize) -> usize {
    let capped = max_total.clamp(1, REGION_THREADS as usize);
    let pow2 = 1usize << (usize::BITS - 1 - capped.leading_zeros());
    pow2.max(32)
}

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
fn streamed_dispatches(
    steps: &[StreamedStep],
    descriptors: &[ValueDesc],
    threads: u32,
    temporary_bytes: u64,
) -> Vec<(StepWord, u32)> {
    let threads = threads.max(1);
    let knobs = streamed_knobs();
    let groups_for = |work: u32| -> u32 { work.div_ceil(threads).clamp(1, knobs.max_groups) };
    let mut out = Vec::with_capacity(steps.len());
    let repeats = |kind: StepKind| -> usize {
        match knobs.repeat_kind {
            Some(k) if k == kind => knobs.repeat,
            Some(_) => 1,
            None if kind == StepKind::Wide => knobs.repeat,
            None => 1,
        }
    };
    for (position, step) in steps.iter().enumerate().take(knobs.limit) {
        let word = StepWord {
            index: u32::try_from(position).unwrap_or(u32::MAX),
            ..StepWord::default()
        };
        match step.kind {
            StepKind::Wide | StepKind::Partial => {
                let sized_by = if step.kind == StepKind::Partial {
                    step.input
                } else {
                    step.result
                };
                let len = descriptors
                    .get(sized_by as usize)
                    .map_or(1, |desc| desc.len);
                for _ in 0..repeats(StepKind::Wide) {
                    out.push((word, groups_for(len)));
                }
            }
            StepKind::Single => {
                for _ in 0..repeats(StepKind::Single) {
                    out.push((word, 1));
                }
            }
            StepKind::Reduce => {
                let desc = descriptors
                    .get(step.input as usize)
                    .copied()
                    .unwrap_or_default();
                let mut count = desc.last;
                let mut at = 0u32;
                for level in reduce_dispatch_levels(desc.last) {
                    while at < level {
                        count = count.div_ceil(32);
                        at += 1;
                    }
                    let chunks = count.div_ceil(32).max(1);
                    let groups = chunks.div_ceil(32 * REDUCE_CHUNKS_PER_GROUP).max(1);
                    for _ in 0..repeats(StepKind::Reduce) {
                        out.push((StepWord { level, ..word }, groups));
                    }
                }
            }
            StepKind::Argmax => {
                let desc = descriptors
                    .get(step.input as usize)
                    .copied()
                    .unwrap_or_default();
                let candidate_bytes = 16u64;
                let fit = (temporary_bytes / candidate_bytes / u64::from(desc.rows.max(1)))
                    .clamp(1, u64::from(STREAMED_MAX_GROUPS)) as u32;
                let groups = groups_for(desc.last).min(fit);
                for _ in 0..repeats(StepKind::Argmax) {
                    out.push((
                        StepWord {
                            level: 0,
                            groups,
                            ..word
                        },
                        groups,
                    ));
                    out.push((
                        StepWord {
                            level: 1,
                            groups,
                            ..word
                        },
                        1,
                    ));
                }
            }
        }
    }
    let nops = crate::diag::on().streamed_nop;
    for _ in 0..nops {
        out.push((
            StepWord {
                index: u32::MAX,
                ..StepWord::default()
            },
            1,
        ));
    }
    if crate::diag::on().streamed_trace {
        let grid: Vec<String> = out
            .iter()
            .map(|(word, groups)| format!("{}:{}x{groups}", word.index, word.level))
            .collect();
        let sizes: Vec<String> = steps
            .iter()
            .map(|step| {
                format!(
                    "{:?}[r{}={} i{}={}]",
                    step.kind,
                    step.result,
                    descriptors.get(step.result as usize).map_or(0, |d| d.len),
                    step.input,
                    descriptors.get(step.input as usize).map_or(0, |d| d.len)
                )
            })
            .collect();
        eprintln!(
            "streamed: {} step(s) -> {} dispatch(es), {threads} threads, repeat {} of {:?}; grid {}; sizes {}",
            steps.len(),
            out.len(),
            knobs.repeat,
            knobs.repeat_kind,
            grid.join(" "),
            sizes.join(" ")
        );
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BatchKey {
    content: u64,
    channel_count: u32,
    channel_slots_per_lane: u32,
    scratch_stride: u32,
    temporary_offset: u32,
    trunk_rows: u32,
    draft_rows: u32,
    regions: usize,
}

impl Prepared {
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

        let mut records = Vec::with_capacity(plan.ops.len());
        let mut declared: [Option<Declared>; INTRINSIC_SLOTS] = [None; INTRINSIC_SLOTS];
        let mut result_base = 0u32;
        for op in &plan.ops {
            let mut record = OpParams::of(op, result_base, OpRuntime::default());
            if let (true, Some(channel)) = (op.tag == tags::CHAN_PUT, op.channel) {
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

        let param_bytes = records_bytes(&records);
        let mut params = Buffer::zeroed(
            device,
            (records.len() * size_of::<OpParams>()).max(size_of::<OpParams>()) as u64,
        )?;
        params.write(0, &param_bytes)?;

        let descriptor_bytes: Vec<u8> = descriptors.iter().flat_map(record_bytes).collect();
        let mut descriptor_buffer = Buffer::zeroed(device, descriptor_bytes.len().max(1) as u64)?;
        descriptor_buffer.write(0, &descriptor_bytes)?;

        let offset_bytes: Vec<u8> = scratch_layout
            .values
            .iter()
            .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut offsets = Buffer::zeroed(device, offset_bytes.len().max(size_of::<u32>()) as u64)?;
        offsets.write(0, &offset_bytes)?;

        let scratch_bytes = u64::from(scratch_stride).max(SCRATCH_ALIGN);
        let scratch = Buffer::zeroed(device, scratch_bytes)?;
        let status = Buffer::zeroed(device, eta_exec::STATUS_BYTES as u64)?;

        let lanes = LaneShape::of(
            GROUPED_LANES,
            u32::try_from(shapes.len()).map_err(|_| {
                Fault::program("program::launch", "more channels than a u32 can count")
            })?,
        );
        let reads = |wanted: IntrinsicId| {
            plan.ops.iter().any(|op| {
                op.intrinsic
                    .is_some_and(|id| id as usize == wanted as usize)
            })
        };
        let trunk_rows = READOUT_INTRINSICS
            .iter()
            .map(|id| {
                declared[*id as usize]
                    .map_or(0, |it| it.rows)
                    .max(u32::from(reads(*id)))
            })
            .max()
            .unwrap_or(0)
            .max(plan.drafts_len);
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
            descriptor_bytes,
            param_bytes,
            offset_bytes,
            descriptor_table: descriptors,
        })
    }

    #[must_use]
    pub fn batch_key(&self) -> Option<BatchKey> {
        use std::hash::{Hash, Hasher};
        let grouped = self.grouped.as_ref()?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.descriptor_bytes.hash(&mut hasher);
        self.param_bytes.hash(&mut hasher);
        self.offset_bytes.hash(&mut hasher);
        self.bindings.hash(&mut hasher);
        Some(BatchKey {
            content: hasher.finish(),
            channel_count: self.channel_count,
            channel_slots_per_lane: grouped.shape.channel_slots_per_lane,
            scratch_stride: self.scratch_stride,
            temporary_offset: self.temporary_offset,
            trunk_rows: grouped.trunk_rows,
            draft_rows: grouped.draft_rows,
            regions: grouped.layouts.len(),
        })
    }

    pub fn refresh(&mut self, rings: &Rings, cursors: &[Cursor]) -> Result<()> {
        self.refresh_cells(rings, cursors)?;
        if scratch_zeroing_skipped() {
            return Ok(());
        }
        let bytes = self.scratch.bytes();
        self.scratch.zero_span(0, bytes)
    }

    pub fn zero_scratch(&mut self) -> Result<()> {
        let bytes = self.scratch.bytes();
        self.scratch.zero_span(0, bytes)
    }

    #[cfg(target_vendor = "apple")]
    pub fn zero_scratch_on(&mut self, frame: &mut Frame) -> Result<()> {
        if scratch_zeroing_skipped() {
            return Ok(());
        }
        let bytes = self.scratch.bytes();
        frame.fill(self.scratch.slab(), 0, bytes)?;
        frame.next_pass()?;
        Ok(())
    }

    pub fn refresh_cells(&mut self, rings: &Rings, cursors: &[Cursor]) -> Result<()> {
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
        let ready = Status {
            state: 1,
            fault: 0,
            reserved0: 0,
            reserved1: 0,
        };
        self.status.write(0, &record_bytes(&ready))?;
        if let Some(grouped) = self.grouped.as_mut() {
            grouped.refresh(&self.bindings, &self.bound)?;
        }
        Ok(())
    }

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
        let declared = *self.declared.get(intrinsic as usize).ok_or_else(|| {
            Fault::program(
                "program::launch",
                format!("{intrinsic:?} is past the pitch the slot table is indexed with"),
            )
        })?;
        let element = eta_compiler::codegen::metal::m2_intrinsic_element_bytes(intrinsic as u16)
            .map(u64::from)
            .ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    format!("{intrinsic:?} has no element width in the M2 slot table"),
                )
            })?;
        let wanted = match intrinsic {
            IntrinsicId::MtpDrafts => Dtype::I32,
            _ if element == 4 => Dtype::F32,
            _ => Dtype::Bf16,
        };
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
            let reach = offset.saturating_add(declared.elements.saturating_mul(element));
            if reach > base.bytes() {
                return Err(Fault::Ceiling {
                    what: "bytes in the rectangle an intrinsic is pointed at",
                    need: reach,
                    have: base.bytes(),
                });
            }
        }
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

    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    fn strideable(&self, intrinsic: IntrinsicId) -> bool {
        self.grouped.is_some()
            && eta_compiler::codegen::metal::m3_intrinsic_bindable(intrinsic as u16)
    }

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

    fn regroup(&mut self) -> Result<()> {
        if self.grouped.is_none() {
            return Ok(());
        }
        let (score_base, score_stride) =
            match self.intrinsics[IntrinsicId::AttnScore as usize].as_ref() {
                Some(slot) => (address_of(&slot.base, slot.offset)?, slot.width),
                None => (0, 0),
            };
        self.grouped
            .as_mut()
            .expect("the seat was there one statement ago")
            .set_scores(score_base, score_stride)?;
        let (drafts_base, drafts_depth) =
            match self.intrinsics[IntrinsicId::MtpDrafts as usize].as_ref() {
                Some(slot) => (address_of(&slot.base, slot.offset)?, slot.width),
                None => (0, 0),
            };
        self.grouped
            .as_mut()
            .expect("the seat was there one statement ago")
            .set_drafts(drafts_base, drafts_depth)?;
        let Some(trunk) = READOUT_INTRINSICS
            .iter()
            .find_map(|id| self.intrinsics[*id as usize].as_ref())
        else {
            return Ok(());
        };
        let width = trunk.width;
        let logits_base = address_of(&trunk.base, trunk.offset)?;
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

    pub fn encode_into(&self, frame: &Frame, region: &Region) -> Result<()> {
        match region.form {
            Form::Fused => self.encode_fused(frame, region, &self.scratch, 0),
            Form::Streamed => self.encode_streamed(frame, region),
            Form::Grouped | Form::GroupedLibrary => self.encode_grouped(frame, region),
        }
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn encode_streamed(&self, frame: &Frame, region: &Region) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2::runtime::ProtocolObject;
            use objc2_metal::{
                MTLComputeCommandEncoder, MTLComputePipelineState, MTLResource, MTLResourceUsage,
            };
            let grouped = self.grouped.as_ref().ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    "a region was compiled for the streamed form and this stage carries no \
                     lane table; the plan said the grouped path could not cover it",
                )
            })?;
            let layout = grouped
                .layouts
                .get(region.region_index as usize)
                .ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!("region {} has no group layout", region.region_index),
                    )
                })?;
            let encoder = frame.encoder();
            encoder.setComputePipelineState(region.pipeline());
            // SAFETY: every reservation is retained by `self`; every offset is zero.
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
                resident(&cell.slab, MTLResourceUsage::Read | MTLResourceUsage::Write);
            }
            for held in self.intrinsics.iter().flatten() {
                resident(&held.base, MTLResourceUsage::Read);
            }
            let threads = streamed_threads(region.pipeline().maxTotalThreadsPerThreadgroup());
            let temporary_bytes =
                u64::from(self.scratch_stride.saturating_sub(self.temporary_offset));
            dispatch_streamed(
                encoder,
                &streamed_dispatches(
                    &region.steps,
                    &self.descriptor_table,
                    u32::try_from(threads).unwrap_or(1),
                    temporary_bytes,
                ),
                threads,
                GROUPED_LANES as usize,
            );
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn encode_fused(
        &self,
        frame: &Frame,
        region: &Region,
        scratch: &Buffer,
        scratch_at: u64,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

            self.no_stride_owed(region)?;
            let encoder = frame.encoder();
            encoder.setComputePipelineState(region.pipeline());
            let scratch_at = usize::try_from(scratch_at).unwrap_or(usize::MAX);
            // SAFETY: every buffer is retained by `self`/`region`/the batch; every offset was bounds-checked.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.status.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.descriptors.raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(self.params.raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(self.offsets.raw()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(scratch.raw()), scratch_at, 4);
                encoder.setBuffer_offset_atIndex(
                    Some(scratch.raw()),
                    scratch_at + self.temporary_offset as usize,
                    5,
                );
                let readout = READOUT_INTRINSICS
                    .iter()
                    .find_map(|id| self.intrinsics[*id as usize].as_ref());
                match readout {
                    Some(bound) => encoder.setBuffer_offset_atIndex(
                        Some(bound.base.raw()),
                        usize::try_from(bound.offset).unwrap_or(0),
                        eta_compiler::codegen::metal::M2_LOGITS_BUFFER,
                    ),
                    None => encoder.setBuffer_offset_atIndex(
                        None,
                        0,
                        eta_compiler::codegen::metal::M2_LOGITS_BUFFER,
                    ),
                }
                for (slot, held) in self.intrinsics.iter().enumerate() {
                    let Some(at) = u16::try_from(slot)
                        .ok()
                        .and_then(eta_compiler::codegen::metal::m2_intrinsic_buffer)
                        .filter(|at| *at != eta_compiler::codegen::metal::M2_LOGITS_BUFFER)
                    else {
                        continue;
                    };
                    if let Some(bound) = held {
                        encoder.setBuffer_offset_atIndex(
                            Some(bound.base.raw()),
                            usize::try_from(bound.offset).unwrap_or(0),
                            at,
                        );
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
            Err(Fault::Deviceless)
        }
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn encode_grouped(&self, frame: &Frame, region: &Region) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
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
            // SAFETY: every reservation is retained by `self`; every offset is zero (the kernel strides off `layout`).
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
                resident(&cell.slab, MTLResourceUsage::Read | MTLResourceUsage::Write);
            }
            for held in self.intrinsics.iter().flatten() {
                resident(&held.base, MTLResourceUsage::Read);
            }

            let rows = grouped
                .layout_words
                .get(region.region_index as usize)
                .map_or(1, |words| words.reserved1 as usize);
            let (groups, threads) = match region.form {
                Form::Fused | Form::Streamed => {
                    unreachable!("`encode_into` routes the single-lane and streamed forms")
                }
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
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }

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

    #[must_use]
    pub const fn channel_count(&self) -> u32 {
        self.channel_count
    }

    #[must_use]
    pub const fn value_count(&self) -> u32 {
        self.value_count
    }

    #[must_use]
    pub const fn scratch_stride(&self) -> u32 {
        self.scratch_stride
    }
}

#[derive(Debug)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Batch {
    key: BatchKey,
    lanes: u32,
    shape: LaneShape,
    table: Buffer,
    descriptors: Buffer,
    params: Buffer,
    offsets: Buffer,
    bindings: Buffer,
    pending_flags: Buffer,
    lane_indices: Buffer,
    row_meta: Buffer,
    row_indices: Buffer,
    scratch: Buffer,
    layouts: Vec<Buffer>,
    layout_words: Vec<GroupLayout>,
    rows_per_lane: u32,
}

impl Batch {
    pub fn build(device: &Context, template: &Prepared, lanes: u32) -> Result<Batch> {
        let key = template.batch_key().ok_or_else(|| {
            Fault::program(
                "program::launch",
                "a batch was asked for a stage with no grouped seat; nothing in it can be \
                 launched together",
            )
        })?;
        let grouped = template
            .grouped
            .as_ref()
            .expect("the key was answered off this seat one statement ago");
        let lanes = lanes.max(1);
        let shape = LaneShape::of(lanes, grouped.shape.channel_slots_per_lane);
        let bytes = shape.bytes().ok_or_else(|| {
            Fault::program(
                "program::launch",
                "a batch lane table whose size does not fit a u64",
            )
        })?;
        let mut table = Buffer::zeroed(device, bytes)?;
        table.write(
            0,
            &record_bytes(&LaneHeader {
                abi_version: LANE_ABI_VERSION,
                lane_count: lanes,
                channel_slots_per_lane: shape.channel_slots_per_lane,
                flags: 0,
            }),
        )?;
        let replicate = |device: &Context, bytes: &[u8]| -> Result<Buffer> {
            let mut buffer =
                Buffer::zeroed(device, (bytes.len() as u64 * u64::from(lanes)).max(4))?;
            for lane in 0..lanes {
                buffer.write(u64::from(lane) * bytes.len() as u64, bytes)?;
            }
            Ok(buffer)
        };
        let descriptors = replicate(device, &template.descriptor_bytes)?;
        let params = replicate(device, &template.param_bytes)?;
        let mut offsets = Buffer::zeroed(device, template.offset_bytes.len().max(4) as u64)?;
        offsets.write(0, &template.offset_bytes)?;
        let binding_bytes: Vec<u8> = template
            .bindings
            .iter()
            .copied()
            .flat_map(u32::to_le_bytes)
            .collect();
        let bindings = replicate(device, &binding_bytes)?;
        let pending_flags = Buffer::zeroed(
            device,
            (u64::from(lanes) * u64::from(shape.channel_slots_per_lane)).max(1),
        )?;
        let index_bytes: Vec<u8> = (0..lanes).flat_map(u32::to_le_bytes).collect();
        let mut lane_indices = Buffer::zeroed(device, index_bytes.len() as u64)?;
        lane_indices.write(0, &index_bytes)?;
        let row_meta = Buffer::zeroed(device, u64::from(lanes) * size_of::<RowMeta>() as u64)?;
        let rows_per_lane = grouped.trunk_rows.saturating_add(grouped.draft_rows);
        let row_indices = Buffer::zeroed(
            device,
            (u64::from(lanes) * u64::from(rows_per_lane) * size_of::<u32>() as u64).max(4),
        )?;
        let scratch = Buffer::zeroed(
            device,
            (u64::from(lanes) * u64::from(template.scratch_stride)).max(SCRATCH_ALIGN),
        )?;
        let mut layouts = Vec::with_capacity(grouped.layout_words.len());
        for words in &grouped.layout_words {
            let mut buffer = Buffer::zeroed(device, size_of::<GroupLayout>() as u64)?;
            buffer.write(0, &record_bytes(words))?;
            layouts.push(buffer);
        }
        Ok(Batch {
            key,
            lanes,
            shape,
            table,
            descriptors,
            params,
            offsets,
            bindings,
            pending_flags,
            lane_indices,
            row_meta,
            row_indices,
            scratch,
            layouts,
            layout_words: grouped.layout_words.clone(),
            rows_per_lane,
        })
    }

    #[must_use]
    pub fn key(&self) -> BatchKey {
        self.key
    }

    #[must_use]
    pub fn lanes(&self) -> u32 {
        self.lanes
    }

    pub fn encode(
        &mut self,
        frame: &mut Frame,
        regions: &[Region],
        members: &mut [&mut Prepared],
    ) -> Result<()> {
        let count = u32::try_from(members.len()).unwrap_or(u32::MAX);
        if count > self.lanes {
            return Err(Fault::Ceiling {
                what: "dispatch lanes in a batch",
                need: u64::from(count),
                have: u64::from(self.lanes),
            });
        }
        if count == 0 {
            return Ok(());
        }
        let stride = u64::from(self.key.scratch_stride);
        self.pending_flags.zero_span(
            0,
            u64::from(count) * u64::from(self.shape.channel_slots_per_lane),
        )?;
        if !scratch_zeroing_skipped() {
            #[cfg(target_vendor = "apple")]
            {
                frame.fill(self.scratch.slab(), 0, u64::from(count) * stride)?;
                frame.next_pass()?;
            }
            #[cfg(not(target_vendor = "apple"))]
            self.scratch.zero_span(0, u64::from(count) * stride)?;
        }
        let frame: &Frame = frame;
        let vocab = members[0]
            .grouped
            .as_ref()
            .and_then(|g| g.layout_words.first())
            .map_or(0, |words| words.vocab);
        for (index, member) in members.iter().enumerate() {
            let lane = u32::try_from(index).unwrap_or(u32::MAX);
            if member.batch_key() != Some(self.key) {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "member {index} of a batch does not share its key; it would read \
                         another member's tables"
                    ),
                ));
            }
            let grouped = member
                .grouped
                .as_ref()
                .expect("the key was answered off this seat");
            if grouped.layout_words.first().map_or(0, |w| w.vocab) != vocab {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "member {index} of a batch reads a rectangle of another width than \
                         member 0; one launch has one row pitch"
                    ),
                ));
            }
            let mut record = grouped.record;
            record.channel_slot_offset = self.shape.slot_index(lane).unwrap_or(0);
            let at = self.shape.record_offset(lane).ok_or_else(|| {
                Fault::program("program::launch", "a batch member outside its lane table")
            })?;
            self.table.write(at, &record_bytes(&record))?;
            for (&dense, cell) in member.bindings.iter().zip(&member.bound) {
                let slot = LaneChannelSlot {
                    committed_cell: address_of(&cell.slab, cell.committed)?,
                    pending_cell: address_of(&cell.slab, cell.pending)?,
                    expected_head: NO_TICKET,
                    expected_tail: NO_TICKET,
                };
                let at = self.shape.slot_offset(lane, dense).ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!(
                            "channel {dense} is outside the batch's slot window, which was \
                             carved for the channels its template carries"
                        ),
                    )
                })?;
                self.table.write(at, &record_bytes(&slot))?;
            }
            let row_base = lane * self.rows_per_lane;
            let meta = RowMeta {
                offset: row_base,
                count: self.rows_per_lane,
                mtp_offset: grouped.trunk_rows,
                reserved: 0,
            };
            self.row_meta.write(
                u64::from(lane) * size_of::<RowMeta>() as u64,
                &record_bytes(&meta),
            )?;
            let rows: Vec<u8> = (0..grouped.trunk_rows)
                .chain((0..grouped.draft_rows).map(|row| grouped.draft_base.saturating_add(row)))
                .flat_map(u32::to_le_bytes)
                .collect();
            if !rows.is_empty() {
                self.row_indices
                    .write(u64::from(row_base) * size_of::<u32>() as u64, &rows)?;
            }
        }
        for (words, buffer) in self.layout_words.iter_mut().zip(self.layouts.iter_mut()) {
            words.lane_count = count;
            words.vocab = vocab;
            buffer.write(0, &record_bytes(words))?;
        }
        for region in regions {
            match region.form {
                Form::Fused => {
                    for (index, member) in members.iter().enumerate() {
                        member.encode_fused(frame, region, &self.scratch, index as u64 * stride)?;
                    }
                }
                Form::Grouped | Form::GroupedLibrary => {
                    self.encode_grouped(frame, region, members, count)?;
                }
                Form::Streamed => {
                    self.encode_streamed(frame, region, members, count)?;
                }
            }
        }
        Ok(())
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn encode_streamed(
        &self,
        frame: &Frame,
        region: &Region,
        members: &[&mut Prepared],
        count: u32,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2::runtime::ProtocolObject;
            use objc2_metal::{
                MTLComputeCommandEncoder, MTLComputePipelineState, MTLResource, MTLResourceUsage,
            };
            let layout = self
                .layouts
                .get(region.region_index as usize)
                .ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!(
                            "region {} has no group layout in this batch",
                            region.region_index
                        ),
                    )
                })?;
            let encoder = frame.encoder();
            encoder.setComputePipelineState(region.pipeline());
            // SAFETY: every buffer is retained by `self`; every offset is zero.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.table.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.descriptors.raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(self.params.raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(self.offsets.raw()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(self.scratch.raw()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(layout.raw()), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(self.bindings.raw()), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(self.pending_flags.raw()), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(self.lane_indices.raw()), 0, 8);
                encoder.setBuffer_offset_atIndex(Some(self.row_meta.raw()), 0, 9);
                encoder.setBuffer_offset_atIndex(Some(self.row_indices.raw()), 0, 10);
            }
            let resident = |buffer: &Buffer, usage: MTLResourceUsage| {
                let resource: &ProtocolObject<dyn MTLResource> =
                    ProtocolObject::from_ref(&**buffer.slab());
                encoder.useResource_usage(resource, usage);
            };
            for member in members {
                resident(
                    &member.status,
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                for cell in &member.bound {
                    resident(&cell.slab, MTLResourceUsage::Read | MTLResourceUsage::Write);
                }
                for held in member.intrinsics.iter().flatten() {
                    resident(&held.base, MTLResourceUsage::Read);
                }
            }
            let template = members
                .first()
                .ok_or_else(|| Fault::program("program::launch", "a batch with no members"))?;
            let threads = streamed_threads(region.pipeline().maxTotalThreadsPerThreadgroup());
            let temporary_bytes = u64::from(
                self.key
                    .scratch_stride
                    .saturating_sub(self.key.temporary_offset),
            );
            self.dump_streamed_tables(region, template)?;
            dispatch_streamed(
                encoder,
                &streamed_dispatches(
                    &region.steps,
                    &template.descriptor_table,
                    u32::try_from(threads).unwrap_or(1),
                    temporary_bytes,
                ),
                threads,
                count as usize,
            );
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }

    #[cfg(target_vendor = "apple")]
    fn dump_streamed_tables(&self, region: &Region, template: &Prepared) -> Result<()> {
        let Some(dir) = crate::diag::on().kernel_dump.as_deref() else {
            return Ok(());
        };
        let path = dir.join(format!("{}.tables", region.module.entry()));
        if path.exists() {
            return Ok(());
        }
        let Some(words) = self.layout_words.get(region.region_index as usize) else {
            return Ok(());
        };
        let value_count = template.descriptor_table.len();
        let params_per_lane = words.reserved2 as usize;
        let mut descriptors = vec![0u8; value_count * size_of::<ValueDesc>()];
        self.descriptors.read(0, &mut descriptors)?;
        let mut params = vec![0u8; params_per_lane * size_of::<OpParams>()];
        self.params.read(0, &mut params)?;
        let mut offsets = vec![0u8; value_count * 4];
        self.offsets.read(0, &mut offsets)?;
        let mut bytes = Vec::new();
        for word in [
            value_count as u32,
            params_per_lane as u32,
            self.key.scratch_stride,
            self.key.temporary_offset,
        ] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(&record_bytes(words));
        bytes.extend_from_slice(&descriptors);
        bytes.extend_from_slice(&params);
        bytes.extend_from_slice(&offsets);
        let _ = std::fs::write(path, bytes);
        Ok(())
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn encode_grouped(
        &self,
        frame: &Frame,
        region: &Region,
        members: &[&mut Prepared],
        count: u32,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2::runtime::ProtocolObject;
            use objc2_metal::{
                MTLComputeCommandEncoder, MTLComputePipelineState, MTLResource, MTLResourceUsage,
                MTLSize,
            };
            let layout = self
                .layouts
                .get(region.region_index as usize)
                .ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!(
                            "region {} has no group layout in this batch",
                            region.region_index
                        ),
                    )
                })?;
            let encoder = frame.encoder();
            encoder.setComputePipelineState(region.pipeline());
            if let Some(template) = members.first() {
                self.dump_streamed_tables(region, template)?;
            }
            // SAFETY: every buffer is retained by `self`; every offset is zero (the kernel strides off `layout`).
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.table.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.descriptors.raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(self.params.raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(self.offsets.raw()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(self.scratch.raw()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(layout.raw()), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(self.bindings.raw()), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(self.pending_flags.raw()), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(self.lane_indices.raw()), 0, 8);
                encoder.setBuffer_offset_atIndex(Some(self.row_meta.raw()), 0, 9);
                encoder.setBuffer_offset_atIndex(Some(self.row_indices.raw()), 0, 10);
            }
            let resident = |buffer: &Buffer, usage: MTLResourceUsage| {
                let resource: &ProtocolObject<dyn MTLResource> =
                    ProtocolObject::from_ref(&**buffer.slab());
                encoder.useResource_usage(resource, usage);
            };
            for member in members {
                resident(
                    &member.status,
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                for cell in &member.bound {
                    resident(&cell.slab, MTLResourceUsage::Read | MTLResourceUsage::Write);
                }
                for held in member.intrinsics.iter().flatten() {
                    resident(&held.base, MTLResourceUsage::Read);
                }
            }
            let rows = self
                .layout_words
                .get(region.region_index as usize)
                .map_or(1, |words| words.reserved1 as usize);
            let (groups, threads) = match region.form {
                Form::Fused | Form::Streamed => {
                    unreachable!("`Batch::encode` routes the single-lane and streamed forms")
                }
                Form::GroupedLibrary => ((count as usize) * rows, LIBRARY_SAMPLER_THREADS),
                Form::Grouped => (
                    count as usize,
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
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }
}

#[cfg(target_vendor = "apple")]
fn dispatch_streamed(
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    dispatches: &[(StepWord, u32)],
    threads: usize,
    lanes: usize,
) {
    use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
    for (word, groups) in dispatches {
        // SAFETY: `word` lives for the call; Metal copies the bytes.
        unsafe {
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::from(word).cast(),
                size_of::<StepWord>(),
                11,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (*groups).max(1) as usize,
                height: lanes.max(1),
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
    }
}

fn record_bytes<T: Copy>(record: &T) -> Vec<u8> {
    // SAFETY: as stated above; the slice's life is this expression's.
    let bytes =
        unsafe { std::slice::from_raw_parts((record as *const T).cast::<u8>(), size_of::<T>()) };
    bytes.to_vec()
}

fn records_bytes<T: Copy>(records: &[T]) -> Vec<u8> {
    records.iter().flat_map(record_bytes).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_every_case() {
        a_channel_this_instance_does_not_carry_is_refused_by_number();
        the_shared_op_record_is_the_emitted_one();
        the_first_channel_binds_where_the_emitter_writes_it();
        the_group_layout_matches_the_emitted_struct();
        the_grouped_samplers_take_the_bindings_this_file_writes();
    }

    fn a_channel_this_instance_does_not_carry_is_refused_by_number() {
        let rings = Rings {
            slabs: Vec::new(),
            shapes: Vec::new(),
            shared: Vec::new(),
        };
        let said = rings
            .cell_offset(2, 0)
            .expect_err("no such channel")
            .to_string();
        assert!(said.contains('2'), "the refusal names the channel: {said}");
    }

    fn the_shared_op_record_is_the_emitted_one() {
        assert_eq!(size_of::<OpParams>(), 64);
        assert_eq!(size_of::<Status>(), 16);
    }

    fn the_first_channel_binds_where_the_emitter_writes_it() {
        assert_eq!(FIRST_CHANNEL_BUFFER, 7);
    }

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
        assert_eq!(size_of::<GroupLayout>(), (u32::BITS as usize));
        assert_eq!(
            fields("M3RowMeta"),
            [
                "uint offset",
                "uint count",
                "uint mtp_offset",
                "uint reserved"
            ],
            "`RowMeta` in this file is the host half of the emitted `M3RowMeta`, \
             and the two have parted"
        );
        assert_eq!(size_of::<RowMeta>(), 4 * size_of::<u32>());
    }

    fn the_grouped_samplers_take_the_bindings_this_file_writes() {
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
