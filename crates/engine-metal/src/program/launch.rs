#![allow(unsafe_code)]
use std::collections::BTreeMap;
use std::mem::size_of;
use std::sync::Arc;

use eta_compiler::codegen::lane_scratch::LaneScratch;
use eta_compiler::codegen::launch::{LaunchChannel, LaunchStagePlan};
use eta_compiler::codegen::metal::StepNodes;
use eta_exec::{Launch, Launches, Layout, TooLarge};
use eta_exec::{
    Extents, LANE_ABI_VERSION, LaneChannelSlot, LaneHeader, LaneRecord, LaneShape, NO_TICKET,
    OpParams, SCRATCH_ALIGN, Status, ValueDesc, describe,
};

/// The stage as Metal runs it, region by region as each was compiled: a
/// streamed region's launches are its steps as the emitter published their
/// nodes, each dispatched after the one before it, so a value whose last
/// reader's last step has run gives its bytes to one a later step writes,
/// and a view no step computes takes no bytes (its readers read its
/// source's); any other form runs its region as one dispatch, one launch.
/// Every other value a region names takes bytes (Metal's streamed form
/// writes them all); the temporary area is what each region's kernel
/// publishes it takes ([`temporary_needs`]), below it the select's reserve
/// as before.
fn launches<'a>(
    plan: &'a LaunchStagePlan,
    regions: &'a [Region],
    descriptors: &[ValueDesc],
) -> std::result::Result<Launches<'a>, eta_exec::TemporaryRefused> {
    let published: Vec<Option<&StepNodes>> = (0..plan.fused.len())
        .map(|index| {
            regions
                .iter()
                .find(|compiled| compiled.region_index as usize == index)
                .filter(|compiled| compiled.form == Form::Streamed)
                .map(|compiled| &*compiled.launches)
        })
        .collect();
    published_launches(plan, &published, descriptors)
}

/// [`launches`] from what each region of `plan` published: a streamed
/// region's step lists (empty where it is one launch), `None` for a region
/// of another form.
pub(crate) fn published_launches<'a>(
    plan: &'a LaunchStagePlan,
    published: &[Option<&'a StepNodes>],
    descriptors: &[ValueDesc],
) -> std::result::Result<Launches<'a>, eta_exec::TemporaryRefused> {
    let mut each = Vec::new();
    let mut idle = Vec::new();
    for (index, region) in plan.fused.iter().enumerate() {
        let published = published
            .get(index)
            .copied()
            .flatten()
            .filter(|published| !published.steps.is_empty());
        if let Some(published) = published {
            // The streamed form dispatches its steps one after another, a
            // launch a step; a node two steps compute lies in both.
            let first = each.len();
            each.extend(published.steps.iter().map(|nodes| Launch {
                nodes,
                stepwise: false,
                blocks: 1,
                row_value: None,
                row_alias: None,
                held: Vec::new(),
                keeps: None,
            }));
            idle.extend_from_slice(&published.views);
            // A folded node's step computes it in another form: its results
            // take no bytes (idle, so hollow), and what it reads is held
            // through that step.
            for &(step, node) in &published.folded {
                idle.push(node);
                if let (Some(launch), Some(op)) =
                    (each.get_mut(first + step as usize), plan.ops.get(node as usize))
                {
                    launch.held.extend(op.reads());
                }
            }
        } else {
            each.push(Launch {
                nodes: &region.nodes,
                stepwise: false,
                ..Launch::of_region(region, 1)
            });
        }
    }
    let hollow = idle
        .iter()
        .flat_map(|&node| results_of(plan, node))
        .collect();
    let streamed: Vec<usize> = (0..plan.fused.len())
        .filter(|&index| published.get(index).copied().flatten().is_some())
        .collect();
    Ok(Launches {
        each,
        idle,
        hollow,
        // Held clear below the temporary area as the whole-value layout held
        // it: a one-row select keeps its state there.
        reserve: eta_exec::SELECT_RESERVE,
        temporary: eta_exec::stage_temporaries(descriptors, &temporary_needs(plan, &streamed))?,
        ..Launches::default()
    })
}

/// What each streamed region's kernel takes of a lane's temporary area, as
/// Metal's emitter publishes it: a top-k's histograms and candidates a row
/// of its input, or each select's state below the reductions' planes, for
/// the regions `streamed` names. The other forms take the whole-value rule
/// alone.
pub(crate) fn temporary_needs(
    plan: &LaunchStagePlan,
    streamed: &[usize],
) -> Vec<eta_exec::TemporaryNeed> {
    use eta_compiler::codegen::metal::{streamed_temporary_need, streamed_topk_temporary_need};
    streamed
        .iter()
        .map(|&index| {
            streamed_topk_temporary_need(plan, index)
                .unwrap_or_else(|| streamed_temporary_need(plan, index))
        })
        .collect()
}

/// The values node `node` of `plan` results in.
fn results_of(plan: &LaunchStagePlan, node: u32) -> std::ops::Range<u32> {
    let base: u32 = plan
        .ops
        .iter()
        .take(node as usize)
        .map(|op| u32::from(op.result_count))
        .sum();
    let count = plan
        .ops
        .get(node as usize)
        .map_or(0, |op| u32::from(op.result_count));
    base..base + count
}

/// A stage's scratch as Metal lays it ([`launches`]): what the kernels'
/// offsets, `FireFacts::a_lane` and the stride all read, the same at
/// every call. A view takes its source's offset.
fn layout_of(
    plan: &LaunchStagePlan,
    regions: &[Region],
    descriptors: &[ValueDesc],
) -> std::result::Result<Layout, String> {
    let launches = launches(plan, regions, descriptors).map_err(|why| why.to_string())?;
    lay_out_launches(plan, &launches, descriptors).map_err(|why: TooLarge| format!("{why:?}"))
}

/// The layout [`layout_of`] takes, over `run`: a view's offset its
/// source's.
pub(crate) fn lay_out_launches(
    plan: &LaunchStagePlan,
    run: &Launches<'_>,
    descriptors: &[ValueDesc],
) -> std::result::Result<Layout, TooLarge> {
    let mut laid = eta_exec::lay_out(plan, descriptors, run)?;
    for &node in &run.idle {
        let Some(&source) = plan.ops.get(node as usize).and_then(|op| op.args.first()) else {
            continue;
        };
        for result in results_of(plan, node) {
            if let Some(&at) = laid.values.get(source as usize)
                && let Some(slot) = laid.values.get_mut(result as usize)
            {
                *slot = at;
            }
        }
    }
    Ok(laid)
}
use eta_ir::Dtype;
use eta_ir::op::tags;

use crate::device::ctx::Frame;
use crate::device::Buffer;
use crate::error::{Fault, Result};

use super::compile::{Form, Region, StreamedStep};
use super::held::{self, Cells, Held, Tables};
use super::shared::SharedRing;
use crate::store::Pools;
use crate::store::layout::CHANNEL;
use eta_compiler::codegen::metal::{StepKind, reduce_dispatch_levels};

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
const FIRST_CHANNEL_BUFFER: usize = 7;

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

const STORAGE_F32: u32 = 0;

const STORAGE_BF16: u32 = 1;

const STORAGE_I32: u32 = 2;

/// A seat whose `at` is a table of row addresses (a `ulong` a row), each
/// row bf16: a lane-0 read whose rows are seated at the instance's lanes,
/// row r at lane r's seat.
const STORAGE_BF16_ROWS: u32 = 3;

/// As [`STORAGE_BF16_ROWS`], each row f32.
const STORAGE_F32_ROWS: u32 = 4;

/// The row tables a stage of `plan` takes: one for each float output read,
/// `rows` addresses each (the load's readout rows a lane bound them).
fn row_table_bytes(plan: &LaunchStagePlan, rows: u32) -> u64 {
    let reads = plan
        .ops
        .iter()
        .filter(|op| op.tag == tags::OUTPUT && op.dtype != Dtype::I32)
        .count() as u64;
    (reads * u64::from(rows.max(1)) * 8).max(8)
}

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

    /// The ring's cells: `capacity + 1` of them.
    #[must_use]
    pub fn cell_count(&self) -> u64 {
        u64::from(self.capacity) + 1
    }

    /// The `Channel` slots this ring takes with its cells packed a slot, or
    /// `None` where its cell is larger than a slot (it then takes a run).
    #[must_use]
    pub fn ring_slots(&self, slot_bytes: u64) -> Option<u32> {
        held::ring_slots(self.cell_count(), self.cell_stride() as u64, slot_bytes)
    }

    /// Zeroed cells for this ring in `slots`, the `Channel` slot refs the
    /// runtime took for it. A ring whose cells pass a slot lies in one run,
    /// which lies in one chunk: a ring past that is refused, never held
    /// beside the pool.
    pub fn cells(&self, pools: &Pools, slots: &[u32]) -> Result<Cells> {
        let stride = self.cell_stride() as u64;
        let slot = pools.slot_bytes(CHANNEL)?;
        let per_chunk = pools.slots_per_chunk(CHANNEL)?;
        let fits = self.ring_slots(slot).is_some()
            || held::ring_run(self.cell_count(), stride, slot, per_chunk).is_some();
        if fits {
            return Cells::pooled(pools, CHANNEL, slots, self.cell_count(), stride);
        }
        Err(Fault::program(
            "program::launch",
            format!(
                "a ring of {} cells of {stride} bytes: its cells pass a {slot}-byte channel slot, \
                 so it lies in one run of adjacent slots, and the run's {} bytes pass a chunk \
                 of {per_chunk} slots; declare fewer cells or a smaller cell",
                self.cell_count(),
                self.cell_count() * stride,
            ),
        ))
    }
}

#[derive(Debug)]
pub struct Rings {
    /// Each channel's cells this instance holds itself; `None` for an
    /// adopted shared ring, whose cells are the ring's.
    own: Vec<Option<Cells>>,
    shapes: Vec<ChannelShape>,
    shared: Vec<Option<Arc<SharedRing>>>,
}

impl Rings {
    /// `rings`: the `Channel` slots the runtime took for each ring this
    /// instance holds itself, by channel.
    pub fn allocate(
        shapes: &[ChannelShape],
        adopted: &[Option<Arc<SharedRing>>],
        pools: &Pools,
        rings: &[engine::program::RingSlots],
    ) -> Result<Rings> {
        let mut own = Vec::with_capacity(shapes.len());
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
                own.push(None);
                shared.push(Some(Arc::clone(ring)));
                continue;
            }
            let slots = rings
                .iter()
                .find(|ring| ring.channel as usize == channel)
                .map_or(&[][..], |ring| ring.slots.as_slice());
            own.push(Some(shape.cells(pools, slots)?));
            shared.push(None);
        }
        Ok(Rings {
            own,
            shapes: shapes.to_vec(),
            shared,
        })
    }

    /// Points every pooled ring this instance holds at the chunks its
    /// slots sit in now.
    pub fn rebind(&mut self, pools: &Pools) -> Result<()> {
        self.own
            .iter_mut()
            .flatten()
            .try_for_each(|cells| cells.rebind(pools))
    }

    /// Moves the pooled rings `rings` names to their new slots.
    pub fn retarget(&mut self, pools: &Pools, rings: &[engine::program::RingSlots]) -> Result<()> {
        for ring in rings {
            let cells = self
                .own
                .get_mut(ring.channel as usize)
                .and_then(Option::as_mut)
                .ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!(
                            "channel {} is not a ring this instance holds itself",
                            ring.channel
                        ),
                    )
                })?;
            cells.retarget(pools, &ring.slots)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn shared(&self, channel: usize) -> Option<&Arc<SharedRing>> {
        self.shared.get(channel).and_then(Option::as_ref)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    #[must_use]
    pub fn shape(&self, channel: usize) -> Option<ChannelShape> {
        self.shapes.get(channel).copied()
    }

    fn index(&self, channel: usize, sequence: u64) -> Result<u64> {
        let shape = self.shape(channel).ok_or_else(|| {
            Fault::program(
                "program::launch",
                format!("channel {channel} is not one this instance carries"),
            )
        })?;
        Ok(sequence % (u64::from(shape.capacity) + 1))
    }

    /// Runs `with` on channel `channel`'s cells, its own or its shared
    /// ring's.
    fn with_cells<T>(&self, channel: usize, with: impl FnOnce(&Cells) -> Result<T>) -> Result<T> {
        if let Some(ring) = self.shared(channel) {
            return ring.with_cells(with);
        }
        match self.own.get(channel).and_then(Option::as_ref) {
            Some(cells) => with(cells),
            None => Err(Fault::program(
                "program::launch",
                format!("channel {channel} is not one this instance carries"),
            )),
        }
    }

    /// The buffer and byte the cell of `sequence` lies at, as a kernel
    /// binds it.
    pub fn cell_at(&self, channel: usize, sequence: u64) -> Result<(Buffer, u64)> {
        let index = self.index(channel, sequence)?;
        self.with_cells(channel, |cells| {
            let (held, at) = cells.locate(index)?;
            Ok((held.buffer().clone(), held.offset() + at))
        })
    }

    pub fn write_cell(&mut self, channel: usize, sequence: u64, bytes: &[u8]) -> Result<()> {
        let index = self.index(channel, sequence)?;
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
        if let Some(ring) = self.shared(channel) {
            return ring.write_cell(index, bytes);
        }
        match self.own.get_mut(channel).and_then(Option::as_mut) {
            Some(cells) => cells.write(index, bytes),
            None => Err(Fault::program(
                "program::launch",
                format!("channel {channel} is not one this instance carries"),
            )),
        }
    }

    pub fn read_cell(&self, channel: usize, sequence: u64) -> Result<Vec<u8>> {
        let index = self.index(channel, sequence)?;
        let width = self.shape(channel).map_or(0, |shape| shape.cell_bytes());
        let mut cell = vec![0u8; width];
        self.with_cells(channel, |cells| cells.read(index, &mut cell))?;
        Ok(cell)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    pub head: u64,
    pub tail: u64,
}

#[derive(Debug, Clone)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Slot {
    base: Buffer,
    offset: u64,
    width: u32,
    /// Its row pitch, in elements, and how it is stored.
    stride: u32,
    storage: u32,
    /// The rows from `offset`, a `stride` apart, that are its lane's own:
    /// no read of it reaches past them into another lane's (another
    /// request's) rows.
    rows_held: u32,
}

impl Slot {
    /// The GPU address of row `row` of this seat.
    fn row_at(&self, row: u32) -> Result<u64> {
        let element = if matches!(self.storage, STORAGE_BF16 | STORAGE_BF16_ROWS) { 2 } else { 4 };
        Ok(address_of(&self.base, self.offset)?
            + u64::from(row) * u64::from(self.stride.max(self.width)) * element)
    }
}

/// How a stage reads an output slot: through the readout plane, or its
/// own table entry as floats or as ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Read {
    Plane,
    Floats,
    Ids,
}

/// An output table entry as the kernels read it (`M1OutputSeat`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct OutputSeat {
    at: u64,
    storage: u32,
    stride: u32,
    /// The rows from `at` that are its lane's own (a table's: its rows).
    rows: u32,
    reserved: u32,
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
    reserved0: u32,
    reserved: u32,
}

#[derive(Debug)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Grouped {
    /// The lane table, the flags, indices and rows, and each fused
    /// region's layout: the instance's, in its `Instance` slots.
    table: Held,
    shape: LaneShape,
    layouts: Vec<Held>,
    bindings: Held,
    pending_flags: Held,
    lane_indices: Held,
    /// The lane's scratch base, a device address
    /// ([`LANE_SCRATCH`](eta_compiler::codegen::lane_scratch::LANE_SCRATCH)), written where a fire seats its scratch.
    lane_scratch: Held,
    row_meta: Held,
    row_indices: Held,
    readout_rows: u32,
    /// Each readout row's first element on the readout plane, from
    /// `logits_base`: a row a pitch where the lane's rows lie one after
    /// another; where the rows are seated at the instance's lanes, each
    /// row's own, wherever it lies.
    row_at: Vec<u32>,
    layout_words: Vec<GroupLayout>,
    record: LaneRecord,
}

const GROUPED_LANES: u32 = 1;

const THE_LANE: u32 = 0;

impl Grouped {
    /// Every table, in the order [`Prepared::table_sizes`] states them.
    fn helds(&mut self) -> impl Iterator<Item = &mut Held> {
        [
            &mut self.bindings,
            &mut self.table,
            &mut self.pending_flags,
            &mut self.lane_indices,
            &mut self.lane_scratch,
            &mut self.row_meta,
            &mut self.row_indices,
        ]
        .into_iter()
        .chain(self.layouts.iter_mut())
    }

    /// The lane record's commit slot names the status word where it lies
    /// now, after the instance's slots moved.
    fn recommit(&mut self, status: &Held) -> Result<()> {
        self.record.commit_slot = status.address_at(0).ok_or_else(|| {
            Fault::program(
                "program::launch",
                "the status word has no GPU address, so a grouped kernel would have \
                 nothing to write its verdict through",
            )
        })?;
        self.write_record()
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        plan: &LaunchStagePlan,
        regions: &[Region],
        descriptors: &[ValueDesc],
        shape: LaneShape,
        status: &Held,
        extents: Extents,
        (readout_rows, rows_bound): (u32, u32),
        tables: &mut Tables<'_>,
    ) -> Result<Option<Grouped>> {
        if !plan.needs.grouped_valid {
            return Ok(None);
        }
        if readout_rows > rows_bound.max(1) {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "this bind reads {readout_rows} readout rows a lane, and its tables are \
                     laid for {} (the rows its extents state, `Plane::rows_for`)",
                    rows_bound.max(1)
                ),
            ));
        }
        let stride = u32::try_from(plan.channel_bindings.len())
            .map_err(|_| Fault::program("program::launch", "more channels than a u32 can count"))?;
        let binding_bytes: Vec<u8> = (0..GROUPED_LANES)
            .flat_map(|_| plan.channel_bindings.iter().copied())
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut bindings = tables.take(binding_bytes.len().max(4) as u64)?;
        bindings.write(0, &binding_bytes)?;

        let Some(bytes) = shape.bytes() else {
            return Err(Fault::program(
                "program::launch",
                "a lane table whose size does not fit a u64: the channel count and the \
                 lane count multiply past what an allocation can be",
            ));
        };
        let mut table = tables.take(bytes)?;
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

        let flags = u64::from(GROUPED_LANES) * u64::from(shape.channel_slots_per_lane);
        let pending_flags = tables.take(flags.max(1))?;

        let index_bytes: Vec<u8> = (0..GROUPED_LANES).flat_map(u32::to_le_bytes).collect();
        let mut lane_indices = tables.take(index_bytes.len() as u64)?;
        lane_indices.write(0, &index_bytes)?;
        let lane_scratch = tables.take(LaneScratch::bytes(GROUPED_LANES))?;

        let row_meta = tables.take((GROUPED_LANES as u64) * size_of::<RowMeta>() as u64)?;
        let row_indices = tables.take(u64::from(rows_bound.max(1)) * size_of::<u32>() as u64)?;

        let scratch_layout = layout_of(plan, regions, descriptors).map_err(|why| {
            Fault::program(
                "program::launch",
                format!("this fire's scratch does not fit: {why}"),
            )
        })?;
        let base = GroupLayout {
            lane_count: GROUPED_LANES,
            value_count: u32::try_from(plan.value_types.len()).unwrap_or(u32::MAX),
            scratch_stride: lane_word(LANE_SCRATCH, scratch_layout.total)?,
            temporary_offset: lane_word(LANE_TEMPORARY, scratch_layout.temporary)?,
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
            let mut buffer = tables.take(size_of::<GroupLayout>() as u64)?;
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
            lane_scratch,
            row_meta,
            row_indices,
            readout_rows,
            row_at: Vec::new(),
            layout_words,
            record,
        }))
    }

    fn set_vocab(&mut self, vocab: u32) -> Result<()> {
        for (words, buffer) in self.layout_words.iter_mut().zip(self.layouts.iter_mut()) {
            words.vocab = vocab;
            buffer.write(0, &record_bytes(words))?;
        }
        Ok(())
    }

    fn set_rows(&mut self) -> Result<()> {
        let bytes: Vec<u8> = self.row_at.iter().copied().flat_map(u32::to_le_bytes).collect();
        if !bytes.is_empty() {
            self.row_indices.write(0, &bytes)?;
        }
        let meta = RowMeta {
            offset: 0,
            count: self.readout_rows,
            ..RowMeta::default()
        };
        self.row_meta.write(0, &record_bytes(&meta))
    }

    fn set_logits(&mut self, base: u64, row_offset: u32, row_count: u32) -> Result<()> {
        self.record.logits_base = base;
        self.record.logits_row_offset = row_offset;
        self.record.logits_row_count = row_count;
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

const LANE_SCRATCH: &str = "bytes of one lane's value scratch (the kernels' `uint scratch_stride`)";
const LANE_TEMPORARY: &str =
    "bytes before one lane's temporaries (the kernels' `uint temporary_offset`)";

/// A lane's scratch extent as the generated kernels' layout word holds it:
/// a `uint`, refused past one. A lane's base is its own entry of the lane
/// scratch table, a device address, so no lane's bytes are reached as an
/// offset from another's.
fn lane_word(what: &'static str, bytes: u64) -> Result<u32> {
    u32::try_from(bytes).map_err(|_| Fault::Ceiling {
        what,
        need: bytes,
        have: u64::from(u32::MAX),
    })
}

/// Where a fire's value scratch lies: the activation span's guest region,
/// laid per fire under the model's carve (REDESIGN 3, guest programs), or
/// nowhere before a fire seats it.
#[derive(Debug, Clone)]
pub struct ScratchAt {
    pub buffer: Buffer,
    pub offset: u64,
}

#[derive(Debug)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Prepared {
    /// The status word and the output address table: the instance's, in
    /// its `Instance` slots, as the tables below.
    status: Held,
    /// The bind-time tables (the instance's state, REDESIGN step 4): in
    /// `Instance` slots where the runtime took them.
    descriptors: Held,
    params: Held,
    offsets: Held,
    /// The value scratch this fire seated ([`Prepared::seat_scratch`]),
    /// `scratch_bytes` of it.
    scratch: Option<ScratchAt>,
    scratch_bytes: u64,
    /// By output slot ([`eta_ir::op::output_slot`]): how the stage reads it,
    /// what is bound there and its table entry.
    reads: Vec<Option<Read>>,
    outputs: Vec<Option<Slot>>,
    seats: Vec<OutputSeat>,
    addresses: Held,
    /// By output slot: the rows the stage's read there covers, and where
    /// its row table lies in `row_tables` (a float read's).
    read_rows: Vec<u32>,
    row_table_at: Vec<Option<u64>>,
    row_tables: Held,
    /// Rows past 0 of a lane-0 read, seated at the instance's lanes: row r
    /// at lane r's seat, by (lane-0 slot, row).
    row_seats: BTreeMap<(usize, u32), Slot>,
    /// The one slot read through the readout plane.
    readout: Option<usize>,
    /// The stage's names and the slots a lane takes in its table.
    names: Vec<String>,
    output_slots: u32,
    channel_count: u32,
    value_count: u32,
    scratch_stride: u32,
    temporary_offset: u32,
    bindings: Vec<u32>,
    bound: Vec<Bound>,
    grouped: Option<Grouped>,
    descriptor_bytes: Vec<u8>,
    param_bytes: Vec<u8>,
    offset_bytes: Vec<u8>,
    descriptor_table: Vec<ValueDesc>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub(crate) struct StepWord {
    index: u32,
    level: u32,
    groups: u32,
    reserved: u32,
}

/// Test builds (`test-slot-fill`): `len` bytes of guest scratch from `at`
/// set to 0xFF (NaN in every float width) in `frame`, so a value read
/// before its step writes it fails the test that reads it. Elsewhere
/// nothing: every guest value is written before it is read (a streamed
/// step writes each node it publishes, the lay_out gives a value's bytes on
/// only after its last reader, and the kernels that accumulate clear their
/// own buffers), so the scratch goes unfilled.
#[cfg(target_vendor = "apple")]
pub(crate) fn poison_scratch(
    frame: &mut Frame,
    slab: &crate::device::alloc::Slab,
    at: u64,
    len: u64,
) -> Result<()> {
    #[cfg(feature = "test-slot-fill")]
    {
        frame.fill_with(slab, at, len, 0xff)?;
        frame.next_pass()?;
    }
    let _ = (frame, slab, at, len);
    Ok(())
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct StreamedKnobs {
    max_groups: u32,
    repeat: usize,
    repeat_kind: Option<StepKind>,
    limit: usize,
}

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
                    // A row's groups, for every row (the kernel folds the
                    // rows side by side).
                    let chunks = count.div_ceil(32).max(1);
                    let groups = chunks.div_ceil(32 * REDUCE_CHUNKS_PER_GROUP).max(1) * desc.rows.max(1);
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
    readout_rows: u32,
    regions: usize,
}

impl Prepared {
    /// Moves each pooled table to the ref `to` maps its slot to.
    pub fn retarget(&mut self, pools: &Pools, to: &impl Fn(u32) -> Result<u32>) -> Result<()> {
        let grouped = self.grouped.as_mut().into_iter().flat_map(Grouped::helds);
        for held in [
            &mut self.descriptors,
            &mut self.params,
            &mut self.offsets,
            &mut self.status,
            &mut self.addresses,
            &mut self.row_tables,
        ]
        .into_iter()
        .chain(grouped)
        {
            if let Some(slot) = held.slot() {
                held.retarget(pools, to(slot)?)?;
            }
        }
        self.recommit()
    }

    /// Points the pooled bind-time tables at the chunks their slots sit in
    /// now.
    pub fn rebind(&mut self, pools: &Pools) -> Result<()> {
        let grouped = self.grouped.as_mut().into_iter().flat_map(Grouped::helds);
        for held in [
            &mut self.descriptors,
            &mut self.params,
            &mut self.offsets,
            &mut self.status,
            &mut self.addresses,
            &mut self.row_tables,
        ]
        .into_iter()
        .chain(grouped)
        {
            held.rebind(pools)?;
        }
        self.recommit()
    }

    /// Whether the lane table, as the pool holds it, names the status word
    /// where it lies now; `None` for a stage with no lane table.
    #[cfg(test)]
    pub fn commit_slot_names_status(&self) -> Result<Option<bool>> {
        let Some(grouped) = self.grouped.as_ref() else {
            return Ok(None);
        };
        let at = grouped.shape.record_offset(THE_LANE).ok_or_else(|| {
            Fault::program("program::launch", "the one lane is outside the lane table")
        })?;
        let mut word = [0u8; 8];
        grouped.table.read(
            at + std::mem::offset_of!(LaneRecord, commit_slot) as u64,
            &mut word,
        )?;
        Ok(Some(
            Some(u64::from_le_bytes(word)) == self.status.address_at(0),
        ))
    }

    /// The lane record names the status word where it lies now: its GPU
    /// address is baked into the lane table.
    fn recommit(&mut self) -> Result<()> {
        match self.grouped.as_mut() {
            Some(grouped) => grouped.recommit(&self.status),
            None => Ok(()),
        }
    }

    /// The value scratch one lane of a stage of `plan` takes in a fire at
    /// `extents`: its stride, aligned, as [`Prepared::build`] lays it.
    pub fn scratch_for(plan: &LaunchStagePlan, regions: &[Region], extents: &Extents) -> Result<u64> {
        Ok(u64::from(Self::stride_for(plan, regions, extents)?)
            .max(SCRATCH_ALIGN)
            .next_multiple_of(SCRATCH_ALIGN))
    }

    /// One lane's value scratch for a stage of `plan` at `extents`, as its
    /// kernels stride it (`uint scratch_stride`): what a batch's lanes hold
    /// a stride each of.
    pub fn stride_for(plan: &LaunchStagePlan, regions: &[Region], extents: &Extents) -> Result<u32> {
        let descriptors: Vec<ValueDesc> = plan
            .value_types
            .iter()
            .map(|value| {
                describe(value, extents).map_err(|why| {
                    Fault::program(
                        "program::launch",
                        format!("a value's shape does not resolve against these extents: {why:?}"),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let total = layout_of(plan, regions, &descriptors)
            .map_err(|why| {
                Fault::program(
                    "program::launch",
                    format!("a stage's scratch does not fit: {why}"),
                )
            })?
            .total;
        lane_word(LANE_SCRATCH, total)
    }

    /// The bind-time tables a stage of `plan` takes, in the order `build`
    /// takes them from its [`Tables`]: params, descriptors, offsets, the
    /// status word, the output addresses and the float reads' row tables
    /// and, for a grouped stage, the
    /// channel bindings, the lane table (`channels` slots a lane), its
    /// flags, indices, row meta, `rows` readout rows and each fused
    /// region's layout. Counts alone size them, so a program states them
    /// at registration.
    #[must_use]
    pub fn table_sizes(plan: &LaunchStagePlan, channels: u32, rows: u32) -> Vec<u64> {
        let mut sizes = vec![
            (plan.ops.len() * size_of::<OpParams>()).max(size_of::<OpParams>()) as u64,
            (plan.value_types.len() * size_of::<ValueDesc>()).max(1) as u64,
            (plan.value_types.len() * size_of::<u32>()).max(size_of::<u32>()) as u64,
            eta_exec::STATUS_BYTES as u64,
            ((plan.output_pitch as usize).max(1) * size_of::<OutputSeat>()) as u64,
            row_table_bytes(plan, rows),
        ];
        if plan.needs.grouped_valid {
            let shape = LaneShape::of(GROUPED_LANES, channels);
            sizes.extend([
                (GROUPED_LANES as usize * plan.channel_bindings.len() * size_of::<u32>()).max(4)
                    as u64,
                shape.bytes().unwrap_or(u64::MAX),
                (u64::from(GROUPED_LANES) * u64::from(shape.channel_slots_per_lane)).max(1),
                u64::from(GROUPED_LANES) * size_of::<u32>() as u64,
                LaneScratch::bytes(GROUPED_LANES),
                u64::from(GROUPED_LANES) * size_of::<RowMeta>() as u64,
                u64::from(rows.max(1)) * size_of::<u32>() as u64,
            ]);
            sizes.extend(plan.fused.iter().map(|_| size_of::<GroupLayout>() as u64));
        }
        sizes
    }

    pub fn build(
        plan: &LaunchStagePlan,
        regions: &[Region],
        shapes: &[ChannelShape],
        extents: Extents,
        rows: u32,
        tables: &mut Tables<'_>,
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
        let scratch_layout = layout_of(plan, regions, &descriptors).map_err(|why| {
            Fault::program(
                "program::launch",
                format!("this fire's scratch does not fit: {why}"),
            )
        })?;
        let scratch_stride = lane_word(LANE_SCRATCH, scratch_layout.total)?;
        let temporary_offset = lane_word(LANE_TEMPORARY, scratch_layout.temporary)?;

        let mut records = Vec::with_capacity(plan.ops.len());
        let pitch = plan.output_pitch as usize;
        let mut reads: Vec<Option<Read>> = vec![None; pitch];
        let mut read_rows: Vec<u32> = vec![1; pitch];
        let mut row_table_at: Vec<Option<u64>> = vec![None; pitch];
        let mut float_reads = 0u64;
        let mut readout = None;
        let mut readout_rows = 0u32;
        let mut result_base = 0u32;
        for op in &plan.ops {
            let mut record = OpParams::of(op, result_base, plan.output_slots);
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
            if op.tag == tags::OUTPUT {
                let slot = record.intr as usize;
                let plane = plan
                    .value_types
                    .get(record.o0 as usize)
                    .and_then(|value| value.axes.first())
                    == Some(&eta_compiler::plan::Dimension::Symbolic(
                        eta_compiler::plan::SymbolicExtent::SampledRows,
                    ));
                let read = if plane {
                    if readout.is_some_and(|other| other != slot) {
                        return Err(Fault::program(
                            "program::launch",
                            "this stage reads two outputs' readout rows, and the Metal \
                             kernels read one readout plane a stage",
                        ));
                    }
                    readout = Some(slot);
                    readout_rows = readout_rows.max(
                        descriptors
                            .get(record.o0 as usize)
                            .map_or(1, |out| out.rows.max(1)),
                    );
                    Read::Plane
                } else if op.dtype == Dtype::I32 {
                    Read::Ids
                } else {
                    Read::Floats
                };
                if let Some(seat) = reads.get_mut(slot) {
                    *seat = Some(read);
                }
                if let Some(at) = read_rows.get_mut(slot) {
                    *at = descriptors
                        .get(record.o0 as usize)
                        .map_or(1, |out| out.rows.max(1));
                }
                if op.dtype != Dtype::I32 {
                    if let Some(at) = row_table_at.get_mut(slot) {
                        *at = Some(float_reads * u64::from(rows.max(1)) * 8);
                    }
                    float_reads += 1;
                }
            }
            records.push(record);
            result_base += u32::from(op.result_count);
        }

        let param_bytes = records_bytes(&records);
        let mut params = tables
            .take((records.len() * size_of::<OpParams>()).max(size_of::<OpParams>()) as u64)?;
        params.write(0, &param_bytes)?;

        let descriptor_bytes: Vec<u8> = descriptors.iter().flat_map(record_bytes).collect();
        let mut descriptor_buffer = tables.take(descriptor_bytes.len().max(1) as u64)?;
        descriptor_buffer.write(0, &descriptor_bytes)?;

        let offset_bytes: Vec<u8> = scratch_layout
            .values
            .iter()
            .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut offsets = tables.take(offset_bytes.len().max(size_of::<u32>()) as u64)?;
        offsets.write(0, &offset_bytes)?;

        let scratch_bytes = u64::from(scratch_stride)
            .max(SCRATCH_ALIGN)
            .next_multiple_of(SCRATCH_ALIGN);
        let status = tables.take(eta_exec::STATUS_BYTES as u64)?;
        let addresses = tables.take((pitch.max(1) * size_of::<OutputSeat>()) as u64)?;
        let row_tables = tables.take(row_table_bytes(plan, rows))?;

        let lanes = LaneShape::of(
            GROUPED_LANES,
            u32::try_from(shapes.len()).map_err(|_| {
                Fault::program("program::launch", "more channels than a u32 can count")
            })?,
        );
        let grouped = Grouped::build(
            plan,
            regions,
            &descriptors,
            lanes,
            &status,
            extents,
            (readout_rows, rows),
            tables,
        )?;

        Ok(Prepared {
            status,
            descriptors: descriptor_buffer,
            params,
            offsets,
            scratch: None,
            scratch_bytes,
            reads,
            outputs: vec![None; pitch],
            seats: vec![OutputSeat::default(); pitch],
            addresses,
            read_rows,
            row_table_at,
            row_tables,
            row_seats: BTreeMap::new(),
            readout,
            names: plan.names.clone(),
            output_slots: plan.output_slots,
            channel_count,
            value_count,
            scratch_stride,
            temporary_offset,
            bindings: plan.channel_bindings.clone(),
            bound: Vec::new(),
            grouped,
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
            readout_rows: grouped.readout_rows,
            regions: grouped.layouts.len(),
        })
    }

    pub fn refresh(&mut self, rings: &Rings, cursors: &[Cursor]) -> Result<()> {
        self.refresh_cells(rings, cursors)
    }

    /// The value scratch one fire of this stage takes: its stride, aligned.
    #[must_use]
    pub fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }

    /// Seats this fire's value scratch at `at` (the guest region under the
    /// fire's model carve); a later fire seats its own. A grouped kernel
    /// finds it by its address in the lane scratch table.
    pub fn seat_scratch(&mut self, at: ScratchAt) -> Result<()> {
        // Off Apple nothing encodes (a grouped encode is `Deviceless`), and
        // no reservation has an address to write.
        #[cfg(target_vendor = "apple")]
        if let Some(grouped) = self.grouped.as_mut() {
            let base = address_of(&at.buffer, at.offset)?;
            grouped.lane_scratch.write(0, &base.to_le_bytes())?;
        }
        self.scratch = Some(at);
        Ok(())
    }

    /// Where this fire's value scratch lies: what a batch's lane scratch
    /// table holds for this member.
    pub fn scratch_at(&self) -> Result<ScratchAt> {
        self.scratch().cloned()
    }

    fn scratch(&self) -> Result<&ScratchAt> {
        self.scratch.as_ref().ok_or_else(|| {
            Fault::program(
                "program::launch",
                "a stage fired before its fire seated its value scratch in the span",
            )
        })
    }

    /// Test builds: poisons the seated scratch ([`poison_scratch`]) ahead
    /// of the stage's passes.
    #[cfg(target_vendor = "apple")]
    pub fn poison_scratch_on(&mut self, frame: &mut Frame) -> Result<()> {
        let scratch = self.scratch()?.clone();
        poison_scratch(frame, scratch.buffer.slab(), scratch.offset, self.scratch_bytes)
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
            // A ring's cells lie in one buffer: its own, or the pool's view.
            let (slab, committed) = rings.cell_at(channel, cursor.head)?;
            let (_, pending) = rings.cell_at(channel, cursor.tail)?;
            self.bound.push(Bound {
                slab,
                committed,
                pending,
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

    /// Points this stage's reads of output `name` for `lane` at `offset` of
    /// `base`. A lane past 0 that the stage reads no output at is a row of
    /// its lane-0 read where that read covers rows: row r is lane r's
    /// readout row. Returns whether the stage reads the seat; one it does
    /// not is let go here, and the session refuses a seat no stage reads.
    #[allow(clippy::too_many_arguments)]
    /// Whether this stage reads output `name` at `lane` (what
    /// [`Self::bind_output`] would bind), without binding it.
    #[must_use]
    pub fn reads_output(&self, name: &str, lane: u32) -> bool {
        let Some(named) = self.names.iter().position(|named| named == name) else {
            return false;
        };
        let slot = named + (self.output_slots * lane) as usize;
        match self.reads.get(slot).copied().flatten() {
            Some(_) => true,
            None => matches!(self.reads.get(named).copied().flatten(), Some(_) if lane > 0 && lane < self.read_rows[named]),
        }
    }

    pub fn bind_output(
        &mut self,
        name: &str,
        lane: u32,
        base: &Buffer,
        offset: u64,
        dtype: Dtype,
        width: u32,
        stride: u32,
        rows_held: u32,
    ) -> Result<bool> {
        let Some(named) = self.names.iter().position(|named| named == name) else {
            return Ok(false);
        };
        let slot = named + (self.output_slots * lane) as usize;
        let (read, first) = match self.reads.get(slot).copied().flatten() {
            Some(read) => (read, slot),
            None => match self.reads.get(named).copied().flatten() {
                Some(read) if lane > 0 && lane < self.read_rows[named] => (read, named),
                _ => return Ok(false),
            },
        };
        let storage = match (read, dtype) {
            (Read::Plane, Dtype::Bf16) | (Read::Floats, Dtype::Bf16) => STORAGE_BF16,
            (Read::Floats, Dtype::F32) => STORAGE_F32,
            (Read::Ids, Dtype::I32) if first == slot => STORAGE_I32,
            _ => {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "output `{name}` landed as {dtype:?} at lane {lane}, which this \
                         stage's {read:?} read of it has no gather for"
                    ),
                ));
            }
        };
        if read == Read::Plane && stride != width {
            return Err(Fault::program(
                "program::launch",
                format!("output `{name}`'s readout rows are not one row after another"),
            ));
        }
        let seated = Slot {
            base: base.clone(),
            offset,
            width,
            stride,
            storage,
            rows_held,
        };
        if first == slot {
            self.seats[slot] = OutputSeat {
                at: address_of(base, offset)?,
                storage,
                stride,
                rows: rows_held,
                reserved: 0,
            };
            self.addresses.write(
                (slot * size_of::<OutputSeat>()) as u64,
                &record_bytes(&self.seats[slot]),
            )?;
            self.outputs[slot] = Some(seated);
        } else {
            self.row_seats.insert((first, lane), seated);
        }
        self.seat_rows(first)?;
        self.regroup()?;
        Ok(true)
    }

    /// Whether this stage's lane-0 read of output `name` covers the row of
    /// the instance's lane `lane`: that lane's seat is where the row is
    /// read from, bound though the program names no read at that lane.
    #[must_use]
    pub fn covers(&self, name: &str, lane: u32) -> bool {
        self.names.iter().position(|named| named == name).is_some_and(|named| {
            lane > 0
                && self.reads.get(named).copied().flatten().is_some()
                && lane < self.read_rows.get(named).copied().unwrap_or(1)
        })
    }

    /// Why a read of this stage reaches past the rows its seats hold, if
    /// one does: a read of `n` rows takes row `r` from the instance's lane
    /// `r` seat or from rows the seat itself holds, and a row neither holds
    /// would be another lane's, another request's (a guest declaring more
    /// rows than its lane holds read its neighbours' rows on Metal).
    pub fn rows_unbounded(&self) -> Option<String> {
        let readout_rows = self.grouped.as_ref().map_or(0, |grouped| grouped.readout_rows);
        for (slot, seat) in self.outputs.iter().enumerate() {
            let Some(seat) = seat else {
                continue;
            };
            if self.reads.get(slot).copied().flatten().is_none() {
                continue;
            }
            let mut rows = self.read_rows.get(slot).copied().unwrap_or(1).max(1);
            if self.readout == Some(slot) {
                rows = rows.max(readout_rows);
            }
            let name = &self.names[slot % self.output_slots as usize];
            if let Some(row) = (0..rows)
                .find(|&row| row >= seat.rows_held && !self.row_seats.contains_key(&(slot, row)))
            {
                return Some(format!(
                    "a read of {rows} rows of output `{name}`, whose seat holds {} and no lane \
                     {row} seat holds row {row}: past them it would read another lane's rows",
                    seat.rows_held
                ));
            }
        }
        None
    }

    /// Every seat this stage reads, its lanes' rows among them.
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    fn seated(&self) -> impl Iterator<Item = &Slot> {
        self.outputs.iter().flatten().chain(self.row_seats.values())
    }

    /// The row seats of the lane-0 read at `first`, by row; empty where
    /// its rows lie one after another from lane 0's seat.
    fn rows_of(&self, first: usize) -> Vec<(u32, &Slot)> {
        self.row_seats
            .range((first, 0)..(first + 1, 0))
            .map(|(&(_, row), seat)| (row, seat))
            .collect()
    }

    /// Each row's address of the lane-0 read at `first`: row 0 at lane 0's
    /// seat, a row seated at its lane there, a row lane 0 holds on its
    /// pitch, and a row no seat holds (yet: its lane binds later, or never)
    /// at lane 0's own first row. Never another lane's row on lane 0's
    /// pitch: that row is another request's. A read whose rows its seats do
    /// not hold is refused before it launches ([`Self::rows_unbounded`]).
    fn row_addresses(&self, first: usize) -> Result<Option<Vec<u64>>> {
        let seated = self.rows_of(first);
        let Some(zero) = self.outputs.get(first).and_then(Option::as_ref) else {
            return Ok(None);
        };
        if seated.is_empty() {
            return Ok(None);
        }
        (0..self.read_rows[first])
            .map(|row| match seated.iter().find(|(at, _)| *at == row) {
                Some((_, seat)) => seat.row_at(0),
                None if row < zero.rows_held => zero.row_at(row),
                None => zero.row_at(0),
            })
            .collect::<Result<Vec<u64>>>()
            .map(Some)
    }

    /// Seats a float read's rows through its row table where its lanes'
    /// rows are seated apart: the table holds each row's address and the
    /// read's seat names the table ([`STORAGE_BF16_ROWS`]).
    fn seat_rows(&mut self, first: usize) -> Result<()> {
        if self.reads.get(first).copied().flatten() != Some(Read::Floats) {
            return Ok(());
        }
        let Some(rows) = self.row_addresses(first)? else {
            return Ok(());
        };
        let at = self.row_table_at[first].ok_or_else(|| {
            Fault::program("program::launch", "a float read with no row table laid for it")
        })?;
        let zero = self.outputs[first].as_ref().expect("row_addresses saw lane 0's seat");
        let storage = if zero.storage == STORAGE_BF16 { STORAGE_BF16_ROWS } else { STORAGE_F32_ROWS };
        let bytes: Vec<u8> = rows.iter().flat_map(|address| address.to_le_bytes()).collect();
        self.row_tables.write(at, &bytes)?;
        self.seats[first] = OutputSeat {
            at: self.row_tables.address_at(at).ok_or_else(|| {
                Fault::program("program::launch", "a row table with no GPU address")
            })?,
            storage,
            stride: 0,
            rows: u32::try_from(rows.len()).unwrap_or(u32::MAX),
            reserved: 0,
        };
        self.addresses.write(
            (first * size_of::<OutputSeat>()) as u64,
            &record_bytes(&self.seats[first]),
        )
    }

    fn regroup(&mut self) -> Result<()> {
        if self.grouped.is_none() {
            return Ok(());
        }
        let Some(slot) = self.readout else {
            return Ok(());
        };
        let Some(plane) = self.outputs[slot].as_ref() else {
            return Ok(());
        };
        let width = plane.width;
        // Each row's first element on the plane: from lane 0's seat one
        // after another, or, where the lanes' rows are seated apart, each
        // row's own from the lowest.
        let element = |apart: u64| -> Result<u32> {
            u32::try_from(apart / 2).map_err(|_| {
                Fault::program(
                    "program::launch",
                    format!(
                        "output `{}`'s rows lie {apart} bytes apart, past a u32 of \
                         elements the grouped kernels address rows by",
                        self.names[slot % self.output_slots as usize]
                    ),
                )
            })
        };
        let pitch = u64::from(width) * 2;
        let readout_rows = self.grouped.as_ref().map_or(0, |grouped| grouped.readout_rows);
        let (base, rows) = match self.row_addresses(slot)? {
            None => (
                address_of(&plane.base, plane.offset)?,
                (0..u64::from(readout_rows))
                    .map(|row| element(row * pitch))
                    .collect::<Result<Vec<u32>>>()?,
            ),
            Some(addresses) => {
                let low = addresses.iter().copied().min().unwrap_or(0);
                (
                    low,
                    addresses
                        .iter()
                        .map(|&address| element(address - low))
                        .collect::<Result<Vec<u32>>>()?,
                )
            }
        };
        let grouped = self.grouped.as_mut().expect("checked above");
        grouped.set_vocab(width)?;
        grouped.set_logits(base, 0, grouped.readout_rows)?;
        grouped.row_at = rows;
        grouped.set_rows()
    }

    pub fn encode_into(&self, frame: &Frame, region: &Region) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        if crate::encode::profiling() {
            let (_, seconds) = frame.timed(|| self.encode_one_region(frame, region))?;
            if let Some(seconds) = seconds {
                crate::encode::record_kernel(&guest_key(region, 1), seconds);
            }
            return Ok(());
        }
        self.encode_one_region(frame, region)
    }

    fn encode_one_region(&self, frame: &Frame, region: &Region) -> Result<()> {
        match region.form {
            Form::Fused => {
                let scratch = self.scratch()?;
                self.encode_fused(frame, region, &scratch.buffer, scratch.offset)
            }
            Form::Streamed => self.encode_streamed(frame, region),
            Form::Grouped | Form::GroupedLibrary => self.encode_grouped(frame, region),
        }
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn encode_streamed(&self, frame: &Frame, region: &Region) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            let encoder = frame.encoder();
            let (dispatches, threads) = self.bind_streamed(&encoder, region)?;
            dispatch_streamed(&encoder, &dispatches, threads, GROUPED_LANES as usize, region);
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (frame, region);
            Err(Fault::Deviceless)
        }
    }

    /// Binds this member's state for a streamed region on `encoder` and
    /// returns its steps' dispatches (step word, threadgroups) and threads
    /// a threadgroup, for the caller to dispatch: all at once, or one step
    /// at a time interleaved with other members' ([`dispatch_streamed_step`]).
    #[cfg(target_vendor = "apple")]
    pub(crate) fn bind_streamed(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        region: &Region,
    ) -> Result<(Vec<(StepWord, u32)>, usize)> {
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
            encoder.setComputePipelineState(region.pipeline());
            // SAFETY: every reservation is retained by `self`; every offset lies inside its slot.
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.table.raw()),
                    grouped.table.offset() as usize,
                    0,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.descriptors.raw()),
                    self.descriptors.offset() as usize,
                    1,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.params.raw()),
                    self.params.offset() as usize,
                    2,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.offsets.raw()),
                    self.offsets.offset() as usize,
                    3,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.lane_scratch.raw()),
                    grouped.lane_scratch.offset() as usize,
                    eta_compiler::codegen::lane_scratch::LANE_SCRATCH.metal_buffer,
                );
                encoder.setBuffer_offset_atIndex(Some(layout.raw()), layout.offset() as usize, 5);
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.bindings.raw()),
                    grouped.bindings.offset() as usize,
                    6,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.pending_flags.raw()),
                    grouped.pending_flags.offset() as usize,
                    7,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.lane_indices.raw()),
                    grouped.lane_indices.offset() as usize,
                    8,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.row_meta.raw()),
                    grouped.row_meta.offset() as usize,
                    9,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.row_indices.raw()),
                    grouped.row_indices.offset() as usize,
                    10,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.addresses.raw()),
                    self.addresses.offset() as usize,
                    eta_compiler::codegen::metal::M3_INTRINSIC_TABLE_BUFFER,
                );
            }
            let resident = |buffer: &Buffer, usage: MTLResourceUsage| {
                let resource: &ProtocolObject<dyn MTLResource> =
                    ProtocolObject::from_ref(&**buffer.slab());
                encoder.useResource_usage(resource, usage);
            };
            // The lanes reach their scratch by the addresses in the lane
            // scratch table.
            resident(
                &self.scratch()?.buffer,
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            resident(
                self.status.buffer(),
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            for cell in &self.bound {
                resident(&cell.slab, MTLResourceUsage::Read | MTLResourceUsage::Write);
            }
            for held in self.seated() {
                resident(&held.base, MTLResourceUsage::Read);
            }
            let threads = streamed_threads(region.pipeline().maxTotalThreadsPerThreadgroup());
            let temporary_bytes =
                u64::from(self.scratch_stride.saturating_sub(self.temporary_offset));
            Ok((
                streamed_dispatches(
                    &region.steps,
                    &self.descriptor_table,
                    u32::try_from(threads).unwrap_or(1),
                    temporary_bytes,
                ),
                threads,
            ))
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
            use objc2::runtime::ProtocolObject;
            use objc2_metal::{MTLComputeCommandEncoder, MTLResource, MTLResourceUsage, MTLSize};

            let encoder = frame.encoder();
            encoder.setComputePipelineState(region.pipeline());
            for held in self.seated() {
                let resource: &ProtocolObject<dyn MTLResource> =
                    ProtocolObject::from_ref(&**held.base.slab());
                encoder.useResource_usage(resource, MTLResourceUsage::Read);
            }
            let scratch_at = usize::try_from(scratch_at).unwrap_or(usize::MAX);
            // SAFETY: every buffer is retained by `self`/`region`/the batch; every offset was bounds-checked.
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(self.status.raw()),
                    self.status.offset() as usize,
                    0,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.descriptors.raw()),
                    self.descriptors.offset() as usize,
                    1,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.params.raw()),
                    self.params.offset() as usize,
                    2,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.offsets.raw()),
                    self.offsets.offset() as usize,
                    3,
                );
                encoder.setBuffer_offset_atIndex(Some(scratch.raw()), scratch_at, 4);
                encoder.setBuffer_offset_atIndex(
                    Some(scratch.raw()),
                    scratch_at + self.temporary_offset as usize,
                    5,
                );
                if let Some(slot) = self.readout
                    && !self.rows_of(slot).is_empty()
                {
                    return Err(Fault::program(
                        "program::launch",
                        format!(
                            "output `{}`'s rows are seated at the instance's lanes, and \
                             this stage's fused form reads the readout plane's rows one \
                             after another from lane 0's",
                            self.names[slot % self.output_slots as usize]
                        ),
                    ));
                }
                match self.readout.and_then(|slot| self.outputs[slot].as_ref()) {
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
                encoder.setBuffer_offset_atIndex(
                    Some(self.addresses.raw()),
                    self.addresses.offset() as usize,
                    eta_compiler::codegen::metal::M2_INTRINSIC_TABLE_BUFFER,
                );
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
            // SAFETY: every reservation is retained by `self`; every offset lies inside its slot (the kernel strides off `layout`).
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.table.raw()),
                    grouped.table.offset() as usize,
                    0,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.descriptors.raw()),
                    self.descriptors.offset() as usize,
                    1,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.params.raw()),
                    self.params.offset() as usize,
                    2,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.offsets.raw()),
                    self.offsets.offset() as usize,
                    3,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.lane_scratch.raw()),
                    grouped.lane_scratch.offset() as usize,
                    eta_compiler::codegen::lane_scratch::LANE_SCRATCH.metal_buffer,
                );
                encoder.setBuffer_offset_atIndex(Some(layout.raw()), layout.offset() as usize, 5);
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.bindings.raw()),
                    grouped.bindings.offset() as usize,
                    6,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.pending_flags.raw()),
                    grouped.pending_flags.offset() as usize,
                    7,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.lane_indices.raw()),
                    grouped.lane_indices.offset() as usize,
                    8,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.row_meta.raw()),
                    grouped.row_meta.offset() as usize,
                    9,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(grouped.row_indices.raw()),
                    grouped.row_indices.offset() as usize,
                    10,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.addresses.raw()),
                    self.addresses.offset() as usize,
                    eta_compiler::codegen::metal::M3_INTRINSIC_TABLE_BUFFER,
                );
            }

            let resident = |buffer: &Buffer, usage: MTLResourceUsage| {
                let resource: &ProtocolObject<dyn MTLResource> =
                    ProtocolObject::from_ref(&**buffer.slab());
                encoder.useResource_usage(resource, usage);
            };
            // The lanes reach their scratch by the addresses in the lane
            // scratch table.
            resident(
                &self.scratch()?.buffer,
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            resident(
                self.status.buffer(),
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            for cell in &self.bound {
                resident(&cell.slab, MTLResourceUsage::Read | MTLResourceUsage::Write);
            }
            for held in self.seated() {
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

    /// Fires `region` in a frame of its own; `zero` (a stage's first
    /// region) first poisons the seated scratch in test builds.
    /// `region` encoded into `frame` as a pass of its own after what the
    /// frame holds (a stage's first poisons its scratch first): in order
    /// with the regions before it, as a command buffer each was.
    pub fn encode_region(&mut self, frame: &mut Frame, region: &Region, zero: bool) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            if zero {
                self.poison_scratch_on(frame)?;
            }
            frame.next_pass()?;
            self.encode_into(frame, region)?;
            frame.next_pass()?;
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (frame, region, zero);
            Err(Fault::Deviceless)
        }
    }

    /// The status word's region, as a completion handler reads it
    /// ([`super::session::EpilogueCheck`]).
    #[must_use]
    pub(crate) fn status_region(&self) -> Held {
        self.status.clone()
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
    /// Every table, a region of the span's batch area ([`Batch::lay`]):
    /// this batch's frame in flight writes them, and no other frame reads
    /// them while it is airborne.
    table: SpanRegion,
    descriptors: SpanRegion,
    params: SpanRegion,
    offsets: SpanRegion,
    bindings: SpanRegion,
    pending_flags: SpanRegion,
    lane_indices: SpanRegion,
    /// Each lane's scratch base, a device address ([`LANE_SCRATCH`](eta_compiler::codegen::lane_scratch::LANE_SCRATCH)),
    /// written where a fire seats its scratch.
    lane_scratch: SpanRegion,
    row_meta: SpanRegion,
    row_indices: SpanRegion,
    /// This fire's lanes' value scratch, each its member's own
    /// ([`Batch::seat_scratch`]): what the lane scratch table addresses.
    scratch: Vec<ScratchAt>,
    addresses: SpanRegion,
    table_bytes: u64,
    layouts: Vec<SpanRegion>,
    layout_words: Vec<GroupLayout>,
    rows_per_lane: u32,
    /// What [`Batch::lay`] writes again wherever the batch is laid: the
    /// template's tables, a copy a lane where they are per lane.
    descriptor_bytes: Vec<u8>,
    param_bytes: Vec<u8>,
    offset_bytes: Vec<u8>,
    binding_bytes: Vec<u8>,
}

/// A batch table's region of the span.
type SpanRegion = crate::inputs::Region;

/// A batch's tables' bytes, in the order [`Batch::lay`] lays them: the lane
/// table, descriptors, params, offsets, bindings, pending flags, lane
/// indices, lane scratch bases, row meta, row indices, output addresses and
/// each fused region's layout.
#[allow(clippy::too_many_arguments)]
fn batch_pieces(
    lanes: u32,
    channel_slots: u32,
    (descriptor, param, offset, binding): (u64, u64, u64, u64),
    rows: u32,
    table_bytes: u64,
    fused: usize,
) -> Vec<u64> {
    let lanes64 = u64::from(lanes);
    let mut pieces = vec![
        LaneShape::of(lanes, channel_slots)
            .bytes()
            .unwrap_or(u64::MAX),
        (descriptor * lanes64).max(4),
        (param * lanes64).max(4),
        offset.max(4),
        (binding * lanes64).max(4),
        (lanes64 * u64::from(channel_slots)).max(1),
        lanes64 * size_of::<u32>() as u64,
        LaneScratch::bytes(lanes),
        lanes64 * size_of::<RowMeta>() as u64,
        (lanes64 * u64::from(rows) * size_of::<u32>() as u64).max(4),
        lanes64 * table_bytes,
    ];
    pieces.extend((0..fused).map(|_| size_of::<GroupLayout>() as u64));
    pieces
}

/// The span bytes `pieces` take, each on the arena's boundary.
fn batch_area(pieces: &[u64]) -> u64 {
    pieces.iter().map(|&bytes| crate::arena::room(bytes)).sum()
}

impl Batch {
    /// The span bytes this batch's tables take where it is laid.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.regions()
            .map(|region| crate::arena::room(region.bytes()))
            .sum()
    }

    /// The span bytes a batch of `lanes` lanes of a stage of `plan` takes,
    /// its instances carrying `channels` channels and reading at most
    /// `rows` readout rows a lane: what a fire term states before any is
    /// laid; 0 for a stage with no grouped seat.
    #[must_use]
    pub fn bytes_for(
        plan: &LaunchStagePlan,
        channels: u32,
        lanes: u32,
        rows: u32,
    ) -> u64 {
        if !plan.needs.grouped_valid {
            return 0;
        }
        let values = plan.value_types.len() as u64;
        batch_area(&batch_pieces(
            lanes.max(1),
            LaneShape::of(GROUPED_LANES, channels).channel_slots_per_lane,
            (
                values * size_of::<ValueDesc>() as u64,
                (plan.ops.len() * size_of::<OpParams>()) as u64,
                values * size_of::<u32>() as u64,
                (plan.channel_bindings.len() * size_of::<u32>()) as u64,
            ),
            rows.max(1),
            ((plan.output_pitch as usize).max(1) * size_of::<OutputSeat>()) as u64,
            plan.fused.len(),
        ))
    }

    fn regions(&self) -> impl Iterator<Item = &SpanRegion> {
        [
            &self.table,
            &self.descriptors,
            &self.params,
            &self.offsets,
            &self.bindings,
            &self.pending_flags,
            &self.lane_indices,
            &self.lane_scratch,
            &self.row_meta,
            &self.row_indices,
            &self.addresses,
        ]
        .into_iter()
        .chain(&self.layouts)
    }

    fn regions_mut(&mut self) -> impl Iterator<Item = &mut SpanRegion> {
        [
            &mut self.table,
            &mut self.descriptors,
            &mut self.params,
            &mut self.offsets,
            &mut self.bindings,
            &mut self.pending_flags,
            &mut self.lane_indices,
            &mut self.lane_scratch,
            &mut self.row_meta,
            &mut self.row_indices,
            &mut self.addresses,
        ]
        .into_iter()
        .chain(self.layouts.iter_mut())
    }

    /// A batch of `lanes` lanes of `template`'s stage, its tables sized and
    /// not yet laid ([`Batch::lay`]): nothing binds or writes them before.
    pub fn plan(template: &Prepared, lanes: u32, empty: &Buffer) -> Result<Batch> {
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
        let table_bytes = (template.seats.len().max(1) * size_of::<OutputSeat>()) as u64;
        let shape = LaneShape::of(lanes, grouped.shape.channel_slots_per_lane);
        if shape.bytes().is_none() {
            return Err(Fault::program(
                "program::launch",
                "a batch lane table whose size does not fit a u64",
            ));
        }
        let binding_bytes: Vec<u8> = template
            .bindings
            .iter()
            .copied()
            .flat_map(u32::to_le_bytes)
            .collect();
        let rows_per_lane = grouped.readout_rows;
        let pieces = batch_pieces(
            lanes,
            grouped.shape.channel_slots_per_lane,
            (
                template.descriptor_bytes.len() as u64,
                template.param_bytes.len() as u64,
                template.offset_bytes.len() as u64,
                binding_bytes.len() as u64,
            ),
            rows_per_lane,
            table_bytes,
            grouped.layout_words.len(),
        );
        let mut regions = pieces
            .iter()
            .map(|&bytes| SpanRegion::unplaced(bytes, empty.clone()));
        let mut next = || regions.next().expect("a region a piece");
        Ok(Batch {
            key,
            lanes,
            shape,
            table: next(),
            descriptors: next(),
            params: next(),
            offsets: next(),
            bindings: next(),
            pending_flags: next(),
            lane_indices: next(),
            lane_scratch: next(),
            row_meta: next(),
            row_indices: next(),
            scratch: Vec::new(),
            addresses: next(),
            table_bytes,
            layouts: (0..grouped.layout_words.len()).map(|_| next()).collect(),
            layout_words: grouped.layout_words.clone(),
            rows_per_lane,
            descriptor_bytes: template.descriptor_bytes.clone(),
            param_bytes: template.param_bytes.clone(),
            offset_bytes: template.offset_bytes.clone(),
            binding_bytes,
        })
    }

    /// Lays the batch's tables from `at` of `view` (the span's buffer)
    /// upward, written through `host` (`(at, bytes)` to its host runs), and
    /// writes what a batch holds from its first fire; the caller drained
    /// every frame that read where it was. Returns where the next lies.
    pub fn lay(
        &mut self,
        view: &Buffer,
        at: u64,
        host: &dyn Fn(u64, u64) -> Result<Vec<(u64, u64, std::ptr::NonNull<u8>)>>,
    ) -> Result<u64> {
        let mut next = at;
        for region in self.regions_mut() {
            let bytes = region.bytes();
            region.place(view.clone(), next);
            region.rehost(if bytes > 0 {
                host(next, bytes)?
            } else {
                Vec::new()
            })?;
            next += crate::arena::room(bytes);
        }
        self.table.write(
            0,
            &record_bytes(&LaneHeader {
                abi_version: LANE_ABI_VERSION,
                lane_count: self.lanes,
                channel_slots_per_lane: self.shape.channel_slots_per_lane,
                flags: 0,
            }),
        )?;
        for lane in 0..u64::from(self.lanes) {
            let descriptor = self.descriptor_bytes.len() as u64;
            self.descriptors
                .write(lane * descriptor, &self.descriptor_bytes)?;
            let param = self.param_bytes.len() as u64;
            self.params.write(lane * param, &self.param_bytes)?;
            let binding = self.binding_bytes.len() as u64;
            self.bindings.write(lane * binding, &self.binding_bytes)?;
        }
        self.offsets.write(0, &self.offset_bytes)?;
        let index_bytes: Vec<u8> = (0..self.lanes).flat_map(u32::to_le_bytes).collect();
        self.lane_indices.write(0, &index_bytes)?;
        for (words, region) in self.layout_words.iter().zip(self.layouts.iter_mut()) {
            region.write(0, &record_bytes(words))?;
        }
        Ok(next)
    }

    /// Writes each lane's scratch address into the lane scratch table:
    /// lane `i` fires in `lanes[i]`, its member's own scratch, wherever the
    /// pool laid it. No lane's bytes are reached from another's.
    pub fn seat_scratch(&mut self, lanes: &[ScratchAt]) -> Result<()> {
        if lanes.len() as u64 > u64::from(self.lanes) {
            return Err(Fault::Ceiling {
                what: "lanes of a batch's lane scratch table",
                need: lanes.len() as u64,
                have: u64::from(self.lanes),
            });
        }
        // Off Apple nothing encodes (a batch's encode is `Deviceless`), and
        // no reservation has an address to write.
        #[cfg(target_vendor = "apple")]
        {
            let mut bases = Vec::with_capacity(LaneScratch::bytes(self.lanes) as usize);
            for at in lanes {
                bases.extend(address_of(&at.buffer, at.offset)?.to_le_bytes());
            }
            self.lane_scratch.write(0, &bases)?;
        }
        self.scratch = lanes.to_vec();
        Ok(())
    }

    /// This fire's lanes' scratch, one a member: refused where the fire
    /// seated other than `count` lanes.
    #[cfg(target_vendor = "apple")]
    fn lanes_scratch(&self, count: u32) -> Result<&[ScratchAt]> {
        if self.scratch.len() as u64 != u64::from(count) {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "a batch fires {count} lanes and seated {} lanes' scratch",
                    self.scratch.len()
                ),
            ));
        }
        Ok(&self.scratch)
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
        #[cfg(target_vendor = "apple")]
        let stride = u64::from(self.key.scratch_stride);
        self.pending_flags.zero_span(
            0,
            u64::from(count) * u64::from(self.shape.channel_slots_per_lane),
        )?;
        #[cfg(target_vendor = "apple")]
        for lane in self.lanes_scratch(count)? {
            poison_scratch(frame, lane.buffer.slab(), lane.offset, stride)?;
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
            self.addresses.write(
                u64::from(lane) * self.table_bytes,
                &records_bytes(&member.seats),
            )?;
            let row_base = lane * self.rows_per_lane;
            let meta = RowMeta {
                offset: row_base,
                count: self.rows_per_lane,
                ..RowMeta::default()
            };
            self.row_meta.write(
                u64::from(lane) * size_of::<RowMeta>() as u64,
                &record_bytes(&meta),
            )?;
            let rows: Vec<u8> = grouped
                .row_at
                .iter()
                .copied()
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
            #[cfg(target_vendor = "apple")]
            if crate::encode::profiling() {
                let (_, seconds) =
                    frame.timed(|| self.encode_batched_region(frame, region, members, count))?;
                if let Some(seconds) = seconds {
                    crate::encode::record_kernel(&guest_key(region, count), seconds);
                }
                continue;
            }
            self.encode_batched_region(frame, region, members, count)?;
        }
        Ok(())
    }

    fn encode_batched_region(
        &self,
        frame: &Frame,
        region: &Region,
        members: &[&mut Prepared],
        count: u32,
    ) -> Result<()> {
        {
            match region.form {
                Form::Fused => {
                    // A fused region fires each member in its own scratch.
                    for member in members.iter() {
                        let own = member.scratch_at()?;
                        member.encode_fused(frame, region, &own.buffer, own.offset)?;
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
                encoder.setBuffer_offset_atIndex(
                    Some(self.table.view().raw()),
                    self.table.at(0) as usize,
                    0,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.descriptors.view().raw()),
                    self.descriptors.at(0) as usize,
                    1,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.params.view().raw()),
                    self.params.at(0) as usize,
                    2,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.offsets.view().raw()),
                    self.offsets.at(0) as usize,
                    3,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.lane_scratch.view().raw()),
                    self.lane_scratch.at(0) as usize,
                    eta_compiler::codegen::lane_scratch::LANE_SCRATCH.metal_buffer,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(layout.view().raw()),
                    layout.at(0) as usize,
                    5,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.bindings.view().raw()),
                    self.bindings.at(0) as usize,
                    6,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.pending_flags.view().raw()),
                    self.pending_flags.at(0) as usize,
                    7,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.lane_indices.view().raw()),
                    self.lane_indices.at(0) as usize,
                    8,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.row_meta.view().raw()),
                    self.row_meta.at(0) as usize,
                    9,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.row_indices.view().raw()),
                    self.row_indices.at(0) as usize,
                    10,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.addresses.view().raw()),
                    self.addresses.at(0) as usize,
                    eta_compiler::codegen::metal::M3_INTRINSIC_TABLE_BUFFER,
                );
            }
            let resident = |buffer: &Buffer, usage: MTLResourceUsage| {
                let resource: &ProtocolObject<dyn MTLResource> =
                    ProtocolObject::from_ref(&**buffer.slab());
                encoder.useResource_usage(resource, usage);
            };
            // The lanes reach their scratch by the addresses in the lane
            // scratch table: each member's own, wherever it lies.
            for lane in &self.scratch {
                resident(&lane.buffer, MTLResourceUsage::Read | MTLResourceUsage::Write);
            }
            for member in members {
                resident(
                    member.status.buffer(),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                for cell in &member.bound {
                    resident(&cell.slab, MTLResourceUsage::Read | MTLResourceUsage::Write);
                }
                for held in member.seated() {
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
                &encoder,
                &streamed_dispatches(
                    &region.steps,
                    &template.descriptor_table,
                    u32::try_from(threads).unwrap_or(1),
                    temporary_bytes,
                ),
                threads,
                count as usize,
                region,
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
                encoder.setBuffer_offset_atIndex(
                    Some(self.table.view().raw()),
                    self.table.at(0) as usize,
                    0,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.descriptors.view().raw()),
                    self.descriptors.at(0) as usize,
                    1,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.params.view().raw()),
                    self.params.at(0) as usize,
                    2,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.offsets.view().raw()),
                    self.offsets.at(0) as usize,
                    3,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.lane_scratch.view().raw()),
                    self.lane_scratch.at(0) as usize,
                    eta_compiler::codegen::lane_scratch::LANE_SCRATCH.metal_buffer,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(layout.view().raw()),
                    layout.at(0) as usize,
                    5,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.bindings.view().raw()),
                    self.bindings.at(0) as usize,
                    6,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.pending_flags.view().raw()),
                    self.pending_flags.at(0) as usize,
                    7,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.lane_indices.view().raw()),
                    self.lane_indices.at(0) as usize,
                    8,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.row_meta.view().raw()),
                    self.row_meta.at(0) as usize,
                    9,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.row_indices.view().raw()),
                    self.row_indices.at(0) as usize,
                    10,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(self.addresses.view().raw()),
                    self.addresses.at(0) as usize,
                    eta_compiler::codegen::metal::M3_INTRINSIC_TABLE_BUFFER,
                );
            }
            let resident = |buffer: &Buffer, usage: MTLResourceUsage| {
                let resource: &ProtocolObject<dyn MTLResource> =
                    ProtocolObject::from_ref(&**buffer.slab());
                encoder.useResource_usage(resource, usage);
            };
            // The lanes reach their scratch by the addresses in the lane
            // scratch table: each member's own, wherever it lies.
            for lane in &self.scratch {
                resident(&lane.buffer, MTLResourceUsage::Read | MTLResourceUsage::Write);
            }
            for member in members {
                resident(
                    member.status.buffer(),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                for cell in &member.bound {
                    resident(&cell.slab, MTLResourceUsage::Read | MTLResourceUsage::Write);
                }
                for held in member.seated() {
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

/// One streamed step of a member bound on `encoder` ([`Prepared::bind_streamed`]).
#[cfg(target_vendor = "apple")]
pub(crate) fn dispatch_streamed_step(
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    step: (StepWord, u32),
    threads: usize,
    region: &Region,
) {
    dispatch_streamed(encoder, &[step], threads, GROUPED_LANES as usize, region);
}

#[cfg(target_vendor = "apple")]
fn dispatch_streamed(
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    dispatches: &[(StepWord, u32)],
    threads: usize,
    lanes: usize,
    region: &Region,
) {
    use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
    // Each step on its own pipeline where the region was built one a step
    // (only its case compiled in), else every step on the region's.
    let own = region.module.step_pipelines();
    let mut bound: Option<usize> = None;
    for (word, groups) in dispatches {
        let step = word.index as usize;
        let want = (step < own.len()).then_some(step);
        if want != bound {
            match want {
                Some(at) => encoder.setComputePipelineState(&own[at]),
                None => encoder.setComputePipelineState(region.pipeline()),
            }
            bound = want;
        }
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

    /// A lane's scratch past a `uint` is refused by name, not clamped: the
    /// kernels' layout word would hold another stride than the one laid.
    #[test]
    fn a_lane_scratch_past_a_uint_is_refused_by_name() {
        assert_eq!(
            lane_word(LANE_SCRATCH, u64::from(u32::MAX)).ok(),
            Some(u32::MAX)
        );
        let refused = lane_word(LANE_SCRATCH, 1 << 32).expect_err("past a uint");
        assert!(
            matches!(refused, Fault::Ceiling { what, need, .. }
                if what == LANE_SCRATCH && need == 1 << 32),
            "{refused}"
        );
    }

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
            own: Vec::new(),
            shapes: Vec::new(),
            shared: Vec::new(),
        };
        let said = rings
            .cell_at(2, 0)
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
                "uint reserved0",
                "uint reserved"
            ],
            "`RowMeta` in this file is the host half of the emitted `M3RowMeta`, \
             and the two have parted"
        );
        assert_eq!(size_of::<RowMeta>(), 4 * size_of::<u32>());
        let runtime = eta_compiler::codegen::metal::RUNTIME_TEMPLATE;
        assert!(
            runtime.contains(
                "struct M1OutputSeat {\n  ulong at;\n  uint storage;\n  uint stride;\n  uint rows;\n  uint reserved;\n};"
            ),
            "`OutputSeat` in this file is the host half of the emitted `M1OutputSeat`"
        );
        assert_eq!(size_of::<OutputSeat>(), 24);
    }

    fn the_grouped_samplers_take_the_bindings_this_file_writes() {
        const BOUND: [&str; 11] = [
            "lane_bytes",
            "all_descriptors",
            "params",
            "offsets",
            eta_compiler::codegen::lane_scratch::LANE_SCRATCH.name,
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


/// The kernel profile's row for a guest region: its entry, form, steps
/// and the lanes it ran for.
#[cfg(target_vendor = "apple")]
fn guest_key(region: &Region, lanes: u32) -> String {
    format!(
        "guest {} {:?} steps={} lanes={lanes}",
        region.module.entry(),
        region.form,
        region.steps.len()
    )
}
