#![allow(unsafe_code)]
use eta_compiler::codegen::cuda_steps;
use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_compiler::plan::{LibraryOp, RegionKind};
use eta_exec::tables::SteppedTables;
use eta_exec::{
    Extents, LANE_HEADER_BYTES, LANE_RECORD_BYTES, LaneChannelSlot, LaneHeader, LaneRecord,
    LaneShape, NO_TICKET, OpParams, SCRATCH_ALIGN, SCRATCH_DUMMY_BYTES, SCRATCH_POISON, ValueDesc,
};
use eta_ir::Dtype;
use eta_ir::container::HostRole;
use eta_ir::op::tags;
use eta_ir::types::name_or_unknown;

use std::sync::Arc;

use kernels_cuda::runtime::channel::MAX_RING;

use crate::error::{Fault, Result};

use super::compile::{Module, Stage};
use super::endpoint::Endpoint;

pub const INTRINSIC_STORAGE_F32: u32 = 0;

pub const INTRINSIC_STORAGE_RAW_BF16: u32 = 1;

pub const INTRINSIC_STORAGE_ROW_POINTERS: u32 = 2;

pub const INTRINSIC_STORAGE_RAW_I32: u32 = 3;

pub const BOOL_STORAGE_NATIVE_BYTES: u32 = 0;

#[must_use]
pub fn native_cell_bytes(dtype: Dtype, numel: usize) -> usize {
    if dtype == Dtype::Bool {
        numel.max(1)
    } else {
        numel.max(1) * 4
    }
}

pub fn wire_to_native(dtype: Dtype, numel: usize, wire: &[u8]) -> Result<Vec<u8>> {
    let numel = numel.max(1);
    let want = eta_exec::wire_cell_bytes(dtype, numel);
    if wire.len() != want {
        return Err(Fault::program(
            "program::launch",
            format!(
                "a {} wire cell of {numel} lane(s) is {want} bytes and {} were offered",
                name_or_unknown(dtype),
                wire.len()
            ),
        ));
    }
    if dtype != Dtype::Bool {
        return Ok(wire.to_vec());
    }
    Ok((0..numel)
        .map(|i| u8::from(wire[i / 8] >> (i % 8) & 1 == 1))
        .collect())
}

pub fn native_to_wire(dtype: Dtype, numel: usize, native: &[u8]) -> Result<Vec<u8>> {
    let numel = numel.max(1);
    let want = native_cell_bytes(dtype, numel);
    if native.len() != want {
        return Err(Fault::program(
            "program::launch",
            format!(
                "a {} native cell of {numel} lane(s) is {want} bytes and {} were offered",
                name_or_unknown(dtype),
                native.len()
            ),
        ));
    }
    if dtype != Dtype::Bool {
        return Ok(native.to_vec());
    }
    let mut out = vec![0u8; eta_exec::wire_cell_bytes(dtype, numel)];
    for (i, &byte) in native.iter().enumerate().take(numel) {
        if byte != 0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelShape {
    pub numel: usize,
    pub dtype: Dtype,
    pub capacity: u32,
}

impl ChannelShape {
    #[must_use]
    pub fn of(declared: &eta_compiler::codegen::launch::LaunchChannel) -> ChannelShape {
        ChannelShape {
            numel: declared
                .shape
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1),
            dtype: eta_exec::concrete_dtype(declared.dtype),
            capacity: declared.capacity.max(1),
        }
    }

    #[must_use]
    pub const fn ring(&self) -> u64 {
        self.capacity as u64 + 1
    }

    #[must_use]
    pub fn cell_bytes(&self) -> usize {
        native_cell_bytes(self.dtype, self.numel)
    }
}

/// What an instance of a program keeps in the pool
/// (`new-memory/STATE-ROLES.md`): its own item, the rings' registry and
/// then the native cells of each ring the host reads or writes, then each
/// stage's bind-time tables (`eta_exec::tables`); and the rings only the
/// device touches, an item each.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Held {
    /// The instance's item, whole.
    pub instance_bytes: u64,
    /// Where each channel's native cells lie in the instance's item;
    /// `None` for a ring that is an item of its own.
    pub native_at: Vec<Option<u64>>,
    /// The bytes of each ring that is an item of its own; `None` for the
    /// channels above.
    pub ring_bytes: Vec<Option<u64>>,
    /// Where each stage's tables lie in the instance's item: its
    /// descriptors, op params, value offsets and config, then its steps.
    pub tables: Vec<[u64; 5]>,
}

/// What every cell span in an item starts a whole number of.
const CELL_ALIGN: u64 = 64;

/// What every table in an item starts a whole number of.
const TABLE_ALIGN: u64 = 256;

impl Held {
    #[must_use]
    pub fn of(
        channels: &[eta_compiler::codegen::launch::LaunchChannel],
        plans: &[LaunchStagePlan],
        steps: &[usize],
    ) -> Held {
        let mut at = registry_bytes(channels.len()).next_multiple_of(CELL_ALIGN);
        let mut native_at = Vec::with_capacity(channels.len());
        let mut ring_bytes = Vec::with_capacity(channels.len());
        for declared in channels {
            let shape = ChannelShape::of(declared);
            let bytes = shape.cell_bytes() as u64 * shape.ring();
            if declared.host_role == HostRole::None {
                native_at.push(None);
                ring_bytes.push(Some(bytes));
            } else {
                native_at.push(Some(at));
                ring_bytes.push(None);
                at = (at + bytes).next_multiple_of(CELL_ALIGN);
            }
        }
        let mut tables = Vec::with_capacity(plans.len());
        for (index, plan) in plans.iter().enumerate() {
            let mut stage = [0u64; 5];
            let [descs, params, offs, cfg] = eta_exec::tables::table_sizes(plan);
            let steps = (steps.get(index).copied().unwrap_or(0).max(1)
                * cuda_steps::STEP_RECORD_BYTES) as u64;
            for (slot, bytes) in stage.iter_mut().zip([descs, params, offs, cfg, steps]) {
                at = at.next_multiple_of(TABLE_ALIGN);
                *slot = at;
                at += bytes;
            }
            tables.push(stage);
        }
        Held {
            instance_bytes: at,
            native_at,
            ring_bytes,
            tables,
        }
    }
}

/// The registry of `slots` rings: their head, tail and ring words, then
/// each ring's full bytes.
const fn registry_bytes(slots: usize) -> u64 {
    (3 * slots * size_of::<u32>() + slots * MAX_RING as usize) as u64
}

/// Where an instance's items lie on the device now: the instance's item,
/// and each ring that is an item of its own (0 for a channel whose cells
/// are in the instance's item or its endpoint's).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Place {
    pub instance: u64,
    pub rings: Vec<u64>,
}

#[derive(Debug)]
pub struct Rings {
    /// Each channel's native cells in the pool; 0 where its endpoint
    /// carries them.
    cells: Vec<u64>,
    shapes: Vec<ChannelShape>,
    endpoints: Vec<Option<Arc<Endpoint>>>,
    held: Held,
    registry: u64,
    device: kernels_cuda::runtime::channel::Rings,
    /// The stream the cells are staged and fired on, where a read of one
    /// waits behind what wrote it.
    stream: usize,
}

impl Rings {
    /// The rings of `shapes` at `place`, their cells zeroed and their
    /// registry staged on `stream` ahead of any fire that reads it: a pool
    /// slot holds what its last holder left.
    pub fn allocate(
        shapes: &[ChannelShape],
        endpoints: Vec<Option<Arc<Endpoint>>>,
        held: Held,
        place: &Place,
        stream: *mut core::ffi::c_void,
    ) -> Result<Rings> {
        if endpoints.len() != shapes.len() {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "{} channel shape(s) and {} endpoint slot(s): the two are indexed by \
                     the same dense channel number",
                    shapes.len(),
                    endpoints.len()
                ),
            ));
        }
        for (index, shape) in shapes.iter().enumerate() {
            if shape.cell_bytes() == 0 {
                return Err(Fault::program(
                    "program::launch",
                    format!("channel {index}'s cell is zero bytes, so it can never be ready"),
                ));
            }
        }
        let mut rings = Rings {
            cells: Vec::new(),
            shapes: shapes.to_vec(),
            endpoints,
            held,
            registry: 0,
            device: kernels_cuda::runtime::channel::Rings::new(0, 0, 0, 0, 0),
            stream: stream as usize,
        };
        rings.rebase(place)?;
        crate::device::zero_span_on(
            stream,
            rings.registry,
            usize::try_from(rings.held.instance_bytes).unwrap_or(usize::MAX),
        )?;
        for (channel, shape) in shapes.iter().enumerate() {
            if rings.held.ring_bytes[channel].is_some() && rings.shared_slab(channel).is_none() {
                crate::device::zero_span_on(
                    stream,
                    rings.cells[channel],
                    shape.cell_bytes() * shape.ring() as usize,
                )?;
            }
        }
        let cap1: Vec<u8> = shapes
            .iter()
            .flat_map(|shape| {
                u32::try_from(shape.ring())
                    .unwrap_or(u32::MAX)
                    .to_le_bytes()
            })
            .collect();
        let words = (shapes.len() * size_of::<u32>()) as u64;
        crate::device::stage_raw(stream, rings.registry + 2 * words, &cap1)?;
        Ok(rings)
    }

    /// Points the rings at `place`: where the pool's slots they lie in are
    /// now. Every fire staged after addresses them there.
    pub fn rebase(&mut self, place: &Place) -> Result<()> {
        let slots = self.shapes.len();
        let mut cells = Vec::with_capacity(slots);
        for channel in 0..slots {
            let own = place.rings.get(channel).copied().unwrap_or(0);
            cells.push(match self.held.native_at[channel] {
                Some(at) => place.instance + at,
                None if self.shared_slab(channel).is_some() => 0,
                None if own != 0 => own,
                None => {
                    return Err(Fault::program(
                        "program::launch",
                        format!(
                            "channel {channel} is a ring only the device touches, and its \
                             binding names neither a registered channel nor slots for it"
                        ),
                    ));
                }
            });
        }
        let words = (slots * size_of::<u32>()) as u64;
        self.cells = cells;
        self.registry = place.instance;
        self.device = kernels_cuda::runtime::channel::Rings::new(
            place.instance + 3 * words,
            place.instance,
            place.instance + words,
            place.instance + 2 * words,
            u32::try_from(slots).unwrap_or(u32::MAX),
        );
        Ok(())
    }

    /// Where stage `stage`'s tables lie on the device now: its
    /// descriptors, op params, value offsets and config, then its steps.
    #[must_use]
    pub fn tables(&self, stage: usize) -> Option<[u64; 5]> {
        let offsets = self.held.tables.get(stage)?;
        Some(offsets.map(|at| self.registry + at))
    }

    #[must_use]
    pub const fn device(&self) -> kernels_cuda::runtime::channel::Rings {
        self.device
    }

    #[must_use]
    pub fn endpoint(&self, channel: usize) -> Option<&Arc<Endpoint>> {
        self.endpoints.get(channel).and_then(Option::as_ref)
    }

    pub fn seed_registry(
        &mut self,
        cursors: &[Cursor],
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        let slots = self.shapes.len();
        let words = slots * size_of::<u32>();
        let mut head = Vec::with_capacity(words);
        let mut tail = Vec::with_capacity(words);
        let mut full = vec![0u8; slots * MAX_RING as usize];
        for (channel, shape) in self.shapes.iter().enumerate() {
            let ring = shape.ring();
            let cursor = cursors
                .get(channel)
                .copied()
                .unwrap_or(Cursor { head: 0, tail: 0 });
            head.extend_from_slice(&((cursor.head % ring) as u32).to_le_bytes());
            tail.extend_from_slice(&((cursor.tail % ring) as u32).to_le_bytes());
            for sequence in cursor.head..cursor.tail {
                let at = kernels_cuda::runtime::channel::Rings::full_at(
                    u32::try_from(channel).unwrap_or(u32::MAX),
                    (sequence % ring) as u32,
                ) as usize;
                if let Some(byte) = full.get_mut(at) {
                    *byte = 1;
                }
            }
        }
        crate::device::stage_raw(stream, self.registry, &head)?;
        crate::device::stage_raw(stream, self.registry + words as u64, &tail)?;
        crate::device::stage_raw(stream, self.registry + (3 * words) as u64, &full)?;
        Ok(())
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

    pub fn cell_address(&self, channel: usize, sequence: u64) -> Result<u64> {
        let shape = self.shape_of(channel)?;
        let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
        match self.shared_slab(channel) {
            Some(base) => Ok(base + at),
            None => Ok(self.cells[channel] + at),
        }
    }

    pub fn feed_address(&self, channel: usize, sequence: u64) -> Result<u64> {
        let shape = self.shape_of(channel)?;
        if shape.dtype == Dtype::Bool {
            return Err(Fault::program(
                "program::launch",
                format!("channel {channel} is a bool ring, which no float input reads"),
            ));
        }
        let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
        if let Some(base) = self.shared_slab(channel) {
            return Ok(base + at);
        }
        if let Some(endpoint) = self.endpoint(channel) {
            let cell = (sequence % u64::from(endpoint.cap1())) * u64::from(endpoint.wire_bytes());
            return Ok(endpoint.mirror_device() + cell);
        }
        Ok(self.cells[channel] + at)
    }

    #[must_use]
    pub fn shared_slab(&self, channel: usize) -> Option<u64> {
        self.endpoints
            .get(channel)
            .and_then(Option::as_ref)
            .and_then(|endpoint| endpoint.device_cells())
    }

    pub fn write_cell(
        &mut self,
        channel: usize,
        sequence: u64,
        native: &[u8],
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        let shape = self.shape_of(channel)?;
        if native.len() != shape.cell_bytes() {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "channel {channel}'s native cell is {} bytes and {} were offered",
                    shape.cell_bytes(),
                    native.len()
                ),
            ));
        }
        if let Some(base) = self.shared_slab(channel) {
            let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
            crate::device::stage_raw(stream, base + at, native)?;
            if let Some(endpoint) = self.endpoint(channel)
                && !endpoint.write_cell(sequence, native)
            {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "channel {channel}'s shared shadow cell is {} bytes and {} were \
                         offered",
                        endpoint.wire_bytes(),
                        native.len()
                    ),
                ));
            }
            return Ok(());
        }
        if let Some(endpoint) = self.endpoint(channel) {
            let wire = native_to_wire(shape.dtype, shape.numel, native)?;
            if !endpoint.write_cell(sequence, &wire) {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "channel {channel}'s mirror cell is {} bytes and {} were offered",
                        endpoint.wire_bytes(),
                        wire.len()
                    ),
                ));
            }
            return Ok(());
        }
        let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
        crate::device::stage_raw(stream, self.cells[channel] + at, native)
    }

    pub fn read_cell(&self, channel: usize, sequence: u64) -> Result<Vec<u8>> {
        let shape = self.shape_of(channel)?;
        if let Some(endpoint) = self.endpoint(channel)
            && endpoint.role() == HostRole::None
        {
            return Ok(endpoint.read_cell(sequence));
        }
        if let Some(endpoint) = self.endpoint(channel) {
            return wire_to_native(shape.dtype, shape.numel, &endpoint.read_cell(sequence));
        }
        let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
        let mut out = vec![0u8; shape.cell_bytes()];
        crate::device::copy_d2h_on(
            self.stream as *mut core::ffi::c_void,
            self.cells[channel] + at,
            &mut out,
        )?;
        Ok(out)
    }

    pub(crate) fn shape_of(&self, channel: usize) -> Result<ChannelShape> {
        self.shapes.get(channel).copied().ok_or_else(|| {
            Fault::program(
                "program::launch",
                format!("channel {channel} is not one this instance carries"),
            )
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    pub head: u64,
    pub tail: u64,
}
#[derive(Debug)]
pub struct Prepared {
    /// The stage's bind-time tables (descriptors, op params, value offsets,
    /// config; `eta_exec::tables::stage_tables` at the bind's extents), as
    /// a bind writes them into each instance's `Instance` slots
    /// (`Held::tables`).
    tables: [Vec<u8>; 4],
    /// Each node's temporary area, a word offset in a lane's scratch, which
    /// its steps' records carry.
    temporaries: Vec<u32>,
    /// The bytes a sort's ladder keeps, one lane slot's share of the fire's
    /// arena, and where that arena lies in the fire's span.
    arena_bytes: u32,
    arena: u64,
    /// The blocks a step runs a lane on.
    groups: u32,
    /// Where the fire's lane table, scratch and cells lie in its span,
    /// laid by [`Prepared::lay`]: each an address in it.
    table: u64,
    table_host: Vec<u8>,
    scratch: u64,
    /// Two words a lane slot, where a lane's blocks meet between steps.
    barriers: u64,
    /// The lanes' intrinsic tables (bases, modes, widths, strides,
    /// offsets) and pending flags, one span staged in one copy a commit
    /// (`cell_at` their offsets, `cells_host` its image).
    cells: u64,
    cell_at: [usize; 6],
    cells_host: Vec<u8>,
    intrinsic_bases_host: Vec<u64>,
    intrinsic_modes_host: Vec<u32>,
    intrinsic_widths_host: Vec<u32>,
    intrinsic_strides_host: Vec<u32>,
    intrinsic_offsets_host: Vec<u32>,
    /// The side-table slots a lane binds ([`eta_ir::op::output_pitch`]).
    pitch: usize,
    /// The outputs this stage reads, an output's slot its index among them
    /// plus `output_slots` a lane.
    names: Vec<String>,
    output_slots: u32,
    channel_count: u32,
    scratch_stride: u32,
    lanes: u32,
    filled: u32,
    bindings: Vec<u32>,
}

/// What every span of a stage starts a whole number of in the fire's span.
const SPAN_ALIGN: u64 = 256;

/// The spans a stage lays in a fire's span: its lane table, scratch, cells,
/// barriers and arena.
const SPANS: u64 = 5;

/// The most blocks a lane's stage runs on.
const MAX_GROUPS: u32 = 64;

/// The blocks a step of a stage runs a lane on: enough for its widest
/// value a thread an element, at most four a multiprocessor.
fn groups_for(descriptors: &[ValueDesc]) -> u32 {
    let widest = descriptors.iter().map(|d| d.len).max().unwrap_or(1).max(1);
    widest
        .div_ceil(cuda_steps::BLOCK)
        .clamp(1, device_units().saturating_mul(4).max(1))
}

/// `plan`'s tables and a lane's scratch at `extents`. A sink's operands (an
/// adapter's planes, landed at bind) are the engine's, and take no bytes of
/// a lane.
fn staged(plan: &LaunchStagePlan, extents: Extents) -> Result<SteppedTables> {
    eta_exec::tables::stepped_tables(plan, &extents, |region| {
        region.kind != RegionKind::Library(LibraryOp::SecondParty)
    })
    .map_err(|why| Fault::program("program::launch", why))
}

/// Each of `plan`'s values as a bind at `extents` describes it.
pub fn describe_values(plan: &LaunchStagePlan, extents: Extents) -> Result<Vec<ValueDesc>> {
    staged(plan, extents).map(|staged| staged.tables.descriptors)
}

/// The bytes a lane of `plan` at `extents` takes of a fire's scratch.
pub fn scratch_bytes(plan: &LaunchStagePlan, extents: Extents) -> Result<u64> {
    staged(plan, extents).map(|staged| staged.tables.scratch)
}

/// Where each of `plan`'s values lies in a lane's scratch at `extents`.
pub fn scratch_offsets(plan: &LaunchStagePlan, extents: Extents) -> Result<Vec<u64>> {
    staged(plan, extents).map(|staged| staged.tables.offsets)
}

impl Prepared {
    pub fn build(
        plan: &LaunchStagePlan,
        shapes: &[ChannelShape],
        extents: Extents,
    ) -> Result<Prepared> {
        let channel_count = u32::try_from(plan.channel_bindings.len())
            .map_err(|_| Fault::program("program::launch", "more channels than a u32 can count"))?;
        let SteppedTables {
            tables: staged,
            temporaries,
            arena,
        } = staged(plan, extents)?;
        let mut tables = staged.tables;
        // A put's cell bytes, which the shared params leave to the engine.
        let record = size_of::<OpParams>();
        for (at, op) in plan.ops.iter().enumerate() {
            let (true, Some(channel)) = (op.tag == tags::CHAN_PUT, op.channel) else {
                continue;
            };
            let dense = plan
                .channel_bindings
                .get(channel as usize)
                .copied()
                .ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!(
                            "a put names stage-local channel {channel}, which the plan does \
                             not bind"
                        ),
                    )
                })?;
            let shape = shapes.get(dense as usize).copied().ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    format!("a put targets channel {dense}, which this instance does not carry"),
                )
            })?;
            let bytes = u32::try_from(shape.cell_bytes()).map_err(|_| {
                Fault::program("program::launch", "a channel cell past what a u32 counts")
            })?;
            let word = at * record + 15 * size_of::<u32>();
            tables[1][word..word + 4].copy_from_slice(&bytes.to_le_bytes());
        }
        let scratch_stride = u32::try_from(staged.scratch)
            .map_err(|_| Fault::program("program::launch", "a scratch stride past a u32"))?;
        Ok(Prepared {
            tables,
            temporaries,
            arena_bytes: u32::try_from(arena)
                .map_err(|_| Fault::program("program::launch", "a sort's arena past a u32"))?,
            arena: 0,
            groups: groups_for(&staged.descriptors),
            table: 0,
            table_host: Vec::new(),
            scratch: 0,
            barriers: 0,
            cells: 0,
            cell_at: [0; 6],
            cells_host: Vec::new(),
            intrinsic_bases_host: Vec::new(),
            intrinsic_modes_host: Vec::new(),
            intrinsic_widths_host: Vec::new(),
            intrinsic_strides_host: Vec::new(),
            intrinsic_offsets_host: Vec::new(),
            pitch: plan.output_pitch as usize,
            names: plan.names.clone(),
            output_slots: plan.output_slots,
            channel_count,
            scratch_stride,
            lanes: 0,
            filled: 0,
            bindings: plan.channel_bindings.clone(),
        })
    }

    /// The stage's tables, in [`Held::tables`] order, as a bind writes them.
    #[must_use]
    pub const fn tables(&self) -> &[Vec<u8>; 4] {
        &self.tables
    }

    /// Each node's temporary area at this stage's extents.
    #[must_use]
    pub fn temporaries(&self) -> &[u32] {
        &self.temporaries
    }

    /// The bytes of the spans whose size no lane count changes: the lane
    /// table's header, and the room every span's alignment may take.
    #[must_use]
    pub fn fixed_bytes(&self) -> u64 {
        LANE_HEADER_BYTES + SPANS * SPAN_ALIGN + 6 * 16
    }

    /// The bytes each lane adds: its record and channel slots, its scratch,
    /// its intrinsic and pending cells, its barrier and its arena share.
    #[must_use]
    pub fn lane_bytes(&self) -> u64 {
        let words = self.pitch as u64;
        LANE_RECORD_BYTES
            + u64::from(self.channel_count) * size_of::<LaneChannelSlot>() as u64
            + u64::from(self.scratch_stride)
            + words * (size_of::<u64>() + 4 * size_of::<u32>()) as u64
            + u64::from(self.channel_count).max(1)
            + 2 * size_of::<u32>() as u64
            // A lane's share of the arena: a fire lays one a lane.
            + u64::from(self.arena_bytes)
    }

    /// Lays this stage for `lanes` lanes in the fire's span from `at` up,
    /// and returns where its bytes end. Nothing is kept there between
    /// fires: a span is the fire's.
    pub fn lay(&mut self, extents: Extents, lanes: u32, at: u64) -> Result<u64> {
        self.filled = 0;
        let lanes = lanes.max(1);
        let total = u64::from(self.scratch_stride) * u64::from(lanes);
        if total > eta_exec::SCRATCH_MAX_BYTES {
            return Err(Fault::Ceiling {
                what: "a wave's guest scratch",
                need: total,
                have: eta_exec::SCRATCH_MAX_BYTES,
            });
        }
        let mut next = at.next_multiple_of(SPAN_ALIGN);
        let mut take = |bytes: usize| {
            let here = next;
            next = (here + bytes as u64).next_multiple_of(SPAN_ALIGN);
            here
        };
        let shape = LaneShape::of(lanes, self.channel_count);
        let table_bytes = shape
            .bytes()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| Fault::program("program::launch", "a lane table past what fits"))?;
        if lanes != self.lanes {
            let mut table_host = vec![0u8; table_bytes];
            write_record(
                &mut table_host,
                0,
                &LaneHeader {
                    abi_version: eta_exec::LANE_ABI_VERSION,
                    lane_count: lanes,
                    channel_slots_per_lane: self.channel_count,
                    flags: 0,
                },
            );
            for lane in 0..lanes {
                let at = shape
                    .record_offset(lane)
                    .and_then(|at| usize::try_from(at).ok())
                    .ok_or_else(|| {
                        Fault::program("program::launch", "a lane record past the table")
                    })?;
                write_record(
                    &mut table_host,
                    at,
                    &LaneRecord {
                        row_count: extents.row_count,
                        token_count: extents.token_count,
                        sampled_rows: extents.sampled_rows,
                        query_len: extents.query_len,
                        key_len: extents.key_len,
                        channel_slot_offset: shape.slot_index(lane).ok_or_else(|| {
                            Fault::program("program::launch", "a lane's slot row past the table")
                        })?,
                        ..LaneRecord::default()
                    },
                );
            }
            self.table_host = table_host;
        }
        self.table = take(table_bytes);
        self.scratch = take(
            (self.scratch_stride as usize * lanes as usize)
                .max(usize::try_from(SCRATCH_ALIGN).unwrap_or(256)),
        );
        let words = self.pitch * lanes as usize;
        self.intrinsic_bases_host.resize(words, 0u64);
        self.intrinsic_modes_host.resize(words, 0u32);
        self.intrinsic_widths_host.resize(words, 0u32);
        self.intrinsic_strides_host.resize(words, 0u32);
        self.intrinsic_offsets_host.resize(words, 0u32);
        let pending = (self.channel_count as usize * lanes as usize).max(1);
        let mut cell_at = 0usize;
        for (cell, bytes) in [
            words * size_of::<u64>(),
            words * size_of::<u32>(),
            words * size_of::<u32>(),
            words * size_of::<u32>(),
            words * size_of::<u32>(),
            pending,
        ]
        .into_iter()
        .enumerate()
        {
            self.cell_at[cell] = cell_at;
            cell_at += bytes.div_ceil(16) * 16;
        }
        if self.cells_host.len() != cell_at {
            self.cells_host = vec![0u8; cell_at];
        }
        self.cells = take(cell_at);
        self.barriers = take(2 * size_of::<u32>() * lanes as usize);
        // After the scratch, so a lane finds its share words past it.
        self.arena = take(self.arena_bytes as usize * lanes as usize);
        self.lanes = lanes;
        Ok(next)
    }

    pub fn stage_lane(&mut self, rings: &Rings, cursors: &[Cursor], commit: u64) -> Result<u32> {
        let lane = self.filled;
        if lane >= self.lanes {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "a boundary staged lane {lane} of a batch cut for {}: `begin` is what \
                     sizes it and it was told a smaller number",
                    self.lanes
                ),
            ));
        }
        let shape = LaneShape::of(self.lanes, self.channel_count);
        let record_at = shape
            .record_offset(lane)
            .and_then(|at| usize::try_from(at).ok())
            .ok_or_else(|| Fault::program("program::launch", "a lane record past the table"))?;
        let commit_at = record_at + std::mem::offset_of!(LaneRecord, commit_slot);
        self.table_host[commit_at..commit_at + size_of::<u64>()]
            .copy_from_slice(&commit.to_le_bytes());

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
            let slot_at = shape
                .slot_offset(lane, u32::try_from(local).unwrap_or(u32::MAX))
                .and_then(|at| usize::try_from(at).ok())
                .ok_or_else(|| {
                    Fault::program("program::launch", "a channel slot past the table")
                })?;
            write_record(
                &mut self.table_host,
                slot_at,
                &LaneChannelSlot {
                    committed_cell: rings.cell_address(channel, cursor.head)?,
                    pending_cell: rings.cell_address(channel, cursor.tail)?,
                    expected_head: NO_TICKET,
                    expected_tail: NO_TICKET,
                },
            );
        }
        let row = lane as usize * self.pitch;
        self.intrinsic_bases_host[row..row + self.pitch].fill(0);
        self.intrinsic_modes_host[row..row + self.pitch].fill(0);
        self.intrinsic_widths_host[row..row + self.pitch].fill(0);
        self.intrinsic_strides_host[row..row + self.pitch].fill(0);
        self.intrinsic_offsets_host[row..row + self.pitch].fill(0);
        self.filled += 1;
        Ok(lane)
    }

    pub fn commit_lanes(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        if self.filled == 0 {
            return Ok(());
        }
        write_record(
            &mut self.table_host,
            0,
            &LaneHeader {
                abi_version: eta_exec::LANE_ABI_VERSION,
                lane_count: self.filled,
                channel_slots_per_lane: self.channel_count,
                flags: 0,
            },
        );
        crate::device::stage_raw(stream, self.table, &self.table_host)?;
        // The intrinsic tables and the pending flags (zeroed) in one copy:
        // seven small ones each cost the stream a few microseconds a fire.
        let at = self.cell_at;
        let image = &mut self.cells_host;
        for (cell, bytes) in [
            slice_bytes(&self.intrinsic_bases_host),
            slice_bytes(&self.intrinsic_modes_host),
            slice_bytes(&self.intrinsic_widths_host),
            slice_bytes(&self.intrinsic_strides_host),
            slice_bytes(&self.intrinsic_offsets_host),
        ]
        .iter()
        .enumerate()
        {
            image[at[cell]..at[cell] + bytes.len()].copy_from_slice(bytes);
        }
        image[at[5]..].fill(0);
        crate::device::stage_raw(stream, self.cells, image)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// Where output `name` of lane `lane` sits in this stage's table; none
    /// when the stage reads no such output.
    #[must_use]
    pub fn slot_of(&self, name: &str, lane: u32) -> Option<u32> {
        let index = self.names.iter().position(|named| named == name)?;
        u32::try_from(index)
            .ok()?
            .checked_add(self.output_slots.checked_mul(lane)?)
    }

    pub fn bind_intrinsic(
        &mut self,
        lane: u32,
        slot: u32,
        base: u64,
        storage: u32,
        width: u32,
        row_stride: u32,
        row_offset: u32,
    ) -> Result<()> {
        let slot = slot as usize;
        if slot >= self.pitch {
            // A lane this stage reads no intrinsic of.
            return Ok(());
        }
        if lane >= self.lanes {
            return Err(Fault::program(
                "program::launch",
                format!("lane {lane} is past a batch cut for {}", self.lanes),
            ));
        }
        let at = lane as usize * self.pitch + slot;
        self.intrinsic_bases_host[at] = base;
        self.intrinsic_modes_host[at] = storage;
        self.intrinsic_widths_host[at] = width;
        self.intrinsic_strides_host[at] = row_stride;
        self.intrinsic_offsets_host[at] = row_offset;
        Ok(())
    }

    fn cell(&self, cell: usize) -> u64 {
        self.cells + self.cell_at[cell] as u64
    }

    /// Launches `stage` for the lanes staged: one launch of the stage
    /// kernel, which runs the stage's steps in order over every lane, the
    /// stage's tables at `tables` (where a bind laid them, in an instance
    /// of the flight; every instance of it lays the same bytes), each lane
    /// over its own scratch.
    pub fn launch_steps(
        &self,
        stage: &Stage,
        tables: [u64; 5],
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        if self.filled == 0 || stage.steps.is_empty() {
            return Ok(());
        }
        // A value stated to take no bytes lies on the lane's dummy bytes: a
        // build with `test-slot-fill` poisons them before every launch, so a
        // kernel that reads one answers wrongly where a golden sees it.
        // Without the feature no build writes anything here.
        if cfg!(feature = "test-slot-fill") {
            self.poison_dummy(stream)?;
        }
        let (groups, slots) = self.shape();
        // The lane slots' barriers start every launch at zero.
        crate::device::zero_span_on(stream, self.barriers, 2 * size_of::<u32>() * slots as usize)?;
        let pitch = u32::try_from(self.pitch).unwrap_or(u32::MAX);
        let steps = u32::try_from(stage.steps.len()).unwrap_or(u32::MAX);
        let mut args = Args::new();
        args.ptr(self.table)
            .ptr(self.lane_records_ptr())
            .ptr(self.channel_slots_ptr())
            .ptr(tables[0])
            .ptr(tables[1])
            .ptr(tables[2])
            .ptr(tables[3])
            .ptr(self.scratch)
            .u32(self.scratch_stride)
            .ptr(self.cell(5))
            .ptr(self.cell(0))
            .ptr(self.cell(1))
            .ptr(self.cell(2))
            .ptr(self.cell(3))
            .ptr(self.cell(4))
            .u32(pitch)
            .u32(groups)
            .ptr(tables[4])
            .u32(steps)
            .ptr(self.barriers)
            .ptr(self.arena)
            .u32(self.arena_bytes);
        launch(
            &stage.module,
            slots * groups,
            cuda_steps::BLOCK,
            &mut args,
            cuda_steps::ARITY,
            stream,
        )
    }

    /// The blocks a lane runs on and the lane slots one launch holds: every
    /// lane a slot of its own, so no fire loops its lanes; a lane's blocks
    /// as many as the device holds at once with the fire's lanes (they meet
    /// between steps, so they must be resident together), from one (which
    /// meets no other, past the units) to [`MAX_GROUPS`], and no more than
    /// the stage's widest value asks.
    fn shape(&self) -> (u32, u32) {
        let lanes = self.filled.max(1);
        let groups = (device_units().max(1) / lanes)
            .clamp(1, MAX_GROUPS)
            .min(self.groups.max(1));
        (groups, lanes)
    }

    #[must_use]
    pub fn lane_ceiling(&self) -> u32 {
        if self.scratch_stride == 0 {
            return u32::MAX;
        }
        u32::try_from(eta_exec::SCRATCH_MAX_BYTES / u64::from(self.scratch_stride))
            .unwrap_or(u32::MAX)
            .max(1)
    }

    #[must_use]
    pub const fn filled(&self) -> u32 {
        self.filled
    }

    #[must_use]
    pub const fn lanes(&self) -> u32 {
        self.lanes
    }

    #[must_use]
    pub const fn channel_count(&self) -> u32 {
        self.channel_count
    }

    fn lane_records_ptr(&self) -> u64 {
        self.table + LANE_HEADER_BYTES
    }

    fn channel_slots_ptr(&self) -> u64 {
        self.table + LANE_HEADER_BYTES + u64::from(self.lanes) * LANE_RECORD_BYTES
    }
}

fn write_record<T: Copy>(into: &mut [u8], at: usize, record: &T) {
    // SAFETY: `T` is a `#[repr(C)]` mirror of a device struct with every field
    // written by the caller, so reading it as bytes reads only initialised
    // memory.
    let bytes = unsafe {
        std::slice::from_raw_parts(std::ptr::from_ref(record).cast::<u8>(), size_of::<T>())
    };
    into[at..at + bytes.len()].copy_from_slice(bytes);
}

pub(super) fn slice_bytes<T: Copy>(records: &[T]) -> Vec<u8> {
    // SAFETY: as `write_record`, over a contiguous run of fully-initialised
    // `#[repr(C)]` mirrors.
    unsafe {
        std::slice::from_raw_parts(
            records.as_ptr().cast::<u8>(),
            std::mem::size_of_val(records),
        )
        .to_vec()
    }
}

#[derive(Default)]
pub struct Args {
    #[allow(clippy::vec_box)]
    storage: Vec<Box<u64>>,
    slots: Vec<*mut std::ffi::c_void>,
}

impl Args {
    #[must_use]
    pub fn new() -> Args {
        Args::default()
    }

    pub fn ptr(&mut self, pointer: u64) -> &mut Args {
        self.scalar(pointer)
    }

    pub fn u32(&mut self, value: u32) -> &mut Args {
        self.scalar(u64::from(value))
    }

    fn scalar(&mut self, value: u64) -> &mut Args {
        let mut cell = Box::new(value);
        let at: *mut u64 = &raw mut *cell;
        self.storage.push(cell);
        self.slots.push(at.cast());
        self
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

impl Prepared {
    /// The fire's lanes' dummy bytes (the first [`SCRATCH_DUMMY_BYTES`] of
    /// each lane's scratch, where every value that takes no bytes lies)
    /// filled with [`SCRATCH_POISON`].
    fn poison_dummy(&self, stream: *mut core::ffi::c_void) -> Result<()> {
        #[cfg(feature = "device")]
        {
            use crate::device::dr;
            // SAFETY: the scratch holds `filled` lanes `scratch_stride` apart,
            // each no shorter than its dummy bytes.
            let code = unsafe {
                dr::cuMemsetD2D32Async(
                    self.scratch,
                    self.scratch_stride as usize,
                    SCRATCH_POISON,
                    (SCRATCH_DUMMY_BYTES / 4) as usize,
                    self.filled as usize,
                    stream.cast(),
                )
            };
            if code != dr::CUresult::CUDA_SUCCESS {
                return Err(Fault::Device {
                    call: "cuMemsetD2D32Async",
                    code: code as i32,
                });
            }
        }
        #[cfg(not(feature = "device"))]
        let _ = (stream, SCRATCH_POISON, SCRATCH_DUMMY_BYTES);
        Ok(())
    }
}

/// The device's multiprocessors, which a sampler fire's blocks may fill.
fn device_units() -> u32 {
    static UNITS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *UNITS.get_or_init(|| {
        #[cfg(feature = "device")]
        {
            use crate::device::dr;
            let mut device: dr::CUdevice = 0;
            let mut count: core::ffi::c_int = 0;
            // SAFETY: out-parameters for the current context's device.
            unsafe {
                if dr::cuCtxGetDevice(&raw mut device) != dr::CUresult::CUDA_SUCCESS
                    || dr::cuDeviceGetAttribute(
                        &raw mut count,
                        dr::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                        device,
                    ) != dr::CUresult::CUDA_SUCCESS
                {
                    return 1;
                }
            }
            u32::try_from(count).unwrap_or(1).max(1)
        }
        #[cfg(not(feature = "device"))]
        {
            1
        }
    })
}

pub fn launch(
    module: &Module,
    grid: u32,
    block: u32,
    args: &mut Args,
    expected: usize,
    stream: *mut core::ffi::c_void,
) -> Result<()> {
    if args.len() != expected {
        return Err(Fault::program(
            "cuLaunchKernel",
            format!(
                "`{}` takes {expected} arguments and {} were bound; CUDA reads the rest \
                 from whatever follows the array",
                module.entry_name(),
                args.len()
            ),
        ));
    }
    if grid == 0 {
        return Err(Fault::program(
            "cuLaunchKernel",
            format!(
                "`{}` was launched with an empty grid, which launches nothing and \
                 returns success",
                module.entry_name()
            ),
        ));
    }
    #[cfg(feature = "device")]
    {
        use crate::device::dr;

        // SAFETY: the function is live for this borrow, and `args` holds every
        // scalar the pointer array points at for the duration of the call.
        let code = unsafe {
            dr::cuLaunchKernel(
                module.function(),
                grid,
                1,
                1,
                block,
                1,
                1,
                module.dynamic_shared(),
                stream.cast(),
                args.slots.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(Fault::Device {
                call: "cuLaunchKernel",
                code: code as i32,
            });
        }
        Ok(())
    }
    #[cfg(not(feature = "device"))]
    {
        let _ = (block, stream);
        Err(Fault::Runtimeless)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod shipped {
        use eta_compiler::codegen::launch::LaunchPackage;
        use eta_compiler::plan::{CompiledStage, compile_bound};
        use eta_dsl::Traced;
        use eta_dsl::prelude::*;
        use eta_ir::registry::{Axis, EngineCaps, OutputDecl, Port};

        pub const QWEN_VOCAB: u32 = 151_936;

        fn leak(channel: Channel) -> &'static Channel {
            Box::leak(Box::new(channel))
        }

        fn logits(vocab: u32) -> Vec<Vec<OutputDecl>> {
            vec![vec![OutputDecl {
                name: "logits".into(),
                axis: Axis::Row,
                width: vocab,
                dtype: eta_ir::Dtype::F32,
                layers: 0,
                readout: true,
            }]]
        }

        fn builder(vocab: u32) -> (Builder<'static>, &'static Channel) {
            let tok_in = leak(Channel::from([42i32]));
            let mut b = Builder::new(vocab, 16);
            b.outputs(logits(vocab));
            b.bind_port(Port::EmbedTokens, tok_in);
            b.bind_port(Port::EmbedIndptr, leak(Channel::from([0u32, 1])));
            b.bind_port(Port::Fold, leak(Channel::from([u32::MAX])));
            (b, tok_in)
        }

        /// `text-completion-bench`'s decode epilogue: a temperature-scaled
        /// top-p Gumbel-max draw.
        pub fn top_p(vocab: u32) -> Traced {
            let (temperature, top_p) = (0.6f32, 0.95f32);
            let rng = leak(Channel::from([0x7ce1u32, 0]));
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let logits = intrinsics::output(0, "logits");
                let scaled = logits / temperature;
                let probs = softmax(&scaled);
                let keep = pivot_threshold(probs, cummass_le(top_p));
                let neg_inf = broadcast(f32::NEG_INFINITY, [vocab]);
                let masked = select(&keep, &scaled, &neg_inf);
                let r = rng.take();
                let g = gumbel(&r, [vocab]);
                let r_next = &r + iota(2);
                rng.put(&r_next);
                let t = reshape(reduce_argmax(masked + g), [1]);
                tok_in.put(&t);
                out.put(&t);
            });
            out.note_host_take();
            b.build().expect("the top-p epilogue traces")
        }

        /// The `compat` generate step's sampler with a top-k and a top-p
        /// cut: both pivots, their `and`, the mask and a Gumbel-max draw.
        pub fn top_k_top_p(vocab: u32) -> Traced {
            let temperature = 0.8f32;
            let rng = leak(Channel::from([0x7ce1u32, 0]));
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let r = rng.take();
                let logits = reshape(intrinsics::output(0, "logits"), [vocab]);
                let scaled = &logits / temperature;
                let probs = softmax(&scaled);
                let ranked = pivot_threshold(&probs, rank_le(40u32));
                let nucleus = pivot_threshold(&probs, cummass_le(0.9f32));
                let keep = and(&ranked, &nucleus);
                let token = reshape(gumbel_max(mask_apply(&scaled, keep), &r), [1]);
                rng.put(&r + iota(2));
                tok_in.put(&token);
                out.put(&token);
            });
            out.note_host_take();
            b.build().expect("the top-k top-p epilogue traces")
        }

        /// A temperature Gumbel-max draw with no cut.
        pub fn gumbel_max_only(vocab: u32) -> Traced {
            let rng = leak(Channel::from([0x7ce1u32, 0]));
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let r = rng.take();
                let scaled = intrinsics::output(0, "logits") / 0.7f32;
                let token = reshape(gumbel_max(scaled, &r), [1]);
                rng.put(&r + iota(2));
                tok_in.put(&token);
                out.put(&token);
            });
            out.note_host_take();
            b.build().expect("the gumbel-max epilogue traces")
        }

        /// The greedy decode epilogue.
        pub fn greedy(vocab: u32) -> Traced {
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let t = reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]);
                tok_in.put(&t);
                out.put(&t);
            });
            out.note_host_take();
            b.build().expect("the greedy epilogue traces")
        }

        /// A greedy step that lands its top logprobs: a top-k region.
        pub fn top_logprobs(vocab: u32) -> Traced {
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let values_out = leak(Channel::new([5], dtype::f32).capacity(8));
            let ids_out = leak(Channel::new([5], dtype::i32).capacity(8));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let logits = reshape(intrinsics::output(0, "logits"), [vocab]);
                let t = reshape(reduce_argmax(&logits), [1]);
                let (values, ids) = top_k(log_softmax(&logits), 5);
                values_out.put(&values);
                ids_out.put(cast(&ids, dtype::i32));
                tok_in.put(&t);
                out.put(&t);
            });
            values_out.note_host_take();
            ids_out.note_host_take();
            out.note_host_take();
            b.build().expect("the top-logprobs epilogue traces")
        }

        /// A greedy step that lands the whole row in order: a sort region.
        pub fn sorted(vocab: u32) -> Traced {
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let ids_out = leak(Channel::new([vocab], dtype::i32).capacity(2));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let logits = reshape(intrinsics::output(0, "logits"), [vocab]);
                let t = reshape(reduce_argmax(&logits), [1]);
                let (_, ids) = sort_desc(softmax(&logits));
                ids_out.put(cast(&ids, dtype::i32));
                tok_in.put(&t);
                out.put(&t);
            });
            ids_out.note_host_take();
            out.note_host_take();
            b.build().expect("the sorted epilogue traces")
        }

        /// Every program here, by name.
        pub fn all(vocab: u32) -> Vec<(&'static str, Traced)> {
            vec![
                ("top-p", top_p(vocab)),
                ("top-k + top-p", top_k_top_p(vocab)),
                ("gumbel-max", gumbel_max_only(vocab)),
                ("greedy", greedy(vocab)),
                ("top logprobs", top_logprobs(vocab)),
                ("sorted", sorted(vocab)),
            ]
        }

        pub fn package(traced: &Traced, vocab: u32) -> LaunchPackage {
            compiled(traced, vocab).1
        }

        /// The program's stages as compiled, and the package built of them.
        pub fn compiled(traced: &Traced, vocab: u32) -> (Vec<CompiledStage>, LaunchPackage) {
            let caps = EngineCaps {
                lora: true,
                attn_page_mask: true,
            };
            let bound = eta_ir::validate::bind(traced.container().clone(), logits(vocab), caps)
                .expect("the program binds");
            let stages = compile_bound(&bound);
            let package = eta_compiler::codegen::launch::build(&bound, &stages);
            (stages, package)
        }
    }

    /// What a lane's scratch is for the programs inferlets ship, at a
    /// vocabulary of 151,936: each value its own bytes; as laid here, with
    /// the temporary area's share of it; and what it was while the rows a
    /// scan of the model's row leaves unwritten, and the reshapes read as
    /// their sources, held bytes.
    /// Every program inferlets ship runs as steps of the closed set, and a
    /// lane of it takes no more scratch than the fused regions laid for it
    /// (`WAS`, a lane at a vocabulary of 151,936): an output is read where
    /// it lies, a value its one elementwise reader can compute is computed
    /// there, and a ladder's buffers are the fire's arena's.
    #[test]
    fn a_shipped_program_runs_as_steps_of_the_set() {
        const WAS: [(&str, u64); 6] = [
            ("top-p", 2_898_432),
            ("top-k + top-p", 3_658_240),
            ("gumbel-max", 1_792),
            ("greedy", 768),
            ("top logprobs", 1_900_800),
            ("sorted", 3_040_000),
        ];
        let vocab = shipped::QWEN_VOCAB;
        for ((name, traced), (was_name, was)) in shipped::all(vocab).into_iter().zip(WAS) {
            assert_eq!(name, was_name);
            let package = shipped::package(&traced, vocab);
            let [plan] = &package.plans[..] else {
                panic!("{name}: one stage");
            };
            let steps = eta_compiler::codegen::cuda_steps::stage_steps(plan)
                .unwrap_or_else(|tag| panic!("{name}: no kernel serves tag {tag:#x}"));
            assert!(!steps.is_empty(), "{name}: a stage of no steps");
            let scratch = staged(plan, Extents::default())
                .unwrap_or_else(|why| panic!("{name}: {why:?}"))
                .tables
                .scratch;
            assert!(
                scratch <= was,
                "{name}: {scratch} bytes of scratch a lane, {was} before"
            );
        }
    }

    #[test]
    fn only_bool_is_packed_and_the_round_trip_is_lossless() {
        for (dtype, numel) in [
            (Dtype::F32, 3),
            (Dtype::I32, 5),
            (Dtype::U32, 1),
            (Dtype::Bool, 11),
        ] {
            let native = native_cell_bytes(dtype, numel);
            let wire = eta_exec::wire_cell_bytes(dtype, numel);
            if dtype == Dtype::Bool {
                assert_eq!(native, numel, "a byte per lane on the device");
                assert_eq!(wire, numel.div_ceil(8), "a bit per lane on the wire");
            } else {
                assert_eq!(native, wire, "four bytes either way");
            }
            let bytes: Vec<u8> = (0..wire).map(|i| (i as u8).wrapping_mul(37) | 1).collect();
            let there = wire_to_native(dtype, numel, &bytes).expect("one wire cell");
            assert_eq!(there.len(), native);
            let back = native_to_wire(dtype, numel, &there).expect("one native cell");
            if dtype == Dtype::Bool {
                for lane in 0..numel {
                    assert_eq!(
                        back[lane / 8] >> (lane % 8) & 1,
                        bytes[lane / 8] >> (lane % 8) & 1,
                        "lane {lane} did not survive the round trip"
                    );
                }
            } else {
                assert_eq!(back, bytes);
            }
        }
    }
}
