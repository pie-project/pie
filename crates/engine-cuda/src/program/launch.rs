#![allow(unsafe_code)]
use eta_compiler::codegen::launch::{LaunchRegion, LaunchStagePlan};
use eta_compiler::plan::{LibraryOp, RegionKind};
use eta_exec::{
    Extents, LANE_HEADER_BYTES, LANE_RECORD_BYTES, LaneChannelSlot, LaneHeader, LaneRecord,
    LaneShape, Launch, Launches, Layout, NO_TICKET, OpParams, SCRATCH_ALIGN, SCRATCH_DUMMY_BYTES,
    SCRATCH_POISON, ValueDesc, describe_launched, region_rows,
};
use eta_ir::Dtype;
use eta_ir::container::HostRole;
use eta_ir::op::tags;
use eta_ir::types::name_or_unknown;

use std::sync::Arc;

use kernels_cuda::runtime::channel::MAX_RING;

use crate::error::{Fault, Result};

use super::compile::{Module, Region};
use super::endpoint::Endpoint;

const FUSED_ARITY: usize = 19;

pub const ORDER_ROW_BLOCKS: u32 = 32;
/// The most blocks a lane's sampler region runs on (`sample_lane.cuh`); a
/// fire runs as many as the device holds at once with its lanes, from one.
pub const LANE_SAMPLER_BLOCKS: u32 = 16;
/// A sampler block's arena: a 64-byte head (unused: the lane's barrier
/// words are its line of the fire's tables), two exchange buffers and two
/// buffers of 512 gathered elements (`M1_FAST_GATHER`).
const LANE_SAMPLER_ARENA: u64 = 64 + 2 * 24_576 + 2 * 512 * 8;

/// Whether `kind` is a lane sampler region: its lane's blocks split each row
/// and meet at a barrier (`LANE_SAMPLER_BLOCKS`).
#[must_use]
pub fn lane_sampler(kind: RegionKind) -> bool {
    matches!(
        kind,
        RegionKind::Library(LibraryOp::NucleusSample | LibraryOp::RankSample | LibraryOp::LaneCut)
    )
}

pub const INTRINSIC_STORAGE_F32: u32 = 0;

pub const INTRINSIC_STORAGE_RAW_BF16: u32 = 1;

pub const INTRINSIC_STORAGE_ROW_POINTERS: u32 = 2;

pub const INTRINSIC_STORAGE_RAW_I32: u32 = 3;

pub const BOOL_STORAGE_NATIVE_BYTES: u32 = 0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CudaOpParams {
    pub tag: u32,
    pub a0: u32,
    pub a1: u32,
    pub a2: u32,
    pub o0: u32,
    pub o1: u32,
    pub imm: u32,
    pub imm2: u32,
    pub imm3: u32,
    pub kind: u32,
    pub pred_tag: u32,
    pub lit_dtype: u32,
    pub lit_bits: u32,
    pub channel_slot: u32,
    pub intr: u32,
    pub sink_bytes: u32,

    pub intrinsic_dtype: u32,
    pub bool_storage: u32,
    pub intrinsic_row_stride: u32,
    pub intrinsic_row_offset: u32,
    pub rng_seed: u64,
}

const _: () = assert!(size_of::<CudaOpParams>() == 88);

const _: () = {
    assert!(std::mem::offset_of!(CudaOpParams, tag) == 0);
    assert!(std::mem::offset_of!(CudaOpParams, a0) == 4);
    assert!(std::mem::offset_of!(CudaOpParams, a1) == 8);
    assert!(std::mem::offset_of!(CudaOpParams, a2) == 12);
    assert!(std::mem::offset_of!(CudaOpParams, o0) == 16);
    assert!(std::mem::offset_of!(CudaOpParams, o1) == 20);
    assert!(std::mem::offset_of!(CudaOpParams, imm) == 24);
    assert!(std::mem::offset_of!(CudaOpParams, imm2) == 28);
    assert!(std::mem::offset_of!(CudaOpParams, imm3) == 32);
    assert!(std::mem::offset_of!(CudaOpParams, kind) == 36);
    assert!(std::mem::offset_of!(CudaOpParams, pred_tag) == 40);
    assert!(std::mem::offset_of!(CudaOpParams, lit_dtype) == 44);
    assert!(std::mem::offset_of!(CudaOpParams, lit_bits) == 48);
    assert!(std::mem::offset_of!(CudaOpParams, channel_slot) == 52);
    assert!(std::mem::offset_of!(CudaOpParams, intr) == 56);
    assert!(std::mem::offset_of!(CudaOpParams, sink_bytes) == 60);
    assert!(std::mem::offset_of!(CudaOpParams, intrinsic_dtype) == 64);
    assert!(std::mem::offset_of!(CudaOpParams, bool_storage) == 68);
    assert!(std::mem::offset_of!(CudaOpParams, intrinsic_row_stride) == 72);
    assert!(std::mem::offset_of!(CudaOpParams, intrinsic_row_offset) == 76);
    assert!(std::mem::offset_of!(CudaOpParams, rng_seed) == 80);
};

impl CudaOpParams {
    #[must_use]
    pub const fn widen(shared: OpParams) -> CudaOpParams {
        CudaOpParams {
            tag: shared.tag,
            a0: shared.a0,
            a1: shared.a1,
            a2: shared.a2,
            o0: shared.o0,
            o1: shared.o1,
            imm: shared.imm,
            imm2: shared.imm2,
            imm3: shared.imm3,
            kind: shared.kind,
            pred_tag: shared.pred_tag,
            lit_dtype: shared.lit_dtype,
            lit_bits: shared.lit_bits,
            channel_slot: shared.channel_slot,
            intr: shared.intr,
            sink_bytes: shared.sink_bytes,
            intrinsic_dtype: INTRINSIC_STORAGE_F32,
            bool_storage: BOOL_STORAGE_NATIVE_BYTES,
            intrinsic_row_stride: 0,
            intrinsic_row_offset: 0,
            rng_seed: 0,
        }
    }
}

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
/// then the native cells of each ring the host reads or writes, and the
/// rings only the device touches, an item each.
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
}

/// What every cell span in an item starts a whole number of.
const CELL_ALIGN: u64 = 64;

impl Held {
    #[must_use]
    pub fn of(channels: &[eta_compiler::codegen::launch::LaunchChannel]) -> Held {
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
        Held {
            instance_bytes: at,
            native_at,
            ring_bytes,
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
/// are in the instance's item or its endpoint's), and its scratch: each
/// piece's address (`engine::InstanceBinding::scratch`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Place {
    pub instance: u64,
    pub rings: Vec<u64>,
    pub scratch: Vec<u64>,
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
    /// Where this stage's tables lie in its (program, extents) pair's
    /// state ([`Prepared::place`]): each an address there. A lane's
    /// scratch is its instance's: each of this stage's pieces at
    /// [`Prepared::pieces_at`] of it.
    table: u64,
    table_host: Vec<u8>,
    descriptors: u64,
    descriptor_row: Vec<u8>,
    descriptors_host: Vec<u8>,
    params: u64,
    params_host: Vec<u8>,
    offsets: u64,
    offsets_host: Vec<u8>,
    /// The lanes' intrinsic tables (bases, modes, widths, strides,
    /// offsets), pending flags, lines and scratch base tables, one span
    /// staged in one copy a commit (`cell_at` their offsets, `cells_host`
    /// its image).
    cells: u64,
    cell_at: [usize; CELLS],
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
    value_count: u32,
    /// Each of this stage's scratch pieces' bytes in use
    /// ([`eta_exec::ChunkLayout::ends`]).
    ends: Vec<u64>,
    /// Where each of its pieces lies in its instance's scratch, and the
    /// bytes a piece of that holds ([`Prepared::set_pieces_at`]).
    pieces_at: Vec<u64>,
    unit: u64,
    temporary_piece: u32,
    temporary_offset: u32,
    temporary_bytes: u32,
    region_rows: Vec<u32>,
    samplers: Vec<bool>,
    lanes: u32,
    /// The tables were placed anew since they were last staged.
    moved: bool,
    filled: u32,
    bindings: Vec<u32>,
}

/// What every span of a stage starts a whole number of.
pub const SPAN_ALIGN: u64 = 256;

/// The spans a stage lays in its pair's state: its params, offsets, lane
/// table, descriptors and cells ([`Prepared::spans`]).
pub const SPANS: usize = 5;

/// The cells a stage stages each commit: five intrinsic tables, the
/// pending flags, the lanes' lines and their scratch base tables.
const CELLS: usize = 8;
const PENDING: usize = 5;
const LINES: usize = 6;
const BASES: usize = 7;

/// A lane's line of a stage's cells, zeroed each commit: a lane sampler's
/// barrier words, one a block (`sample_lane.cuh`).
const LINE_BYTES: usize = 64;

#[must_use]
pub(crate) fn shell_launches(region: &LaunchRegion) -> bool {
    region.kind != RegionKind::Library(LibraryOp::SecondParty)
}

pub fn describe_values(plan: &LaunchStagePlan, extents: Extents) -> Result<Vec<ValueDesc>> {
    describe_launched(plan, &extents, shell_launches).map_err(|why| {
        Fault::program(
            "program::launch",
            format!("a value's shape does not resolve against this fire: {why:?}"),
        )
    })
}

/// The blocks each region runs across.
fn region_blocks(plan: &LaunchStagePlan, descriptors: &[ValueDesc]) -> Vec<u32> {
    plan.fused
        .iter()
        .zip(region_rows(plan, descriptors))
        .map(|(region, rows)| match region.kind {
            RegionKind::Generated => rows,
            kind if lane_sampler(kind) => LANE_SAMPLER_BLOCKS,
            RegionKind::Library(LibraryOp::TopK) if selects(plan, region) => rows,
            RegionKind::Library(LibraryOp::TopK | LibraryOp::Sort) => rows.min(ORDER_ROW_BLOCKS),
            RegionKind::Library(_) => 1,
        })
        .collect()
}

fn selects(plan: &LaunchStagePlan, region: &LaunchRegion) -> bool {
    region
        .nodes
        .first()
        .and_then(|&node| plan.ops.get(node as usize))
        .is_some_and(|op| {
            op.tag == tags::TOP_K
                && (1..=eta_compiler::codegen::cuda::order::TOP_K_SELECT_MAX).contains(&op.imm)
        })
}

/// The bytes a lane's temporary area holds: the most a region's blocks keep
/// there between them (a launch hands each block its share,
/// `launch_region`).
fn temporary_bytes(plan: &LaunchStagePlan, descriptors: &[ValueDesc], blocks: &[u32]) -> u64 {
    plan.fused
        .iter()
        .zip(blocks)
        .map(|(region, &blocks)| {
            block_temporary(plan, descriptors, region)
                .checked_next_multiple_of(SCRATCH_ALIGN)
                .unwrap_or(u64::MAX)
                .saturating_mul(u64::from(blocks.max(1)))
        })
        .max()
        .unwrap_or(0)
}

/// The bytes a block of `region` keeps in the temporary area: what its
/// kernel does, by the emitter that writes it.
fn block_temporary(
    plan: &LaunchStagePlan,
    descriptors: &[ValueDesc],
    region: &LaunchRegion,
) -> u64 {
    let widest = descriptors
        .iter()
        .map(|d| u64::from(d.last))
        .max()
        .unwrap_or(1)
        .max(1);
    match region.kind {
        // A generated region's block keeps its reductions' levels here and
        // nothing else.
        RegionKind::Generated => {
            eta_compiler::codegen::launch::generated_temporary_bytes(plan, region, widest)
        }
        // A sampler block's arena (`sample_lane.cuh`).
        kind if lane_sampler(kind) => LANE_SAMPLER_ARENA,
        // A select keeps its candidates in the block's shared memory.
        RegionKind::Library(LibraryOp::TopK) if selects(plan, region) => 0,
        // An order block sorts through two rows of indices (`order_a`,
        // `order_b`).
        RegionKind::Library(LibraryOp::TopK | LibraryOp::Sort) => {
            8 * order_width(plan, descriptors, region).unwrap_or(widest)
        }
        // A scan and a matmul keep nothing here.
        RegionKind::Library(_) => 0,
    }
}

/// The width of the rows an order region sorts: its one node's input's.
fn order_width(
    plan: &LaunchStagePlan,
    descriptors: &[ValueDesc],
    region: &LaunchRegion,
) -> Option<u64> {
    let op = plan.ops.get(*region.nodes.first()? as usize)?;
    let input = descriptors.get(*op.args.first()? as usize)?;
    Some(u64::from(input.last))
}

/// The stage as this engine runs it: a launch a region across its blocks,
/// a value a region's kernel never touches (what it streams through
/// registers, what a scan of the model's row leaves unwritten, a reshape
/// read as its source, a sampler's interior) taking no bytes, the temporary
/// area what its kernels keep there.
fn launches<'a>(plan: &'a LaunchStagePlan, blocks: &[u32], temporary: u64) -> Launches<'a> {
    Launches {
        each: plan
            .fused
            .iter()
            .zip(blocks)
            .map(|(region, &blocks)| Launch::of_region(region, blocks))
            .collect(),
        hollow: plan
            .fused
            .iter()
            .flat_map(|region| region.spent.iter().copied())
            .collect(),
        temporary,
        ..Launches::default()
    }
}

/// The launches `plan`'s regions make over `descriptors`, and what its
/// temporary area keeps.
fn run_of<'a>(plan: &'a LaunchStagePlan, descriptors: &[ValueDesc]) -> Launches<'a> {
    let blocks = region_blocks(plan, descriptors);
    let temporary = temporary_bytes(plan, descriptors, &blocks);
    launches(plan, &blocks, temporary)
}

fn lay_out(plan: &LaunchStagePlan, descriptors: &[ValueDesc]) -> Result<Layout> {
    eta_exec::lay_out(plan, descriptors, &run_of(plan, descriptors)).map_err(|why| {
        Fault::program(
            "program::launch",
            format!("this fire's scratch does not fit: {why:?}"),
        )
    })
}

pub fn scratch_bytes(plan: &LaunchStagePlan, extents: Extents) -> Result<u64> {
    let descriptors = describe_values(plan, extents)?;
    lay_out(plan, &descriptors).map(|layout| layout.total)
}

pub fn scratch_offsets(plan: &LaunchStagePlan, extents: Extents) -> Result<Vec<u64>> {
    let descriptors = describe_values(plan, extents)?;
    lay_out(plan, &descriptors).map(|layout| layout.values)
}

/// `plan`'s scratch at `extents` in `pieces`, the layout a guest lane's
/// scratch takes: each piece a run of adjacent chunks that holds the
/// widest item whole; an item past a segment is refused by name
/// ([`eta_exec::TooLarge::PastSegment`]).
pub fn scratch_chunked(
    plan: &LaunchStagePlan,
    extents: Extents,
    pieces: eta_exec::Pieces,
) -> Result<eta_exec::ChunkLayout> {
    let descriptors = describe_values(plan, extents)?;
    lay_out_chunked(plan, &descriptors, pieces)
}

fn lay_out_chunked(
    plan: &LaunchStagePlan,
    descriptors: &[ValueDesc],
    pieces: eta_exec::Pieces,
) -> Result<eta_exec::ChunkLayout> {
    eta_exec::lay_out_chunked(plan, descriptors, &run_of(plan, descriptors), pieces).map_err(
        |why| {
            Fault::program(
                "program::launch",
                format!(
                    "a lane's scratch does not lie in runs of chunks of {} bytes: {why}",
                    pieces.chunk
                ),
            )
        },
    )
}

impl Prepared {
    /// The stage at `extents`, its scratch laid in `pieces`.
    pub fn build(
        plan: &LaunchStagePlan,
        shapes: &[ChannelShape],
        extents: Extents,
        pieces: eta_exec::Pieces,
    ) -> Result<Prepared> {
        let channel_count = u32::try_from(plan.channel_bindings.len())
            .map_err(|_| Fault::program("program::launch", "more channels than a u32 can count"))?;
        let value_count = u32::try_from(plan.value_types.len())
            .map_err(|_| Fault::program("program::launch", "more values than a u32 can count"))?;

        let descriptors = describe_values(plan, extents)?;
        let scratch_layout = lay_out_chunked(plan, &descriptors, pieces)?;
        let temporary_offset = u32::try_from(scratch_layout.temporary.offset)
            .map_err(|_| Fault::program("program::launch", "a temporary offset past a u32"))?;
        let temporary_bytes = u32::try_from(scratch_layout.temporary_bytes)
            .map_err(|_| Fault::program("program::launch", "a temporary arena past a u32"))?;
        let region_rows = region_blocks(plan, &descriptors);
        let samplers = plan
            .fused
            .iter()
            .map(|region| lane_sampler(region.kind))
            .collect();

        let mut records = Vec::with_capacity(plan.ops.len());
        let mut result_base = 0u32;
        for op in &plan.ops {
            let mut record = CudaOpParams::widen(OpParams::of(op, result_base, plan.output_slots));
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
            records.push(record);
            result_base += u32::from(op.result_count);
        }

        // Each value's offset in its piece, then each value's piece: one
        // span, the pieces `value_count` words past the offsets.
        let offsets_host: Vec<u8> = scratch_layout
            .values
            .iter()
            .map(|placed| u32::try_from(placed.offset).unwrap_or(u32::MAX))
            .chain(scratch_layout.values.iter().map(|placed| placed.piece))
            .flat_map(u32::to_le_bytes)
            .collect();

        Ok(Prepared {
            table: 0,
            table_host: Vec::new(),
            descriptors: 0,
            descriptor_row: descriptors.iter().flat_map(record_bytes).collect(),
            descriptors_host: Vec::new(),
            params: 0,
            params_host: records_bytes(&records),
            offsets: 0,
            offsets_host,
            cells: 0,
            cell_at: [0; CELLS],
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
            value_count,
            pieces_at: Vec::new(),
            unit: scratch_layout.piece_bytes(),
            temporary_piece: scratch_layout.temporary.piece,
            ends: scratch_layout.ends,
            temporary_offset,
            temporary_bytes,
            region_rows,
            samplers,
            lanes: 0,
            moved: true,
            filled: 0,
            bindings: plan.channel_bindings.clone(),
        })
    }

    /// Each of this stage's scratch pieces' bytes: its share of an
    /// instance's scratch (`engine::GuestState::per_instance`).
    #[must_use]
    pub fn pieces(&self) -> &[u64] {
        &self.ends
    }

    /// This stage's pieces lie at `at` of its instance's scratch, laid in
    /// pieces of `unit` bytes ([`Prepared::pieces`]' order).
    pub fn set_pieces_at(&mut self, at: Vec<u64>, unit: u64) {
        self.pieces_at = at;
        self.unit = unit.max(1);
    }

    /// The bytes of this stage's spans for flights of up to `lanes` lanes,
    /// in [`SPANS`] order, each a whole number of [`SPAN_ALIGN`].
    #[must_use]
    pub fn spans(&self, lanes: u32) -> [u64; SPANS] {
        let lanes = lanes.max(1);
        let table = LaneShape::of(lanes, self.channel_count)
            .bytes()
            .unwrap_or(u64::MAX);
        [
            self.params_host.len().max(size_of::<CudaOpParams>()) as u64,
            self.offsets_host.len().max(size_of::<u32>()) as u64,
            table,
            (self.descriptor_row.len().max(1) * lanes as usize) as u64,
            Self::cell_layout(
                self.pitch,
                self.channel_count,
                self.lane_lines(),
                self.ends.len(),
                lanes,
            )
            .1 as u64,
        ]
        .map(|bytes| bytes.next_multiple_of(SPAN_ALIGN))
    }

    /// Whether a lane of this stage has a line ([`LINE_BYTES`]).
    fn lane_lines(&self) -> bool {
        self.meets_across_blocks()
    }

    /// The cells' offsets in their span for `lanes` lanes, and its bytes.
    fn cell_layout(
        pitch: usize,
        channels: u32,
        lines: bool,
        pieces: usize,
        lanes: u32,
    ) -> ([usize; CELLS], usize) {
        let words = pitch * lanes as usize;
        let pending = (channels as usize * lanes as usize).max(1);
        let mut cell_at = [0usize; CELLS];
        let mut at = 0usize;
        for (cell, bytes) in [
            words * size_of::<u64>(),
            words * size_of::<u32>(),
            words * size_of::<u32>(),
            words * size_of::<u32>(),
            words * size_of::<u32>(),
            pending,
            if lines {
                LINE_BYTES * lanes as usize
            } else {
                0
            },
            size_of::<u64>() * pieces * lanes as usize,
        ]
        .into_iter()
        .enumerate()
        {
            cell_at[cell] = at;
            at += bytes.div_ceil(16) * 16;
        }
        (cell_at, at)
    }

    /// This stage's spans lie at `at` ([`Prepared::spans`]' order), laid
    /// for flights of up to `lanes` lanes: the next lay stages its tables
    /// there again.
    pub fn place(&mut self, at: [u64; SPANS]) {
        let [params, offsets, table, descriptors, cells] = at;
        self.params = params;
        self.offsets = offsets;
        self.table = table;
        self.descriptors = descriptors;
        self.cells = cells;
        self.moved = true;
    }

    /// Lays this stage for a flight of `lanes` lanes where it was placed
    /// ([`Prepared::place`], for no fewer lanes): its lane table and
    /// descriptor rows built again where the lanes differ from the last
    /// lay, and its tables staged again where they moved too.
    pub fn lay(
        &mut self,
        extents: Extents,
        lanes: u32,
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        self.filled = 0;
        let lanes = lanes.max(1);
        let moved = self.moved || lanes != self.lanes;
        self.moved = false;
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
            self.descriptors_host = self
                .descriptor_row
                .iter()
                .copied()
                .cycle()
                .take(self.descriptor_row.len() * lanes as usize)
                .collect();
            let words = self.pitch * lanes as usize;
            self.intrinsic_bases_host.resize(words, 0u64);
            self.intrinsic_modes_host.resize(words, 0u32);
            self.intrinsic_widths_host.resize(words, 0u32);
            self.intrinsic_strides_host.resize(words, 0u32);
            self.intrinsic_offsets_host.resize(words, 0u32);
            let (cell_at, bytes) = Self::cell_layout(
                self.pitch,
                self.channel_count,
                self.lane_lines(),
                self.ends.len(),
                lanes,
            );
            self.cell_at = cell_at;
            self.cells_host = vec![0u8; bytes];
        }
        self.lanes = lanes;
        if moved {
            crate::device::stage_raw(stream, self.params, &self.params_host)?;
            crate::device::stage_raw(stream, self.offsets, &self.offsets_host)?;
            crate::device::stage_raw(stream, self.descriptors, &self.descriptors_host)?;
        }
        Ok(())
    }

    /// Stages the next lane of the flight: its commit slot, its channels'
    /// cells, and its scratch where its instance holds it (`scratch`, the
    /// address of each piece of the instance's): the lane's base table, an
    /// address for each of this stage's pieces, which its record names.
    pub fn stage_lane(
        &mut self,
        rings: &Rings,
        cursors: &[Cursor],
        commit: u64,
        scratch: &[u64],
    ) -> Result<u32> {
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
        let pieces = self.ends.len();
        let row = self.cell_at[BASES] + lane as usize * pieces * size_of::<u64>();
        for (index, &at) in self.pieces_at.iter().enumerate() {
            let base = scratch
                .get(usize::try_from(at / self.unit).unwrap_or(usize::MAX))
                .ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!(
                            "a lane's scratch piece {index} lies at byte {at} of its \
                             instance's, which holds {} piece(s)",
                            scratch.len()
                        ),
                    )
                })?
                + at % self.unit;
            let entry = row + index * size_of::<u64>();
            self.cells_host[entry..entry + size_of::<u64>()].copy_from_slice(&base.to_le_bytes());
        }
        let table_at = record_at + std::mem::offset_of!(LaneRecord, scratch_table);
        self.table_host[table_at..table_at + size_of::<u64>()]
            .copy_from_slice(&(self.cells + row as u64).to_le_bytes());

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
        image[at[PENDING]..at[BASES]].fill(0);
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

    pub fn launch_region(&self, region: &Region, stream: *mut core::ffi::c_void) -> Result<()> {
        if self.filled == 0 {
            return Ok(());
        }
        let mut args = Args::new();
        args.ptr(self.table)
            .ptr(self.lane_records_ptr())
            .ptr(self.channel_slots_ptr())
            .ptr(self.descriptors)
            .ptr(self.params)
            .ptr(self.offsets)
            .ptr(self.offsets + u64::from(self.value_count) * size_of::<u32>() as u64)
            .ptr(self.cell(LINES))
            .u32(self.value_count)
            .u32(self.temporary_piece)
            .u32(self.temporary_offset)
            .ptr(self.cell(PENDING))
            .ptr(self.cell(0))
            .ptr(self.cell(1))
            .ptr(self.cell(2))
            .ptr(self.cell(3))
            .ptr(self.cell(4));
        let rows = self
            .region_rows
            .get(region.region_index as usize)
            .copied()
            .unwrap_or(1)
            .max(1);
        let stride = (self.temporary_bytes / rows) & !(SCRATCH_ALIGN as u32 - 1);
        let sampler = self
            .samplers
            .get(region.region_index as usize)
            .copied()
            .unwrap_or(false);
        // A sampler lane runs on as many blocks as the device holds at once
        // with the fire's lanes (its blocks wait on one another), each over
        // its own arena; the lanes' barrier words (their lines) start at
        // zero, staged so with the cells.
        let rows = if sampler {
            (device_units() / self.filled.max(1)).clamp(1, LANE_SAMPLER_BLOCKS.min(rows))
        } else {
            rows
        };
        args.u32(rows).u32(stride);
        // A value stated to take no bytes lies on the lane's dummy bytes: a
        // build with `test-slot-fill` poisons them before every launch, so a
        // kernel that reads one answers wrongly where a golden sees it.
        // Without the feature no build writes anything here.
        if cfg!(feature = "test-slot-fill") {
            self.poison_dummy(stream)?;
        }
        launch(
            &region.module,
            self.filled.saturating_mul(rows),
            region.module.block_threads(),
            &mut args,
            FUSED_ARITY,
            stream,
        )
    }

    /// Whether a region of this stage is a lane sampler, whose blocks wait
    /// for one another and so need their grid resident at once.
    #[must_use]
    pub fn meets_across_blocks(&self) -> bool {
        self.samplers.iter().any(|&sampler| sampler)
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

pub(super) fn record_bytes<T: Copy>(record: &T) -> Vec<u8> {
    // SAFETY: as `write_record` — a fully-initialised `#[repr(C)]` mirror.
    unsafe {
        std::slice::from_raw_parts(std::ptr::from_ref(record).cast::<u8>(), size_of::<T>()).to_vec()
    }
}

pub(super) fn slice_bytes<T: Copy>(records: &[T]) -> Vec<u8> {
    // SAFETY: as `record_bytes`, over a contiguous run of them.
    unsafe {
        std::slice::from_raw_parts(
            records.as_ptr().cast::<u8>(),
            std::mem::size_of_val(records),
        )
        .to_vec()
    }
}

fn records_bytes(params: &[CudaOpParams]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_of_val(params));
    for record in params {
        bytes.extend_from_slice(&record_bytes(record));
    }
    bytes
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
            for lane in 0..self.filled as usize {
                // Piece 0 opens with the dummy bytes.
                let row = self.cell_at[BASES] + lane * self.ends.len() * size_of::<u64>();
                let mut word = [0u8; size_of::<u64>()];
                word.copy_from_slice(&self.cells_host[row..row + size_of::<u64>()]);
                // SAFETY: the lane's scratch, its instance's, is no shorter
                // than its dummy bytes.
                let code = unsafe {
                    dr::cuMemsetD2D32Async(
                        u64::from_le_bytes(word),
                        SCRATCH_DUMMY_BYTES as usize,
                        SCRATCH_POISON,
                        (SCRATCH_DUMMY_BYTES / 4) as usize,
                        1,
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
    use eta_ir::types::MAX_RANK;

    /// A take its region runs after a constant cannot hand the constant its
    /// slot: the scratch is laid out in the order the emitter runs the ops,
    /// not the order the stage states them.
    #[test]
    fn a_region_lays_its_scratch_out_in_the_order_it_runs() {
        let op =
            |tag, result_count, args: Vec<u32>, channel| eta_compiler::codegen::launch::LaunchOp {
                tag,
                result_count,
                args,
                channel,
                ..Default::default()
            };
        let mut plan = LaunchStagePlan {
            ops: vec![
                op(tags::CHAN_TAKE, 1, vec![], Some(1)),
                op(tags::CONST, 1, vec![], None),
                op(tags::CHAN_PUT, 0, vec![1], Some(0)),
            ],
            value_types: vec![
                eta_compiler::codegen::launch::LaunchPlanValue {
                    dtype: Dtype::U32,
                    axes: vec![eta_compiler::plan::Dimension::Static(1)],
                };
                2
            ],
            fused: vec![LaunchRegion {
                kind: RegionKind::Generated,
                nodes: vec![0, 1, 2],
                order: vec![1, 0, 2],
                ..Default::default()
            }],
            ..Default::default()
        };
        let one = ValueDesc {
            len: 1,
            rows: 1,
            last: 1,
            rank: 1,
            dtype: 0,
            dims: [0; MAX_RANK],
        };
        let offsets = |plan: &LaunchStagePlan| {
            eta_exec::lay_out(plan, &[one, one], &launches(plan, &[1], 0))
                .expect("fits")
                .values
        };
        let laid = offsets(&plan);
        assert_ne!(
            laid[0], laid[1],
            "the take lands over the constant the put reads after it"
        );
        plan.fused[0].order.clear();
        let stated = offsets(&plan);
        assert_eq!(
            stated[0], stated[1],
            "in the stated order the constant may reuse the dead take's slot"
        );
    }

    /// Guest programs as inferlets ship them, traced here: the scratch
    /// they are laid is read off these.
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
    #[test]
    fn a_shipped_programs_scratch_a_lane() {
        const WAS: [u64; 6] = [
            3_505_408, 4_265_984, 3_040_512, 1_216_256, 2_508_544, 3_647_744,
        ];
        let vocab = shipped::QWEN_VOCAB;
        let totals: Vec<(&str, u64, u64, u64)> = shipped::all(vocab)
            .iter()
            .map(|(name, traced)| {
                let package = shipped::package(traced, vocab);
                let [plan] = &package.plans[..] else {
                    panic!("{name}: one stage");
                };
                let descriptors = describe_values(plan, Extents::default()).expect("static shapes");
                let own = eta_exec::layout(&descriptors).expect("fits").total;
                let laid = lay_out(plan, &descriptors).expect("fits");
                (*name, own, laid.total, laid.temporary_bytes)
            })
            .collect();
        assert_eq!(
            totals,
            [
                ("top-p", 9_312_512, 2_898_432, 921_600),
                ("top-k + top-p", 9_616_896, 3_658_240, 921_600),
                ("gumbel-max", 5_512_960, 1_792, 0),
                ("greedy", 3_688_704, 768, 0),
                ("top logprobs", 6_728_960, 1_900_800, 76_032),
                ("sorted", 8_551_168, 3_040_000, 1_215_488),
            ]
        );
        for ((name, _, now, _), was) in totals.iter().zip(WAS) {
            assert!(*now < was, "{name}: {now} bytes a lane, {was} before");
        }
    }

    /// No generated kernel of the programs inferlets ship names the bytes
    /// of a value its stage states to take none (a generated kernel
    /// addresses a value as `offsets[<its id>]` and no other way), and no
    /// such value is one a region is handed, hands on or puts. A greedy or
    /// a Gumbel-max lane, whose one argmax scans the model's row, holds no
    /// row of the vocabulary at all.
    #[test]
    fn no_kernel_names_a_value_that_takes_no_bytes() {
        let vocab = shipped::QWEN_VOCAB;
        let mut unwritten_rows = Vec::new();
        for (name, traced) in shipped::all(vocab) {
            let (stages, package) = shipped::compiled(&traced, vocab);
            let ([stage], [plan]) = (&stages[..], &package.plans[..]) else {
                panic!("{name}: one stage");
            };
            let descriptors = describe_values(plan, Extents::default()).expect("static shapes");
            let laid = lay_out(plan, &descriptors).expect("fits");
            let hollow: Vec<u32> = plan
                .fused
                .iter()
                .flat_map(|region| region.spent.iter().copied())
                .collect();
            for &value in &hollow {
                assert_eq!(laid.values[value as usize], 0, "{name}: value {value}");
                for region in &plan.fused {
                    assert!(
                        !region.inputs.contains(&value)
                            && !region.outputs.contains(&value)
                            && region.sinks.iter().all(|sink| sink.value != value),
                        "{name}: value {value} takes no bytes and a region holds it"
                    );
                }
            }
            for (index, (region, lowered)) in
                stage.fused.regions.iter().zip(&plan.fused).enumerate()
            {
                if lowered.kind != RegionKind::Generated {
                    continue;
                }
                let source = eta_compiler::codegen::cuda::emit_region("kernel", stage, region)
                    .unwrap_or_else(|why| panic!("{name}: region {index}: {why:?}"))
                    .replace("intrinsic_offsets[", "");
                for &value in &hollow {
                    assert!(
                        !source.contains(&format!("offsets[{value}]")),
                        "{name}: region {index} names value {value}, which takes no bytes"
                    );
                }
            }
            let rows = descriptors
                .iter()
                .zip(&laid.values)
                .filter(|(descriptor, at)| descriptor.last == vocab && **at != 0)
                .count();
            if rows == 0 {
                unwritten_rows.push(name);
            }
        }
        assert_eq!(unwritten_rows, ["gumbel-max", "greedy"]);
    }

    /// The kernel of a region this engine states no temporary bytes for
    /// names no temporary area, and one it states bytes for does: the
    /// statement follows the emitters over the programs inferlets ship.
    /// (An op the interpreter runs is handed the area whatever it is; the
    /// interpreter goes to it for a reduction alone,
    /// `ptir_m1_runtime_body.cuh`.)
    #[test]
    fn a_kernel_names_the_temporary_area_only_where_bytes_are_stated() {
        // What every kernel's preamble says of the area, and a stream's
        // pointer to it, which only its reductions go through.
        const UNUSED: [&str; 3] = [
            "m1_u8* temporary = reinterpret_cast<m1_u8*>(scratch_bases[temporary_piece]) +\n      temporary_offset + (m1_u64)lane_row * temporary_stride;",
            "(void)temporary;",
            "float* stream_work = reinterpret_cast<float*>(temporary);",
        ];
        // The tag of the op a line hands the interpreter, if it does.
        let interpreted = |line: &str| -> Option<u8> {
            let (_, call) = line.split_once("ptir_m1_execute(")?;
            call.split_once('u')?.0.parse().ok()
        };
        let reduces =
            |tag: u8| matches!(tag, tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN);
        let vocab = shipped::QWEN_VOCAB;
        let mut stated_some = 0;
        let mut stated_none = 0;
        for (name, traced) in shipped::all(vocab) {
            let (stages, package) = shipped::compiled(&traced, vocab);
            let ([stage], [plan]) = (&stages[..], &package.plans[..]) else {
                panic!("{name}: one stage");
            };
            let descriptors = describe_values(plan, Extents::default()).expect("static shapes");
            assert_eq!(stage.fused.regions.len(), plan.fused.len());
            for (index, (region, lowered)) in
                stage.fused.regions.iter().zip(&plan.fused).enumerate()
            {
                let source = eta_compiler::codegen::cuda::emit_region("kernel", stage, region)
                    .unwrap_or_else(|why| panic!("{name}: region {index}: {why:?}"));
                let entry = source
                    .rfind("extern \"C\" __global__ void ")
                    .expect("a kernel has its entry");
                let (_, block) = source[entry..]
                    .split_once(") {")
                    .expect("its parameters end where its block opens");
                let mut body: String = block
                    .lines()
                    .filter(|line| interpreted(line).is_none_or(reduces))
                    .collect::<Vec<_>>()
                    .join("\n");
                for unused in UNUSED {
                    body = body.replace(unused, "");
                }
                let names: Vec<&str> = body
                    .lines()
                    .filter(|line| line.contains("temporary") || line.contains("stream_work"))
                    .collect();
                let stated = block_temporary(plan, &descriptors, lowered);
                assert_eq!(
                    !names.is_empty(),
                    stated != 0,
                    "{name}: region {index} ({:?}): {stated} bytes a block stated, named by {names:#?}",
                    lowered.kind
                );
                if stated == 0 {
                    stated_none += 1;
                } else {
                    stated_some += 1;
                }
            }
        }
        assert!(stated_some != 0 && stated_none != 0, "both kinds were seen");
    }

    /// A sampler region's kernel reads its logits, threshold and state and
    /// writes its probabilities, kept flags and token: what its nodes
    /// compute between those takes no bytes, what it touches keeps its own.
    #[test]
    fn a_samplers_untouched_values_take_no_bytes() {
        let vocab = shipped::QWEN_VOCAB;
        let package = shipped::package(&shipped::top_p(vocab), vocab);
        let [plan] = &package.plans[..] else {
            panic!("one stage");
        };
        let descriptors = describe_values(plan, Extents::default()).expect("static shapes");
        let laid = lay_out(plan, &descriptors).expect("fits");
        let bases: Vec<u32> = plan
            .ops
            .iter()
            .scan(0u32, |next, op| {
                let base = *next;
                *next += u32::from(op.result_count);
                Some(base)
            })
            .collect();
        let sampler = plan
            .fused
            .iter()
            .find(|region| lane_sampler(region.kind))
            .expect("top-p's cut and draw are a sampler region");

        let rows = sampler
            .spent
            .iter()
            .filter(|&&value| descriptors[value as usize].last == vocab)
            .count();
        assert_eq!(rows, 7, "rows of the vocabulary the kernel never touches");
        for &value in &sampler.spent {
            assert_eq!(
                laid.values[value as usize], 0,
                "value {value} takes no bytes"
            );
            assert!(!sampler.inputs.contains(&value));
            assert!(!sampler.outputs.contains(&value));
            assert!(sampler.sinks.iter().all(|sink| sink.value != value));
        }

        let node = |tag: u8| {
            sampler
                .nodes
                .iter()
                .copied()
                .find(|&node| plan.ops[node as usize].tag == tag)
                .expect("a sampler region holds its cut and its draw") as usize
        };
        let cut = node(tags::PIVOT_THRESHOLD);
        let touched = [
            plan.ops[node(tags::REDUCE_MAX)].args[0],
            plan.ops[cut].args[0],
            bases[cut],
            plan.ops[node(tags::RNG_KEYED)].args[0],
            bases[node(tags::REDUCE_ARGMAX)],
        ];
        let mut at: Vec<u64> = touched
            .iter()
            .map(|&value| laid.values[value as usize])
            .collect();
        assert!(at.iter().all(|&offset| offset != 0));
        at.sort_unstable();
        at.dedup();
        assert_eq!(at.len(), touched.len(), "each its own bytes");
        assert_eq!(
            laid.temporary_bytes,
            u64::from(LANE_SAMPLER_BLOCKS) * LANE_SAMPLER_ARENA.next_multiple_of(SCRATCH_ALIGN),
            "the sampler's blocks' arenas and nothing else"
        );
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
