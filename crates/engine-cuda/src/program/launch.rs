#![allow(unsafe_code)]
use eta_compiler::codegen::lane_scratch::LaneScratch;
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

const FUSED_ARITY: usize = 20;

/// The spans of a stage's cells ([`Prepared::cells`]): five intrinsic
/// tables, the pending flags, the row tables, the lane scratch table and
/// the intrinsics' row bounds.
const CELLS: usize = 9;

/// The cell that holds the lane scratch table.
const LANE_SCRATCH_CELL: usize = 7;

/// The cell that holds each intrinsic's row bound: the row its reads stop
/// short of.
const INTRINSIC_ROWS_CELL: usize = 8;

pub const ORDER_ROW_BLOCKS: u32 = 32;
/// The most blocks a lane's sampler region runs on (`sample_lane.cuh`); a
/// fire runs as many as the device holds at once with its lanes, from one.
pub const LANE_SAMPLER_BLOCKS: u32 = 16;
/// A sampler block's arena: a barrier word's line and two exchange buffers.
const LANE_SAMPLER_ARENA: u64 = 64 + 2 * 24_576;

/// A sampler lane's gather buffers, after its blocks' arenas: two of 512
/// gathered elements (`M1_FAST_GATHER`) for each of the
/// [`LANE_SAMPLER_BLOCKS`] homes a lane's gathers spread over, however few
/// blocks the lane runs on (`M1_LANE_GATHERS`).
const LANE_SAMPLER_GATHERS: u64 = LANE_SAMPLER_BLOCKS as u64 * 2 * 512 * 8;

/// What a sampler block's arena takes of the lane arenas, a block to the next.
fn lane_arena_stride() -> u64 {
    LANE_SAMPLER_ARENA.next_multiple_of(SCRATCH_ALIGN)
}

/// What a sampler lane on `blocks` blocks takes of the lane arenas, a lane to
/// the next (`M1_LANE_SPAN`).
fn lane_span(blocks: u32) -> u64 {
    u64::from(blocks) * lane_arena_stride() + LANE_SAMPLER_GATHERS
}

/// The engine's lane arenas: where a sampler's blocks meet, laid once at
/// load for the most lanes' spans a fire's sampler launch lays (lanes run their
/// launches one after another on the fire's main stream, `fan::Fan::fork`).
/// They are no lane's: a lane's scratch holds none of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Arenas {
    /// Their device address.
    pub at: u64,
    pub bytes: u64,
}

impl Arenas {
    /// The bytes they take for flights of up to `flight` lanes on a device
    /// of `units` multiprocessors: the most one launch lays, its lanes'
    /// spans on the blocks each is given (`launch_region`).
    #[must_use]
    pub fn bytes_for(flight: u32, units: u32) -> u64 {
        (1..=flight.max(1))
            .map(|filled| u64::from(filled) * lane_span(lane_blocks(units, filled)))
            .max()
            .unwrap_or(0)
    }
}

/// The blocks each of `filled` lanes' samplers runs on: as many as the
/// `units` this process runs on hold at once with them, a block a unit, from
/// one to [`LANE_SAMPLER_BLOCKS`] (the launch checks the kernel's occupancy
/// holds them).
fn lane_blocks(units: u32, filled: u32) -> u32 {
    (units / filled.max(1)).clamp(1, LANE_SAMPLER_BLOCKS)
}

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

/// A table of pointers to f32 rows, a row a lane wherever each lies.
pub const INTRINSIC_STORAGE_ROW_POINTERS_F32: u32 = 4;

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
    /// The row an intrinsic's reads stop short of, set where the kernel
    /// reads its output (`intrinsic_rows`).
    pub intrinsic_rows: u32,
    pub reserved: u32,
    pub rng_seed: u64,
}

const _: () = assert!(size_of::<CudaOpParams>() == 96);

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
    assert!(std::mem::offset_of!(CudaOpParams, intrinsic_rows) == 80);
    assert!(std::mem::offset_of!(CudaOpParams, rng_seed) == 88);
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
            intrinsic_rows: 0,
            reserved: 0,
            rng_seed: 0,
        }
    }
}

#[must_use]
pub fn native_cell_bytes(dtype: Dtype, numel: usize) -> usize {
    match dtype {
        Dtype::Bool => numel.max(1),
        Dtype::Bf16 => numel.max(1) * 2,
        _ => numel.max(1) * 4,
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

/// What an instance of a program keeps in the pool: its own item, the rings' registry and
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
    /// Where a reader of captured scores keeps a capturing lane's rows in
    /// the instance's item (`crate::scores`); `None` for any other program.
    pub score_at: Option<u64>,
    /// Where a beam group keeps its page lists in the instance's item:
    /// zeroed with the item at bind, so the group's first fire primes it. `None` for any other.
    pub beam: Option<BeamLayout>,
}

/// A beam group's resident state in its instance's item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeamLayout {
    /// Where it starts in the item.
    pub at: u64,
    pub lanes: u32,
    /// The list width and pool pages it is laid for.
    pub cap: u32,
    /// Its ring of pending pages.
    pub frames: u32,
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
            score_at: None,
            beam: None,
        }
    }

    /// The same item with a beam group of `lanes` beams' state past it,
    /// laid for `cap` pages a list and pool and `frames` pending rings.
    #[must_use]
    pub fn with_beam(mut self, lanes: u32, cap: u32, frames: u32) -> Held {
        let at = self.instance_bytes.next_multiple_of(CELL_ALIGN);
        let words = kernels_cuda::runtime::channel::beam_state_words(lanes, cap, frames);
        self.beam = Some(BeamLayout {
            at,
            lanes,
            cap,
            frames,
        });
        self.instance_bytes = at + words * 4;
        self
    }

    /// The same item with a capturing lane's rows of `bytes` past it.
    #[must_use]
    pub fn with_scores(mut self, bytes: u64) -> Held {
        let at = self.instance_bytes.next_multiple_of(CELL_ALIGN);
        self.score_at = Some(at);
        self.instance_bytes = at + bytes;
        self
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
        let cells = self.cells(place)?;
        self.rebase_to(place, cells);
        Ok(())
    }

    /// Each ring's cell at `place`: refused where a ring only the device
    /// touches has nowhere to lie. Nothing is pointed.
    pub fn cells(&self, place: &Place) -> Result<Vec<u64>> {
        let mut cells = Vec::with_capacity(self.shapes.len());
        for channel in 0..self.shapes.len() {
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
        Ok(cells)
    }

    /// The rings pointed at `place`, their `cells` there ([`Self::cells`]).
    pub fn rebase_to(&mut self, place: &Place, cells: Vec<u64>) {
        let slots = self.shapes.len();
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
    }

    /// Where the instance's item lies now.
    #[must_use]
    pub const fn instance_base(&self) -> u64 {
        self.registry
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
    /// Where this stage's tables and scratch lie in its pair's state, laid
    /// by [`Prepared::lay`]: each an address in it.
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
    /// offsets), pending flags, row tables and lane scratch table, one span
    /// staged in one copy a commit (`cell_at` their offsets, `cells_host`
    /// its image).
    cells: u64,
    cell_at: [usize; CELLS],
    cells_host: Vec<u8>,
    /// Each lane's scratch base, the device address the grouped kernels
    /// read at their `lane_scratch` parameter
    /// ([`eta_compiler::codegen::lane_scratch::LANE_SCRATCH`]).
    lane_scratch_host: Vec<u64>,
    intrinsic_bases_host: Vec<u64>,
    intrinsic_modes_host: Vec<u32>,
    intrinsic_widths_host: Vec<u32>,
    intrinsic_strides_host: Vec<u32>,
    intrinsic_offsets_host: Vec<u32>,
    intrinsic_rows_host: Vec<u32>,
    /// The side-table slots a lane binds ([`eta_ir::op::output_pitch`]).
    pitch: usize,
    /// Each slot's row table: where it starts among a lane's pointers and
    /// the rows it holds (0 for a slot no output op reads), for an output
    /// whose rows are the readout rows of an instance's lanes, each its own
    /// seat ([`Prepared::bind_rows`]).
    row_tables: Vec<(u32, u32)>,
    /// A lane's row-table pointers, every slot's.
    row_words: u32,
    row_pointers_host: Vec<u64>,
    /// The outputs this stage reads, an output's slot its index among them
    /// plus `output_slots` a lane.
    names: Vec<String>,
    output_slots: u32,
    channel_count: u32,
    value_count: u32,
    scratch_stride: u32,
    temporary_offset: u32,
    temporary_bytes: u32,
    region_rows: Vec<u32>,
    samplers: Vec<bool>,
    /// The most blocks each select split across a lane's blocks takes
    /// ([`splits`]), a region each.
    splits: Vec<Option<u32>>,
    /// The rows of each region whose rows split across blocks
    /// ([`splits_rows`]).
    row_splits: Vec<Option<u32>>,
    /// The most blocks each such region's rows split across
    /// ([`row_split_max`]), a region each.
    row_split_most: Vec<u32>,
    /// The launches (phases) each region runs in ([`LaunchRegion::phases`]):
    /// a region compiled to split its rows, once a phase, whether or not
    /// this fire splits them (past [`LANE_SAMPLER_BLOCKS`] rows each runs
    /// on one block, its phases one after another); every other region once.
    phases: Vec<u32>,
    lanes: u32,
    filled: u32,
    bindings: Vec<u32>,
}

/// What every span of a stage starts a whole number of in its pair's state.
pub const SPAN_ALIGN: u64 = 256;

/// The spans a stage lays: its params, offsets, table, descriptors and
/// cells.
const SPANS: u64 = 5;

/// The value scratch a lane of stages whose widest lane scratch is
/// `stride` bytes holds: one region every stage's lane lies at. A program's
/// stages run one after another on one stream, each over values it writes
/// before it reads them, and none reads another's scratch: what passes
/// between stages passes through channels. So the stages share one
/// region, as wide as the widest, and none lays its own.
#[must_use]
pub fn lane_scratch(stride: u64) -> u64 {
    stride.max(SCRATCH_ALIGN).next_multiple_of(SPAN_ALIGN)
}

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
            // A row of a region that splits its rows runs on up to
            // [`row_split_max`] blocks.
            RegionKind::Generated if splits_rows(region, rows) => {
                rows.saturating_mul(row_split_max(plan, descriptors, region, rows))
            }
            RegionKind::Generated => rows,
            kind if lane_sampler(kind) => LANE_SAMPLER_BLOCKS,
            RegionKind::Library(LibraryOp::TopK) if splits(plan, region, rows).is_some() => 1,
            RegionKind::Library(LibraryOp::TopK) if selects(plan, region) => rows,
            RegionKind::Library(LibraryOp::TopK | LibraryOp::Sort) => rows.min(ORDER_ROW_BLOCKS),
            RegionKind::Library(_) => 1,
        })
        .collect()
}

/// The most blocks a row of a region that splits its `rows` runs across, a
/// power of two: as many as the device holds at once with them, and no more
/// than its row's vector passes (a block of
/// [`super::compile::MAX_BLOCK_THREADS`] threads, four elements a thread),
/// rounded up to a power of two: more blocks would stream with idle threads
/// and still pay every meet. A
/// region whose rows argmax in place keeps its parts' candidates in the lane
/// arena's sixteen slots a row, so it splits across [`LANE_SAMPLER_BLOCKS`]
/// at most.
fn row_split_max(
    plan: &LaunchStagePlan,
    descriptors: &[ValueDesc],
    region: &LaunchRegion,
    rows: u32,
) -> u32 {
    let argmaxes = region.nodes.iter().any(|&node| {
        plan.ops
            .get(node as usize)
            .is_some_and(|op| matches!(op.tag, tags::REDUCE_ARGMAX | tags::GUMBEL_ARGMAX))
    });
    // The widest row it streams: its reductions' inputs', its own inputs'
    // and outputs'.
    let reduced = region
        .nodes
        .iter()
        .filter_map(|&node| plan.ops.get(node as usize))
        .filter(|op| matches!(op.tag, tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN))
        .filter_map(|op| op.args.first().copied());
    let width = reduced
        .chain(region.inputs.iter().copied())
        .chain(region.outputs.iter().copied())
        .filter_map(|value| descriptors.get(value as usize))
        .map(|row| row.last)
        .max()
        .unwrap_or(1)
        .max(1);
    let passes = width
        .div_ceil(super::compile::MAX_BLOCK_THREADS * 4)
        .next_power_of_two();
    let held = (device_units() / rows.max(1)).clamp(1, passes);
    let most = if argmaxes { held.min(LANE_SAMPLER_BLOCKS) } else { held };
    1u32 << (31 - most.leading_zeros())
}

/// Whether `region`'s kernel runs each of its `rows` across several blocks
/// (its rows meet at a word each of the lane arenas' barrier line, which
/// holds a word for each of [`LANE_SAMPLER_BLOCKS`] rows).
fn splits_rows(region: &LaunchRegion, rows: u32) -> bool {
    region.splits_rows && (1..=LANE_SAMPLER_BLOCKS).contains(&rows)
}

/// The most blocks `region` splits across, where it is a select over a
/// lane's one row (`rows` 1) narrow enough a cut to split
/// ([`eta_compiler::codegen::cuda::order::top_k_splits`]): its blocks' merged
/// candidates meet in the lane's first arena.
fn splits(plan: &LaunchStagePlan, region: &LaunchRegion, rows: u32) -> Option<u32> {
    let op = region
        .nodes
        .first()
        .and_then(|&node| plan.ops.get(node as usize))?;
    (rows == 1
        && selects(plan, region)
        && eta_compiler::codegen::cuda::order::top_k_splits(op.imm))
    .then(|| eta_compiler::codegen::cuda::order::top_k_split_blocks(op.imm))
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
        .zip(region_rows(plan, descriptors))
        .map(|((region, &blocks), rows)| {
            let need = block_temporary(plan, descriptors, region);
            if region.kind == RegionKind::Generated && splits_rows(region, rows) {
                // A split row's blocks fold in shared memory: only the row's
                // first level of partials lies here, a row's area each,
                // whatever its split (its kernel strides the area by row).
                return need
                    .checked_next_multiple_of(SCRATCH_ALIGN)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(u64::from(rows.max(1)));
            }
            need.checked_next_multiple_of(SCRATCH_ALIGN)
                .unwrap_or(u64::MAX)
                .saturating_mul(u64::from(blocks.max(1)))
        })
        .max()
        .unwrap_or(0)
}

/// The bytes a block of `region` keeps in the temporary area: what its
/// kernel does, by the emitter that writes it. This is the SIMT emitter's
/// published need, and it stands alone: the shared whole-value floor
/// (`eta_exec::stage_temporaries`, 16 B an element of the widest value) is
/// a figure for kernels that sort or reduce a whole value in the area,
/// which no SIMT kernel does past what is counted here (a select keeps its
/// candidates in shared memory, a sampler in the engine's lane arenas, an
/// order block two rows of indices), and it would lay a top-p lane its
/// vocabulary's 16 B an element where its kernels keep none.
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
        // nothing else, laid from the rows its reductions stream (each
        // reduction's input's row, `stream_width`), not the stage's widest
        // value: a beam step's flattened candidates are a value of the
        // stage no reduction of its log-softmax streams.
        RegionKind::Generated => {
            let streamed = region
                .nodes
                .iter()
                .filter_map(|&node| plan.ops.get(node as usize))
                .filter(|op| {
                    matches!(
                        op.tag,
                        tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN
                    )
                })
                .filter_map(|op| descriptors.get(*op.args.first()? as usize))
                .map(|d| u64::from(d.last))
                .max()
                .unwrap_or(1)
                .max(1);
            eta_compiler::codegen::launch::generated_temporary_bytes(plan, region, streamed)
        }
        // A sampler's blocks meet in the engine's lane arenas, not here.
        kind if lane_sampler(kind) => 0,
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

fn lay_out(plan: &LaunchStagePlan, descriptors: &[ValueDesc]) -> Result<Layout> {
    let blocks = region_blocks(plan, descriptors);
    let temporary = temporary_bytes(plan, descriptors, &blocks);
    eta_exec::lay_out(plan, descriptors, &launches(plan, &blocks, temporary)).map_err(|why| {
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

impl Prepared {
    pub fn build(
        plan: &LaunchStagePlan,
        shapes: &[ChannelShape],
        extents: Extents,
    ) -> Result<Prepared> {
        let channel_count = u32::try_from(plan.channel_bindings.len())
            .map_err(|_| Fault::program("program::launch", "more channels than a u32 can count"))?;
        let value_count = u32::try_from(plan.value_types.len())
            .map_err(|_| Fault::program("program::launch", "more values than a u32 can count"))?;

        let descriptors = describe_values(plan, extents)?;
        let scratch_layout = lay_out(plan, &descriptors)?;
        let scratch_stride = u32::try_from(scratch_layout.total)
            .map_err(|_| Fault::program("program::launch", "a scratch stride past a u32"))?;
        let temporary_offset = u32::try_from(scratch_layout.temporary)
            .map_err(|_| Fault::program("program::launch", "a temporary offset past a u32"))?;
        let temporary_bytes = u32::try_from(scratch_layout.temporary_bytes)
            .map_err(|_| Fault::program("program::launch", "a temporary arena past a u32"))?;
        let region_rows = region_blocks(plan, &descriptors);
        let rows_of = eta_exec::region_rows(plan, &descriptors);
        // A sampler, or a select split across a lane's blocks: both meet in
        // the lane arenas, on as many blocks as the device holds with the
        // fire's lanes.
        let samplers = plan
            .fused
            .iter()
            .map(|region| lane_sampler(region.kind))
            .collect();
        let splits = plan
            .fused
            .iter()
            .zip(&rows_of)
            .map(|(region, &rows)| splits(plan, region, rows))
            .collect();
        let row_splits = plan
            .fused
            .iter()
            .zip(&rows_of)
            .map(|(region, &rows)| splits_rows(region, rows).then_some(rows))
            .collect();
        let row_split_most = plan
            .fused
            .iter()
            .zip(&rows_of)
            .map(|(region, &rows)| row_split_max(plan, &descriptors, region, rows))
            .collect();
        let phases = plan.fused.iter().map(|region| region.phases.max(1)).collect();

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

        // A slot's row table holds the rows its output op reads, from the
        // op's own first.
        let mut row_rows = vec![0u32; plan.output_pitch as usize];
        let mut base = 0u32;
        for op in &plan.ops {
            if op.tag == tags::OUTPUT {
                let slot = op.name_index + plan.output_slots * op.imm;
                let rows = descriptors.get(base as usize).map_or(1, |d| d.rows.max(1));
                if let Some(held) = row_rows.get_mut(slot as usize) {
                    *held = (*held).max(op.imm2 + rows);
                }
            }
            base += u32::from(op.result_count);
        }
        let mut row_words = 0u32;
        let row_tables: Vec<(u32, u32)> = row_rows
            .iter()
            .map(|&rows| {
                let at = row_words;
                row_words += rows;
                (at, rows)
            })
            .collect();

        let offsets_host: Vec<u8> = scratch_layout
            .values
            .iter()
            .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
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
            lane_scratch_host: Vec::new(),
            intrinsic_bases_host: Vec::new(),
            intrinsic_modes_host: Vec::new(),
            intrinsic_widths_host: Vec::new(),
            intrinsic_strides_host: Vec::new(),
            intrinsic_offsets_host: Vec::new(),
            intrinsic_rows_host: Vec::new(),
            pitch: plan.output_pitch as usize,
            row_tables,
            row_words,
            row_pointers_host: Vec::new(),
            names: plan.names.clone(),
            output_slots: plan.output_slots,
            channel_count,
            value_count,
            scratch_stride,
            temporary_offset,
            temporary_bytes,
            region_rows,
            samplers,
            splits,
            row_splits,
            row_split_most,
            phases,
            lanes: 0,
            filled: 0,
            bindings: plan.channel_bindings.clone(),
        })
    }

    /// The bytes of the spans whose size no lane count changes: the params
    /// and offsets, the table's header, and the room every span's
    /// alignment may take.
    #[must_use]
    pub fn fixed_bytes(&self) -> u64 {
        (self.params_host.len().max(size_of::<CudaOpParams>())
            + self.offsets_host.len().max(size_of::<u32>())) as u64
            + LANE_HEADER_BYTES
            + SPANS * SPAN_ALIGN
            // Each cell [`Prepared::lay`] lays is rounded up to 16 bytes.
            + CELLS as u64 * 16
    }

    /// The bytes of each cell a stage lays for `lanes` lanes, before each
    /// is rounded up to 16 ([`Prepared::lay`]): its intrinsics' bases,
    /// modes, widths, strides, offsets and rows, its pending flags, its row
    /// pointers and its lane scratch table. What sizes the pair's state and
    /// what is laid in it are this one list.
    fn cell_bytes(&self, lanes: u32) -> [usize; CELLS] {
        let words = self.pitch * lanes as usize;
        let rows = self.row_words as usize * lanes as usize;
        let pending = (self.channel_count as usize * lanes as usize).max(1);
        [
            words * size_of::<u64>(),
            words * size_of::<u32>(),
            words * size_of::<u32>(),
            words * size_of::<u32>(),
            words * size_of::<u32>(),
            pending,
            rows * size_of::<u64>(),
            LaneScratch::bytes(lanes) as usize,
            words * size_of::<u32>(),
        ]
    }

    /// The bytes each lane adds to the stage's tables: its record and
    /// channel slots, its descriptor row and its intrinsic and pending
    /// cells. Its value scratch is its instance's ([`Prepared::scratch_stride`],
    /// [`lane_scratch`]).
    #[must_use]
    pub fn lane_bytes(&self) -> u64 {
        // Every cell grows a lane's share a lane (the pending flags at
        // least one a lane, as one lane lays them).
        let cells: usize = self.cell_bytes(1).iter().sum();
        LANE_RECORD_BYTES
            + u64::from(self.channel_count) * size_of::<LaneChannelSlot>() as u64
            + self.descriptor_row.len().max(1) as u64
            + cells as u64
    }

    /// A lane's value scratch: what the kernels step `dispatch_lane` by.
    #[must_use]
    pub const fn scratch_stride(&self) -> u32 {
        self.scratch_stride
    }

    /// Lays this stage's tables for `lanes` lanes in its pair's state from
    /// `at` up (its lanes' scratch is each one's instance's,
    /// [`Prepared::stage_lane`]), and returns where its tables end. `fresh`: the
    /// state moved or came back since this stage last lay there, so the
    /// tables no lane changes are staged again.
    pub fn lay(
        &mut self,
        extents: Extents,
        lanes: u32,
        at: u64,
        fresh: bool,
        stream: *mut core::ffi::c_void,
    ) -> Result<u64> {
        self.filled = 0;
        let lanes = lanes.max(1);
        let mut next = at.next_multiple_of(SPAN_ALIGN);
        let mut take = |bytes: usize| {
            let here = next;
            next = (here + bytes as u64).next_multiple_of(SPAN_ALIGN);
            here
        };
        let params = take(self.params_host.len().max(size_of::<CudaOpParams>()));
        let offsets = take(self.offsets_host.len().max(size_of::<u32>()));
        let moved = fresh || params != self.params || lanes != self.lanes;
        self.params = params;
        self.offsets = offsets;
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

        let row = self.descriptor_row.len().max(1);
        self.descriptors = take(row * lanes as usize);
        if lanes != self.lanes {
            self.descriptors_host = self
                .descriptor_row
                .iter()
                .copied()
                .cycle()
                .take(self.descriptor_row.len() * lanes as usize)
                .collect();
        }

        let words = self.pitch * lanes as usize;
        self.intrinsic_bases_host.resize(words, 0u64);
        self.intrinsic_modes_host.resize(words, 0u32);
        self.intrinsic_widths_host.resize(words, 0u32);
        self.intrinsic_strides_host.resize(words, 0u32);
        self.intrinsic_offsets_host.resize(words, 0u32);
        self.intrinsic_rows_host.resize(words, 0u32);
        let rows = self.row_words as usize * lanes as usize;
        self.row_pointers_host.resize(rows, 0u64);
        self.lane_scratch_host.resize(lanes as usize, 0u64);
        let mut cell_at = 0usize;
        for (cell, bytes) in self.cell_bytes(lanes).into_iter().enumerate() {
            self.cell_at[cell] = cell_at;
            cell_at += bytes.div_ceil(16) * 16;
        }
        if self.cells_host.len() != cell_at {
            self.cells_host = vec![0u8; cell_at];
        }
        self.cells = take(cell_at);
        self.lanes = lanes;
        if moved {
            crate::device::stage_raw(stream, self.params, &self.params_host)?;
            crate::device::stage_raw(stream, self.offsets, &self.offsets_host)?;
            crate::device::stage_raw(stream, self.descriptors, &self.descriptors_host)?;
        }
        Ok(next)
    }

    /// Stages the next lane: its tables' rows, and `scratch`, its
    /// instance's value scratch, in the lane scratch table.
    pub fn stage_lane(
        &mut self,
        rings: &Rings,
        cursors: &[Cursor],
        commit: u64,
        scratch: u64,
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
        self.intrinsic_rows_host[row..row + self.pitch].fill(0);
        let words = self.row_words as usize;
        self.row_pointers_host[lane as usize * words..(lane as usize + 1) * words].fill(0);
        // The lane's scratch: its instance's own, where it lies now.
        self.lane_scratch_host[lane as usize] = scratch;
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
        image[at[5]..at[6]].fill(0);
        let rows = slice_bytes(&self.row_pointers_host);
        image[at[6]..at[6] + rows.len()].copy_from_slice(&rows);
        let bases = slice_bytes(&self.lane_scratch_host);
        image[at[LANE_SCRATCH_CELL]..at[LANE_SCRATCH_CELL] + bases.len()].copy_from_slice(&bases);
        let bounds = slice_bytes(&self.intrinsic_rows_host);
        image[at[INTRINSIC_ROWS_CELL]..at[INTRINSIC_ROWS_CELL] + bounds.len()]
            .copy_from_slice(&bounds);
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

    /// The rows this stage reads of each output it reads, by name and
    /// lane: from an op's own first row, the most any op reads.
    pub fn rows_read(&self) -> impl Iterator<Item = (&str, u32, u32)> + '_ {
        let slots = self.output_slots.max(1) as usize;
        self.row_tables
            .iter()
            .enumerate()
            .filter(|(_, (_, rows))| *rows > 0)
            .filter_map(move |(slot, &(_, rows))| {
                let name = self.names.get(slot % slots)?;
                Some((name.as_str(), u32::try_from(slot / slots).ok()?, rows))
            })
    }

    /// Whether this stage reads output `name` of `lane`.
    #[must_use]
    pub fn reads(&self, name: &str, lane: u32) -> bool {
        self.slot_of(name, lane)
            .and_then(|slot| self.row_tables.get(slot as usize))
            .is_some_and(|&(_, rows)| rows > 0)
    }

    /// Binds output slot `slot` of `lane` to rows each its own: row `r` the
    /// `width` wide row at `rows[r]`, f32 (`bf16` false) or bf16.
    pub fn bind_rows(
        &mut self,
        lane: u32,
        slot: u32,
        rows: &[u64],
        width: u32,
        bf16: bool,
    ) -> Result<()> {
        let Some(&(at, held)) = self.row_tables.get(slot as usize) else {
            return Err(Fault::program(
                "program::launch",
                format!("slot {slot} is past the {} this stage's outputs take", self.pitch),
            ));
        };
        // Every pointer the table holds is written: the stage reads all
        // `held` of them, and a seat past those is a row it does not read.
        let Some(rows) = rows.get(..held as usize) else {
            return Err(Fault::Ceiling {
                what: "rows seated for a stage's output read a row a lane",
                need: u64::from(held),
                have: rows.len() as u64,
            });
        };
        if lane >= self.lanes {
            return Err(Fault::program(
                "program::launch",
                format!("lane {lane} is past a batch cut for {}", self.lanes),
            ));
        }
        let first = lane as usize * self.row_words as usize + at as usize;
        self.row_pointers_host[first..first + rows.len()].copy_from_slice(rows);
        let table = self.cells
            + self.cell_at[6] as u64
            + first as u64 * size_of::<u64>() as u64;
        let mode = if bf16 {
            INTRINSIC_STORAGE_ROW_POINTERS
        } else {
            INTRINSIC_STORAGE_ROW_POINTERS_F32
        };
        self.bind_intrinsic(lane, slot, table, mode, width, width, 0, held)
    }

    /// Binds output slot `slot` of `lane` to `width` wide rows at `base`,
    /// read from row `row_offset` on and never at row `rows` or past it.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_intrinsic(
        &mut self,
        lane: u32,
        slot: u32,
        base: u64,
        storage: u32,
        width: u32,
        row_stride: u32,
        row_offset: u32,
        rows: u32,
    ) -> Result<()> {
        let slot = slot as usize;
        if slot >= self.pitch {
            return Err(Fault::program(
                "program::launch",
                format!("slot {slot} is past the {} this stage's outputs take", self.pitch),
            ));
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
        self.intrinsic_rows_host[at] = rows;
        Ok(())
    }

    fn cell(&self, cell: usize) -> u64 {
        self.cells + self.cell_at[cell] as u64
    }

    pub fn launch_region(
        &self,
        region: &Region,
        arenas: Arenas,
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
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
            .ptr(self.cell(LANE_SCRATCH_CELL))
            .u32(self.value_count)
            .u32(self.temporary_offset)
            .ptr(self.cell(5))
            .ptr(self.cell(0))
            .ptr(self.cell(1))
            .ptr(self.cell(2))
            .ptr(self.cell(3))
            .ptr(self.cell(4))
            .ptr(self.cell(INTRINSIC_ROWS_CELL));
        let rows = self
            .region_rows
            .get(region.region_index as usize)
            .copied()
            .unwrap_or(1)
            .max(1);
        let at = region.region_index as usize;
        let sampler = self.samplers.get(at).copied().unwrap_or(false);
        // A sampler lane runs on as many blocks as the device holds at once
        // with the fire's lanes (its blocks wait on one another), in its own
        // span of the engine's lane arenas: its blocks' arenas and its
        // gather buffers; the lanes' barrier words start at zero.
        let (rows, strided) = if let Some(Some(most)) = self.splits.get(at).copied() {
            // A split select's lane runs on as many blocks as the device
            // holds with the fire's lanes, its candidates and its count in
            // its first arena; its last block merges (none waits).
            let blocks = (device_units() / self.filled.max(1)).clamp(1, most);
            let need = u64::from(self.filled) * lane_span(1);
            if need > arenas.bytes {
                return Err(Fault::Ceiling {
                    what: "bytes of the engine's lane arenas a fire's split selects meet in",
                    need,
                    have: arenas.bytes,
                });
            }
            self.zero_barriers(arenas, lane_span(1), stream)?;
            (blocks, blocks)
        } else if sampler {
            // A sampler lane's blocks wait on one another at its barrier
            // (`M1Lane::barrier`), so every block of the launch is resident
            // at once: the lanes' blocks are never more than the units this
            // process's queues run on ([`device_units`], a CU mask's and a
            // partition's and MPS's count, not the device's) hold of this
            // kernel at once. Nothing in the process spins waiting on a
            // sampler, so another launch's blocks only delay its own; a lane
            // on one block waits on no other.
            let units = device_units();
            let rows = lane_blocks(units, self.filled).min(rows);
            let held = u64::from(units) * u64::from(region.module.resident());
            if rows > 1 && u64::from(self.filled) * u64::from(rows) > held {
                return Err(Fault::program(
                    "cuLaunchKernel",
                    format!(
                        "`{}`: {} lanes' samplers on {rows} blocks each wait on one another, \
                         and the {units} compute units this process runs on hold {held} of its \
                         blocks at once",
                        region.module.entry_name(),
                        self.filled
                    ),
                ));
            }
            let need = u64::from(self.filled) * lane_span(rows);
            if need > arenas.bytes {
                return Err(Fault::Ceiling {
                    what: "bytes of the engine's lane arenas a fire's samplers meet in",
                    need,
                    have: arenas.bytes,
                });
            }
            self.zero_barriers(arenas, lane_span(rows), stream)?;
            (rows, rows)
        } else if let Some(Some(region_rows)) = self.row_splits.get(at).copied() {
            // Each row on as many blocks as the device holds at once with
            // the fire's lanes' rows (they wait on one another), meeting at
            // its word of its lane's barrier line.
            let most = self.row_split_most.get(at).copied().unwrap_or(1).max(1);
            let held = (device_units() / self.filled.saturating_mul(region_rows).max(1)).clamp(1, most);
            let split = 1u32 << (31 - held.leading_zeros());
            if split > 1 {
                let need = u64::from(self.filled) * lane_span(1);
                if need > arenas.bytes {
                    return Err(Fault::Ceiling {
                        what: "bytes of the engine's lane arenas a fire's split rows meet in",
                        need,
                        have: arenas.bytes,
                    });
                }
                self.zero_barriers(arenas, lane_span(1), stream)?;
            }
            (region_rows * split, region_rows)
        } else {
            (rows, rows)
        };
        // A split row's temporary area is its row's (its blocks fold in
        // shared memory), strided by row; every other block's is its own.
        let stride = (self.temporary_bytes / strided) & !(SCRATCH_ALIGN as u32 - 1);
        args.u32(rows).u32(stride).ptr(arenas.at).u32(0);
        // A value stated to take no bytes lies on the lane's dummy bytes: a
        // build with `test-slot-fill` poisons them before every launch, so a
        // kernel that reads one answers wrongly where a golden sees it.
        // Without the feature no build writes anything here.
        if cfg!(feature = "test-slot-fill") {
            self.poison_dummy(stream)?;
        }
        // A region whose rows split runs a launch a phase: every block's
        // partials are written by a launch's end, and the next folds them, so
        // no block of it waits on another (its grid need not be resident at
        // once).
        let phases = self.phases.get(at).copied().unwrap_or(1).max(1);
        let phase_at = args.len() - 1;
        for phase in 0..phases {
            args.set(phase_at, u64::from(phase));
            launch(
                &region.module,
                self.filled.saturating_mul(rows),
                region.module.block_threads(),
                &mut args,
                FUSED_ARITY,
                stream,
            )?;
        }
        Ok(())
    }

    /// Whether a region of this stage meets across its lane's blocks in the
    /// engine's lane arenas (a lane sampler, a split select, a split row):
    /// its blocks may wait for one another, so their grid is resident at
    /// once, and the arenas are one buffer every launch's lanes lay at the
    /// same offsets, so no other group's launch (another program's select,
    /// a decode's sampler) may run beside it on a side stream.
    #[must_use]
    pub fn meets_across_blocks(&self) -> bool {
        self.samplers.iter().any(|&sampler| sampler)
            || self.splits.iter().any(Option::is_some)
            || self.row_splits.iter().any(Option::is_some)
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

    /// Rebinds the scalar at `at` (a bound argument) to `value`.
    pub fn set(&mut self, at: usize, value: u64) {
        if let Some(cell) = self.storage.get_mut(at) {
            **cell = value;
        }
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
    /// The fire's lanes' sampler barrier words (the first 64 bytes of each
    /// lane's temporary arena: a word a block) back to zero.
    /// The first line of each lane's arenas, `pitch` apart from the lane
    /// arenas' start: the lanes' barrier words.
    fn zero_barriers(
        &self,
        arenas: Arenas,
        pitch: u64,
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        #[cfg(feature = "device")]
        {
            use crate::device::dr;
            // SAFETY: the lane arenas hold `filled` lanes' blocks `pitch`
            // apart, checked against their bytes before this.
            let code = unsafe {
                dr::cuMemsetD2D32Async(
                    arenas.at,
                    pitch as usize,
                    0,
                    16,
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
        let _ = (arenas, pitch, stream);
        Ok(())
    }

    /// The fire's lanes' dummy bytes (the first [`SCRATCH_DUMMY_BYTES`] of
    /// each lane's scratch, where every value that takes no bytes lies)
    /// filled with [`SCRATCH_POISON`].
    fn poison_dummy(&self, stream: *mut core::ffi::c_void) -> Result<()> {
        #[cfg(feature = "device")]
        {
            use crate::device::dr;
            for &scratch in &self.lane_scratch_host[..self.filled as usize] {
                // SAFETY: each lane's scratch, its instance's, is no shorter
                // than its dummy bytes.
                let code = unsafe {
                    dr::cuMemsetD2D32Async(
                        scratch,
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

/// The device's multiprocessors this process's queues run on, which a
/// sampler fire's blocks may fill and its lane arenas are sized by
/// ([`Arenas::bytes_for`]): the device's count (a partition's own, and under
/// CUDA's MPS the share it is limited to), less those a CU mask keeps its
/// queues off on ROCm ([`queue_units`]). A sampler's blocks meet at a
/// barrier, so it never runs on more than these hold at once.
/// The queue masks the load states ([`crate::serve::Knobs::queue_masks`]),
/// held for [`device_units`]: the process's, so the first load's.
static QUEUE_MASKS: std::sync::OnceLock<crate::serve::QueueMasks> =
    std::sync::OnceLock::new();

/// States the process's queue masks, ahead of any sampler laid
/// ([`device_units`]). They are the process's start's, so a later load's
/// are the same words and keep the first's.
pub fn state_queue_masks(masks: &crate::serve::QueueMasks) {
    let _ = QUEUE_MASKS.set(masks.clone());
}

/// Of device `ordinal`'s `units` compute units, those this process's queues
/// run on under the CU masks the load stated ([`state_queue_masks`]), and
/// named at load where a mask keeps any off.
#[cfg(feature = "rocm")]
#[must_use]
pub fn masked_units(units: u32, ordinal: u32) -> u32 {
    let masks = QUEUE_MASKS.get().cloned().unwrap_or_default();
    // The HSA agent HIP's ordinal names: the ordinal's entry of a
    // visible-devices list, where one is set.
    let agent = masks
        .visible_devices
        .as_deref()
        .and_then(|list| list.split(',').nth(ordinal as usize)?.trim().parse().ok())
        .unwrap_or(ordinal);
    let held = queue_units(
        units,
        agent,
        masks.hsa_cu_mask.as_deref(),
        masks.global_cu_mask.as_deref(),
    );
    if held != units {
        tracing::info!(
            "{held} of the device's {units} compute units are this process's \
             queues' (a CU mask); every grid, arena and sampler is laid over those"
        );
    }
    held
}

#[must_use]
pub fn device_units() -> u32 {
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
            let units = u32::try_from(count).unwrap_or(1).max(1);
            // On ROCm the count is the one the device context stated from the
            // process's CU masks (`masked_units`), as every launcher reads it.
            #[cfg(feature = "rocm")]
            {
                let _ = units;
                kernels_cuda::hip::multiprocessors(device.unsigned_abs()).unwrap_or(1).max(1)
            }
            #[cfg(not(feature = "rocm"))]
            units
        }
        #[cfg(not(feature = "device"))]
        {
            1
        }
    })
}

/// Of a device's `units` compute units, those its queues run on under ROCm's
/// CU masks: `hsa` (`HSA_CU_MASK`, `<agents>:<units>[;...]`, each list of
/// indices and `a-b` ranges, naming HSA agent `agent` or not) and `global`
/// (`ROC_GLOBAL_CU_MASK`, a hex mask HIP gives every queue it makes), both
/// applying where both are set. A mask this cannot read answers one unit:
/// blocks that wait on one another are then never more than it holds.
#[must_use]
pub fn queue_units(units: u32, agent: u32, hsa: Option<&str>, global: Option<&str>) -> u32 {
    fn indices(list: &str) -> Option<Vec<u32>> {
        let mut out = Vec::new();
        for part in list.split(',').map(str::trim).filter(|part| !part.is_empty()) {
            match part.split_once('-') {
                Some((a, b)) => {
                    let (a, b): (u32, u32) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
                    if a > b {
                        return None;
                    }
                    out.extend(a..=b.min(a.saturating_add(4096)));
                }
                None => out.push(part.parse().ok()?),
            }
        }
        Some(out)
    }
    let mut held: Vec<bool> = vec![true; units as usize];
    if let Some(hsa) = hsa.map(str::trim).filter(|mask| !mask.is_empty()) {
        for entry in hsa.split(';').map(str::trim).filter(|entry| !entry.is_empty()) {
            let Some((agents, list)) = entry.split_once(':') else {
                return 1;
            };
            let (Some(agents), Some(list)) = (indices(agents), indices(list)) else {
                return 1;
            };
            if agents.contains(&agent) {
                let named: std::collections::BTreeSet<u32> = list.into_iter().collect();
                for (unit, on) in held.iter_mut().enumerate() {
                    *on &= named.contains(&(unit as u32));
                }
            }
        }
    }
    if let Some(global) = global.map(str::trim).filter(|mask| !mask.is_empty()) {
        let hex = global.trim_start_matches("0x").trim_start_matches("0X");
        if hex.is_empty() || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return 1;
        }
        // Its last digit names units 0-3.
        let digits: Vec<u32> = hex.chars().rev().filter_map(|c| c.to_digit(16)).collect();
        for (unit, on) in held.iter_mut().enumerate() {
            let digit = digits.get(unit / 4).copied().unwrap_or(0);
            *on &= digit >> (unit % 4) & 1 == 1;
        }
    }
    u32::try_from(held.iter().filter(|&&on| on).count()).unwrap_or(1).max(1)
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

    /// The kernel steps a block's arena and a lane's span as the launcher
    /// lays the lane arenas, and they hold every lane a launch runs, its
    /// gather buffers with it however few blocks it runs on.
    #[test]
    fn the_lane_arenas_hold_every_launch_s_lanes() {
        let (stage, _) = shipped::compiled(&shipped::all(shipped::QWEN_VOCAB)[0].1, shipped::QWEN_VOCAB);
        let stage = &stage[0];
        let region = stage
            .fused
            .regions
            .iter()
            .find(|region| lane_sampler(region.kind))
            .expect("top-p has a sampler");
        let source = eta_compiler::codegen::cuda::emit_region("kernel", stage, region)
            .expect("the sampler emits");
        assert!(source.contains("g.stride = M1_LANE_ARENA_STRIDE;"));
        assert!(source.contains("M1_LANE_SPAN(lane_blocks)"));
        assert_eq!(
            lane_arena_stride(),
            (64 + 2 * 24_576 + 255) & !255,
            "the kernel's M1_LANE_ARENA_STRIDE"
        );
        assert_eq!(
            LANE_SAMPLER_GATHERS,
            16 * 2 * 512 * 8,
            "the kernel's M1_LANE_GATHERS"
        );
        for (units, flight) in [(142, 128), (304, 256), (1, 1), (8, 4), (20, 32)] {
            for filled in 1..=flight {
                assert!(
                    u64::from(filled) * lane_span(lane_blocks(units, filled))
                        <= Arenas::bytes_for(flight, units)
                );
            }
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
