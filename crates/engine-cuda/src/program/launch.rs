//! Device residency for one instance's channel rings, the lane-table bytes,
//! and the launch of a fire's regions.

use eta_compiler::codegen::launch::{LaunchRegion, LaunchStagePlan};
use eta_compiler::plan::{LibraryOp, RegionKind};
use eta_exec::{
    Extents, LANE_HEADER_BYTES, LANE_RECORD_BYTES, LaneChannelSlot, LaneHeader,
    LaneRecord, LaneShape, Layout, Lifetime, NO_TICKET, OpParams, OpRuntime, SCRATCH_ALIGN,
    ValueDesc, describe, layout, layout_reusing,
};
use eta_ir::Dtype;
use eta_ir::container::HostRole;
use eta_ir::op::{IntrinsicId, tags};
use eta_ir::types::{MAX_RANK, name_or_unknown};

use std::sync::Arc;

use kernels_cuda::channel::MAX_RING;

use crate::device::Buffer;
use crate::error::{Fault, Result};

use super::compile::{Module, Region};
use super::endpoint::Endpoint;

/// The sixteen arguments a generated fused region takes, matching
/// `fused_block1.cuh`'s signature. CUDA does not validate this: a mismatched
/// count reads garbage rather than erroring.
const FUSED_ARITY: usize = 18;

/// How many blocks per lane a `top_k`/`sort_desc` region gets: each sorts
/// every `ORDER_ROW_BLOCKS`-th row of the lane on its own slice of the
/// temporary arena (two `u32` order arrays over one row). A block per row
/// would want that arena per row — a gigabyte for 256 rows of 262 144.
pub const ORDER_ROW_BLOCKS: u32 = 32;

/// How many intrinsic slots the five side tables carry per lane. Tables are
/// indexed `lane * INTRINSIC_SLOTS + intrinsic`; a mismatch with the
/// kernel's own stride corrupts every intrinsic of every lane but the
/// first.
pub const INTRINSIC_SLOTS: usize = IntrinsicId::SLOTS as usize;

/// `IntrinsicStorageMode::F32` — the bound buffer holds `f32` elements.
pub const INTRINSIC_STORAGE_F32: u32 = 0;

/// `IntrinsicStorageMode::RawBf16` — the bound buffer holds raw `bf16`
/// elements the kernel widens as it reads.
pub const INTRINSIC_STORAGE_RAW_BF16: u32 = 1;

/// `IntrinsicStorageMode::RowPointers` — the bound buffer is a table of row
/// addresses, one `u64` per row, for a readout whose rows are not
/// consecutive. `row_offset` still indexes the table; `row_stride` is
/// ignored.
pub const INTRINSIC_STORAGE_ROW_POINTERS: u32 = 2;

/// `BoolStorageMode::NativeBytes` — one byte per lane, which is what every
/// device-side bool cell is.
pub const BOOL_STORAGE_NATIVE_BYTES: u32 = 0;

/// `BoolStorageMode::WirePacked` — one bit per lane, which is what a bool cell
/// becomes on the way to the host mirror.
pub const BOOL_STORAGE_WIRE_PACKED: u32 = 1;

// ─────────────────────────────────────────────────────────────────────────────
// The op record, in CUDA's spelling
// ─────────────────────────────────────────────────────────────────────────────

/// One op's parameters, in the layout the generated kernels read.
///
/// `#[repr(C)]`, field for field with the device struct `M1OpParams` (88
/// bytes: sixteen shared `u32`s plus five CUDA-only fields and a `u64
/// rng_seed`, whose alignment pads the struct to 88, not 84). Field order
/// must not change; the kernels index by offset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CudaOpParams {
    /// `PTIR_OP_*`.
    pub tag: u32,
    /// First argument's value slot.
    pub a0: u32,
    /// Second argument's value slot, or `pivot_threshold`'s predicate.
    pub a1: u32,
    /// Third argument's value slot.
    pub a2: u32,
    /// First result's value slot, or `a0` for an op with no results.
    pub o0: u32,
    /// Second result's value slot, or `o0` for an op with fewer than two.
    pub o1: u32,
    /// The op's immediate, or the vocabulary for an intrinsic.
    pub imm: u32,
    /// The op's second immediate, or the MTP draft row.
    pub imm2: u32,
    /// The op's third immediate.
    pub imm3: u32,
    /// RNG kind: 0 uniform, 1 gumbel.
    pub kind: u32,
    /// `pivot_threshold`'s predicate tag.
    pub pred_tag: u32,
    /// A const literal's dtype.
    pub lit_dtype: u32,
    /// A const literal's raw bits.
    pub lit_bits: u32,
    /// The stage-local channel slot a channel op targets.
    pub channel_slot: u32,
    /// `PTIR_INTR_*`, for `intrinsic_val`.
    pub intr: u32,
    /// The fixed cell size a `chan_put` writes into. The kernel faults every
    /// put whose logical bytes exceed it, so zero refuses.
    pub sink_bytes: u32,

    // Past here is CUDA's alone; the shared record ends above.
    /// How the intrinsic's buffer stores its elements:
    /// [`INTRINSIC_STORAGE_F32`] or [`INTRINSIC_STORAGE_RAW_BF16`]. A per-fire
    /// fact about the bound buffer, not about the trace.
    pub intrinsic_dtype: u32,
    /// How a bool cell is stored: [`BOOL_STORAGE_NATIVE_BYTES`] or
    /// [`BOOL_STORAGE_WIRE_PACKED`]. Always native on the device; packing
    /// happens at the host boundary.
    pub bool_storage: u32,
    /// Elements — not bytes — between rows in the intrinsic's buffer.
    pub intrinsic_row_stride: u32,
    /// Which row of the intrinsic's buffer this op reads.
    pub intrinsic_row_offset: u32,
    /// The per-op RNG seed. A `u64`: its eight-byte alignment is what pads the
    /// record to 88.
    pub rng_seed: u64,
}

/// The record's size, as `ptir_m1_runtime_prologue.cuh` asserts it.
const _: () = assert!(size_of::<CudaOpParams>() == 88);

/// Every field's offset, pinned individually — a size check alone would not
/// catch a field transposition.
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
    // 80..84 is the `u64`'s padding; asserting 84 here would be the bug.
    assert!(std::mem::offset_of!(CudaOpParams, rng_seed) == 80);
};

impl CudaOpParams {
    /// The shared record, widened, with CUDA's five extra fields at the
    /// defaults for an op that binds no intrinsic. Fields are copied by name
    /// rather than transmuted, so an added field in [`eta_exec::OpParams`] is
    /// a compile error here instead of a silent shift.
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

// ─────────────────────────────────────────────────────────────────────────────
// Cells: native on the device, packed on the wire
// ─────────────────────────────────────────────────────────────────────────────

/// Native device bytes for a cell of `numel` lanes of `dtype`: one byte per
/// bool lane, four per anything else.
#[must_use]
pub fn native_cell_bytes(dtype: Dtype, numel: usize) -> usize {
    if dtype == Dtype::Bool {
        numel.max(1)
    } else {
        numel.max(1) * 4
    }
}

/// A wire cell, as the device wants it.
///
/// # Errors
///
/// [`Fault::Program`] when `wire` is not exactly one wire cell; a short cell
/// reads real-looking garbage past its end.
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

/// A native cell, as the wire wants it.
///
/// Any nonzero byte is `true`: the device promises only nonzero-means-set, so
/// reading `== 1` would drop a `0xff` mask byte.
///
/// # Errors
///
/// [`Fault::Program`] when `native` is not exactly one native cell.
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

/// One channel's geometry, as the ring needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelShape {
    /// Lanes in one cell.
    pub numel: usize,
    /// The cell's element type.
    pub dtype: Dtype,
    /// How many unconsumed items the channel holds. The ring is one longer.
    pub capacity: u32,
}

impl ChannelShape {
    /// The channel a launch package declares at this slot.
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

    /// The ring length: `capacity + 1`, for the sentinel. Identical to the
    /// host half's, which is what makes a slot-for-slot diff meaningful.
    #[must_use]
    pub const fn ring(&self) -> u64 {
        self.capacity as u64 + 1
    }

    /// Native bytes in one cell.
    #[must_use]
    pub fn cell_bytes(&self) -> usize {
        native_cell_bytes(self.dtype, self.numel)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The rings
// ─────────────────────────────────────────────────────────────────────────────

/// One instance's channel state: the device cells, the device registry the
/// control kernels move, and the pinned endpoint of every channel with a
/// host end.
///
/// The registry holds the full/empty byte of `(slot, ring)`, the two ring
/// positions per slot, and each slot's `cap1`; `channel::commit_bump` is the
/// only writer. The host's [`Cursor`] is a prediction `channel::pull_validate`
/// checks against the live registry before commit.
#[derive(Debug)]
pub struct Rings {
    cells: Vec<Buffer>,
    shapes: Vec<ChannelShape>,
    /// The pinned mirror and counters of every channel with a host end;
    /// `None` for a channel whose cells never leave the device.
    endpoints: Vec<Option<Arc<Endpoint>>>,
    /// `head[slots]`, `tail[slots]`, `cap1[slots]`, `full[slots * MAX_RING]`,
    /// one allocation since they are indexed together.
    registry: Buffer,
    /// The addresses of the four arrays above, as the kernels take them.
    device: kernels_cuda::channel::Rings,
}

impl Rings {
    /// Allocate a zeroed ring per channel.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a channel whose cell is zero bytes (it holds
    /// nothing and can never be ready), or whatever `cudaMalloc` said.
    pub fn allocate(shapes: &[ChannelShape], endpoints: Vec<Option<Arc<Endpoint>>>) -> Result<Rings> {
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
        let mut cells = Vec::with_capacity(shapes.len());
        for (index, shape) in shapes.iter().enumerate() {
            let bytes = shape.cell_bytes();
            if bytes == 0 {
                return Err(Fault::program(
                    "program::launch",
                    format!("channel {index}'s cell is zero bytes, so it can never be ready"),
                ));
            }
            // A shared ring's cells belong to the channel, not this
            // instance; the placeholder keeps `cells` indexed densely.
            let shared = endpoints
                .get(index)
                .and_then(Option::as_ref)
                .and_then(|endpoint| endpoint.device_cells());
            cells.push(match shared {
                Some(_) => Buffer::zeroed(0)?,
                None => Buffer::zeroed(bytes * shape.ring() as usize)?,
            });
        }

        // One allocation: the u32 arrays first, then the full/empty bytes,
        // one per `(slot, ring)`, MAX_RING apart.
        let slots = shapes.len();
        let words = slots * size_of::<u32>();
        let full_at = 3 * words;
        let mut registry = Buffer::zeroed(full_at + slots * MAX_RING as usize)?;
        let cap1: Vec<u8> = shapes
            .iter()
            .flat_map(|shape| u32::try_from(shape.ring()).unwrap_or(u32::MAX).to_le_bytes())
            .collect();
        registry.write(2 * words as u64, &cap1)?;
        let device = kernels_cuda::channel::Rings::new(
            registry.at(full_at as u64)?,
            registry.ptr(),
            registry.at(words as u64)?,
            registry.at(2 * words as u64)?,
            u32::try_from(slots).unwrap_or(u32::MAX),
        );

        Ok(Rings {
            cells,
            shapes: shapes.to_vec(),
            endpoints,
            registry,
            device,
        })
    }

    /// The four arrays the control kernels move.
    #[must_use]
    pub const fn device(&self) -> kernels_cuda::channel::Rings {
        self.device
    }

    /// Channel `channel`'s pinned endpoint, or `None` when its cells never
    /// leave the device.
    #[must_use]
    pub fn endpoint(&self, channel: usize) -> Option<&Arc<Endpoint>> {
        self.endpoints.get(channel).and_then(Option::as_ref)
    }

    /// Seed the registry's ring positions and full bytes from the cursors.
    /// Bind-time only; on the fire path the registry belongs to
    /// `commit_bump` alone.
    ///
    /// # Errors
    ///
    /// Whatever the copies said.
    pub fn seed_registry(&mut self, cursors: &[Cursor]) -> Result<()> {
        let slots = self.shapes.len();
        let words = slots * size_of::<u32>();
        let mut head = Vec::with_capacity(words);
        let mut tail = Vec::with_capacity(words);
        let mut full = vec![0u8; slots * MAX_RING as usize];
        for (channel, shape) in self.shapes.iter().enumerate() {
            let ring = shape.ring();
            let cursor = cursors.get(channel).copied().unwrap_or(Cursor { head: 0, tail: 0 });
            head.extend_from_slice(&((cursor.head % ring) as u32).to_le_bytes());
            tail.extend_from_slice(&((cursor.tail % ring) as u32).to_le_bytes());
            for sequence in cursor.head..cursor.tail {
                let at = kernels_cuda::channel::Rings::full_at(
                    u32::try_from(channel).unwrap_or(u32::MAX),
                    (sequence % ring) as u32,
                ) as usize;
                if let Some(byte) = full.get_mut(at) {
                    *byte = 1;
                }
            }
        }
        self.registry.write(0, &head)?;
        self.registry.write(words as u64, &tail)?;
        self.registry.write((3 * words) as u64, &full)?;
        Ok(())
    }

    /// How many channels this instance carries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    /// Whether it carries none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// Channel `channel`'s geometry.
    #[must_use]
    pub fn shape(&self, channel: usize) -> Option<ChannelShape> {
        self.shapes.get(channel).copied()
    }

    /// The device address of channel `channel`'s cell at ring position
    /// `sequence`.
    ///
    /// `sequence` is a free-running cursor and is reduced modulo the ring
    /// here, not refused: the ring position is the cursor's residue.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when `channel` is not one this instance carries.
    pub fn cell_address(&self, channel: usize, sequence: u64) -> Result<u64> {
        let shape = self.shape_of(channel)?;
        let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
        match self.shared_slab(channel) {
            // The shared slab's bound is the endpoint's own: `cap1` cells of
            // this width, which is the same ring `shape.ring()` counts.
            Some(base) => Ok(base + at),
            None => self.cells[channel].at(at),
        }
    }

    /// Channel `channel`'s shared device slab, or `None` for one whose cells
    /// this session cut for itself. See
    /// [`Endpoint::device_cells`](super::Endpoint::device_cells).
    #[must_use]
    pub fn shared_slab(&self, channel: usize) -> Option<u64> {
        self.endpoints
            .get(channel)
            .and_then(Option::as_ref)
            .and_then(|endpoint| endpoint.device_cells())
    }

    /// Write one native cell into channel `channel` at `sequence`.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown channel or a cell of the wrong width;
    /// a short write leaves real-looking garbage in the cell's tail.
    pub fn write_cell(&mut self, channel: usize, sequence: u64, native: &[u8]) -> Result<()> {
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
        // A host-visible channel's cell lives in the pinned mirror; a shared
        // ring has no mirror, so its seed goes straight to the slab.
        if let Some(base) = self.shared_slab(channel) {
            let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
            crate::device::write_raw(base + at, native)?;
            // Also written to the pinned shadow, which `read_cell` reads.
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
        self.cells[channel].write(at, native)
    }

    /// Read channel `channel`'s cell at `sequence` back, in native form.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown channel, or whatever the copy said.
    pub fn read_cell(&self, channel: usize, sequence: u64) -> Result<Vec<u8>> {
        let shape = self.shape_of(channel)?;
        // A shared ring has no mirror; its committed cell is read from the
        // pinned shadow `channel::scatter_publish` writes (native bytes,
        // committed only, not pending).
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
        self.cells[channel].read(at, &mut out)?;
        Ok(out)
    }

    fn shape_of(&self, channel: usize) -> Result<ChannelShape> {
        self.shapes.get(channel).copied().ok_or_else(|| {
            Fault::program(
                "program::launch",
                format!("channel {channel} is not one this instance carries"),
            )
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// One stage's fire-path buffers
// ─────────────────────────────────────────────────────────────────────────────

/// Where a channel's two cells are, for one fire.
///
/// A `take`/`read` reads `committed`; a `put` writes `pending`. Both are
/// resolved on the host out of the ring cursors, because the kernel does no
/// ring arithmetic at all.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    /// The sequence a consumer reads.
    pub head: u64,
    /// The sequence a producer writes.
    pub tail: u64,
}
/// One stage's device state, cut for `lanes` lanes and keyed by `Extents`.
///
/// Nothing allocates on the fire path: every buffer is carved once (sized by
/// the plan, not the fire) and only a boundary's per-lane fields are
/// rewritten — its record, channel slots, and intrinsic row. `offsets` and
/// `params` are shared across lanes (they depend only on `Extents`); lane
/// channel/pending-flag arrays are indexed by the same `channel_slot_offset`.
#[derive(Debug)]
pub struct Prepared {
    /// Buffers a grow replaced, kept alive: `cudaFree` takes effect
    /// immediately but a previous boundary's enqueued kernels may still be
    /// reading them, so freeing here would be unsafe.
    retired: Vec<Buffer>,
    table: Buffer,
    /// The host mirror of the table. Kept so a boundary patches each lane's
    /// record and slots rather than rebuilding a header it already wrote.
    table_host: Vec<u8>,
    descriptors: Buffer,
    /// One lane's descriptor row, kept so a grow can repeat it.
    descriptor_row: Vec<u8>,
    params: Buffer,
    offsets: Buffer,
    scratch: Buffer,
    pending: Buffer,
    intrinsic_bases: Buffer,
    intrinsic_modes: Buffer,
    intrinsic_widths: Buffer,
    intrinsic_strides: Buffer,
    intrinsic_offsets: Buffer,
    /// Host mirrors of the five side tables, `lanes * INTRINSIC_SLOTS` each.
    /// A binding survives across fires, so these are restaged whole once
    /// per boundary rather than rebuilt.
    intrinsic_bases_host: Vec<u64>,
    intrinsic_modes_host: Vec<u32>,
    intrinsic_widths_host: Vec<u32>,
    intrinsic_strides_host: Vec<u32>,
    intrinsic_offsets_host: Vec<u32>,
    channel_count: u32,
    value_count: u32,
    scratch_stride: u32,
    temporary_offset: u32,
    /// The temporary arena's bytes, shared by a region's blocks per lane.
    temporary_bytes: u32,
    /// Blocks per lane for each region, by region index.
    region_rows: Vec<u32>,
    /// How many lanes every buffer above is cut for: a high-water mark that
    /// only grows, since a grow reallocates and is only safe before a
    /// boundary has staged anything.
    lanes: u32,
    /// How many lanes this boundary has taken: the launch's grid and the
    /// table header's `lane_count`, which the kernel treats as its own
    /// bound.
    filled: u32,
    /// This stage's local channel slot → the instance's dense channel index.
    bindings: Vec<u32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Which of a stage's values a fire actually carries
// ─────────────────────────────────────────────────────────────────────────────

/// Whether this shell launches `region`'s generated body. A
/// `Library(SecondParty)` region is a `kernel_call`/`sink_call` the shell
/// would have to run itself, so it does not run. Shared by the compiler and
/// [`describe_values`] so the two never drift apart.
#[must_use]
pub(crate) fn shell_launches(region: &LaunchRegion) -> bool {
    region.kind != RegionKind::Library(LibraryOp::SecondParty)
}

/// A value read only by a region this shell does not launch is not
/// materialised: described as empty when it is read only by a skipped
/// region's ops, read by none it does launch, and produced by
/// `chan_read`/`chan_take` (so an empty descriptor makes the emitted copy
/// fill nothing).
#[must_use]
fn read_only_by_skipped_regions(plan: &LaunchStagePlan) -> Vec<bool> {
    let values = plan.value_types.len();

    // Value -> the op that defines it.
    let mut producer = vec![usize::MAX; values];
    let mut result_base = 0u32;
    for (node, op) in plan.ops.iter().enumerate() {
        for result in 0..u32::from(op.result_count) {
            if let Some(slot) = producer.get_mut((result_base + result) as usize) {
                *slot = node;
            }
        }
        result_base += u32::from(op.result_count);
    }

    // Ops inside a skipped region, and the values they read.
    let mut skipped = vec![false; plan.ops.len()];
    let mut unread = vec![false; values];
    for region in plan.fused.iter().filter(|region| !shell_launches(region)) {
        for &node in &region.nodes {
            if let Some(node) = skipped.get_mut(node as usize) {
                *node = true;
            }
        }
        for &value in region.inputs.iter().chain(
            region
                .nodes
                .iter()
                .filter_map(|&node| plan.ops.get(node as usize))
                .flat_map(|op| op.args.iter()),
        ) {
            if let Some(value) = unread.get_mut(value as usize) {
                *value = true;
            }
        }
    }

    // Anything an op that runs reads is read, and that settles it.
    for (node, op) in plan.ops.iter().enumerate() {
        if skipped.get(node).copied().unwrap_or(false) {
            continue;
        }
        let predicate = (op.tag == tags::PIVOT_THRESHOLD).then_some(op.pred_payload);
        for value in op.args.iter().copied().chain(predicate) {
            if let Some(value) = unread.get_mut(value as usize) {
                *value = false;
            }
        }
    }
    // So is anything a launched region commits to a channel.
    for region in plan.fused.iter().filter(|region| shell_launches(region)) {
        for sink in &region.sinks {
            if let Some(value) = unread.get_mut(sink.value as usize) {
                *value = false;
            }
        }
    }

    // Condition 3: only a channel materialisation may be described as empty.
    for (value, drop) in unread.iter_mut().enumerate() {
        let tag = producer
            .get(value)
            .and_then(|&node| plan.ops.get(node))
            .map(|op| op.tag);
        if !matches!(tag, Some(tags::CHAN_READ | tags::CHAN_TAKE)) {
            *drop = false;
        }
    }
    unread
}

/// Every value's device descriptor for one fire's extents, with the ones
/// [`read_only_by_skipped_regions`] names emptied: rank and dtype stay, the
/// extents (and `len`) go to zero. A dropped value is still resolved first,
/// so a plan whose shapes do not resolve is still refused.
///
/// # Errors
///
/// [`Fault::Program`] when a value's shape does not resolve against
/// `extents`.
pub fn describe_values(plan: &LaunchStagePlan, extents: Extents) -> Result<Vec<ValueDesc>> {
    let empty = read_only_by_skipped_regions(plan);
    plan.value_types
        .iter()
        .enumerate()
        .map(|(value, declared)| {
            let described = describe(declared, &extents).map_err(|why| {
                Fault::program(
                    "program::launch",
                    format!("a value's shape does not resolve against this fire: {why:?}"),
                )
            })?;
            Ok(if empty.get(value).copied().unwrap_or(false) {
                ValueDesc {
                    len: 0,
                    rows: 0,
                    last: 0,
                    dims: [0; MAX_RANK],
                    ..described
                }
            } else {
                described
            })
        })
        .collect()
}

/// Every value's life on the launch clock. A generated region is one block
/// per lane running its nodes in emission order with a `__syncthreads()`
/// after each, so each of its nodes is a step of its own; a library region
/// is one kernel whose internal order is its own, so it is a single step.
/// Regions launch in index order on one stream. A value is defined at its
/// producing node's step and last read at the last step whose node names it
/// as an operand (a `pivot_threshold`'s predicate included), or whose region
/// names it as an input, output or sink. A reshape's consumers may read its
/// SOURCE (`cuda::fused` elides the copy), so a read of a reshape result also
/// extends the source's life. Only a result its op always writes in full may
/// take a vacated slot; a `pivot_threshold` (predicated) or a channel
/// materialisation (may be described empty) keeps a fresh, per-fire-zeroed
/// one. `None` when some op sits in no region, since then there is no launch
/// order to reason from.
fn value_lifetimes(plan: &LaunchStagePlan, rows: &[u32]) -> Option<Vec<Lifetime>> {
    // node -> its step and region; region -> its first and last step.
    let mut step_of = vec![u32::MAX; plan.ops.len()];
    let mut region_of = vec![u32::MAX; plan.ops.len()];
    let mut region_first = Vec::with_capacity(plan.fused.len());
    let mut region_last = Vec::with_capacity(plan.fused.len());
    let mut step = 0u32;
    for (index, region) in plan.fused.iter().enumerate() {
        let per_node = region.kind == RegionKind::Generated;
        region_first.push(step);
        for &node in &region.nodes {
            *step_of.get_mut(node as usize)? = step;
            *region_of.get_mut(node as usize)? = u32::try_from(index).ok()?;
            if per_node {
                step = step.checked_add(1)?;
            }
        }
        if !per_node {
            step = step.checked_add(1)?;
        }
        region_last.push(step.saturating_sub(1));
    }
    if step_of.contains(&u32::MAX) {
        return None;
    }
    let region_of_step = |step: u32| -> u32 {
        region_first
            .partition_point(|&first| first <= step)
            .saturating_sub(1) as u32
    };
    // How a launch's blocks touch a value: see `Lifetime::class_def`. A
    // many-block launch slices a value of its row geometry by row (class
    // 1 + the row's bytes), a per-row vector by element (class 2 + the
    // element's bytes), and reads anything else whole (class 0). The
    // kinds are the emitter's (`cuda::fused`, `Region::row_value` /
    // `row_alias`), computed here from the same symbolic dims.
    let class_of = |region: u32, value: u32| -> u64 {
        let Some(fused) = plan.fused.get(region as usize) else {
            return Lifetime::SEQUENTIAL;
        };
        if rows.get(region as usize).copied().unwrap_or(1) <= 1 {
            return Lifetime::SEQUENTIAL;
        }
        let geometry = fused
            .row_value
            .and_then(|witness| plan.value_types.get(witness as usize))
            .and_then(|ty| eta_compiler::plan::value_rows(&ty.axes));
        let Some(geometry) = geometry else {
            return 0;
        };
        let Some(ty) = plan.value_types.get(value as usize) else {
            return 0;
        };
        let elem: u64 = if ty.dtype == Dtype::Bool { 1 } else { 4 };
        let alias = fused.row_alias;
        if ty.axes.len() >= 2
            && eta_compiler::plan::value_rows(&ty.axes)
                .is_some_and(|shape| eta_compiler::plan::same_rows(shape, geometry, alias))
        {
            let width = match ty.axes.last() {
                Some(eta_compiler::plan::Dimension::Static(width)) => u64::from(*width),
                _ => return 0,
            };
            (1u64 << 40) | (width * elem)
        } else if eta_compiler::plan::is_row_vector(&ty.axes, geometry.0, geometry.1, alias) {
            (2u64 << 40) | elem
        } else {
            0
        }
    };
    let values = plan.value_types.len();
    let mut lifetimes = vec![
        Lifetime {
            def: 0,
            last: 0,
            reusable: false,
            launch_def: 0,
            launch_last: 0,
            class_def: Lifetime::SEQUENTIAL,
            class_last: Lifetime::SEQUENTIAL,
        };
        values
    ];
    let mut alias_of = vec![u32::MAX; values];
    let bump = |lifetimes: &mut Vec<Lifetime>, alias_of: &[u32], mut value: u32, step: u32| {
        // The value, then the source chain a reshape of it may read instead.
        for _ in 0..=alias_of.len() {
            let Some(life) = lifetimes.get_mut(value as usize) else {
                return;
            };
            life.last = life.last.max(step);
            match alias_of.get(value as usize).copied() {
                Some(source) if source != u32::MAX => value = source,
                _ => return,
            }
        }
    };
    let mut result_base = 0u32;
    for (node, op) in plan.ops.iter().enumerate() {
        let step = step_of[node];
        let written_in_full = !matches!(
            op.tag,
            tags::PIVOT_THRESHOLD | tags::CHAN_READ | tags::CHAN_TAKE
        );
        for result in 0..u32::from(op.result_count) {
            if let Some(life) = lifetimes.get_mut((result_base + result) as usize) {
                life.def = step;
                life.last = life.last.max(step);
                life.reusable = written_in_full;
                life.launch_def = region_of[node];
            }
        }
        if op.tag == tags::RESHAPE
            && let Some(&source) = op.args.first()
            && let Some(slot) = alias_of.get_mut(result_base as usize)
        {
            *slot = source;
        }
        result_base += u32::from(op.result_count);
        let predicate = (op.tag == tags::PIVOT_THRESHOLD).then_some(op.pred_payload);
        for value in op.args.iter().copied().chain(predicate) {
            bump(&mut lifetimes, &alias_of, value, step);
        }
    }
    for (region, fused) in plan.fused.iter().enumerate() {
        let step = region_last[region];
        let named = fused
            .inputs
            .iter()
            .chain(fused.outputs.iter())
            .copied()
            .chain(fused.sinks.iter().map(|sink| sink.value));
        for value in named {
            bump(&mut lifetimes, &alias_of, value, step);
        }
    }
    // Launches and classes; a value a many-block launch reads whole stays
    // until that launch has finished, since any of its blocks may still be
    // reading it.
    for (value, life) in lifetimes.iter_mut().enumerate() {
        let value = value as u32;
        life.launch_last = region_of_step(life.last);
        life.class_def = class_of(life.launch_def, value);
        life.class_last = class_of(life.launch_last, value);
        if life.class_last == 0 {
            life.last = life.last.max(region_last[life.launch_last as usize]);
        }
    }
    Some(lifetimes)
}

/// Blocks per lane for each of `plan.fused`'s regions at these descriptors:
/// the witness value's rows for a row-parallel generated region, at most
/// [`ORDER_ROW_BLOCKS`] for a `top_k`/`sort_desc` region, one otherwise.
fn region_rows(plan: &LaunchStagePlan, descriptors: &[ValueDesc]) -> Vec<u32> {
    plan.fused
        .iter()
        .map(|region| {
            let rows = region
                .row_value
                .and_then(|value| descriptors.get(value as usize))
                .map_or(1, |descriptor| descriptor.rows.max(1));
            match region.kind {
                RegionKind::Generated => rows,
                RegionKind::Library(LibraryOp::TopK) if selects(plan, region) => rows,
                RegionKind::Library(LibraryOp::TopK | LibraryOp::Sort) => rows.min(ORDER_ROW_BLOCKS),
                RegionKind::Library(_) => 1,
            }
        })
        .collect()
}

/// Whether a `top_k` region is emitted as the selection kernel (no
/// temporary arena, a block per row) — the emitter's rule, read off the
/// op's width: `eta_compiler::codegen::cuda::order`.
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

/// The least the temporary arena may be, so that every block of every
/// region has its slice (`temporary_bytes / blocks`): a row block of a
/// generated region needs the parallel reduction's two work arrays over
/// one row (`ptir_parallel_reduce_f32`: `ceil(last / 32)` floats each);
/// an order block needs two `u32` arrays over one row.
fn temporary_floor(plan: &LaunchStagePlan, descriptors: &[ValueDesc], rows: &[u32]) -> u64 {
    let align = |bytes: u64| bytes.next_multiple_of(u64::from(SCRATCH_ALIGN));
    let widest = descriptors.iter().map(|d| u64::from(d.last)).max().unwrap_or(1).max(1);
    plan.fused
        .iter()
        .zip(rows)
        .map(|(region, &blocks)| {
            // A one-block launch is covered by `layout`'s own arena (four
            // words an element of the widest row); only many blocks sharing
            // it need the floor.
            if blocks <= 1 {
                return 0;
            }
            let per_block = match region.kind {
                RegionKind::Generated => align(2 * widest.div_ceil(32) * 4 + 4096),
                RegionKind::Library(LibraryOp::TopK) if selects(plan, region) => 0,
                RegionKind::Library(LibraryOp::TopK | LibraryOp::Sort) => align(8 * widest + 256),
                RegionKind::Library(_) => 0,
            };
            per_block * u64::from(blocks)
        })
        .max()
        .unwrap_or(0)
}

/// One lane's scratch layout for `plan`'s `descriptors`: slots reused by
/// liveness when the plan's regions give a launch order, one per value
/// otherwise; the temporary arena floored for the blocks that share it.
fn lay_out(plan: &LaunchStagePlan, descriptors: &[ValueDesc]) -> Result<Layout> {
    let rows = region_rows(plan, descriptors);
    let lifetimes = value_lifetimes(plan, &rows);
    let floor = temporary_floor(plan, descriptors, &rows);
    if std::env::var_os("PIE_ETA_REGIONS").is_some() {
        // A reading aid, not a contract: the stage's regions as this shell
        // launches them, with the bytes each one's values name.
        eprintln!(
            "[eta-regions] ops={} values={} regions={} temporary_floor={}",
            plan.ops.len(),
            plan.value_types.len(),
            plan.fused.len(),
            floor
        );
        for (index, (region, &blocks)) in plan.fused.iter().zip(&rows).enumerate() {
            let tags: Vec<String> = region
                .nodes
                .iter()
                .filter_map(|&n| plan.ops.get(n as usize).map(|op| format!("{:02x}", op.tag)))
                .collect();
            let named: u64 = region
                .nodes
                .iter()
                .filter_map(|&n| plan.ops.get(n as usize))
                .flat_map(|op| op.args.iter().copied().chain((0..u32::from(op.result_count)).map(move |r| op.result_id + r)))
                .collect::<std::collections::BTreeSet<u32>>()
                .into_iter()
                .filter_map(|v| descriptors.get(v as usize))
                .map(ValueDesc::device_bytes)
                .sum();
            eprintln!(
                "[eta-regions]  r{index} {:?} blocks/lane={blocks} nodes={} named={}MB tags=[{}]",
                region.kind,
                region.nodes.len(),
                named >> 20,
                tags.join(" ")
            );
            if blocks == 1 && named >> 20 > 0 {
                // A one-block region naming megabytes: say what its ops see.
                for &n in &region.nodes {
                    let Some(op) = plan.ops.get(n as usize) else { continue };
                    let axes = |v: u32| {
                        plan.value_types
                            .get(v as usize)
                            .map(|t| format!("v{v}:{:?}", t.axes))
                            .unwrap_or_default()
                    };
                    let args: Vec<String> = op.args.iter().map(|&v| axes(v)).collect();
                    let results: Vec<String> =
                        (0..u32::from(op.result_count)).map(|r| axes(op.result_id + r)).collect();
                    eprintln!(
                        "[eta-regions]    n{n} {:02x} args=[{}] results=[{}]",
                        op.tag,
                        args.join(", "),
                        results.join(", ")
                    );
                }
            }
        }
    }
    match lifetimes {
        Some(lifetimes) => layout_reusing(descriptors, &lifetimes, floor),
        None => layout(descriptors),
    }
    .map_err(|why| {
        Fault::program(
            "program::launch",
            format!("this fire's scratch does not fit: {why:?}"),
        )
    })
}

/// What one lane's scratch costs for `plan` at `extents` — the stride
/// [`Prepared::build`] cuts and [`Prepared::commit_lanes`] re-zeroes.
///
/// # Errors
///
/// [`Fault::Program`] when a value's shape does not resolve or the scratch
/// exceeds what [`eta_exec::layout`] permits.
pub fn scratch_bytes(plan: &LaunchStagePlan, extents: Extents) -> Result<u64> {
    let descriptors = describe_values(plan, extents)?;
    lay_out(plan, &descriptors).map(|layout| layout.total)
}

impl Prepared {
    /// Carve every buffer one stage needs, for `lanes` lanes.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when a value's shape does not resolve against
    /// `extents` or the scratch exceeds what [`eta_exec::layout`] permits, and
    /// whatever the allocations said.
    pub fn build(
        plan: &LaunchStagePlan,
        shapes: &[ChannelShape],
        extents: Extents,
        lanes: u32,
    ) -> Result<Prepared> {
        let channel_count = u32::try_from(plan.channel_bindings.len())
            .map_err(|_| Fault::program("program::launch", "more channels than a u32 can count"))?;
        let value_count = u32::try_from(plan.value_types.len())
            .map_err(|_| Fault::program("program::launch", "more values than a u32 can count"))?;

        // ── The value descriptors, and the scratch they size. ──
        let descriptors = describe_values(plan, extents)?;
        let scratch_layout = lay_out(plan, &descriptors)?;
        let scratch_stride = u32::try_from(scratch_layout.total)
            .map_err(|_| Fault::program("program::launch", "a scratch stride past a u32"))?;
        let temporary_offset = u32::try_from(scratch_layout.temporary)
            .map_err(|_| Fault::program("program::launch", "a temporary offset past a u32"))?;
        let temporary_bytes = u32::try_from(scratch_layout.temporary_bytes)
            .map_err(|_| Fault::program("program::launch", "a temporary arena past a u32"))?;
        let region_rows = region_rows(plan, &descriptors);

        // ── Op params, widened to CUDA's 88-byte record. ──
        let mut records = Vec::with_capacity(plan.ops.len());
        let mut result_base = 0u32;
        for op in &plan.ops {
            let mut record =
                CudaOpParams::widen(OpParams::of(op, result_base, OpRuntime::default()));
            if let (true, Some(channel)) = (op.tag == tags::CHAN_PUT, op.channel) {
                // `sink_bytes` is the cell exactly: the emitted put writes
                // `0..sink_bytes`, zero-filling past the value's own bytes,
                // and faults when the value is wider.
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

        let mut params = Buffer::zeroed(
            (records.len() * size_of::<CudaOpParams>()).max(size_of::<CudaOpParams>()),
        )?;
        params.write(0, &records_bytes(&records))?;

        let offset_bytes: Vec<u8> = scratch_layout
            .values
            .iter()
            .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut offsets = Buffer::zeroed(offset_bytes.len().max(size_of::<u32>()))?;
        offsets.write(0, &offset_bytes)?;

        let mut prepared = Prepared {
            retired: Vec::new(),
            table: Buffer::zeroed(0)?,
            table_host: Vec::new(),
            descriptors: Buffer::zeroed(0)?,
            descriptor_row: descriptors.iter().flat_map(record_bytes).collect(),
            params,
            offsets,
            scratch: Buffer::zeroed(0)?,
            pending: Buffer::zeroed(0)?,
            intrinsic_bases: Buffer::zeroed(0)?,
            intrinsic_modes: Buffer::zeroed(0)?,
            intrinsic_widths: Buffer::zeroed(0)?,
            intrinsic_strides: Buffer::zeroed(0)?,
            intrinsic_offsets: Buffer::zeroed(0)?,
            intrinsic_bases_host: Vec::new(),
            intrinsic_modes_host: Vec::new(),
            intrinsic_widths_host: Vec::new(),
            intrinsic_strides_host: Vec::new(),
            intrinsic_offsets_host: Vec::new(),
            channel_count,
            value_count,
            scratch_stride,
            temporary_offset,
            temporary_bytes,
            region_rows,
            lanes: 0,
            filled: 0,
            bindings: plan.channel_bindings.clone(),
        };
        prepared.grow(extents, lanes.max(1))?;
        Ok(prepared)
    }

    /// Cut the per-lane buffers for `lanes` lanes, keeping every intrinsic
    /// binding the smaller table held. A high-water mark, not a fit; only
    /// safe to call from [`Prepared::begin`] before anything is staged.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the lane table or the scratch outgrows what
    /// [`eta_exec::layout`] permits, and whatever the allocations said.
    fn grow(&mut self, extents: Extents, lanes: u32) -> Result<()> {
        if lanes <= self.lanes {
            return Ok(());
        }
        // The scratch ceiling is a wave's, not a lane's; a named refusal the
        // caller can retry with a smaller batch.
        let total = u64::from(self.scratch_stride) * u64::from(lanes);
        if total > eta_exec::SCRATCH_MAX_BYTES {
            return Err(Fault::Ceiling {
                what: "a wave's fused-region scratch",
                need: total,
                have: eta_exec::SCRATCH_MAX_BYTES,
            });
        }
        let shape = LaneShape::of(lanes, self.channel_count);
        let table_bytes = shape
            .bytes()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| Fault::program("program::launch", "a lane table past what fits"))?;
        let mut table_host = vec![0u8; table_bytes];
        write_record(
            &mut table_host,
            0,
            &LaneHeader {
                abi_version: eta_exec::LANE_ABI_VERSION,
                // Rewritten at every commit; the kernel reads it as its bound.
                lane_count: lanes,
                channel_slots_per_lane: self.channel_count,
                flags: 0,
            },
        );
        // Every lane's record carries the same extents; differs only in its
        // channel slot offset and commit word (set at `stage_lane`).
        for lane in 0..lanes {
            let at = shape
                .record_offset(lane)
                .and_then(|at| usize::try_from(at).ok())
                .ok_or_else(|| Fault::program("program::launch", "a lane record past the table"))?;
            write_record(
                &mut table_host,
                at,
                &LaneRecord {
                    kv_len: extents.kv_len,
                    page_count: extents.page_count,
                    row_count: extents.row_count,
                    token_count: extents.token_count,
                    sampled_rows: extents.sampled_rows,
                    query_len: extents.query_len,
                    key_len: extents.key_len,
                    // The lane's row in the flat slot array; the kernel
                    // indexes both channels and pending_flags with it.
                    channel_slot_offset: shape.slot_index(lane).ok_or_else(|| {
                        Fault::program("program::launch", "a lane's slot row past the table")
                    })?,
                    ..LaneRecord::default()
                },
            );
        }
        // The header and records above are freshly written; channel slots
        // are rewritten by every `stage_lane`.
        let mut table = Buffer::zeroed(table_bytes)?;
        table.write(0, &table_host)?;
        self.retired.push(std::mem::replace(&mut self.table, table));
        self.table_host = table_host;

        // Every lane's descriptor row is the same row, so this is the one
        // write it ever needs.
        let row = self.descriptor_row.len().max(1);
        let mut descriptors = Buffer::zeroed(row * lanes as usize)?;
        if !self.descriptor_row.is_empty() {
            let repeated: Vec<u8> = self
                .descriptor_row
                .iter()
                .copied()
                .cycle()
                .take(self.descriptor_row.len() * lanes as usize)
                .collect();
            descriptors.write(0, &repeated)?;
        }
        self.retired
            .push(std::mem::replace(&mut self.descriptors, descriptors));

        let scratch_bytes = (self.scratch_stride as usize * lanes as usize)
            .max(usize::try_from(SCRATCH_ALIGN).unwrap_or(256));
        self.retired.push(std::mem::replace(
            &mut self.scratch,
            Buffer::zeroed(scratch_bytes)?,
        ));
        // One byte per channel per lane, indexed as the channel slots are:
        // zero means a take reads the committed cell.
        self.retired.push(std::mem::replace(
            &mut self.pending,
            Buffer::zeroed((self.channel_count as usize * lanes as usize).max(1))?,
        ));

        let words = INTRINSIC_SLOTS * lanes as usize;
        self.intrinsic_bases_host.resize(words, 0u64);
        self.intrinsic_modes_host.resize(words, 0u32);
        self.intrinsic_widths_host.resize(words, 0u32);
        self.intrinsic_strides_host.resize(words, 0u32);
        self.intrinsic_offsets_host.resize(words, 0u32);
        for (slot, bytes) in [
            (&mut self.intrinsic_bases, words * size_of::<u64>()),
            (&mut self.intrinsic_modes, words * size_of::<u32>()),
            (&mut self.intrinsic_widths, words * size_of::<u32>()),
            (&mut self.intrinsic_strides, words * size_of::<u32>()),
            (&mut self.intrinsic_offsets, words * size_of::<u32>()),
        ] {
            self.retired
                .push(std::mem::replace(slot, Buffer::zeroed(bytes)?));
        }
        self.lanes = lanes;
        Ok(())
    }

    /// Open a boundary, growing to `lanes` if this batch has never carried
    /// that many. [`Prepared::grow`] retires rather than drops replaced
    /// buffers, since a previous boundary's kernels may still read them.
    ///
    /// # Errors
    ///
    /// As [`Prepared::grow`].
    pub fn begin(&mut self, extents: Extents, lanes: u32) -> Result<()> {
        self.filled = 0;
        self.grow(extents, lanes)
    }

    /// Take the next lane of this boundary, pointing its channel slots at
    /// the cells `cursors` name and its record at `commit`. Answers the
    /// lane index, which is the row a later [`Prepared::bind_intrinsic`]
    /// names.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when a stage-local slot names a channel this
    /// instance does not carry, or the batch has no lane left.
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
        // The commit word lets a refused lane early-return while others run.
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
                    // Not a ticket: nothing stages a table ahead of the fire.
                    expected_head: NO_TICKET,
                    expected_tail: NO_TICKET,
                },
            );
        }
        // Lanes are reused; clearing makes an unbound intrinsic read zero.
        let row = lane as usize * INTRINSIC_SLOTS;
        self.intrinsic_bases_host[row..row + INTRINSIC_SLOTS].fill(0);
        self.intrinsic_modes_host[row..row + INTRINSIC_SLOTS].fill(0);
        self.intrinsic_widths_host[row..row + INTRINSIC_SLOTS].fill(0);
        self.intrinsic_strides_host[row..row + INTRINSIC_SLOTS].fill(0);
        self.intrinsic_offsets_host[row..row + INTRINSIC_SLOTS].fill(0);
        self.filled += 1;
        Ok(lane)
    }

    /// The boundary's tables, staged on the stream; pending flags cleared
    /// and scratch zeroed. The commit word is not reset here; it is seeded
    /// on the stream by `channel::pull_validate`.
    ///
    /// # Errors
    ///
    /// Whatever the copies said.
    pub fn commit_lanes(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        if self.filled == 0 {
            return Ok(());
        }
        // The header is the kernel's own bound (`dispatch_lane >= lane_count`
        // returns early).
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
        self.table.stage(stream, 0, &self.table_host)?;
        self.intrinsic_bases
            .stage(stream, 0, &slice_bytes(&self.intrinsic_bases_host))?;
        self.intrinsic_modes
            .stage(stream, 0, &slice_bytes(&self.intrinsic_modes_host))?;
        self.intrinsic_widths
            .stage(stream, 0, &slice_bytes(&self.intrinsic_widths_host))?;
        self.intrinsic_strides
            .stage(stream, 0, &slice_bytes(&self.intrinsic_strides_host))?;
        self.intrinsic_offsets
            .stage(stream, 0, &slice_bytes(&self.intrinsic_offsets_host))?;
        // Only what this boundary uses, not the whole table.
        let lanes = self.filled as usize;
        self.pending
            .zero_span_on(stream, 0, lanes * self.channel_count as usize)?;
        // An unwritten value slot must read back as zero.
        self.scratch
            .zero_span_on(stream, 0, lanes * self.scratch_stride as usize)?;
        Ok(())
    }

    /// Point one lane's intrinsic at the buffer a model fire produced.
    /// Host-side; staged with the rest at [`Prepared::commit_lanes`].
    ///
    /// `storage` is a storage mode, not a `Dtype` wire code: they collide at
    /// `INTRINSIC_STORAGE_F32 == 0`, so passing a dtype silently misreads.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an intrinsic past the table's pitch or a lane
    /// past the batch.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_intrinsic(
        &mut self,
        lane: u32,
        intrinsic: IntrinsicId,
        base: u64,
        storage: u32,
        width: u32,
        row_stride: u32,
        row_offset: u32,
    ) -> Result<()> {
        let slot = intrinsic as usize;
        if slot >= INTRINSIC_SLOTS {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "intrinsic {slot} is past the {INTRINSIC_SLOTS}-slot pitch the side \
                     tables are indexed with"
                ),
            ));
        }
        if lane >= self.lanes {
            return Err(Fault::program(
                "program::launch",
                format!("lane {lane} is past a batch cut for {}", self.lanes),
            ));
        }
        let at = lane as usize * INTRINSIC_SLOTS + slot;
        self.intrinsic_bases_host[at] = base;
        self.intrinsic_modes_host[at] = storage;
        self.intrinsic_widths_host[at] = width;
        self.intrinsic_strides_host[at] = row_stride;
        self.intrinsic_offsets_host[at] = row_offset;
        Ok(())
    }

    /// Launch one generated region over every lane this boundary staged:
    /// one CTA per lane, or per row of a row-parallel region (the kernel
    /// reads `blockIdx.x / rows_per_lane` as its lane).
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] when the driver refuses the launch. A fault inside
    /// the kernel is asynchronous and surfaces at the next synchronize.
    pub fn launch_region(&self, region: &Region, stream: *mut core::ffi::c_void) -> Result<()> {
        if self.filled == 0 {
            return Ok(());
        }
        let mut args = Args::new();
        args.ptr(self.table.ptr())
            .ptr(self.lane_records_ptr())
            .ptr(self.channel_slots_ptr())
            .ptr(self.descriptors.ptr())
            .ptr(self.params.ptr())
            .ptr(self.offsets.ptr())
            .ptr(self.scratch.ptr())
            .u32(self.value_count)
            .u32(self.scratch_stride)
            .u32(self.temporary_offset)
            .ptr(self.pending.ptr())
            .ptr(self.intrinsic_bases.ptr())
            .ptr(self.intrinsic_modes.ptr())
            .ptr(self.intrinsic_widths.ptr())
            .ptr(self.intrinsic_strides.ptr())
            .ptr(self.intrinsic_offsets.ptr());
        // A row-parallel region launches `rows` blocks per lane, each on its
        // own slice of the temporary arena.
        let rows = self
            .region_rows
            .get(region.region_index as usize)
            .copied()
            .unwrap_or(1)
            .max(1);
        let stride = (self.temporary_bytes / rows) & !(SCRATCH_ALIGN as u32 - 1);
        args.u32(rows).u32(stride);
        launch(
            &region.module,
            self.filled.saturating_mul(rows),
            region.module.block_threads(),
            &mut args,
            FUSED_ARITY,
            stream,
        )
    }

    /// How many lanes this stage can carry in one launch, so a caller can
    /// split a group into launches that fit. At least one: a lane that alone
    /// doesn't fit is refused at bind by [`eta_exec::layout`].
    #[must_use]
    pub fn lane_ceiling(&self) -> u32 {
        if self.scratch_stride == 0 {
            return u32::MAX;
        }
        u32::try_from(eta_exec::SCRATCH_MAX_BYTES / u64::from(self.scratch_stride))
            .unwrap_or(u32::MAX)
            .max(1)
    }

    /// How many lanes this boundary staged.
    #[must_use]
    pub const fn filled(&self) -> u32 {
        self.filled
    }

    /// How many lanes the buffers are cut for.
    #[must_use]
    pub const fn lanes(&self) -> u32 {
        self.lanes
    }

    /// How many channel slots one lane of this stage's table carries.
    #[must_use]
    pub const fn channel_count(&self) -> u32 {
        self.channel_count
    }

    /// The lane records, which begin one header into the table.
    fn lane_records_ptr(&self) -> u64 {
        self.table.ptr() + LANE_HEADER_BYTES
    }

    /// The flat channel-slot array, which begins after every lane record.
    fn channel_slots_ptr(&self) -> u64 {
        self.table.ptr()
            + LANE_HEADER_BYTES
            + u64::from(self.lanes) * LANE_RECORD_BYTES
    }
}

/// A `#[repr(C)]` record's bytes, written into `into` at `at`.
fn write_record<T: Copy>(into: &mut [u8], at: usize, record: &T) {
    // SAFETY: `T` is a `#[repr(C)]` mirror of a device struct with every field
    // written by the caller, so reading it as bytes reads only initialised
    // memory.
    let bytes = unsafe {
        std::slice::from_raw_parts(std::ptr::from_ref(record).cast::<u8>(), size_of::<T>())
    };
    into[at..at + bytes.len()].copy_from_slice(bytes);
}

/// One `#[repr(C)]` record as an owned byte vector.
pub(super) fn record_bytes<T: Copy>(record: &T) -> Vec<u8> {
    // SAFETY: as `write_record` — a fully-initialised `#[repr(C)]` mirror.
    unsafe {
        std::slice::from_raw_parts(std::ptr::from_ref(record).cast::<u8>(), size_of::<T>()).to_vec()
    }
}

/// A slice of op records as the flat bytes one upload copies.
pub(super) fn slice_bytes<T: Copy>(records: &[T]) -> Vec<u8> {
    // SAFETY: as `record_bytes`, over a contiguous run of them.
    unsafe {
        std::slice::from_raw_parts(records.as_ptr().cast::<u8>(), std::mem::size_of_val(records))
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

// ─────────────────────────────────────────────────────────────────────────────
// cuLaunchKernel
// ─────────────────────────────────────────────────────────────────────────────

/// A kernel's argument list, kept alive across the launch.
///
/// `cuLaunchKernel` takes `void**` — pointers to each argument's storage,
/// not the values — so a scalar must outlive the call.
#[derive(Default)]
pub struct Args {
    /// Boxed so a later append cannot move an earlier scalar and dangle its
    /// pointer in `slots` — a `Vec<u64>` would reallocate.
    #[allow(clippy::vec_box)]
    storage: Vec<Box<u64>>,
    slots: Vec<*mut std::ffi::c_void>,
}

impl Args {
    /// An empty list.
    #[must_use]
    pub fn new() -> Args {
        Args::default()
    }

    /// Append a device pointer argument.
    pub fn ptr(&mut self, pointer: u64) -> &mut Args {
        self.scalar(pointer)
    }

    /// Append a `u32` argument, stored in a `u64` cell and pointed at its
    /// first four bytes — correct on the little-endian hosts CUDA runs on.
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
}

/// Launch `module`'s entry with `grid` blocks of `block` threads.
///
/// # Errors
///
/// [`Fault::Program`] when the argument count is not `expected` or the grid
/// is empty — CUDA does not check either and both look like a fire that ran
/// — and [`Fault::Device`] when the driver refuses.
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
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys as dr;

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
                0,
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
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (block, stream);
        Err(Fault::Runtimeless)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    /// The round trip is lossless in both directions.
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
                // High bits past `numel` are not carried; compare only real lanes.
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
