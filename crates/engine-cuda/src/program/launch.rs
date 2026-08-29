//! Device residency for one instance's channel rings, the lane-table bytes,
//! and the launch of a fire's regions.
//!
//! **THE BYTES ARE THE INTERFACE** (design §6). Everything an emitted region
//! reads is a `#[repr(C)]` record whose one definition lives in the shared
//! crate — [`engine::LaneHeader`], [`engine::LaneRecord`],
//! [`engine::LaneChannelSlot`], [`engine::ValueDesc`], [`engine::OpParams`] —
//! and this module's whole job is to pack those records and hand their
//! addresses to `cuLaunchKernel`. Nothing here decides anything: which values
//! exist is the plan's, where they sit in scratch is [`engine::layout`]'s,
//! what each op does is the emitter's.
//!
//! **TWO SPELLINGS OF A CELL, AND ONLY BOOL DIFFERS.** On the device a cell is
//! native — one byte per bool lane, because a kernel indexes lane by lane —
//! and on the wire it is packed, one bit per bool lane, which is what the host
//! interpreter's rings hold. Every other dtype is four bytes either way, so a
//! plain byte copy is right for three of the four and catastrophic for the
//! fourth; [`wire_to_native`] and [`native_to_wire`] are the door.
//!
//! **THE RING IS `capacity + 1`.** The extra slot is a sentinel, which is what
//! makes empty and full distinguishable. The host half's [`ChannelState`] uses
//! the same `capacity + 1`, and the cursors on both sides are the same
//! sequence numbers — the device just takes them modulo the ring, because a
//! cell address is `base + (sequence % ring) * cell_bytes`. That agreement is
//! not decoration: it is what lets the parity test compare two rings slot for
//! slot.
//!
//! [`ChannelState`]: engine::ChannelState

use engine::engine_api::program::LaunchStagePlan;
use engine::tensor_ir::DType;
use engine::tensor_ir::op::{IntrinsicId, tags};
use engine::{
    Extents, LANE_HEADER_BYTES, LANE_RECORD_BYTES, LANE_SLOT_BYTES, LaneChannelSlot, LaneHeader,
    LaneRecord, LaneShape, NO_TICKET, OpParams, OpRuntime, SCRATCH_ALIGN, ValueDesc, describe,
    layout,
};

use std::sync::Arc;

use kernels_cuda::channel::MAX_RING;

use crate::device::Buffer;
use crate::error::{Fault, Result};

use super::compile::{Module, Region};
use super::endpoint::Endpoint;

/// The sixteen arguments a generated fused region takes.
///
/// From `tensor-compiler/runtime/cuda/fused_block1.cuh`, which is the
/// signature every emitted region is spliced into. CUDA validates nothing:
/// fifteen bound arguments read the sixteenth out of whatever follows the
/// array, which is a wrong answer rather than an error.
const FUSED_ARITY: usize = 16;

/// How many intrinsic slots the five side tables carry per lane.
///
/// PROJECTED FROM THE ABI, NOT WRITTEN. The tables are indexed
/// `lane * INTRINSIC_SLOTS + intrinsic`, so this is a stride, and a stride
/// that disagrees with the emitted kernel's misdirects every intrinsic of
/// every lane but the first — which is why a single-lane fire never shows it.
pub const INTRINSIC_SLOTS: usize = IntrinsicId::SLOTS as usize;

/// `IntrinsicStorageMode::F32` — the bound buffer holds `f32` elements.
pub const INTRINSIC_STORAGE_F32: u32 = 0;

/// `IntrinsicStorageMode::RawBf16` — the bound buffer holds raw `bf16`
/// elements the kernel widens as it reads.
pub const INTRINSIC_STORAGE_RAW_BF16: u32 = 1;

/// `IntrinsicStorageMode::RowPointers` — the bound buffer is a TABLE OF ROW
/// ADDRESSES, one `u64` per row of the value the guest declared, and each
/// entry addresses one `bf16` row of the real rectangle.
///
/// **WHAT IT IS FOR: A READOUT WHOSE ROWS ARE NOT CONSECUTIVE.**
/// [`INTRINSIC_STORAGE_RAW_BF16`] addresses row `r` as
/// `base + (row_offset + r) * row_stride`, which can name any contiguous run
/// and nothing else. A lane that states `Readout::Rows([0, 7, 3])` is asking
/// for three rows in that order, and the only shape that expresses it is a
/// list. The emitted kernel has read this mode since the runtime prologue was
/// written (`m1_intrinsic_row_base`'s `mode == 2` arm, and
/// `ptir_fast_argmax_intrinsic` beside it); what did not exist until the
/// row-selected readout landed was a producer.
///
/// `row_offset` still applies — it indexes the TABLE — and `row_stride` is
/// ignored, because each entry carries its own address. The element type is
/// `bf16`, the same as [`INTRINSIC_STORAGE_RAW_BF16`]: a row pointer says
/// where a row is and not what it holds.
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
/// [`engine::OpParams`] is 64 bytes (sixteen `u32`); CUDA's `M1OpParams` is 88
/// — twenty `u32` plus a `u64` `rng_seed` whose eight-byte alignment pads the
/// record to 88 and not the 84 a hand-summed field list gives. The first
/// sixteen words match by name and order; [`CudaOpParams::widen`] adds the
/// five CUDA-only fields.
///
/// `#[repr(C)]`, field for field with the device struct. The kernels index it
/// by offset, so the field order may not be tidied.
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

/// Every field's offset, pinned individually: `sizeof == 88` holds under any
/// permutation of the twenty `u32`s, so a size check cannot catch a
/// transposition.
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
    /// defaults the C++ wrote for an op that binds no intrinsic.
    ///
    /// The sixteen shared words are copied BY NAME rather than transmuted out
    /// of the 64-byte prefix, so a field inserted into [`engine::OpParams`] is
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
pub fn native_cell_bytes(dtype: DType, numel: usize) -> usize {
    if dtype == DType::Bool {
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
pub fn wire_to_native(dtype: DType, numel: usize, wire: &[u8]) -> Result<Vec<u8>> {
    let numel = numel.max(1);
    let want = engine::wire_cell_bytes(dtype, numel);
    if wire.len() != want {
        return Err(Fault::program(
            "program::launch",
            format!(
                "a {} wire cell of {numel} lane(s) is {want} bytes and {} were offered",
                dtype.name(),
                wire.len()
            ),
        ));
    }
    if dtype != DType::Bool {
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
pub fn native_to_wire(dtype: DType, numel: usize, native: &[u8]) -> Result<Vec<u8>> {
    let numel = numel.max(1);
    let want = native_cell_bytes(dtype, numel);
    if native.len() != want {
        return Err(Fault::program(
            "program::launch",
            format!(
                "a {} native cell of {numel} lane(s) is {want} bytes and {} were offered",
                dtype.name(),
                native.len()
            ),
        ));
    }
    if dtype != DType::Bool {
        return Ok(native.to_vec());
    }
    let mut out = vec![0u8; engine::wire_cell_bytes(dtype, numel)];
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
    pub dtype: DType,
    /// How many unconsumed items the channel holds. The ring is one longer.
    pub capacity: u32,
}

impl ChannelShape {
    /// The channel a launch package declares at this slot.
    #[must_use]
    pub fn of(declared: &engine::engine_api::program::LaunchChannel) -> ChannelShape {
        ChannelShape {
            numel: declared
                .shape
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1),
            dtype: engine::concrete_dtype(declared.dtype),
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

/// One instance's channel state: the device cells, the device REGISTRY the
/// control kernels move, and the pinned endpoint of every channel with a host
/// end.
///
/// **THE CURSORS CAME BACK TO THE DEVICE** (alto design §5, survey §7 I1/I2).
/// What stood here was cells and nothing else, because readiness and commit
/// were host arithmetic after a per-stage synchronize — the inversion the
/// survey calls the root of six of the ten violations. The registry below is
/// dev's `ChannelArena` and the four arrays `channel::Rings` names: the
/// full/empty byte of `(slot, ring)` at `slot * MAX_RING + ring`, the two ring
/// POSITIONS per slot, and each slot's `cap1`. `channel::commit_bump` is the
/// only writer of any of them.
///
/// The host still keeps a [`Cursor`] per channel and it is no longer the
/// truth: it is a PREDICTION, minted by counting, that this fire's cell
/// addresses are arithmetic on and that `channel::pull_validate` checks
/// against the live pinned words before anything commits.
///
/// [`session`]: super::session
#[derive(Debug)]
pub struct Rings {
    cells: Vec<Buffer>,
    shapes: Vec<ChannelShape>,
    /// The pinned mirror and counters of every channel with a host end;
    /// `None` for a channel whose cells never leave the device.
    endpoints: Vec<Option<Arc<Endpoint>>>,
    /// `head[slots]`, `tail[slots]`, `cap1[slots]`, `full[slots * MAX_RING]`,
    /// in one allocation because they are indexed together and a lane that
    /// carried a different `full` than the pull that set the byte would
    /// publish into a ring nobody reads.
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
            // **A SHARED RING IS NOT CUT HERE, AND THAT IS THE WHOLE FIX**
            // (design §5). A channel whose endpoint carries its own device
            // slab is one the CHANNEL owns rather than this instance: the
            // pass that puts and the pass that takes are two sessions, and a
            // slab cut per session is two slabs that never meet. The
            // placeholder keeps `cells` indexed by dense channel like every
            // other array here; `Rings::slab` is what reads through it.
            let shared = endpoints
                .get(index)
                .and_then(Option::as_ref)
                .and_then(|endpoint| endpoint.device_cells());
            cells.push(match shared {
                Some(_) => Buffer::zeroed(0)?,
                None => Buffer::zeroed(bytes * shape.ring() as usize)?,
            });
        }

        // ── The registry, in one allocation. The u32 arrays go first so that
        //    every one of them is 4-aligned off a `cudaMalloc` base, and the
        //    full/empty BYTES — one per `(slot, ring)`, MAX_RING apart
        //    whatever a slot's own cap1 is — go last.
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

    /// Seed the registry's ring POSITIONS and full bytes from the cursors the
    /// seeds left behind.
    ///
    /// **BIND-TIME ONLY, AND THE ONLY WRITE TO THESE WORDS THAT IS NOT
    /// `commit_bump`'s.** A channel declared `from(seed)` starts with its
    /// first cell full and its tail one on (`channel::Rings`' own note); the
    /// host plants the cells through [`Rings::write_cell`] and this states
    /// where they left the ring standing. On the fire path the registry is
    /// the bump kernel's alone.
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
    /// here, not refused: the ring position IS the cursor's residue, and that
    /// is the one place the two halves' cursor spellings meet.
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

    /// **Channel `channel`'s SHARED device slab**, or `None` for one whose
    /// cells this session cut for itself.
    ///
    /// See [`Endpoint::device_cells`](super::Endpoint::device_cells): a
    /// device-only ring belongs to the channel and every attachment addresses
    /// one slab; a host-visible one's device cells are this pass's staging for
    /// a crossing and belong to the session.
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
        // ── THE HOST-VISIBLE CELL LIVES IN THE PINNED MIRROR, NOT HERE.
        //    A channel with a host end crosses by device ACCESS to mapped
        //    pinned memory (survey §7 I5): the guest's cell is pulled into
        //    this slab by `channel::pull_validate` at fire time and the pass's
        //    cell is scattered out of it by `channel::scatter_publish` at
        //    commit time. Writing the slab here instead would put the bytes
        //    where the guest cannot see them and where the next pull will
        //    overwrite them.
        // ── A SHARED RING'S SEED GOES TO THE SLAB ITSELF. It has no guest
        //    at either end, so there is no mirror for a cell to cross through
        //    and nothing will ever pull it in: the passes that take from this
        //    ring read the device cells directly, so the seed has to BE one.
        if let Some(base) = self.shared_slab(channel) {
            let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
            return crate::device::write_raw(base + at, native);
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
        // As `write_cell`: the mirror is where a host-visible channel's cells
        // are, in both directions. A descriptor port resolved off this call
        // therefore reads the cell the GUEST published this fire, before
        // anything has launched and without a device read.
        // ── A SHARED RING HAS NO MIRROR TO READ. Its endpoint carries the
        //    counters and the SLAB and nothing else: there is no guest at
        //    either end, so no cell ever crosses into pinned memory and the
        //    mirror stays as `Endpoint::open` left it. The cell a descriptor
        //    port wants is the device one, which is where both attachments
        //    write it — the same read the per-session branch below took while
        //    the ring was per session.
        if let Some(base) = self.shared_slab(channel) {
            let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
            let mut out = vec![0u8; shape.cell_bytes()];
            crate::device::copy_d2h(base + at, &mut out)?;
            return Ok(out);
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

/// One stage's device state, allocated once and refreshed per fire.
///
/// **NOTHING ALLOCATES ON THE FIRE PATH** (design §0). A stage's value types
/// size its scratch, its bindings index its lane table and its op count
/// strides its params — all three are properties of the PLAN, not of the fire
/// — so every buffer below is carved at bind and only the four things a fire
/// actually changes are written again: the cell addresses in the lane table,
/// the pending flags, the scratch, and the commit slot.
#[derive(Debug)]
pub struct Prepared {
    table: Buffer,
    /// The host mirror of the table. Kept so a fire patches the sixteen bytes
    /// of each channel slot rather than rebuilding a header and a record it
    /// already wrote.
    table_host: Vec<u8>,
    descriptors: Buffer,
    params: Buffer,
    offsets: Buffer,
    scratch: Buffer,
    pending: Buffer,
    intrinsic_bases: Buffer,
    intrinsic_modes: Buffer,
    intrinsic_widths: Buffer,
    intrinsic_strides: Buffer,
    intrinsic_offsets: Buffer,
    /// Where the kernel writes its verdict — **the fire's commit word, not
    /// this stage's**, and it is not allocated here.
    ///
    /// `channel::pull_validate` SEEDS this word (to one, or to zero when a
    /// ticket's prediction was stale), every stage's kernel reads it first and
    /// early-returns when it is clear, `channel::commit_bump` moves durable
    /// ring state only if it survived, and `channel::scatter_publish`
    /// publishes only then. One word, four readers, one meaning — which is
    /// what makes a refused fire a DUMMY RUN rather than a fire that did not
    /// happen (survey §7 I4). It lives in the session's pinned commit pair, so
    /// the host reads the verdict at settle without a device read (I3).
    commit: u64,
    channel_count: u32,
    value_count: u32,
    scratch_stride: u32,
    temporary_offset: u32,
    /// This stage's local channel slot → the instance's dense channel index.
    bindings: Vec<u32>,
}

impl Prepared {
    /// Carve every buffer one stage needs, for a single-lane fire.
    ///
    /// **ONE LANE.** The lane table's shape already admits many — that is what
    /// [`LaneShape`] is — and the emitted kernel reads `blockIdx.x` as its
    /// lane. What a grouped fire additionally needs is one ring registry
    /// shared across instances, which is a decision about the runtime's
    /// batching and not about this plane; step 7 binds one instance per
    /// session and the shape is left reachable.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when a value's shape does not resolve against
    /// `extents` or the scratch exceeds what [`engine::layout`] permits, and
    /// whatever the allocations said.
    pub fn build(
        plan: &LaunchStagePlan,
        shapes: &[ChannelShape],
        extents: Extents,
        commit: u64,
    ) -> Result<Prepared> {
        let channel_count = u32::try_from(plan.channel_bindings.len())
            .map_err(|_| Fault::program("program::launch", "more channels than a u32 can count"))?;
        let value_count = u32::try_from(plan.value_types.len())
            .map_err(|_| Fault::program("program::launch", "more values than a u32 can count"))?;

        // ── The value descriptors, and the scratch they size. ──
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

        // ── The lane table: header, one record, its channel slots. ──
        let shape = LaneShape::of(1, channel_count);
        let table_bytes = shape
            .bytes()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| Fault::program("program::launch", "a lane table past what fits"))?;
        let mut table_host = vec![0u8; table_bytes];
        write_record(
            &mut table_host,
            0,
            &LaneHeader {
                abi_version: engine::LANE_ABI_VERSION,
                lane_count: 1,
                channel_slots_per_lane: channel_count,
                flags: 0,
            },
        );
        write_record(
            &mut table_host,
            LANE_HEADER_BYTES as usize,
            &LaneRecord {
                kv_len: extents.kv_len,
                page_count: extents.page_count,
                row_count: extents.row_count,
                token_count: extents.token_count,
                sampled_rows: extents.sampled_rows,
                query_len: extents.query_len,
                key_len: extents.key_len,
                // The lane's row in the flat slot array. Lane zero starts at
                // zero; the kernel indexes `channels[offset + n]`.
                channel_slot_offset: 0,
                commit_slot: commit,
                ..LaneRecord::default()
            },
        );
        let mut table = Buffer::zeroed(table_bytes)?;
        table.write(0, &table_host)?;

        // ── Op params, widened to CUDA's 88-byte record. ──
        let mut records = Vec::with_capacity(plan.ops.len());
        let mut result_base = 0u32;
        for op in &plan.ops {
            let mut record =
                CudaOpParams::widen(OpParams::of(op, result_base, OpRuntime::default()));
            if let (true, Some(channel)) = (op.tag == tags::CHAN_PUT, op.channel) {
                // `sink_bytes` IS THE CELL, exactly: the emitted put writes
                // `0..sink_bytes` of the pending cell (zero-filling the tail
                // past the value's own bytes) and faults when the value is
                // wider. Zero refuses every put silently; anything larger
                // runs off the end of the ring slot.
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

        let descriptor_bytes: Vec<u8> = descriptors.iter().flat_map(record_bytes).collect();
        let mut descriptor_buffer = Buffer::zeroed(descriptor_bytes.len().max(1))?;
        descriptor_buffer.write(0, &descriptor_bytes)?;

        let offset_bytes: Vec<u8> = scratch_layout
            .values
            .iter()
            .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut offsets = Buffer::zeroed(offset_bytes.len().max(size_of::<u32>()))?;
        offsets.write(0, &offset_bytes)?;

        let scratch_bytes =
            (scratch_stride as usize).max(usize::try_from(SCRATCH_ALIGN).unwrap_or(256));
        let scratch = Buffer::zeroed(scratch_bytes)?;
        // One byte per channel: zero means a take reads the committed cell,
        // and the kernel sets it as puts land within the fire.
        let pending = Buffer::zeroed((channel_count as usize).max(1))?;

        let words = INTRINSIC_SLOTS;
        Ok(Prepared {
            table,
            table_host,
            descriptors: descriptor_buffer,
            params,
            offsets,
            scratch,
            pending,
            intrinsic_bases: Buffer::zeroed(words * size_of::<u64>())?,
            intrinsic_modes: Buffer::zeroed(words * size_of::<u32>())?,
            intrinsic_widths: Buffer::zeroed(words * size_of::<u32>())?,
            intrinsic_strides: Buffer::zeroed(words * size_of::<u32>())?,
            intrinsic_offsets: Buffer::zeroed(words * size_of::<u32>())?,
            commit,
            channel_count,
            value_count,
            scratch_stride,
            temporary_offset,
            bindings: plan.channel_bindings.clone(),
        })
    }

    /// Point this stage's lane table at the cells `cursors` name, and reset
    /// everything a fire starts from: pending flags clear, scratch zeroed,
    /// commit slot set.
    ///
    /// The commit word is NOT reset here: `channel::pull_validate` seeds it
    /// on the stream, ahead of this stage's launches, from this fire's
    /// tickets.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when a stage-local slot names a channel this
    /// instance does not carry, and whatever the copies said.
    pub fn refresh(
        &mut self,
        rings: &Rings,
        cursors: &[Cursor],
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        let slots_at = LANE_HEADER_BYTES as usize + LANE_RECORD_BYTES as usize;
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
            write_record(
                &mut self.table_host,
                slots_at + local * LANE_SLOT_BYTES as usize,
                &LaneChannelSlot {
                    committed_cell: rings.cell_address(channel, cursor.head)?,
                    pending_cell: rings.cell_address(channel, cursor.tail)?,
                    // Not a ticket: nothing stages a table ahead of the fire,
                    // so claiming one would pass a staleness check for the
                    // wrong reason.
                    expected_head: NO_TICKET,
                    expected_tail: NO_TICKET,
                },
            );
        }
        self.table.stage(stream, 0, &self.table_host)?;
        self.pending.clear(stream)?;
        // Zeroed every fire, not once: a value slot no op writes reads back as
        // whatever the LAST fire left there, and zeros are the state the
        // emitted kernels — and the host interpreter they are diffed against —
        // both assume.
        self.scratch.clear(stream)?;
        // THE COMMIT WORD IS NOT SEEDED HERE ANY MORE, and that is the whole
        // of F2a in one deletion: `channel::pull_validate` seeds it on the
        // stream, in front of these launches, after checking this fire's
        // tickets against the live pinned counters. A host store of one here
        // would overwrite the device's verdict with the host's opinion.
        Ok(())
    }

    /// Point one intrinsic at the buffer a model fire produced.
    ///
    /// The side tables are zeroed at bind, so an unbound intrinsic reads
    /// address zero. `modes` is a STORAGE mode, not a `DType` wire code: they
    /// collide only at `DType::F32 as u8 == 0 == INTRINSIC_STORAGE_F32`, so
    /// passing a dtype for a bf16 buffer misreads every logit, silently.
    ///
    /// **THE SEAM, NOT THE ATTACHMENT.** What is missing is not this call —
    /// [`Session::bind_intrinsic`](super::session::Session::bind_intrinsic)
    /// reaches it and the parity test drives it — but the shell deciding WHEN
    /// to make it: pointing this at the buffer `Shell::fire` produced, in the
    /// order a prologue and an epilogue run around a model fire, is the
    /// runtime's step.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an intrinsic past the table's pitch.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_intrinsic(
        &mut self,
        intrinsic: IntrinsicId,
        base: u64,
        storage: u32,
        width: u32,
        row_stride: u32,
        row_offset: u32,
        stream: *mut core::ffi::c_void,
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
        self.intrinsic_bases.stage(
            stream,
            (slot * size_of::<u64>()) as u64,
            &base.to_le_bytes(),
        )?;
        let word = |buffer: &mut Buffer, value: u32| -> Result<()> {
            buffer.stage(
                stream,
                (slot * size_of::<u32>()) as u64,
                &value.to_le_bytes(),
            )
        };
        word(&mut self.intrinsic_modes, storage)?;
        word(&mut self.intrinsic_widths, width)?;
        word(&mut self.intrinsic_strides, row_stride)?;
        word(&mut self.intrinsic_offsets, row_offset)?;
        Ok(())
    }

    /// Launch one generated region over this fire's single lane.
    ///
    /// One CTA per lane at the compiled function's own block width — the
    /// kernel's contract, not a tuning choice: it reads `blockIdx.x` as its
    /// lane and reduces with a halving tree over `blockDim.x`.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] when the driver refuses the launch. A fault INSIDE
    /// the kernel is asynchronous and surfaces at the next synchronize.
    pub fn launch_region(&self, region: &Region, stream: *mut core::ffi::c_void) -> Result<()> {
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
        launch(
            &region.module,
            1,
            region.module.block_threads(),
            &mut args,
            FUSED_ARITY,
            stream,
        )
    }

    /// The fire's commit word, as this stage's lane record points at it.
    #[must_use]
    pub const fn commit(&self) -> u64 {
        self.commit
    }

    /// How many channel slots this stage's table carries.
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
        self.table.ptr() + LANE_HEADER_BYTES + LANE_RECORD_BYTES
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
/// `cuLaunchKernel` takes `void**` — pointers to each argument's STORAGE, not
/// the values — so a scalar must outlive the call and a device pointer is
/// passed by the address of the variable holding it. It validates nothing, so
/// the marshalling lives behind this type.
#[derive(Default)]
pub struct Args {
    /// Boxed so that a later append cannot move an earlier scalar and dangle
    /// its pointer in `slots` — a `Vec<u64>` would reallocate. Hence
    /// `clippy::vec_box`.
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
/// [`Fault::Program`] when the argument count is not `expected` or the grid is
/// empty — two things CUDA does not check and both of which produce a fire
/// that looks like it ran — and [`Fault::Device`] when the driver refuses.
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
    #[cfg(feature = "_cuda")]
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
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (block, stream);
        Err(Fault::Runtimeless)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One wrong record size has every lane after the first read the last
    /// one's tail, and a single-lane fire never shows it.
    #[test]
    fn the_table_records_are_the_sizes_the_kernel_indexes_by() {
        assert_eq!(LANE_HEADER_BYTES, 16, "four u32");
        assert_eq!(size_of::<LaneHeader>(), 16);
        assert_eq!(size_of::<LaneRecord>(), LANE_RECORD_BYTES as usize);
        assert_eq!(size_of::<LaneChannelSlot>(), LANE_SLOT_BYTES as usize);
        assert_eq!(LANE_SLOT_BYTES, 32, "four u64");
    }

    /// The slot array begins past every lane record; the arithmetic
    /// `Prepared` uses for a single lane must agree with `LaneShape`'s.
    #[test]
    fn the_slot_array_begins_past_every_lane_record() {
        let one = LaneShape::of(1, 2).bytes().expect("fits");
        assert_eq!(one, 16 + LANE_RECORD_BYTES + 2 * LANE_SLOT_BYTES);
        assert_eq!(
            LaneShape::of(1, 2).slots_offset().expect("fits"),
            LANE_HEADER_BYTES + LANE_RECORD_BYTES
        );
    }

    /// A kernel handed fifteen of sixteen arguments reads the last past the
    /// array's end.
    #[test]
    fn a_fused_region_takes_sixteen_arguments() {
        assert_eq!(FUSED_ARITY, 16);
    }

    /// The record is 88 bytes; a hand-summed field list would give 84.
    #[test]
    fn the_cuda_op_record_is_eighty_eight_bytes_because_of_the_u64() {
        assert_eq!(size_of::<CudaOpParams>(), 88);
        assert_eq!(align_of::<CudaOpParams>(), 8, "the u64 sets the alignment");
    }

    /// Widening moves each of the sixteen shared words to its like-named
    /// field; distinct values are what make a transposition fail.
    #[test]
    fn widening_carries_all_sixteen_shared_words_to_their_own_fields() {
        let shared = OpParams {
            tag: 1,
            a0: 2,
            a1: 3,
            a2: 4,
            o0: 5,
            o1: 6,
            imm: 7,
            imm2: 8,
            imm3: 9,
            kind: 10,
            pred_tag: 11,
            lit_dtype: 12,
            lit_bits: 13,
            channel_slot: 14,
            intr: 15,
            sink_bytes: 16,
        };
        let cuda = CudaOpParams::widen(shared);
        assert_eq!(
            (
                cuda.tag, cuda.a0, cuda.a1, cuda.a2, cuda.o0, cuda.o1, cuda.imm, cuda.imm2
            ),
            (1, 2, 3, 4, 5, 6, 7, 8)
        );
        assert_eq!(
            (
                cuda.imm3,
                cuda.kind,
                cuda.pred_tag,
                cuda.lit_dtype,
                cuda.lit_bits,
                cuda.channel_slot,
                cuda.intr,
                cuda.sink_bytes
            ),
            (9, 10, 11, 12, 13, 14, 15, 16)
        );
        assert_eq!(cuda.intrinsic_dtype, INTRINSIC_STORAGE_F32);
        assert_eq!(cuda.bool_storage, BOOL_STORAGE_NATIVE_BYTES);
        assert_eq!(cuda.rng_seed, 0);
    }

    /// Records pack at the `index * 88` stride the kernel indexes; at 64 the
    /// second op's head would be the first op's tail, and every field would
    /// still be a plausible small integer.
    #[test]
    fn op_records_pack_at_the_stride_the_kernel_indexes() {
        let params = vec![
            CudaOpParams {
                tag: 0xAA,
                ..CudaOpParams::default()
            },
            CudaOpParams {
                tag: 0xBB,
                ..CudaOpParams::default()
            },
        ];
        let bytes = records_bytes(&params);
        assert_eq!(bytes.len(), 176);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 0xAA);
        assert_eq!(u32::from_le_bytes(bytes[88..92].try_into().unwrap()), 0xBB);
    }

    /// The intrinsic stride the emitted kernels share, pinned as an EQUALITY:
    /// a `>=` would pass forever while host and kernel silently disagreed.
    #[test]
    fn the_intrinsic_stride_is_the_slot_count_the_abi_declares() {
        assert_eq!(INTRINSIC_SLOTS, 8, "AttnScore + 1");
        assert_eq!(INTRINSIC_SLOTS, IntrinsicId::SLOTS as usize);
    }

    /// The two cell spellings differ for bool and for nothing else, and the
    /// round trip is lossless in both directions.
    #[test]
    fn only_bool_is_packed_and_the_round_trip_is_lossless() {
        for (dtype, numel) in [
            (DType::F32, 3),
            (DType::I32, 5),
            (DType::U32, 1),
            (DType::Bool, 11),
        ] {
            let native = native_cell_bytes(dtype, numel);
            let wire = engine::wire_cell_bytes(dtype, numel);
            if dtype == DType::Bool {
                assert_eq!(native, numel, "a byte per lane on the device");
                assert_eq!(wire, numel.div_ceil(8), "a bit per lane on the wire");
            } else {
                assert_eq!(native, wire, "four bytes either way");
            }
            let bytes: Vec<u8> = (0..wire).map(|i| (i as u8).wrapping_mul(37) | 1).collect();
            let there = wire_to_native(dtype, numel, &bytes).expect("one wire cell");
            assert_eq!(there.len(), native);
            let back = native_to_wire(dtype, numel, &there).expect("one native cell");
            if dtype == DType::Bool {
                // The high bits of the last wire byte are past `numel` and are
                // not carried; compare only the lanes that exist.
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

    /// A cell that is not exactly one cell is refused rather than padded: a
    /// short one reads real-looking garbage past its end.
    #[test]
    fn a_cell_of_the_wrong_width_is_refused() {
        assert!(wire_to_native(DType::I32, 2, &[0u8; 4]).is_err());
        assert!(native_to_wire(DType::Bool, 4, &[0u8; 3]).is_err());
    }

    /// A `Vec<u64>` backing would reallocate on append and dangle every
    /// pointer already bound.
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
}
