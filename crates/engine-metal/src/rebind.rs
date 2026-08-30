//! `icb::rebind`, host side: the derived [`DescriptorAbi`] lowered into device
//! tables, and the two-pass fire that reads them.
//!
//! **THIS IS THE JOIN THE PREVIOUS WAVE LEFT OPEN.** Build log 30's verdict
//! was that the ICB makes the exec key unnecessary and the derived table would
//! make the WALK unnecessary, and that the wave built the first and derived
//! the second without wiring them together. The wire is here. What crosses to
//! the device is:
//!
//! ```text
//! once, at load          every fire
//! ─────────────          ──────────
//! the law table          the packed FireDescriptor (one memcpy)
//! the arm table            ↓
//! the binding lists      dispatch icb_rebind, one thread per slot
//! the pipeline table       ↓  (a pass boundary — the commands were WRITTEN)
//! the reservation table  executeCommandsInBuffer(icb, 0..slots)
//! ```
//!
//! and the host does not walk. `model_exec::fire::walk` is still what the eager
//! path runs and still what the recorder recorded — the shader is a
//! translation of `abi::Law::at` and of nothing else, which is why a test can
//! diff the two in host arithmetic before ever dispatching.
//!
//! # What the shader may rewrite, and what it may not
//!
//! `MTLIndirectComputeCommand`'s whole vocabulary: the pipeline state, every
//! kernel buffer binding, the grid, the threadgroup, the barrier, and
//! `reset()`. What it may NOT do is carry inline bytes — there is no
//! `setBytes:` on an indirect compute command — so a scalar argument is a
//! staged cell and the shader writes the CELL. That reservation is this
//! module's own ([`Rebinder::cells`]) rather than the builder's, because a
//! slot's second arm binds a different argument list and needs cells the
//! build never allocated.
//!
//! # Two passes, and the reason is not caution
//!
//! The rebind kernel writes the commands that `executeCommandsInBuffer:` then
//! runs. A compute pass serialises DISPATCHES against each other;
//! `executeCommandsInBuffer:` is not a dispatch. So the two live in two
//! passes of one command buffer ([`Frame::next_pass`](crate::device::Frame)),
//! which is one encoder open and no second commit.

#![cfg(target_vendor = "apple")]

use std::collections::HashMap;

use kernels_metal::icb as layout;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLIndirectCommandBuffer, MTLResource,
    MTLResourceID, MTLResourceUsage, MTLSize,
};
use objc2_foundation::NSRange;

use crate::abi::{Arm, At, DescriptorAbi, Law, Pick};
use crate::device::alloc::{Buffer, Slab, slab_address};
use crate::device::library::Pipeline;
use crate::device::{Context, Pipelines};
use crate::error::{Fault, Result};
use crate::record::{Arg, Slot};

/// What the lowering produced — the census of the device-side table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lowered {
    /// ICB slots, one rebind thread each.
    pub slots: usize,
    /// Arms across every slot; more than `slots` exactly where an entry picks
    /// its shader off the window.
    pub arms: usize,
    /// Law rows — one per moving component of every arm, plus one window-rows
    /// law per slot.
    pub laws: usize,
    /// Binding rows: every argument of every arm, for the re-encode a revived
    /// or re-armed slot is.
    pub binds: usize,
    /// Distinct pipelines the arms name.
    pub pipelines: usize,
    /// Distinct reservations the bindings name.
    pub slabs: usize,
    /// Bytes of staged-scalar arena.
    pub cells: u64,
    /// Bytes of device table, all of it.
    pub bytes: u64,
}

impl std::fmt::Display for Lowered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} slots, {} arms, {} laws, {} bindings, {} pipelines, {} reservations, \
             {:.1} KiB of table and {} B of scalar arena",
            self.slots,
            self.arms,
            self.laws,
            self.binds,
            self.pipelines,
            self.slabs,
            self.bytes as f64 / 1024.0,
            self.cells
        )
    }
}

/// The device-side rebind: every table the shader reads, and the pipeline
/// that reads them.
pub struct Rebinder {
    pipeline: Pipeline,
    /// Argument buffers: the ICB's own resource id, the pipeline states, and
    /// the reservation addresses.
    handle: Buffer,
    pipes: Buffer,
    slabs: Buffer,
    /// The header.
    plan: Buffer,
    /// The packed `model_exec::fire::descriptor` bytes — the ONE thing a fire
    /// writes.
    descriptor: Buffer,
    /// The coordinate recipe.
    konst: Buffer,
    coeff: Buffer,
    /// The tables.
    slot_rows: Buffer,
    arm_rows: Buffer,
    law_rows: Buffer,
    bind_rows: Buffer,
    pipe_rows: Buffer,
    /// One word per slot: which arm is encoded in it, `0` for reset and
    /// `u32::MAX` for "the host touched this, encode it again".
    live: Buffer,
    /// The shader's one output.
    status: Buffer,
    /// The staged scalars of every arm of every slot.
    cells: Buffer,
    /// The compiled pipelines the arm table names, retained so their resource
    /// ids stay valid for the life of the load.
    #[allow(dead_code, reason = "held for the resource ids written into `pipes`")]
    retained: Vec<Pipeline>,
    /// How many bytes of descriptor the reservation holds.
    room: u64,
    census: Lowered,
}

// SAFETY: as `Icb` and `Buffer` — the Metal objects here are documented
// thread-safe for retain/release and for binding, and a `Rebinder` is built
// and used on the one lane thread that owns the shell.
unsafe impl Send for Rebinder {}

impl std::fmt::Debug for Rebinder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rebinder").field("census", &self.census).finish()
    }
}

impl Rebinder {
    /// The census of what was lowered.
    #[must_use]
    pub fn census(&self) -> Lowered {
        self.census
    }

    /// Say that something other than this shader last wrote the buffer, so
    /// every slot is encoded again on the next rebind.
    ///
    /// **THE TWO PATHS SHARE ONE BUFFER AND NEITHER TRUSTS THE OTHER'S
    /// BOOKKEEPING.** `Icb::rebind` diffs against a host-side record of what
    /// is encoded; this shader diffs against `live`. A fire through one after
    /// a fire through the other would diff against a record of somebody
    /// else's writes, so the crossing is marked rather than assumed away.
    pub(crate) fn desync(&mut self) -> Result<()> {
        let mut word = Vec::with_capacity(self.census.slots * 4);
        for _ in 0..self.census.slots {
            word.extend_from_slice(&u32::MAX.to_ne_bytes());
        }
        self.live.write(0, &word)
    }
}

/// One argument's place in the staged-scalar arena, when it has one.
fn cell_of(arg: Arg) -> Option<usize> {
    match arg {
        Arg::I32(_) | Arg::U32(_) | Arg::F32(_) | Arg::Usize(_) => Some(8),
        Arg::Buffer { .. } | Arg::Absent => None,
    }
}

/// A staged scalar's bytes.
fn scalar_bytes(arg: Arg) -> Option<[u8; 8]> {
    let mut out = [0u8; 8];
    match arg {
        Arg::I32(v) => out[..4].copy_from_slice(&v.to_ne_bytes()),
        Arg::U32(v) => out[..4].copy_from_slice(&v.to_ne_bytes()),
        Arg::F32(bits) => out[..4].copy_from_slice(&bits.to_ne_bytes()),
        Arg::Usize(v) => out.copy_from_slice(&v.to_ne_bytes()),
        Arg::Buffer { .. } | Arg::Absent => return None,
    }
    Some(out)
}

/// The `Fire` one recorded slot's pipeline is keyed on.
fn fire_of(slot: &Slot) -> kernels_metal::Fire {
    kernels_metal::Fire {
        file: slot.point.file,
        entrypoint: slot.point.entrypoint,
        stamp: slot.point.stamp,
        lanes: slot.lanes,
        group: slot.group,
    }
}

fn resource_id(id: MTLResourceID) -> u64 {
    // `MTLResourceID` is `#[repr(C)]` over one `u64` and its field is
    // crate-private, so the bytes are taken rather than named — the same
    // reading the kill-factor probe took, and the only one available.
    // SAFETY: both types are eight bytes of plain data with no niche.
    unsafe { std::mem::transmute::<MTLResourceID, u64>(id) }
}

/// Lower one derived table into the device tables the shader reads.
///
/// `icb` is the buffer the shader will rewrite and `slots` its command count;
/// the two must be the artifact this table was derived from, which is checked
/// slot for slot rather than assumed.
///
/// # Errors
///
/// [`Fault::Unstructured`] when the table is not this buffer's,
/// [`Fault::Ceiling`] for a basis, a pipeline count or a reservation count
/// past what the shader's fixed arrays hold, [`Fault::Shader`] for an arm
/// whose entry will not compile, [`Fault::Device`] for a reservation the
/// device declined.
#[allow(clippy::too_many_lines)]
pub(crate) fn lower(
    device: &Context,
    pipelines: &Pipelines,
    icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
    encoded: &[(u32, u32)],
    abi: &DescriptorAbi,
    held: &HashMap<u64, Slab>,
    room: u64,
) -> Result<Rebinder> {
    if abi.len() != encoded.len() {
        return Err(Fault::Unstructured {
            slot: 0,
            why: format!(
                "the derived table holds {} slots and the indirect command buffer was \
                 encoded with {} — one of them is not this artifact's",
                abi.len(),
                encoded.len()
            ),
        });
    }
    for (index, (slot, (region, run))) in abi.slots.iter().zip(encoded).enumerate() {
        if slot.region != *region || slot.run != *run {
            return Err(Fault::Unstructured {
                slot: index as u32,
                why: format!(
                    "the table says region {} run {} and the buffer was encoded at region \
                     {region} run {run} — the table and the buffer were built at two \
                     different compositions",
                    slot.region, slot.run
                ),
            });
        }
    }
    if abi.axes.len() > layout::MAX_AXES {
        return Err(Fault::Ceiling {
            what: "probe directions the rebind shader evaluates",
            need: abi.axes.len() as u64,
            have: layout::MAX_AXES as u64,
        });
    }

    let classes = abi.origin_classes.len();

    // 1. The pipelines the arms name, deduplicated, with the two occupancy
    //    numbers the shader needs to answer a `[0,0,0]` threadgroup.
    let mut pipe_index: HashMap<(&'static str, &'static str), u32> = HashMap::new();
    let mut retained: Vec<Pipeline> = Vec::new();
    let mut pipe_rows: Vec<layout::PipeRow> = Vec::new();
    for slot in &abi.slots {
        for arm in &slot.arms {
            let key = (arm.point.file, arm.point.entrypoint);
            if pipe_index.contains_key(&key) {
                continue;
            }
            let pipeline = pipelines.at(device.device(), fire_of(&arm.skeleton))?;
            pipe_rows.push(layout::PipeRow {
                width: pipeline.threadExecutionWidth() as u32,
                total: pipeline.maxTotalThreadsPerThreadgroup() as u32,
            });
            pipe_index.insert(key, retained.len() as u32);
            retained.push(pipeline);
        }
    }
    if retained.len() > layout::MAX_PIPELINES {
        return Err(Fault::Ceiling {
            what: "distinct pipelines in the rebind shader's argument buffer",
            need: retained.len() as u64,
            have: layout::MAX_PIPELINES as u64,
        });
    }

    // 2. The reservations the bindings name, deduplicated.
    let mut slab_index: HashMap<u64, u32> = HashMap::new();
    let mut addresses: Vec<u64> = Vec::new();
    for slot in &abi.slots {
        for arm in &slot.arms {
            for arg in &arm.skeleton.args {
                if let Arg::Buffer { slab, .. } = arg
                    && !slab_index.contains_key(slab)
                {
                    let reservation = held.get(slab).ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "reservation {slab:#x}, which the derived table binds and the \
                             indirect command buffer's build never saw"
                        ),
                    })?;
                    slab_index.insert(*slab, addresses.len() as u32);
                    addresses.push(slab_address(reservation));
                }
            }
        }
    }
    if addresses.len() > layout::MAX_SLABS {
        return Err(Fault::Ceiling {
            what: "distinct reservations in the rebind shader's argument buffer",
            need: addresses.len() as u64,
            have: layout::MAX_SLABS as u64,
        });
    }

    // 3. The tables. One pass, in slot order, so the arm and law runs are
    //    contiguous and the shader indexes rather than searches.
    let mut slot_rows: Vec<layout::SlotRow> = Vec::with_capacity(abi.len());
    let mut arm_rows: Vec<layout::ArmRow> = Vec::new();
    let mut law_rows: Vec<layout::LawRow> = Vec::new();
    let mut bind_rows: Vec<layout::BindRow> = Vec::new();
    let mut cell_bytes: Vec<u8> = Vec::new();

    for slot in &abi.slots {
        let rows_law = law_rows.len() as u32;
        law_rows.push(law_row(&slot.rows, layout::AT_LANE, 0, None));
        let arm_at = arm_rows.len() as u32;
        for arm in &slot.arms {
            let bind_at = bind_rows.len() as u32;
            // Every argument of this arm, and a cell for every scalar among
            // them: what a revived or re-armed slot is encoded again from.
            let mut cells_of_arg: Vec<Option<u64>> = Vec::with_capacity(arm.skeleton.args.len());
            for (index, arg) in arm.skeleton.args.iter().enumerate() {
                match *arg {
                    Arg::Buffer { slab, offset, .. } => {
                        cells_of_arg.push(None);
                        bind_rows.push(layout::BindRow::new(
                            index as u32,
                            layout::BIND_SLAB,
                            slab_index[&slab],
                            offset,
                        ));
                    }
                    Arg::Absent => {
                        cells_of_arg.push(None);
                        bind_rows.push(layout::BindRow::new(
                            index as u32,
                            layout::BIND_ABSENT,
                            0,
                            0,
                        ));
                    }
                    scalar => {
                        let at = cell_bytes.len() as u64;
                        cell_bytes.extend_from_slice(
                            &scalar_bytes(scalar).expect("a scalar has bytes"),
                        );
                        debug_assert_eq!(cell_of(scalar), Some(8));
                        cells_of_arg.push(Some(at));
                        bind_rows.push(layout::BindRow::new(
                            index as u32,
                            layout::BIND_CELL,
                            0,
                            at,
                        ));
                    }
                }
            }
            let law_at = law_rows.len() as u32;
            for (at, law) in &arm.laws {
                law_rows.push(match *at {
                    At::Grid(axis) => law_row(law, layout::AT_LANE, u32::from(axis), None),
                    At::Block(axis) => law_row(law, layout::AT_GROUP, u32::from(axis), None),
                    At::Arg { at: index, .. } => {
                        let place = argument_place(arm, index as usize, &slab_index, &cells_of_arg);
                        law_row(law, layout::AT_ARG, u32::from(index), Some(place))
                    }
                    // `abi::read` enumerates exactly the three places above,
                    // so a derived table holds no other kind and the shader
                    // has no row shape for one.
                    At::Entry | At::Shared | At::Shape => {
                        return Err(Fault::Unstructured {
                            slot: slot_rows.len() as u32,
                            why: format!(
                                "the law table states a `{at}` component and the Metal \
                                 recorder derives none"
                            ),
                        });
                    }
                });
            }
            arm_rows.push(layout::ArmRow::new(
                pipe_index[&(arm.point.file, arm.point.entrypoint)],
                law_at,
                law_rows.len() as u32 - law_at,
                bind_at,
                bind_rows.len() as u32 - bind_at,
                arm.skeleton.lanes,
                arm.skeleton.group,
            ));
        }
        let (pick, threshold) = match slot.pick {
            Pick::Only => (layout::PICK_ONLY, 0),
            Pick::Rows { at } => (layout::PICK_ROWS, at),
        };
        slot_rows.push(layout::SlotRow::new(
            arm_at,
            arm_rows.len() as u32 - arm_at,
            pick,
            threshold,
            rows_law,
        ));
    }

    // 4. The coordinate recipe, flat: one constant per direction and
    //    `axes × 2 × classes` coefficients.
    let mut konst: Vec<i64> = Vec::with_capacity(abi.axes.len());
    let mut coeff: Vec<i64> = Vec::with_capacity(abi.axes.len() * 2 * classes);
    for row in &abi.recipe {
        konst.push(narrow(row.konst));
        for c in 0..classes {
            coeff.push(narrow(row.rows.get(c).copied().unwrap_or(0)));
            coeff.push(narrow(row.lanes.get(c).copied().unwrap_or(0)));
        }
    }

    // 5. The reservations.
    let plan = layout::Plan::new(
        abi.len() as u32,
        abi.axes.len() as u32,
        classes as u32,
        model_exec::fire::MAGIC,
        model_exec::fire::ABI_VERSION,
    );
    let mut handle = Buffer::zeroed(device, 8)?;
    handle.write(0, &resource_id(icb.gpuResourceID()).to_ne_bytes())?;
    let mut pipes = Buffer::zeroed(device, (layout::MAX_PIPELINES * 8) as u64)?;
    for (at, pipeline) in retained.iter().enumerate() {
        pipes.write(at as u64 * 8, &resource_id(pipeline.gpuResourceID()).to_ne_bytes())?;
    }
    let cells = Buffer::zeroed(device, (cell_bytes.len() as u64).max(16))?;
    let mut cells = cells;
    if !cell_bytes.is_empty() {
        cells.write(0, &cell_bytes)?;
    }
    let mut slabs_arg = Buffer::zeroed(device, (layout::MAX_SLABS * 8) as u64)?;
    for (at, address) in addresses.iter().enumerate() {
        slabs_arg.write(at as u64 * 8, &address.to_ne_bytes())?;
    }

    let census = Lowered {
        slots: abi.len(),
        arms: arm_rows.len(),
        laws: law_rows.len(),
        binds: bind_rows.len(),
        pipelines: retained.len(),
        slabs: addresses.len(),
        cells: cells.bytes(),
        bytes: (std::mem::size_of_val(slot_rows.as_slice())
            + std::mem::size_of_val(arm_rows.as_slice())
            + std::mem::size_of_val(law_rows.as_slice())
            + std::mem::size_of_val(bind_rows.as_slice())) as u64,
    };

    let mut rebinder = Rebinder {
        pipeline: pipelines.at(
            device.device(),
            kernels_metal::Fire::at(layout::FILE, layout::ENTRYPOINT),
        )?,
        handle,
        pipes,
        slabs: slabs_arg,
        plan: stored(device, layout::bytes_of(std::slice::from_ref(&plan)))?,
        descriptor: Buffer::zeroed(device, room.max(64))?,
        konst: stored(device, words(&konst))?,
        coeff: stored(device, words(&coeff))?,
        slot_rows: stored(device, layout::bytes_of(&slot_rows))?,
        arm_rows: stored(device, layout::bytes_of(&arm_rows))?,
        law_rows: stored(device, layout::bytes_of(&law_rows))?,
        bind_rows: stored(device, layout::bytes_of(&bind_rows))?,
        pipe_rows: stored(device, layout::bytes_of(&pipe_rows))?,
        live: Buffer::zeroed(device, (abi.len().max(1) * 4) as u64)?,
        status: Buffer::zeroed(device, 16)?,
        cells,
        retained,
        room: room.max(64),
        census,
    };
    // **NOTHING IS ASSUMED TO BE ENCODED.** The buffer was built by a walk
    // and the shader diffs against `live`; starting every word at "the host
    // touched this" makes the first rebind a full re-encode, which is the one
    // state that is right whatever the builder left behind.
    rebinder.desync()?;
    Ok(rebinder)
}

/// One law, as the shader's row.
fn law_row(law: &Law, at_kind: u32, at_index: u32, place: Option<(u32, u32, u32)>) -> layout::LawRow {
    let mut row = match law {
        Law::Const(v) => {
            let mut row = layout::LawRow::at(layout::LAW_CONST, at_kind, at_index);
            row.base = narrow(*v);
            row
        }
        Law::Affine { base, slope } => {
            let mut row = layout::LawRow::at(layout::LAW_AFFINE, at_kind, at_index);
            row.base = narrow(*base);
            for (k, b) in slope.iter().enumerate().take(layout::MAX_AXES) {
                row.slope[k] = narrow(*b);
            }
            row
        }
        Law::Ceil {
            mul,
            alpha,
            beta,
            div,
        } => {
            let mut row = layout::LawRow::at(layout::LAW_CEIL, at_kind, at_index);
            row.mul = narrow(*mul);
            row.alpha = narrow(*alpha);
            row.beta = narrow(*beta);
            row.div = narrow(*div);
            row
        }
        // The one form no fit produces (`model_exec::law::Law::Slot`): a number
        // read out of the fire's own descriptor rather than solved from its
        // coordinates. The rebind shader has no row for it, and
        // `lower` refuses the table before it reaches here.
        Law::Slot(id) => unreachable!("the Metal fit never states {id}"),
    };
    if let Some((arg_kind, slab, cell)) = place {
        row.arg_kind = arg_kind;
        row.slab = slab;
        row.cell = cell;
    }
    row
}

/// Where an argument law writes: into a binding's offset, or into a staged
/// cell of one of two widths.
fn argument_place(
    arm: &Arm,
    index: usize,
    slab_index: &HashMap<u64, u32>,
    cells: &[Option<u64>],
) -> (u32, u32, u32) {
    match arm.skeleton.args[index] {
        Arg::Buffer { slab, .. } => (layout::ARG_OFFSET, slab_index[&slab], 0),
        Arg::Usize(_) => (
            layout::ARG_WIDE,
            0,
            cells[index].unwrap_or(0) as u32,
        ),
        _ => (layout::ARG_WORD, 0, cells[index].unwrap_or(0) as u32),
    }
}

fn narrow(v: i128) -> i64 {
    i64::try_from(v).unwrap_or(if v < 0 { i64::MIN } else { i64::MAX })
}

fn words(values: &[i64]) -> &[u8] {
    // SAFETY: `i64` is plain data; `u8` has no alignment requirement.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

fn stored(device: &Context, bytes: &[u8]) -> Result<Buffer> {
    let mut buffer = Buffer::zeroed(device, (bytes.len() as u64).max(16))?;
    if !bytes.is_empty() {
        buffer.write(0, bytes)?;
    }
    Ok(buffer)
}

/// Rewrite the buffer from the descriptor and run it: one command buffer, two
/// passes, no walk.
///
/// # Errors
///
/// [`Fault::Ceiling`] for a descriptor wider than the reservation,
/// [`Fault::Device`] when the GPU refused, [`Fault::Unstructured`] when the
/// shader itself refused the descriptor.
pub(crate) fn fire(
    rebinder: &mut Rebinder,
    device: &Context,
    icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
    residents: &[Slab],
    slots: usize,
    descriptor: &[u8],
) -> Result<()> {
    if descriptor.len() as u64 > rebinder.room {
        return Err(Fault::Ceiling {
            what: "bytes of fire descriptor the rebind path reserved",
            need: descriptor.len() as u64,
            have: rebinder.room,
        });
    }
    // **THE ONE HOST WRITE OF A FIRE**, and it is a memcpy into a Shared
    // mapping: the reservation IS the bytes the GPU reads.
    rebinder.descriptor.write(0, descriptor)?;
    rebinder.status.write(0, &0u64.to_ne_bytes())?;

    let mut frame = device.frame()?;
    {
        let encoder = frame.encoder();
        encoder.setComputePipelineState(&rebinder.pipeline);
        let bindings: [(usize, &Buffer); layout::BINDINGS] = [
            (layout::HANDLE, &rebinder.handle),
            (layout::PIPES, &rebinder.pipes),
            (layout::SLABS, &rebinder.slabs),
            (layout::PLAN, &rebinder.plan),
            (layout::DESCRIPTOR, &rebinder.descriptor),
            (layout::KONST, &rebinder.konst),
            (layout::COEFF, &rebinder.coeff),
            (layout::SLOTS, &rebinder.slot_rows),
            (layout::ARMS, &rebinder.arm_rows),
            (layout::LAWS, &rebinder.law_rows),
            (layout::BINDS, &rebinder.bind_rows),
            (layout::PIPE_FACTS, &rebinder.pipe_rows),
            (layout::LIVE, &rebinder.live),
            (layout::STATUS, &rebinder.status),
            (layout::CELLS, &rebinder.cells),
        ];
        for (index, buffer) in bindings {
            // SAFETY: every reservation outlives the command buffer, because
            // the `Rebinder` does.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer.raw()), 0, index) };
        }
        let target: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(icb);
        encoder.useResource_usage(target, MTLResourceUsage::Write);
        let grid = MTLSize {
            width: slots.max(1),
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(
            grid,
            crate::device::ctx::threadgroup(&rebinder.pipeline, [slots.max(1) as u32, 1, 1]),
        );
    }
    {
        // The second pass observes everything the first wrote, which is the
        // whole reason there are two.
        let encoder = frame.next_pass()?;
        for slab in residents {
            let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**slab);
            encoder.useResource_usage(resource, MTLResourceUsage::Read | MTLResourceUsage::Write);
        }
        let cells: &ProtocolObject<dyn MTLResource> =
            ProtocolObject::from_ref(&**rebinder.cells.slab());
        encoder.useResource_usage(cells, MTLResourceUsage::Read);
        // SAFETY: the range is `0..slots` of the buffer the shader above just
        // wrote, and it outlives the command buffer.
        unsafe { encoder.executeCommandsInBuffer_withRange(icb, NSRange::new(0, slots)) };
    }
    frame.commit()?;

    let mut said = [0u8; 4];
    rebinder.status.read(0, &mut said)?;
    match u32::from_ne_bytes(said) {
        0 => Ok(()),
        layout::STATUS_MAGIC => Err(Fault::Unstructured {
            slot: 0,
            why: "the rebind shader read no FIRE magic in the descriptor it was handed"
                .to_string(),
        }),
        layout::STATUS_VERSION => Err(Fault::Unstructured {
            slot: 0,
            why: "the rebind shader read a descriptor ABI version this table was not \
                  lowered for"
                .to_string(),
        }),
        layout::STATUS_CLASSES => Err(Fault::Unstructured {
            slot: 0,
            why: "the descriptor carries a different number of classes than the derived \
                  table's coordinate recipe reads"
                .to_string(),
        }),
        other => Err(Fault::Unstructured {
            slot: 0,
            why: format!("the rebind shader refused with code {other}"),
        }),
    }
}
