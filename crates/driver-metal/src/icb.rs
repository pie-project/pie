//! The indirect command buffer: the walk, encoded ONCE, and rewritten
//! afterwards instead of re-encoded (`.wiki/palo/icb.md` §2).
//!
//! **THE ONE THING THIS BUYS.** A fire's dispatches are encoded into a fresh
//! `MTLComputeCommandEncoder` every time, and that encode is host work
//! proportional to the artifact: measured on the M1 Max, 465 dispatches of
//! eighteen bindings each cost **319 µs of host time per fire**. An
//! `MTLIndirectCommandBuffer` holds the same 465 commands permanently; a fire
//! rewrites the components that moved and issues one
//! `executeCommandsInBuffer:`, which costs **4.4 µs**. Nothing about the GPU
//! work changes — a barriered ICB slot is the same serialized dispatch a
//! compute pass gives — so what is removed is exactly the encode.
//!
//! # What an ICB slot is not, measured on the M1 Max
//!
//! Three differences from a compute pass, none of them optional:
//!
//! - **There is no `setBytes:`.** `MTLIndirectComputeCommand` binds buffers
//!   and nothing else, so every scalar a `kernels-metal` entry marshals as
//!   `ArgValue::I32`/`U32`/`F32`/`Usize` has to live in a device buffer and
//!   be bound at its index. That is what [`Icb::constants`] is: one
//!   reservation, one cell per scalar argument of every slot, written at
//!   build and rewritten in place when a value moves. **The shaders need no
//!   change** — a `constant int&` parameter reads a buffer binding and an
//!   inline `setBytes` identically, which is measured, not assumed.
//! - **Commands are CONCURRENT unless a barrier says otherwise.** A compute
//!   pass is `MTLDispatchTypeSerial` and the walk relies on it. An ICB's
//!   dispatch types are the concurrent ones, so every slot carries
//!   `setBarrier()` — measured: sixteen chained slots produce the serial
//!   answer with barriers and a raced one without, every run.
//! - **A pipeline must be built for it.** `MTLComputePipelineDescriptor`'s
//!   `supportIndirectCommandBuffers` is false by default and a pipeline
//!   without it cannot be set into a slot, so [`Pipelines`] builds every
//!   pipeline through the descriptor now — one path, both consumers.
//!
//! # Rebinding, and what this wave does and does not claim
//!
//! [`Icb::rebind`] takes a [`Recording`] of the walk at the target
//! composition and writes only what differs — grids, buffer offsets, scalar
//! cells, and the pipeline of a slot whose entry picked another arm. So ONE
//! indirect command buffer serves every composition, which is decision #2's
//! own promise, and the exec key is gone: there is no key.
//!
//! What it does not yet do is read the descriptor on the DEVICE. The
//! recording that drives the rebind is host arithmetic — the same walk,
//! without the encode — and `crate::abi` is the table that would replace it
//! with a `v = base + Σ slope · axis` per component. That table is derived
//! and verified separately; wiring it into a rebind SHADER is the next step
//! and is stated here rather than implied.

#![cfg(target_vendor = "apple")]

use std::cell::RefCell;
use std::collections::HashMap;

use kernels_metal::{ArgValue, Encode, Fire, KernelError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLComputeCommandEncoder, MTLDevice, MTLIndirectCommandBuffer,
    MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType, MTLIndirectComputeCommand,
    MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSize,
};
use objc2_foundation::NSRange;

use crate::device::alloc::{Buffer, Slab, slab_id};
use crate::device::library::Pipeline;
use crate::device::{Context, Handles, Pipelines, handles::NIL};
use crate::error::{Fault, Result};
use crate::record::{Arg, Recording, Slot};

/// The widest argument list any `kernels-metal` entry marshals, plus room.
///
/// Measured rather than guessed: `driver-metal`'s own recording of
/// qwen35-d0.8b binds at most eighteen arguments in one dispatch
/// (`descriptor_abi::every_capture_phase_slot_is_a_dispatch`). The ceiling is
/// stated at ICB creation and cannot be raised afterwards, and the M1 Max
/// accepts 8, 16, 31, 32, 64 and 128 alike, so this is a `Budgets`-shaped
/// number with room rather than a device limit.
pub const MAX_BINDINGS: usize = 32;

/// How a rebind writes one argument back.
#[derive(Clone)]
enum Bound {
    /// A buffer binding: the reservation stays, the offset can move.
    Buf { slab: Slab, offset: u64 },
    /// A scalar, living in [`Icb::constants`] at this byte offset. The
    /// BINDING never moves; the bytes do.
    Scalar { at: u64, width: u8 },
    /// An index the shader does not dereference on this arm. Bound to the
    /// constants reservation at zero, because `setKernelBuffer:offset:atIndex:`
    /// takes a buffer and not an option — the one place this path cannot say
    /// what `Sink` says (`setBuffer:nil`), and it is equivalent for exactly
    /// the reason `absent` exists: nothing reads the slot.
    Absent,
}

/// One encoded slot, as the rebind needs to see it.
struct Built {
    point: crate::record::Point,
    /// The compiled pipeline, retained — a slot that was RESET has to be
    /// encoded again from here, and the cache would answer the same object
    /// anyway.
    pipeline: Pipeline,
    args: Vec<Bound>,
    lanes: [u32; 3],
    group: MTLSize,
    /// Which template region and which run of its window this slot stood in.
    /// **THE ALIGNMENT KEY**: a composition with an empty window walks fewer
    /// dispatches, and this is what says which of the buffer's slots the ones
    /// it did walk are.
    region: u32,
    run: u32,
    /// Whether the slot currently holds a command. A slot standing in a
    /// window this composition has no rows for is `reset()` and skipped by
    /// the execution — the device-side reading of `walk`'s rule 1.
    live: bool,
}

/// One artifact's dispatches, encoded once.
pub struct Icb {
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    /// The scalar arena: an ICB slot cannot carry inline bytes, so every
    /// scalar argument of every slot has a cell here.
    constants: Buffer,
    /// Per slot, what was encoded — the rebind's diff base.
    built: Vec<Built>,
    /// Every reservation any slot binds, once each: the residency list a
    /// fire declares with `useResource:`.
    residents: Vec<Slab>,
    /// Reservation identity → the retained buffer, so a rebind can move an
    /// offset without re-resolving a handle table that has been rewound.
    slabs: HashMap<u64, Slab>,
    /// The device-side rebind, once a derived table has been lowered into it
    /// (`crate::rebind`). `None` until [`Icb::attach`], and this buffer is
    /// rewritten by the HOST meanwhile — which is what keeps the shader an
    /// ADDITION rather than a fork.
    rebinder: Option<crate::rebind::Rebinder>,
    /// Whether something other than the host rebind last wrote the commands.
    ///
    /// **TWO REBINDS, ONE BUFFER, AND NEITHER TRUSTS THE OTHER'S RECORD.**
    /// `rebind` diffs against `built`; the shader diffs against its own
    /// `live` words. A fire through one after a fire through the other would
    /// diff against a record of somebody else's writes, so each marks the
    /// other stale and a stale rebind writes every component rather than the
    /// ones it believes moved.
    stale: bool,
}

// SAFETY: as `Buffer` and `Pipelines` — the Metal objects here are
// documented thread-safe for retain/release and for binding, and an `Icb` is
// built, rebound and executed on the one lane thread that owns the shell.
unsafe impl Send for Icb {}

impl std::fmt::Debug for Icb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Icb")
            .field("slots", &self.built.len())
            .field("residents", &self.residents.len())
            .field("constants", &self.constants.bytes())
            .field("rebinder", &self.rebinder)
            .finish()
    }
}

impl Icb {
    /// How many dispatches this artifact holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.built.len()
    }

    /// Whether the artifact dispatches nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.built.is_empty()
    }

    /// How many distinct reservations a fire must declare resident.
    #[must_use]
    pub fn residents(&self) -> usize {
        self.residents.len()
    }

    /// How many bytes the scalar arena holds.
    #[must_use]
    pub fn constant_bytes(&self) -> u64 {
        self.constants.bytes()
    }

    /// Lower a derived [`DescriptorAbi`](crate::abi::DescriptorAbi) into the
    /// device tables `icb::rebind` reads, so this buffer can be rewritten by
    /// the GPU from the descriptor alone.
    ///
    /// `room` is how many bytes of packed fire descriptor to reserve — a
    /// number the caller has, since it is `FireDescriptor::bytes()` at the
    /// widest batch its budgets admit.
    ///
    /// # Errors
    ///
    /// As `crate::rebind::lower`: [`Fault::Unstructured`] when the table is
    /// not this buffer's, [`Fault::Ceiling`] past the shader's fixed arrays,
    /// [`Fault::Shader`] for an arm that will not compile.
    pub fn attach(
        &mut self,
        device: &Context,
        pipelines: &Pipelines,
        abi: &crate::abi::DescriptorAbi,
        room: u64,
    ) -> Result<crate::rebind::Lowered> {
        let encoded: Vec<(u32, u32)> = self.built.iter().map(|s| (s.region, s.run)).collect();
        let rebinder = crate::rebind::lower(
            device,
            pipelines,
            &self.icb,
            &encoded,
            abi,
            &self.slabs,
            room,
        )?;
        let census = rebinder.census();
        self.rebinder = Some(rebinder);
        // The lowering says nothing about what the builder left encoded, and
        // the first shader rebind writes every slot from scratch — so the
        // host's record of what is in the buffer is void from here.
        self.stale = true;
        Ok(census)
    }

    /// Whether a derived table has been lowered into this buffer.
    #[must_use]
    pub fn rebinds_on_device(&self) -> bool {
        self.rebinder.is_some()
    }

    /// What the lowering produced.
    #[must_use]
    pub fn lowered(&self) -> Option<crate::rebind::Lowered> {
        self.rebinder.as_ref().map(crate::rebind::Rebinder::census)
    }

    /// One fire with no host walk at all: write the descriptor, dispatch
    /// `icb::rebind`, execute the buffer.
    ///
    /// **THIS IS THE WAVE'S WHOLE CLAIM IN ONE FUNCTION.** What reaches the
    /// device is the packed descriptor and nothing else; the grids, the
    /// windowed cuts, the staged scalars, the arm of every entry that has two
    /// and the reset of every slot standing in an empty window are all
    /// computed by the GPU out of the law table lowered at load.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] when no table was lowered, and whatever
    /// `crate::rebind::fire` said.
    pub fn execute_rebound(&mut self, device: &Context, descriptor: &[u8]) -> Result<()> {
        let Icb {
            icb,
            built,
            residents,
            rebinder,
            ..
        } = self;
        let rebinder = rebinder.as_mut().ok_or_else(|| Fault::Unbound {
            what: "a lowered descriptor table, which this buffer never had one attached"
                .to_string(),
        })?;
        crate::rebind::fire(rebinder, device, icb, residents, built.len(), descriptor)
    }

    /// Rewrite every component of every slot that this composition moves.
    ///
    /// `taped` is the walk at the target composition, recorded rather than
    /// encoded — so this function compares what the fire WOULD have encoded
    /// against what the buffer holds, and writes the difference.
    ///
    /// # Errors
    ///
    /// [`Fault::Unstructured`] when the recording is not this artifact's —
    /// a different slot count, an argument that changed kind, or a binding
    /// into a reservation this build never saw. [`Fault::Shader`] when a slot
    /// switched to an entry that will not compile.
    pub fn rebind(
        &mut self,
        device: &Context,
        pipelines: &Pipelines,
        taped: &Recording,
    ) -> Result<Rebound> {
        // **THE ALIGNMENT, AND IT IS THE WHOLE OF WHAT MAKES ONE BUFFER SERVE
        // EVERY COMPOSITION.** `driver::fire::walk` skips a zero-row region's
        // nodes, so an all-decode fire walks FEWER dispatches than the mixed
        // one this buffer was built at. Design §5's answer is that the
        // artifact holds every launch and the fire turns the absent ones off,
        // and `(region, run)` is what says which. A region is dispatched
        // whole or not at all — that equality is what defines a region (P2) —
        // so the key is exact rather than a heuristic.
        let mut wanted: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (at, slot) in taped.slots.iter().enumerate() {
            wanted.entry((slot.region, slot.run)).or_default().push(at);
        }
        let mut held: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (at, slot) in self.built.iter().enumerate() {
            held.entry((slot.region, slot.run)).or_default().push(at);
        }
        for key in wanted.keys() {
            if !held.contains_key(key) {
                return Err(Fault::Unstructured {
                    slot: 0,
                    why: format!(
                        "this composition dispatches region {} run {} and the buffer was \
                         built at a composition that did not — an indirect command buffer \
                         holds the launches of the composition it was encoded at, so it \
                         must be encoded at one that holds every class",
                        key.0, key.1
                    ),
                });
            }
        }

        // A buffer the shader last wrote is a buffer this record does not
        // describe, so every slot is written again rather than diffed.
        let forced = std::mem::take(&mut self.stale);
        let mut moved = Rebound::default();
        for (key, slots) in &held {
            match wanted.get(key) {
                None => {
                    // The window is empty in this composition. Turn the slots
                    // off rather than dispatching them at zero rows: measured
                    // on the M1 Max, a reset slot inside an executed range is
                    // skipped with no error, and a barriered slot that merely
                    // exits immediately still costs 2.13 µs.
                    for index in slots {
                        if self.built[*index].live || forced {
                            unsafe { self.icb.indirectComputeCommandAtIndex(*index).reset() };
                            self.built[*index].live = false;
                            moved.turned_off += 1;
                        }
                    }
                }
                Some(theirs) => {
                    if theirs.len() != slots.len() {
                        return Err(Fault::Unstructured {
                            slot: slots[0] as u32,
                            why: format!(
                                "region {} run {} dispatches {} launches here and {} in the \
                                 buffer — a region is dispatched whole or not at all",
                                key.0,
                                key.1,
                                theirs.len(),
                                slots.len()
                            ),
                        });
                    }
                    for (index, source) in slots.iter().zip(theirs) {
                        self.slot(
                            device,
                            pipelines,
                            *index,
                            &taped.slots[*source],
                            forced,
                            &mut moved,
                        )?;
                    }
                }
            }
        }
        // And the shader's own record is void for the same reason this one
        // was: the host has just written every command it names.
        if let Some(rebinder) = self.rebinder.as_mut() {
            rebinder.desync()?;
        }
        Ok(moved)
    }

    /// One slot, rewritten — or encoded again, if it had been turned off.
    fn slot(
        &mut self,
        device: &Context,
        pipelines: &Pipelines,
        index: usize,
        wanted: &Slot,
        forced: bool,
        moved: &mut Rebound,
    ) -> Result<()> {
        let command = unsafe { self.icb.indirectComputeCommandAtIndex(index) };
        // A slot that was reset holds nothing at all, so every part of it is
        // written again rather than diffed against a command that is gone.
        let revived = forced || !self.built[index].live;
        let rearmed = revived || self.built[index].point != wanted.point;
        if rearmed {
            // The entry picked another arm — `linear`'s gemv/gemm split is the
            // live one — so the slot is a different pipeline. Measured on the
            // M1 Max: a compute ICB slot's pipeline state is settable, from
            // the host here and from a shader in the step after this one.
            let pipeline = pipelines.at(device.device(), fire_of(wanted))?;
            command.setComputePipelineState(&pipeline);
            self.built[index].pipeline = pipeline;
            self.built[index].point = wanted.point;
            if !revived {
                moved.pipelines += 1;
            }
        }
        self.arguments(index, wanted, &command, rearmed, revived, moved)?;
        let group = grid_of(device, pipelines, wanted)?;
        if rearmed || self.built[index].lanes != wanted.lanes || self.built[index].group != group {
            command.concurrentDispatchThreads_threadsPerThreadgroup(size(wanted.lanes), group);
            self.built[index].lanes = wanted.lanes;
            self.built[index].group = group;
            if !revived {
                moved.grids += 1;
            }
        }
        if revived {
            command.setBarrier();
            self.built[index].live = true;
            moved.turned_on += 1;
        }
        Ok(())
    }

    /// One slot's arguments, rewritten where they moved.
    fn arguments(
        &mut self,
        index: usize,
        wanted: &Slot,
        command: &ProtocolObject<dyn MTLIndirectComputeCommand>,
        rearmed: bool,
        revived: bool,
        moved: &mut Rebound,
    ) -> Result<()> {
        if wanted.args.len() != self.built[index].args.len() {
            return Err(Fault::Unstructured {
                slot: index as u32,
                why: format!(
                    "{} binds {} arguments and the encoded slot holds {}",
                    wanted.point,
                    wanted.args.len(),
                    self.built[index].args.len()
                ),
            });
        }
        for (at, arg) in wanted.args.iter().enumerate() {
            match (self.built[index].args[at].clone(), *arg) {
                (Bound::Buf { slab, offset }, Arg::Buffer { slab: want, offset: to, .. }) => {
                    if !rearmed && offset == to && slab_id(&slab) == want {
                        continue;
                    }
                    let slab = self.slabs.get(&want).cloned().ok_or(Fault::Unstructured {
                        slot: index as u32,
                        why: format!(
                            "argument {at} binds a reservation this artifact's build never \
                             saw — an ICB slot binds an address, so a fire cannot introduce \
                             one"
                        ),
                    })?;
                    unsafe {
                        command.setKernelBuffer_offset_atIndex(
                            &slab,
                            usize::try_from(to).expect("an offset inside a reservation"),
                            at,
                        );
                    }
                    self.built[index].args[at] = Bound::Buf { slab, offset: to };
                    moved.offsets += 1;
                }
                (Bound::Scalar { at: cell, width }, value) => {
                    let Some(bytes) = scalar_bytes(value) else {
                        return Err(Fault::Unstructured {
                            slot: index as u32,
                            why: format!(
                                "argument {at} is a scalar in the encoded slot and {} in this \
                                 composition",
                                value.kind()
                            ),
                        });
                    };
                    if bytes.len() as u8 != width {
                        return Err(Fault::Unstructured {
                            slot: index as u32,
                            why: format!(
                                "argument {at} is {width} bytes wide in the encoded slot and \
                                 {} here",
                                bytes.len()
                            ),
                        });
                    }
                    // **A REVIVED SLOT HAS NO BINDINGS AT ALL.** `reset()`
                    // clears the command, so a scalar's cell has to be bound
                    // again and not merely refilled — the bug this line exists
                    // to state, and the one a test that never turned a slot
                    // back on would not have found.
                    if revived {
                        unsafe {
                            command.setKernelBuffer_offset_atIndex(
                                self.constants.slab(),
                                usize::try_from(cell).expect("a cell inside the scalar arena"),
                                at,
                            );
                        }
                    }
                    let mut held = [0u8; 8];
                    self.constants.read(cell, &mut held[..bytes.len()])?;
                    if held[..bytes.len()] == bytes[..] {
                        continue;
                    }
                    // The BINDING does not move: the cell is where the shader
                    // already looks, and only the bytes in it change.
                    self.constants.write(cell, &bytes)?;
                    moved.scalars += 1;
                }
                (Bound::Absent, Arg::Absent) => {
                    if revived {
                        // As above: the nil stand-in is a binding like any
                        // other and a reset slot lost it.
                        unsafe {
                            command.setKernelBuffer_offset_atIndex(self.constants.slab(), 0, at);
                        }
                    }
                }
                (held, value) => {
                    return Err(Fault::Unstructured {
                        slot: index as u32,
                        why: format!(
                            "argument {at} is {} in the encoded slot and {} in this \
                             composition",
                            match held {
                                Bound::Buf { .. } => "a buffer",
                                Bound::Scalar { .. } => "a scalar",
                                Bound::Absent => "an absent binding",
                            },
                            value.kind()
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Execute the whole buffer: one command buffer, one pass, one call.
    ///
    /// The residency declaration is the whole of the ceremony an ICB adds —
    /// its commands bind by address, so the reservations behind those
    /// addresses have to be declared. Measured at 135 ns per resource on the
    /// M1 Max, over a list this shell can count on one hand.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] when the queue would not open, or the GPU refused.
    pub fn execute(&self, device: &Context) -> Result<()> {
        let frame = device.frame()?;
        {
            let encoder = frame.encoder();
            for slab in &self.residents {
                let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**slab);
                encoder.useResource_usage(
                    resource,
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
            }
            let constants: &ProtocolObject<dyn MTLResource> =
                ProtocolObject::from_ref(&**self.constants.slab());
            encoder.useResource_usage(constants, MTLResourceUsage::Read);
            // SAFETY: the range is `0..len` of a buffer whose every slot was
            // encoded by `Builder::finish`, and the buffer outlives the
            // command buffer because `self` does.
            unsafe {
                encoder.executeCommandsInBuffer_withRange(
                    &self.icb,
                    NSRange::new(0, self.built.len()),
                );
            }
        }
        frame.commit()
    }
}

/// What one rebind moved — the observable behind "the descriptor is the one
/// mutable channel".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rebound {
    /// Buffer bindings re-pointed.
    pub offsets: usize,
    /// Scalar cells rewritten in place.
    pub scalars: usize,
    /// Grids re-stated.
    pub grids: usize,
    /// Slots whose entry picked another arm.
    pub pipelines: usize,
    /// Slots standing in a window this composition has no rows for, reset.
    pub turned_off: usize,
    /// Slots that had been reset and this composition wants back.
    pub turned_on: usize,
}

impl std::fmt::Display for Rebound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} offsets, {} scalars, {} grids, {} pipelines, {} off, {} on",
            self.offsets, self.scalars, self.grids, self.pipelines, self.turned_off, self.turned_on
        )
    }
}

impl Rebound {
    /// Everything, added up.
    #[must_use]
    pub fn total(self) -> usize {
        self.offsets + self.scalars + self.grids + self.pipelines + self.turned_off
            + self.turned_on
    }
}

/// The `Encode` that builds an [`Icb`] instead of encoding a pass.
///
/// Stands exactly where [`Sink`](crate::Sink) and [`Tape`](crate::Tape)
/// stand, over the same walk and the same `Run`: one `Encode::fire` is one
/// Metal dispatch is one ICB slot, and this is where the third reading of
/// that sentence is written down.
pub struct Builder<'a> {
    device: &'a Context,
    pipelines: &'a Pipelines,
    handles: &'a Handles,
    /// Where the walk is — the region and run a slot stood in, which is the
    /// key a later composition is aligned onto.
    place: &'a crate::window::At,
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    constants: RefCell<Buffer>,
    /// The next free byte of the scalar arena.
    cursor: std::cell::Cell<u64>,
    built: RefCell<Vec<Built>>,
    slabs: RefCell<HashMap<u64, Slab>>,
    ceiling: usize,
}

impl<'a> Builder<'a> {
    /// Reserve an indirect command buffer for `slots` dispatches and a scalar
    /// arena of `constants` bytes, both sized from a prior recording of the
    /// same walk.
    ///
    /// **THE CEILING IS FIXED AT CREATION AND THAT IS FINE.** `maxCommandCount`
    /// cannot be raised, so it is known after the recording and allocated
    /// after it — `.wiki/palo/icb.md` §8's own answer. Measured on the M1 Max:
    /// 4 194 304 commands allocate (729 bytes each); 8 388 608 aborts the
    /// process inside the driver rather than answering nil, so a shell that
    /// asks for one must have counted first.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] when the device would not reserve the buffer.
    pub fn new(
        device: &'a Context,
        pipelines: &'a Pipelines,
        handles: &'a Handles,
        place: &'a crate::window::At,
        slots: usize,
        constants: u64,
    ) -> Result<Builder<'a>> {
        let descriptor = MTLIndirectCommandBufferDescriptor::new();
        descriptor.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatchThreads);
        descriptor.setInheritPipelineState(false);
        descriptor.setInheritBuffers(false);
        descriptor.setMaxKernelBufferBindCount(MAX_BINDINGS);
        // SAFETY: the descriptor is fully stated above and the count is a
        // number this shell counted from a recording of the same walk.
        let icb = unsafe {
            device
                .device()
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &descriptor,
                    slots.max(1),
                    MTLResourceOptions::StorageModeShared,
                )
        }
        .ok_or(Fault::Device {
                call: "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
            why: format!("the device would not reserve {slots} indirect commands"),
        })?;
        Ok(Builder {
            device,
            pipelines,
            handles,
            place,
            icb,
            constants: RefCell::new(Buffer::zeroed(device, constants.max(16))?),
            cursor: std::cell::Cell::new(0),
            built: RefCell::new(Vec::with_capacity(slots)),
            slabs: RefCell::new(HashMap::new()),
            ceiling: slots,
        })
    }

    /// Seal the buffer.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] when the walk encoded fewer slots than were
    /// reserved — a buffer with an unencoded slot in its range is a command
    /// nothing wrote, and executing it is undefined.
    pub fn finish(self) -> Result<Icb> {
        let built = self.built.into_inner();
        if built.len() != self.ceiling {
            return Err(Fault::Ceiling {
                what: "indirect commands encoded against the count reserved",
                need: built.len() as u64,
                have: self.ceiling as u64,
            });
        }
        let slabs = self.slabs.into_inner();
        let mut residents: Vec<Slab> = slabs.values().cloned().collect();
        residents.sort_by_key(slab_id);
        Ok(Icb {
            icb: self.icb,
            constants: self.constants.into_inner(),
            built,
            residents,
            slabs,
            rebinder: None,
            stale: false,
        })
    }

    /// A cell in the scalar arena for `bytes`, eight-byte aligned so a
    /// `size_t` seat is legal at it.
    fn cell(&self, bytes: &[u8]) -> Result<u64> {
        let at = self.cursor.get();
        self.cursor.set(at + 8);
        self.constants.borrow_mut().write(at, bytes)?;
        Ok(at)
    }
}

impl Encode for Builder<'_> {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> std::result::Result<(), KernelError> {
        self.encode(fire, args).map_err(|fault| KernelError::Backend {
            op: fire.entrypoint,
            detail: fault.to_string(),
        })
    }

    fn absent(&self) -> std::result::Result<ArgValue, KernelError> {
        Ok(ArgValue::Buffer(NIL))
    }
}

impl Builder<'_> {
    /// One dispatch into one slot.
    fn encode(&self, fire: Fire, args: &[ArgValue]) -> Result<()> {
        let index = self.built.borrow().len();
        if index >= self.ceiling {
            return Err(Fault::Ceiling {
                what: "indirect commands",
                need: index as u64 + 1,
                have: self.ceiling as u64,
            });
        }
        if args.len() > MAX_BINDINGS {
            return Err(Fault::Ceiling {
                what: "kernel buffer bindings in one indirect command",
                need: args.len() as u64,
                have: MAX_BINDINGS as u64,
            });
        }
        let pipeline = self.pipelines.at(self.device.device(), fire)?;
        let command = unsafe { self.icb.indirectComputeCommandAtIndex(index) };
        command.setComputePipelineState(&pipeline);

        let mut bound = Vec::with_capacity(args.len());
        for (at, arg) in args.iter().enumerate() {
            bound.push(self.bind(&command, fire, at, *arg)?);
        }

        let group = if fire.group == [0, 0, 0] {
            crate::device::ctx::threadgroup(&pipeline, fire.lanes)
        } else {
            size(fire.group)
        };
        command.concurrentDispatchThreads_threadsPerThreadgroup(size(fire.lanes), group);
        // **EVERY SLOT CARRIES A BARRIER.** An indirect command buffer's
        // dispatches are concurrent by kind; the walk assumes a serial pass.
        // Measured: without this, sixteen slots chained through one word
        // produce the raced answer on every run.
        command.setBarrier();

        self.built.borrow_mut().push(Built {
            point: point_of(fire),
            pipeline,
            args: bound,
            lanes: fire.lanes,
            group,
            region: self.place.region.get(),
            run: self.place.run.get(),
            live: true,
        });
        Ok(())
    }

    /// One argument at one index, in the ICB's vocabulary.
    fn bind(
        &self,
        command: &ProtocolObject<dyn MTLIndirectComputeCommand>,
        fire: Fire,
        at: usize,
        arg: ArgValue,
    ) -> Result<Bound> {
        let (handle, _) = match arg {
            ArgValue::Buffer(handle) => (handle, false),
            ArgValue::BufferMut(handle) => (handle, true),
            scalar => {
                let bytes = scalar_bytes_of(scalar);
                let cell = self.cell(&bytes)?;
                let constants = self.constants.borrow();
                unsafe {
                    command.setKernelBuffer_offset_atIndex(
                        constants.slab(),
                        usize::try_from(cell).expect("a cell inside the scalar arena"),
                        at,
                    );
                }
                return Ok(Bound::Scalar {
                    at: cell,
                    width: bytes.len() as u8,
                });
            }
        };
        if handle == NIL {
            let constants = self.constants.borrow();
            unsafe { command.setKernelBuffer_offset_atIndex(constants.slab(), 0, at) };
            return Ok(Bound::Absent);
        }
        let binding = self.handles.get(handle).ok_or_else(|| Fault::Unbound {
            what: format!(
                "handle {handle} at argument {at} of {}, which this fire minted no row for",
                fire.entrypoint
            ),
        })?;
        let slab = binding.slab().clone();
        let offset = binding.offset();
        drop(binding);
        self.slabs.borrow_mut().insert(slab_id(&slab), slab.clone());
        unsafe {
            command.setKernelBuffer_offset_atIndex(
                &slab,
                usize::try_from(offset).expect("an offset inside a reservation"),
                at,
            );
        }
        Ok(Bound::Buf { slab, offset })
    }
}

/// One recorded slot, back as the `Fire` its pipeline is keyed on.
fn fire_of(slot: &Slot) -> Fire {
    Fire {
        file: slot.point.file,
        entrypoint: slot.point.entrypoint,
        stamp: slot.point.stamp,
        lanes: slot.lanes,
        group: slot.group,
    }
}

fn point_of(fire: Fire) -> crate::record::Point {
    crate::record::Point {
        file: fire.file,
        entrypoint: fire.entrypoint,
        stamp: fire.stamp,
    }
}

/// The threadgroup a recorded slot dispatches at — its own, or the one the
/// pipeline's occupancy answers.
fn grid_of(device: &Context, pipelines: &Pipelines, slot: &Slot) -> Result<MTLSize> {
    if slot.group == [0, 0, 0] {
        let pipeline: Pipeline = pipelines.at(device.device(), fire_of(slot))?;
        Ok(crate::device::ctx::threadgroup(&pipeline, slot.lanes))
    } else {
        Ok(size(slot.group))
    }
}

fn size(axes: [u32; 3]) -> MTLSize {
    MTLSize {
        width: axes[0].max(1) as usize,
        height: axes[1].max(1) as usize,
        depth: axes[2].max(1) as usize,
    }
}

/// A marshalled scalar's bytes, as `setBytes:length:` would have copied them.
fn scalar_bytes_of(arg: ArgValue) -> Vec<u8> {
    match arg {
        ArgValue::I32(v) => v.to_ne_bytes().to_vec(),
        ArgValue::U32(v) => v.to_ne_bytes().to_vec(),
        ArgValue::F32(v) => v.to_ne_bytes().to_vec(),
        ArgValue::Usize(v) => v.to_ne_bytes().to_vec(),
        ArgValue::Buffer(_) | ArgValue::BufferMut(_) => Vec::new(),
    }
}

/// The same, from a recorded argument.
fn scalar_bytes(arg: Arg) -> Option<Vec<u8>> {
    match arg {
        Arg::I32(v) => Some(v.to_ne_bytes().to_vec()),
        Arg::U32(v) => Some(v.to_ne_bytes().to_vec()),
        Arg::F32(bits) => Some(bits.to_ne_bytes().to_vec()),
        Arg::Usize(v) => Some(v.to_ne_bytes().to_vec()),
        Arg::Buffer { .. } | Arg::Absent => None,
    }
}

/// How many bytes of scalar arena a recording needs.
///
/// One eight-byte cell per scalar argument, which is the widest seat
/// (`size_t` in MSL) and keeps every cell legally aligned for every other.
#[must_use]
pub fn constants_for(taped: &Recording) -> u64 {
    let cells: usize = taped
        .slots
        .iter()
        .map(|slot| {
            slot.args
                .iter()
                .filter(|arg| scalar_bytes(**arg).is_some())
                .count()
        })
        .sum();
    cells as u64 * 8 + 16
}
