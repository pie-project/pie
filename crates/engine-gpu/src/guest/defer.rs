//! A guest pass recorded behind its step's forward and landed later, as
//! every backend runs one: the cells a channel keeps on the device between
//! fires, the ring cells a pass on the queue owes the host, how a landed
//! pass's cells reach their rings, and a step's passes in one frame with one
//! submit. What is a backend's own (its frame, its berths, how a landed pass
//! is read) it supplies: [`LandedPass`], and the recording each pass makes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use engine::channel::ChannelId;
use engine::program::{Footprint, InstanceId, ProgramId, RingFacts};
use eta_compiler::codegen::launch::LaunchPackage;
use eta_exec::{ChannelState, ExecPlan, InterpInstance, StepOutcome, Value};
use eta_ir::container::{ChanDType, HostRole};
use eta_ir::registry::{GeometryClass, Port, PortMask};

use crate::backend::{Backend, Frame as _};
use crate::error::{Fault, Result};
use crate::guest::held::{self, Held};
use crate::program::{Refusal, numel_of};
use crate::serve::DeviceCell;
use crate::store::Pools;
use crate::store::layout::CHANNEL;

/// Ring cells a pass on the queue has yet to land, as (channel, sequence):
/// the ring holds a cell for each from the pass's submit, and its value only
/// once the pass's frame lands. Each landing wakes whoever waits on one.
#[derive(Debug, Default)]
pub struct Owed {
    cells: Mutex<Vec<(u32, u64)>>,
    landed: Condvar,
}

impl Owed {
    pub fn owe(&self, cells: &[(u32, u64)]) {
        self.cells
            .lock()
            .expect("landing lock poisoned")
            .extend(cells.iter().copied());
    }

    #[must_use]
    pub fn owes(&self, cell: (u32, u64)) -> bool {
        self.cells
            .lock()
            .expect("landing lock poisoned")
            .contains(&cell)
    }

    /// The pass that owed `cells` landed them (or faulted).
    pub fn land(&self, cells: &[(u32, u64)]) {
        self.cells
            .lock()
            .expect("landing lock poisoned")
            .retain(|cell| !cells.contains(cell));
        self.landed.notify_all();
    }

    /// Waits, woken by each landing, until `cell` holds its value or the
    /// instance faulted.
    pub fn wait(&self, cell: (u32, u64), faulted: &AtomicBool) {
        let mut cells = self.cells.lock().expect("landing lock poisoned");
        while cells.contains(&cell) && !faulted.load(Ordering::Acquire) {
            cells = self.landed.wait(cells).expect("landing lock poisoned");
        }
    }
}

/// The count `ring`, channel `channel` of `instance`'s `fold` port, holds at
/// the fire. The next fire's count is what this one's pass puts, so a fire
/// staged while that pass is on the queue finds its cell in the ring before
/// the value is: it waits for the landing, which keeps the fold exact at the
/// cost of running such fires one after another. A fire whose count is
/// already in the ring reads it at once.
pub fn fold_count(
    instance: InstanceId,
    channel: u32,
    ring: &ChannelState,
    owed: &Owed,
    faulted: &AtomicBool,
) -> std::result::Result<Option<u32>, Refusal> {
    owed.wait((channel, ring.head()), faulted);
    if faulted.load(Ordering::Acquire) {
        return Err(Refusal::Program(format!("instance {instance} is poisoned")));
    }
    if ring.is_empty() {
        return Err(Refusal::Program(format!(
            "instance {instance}: the `fold` ring is empty at the fire"
        )));
    }
    let len = match ring.front() {
        Value::U32(cells) => cells.first().copied(),
        Value::I32(cells) => cells.first().map(|&n| n.max(0) as u32),
        other => {
            return Err(Refusal::Program(format!(
                "instance {instance}: `fold` holds {:?}, not a count",
                other.dtype()
            )));
        }
    };
    len.map(Some)
        .ok_or_else(|| Refusal::Program(format!("instance {instance}: `fold` holds an empty cell")))
}

/// The parents `ring`, channel `channel` of `instance`'s `parents` port,
/// holds at the fire: each beam's parent (`crate::beam`). As with
/// [`fold_count`], the next fire's parents are what this one's pass puts, so
/// a fire staged while that pass is on the queue finds its cell in the ring
/// before the value (a placeholder of zeros, which would continue every beam
/// from beam 0): it waits for the landing. A fire whose parents are already
/// in the ring reads them at once.
pub fn parents(
    instance: InstanceId,
    channel: u32,
    ring: &ChannelState,
    owed: &Owed,
    faulted: &AtomicBool,
) -> std::result::Result<Vec<u32>, Refusal> {
    owed.wait((channel, ring.head()), faulted);
    if faulted.load(Ordering::Acquire) {
        return Err(Refusal::Program(format!("instance {instance} is poisoned")));
    }
    if ring.is_empty() {
        return Err(Refusal::Program(format!(
            "instance {instance}: the `parents` ring is empty at the fire"
        )));
    }
    match ring.front() {
        Value::U32(cells) => Ok(cells),
        Value::I32(cells) => Ok(cells.iter().map(|&p| p as u32).collect()),
        other => Err(Refusal::Program(format!(
            "instance {instance}: `parents` holds {:?}, not beams",
            other.dtype()
        ))),
    }
}

/// The cells of channels the device keeps between fires, by channel: one
/// set for the engine, so a pass reads what another program's pass kept.
/// A channel's cell lies in the `Channel` slots the runtime took for it
/// when it was registered (or its instance bound), whether or not a pass
/// keeps a cell in it now.
pub type Kept<B> = Arc<Mutex<HashMap<ChannelId, Cell<B>>>>;

/// Where a channel keeps a cell on the device.
#[derive(Debug)]
pub struct Cell<B: Backend> {
    place: CellPlace<B>,
    bytes: u64,
    /// A pass kept a cell here the host has not superseded.
    live: bool,
}

#[derive(Debug)]
enum CellPlace<B: Backend> {
    /// In the channel's `Channel` slots (one run where the cell is past a
    /// slot).
    Pooled(Held<B>),
    /// A cell past a chunk, which the contract states no slots for: a
    /// buffer of the engine's own.
    Own(B::Buffer),
}

impl<B: Backend> Cell<B> {
    /// A cell of `bytes` for channel `id` in `slots`: a run of `Channel`
    /// slots, or, past a chunk, a buffer of the engine's own on `device`.
    pub fn of(
        pools: &Pools<B>,
        device: &B::Context,
        id: ChannelId,
        bytes: u64,
        slots: &[u32],
    ) -> std::result::Result<Cell<B>, Refusal> {
        let program = Refusal::fault;
        let slot_bytes = pools.slot_bytes(CHANNEL).map_err(program)?;
        let per_chunk = pools.slots_per_chunk(CHANNEL).map_err(program)?;
        match held::item_slots(bytes, slot_bytes, per_chunk) {
            Some((need, _)) if slots.len() == need as usize => {
                Cell::pooled(pools, slots, bytes).map_err(program)
            }
            Some((need, _)) => Err(Refusal::Program(format!(
                "channel {id}'s device cell of {bytes} bytes takes {need} channel slot(s) and \
                 the binding names {slots:?}"
            ))),
            None if slots.is_empty() => Cell::own(device, bytes).map_err(program),
            None => Err(Refusal::Program(format!(
                "channel {id}'s device cell of {bytes} bytes is past a chunk, the engine's to \
                 keep, and the binding names {} channel slot(s) for it",
                slots.len()
            ))),
        }
    }

    /// A cell of `bytes` in `slots` (`Channel` slot refs, one run where
    /// `bytes` is past a slot).
    pub fn pooled(pools: &Pools<B>, slots: &[u32], bytes: u64) -> Result<Cell<B>> {
        let count = u32::try_from(slots.len()).unwrap_or(u32::MAX);
        let first = *slots.first().ok_or_else(|| Fault::Program {
            at: "guest::defer",
            why: "a channel cell placed in no Channel slots".into(),
        })?;
        if !held::consecutive(slots) {
            return Err(Fault::Program {
                at: "guest::defer",
                why: format!("a channel cell of {bytes} bytes in slots {slots:?}, not one run"),
            });
        }
        Ok(Cell {
            place: CellPlace::Pooled(Held::in_run(pools, CHANNEL, first, count, 0, bytes)?),
            bytes,
            live: false,
        })
    }

    /// A cell past a chunk, in a buffer of its own.
    pub fn own(device: &B::Context, bytes: u64) -> Result<Cell<B>> {
        use crate::backend::Buffer as _;
        Ok(Cell {
            place: CellPlace::Own(B::Buffer::zeroed(
                device,
                bytes,
                crate::books::owners::GUEST_STATE,
            )?),
            bytes,
            live: false,
        })
    }

    fn span(&self) -> (&B::Buffer, u64) {
        match &self.place {
            CellPlace::Pooled(held) => (held.buffer(), held.offset()),
            CellPlace::Own(buffer) => (buffer, 0),
        }
    }

    /// The cell as the forward reads it.
    #[must_use]
    pub fn device_cell(&self) -> DeviceCell<B> {
        let (buffer, at) = self.span();
        DeviceCell {
            buffer: buffer.clone(),
            at,
            bytes: self.bytes,
        }
    }

    /// Where the cell lies under `pools`' chunk table, an evacuation's not
    /// yet committed; `None` for a cell of its own buffer, which no
    /// evacuation moves. Nothing changed.
    pub fn resolve(&self, pools: &Pools<B>) -> Result<Option<Held<B>>> {
        match &self.place {
            CellPlace::Pooled(held) => {
                let mut held = held.clone();
                held.rebind(pools)?;
                Ok(Some(held))
            }
            CellPlace::Own(_) => Ok(None),
        }
    }

    /// Puts a pooled cell where [`Self::resolve`] resolved it.
    pub fn install(&mut self, held: Held<B>) {
        if let CellPlace::Pooled(place) = &mut self.place {
            *place = held;
        }
    }

    pub fn retarget(&mut self, pools: &Pools<B>, slots: &[u32]) -> Result<()> {
        let need = held::item_slots(
            self.bytes,
            pools.slot_bytes(CHANNEL)?,
            pools.slots_per_chunk(CHANNEL)?,
        )
        .map_or(0, |(slots, _)| slots);
        match &mut self.place {
            CellPlace::Pooled(held) if slots.len() == need as usize && held::consecutive(slots) => {
                held.retarget(pools, slots[0])
            }
            _ => Err(Fault::Program {
                at: "guest::defer",
                why: format!(
                    "a {}-byte channel cell rebound to {} slot(s), not the {need} it takes in \
                     one run",
                    self.bytes,
                    slots.len()
                ),
            }),
        }
    }
}

/// An instance's view of the kept cells: its channels by their program's
/// index, over the engine's one set.
#[derive(Debug)]
pub struct Cells<B: Backend> {
    channels: Vec<ChannelId>,
    kept: Kept<B>,
}

impl<B: Backend> Cells<B> {
    #[must_use]
    pub fn new(channels: Vec<ChannelId>, kept: Kept<B>) -> Cells<B> {
        Cells { channels, kept }
    }

    /// The engine's id of the program's `channel`.
    fn channel(&self, channel: u32) -> Result<ChannelId> {
        self.channels
            .get(channel as usize)
            .copied()
            .ok_or_else(|| Fault::Program {
                at: "guest::defer",
                why: format!(
                    "channel {channel} is past the {} this instance binds",
                    self.channels.len()
                ),
            })
    }

    /// Copies `len` bytes at `at` of `heap` into the device's cell for
    /// `channel`, in place of a read back: the next fire's take of the
    /// channel restores it.
    pub fn keep(
        &self,
        frame: &mut B::Frame,
        heap: &B::Buffer,
        at: u64,
        len: u64,
        channel: u32,
    ) -> Result<()> {
        let id = self.channel(channel)?;
        let mut kept = self.kept.lock().expect("kept cells lock poisoned");
        let cell = kept.get_mut(&id).ok_or_else(|| Fault::Program {
            at: "guest::defer",
            why: format!(
                "channel {id} keeps a cell on the device and holds no place for one: its \
                 registration took no Channel slots"
            ),
        })?;
        if cell.bytes < len {
            return Err(Fault::Program {
                at: "guest::defer",
                why: format!(
                    "channel {id} keeps a {len}-byte value in a {}-byte cell",
                    cell.bytes
                ),
            });
        }
        let (into, into_at) = cell.span();
        frame.copy(heap, at, into, into_at, len)?;
        cell.live = true;
        Ok(())
    }

    /// The device cell a kept channel holds.
    #[must_use]
    pub fn kept_cell(&self, channel: u32) -> Option<DeviceCell<B>> {
        let id = self.channels.get(channel as usize)?;
        self.kept
            .lock()
            .expect("kept cells lock poisoned")
            .get(id)
            .filter(|cell| cell.live)
            .map(Cell::device_cell)
    }

    #[must_use]
    pub fn keeps(&self, channel: u32) -> bool {
        self.kept_cell(channel).is_some()
    }

    /// Drops the device's cell for `channel`, so the next take reads the
    /// ring, which the host has written.
    pub fn forget(&self, channel: u32) {
        if let Some(id) = self.channels.get(channel as usize)
            && let Some(cell) = self
                .kept
                .lock()
                .expect("kept cells lock poisoned")
                .get_mut(id)
        {
            cell.live = false;
        }
    }

    /// Copies `channel`'s kept cell into `len` bytes at `at` of `heap`;
    /// false when the device keeps no cell for it.
    pub fn restore(
        &self,
        frame: &mut B::Frame,
        heap: &B::Buffer,
        at: u64,
        len: u64,
        channel: u32,
    ) -> Result<bool> {
        let Some(cell) = self.kept_cell(channel) else {
            return Ok(false);
        };
        frame.copy(&cell.buffer, cell.at, heap, at, len.min(cell.bytes))?;
        Ok(true)
    }
}

/// Whether any stage of `plan` puts into `channel`.
#[must_use]
pub fn puts_into(plan: &ExecPlan, channel: u32) -> bool {
    plan.package
        .stages
        .iter()
        .any(|stage| stage.puts.iter().any(|put| put.channel == channel))
}

/// The bytes a channel's cell takes on the device, where a pass may keep one
/// there: a channel the host does not read. A value lies in a guest heap at
/// its dtype's width an element (`eta_exec::element_bytes`, as
/// `ValueDesc::device_bytes` lays it).
#[must_use]
pub fn device_cell_bytes(shape: &[u32], dtype: ChanDType, host_role: HostRole) -> Option<u64> {
    if host_role == HostRole::Reader {
        return None;
    }
    let numel = numel_of(shape) as u64;
    let element = eta_exec::element_bytes(eta_exec::concrete_dtype(dtype));
    Some(numel.saturating_mul(element).max(4))
}

/// A channel's device cell as `Channel` slots: a slot where the cell fits
/// one, else a run of adjacent slots (`Footprint::contiguous`); none where
/// the host reads the channel, or the cell is past a chunk (the engine
/// keeps that one).
pub fn cell_facts<B: Backend>(
    channel: u32,
    shape: &[u32],
    dtype: ChanDType,
    capacity: u32,
    host_role: HostRole,
    pools: &Pools<B>,
) -> std::result::Result<Option<RingFacts>, Refusal> {
    let Some(bytes) = device_cell_bytes(shape, dtype, host_role) else {
        return Ok(None);
    };
    let program = Refusal::fault;
    let slot_bytes = pools.slot_bytes(CHANNEL).map_err(program)?;
    let per_chunk = pools.slots_per_chunk(CHANNEL).map_err(program)?;
    Ok(
        held::item_slots(bytes, slot_bytes, per_chunk).map(|(slots, contiguous)| RingFacts {
            channel,
            cell_bytes: eta_exec::wire_cell_bytes(eta_exec::concrete_dtype(dtype), numel_of(shape))
                as u64,
            capacity,
            footprint: Footprint {
                bytes,
                slots,
                contiguous,
            },
        }),
    )
}

/// Cells at least this large stay on the device between fires when only the
/// device reads them: a sampler's token histograms, a vocabulary wide each.
pub const KEEP_BYTES: u64 = 64 << 10;

/// Whether a pass run to completion keeps `channel`'s put of `bytes` on the
/// device: a cell only the device reads, one at a time, large enough that
/// reading it back each fire costs.
#[must_use]
pub fn keeps(package: &LaunchPackage, channel: u32, bytes: u64) -> bool {
    package.channels.get(channel as usize).is_some_and(|decl| {
        decl.capacity == 1 && decl.host_role != HostRole::Reader && bytes >= KEEP_BYTES
    })
}

/// Whether a deferred pass keeps `channel`'s puts on the device: every one
/// the host does not read, at the fire (a port of `reads`: a `fold` count,
/// an `attn_mask`, a device-geometry descriptor) or after.
#[must_use]
pub fn host_reads_not(package: &LaunchPackage, channel: u32, reads: PortMask) -> bool {
    let read_at_fire = package.ports.iter().any(|binding| {
        binding.channel == channel && !binding.is_const && reads.contains(binding.port)
    });
    !read_at_fire
        && package
            .channels
            .get(channel as usize)
            .is_some_and(|decl| decl.host_role != HostRole::Reader)
}

/// The device cell holding lane `lane_at`'s one token, where the device
/// keeps the instance's `embed_tokens` cell: what the next forward copies
/// its token from, queued behind the pass that kept it.
pub fn token_feed<B: Backend>(
    plan: &ExecPlan,
    inst: &InterpInstance,
    geometry: GeometryClass,
    cells: &Cells<B>,
    lane_at: usize,
) -> Option<DeviceCell<B>> {
    if !geometry.ports().contains(Port::EmbedTokens) {
        return None;
    }
    let binding = plan
        .package
        .ports
        .iter()
        .find(|binding| binding.port == Port::EmbedTokens && !binding.is_const)?;
    let mut cell = cells.kept_cell(binding.channel)?;
    if lane_at > 0 {
        let indptr = crate::program::embed_indptr(plan, inst);
        let start = u64::from(*indptr.get(lane_at)?);
        let end = u64::from(*indptr.get(lane_at + 1)?);
        if end != start + 1 || end * 4 > cell.bytes {
            return None;
        }
        cell.at += start * 4;
        cell.bytes = 4;
    }
    Some(cell)
}

/// A put the host reads, landing in its pass's berth.
#[derive(Clone, Copy, Debug)]
pub struct Landing {
    pub channel: u32,
    /// Its value in the stage.
    pub value: u32,
    pub dtype: eta_ir::Dtype,
    pub held: u64,
}

/// The passes of one frame, landed: `Some(why)` where they cannot be read.
pub type Then<P> = Box<dyn FnOnce(Vec<Recorded<P>>, Option<String>) + Send>;

/// A pass that landed, as its backend reads it.
pub trait LandedPass: Send + Sized + 'static {
    /// The fault code the pass's stage wrote, if any.
    fn status(&self) -> Result<Option<u32>>;

    /// The value the pass read back for `landing`.
    fn value(&self, landing: &Landing) -> Result<Value>;

    /// Makes what a landed frame's `passes` read back readable, then hands
    /// them to `then`: at once where the backend reads its berths in place,
    /// once mapped where it maps them first.
    fn readable(passes: Vec<Recorded<Self>>, then: Then<Self>) {
        then(passes, None);
    }
}

/// One attachment's pass the step's deferred frame carries: its instance,
/// its lanes, and what its fire read out for each of them.
pub struct DeferredPass<S> {
    pub instance: InstanceId,
    pub lanes: std::ops::Range<u32>,
    pub seats: S,
}

/// A pass recorded into the step's deferred frame: its berth, the ring cells
/// it owes the host, and where its instance learns it landed.
pub struct Recorded<P> {
    instance: InstanceId,
    program: ProgramId,
    lanes: std::ops::Range<u32>,
    pass: P,
    fixups: Vec<(Arc<ChannelState>, u64, Landing)>,
    owed: Vec<(u32, u64)>,
    owing: Arc<Owed>,
    faulted: Arc<AtomicBool>,
}

impl<P> Recorded<P> {
    /// The backend's pass.
    #[must_use]
    pub fn pass(&self) -> &P {
        &self.pass
    }
}

impl<P: LandedPass> Recorded<P> {
    /// An instance's pass, its cells owed from now until it lands: `rings`
    /// are the instance's channels, `owing` and `faulted` where it waits on
    /// its landings and learns of a fault.
    #[allow(clippy::too_many_arguments)]
    pub fn of(
        instance: InstanceId,
        program: ProgramId,
        lanes: std::ops::Range<u32>,
        pass: P,
        landings: Vec<Landing>,
        rings: &[Arc<ChannelState>],
        owing: &Arc<Owed>,
        faulted: &Arc<AtomicBool>,
    ) -> Recorded<P> {
        let fixups: Vec<(Arc<ChannelState>, u64, Landing)> = landings
            .into_iter()
            .map(|landing| {
                let ring = Arc::clone(&rings[landing.channel as usize]);
                let sequence = ring.tail().saturating_sub(1);
                (ring, sequence, landing)
            })
            .collect();
        let owed: Vec<(u32, u64)> = fixups
            .iter()
            .map(|(_, sequence, landing)| (landing.channel, *sequence))
            .collect();
        owing.owe(&owed);
        Recorded {
            instance,
            program,
            lanes,
            pass,
            fixups,
            owed,
            owing: Arc::clone(owing),
            faulted: Arc::clone(faulted),
        }
    }

    /// The pass never reaches the queue: what waits on its cells finds the
    /// instance poisoned rather than waiting on.
    pub fn never(self) {
        self.faulted.store(true, Ordering::Release);
        self.owing.land(&self.owed);
    }

    /// The frame landed (or the queue refused it, `refused`): the host's
    /// cells into the rings, the pass's berth given back as it drops, and
    /// why its lanes fault, if they do.
    pub fn land(self, refused: Option<String>) -> Option<String> {
        let Recorded {
            instance,
            program,
            pass,
            fixups,
            owed,
            owing,
            faulted,
            ..
        } = self;
        let mut fault = refused.or_else(|| match pass.status() {
            Ok(None) => None,
            Ok(Some(code)) => Some(format!(
                "instance {instance} (program {program}) faulted on the device with code {code}"
            )),
            Err(why) => Some(why.to_string()),
        });
        if fault.is_none() {
            for (ring, sequence, landing) in &fixups {
                match pass.value(landing) {
                    Ok(value) => ring.encode_sequence(*sequence, &value),
                    Err(why) => {
                        fault = Some(why.to_string());
                        break;
                    }
                }
            }
        }
        if fault.is_some() {
            faulted.store(true, Ordering::Release);
        }
        owing.land(&owed);
        drop(pass);
        fault
    }
}

/// What recording one pass into the step's frame came to: the frame to hand
/// the next pass (`None` where the pass faulted with it in hand), and the
/// pass recorded, or why it was not.
pub struct Recording<F, P> {
    pub frame: Option<F>,
    pub pass: std::result::Result<Option<Recorded<P>>, String>,
}

/// The lanes a pass's host fault faults, its instance, and why.
type HostFault = (std::ops::Range<u32>, InstanceId, String);

/// Runs a step's passes behind its forward on the queue, all in one frame
/// with one submit, and returns before any lands: `record` records each into
/// the frame the passes before it recorded into (or a frame of its own, for
/// the first), and `depart` hands each recorded pass's instance the
/// submission. The frame's landing writes each pass's host cells into its
/// rings and reports the step to `done` once, faulting the lanes of each
/// pass that faulted.
///
/// A pass that faults on the host faults its own lanes. Where it faulted
/// with the frame in hand, the frame is lost with it, and so are the passes
/// recorded into it before: their lanes fault, and their instances are
/// poisoned, since what they keep on the device was never written.
pub fn fire_deferred<B, P, S>(
    passes: Vec<DeferredPass<S>>,
    mut record: impl FnMut(DeferredPass<S>, Option<B::Frame>) -> Recording<B::Frame, P>,
    done: Option<crate::settle::Done>,
    mut depart: impl FnMut(InstanceId, B::Pending),
) -> std::result::Result<(), Refusal>
where
    B: Backend,
    B::Pending: Clone,
    P: LandedPass,
{
    let mut frame: Option<B::Frame> = None;
    let mut recorded: Vec<Recorded<P>> = Vec::new();
    let mut faults: Vec<HostFault> = Vec::new();
    for pass in passes {
        let (instance, lanes) = (pass.instance, pass.lanes.clone());
        let carried = frame.is_some();
        let Recording { frame: left, pass } = record(pass, frame.take());
        let lost = carried && left.is_none();
        frame = left;
        match pass {
            Ok(Some(pass)) => recorded.push(pass),
            Ok(None) => {}
            Err(why) => {
                tracing::warn!(instance = instance.0, %why, "a deferred guest epilogue faulted its lanes");
                faults.push((lanes, instance, why));
                if lost {
                    for pass in recorded.drain(..) {
                        let why = format!(
                            "instance {} (program {}): the frame its pass was recorded into was \
                             lost when a later pass of the step faulted",
                            pass.instance, pass.program
                        );
                        faults.push((pass.lanes.clone(), pass.instance, why));
                        pass.never();
                    }
                }
            }
        }
    }
    let Some(frame) = frame else {
        if let Some(done) = done {
            tell_settled(done, faults);
        }
        return Ok(());
    };
    let instances: Vec<InstanceId> = recorded.iter().map(|pass| pass.instance).collect();
    let held = Arc::new(Mutex::new(recorded));
    let landing = Arc::clone(&held);
    let on_done = move |refused: Option<String>| {
        let passes: Vec<Recorded<P>> = landing
            .lock()
            .expect("passes lock poisoned")
            .drain(..)
            .collect();
        // The queue refusing the buffer the step's forward and these passes
        // rode is the step lost whole; what each pass's own status and
        // landing say is its lanes'.
        let lost = refused.clone();
        let refused_now = lost.is_some();
        let report = move |passes: Vec<Recorded<P>>, unread: Option<String>| {
            let mut faults = faults;
            let why = refused.or(unread);
            for pass in passes {
                let (lanes, instance) = (pass.lanes.clone(), pass.instance);
                if let Some(why) = pass.land(why.clone()) {
                    faults.push((lanes, instance, why));
                }
            }
            if let Some(done) = done {
                if lost.is_some() {
                    done.tell(lost, "guest pass");
                    return;
                }
                // This runs where the frame's readback landed (the thread
                // that polls the device): the forward's fed ids, read by its
                // own landing on that thread, are joined, never waited on.
                tell_settled(done, faults);
            }
        };
        if refused_now {
            report(passes, None);
        } else {
            P::readable(passes, Box::new(report));
        }
    };
    let pending = match frame.commit_async(Some(Box::new(on_done))) {
        Ok(pending) => pending,
        Err(fault) => {
            // A frame that never went to the queue lands nothing: what waits
            // on its cells finds the instances poisoned rather than waiting
            // on.
            for pass in held.lock().expect("passes lock poisoned").drain(..) {
                pass.never();
            }
            return Err(Refusal::Program(format!(
                "a deferred guest frame of {} pass(es): {fault}",
                instances.len()
            )));
        }
    };
    for instance in instances {
        depart(instance, pending.clone());
    }
    Ok(())
}

/// Tells `done`'s sink the step once its forward's fed ids are checked, the
/// lanes `faults` names faulted beside what the forward says.
fn tell_settled(done: crate::settle::Done, faults: Vec<HostFault>) {
    done.when_settled(move |done, mut outcome| {
        for (lanes, instance, why) in &faults {
            outcome.fault_lanes(lanes.clone(), Some(*instance), why);
        }
        (done.sink)(done.at, outcome);
    });
}

/// What a deferred step's outcome says of one pass's run on the host: the
/// pass recorded, or why its lanes fault.
pub fn outcome_of<P>(
    outcome: StepOutcome,
    instance: InstanceId,
    program: ProgramId,
    recorded: impl FnOnce() -> Option<Recorded<P>>,
) -> std::result::Result<Option<Recorded<P>>, String> {
    match outcome {
        StepOutcome::Committed => Ok(recorded()),
        StepOutcome::Blocked(channel) => Err(format!(
            "instance {instance} (program {program}) blocked on channel {channel}"
        )),
        StepOutcome::Faulted(why) => Err(format!(
            "instance {instance} (program {program}) faulted: {why}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{ChannelState, InstanceId, Owed, Value, fold_count};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Two fires in flight, each folding by the count the pass before it
    /// puts: the pass's cell is in the `fold` ring from its submit, its
    /// value only once its frame lands. A fire staged behind a pass on the
    /// queue reads the count that pass lands, not the cell's placeholder,
    /// and a fire whose count is in the ring reads it at once.
    #[test]
    fn a_fire_staged_behind_a_pass_in_flight_reads_the_count_it_lands() {
        const FOLD: u32 = 3;
        let instance = InstanceId(1);
        let ring = Arc::new(ChannelState::host(eta_ir::Dtype::U32, 1, 4));
        let owed = Arc::new(Owed::default());
        let faulted = Arc::new(AtomicBool::new(false));

        // The seeded count: the first fire reads it at once.
        assert!(ring.push(&Value::U32(vec![4])));
        assert_eq!(
            fold_count(instance, FOLD, &ring, &owed, &faulted).unwrap(),
            Some(4)
        );
        assert!(ring.pop().is_some());
        // Its pass and the next fire's go to the queue: each puts a cell
        // whose value lands later.
        for _ in 0..2 {
            let sequence = ring.tail();
            assert!(ring.push(&Value::U32(vec![0])));
            owed.owe(&[(FOLD, sequence)]);
        }
        let lands = {
            let (ring, owed) = (Arc::clone(&ring), Arc::clone(&owed));
            std::thread::spawn(move || {
                for (sequence, count) in [(1, 4), (2, 4)] {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    ring.encode_sequence(sequence, &Value::U32(vec![count]));
                    owed.land(&[(FOLD, sequence)]);
                }
            })
        };
        // The second fire, staged while the first's pass is on the queue.
        assert_eq!(
            fold_count(instance, FOLD, &ring, &owed, &faulted).unwrap(),
            Some(4)
        );
        assert!(ring.pop().is_some());
        // The third, staged while the second's is.
        assert_eq!(
            fold_count(instance, FOLD, &ring, &owed, &faulted).unwrap(),
            Some(4)
        );
        lands.join().unwrap();
    }

    /// A pass that faults lands nothing; the fire waiting on its cell is
    /// refused, not left waiting.
    #[test]
    fn a_fire_behind_a_pass_that_faulted_is_refused() {
        let ring = ChannelState::host(eta_ir::Dtype::U32, 1, 4);
        let owed = Arc::new(Owed::default());
        let faulted = Arc::new(AtomicBool::new(false));
        assert!(ring.push(&Value::U32(vec![0])));
        owed.owe(&[(3, 0)]);
        let faults = {
            let (owed, faulted) = (Arc::clone(&owed), Arc::clone(&faulted));
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(20));
                faulted.store(true, Ordering::Release);
                owed.land(&[(3, 0)]);
            })
        };
        let refused = fold_count(InstanceId(1), 3, &ring, &owed, &faulted);
        assert!(
            matches!(refused, Err(ref why) if why.to_string().contains("poisoned")),
            "{refused:?}"
        );
        faults.join().unwrap();
    }
}
