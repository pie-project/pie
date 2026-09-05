//! The runtime's door: one loaded model, its boot, and one fire in call order.
//!
//! # The words this tree uses, in the conventional ones
//!
//! * [`Shell`] — a loaded model (the engine instance); [`Boot`] — its
//!   config; [`Knobs`] — its flags.
//! * [`Lane`]/[`Seated`] — one sequence in a batch; `slot` — the sequence id
//!   owning a kv block table; `held` — the tokens that slot already holds in
//!   the kv cache; [`Shell::open`] — reset a sequence.
//! * `bucket` — a row-count lattice point a fire's rows round up to, one
//!   captured graph per entry (the cudagraph capture size); `rung`/`ladder` —
//!   the per-class row ceiling inside a body key (`record::Ladder`).
//! * `body` — the captured CUDA graph for one composition; `arming` — the
//!   load-time warm-up and capture pass; [`Golden`] — its replay-vs-eager
//!   parity check.
//! * [`Fault`] — this crate's error.

mod arming;
mod boot;
mod enqueue;
mod lanes;
mod load;
mod prepare;
mod segments;
mod settle;
mod stats;

pub use arming::{Armed, Kind, Seal};
pub use boot::{
    Boot, DEFAULT_BODIES_MEGABYTES, DEFAULT_GPU_MEM_UTILIZATION, Golden, Graphs, Knobs, Recording,
};
pub use lanes::{Attached, Lane, Media, Seated};
pub(crate) use lanes::{MROPE_COORDS, PATCH_ROUTE_DROP};
pub(crate) use settle::Readback;
pub use settle::{Done, Settled};

use engine::fire::LayerScores;
use engine::frame::{
    Demand, Enqueued as EnqueuedPhase, Prepared as PreparedPhase, Shell as FrameShell, Supply,
};
use model_compiler::{Budget, Budgets, CompiledModel};
use model_exec::fire::{Composition, FireDescriptor};
use model_ir::Trace;

use crate::arena::Arena;
use crate::device::Context;
use crate::error::{Fault, Result};
use crate::exports::Exports;
use crate::inputs::Inputs;
use crate::program::{Fired, Plane as ProgramPlane, Session as ProgramSession};
use crate::record::{self, Bodies as GraphCache};
use crate::run::RsMove;
use crate::store::Pools;
use crate::store::kv::{self, Seat};
use crate::store::rs::Buffers;
use crate::weights::{AdapterPlane, Weights};
use crate::window::Windows;
use enqueue::{GuestBatch, reap_guest_fires};
use segments::Segmented;

/// What one fire's window table cost, in launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FireCost {
    /// Every region's launches, summed.
    pub launches: u32,
    /// How many regions were gathered into one launch instead of split.
    pub copied: u32,
}

/// One loaded model, serving.
pub struct Shell {
    device: Context,
    /// The accounting sentence this load was admitted under.
    accounting: crate::store::Accounting,
    trace: Trace,
    compiled: CompiledModel,
    budget: Budget,
    /// Bytes per patch row, or `None` for a plan with no patch row.
    patch_seat: Option<crate::inputs::PatchSeat>,
    /// Whether the plan declares `RuntimeInput::MropePositions`.
    mrope_seat: bool,
    /// Whether the plan declares `layout.scatter_live_rows`.
    drops_patch_rows: bool,
    /// Whether the artifact states a patch axis at all.
    towered: bool,
    /// How many patch rows the plan folds into one (`1` for none).
    patch_fold: u32,
    /// Both axes' ceilings; `budgets.tokens == budget`.
    budgets: Budgets,
    weights: Weights,
    arena: Arena,
    pools: Pools,
    /// The buffered-activation pool, or `None` for a plan with nothing to buffer.
    buffers: Option<Buffers>,
    /// The fold predicate and the accepted lengths, resident at the lane ceiling.
    predicate: crate::store::rs::Predicate,
    inputs: Inputs,
    /// What the plan restates about its own caches.
    facts: kv::Facts,
    /// How many kv geometry spaces the plan declares.
    spaces: usize,
    /// The classes whose window runs an `attention.masked` arm.
    masked: model_ir::ClassSet,
    /// The classes whose window runs a `linear.lora_correct` arm.
    corrected: model_ir::ClassSet,
    /// The classes whose window runs a one-row-per-lane op.
    decoding: model_ir::ClassSet,
    /// Per class, the requests that land in it.
    landing: Vec<Vec<model_ir::Request>>,
    /// What the arming pass did, once it ran.
    armed: Option<Armed>,
    classify: model_ir::ClassifyFn,
    /// The classes whose window runs the embed merge.
    media: model_ir::ClassSet,
    /// Per template region: every op reads the staged seat's start.
    shifted: Vec<bool>,
    /// Per template region: every op finds its own lane.
    lane_shifted: Vec<bool>,
    /// Which fact bit puts a lane in the correction's window, or `None`.
    adapter_fact: Option<u32>,
    /// The shared-adapter store.
    adapters: crate::blob::Adapters,
    /// Per slot: how many kv tokens it holds.
    held: Vec<u32>,
    /// The row-pointer tables a non-consecutive readout binds through.
    readout_rows: crate::device::Buffer,
    /// This load's declared exports.
    exports: Exports,
    /// The observability slab, or `None` for a plan with no `attn.scores` export.
    scores: Option<crate::scores::Scores>,
    graphs: Graphs,
    /// Serve `Fallback::Copy` where P4's table asks for one.
    copies: bool,
    /// Arm D4's pad before each walk (`Knobs::pad`).
    pad: bool,
    /// Serve fires from a recorded body; read in `prepare`, not at the router.
    bodies: bool,
    /// Bytes of graph exec the arming pass may spend (`Knobs::bodies_mem`, converted).
    bodies_mem: usize,
    /// Is a synthetic arming pass firing?
    arming: bool,
    /// Which half of the golden pass is firing.
    golden_arm: Golden,
    /// Whether `Knobs::golden` asked for the pass at all.
    golden: bool,
    /// The body key the last arming fire composed, or `None` for a refused synthetic.
    armed_body: Option<record::BodyKey>,
    /// The segmentation of every key this load has derived one for.
    segments: std::collections::HashMap<record::BodyKey, Segmented>,
    /// What the last fire's window table cost.
    last: FireCost,
    cache: GraphCache,
    /// The guest-program plane.
    programs: ProgramPlane,
    /// One event per in-flight step.
    settlement: crate::settle::Settlement,
    /// How far ahead of the device this shell is.
    airborne: crate::settle::Airborne,
    /// The epilogue boundary's fires, enqueued and not yet reaped.
    owed: Option<GuestBatch>,
    /// The point on the compute stream `owed`'s verdicts become readable.
    guest_landed: crate::device::graph::Event,
}

impl Shell {
    /// Move one sequence's recurrent state onto another slot, synchronized.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a slot past the pool, [`Fault::Device`] for the copy.
    pub fn copy_state(&mut self, src: u32, dst: u32) -> Result<()> {
        self.pools.copy_slot(self.device.stream(), src, dst)?;
        self.device.synchronize()
    }

    /// Graft kv cells onto other pages of this load's pools, enqueued and not synchronized.
    ///
    /// # Errors
    ///
    /// As [`Pools::copy_kv`](crate::store::Pools::copy_kv).
    pub fn copy_kv(&mut self, moves: &[crate::store::Move]) -> Result<()> {
        self.pools.copy_kv(self.device.stream(), moves)
    }

    /// Bytes one recurrent slot occupies across the plan's state rows — zero for a plan with none (`PoolFacts::state_slot_bytes`).
    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.pools.state_slot_bytes()
    }

    /// One slot's recurrent banks, read back.
    ///
    /// # Errors
    ///
    /// As [`Pools::state_bytes`](crate::store::Pools::state_bytes).
    pub fn state_bytes(&mut self, slot: u32) -> Result<Vec<u8>> {
        self.pools.state_bytes(slot)
    }

    /// The fold predicate this shell's last predicated fire wrote, one byte per lane.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for the read.
    pub fn fold_predicate(&self, lanes: u32) -> Result<Vec<u8>> {
        self.predicate.read_mask(lanes)
    }

    /// Write one adapter's planes into this load's banks; never a recapture.
    ///
    /// # Errors
    ///
    /// [`Fault::Adapter`] for a bank this plan does not declare, an id past
    /// the declared capacity, or a plane that is not one slot's bytes;
    /// [`Fault::Device`] for the copy.
    pub fn register_adapter(&mut self, id: u32, planes: &[AdapterPlane<'_>]) -> Result<()> {
        self.weights.register_adapter(id, planes)
    }

    /// State where the shared adapters live; `None` is the feature off.
    pub fn mount_adapters(&mut self, root: Option<std::path::PathBuf>) {
        self.adapters.mount(root);
    }

    /// Bind one instance to one adapter; answer the slot its lanes route to.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for a mount, a manifest or a shape that disagrees with
    /// this load's banks; [`Fault::AdapterSlots`] when every slot is pinned;
    /// [`Fault::Adapter`] and [`Fault::Device`] from the landing itself.
    pub fn bind_adapter(
        &mut self,
        source: crate::blob::Source<'_>,
    ) -> Result<crate::blob::Binding> {
        let seats = self.weights.seats();
        let weights = &mut self.weights;
        self.adapters.bind(source, &seats, |slot, planes| {
            weights.register_adapter(slot, planes)
        })
    }

    /// Give a bind back; the slot keeps its contents until pressure reclaims it.
    pub fn release_adapter(&mut self, binding: crate::blob::Binding) {
        self.adapters.release(binding);
    }

    /// The banks, as the resolver reads them.
    #[must_use]
    pub fn bank_seats(&self) -> Vec<crate::weights::BankSeat> {
        self.weights.seats()
    }

    /// The `lora` sink one registered program declares, or `Ok(None)`.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a program this shell never registered, and
    /// [`Fault::Adapter`] for a sink this shell cannot serve.
    pub fn program_adapter_sink(&self, program_id: u64) -> Result<Option<crate::adapter::Sink>> {
        let program = self.programs.program(program_id).ok_or_else(|| {
            Fault::program(
                "serve::shell",
                format!("no program {program_id} to read an adapter sink off"),
            )
        })?;
        crate::adapter::sink_of(&program.plan.package)
    }

    /// `word` with the correction window's bit set, or `None` when this bake cannot carry the lane.
    #[must_use]
    pub fn adapted_word(&self, word: u64) -> Option<u64> {
        // One rule for every shell: `model_ir::ClassTable::adapted_word`.
        let bit = self.adapter_fact?;
        self.compiled.classes.adapted_word(&self.corrected, bit, word)
    }

    /// The shared-adapter store.
    #[must_use]
    pub fn adapters(&self) -> &crate::blob::Adapters {
        &self.adapters
    }

    /// Open a slot for a fresh sequence: clear its recurrent banks, zero its count.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a slot the pools do not seat.
    pub fn open(&mut self, slot: u32) -> Result<()> {
        self.pools.clear(slot)?;
        let seats = self.held.len() as u64;
        let held = self.held.get_mut(slot as usize).ok_or(Fault::Ceiling {
            what: "slots",
            need: u64::from(slot) + 1,
            have: seats,
        })?;
        *held = 0;
        Ok(())
    }

    /// Bind the calling thread to this shell's device.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] when the runtime refuses the ordinal.
    pub fn bind_thread(&self) -> Result<()> {
        self.device.bind_thread()
    }

    /// Compile and register a guest program, answering its id.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] for a package that does not adopt, [`Fault::Compile`]
    /// for a region NVRTC refuses.
    pub fn register_program(
        &mut self,
        registration: &engine::program::ProgramRegistration,
    ) -> Result<u64> {
        self.programs.register(&self.device, registration)
    }

    /// Bind an instance of `program_id`, answering its id.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown program or a seed that does not fit.
    pub fn bind_program(
        &mut self,
        program_id: u64,
        seeds: &[(u32, Vec<u8>)],
        extents: eta_exec::Extents,
        geometry: eta_ir::registry::GeometryClass,
        adopted: &[Option<std::sync::Arc<crate::program::Endpoint>>],
        ids: &[u64],
    ) -> Result<u64> {
        self.programs
            .bind(program_id, seeds, extents, geometry, adopted, ids)
    }

    /// The first of `tickets` this instance's own prediction disagrees with.
    #[must_use]
    pub fn program_ticket_disagreement(
        &self,
        instance_id: u64,
        tickets: &[engine::Ticket],
    ) -> Option<String> {
        self.programs.disagreeing_ticket(instance_id, tickets)
    }

    /// The first channel of `instance_id` a fire right now would not meet, or `None`.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance.
    pub fn program_ready(&self, instance_id: u64) -> Result<Option<u32>> {
        self.programs.ready(instance_id)
    }

    /// Collect the deferred epilogue batch.
    ///
    /// # Errors
    ///
    /// As [`reap_guest_fires`].
    pub fn reap_guests(&mut self) -> Result<()> {
        reap_guest_fires(
            &mut self.programs,
            &mut self.owed,
            &self.airborne,
            &self.guest_landed,
        )
    }

    /// One bound instance, reaped first.
    ///
    /// # Errors
    ///
    /// As [`Shell::reap_guests`].
    pub fn program_instance(&mut self, instance_id: u64) -> Result<Option<&mut ProgramSession>> {
        self.reap_guests()?;
        Ok(self.programs.instance_mut(instance_id))
    }

    /// Every bound instance's predicted channel cursors, for the ranks of a
    /// tensor-parallel group to compare.
    #[must_use]
    pub fn channel_predictions(&self) -> Vec<(u64, Vec<crate::program::Cursor>)> {
        self.programs.predictions()
    }

    /// Tear down one bound instance and free its rings, reaped first.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an instance that is already gone, and whatever the reap said.
    pub fn close_program_instance(&mut self, instance_id: u64) -> Result<()> {
        self.reap_guests()?;
        self.programs.close_instance(instance_id)
    }

    /// Fire one guest-program instance standalone.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, and whatever the launches said.
    pub fn fire_program(&mut self, instance_id: u64) -> Result<Fired> {
        self.reap_guests()?;
        self.programs.fire(&self.device, instance_id)
    }

    /// Hand back what the pools no longer need.
    pub fn trim(&mut self, hint: engine::frame::Demand) {
        Supply::trim(&mut self.pools, hint);
    }

    /// Can a `record::BodyKey` name every capture unit of this artifact?
    fn keyable_units(compiled: &CompiledModel) -> bool {
        compiled.units.len() <= 2
    }

    /// The most lanes this load can ever seat at once: `min(slots, max_lanes, max_tokens)`.
    fn lane_ceiling(&self) -> u32 {
        (self.held.len() as u32)
            .min(self.budget.max_lanes)
            .min(self.budget.max_tokens)
    }

    /// Run one fire on the shell's own paging, and hand back each lane's last row of logits.
    ///
    /// # Errors
    ///
    /// As [`Shell::fire_media`].
    pub fn fire(&mut self, lanes: &[Lane<'_>]) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        self.fire_media(&seated, &[], &[], &mut Vec::new())
    }

    /// The one fire door: seated lanes, guest attachments, images, and the
    /// capture columns read back into `scores` (one entry per submitted lane).
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] for a batch the artifact cannot describe or a dispatch
    /// the backend refused, [`Fault::Fragmented`] for a region this fire's
    /// order does not make consecutive, [`Fault::Ceiling`] for a sequence past
    /// its slot's pages, [`Fault::Device`] for a transfer, a capture or a
    /// launch, [`Fault::Program`] for an attachment that cannot fire,
    /// [`Fault::Scoreless`] / [`Fault::ScoreWord`] for a capture the artifact
    /// or the word cannot serve, and the multimodal refusals for an image the
    /// plan or the ladder cannot seat.
    pub fn fire_media(
        &mut self,
        lanes: &[Seated<'_>],
        attachments: &[Attached],
        media: &[Media<'_>],
        scores: &mut Vec<Vec<LayerScores>>,
    ) -> Result<Vec<Vec<f32>>> {
        // The three phases back to back, then the numbers door.
        let prepared = FrameShell::prepare(
            self,
            StepView {
                lanes,
                attachments,
                media,
            },
            None,
        )?;
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let mut settled = FrameShell::settle(self, enqueued)?;
        Shell::read_out(self, &mut settled)?;
        *scores = std::mem::take(&mut settled.scores);
        Ok(std::mem::take(&mut settled.logits))
    }

    /// The compute stream, for a gate that measures it.
    #[must_use]
    pub fn compute_stream(&self) -> *mut core::ffi::c_void {
        self.device.stream()
    }

    /// Wait for everything this shell has enqueued.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for whatever the stream had queued.
    pub fn drain(&mut self) -> Result<()> {
        self.device.synchronize()
    }

    /// May a fire of this load record a body: the bodies knob, the pad it
    /// requires, a recording mode, and weights that do not rotate.
    pub(crate) fn records_bodies(&self) -> bool {
        self.bodies
            && self.pad
            && self.graphs.records()
            && !self.weights.rotating()
            && !self.weights.hosts_experts()
    }

    /// What the arming pass did, or `None` when it did not run.
    #[must_use]
    pub fn armed(&self) -> Option<&Armed> {
        self.armed.as_ref()
    }
}

impl Drop for Shell {
    /// Drain before anything is freed: the staging ring's pinned buffers may still be read.
    fn drop(&mut self) {
        let _ = self.device.synchronize();
    }
}

/// One step's submission, as the shell reads it.
#[derive(Clone, Copy)]
pub struct StepView<'a> {
    /// The lanes, in submission order.
    pub lanes: &'a [Seated<'a>],
    /// The guest programs attached at this step's boundaries.
    pub attachments: &'a [Attached],
    /// The images this step's lanes submitted; empty for a text-only fire.
    pub media: &'a [Media<'a>],
}

/// One fire's recurrent-state plan, resolved on the host; the default is the plain path.
#[derive(Debug, Default, Clone)]
struct RsFire<'a> {
    /// What each lane moves between the arena and the buffer.
    moves: Vec<RsMove<'a>>,
    /// Where each lane's accepted prefix ends.
    lens: Vec<i32>,
    /// Submission index to fire lane.
    order: Vec<u32>,
    /// Does any lane fold at all?
    write_state: bool,
    /// Must the fold predicate be bound this fire?
    predicated: bool,
    /// Must the accepted lengths be bound this fire?
    truncates: bool,
    /// Does some row's fold boundary fall strictly inside its own tokens?
    splits: bool,
    /// Does any lane move buffered bytes? Such a fire cannot graph-replay.
    buffered: bool,
}

/// Every host decision one step needs, made — and not one stream touched.
pub struct Prepared<'a> {
    /// The step this was prepared from.
    lanes: &'a [Seated<'a>],
    /// Its attachments, gated and in order.
    attachments: &'a [Attached],
    /// Words to classes, classes to an order, counts to prefix sums.
    composition: Composition,
    /// What the walk reads to know which nodes have rows.
    descriptor: FireDescriptor,
    /// The patch payload, in fire order; empty for a fire with no image.
    patch_payload: Vec<u8>,
    patch_segments: Vec<i32>,
    patch_routes: Vec<i32>,
    patch_positions: Vec<i32>,
    patch_embed_rows: Vec<i32>,
    patch_embed_weights: Vec<f32>,
    /// The trunk's rotation stream; empty unless the plan declares it.
    mrope_positions: Vec<i32>,
    /// Every region's rows and lanes, bound to a device address only in `enqueue`.
    windows: Windows,
    /// One per lane, in fire (seriated) order.
    seats: Vec<Seat>,
    /// Each lane's stated page table, parallel to `seats`; empty for the shell's.
    tables: Vec<std::borrow::Cow<'a, [u32]>>,
    /// Page arithmetic, once per kv space.
    geometries: Vec<kv::Geometry>,
    /// How many page ids the first space carved.
    pages: u32,
    /// The slots whose recurrent banks this step must zero before it runs.
    fresh: Vec<u32>,
    /// This step's recurrent-state plan.
    rs: RsFire<'a>,
    /// What this step will take from supply.
    demand: Demand,
    /// This step's staging slot; `settle` moves it into the callback.
    slot: Option<crate::inputs::SlotGuard>,
    /// What went into that slot, as lengths.
    lengths: crate::inputs::Staged,
    /// Is this fire a body's? Decided here, because it decides the staging.
    bodied: bool,
    /// Which regions that body holds, per template region.
    admits: std::sync::Arc<[crate::window::Admit]>,
    /// The body key's class ladder.
    ladder: record::Ladder,
    /// The lane ceiling that ladder's decode rungs were taken from.
    lane_ceiling: u32,
    /// The patch unit's ladder, or `None` for an artifact with no patch axis.
    patch_ladder: Option<record::Ladder>,
    /// Does this artifact state a patch axis at all?
    towered: bool,
}

impl Drop for Prepared<'_> {
    fn drop(&mut self) {
        // `abort_step`: the slot is the only thing to release.
        drop(self.slot.take());
    }
}

impl PreparedPhase for Prepared<'_> {
    fn demand(&self) -> Demand {
        self.demand
    }
}

/// One step, on the stream.
pub struct Enqueued<'a> {
    /// The step's host state, carried through for its staging slot.
    prepared: Prepared<'a>,
    /// How many launches went onto the stream.
    launches: u32,
    /// Where a caller that wants numbers would read them; `None` for the arming pass.
    readback: Option<Readback>,
}

impl EnqueuedPhase for Enqueued<'_> {
    fn launches(&self) -> u32 {
        self.launches
    }
}
