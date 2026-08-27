//! The engine's door: boot in call order, and one fire in call order.
//!
//! **THIS FILE HAS NO LOGIC AND THAT IS THE DESIGN** (§6: shells are thin
//! call-order crates). Every decision it looks like it makes was made
//! somewhere else and is being read back here: which windows run is
//! `driver::fire::walk`'s, where a rectangle lives is the compiler's carve,
//! which kernel answers an op is the dispatch arm's, which page a token lands
//! in is [`store::kv`](crate::store::kv)'s arithmetic. What is left — and
//! what a reader should be able to follow top to bottom — is the ORDER.
//!
//! ```text
//! load                                  fire
//! ----                                  ----
//! bind the device, probe it once        lane words  -> compose
//! compile(plan, budgets, profile)       regions     -> windows
//! read the kv spaces off the plan       seats       -> page geometry
//! land the checkpoint                   stage the resident inputs
//! reserve arena, pools, inputs          carve the slot table
//! find the "out" seam                   build the cache table
//!                                       Run::new
//!                                       walk(plan, baked, desc, run, Cursor)
//!                                       synchronize, read the last row
//! ```
//!
//! # The shell holds sequence state, and only this much of it
//!
//! A slot is a sequence's seat in the pools: its kv pages and its recurrent
//! banks. All the shell remembers about one is how many kv tokens it holds —
//! which is what the next fire's positions, page bounds and write descriptors
//! are all derived from. Everything else about a request (its text, its
//! sampler, its adapter) belongs to the engine.
//!
//! # One shell fires at a time, per process
//!
//! `kernels-cuda`'s scratch slabs are process-global and keyed by name
//! (`Ctx::scratch`), and the dense autotuner keeps one device state beside
//! them — that is deliberate, because an entry that allocated per fire could
//! not be captured. The consequence lands here: two shells firing at once, on
//! two streams, stage into the same bytes. It is not a refusal either side
//! can make, because neither knows about the other; it is a fluent-garbage
//! continuation. So a process serves one fire at a time, which is also what
//! the engine's own GPU suite arranges by being thirty binaries rather than
//! one.
//!
//! # Mixed fires
//!
//! A fire whose lanes fall in more than one class is design §0's headline
//! case and this shell runs it: decode attention and prefill attention in ONE
//! fire, each over its own rows. The mechanism is not here either — it is
//! [`window`](crate::window), which resolves every region of the template to
//! its row-and-lane interval, and a [`Run`] that cuts each operand to the
//! interval of the node asking. What this file owns is one more call in the
//! order: [`Windows::of`] before the staging, because the per-window boundary
//! vectors are among the bytes the staging writes.
//!
//! # The three modes, and why the golden one is still first
//!
//! [`Graphs`] is the whole of the shell's capture policy, and it is a word,
//! not a branch in the fire path:
//!
//! ```text
//! Off      the golden path. Schedules are carved to fit this fire, the walk
//!          runs eagerly, no graph exists. Everything else is diffed against
//!          what this mode says.
//! Shaped   the same eager walk, with `FireBindings::capture` set — so the
//!          plan builders carve graph-shaped, padded schedules. It is the
//!          ATTRIBUTION arm: it isolates "the schedules changed" from "the
//!          graph changed", and a difference between Off and Shaped is a
//!          statement about flashinfer's padded split, not about capture.
//! On       Shaped, plus `record.rs`: capture once per shape key, replay
//!          after.
//! ```
//!
//! `PIE_CUDA_GRAPHS=off|shaped|on` overrides what a [`Boot`] asked for, in
//! the idiom `Toggles::from_env` already set on this plane: read once, at
//! load, never on the fire path.
//!
//! # What v1 does not do
//!
//! tp=1, so no collective ever fires, and no bucket padding — a fire's shape
//! IS its key (`record.rs` argues the mechanism, and what padding would take).
//! The PTIR prologue and epilogue are wired ([`Shell::fire_attached`]); what
//! is not is a guest program INSIDE the graph, which design §9 rules out
//! rather than defers.

use std::cell::Cell;
use std::path::Path;

use driver::fire::{FireDescriptor, Lane as FireLane, compose, walk};
use kernels_cuda::attn::plan::Shape;
use model_compiler::{Baked, Budgets, DeviceProfile, compile};
use model_ir::{Dtype, Plan, ValueId};
use model_loader::contract::ModelContract;

use crate::arena::Arena;
use crate::device::Context;
use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use driver::driver_api::fire::{Boundary, Mask};

use crate::program::launch::INTRINSIC_STORAGE_RAW_BF16;
use crate::program::{Fired, Plane as ProgramPlane, Session as ProgramSession};
use crate::record::{self, Graphs as GraphCache};
use crate::run::{CacheGeometry, CachePlanning, FireBindings, FireTables, Run};
use crate::store::kv::{self, Paging, Seat, SpaceFacts};
use crate::store::Pools;
use crate::weights::Weights;
use crate::window::{Cursor, Windows};

/// The name `model_dsl::seam::OUT` states — the one seam whose value a reader
/// touches after the graph has run.
///
/// A literal, because this crate does not depend on the authoring surface;
/// the compiler's arena carries the same literal for the same reason and says
/// so. The coupling is one string, and a plan that lost it would fail
/// [`Shell::load`] with a sentence rather than reading somebody else's bytes.
const OUT_SEAM: &str = "out";

/// How much of a fire this shell records.
///
/// **THE GOLDEN PATH IS A VALUE OF THIS TYPE, NOT AN ABSENCE.** Eager is what
/// every recorded fire is diffed against (decision #11), so it stays a
/// first-class mode of the same shell rather than a build without the other
/// one — and [`Graphs::Shaped`] exists because a difference has two possible
/// authors and a golden that cannot tell them apart is a bisect nobody can
/// finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Graphs {
    /// Eager, with schedules carved to fit each fire. The golden.
    #[default]
    Off,
    /// Eager, with graph-shaped (padded) schedules.
    Shaped,
    /// Captured once per shape key, replayed after.
    On,
}

impl Graphs {
    /// Whether the plan builders are told to carve graph-shaped schedules —
    /// [`FireBindings::capture`], the shell's policy word going in.
    #[must_use]
    pub fn shaped(self) -> bool {
        !matches!(self, Graphs::Off)
    }

    /// Whether fires reach [`record`](crate::record).
    #[must_use]
    pub fn records(self) -> bool {
        matches!(self, Graphs::On)
    }

    /// `PIE_CUDA_GRAPHS`, if it names one of the three; otherwise `stated`.
    ///
    /// Read ONCE, at load, like `Device::probe` and `Toggles::from_env`
    /// beside it: an environment read on the fire path would be a syscall
    /// between two launches.
    #[must_use]
    pub fn from_env(stated: Graphs) -> Graphs {
        match std::env::var("PIE_CUDA_GRAPHS").ok().as_deref() {
            Some("off" | "0" | "eager") => Graphs::Off,
            Some("shaped") => Graphs::Shaped,
            Some("on" | "1" | "graph") => Graphs::On,
            _ => stated,
        }
    }
}

/// Everything a load states.
pub struct Boot<'a> {
    /// The traced supergraph. The ENGINE traces it and hands it across
    /// (decision #18); `Baked` never crosses, which is why this is a `Plan`
    /// and the compile happens on this side.
    pub plan: Plan,
    /// How the checkpoint's bytes become this plan's params. Stated by the
    /// caller for the same reason: it is the model's declaration, and a shell
    /// that derived it would need an arm per family.
    pub contract: &'a ModelContract,
    /// A snapshot directory, or one container file.
    pub checkpoint: &'a Path,
    /// The ceilings every fire is baked against.
    pub budgets: Budgets,
    /// What the device charges. `None` takes the defaults with this device's
    /// measured SM count in them — costs are input, not knowledge, and an
    /// unmeasured deployment should still bake something that runs.
    pub profile: Option<DeviceProfile>,
    /// Tokens per kv page.
    pub page_size: u32,
    /// The most tokens one sequence may hold.
    pub context: u32,
    /// How many sequences the pools seat at once.
    pub slots: u32,
    /// Which device to bind.
    pub ordinal: i32,
    /// How much of a fire to record — overridden by `PIE_CUDA_GRAPHS`.
    pub graphs: Graphs,
}

/// One request inside a fire.
#[derive(Debug, Clone, Copy)]
pub struct Lane<'a> {
    /// Which pool slot this request's sequence lives in.
    pub slot: u32,
    /// Its fact bits, as the model's own `Classify::of` computed them.
    ///
    /// **THE ONE GENUINELY NEW SUBMISSION FIELD** (decision #18). It is
    /// computed engine-side because the engine links `model` anyway, and it
    /// is what `compose` turns into a class and therefore into a window. A
    /// word this artifact has no class for is `Fault::UnknownWord`, which
    /// says the engine and the shell disagree about what is loaded.
    pub word: u64,
    /// The token ids this fire feeds it — a prompt on the first fire, one
    /// token on every fire after.
    pub tokens: &'a [u32],
}

/// One request inside a fire, with the page table its caller owns.
///
/// **THE ONE THING [`Lane`] CANNOT SAY.** A `Lane` is a slot, a word and some
/// tokens, and everything else about where its kv lands is the shell's own
/// paging: a fixed block per slot, and a `held` count the shell keeps. That is
/// right for a deployment whose sequences are seats, and it is exactly wrong
/// for an engine with a real page allocator — copy-on-write forks, a prefix
/// cache, pages that move between sequences — because then the page table is
/// the CALLER's and a block formula names somebody else's pages.
///
/// So the contract's [`KvDelta`](driver_api::KvDelta) states both, and this is
/// its shell-side shape: `pages` empty means the shell owns the table (and
/// `held` is the shell's own count), non-empty means the caller does.
#[derive(Debug, Clone, Copy)]
pub struct Seated<'a> {
    /// The request.
    pub lane: Lane<'a>,
    /// This lane's kv pages, in sequence order. Empty means the shell's.
    pub pages: &'a [u32],
    /// How many kv tokens the slot already holds. `None` asks the shell,
    /// which is the only honest answer when the shell owns the table.
    pub held: Option<u32>,
    /// An explicit attention mask over the lane's readable extent, replacing
    /// the causal bound `attention.prefill` derives — `Some` is what makes
    /// the lane's `masked` fact true, and the word the caller stamped has to
    /// agree with it (design §0: the axis is per LANE).
    ///
    /// It is here rather than on [`Lane`] for the reason the page table is:
    /// a mask is per-fire state the CALLER holds, and a deployment whose
    /// sequences are seats submits neither. [`crate::mask`] is what turns it
    /// into the bits `attention.masked` reads.
    pub mask: Option<&'a Mask>,
}

impl<'a> Seated<'a> {
    /// A lane whose page table, token count and masking are the shell's —
    /// which for the mask means none.
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            lane,
            pages: &[],
            held: None,
            mask: None,
        }
    }

    /// The same lane, reading only `mask`'s positions of its slot.
    #[must_use]
    pub fn masked(lane: Lane<'a>, mask: &'a Mask) -> Seated<'a> {
        Seated {
            mask: Some(mask),
            ..Seated::of(lane)
        }
    }
}

/// One guest program attached to a fire's boundary (design §9).
///
/// The shell's spelling of the contract's
/// [`Attachment`](driver::driver_api::fire::Attachment), and the same rule:
/// one attachment per instance per fire, because a program's stages are ONE
/// pass with one readiness gate and one commit. [`Attached::at`] says which
/// side of the immutable graph that pass runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attached {
    /// Which lane of the submission this instance runs for — whose readout
    /// row an epilogue's `logits` intrinsic is pointed at.
    pub lane: u32,
    /// Which bound instance, as [`Shell::bind_program`] minted it.
    pub instance: u64,
    /// Which side of the graph.
    pub at: Boundary,
}

/// One loaded model, serving.
pub struct Shell {
    device: Context,
    plan: Plan,
    baked: Baked,
    budgets: Budgets,
    weights: Weights,
    arena: Arena,
    pools: Pools,
    inputs: Inputs,
    spaces: Vec<Option<SpaceFacts>>,
    /// The classes whose window runs an `attention.masked` arm — read once
    /// off the bake, because a mask is only ever read by a lane the WORD put
    /// in one of them. Empty for an artifact that declares no masked arm at
    /// all, and then a mask has nowhere to go.
    masked: model_ir::ClassSet,
    /// Per slot: how many kv tokens it holds.
    held: Vec<u32>,
    out: ValueId,
    graphs: Graphs,
    cache: GraphCache,
    /// The guest-program plane (design §9). Empty until something registers a
    /// program, and never touched by [`Shell::fire`] — see
    /// [`Shell::register_program`].
    programs: ProgramPlane,
}

impl Shell {
    /// Boot: bind, bake, land, reserve.
    ///
    /// # Errors
    ///
    /// [`Fault::Bake`] for a plan these budgets do not admit, [`Fault::Load`]
    /// for a checkpoint the contract does not fit, [`Fault::Device`] for the
    /// residency, [`Fault::Unbound`] for a plan naming a seat this shell does
    /// not bind.
    pub fn load(boot: Boot<'_>) -> Result<Shell> {
        let device = Context::bind(boot.ordinal)?;

        // Costs are input (design §6's `layout/` lineage row): the shell
        // measured the device once at bind, and hands the numbers to a
        // compiler that could equally have been run on a laptop.
        let profile = boot.profile.unwrap_or(DeviceProfile {
            sms: device.device().num_sm,
            ..DeviceProfile::default()
        });
        let baked = compile(&boot.plan, &boot.budgets, &profile)?;

        // Heads and head widths are on the ops, not on `CacheRow::Kv`, so
        // they are read off the plan rather than off a config beside it.
        let spaces = kv::probe(&boot.plan)?;
        // The window argument's bake-time half, asked once: no attention
        // schedule may be carved over more classes than the arm consuming it
        // runs in. A per-fire check would be the same answer at a worse
        // instant — region masks are static — and the sentence names the
        // model text rather than the fire.
        crate::window::no_schedule_straddles_its_readers(&boot.plan, &baked)?;
        // Whether this artifact has anywhere for a mask to GO. `masked` is a
        // fact the model declares (design §8), so a plan with no
        // `attention.masked` arm cannot serve one, and accepting the bits
        // anyway would answer with the unmasked continuation.
        //
        // Kept as a CLASS SET rather than a boolean, because the question a
        // fire asks is per lane: does the class this lane's word resolved to
        // run the masked arm? The word and the mask are stamped at two
        // instants by two parties — the engine computes the word from the
        // model's `Classify::of`, the caller states the mask — and this set
        // is what lets the shell check that they agree
        // (`Fault::{Maskless, MaskWord}`).
        let mut masked = model_ir::ClassSet::default();
        for region in baked.template() {
            let runs_masked = region.nodes.clone().any(|node| {
                matches!(
                    boot.plan.nodes.get(node as usize).map(|node| &node.op),
                    Some(model_ir::Operation::Attention(model_ir::Attention::Masked { .. }))
                )
            });
            if runs_masked {
                for class in region.mask.iter() {
                    masked.insert(class);
                }
            }
        }
        let paging = Paging::of(boot.page_size, boot.context, boot.slots)?;

        let weights = Weights::resident(&boot.plan, boot.contract, boot.checkpoint)?;
        let arena = Arena::reserve(&baked.arena)?;
        let pools = Pools::reserve(&boot.plan, paging, &spaces)?;
        let inputs = Inputs::reserve(
            &boot.budgets,
            paging,
            &spaces,
            baked.classes.classes.len(),
            device.device().num_sm,
        )?;

        let out = boot
            .plan
            .seams
            .iter()
            .find(|seam| seam.seam == OUT_SEAM)
            .and_then(|seam| seam.values.first().copied())
            .ok_or_else(|| Fault::Unbound {
                what: format!("no `{OUT_SEAM}` seam, so a fire would compute nothing a reader can take"),
            })?;

        Ok(Shell {
            device,
            plan: boot.plan,
            baked,
            budgets: boot.budgets,
            weights,
            arena,
            pools,
            inputs,
            spaces,
            masked,
            held: vec![0; boot.slots as usize],
            out,
            graphs: Graphs::from_env(boot.graphs),
            cache: GraphCache::new(),
            programs: ProgramPlane::default(),
        })
    }

    /// Open a slot for a fresh sequence.
    ///
    /// The kv pages need no clearing — `kv_len` says nothing before the
    /// append is live — but the recurrent banks do: a linear-attention scan
    /// reads its whole state on its first step, so a slot still holding the
    /// last sequence's history would continue it.
    ///
    /// **A CALLER WITH ITS OWN PAGE TABLE NEVER CALLS THIS**, and says the
    /// same thing by other means: a lane arriving with `held == 0` is a
    /// sequence beginning, and [`Shell::fire_attached`] clears the slot's
    /// banks there for exactly the reason above.
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

    /// How many kv tokens a slot holds.
    #[must_use]
    pub fn held(&self, slot: u32) -> u32 {
        self.held.get(slot as usize).copied().unwrap_or(0)
    }

    /// The plan this shell serves.
    #[must_use]
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The artifact it was baked into.
    #[must_use]
    pub fn baked(&self) -> &Baked {
        &self.baked
    }

    /// The ceilings it was baked against.
    #[must_use]
    pub fn budgets(&self) -> &Budgets {
        &self.budgets
    }

    /// How its pools hand pages out.
    #[must_use]
    pub fn paging(&self) -> Paging {
        self.pools.paging()
    }

    /// Which device it bound.
    #[must_use]
    pub fn ordinal(&self) -> i32 {
        self.device.ordinal()
    }

    /// Bind the CALLING thread to this shell's device — see
    /// [`Context::bind_thread`](crate::device::Context::bind_thread).
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] when the runtime refuses the ordinal.
    pub fn bind_thread(&self) -> Result<()> {
        self.device.bind_thread()
    }

    /// That device's parallel width, probed once at bind.
    #[must_use]
    pub fn sms(&self) -> u32 {
        self.device.device().num_sm
    }

    /// The `out` seam's row width — the vocabulary, for a plan whose out seam
    /// is logits.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for an out value whose width is symbolic.
    pub fn out_width(&self) -> Result<u64> {
        kv::width_of(&self.plan, self.out)
    }

    /// Which mode it is firing in.
    #[must_use]
    pub fn mode(&self) -> Graphs {
        self.graphs
    }

    /// Change the mode between fires.
    ///
    /// **THE A/B IS ONE LOAD, NOT TWO**: 1.7 GB of weights landed twice would
    /// be two residencies, two arenas and two tuner histories, and a
    /// difference between the runs could be any of those. One shell, one set
    /// of addresses, one word changed — then the tokens either match or the
    /// graph is wrong.
    ///
    /// Execs already captured stay cached: their key still means what it
    /// meant, and going Off and back On is a policy change, not an
    /// invalidation.
    pub fn set_mode(&mut self, graphs: Graphs) {
        self.graphs = graphs;
    }

    /// What this load's graph cache has done.
    #[must_use]
    pub fn graph_stats(&self) -> record::Stats {
        self.cache.stats()
    }

    // ── The guest-program plane (design §9) ──
    //
    // THE DOORS, AND `fire_attached` IS THE ONE THAT JOINS THEM: register a
    // program, bind an instance, publish into its channels, fire it at a
    // model fire's boundary, take what it published. The engine still owns
    // the ORDER in the only sense that matters — which lane a program is
    // attached to and at which boundary is what it submits — but the two
    // instants themselves are here, because binding `IntrinsicId::Logits` at
    // the readout needs the arena rectangle this file computes and nobody
    // else sees.

    /// Compile and register a guest program, answering its id.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] for a package that does not adopt, [`Fault::Compile`]
    /// for a region NVRTC refuses.
    pub fn register_program(
        &mut self,
        registration: &driver::driver_api::program::ProgramRegistration,
    ) -> Result<u64> {
        self.programs.register(&self.device, registration)
    }

    /// Bind an instance of `program_id`, answering its id. `seeds` are wire
    /// cells, one per `(channel, bytes)` pair.
    ///
    /// `extents` is what the program's symbolic value shapes resolve against,
    /// and it is an ARGUMENT because a guess zero-fills silently (Build log
    /// 15): every stage's fire-path buffers are carved here, at bind, and one
    /// carved for a single readout row when the fire hands it four leaves
    /// three rows of zeroes that no launch faults on. A program attached to a
    /// model fire is handed that fire's readout shape; a standalone one
    /// resolves entirely from static dims and never reads these at all, which
    /// is what [`driver::Extents::default`] — every extent one — says.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown program or a seed that does not fit.
    pub fn bind_program(
        &mut self,
        program_id: u64,
        seeds: &[(u32, Vec<u8>)],
        extents: driver::Extents,
        geometry: driver::tensor_ir::registry::GeometryClass,
    ) -> Result<u64> {
        self.programs.bind(program_id, seeds, extents, geometry)
    }

    /// How many descriptor-port envelopes have been resolved off guest device
    /// rings in this process. See [`crate::program::ports::resolved`], which
    /// is where the counter lives and why it is process-global.
    #[must_use]
    pub fn envelopes_resolved() -> u64 {
        crate::program::ports::resolved()
    }

    /// The first channel of instance `instance_id` whose declared requirement
    /// a fire right now would not meet, or `None` when it is ready.
    ///
    /// The gate [`Shell::fire_attached`] opens over every attached instance
    /// before it launches anything. See [`ProgramPlane::ready`].
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance.
    pub fn program_ready(&self, instance_id: u64) -> Result<Option<u32>> {
        self.programs.ready(instance_id)
    }

    /// One bound instance, for publishing into and taking out of its channels.
    pub fn program_instance(&mut self, instance_id: u64) -> Option<&mut ProgramSession> {
        self.programs.instance_mut(instance_id)
    }

    /// Tear down one bound instance and free its rings.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an instance that is already gone.
    pub fn close_program_instance(&mut self, instance_id: u64) -> Result<()> {
        self.programs.close_instance(instance_id)
    }

    /// Fire one guest-program instance: readiness, then its stages, then one
    /// commit.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, and whatever the launches
    /// said.
    pub fn fire_program(&mut self, instance_id: u64) -> Result<Fired> {
        self.programs.fire(&self.device, instance_id)
    }

    /// What this load holds on the device: `(weights, arena, pools, inputs)`.
    #[must_use]
    pub fn footprint(&self) -> (u64, u64, u64, u64) {
        (
            self.weights.bytes(),
            self.arena.bytes(),
            self.pools.bytes(),
            self.inputs.bytes(),
        )
    }

    /// Run one fire, and hand back each lane's last row of logits.
    ///
    /// The last row and not every row because that is the row a sampler
    /// reads: a prefill's earlier rows are teacher-forced positions nobody
    /// samples, and they are 0.5 MB each at this vocabulary. Lanes come back
    /// in SUBMISSION order, whatever order the fire ran them in.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] for a batch the artifact cannot describe or a dispatch
    /// the backend refused, [`Fault::Fragmented`] for a region whose classes
    /// this fire's order does not make consecutive, [`Fault::Ceiling`] for a
    /// sequence past its slot's pages, [`Fault::Device`] for a transfer, and
    /// — in [`Graphs::On`] — [`Fault::Schedule`] for a fire whose attention
    /// schedules are not the shape its recorded graph was captured against.
    pub fn fire(&mut self, lanes: &[Lane<'_>]) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        self.fire_seated(&seated)
    }

    /// Run one fire whose lanes may carry their own page tables.
    ///
    /// [`Shell::fire`] is this with every lane seated on the shell's own
    /// paging. The split is here rather than inside because who owns a page
    /// table is a per-lane fact, and a fire may mix the two.
    ///
    /// # Errors
    ///
    /// As [`Shell::fire`], plus [`Fault::Ceiling`] for a lane whose stated
    /// pages do not cover the tokens it is about to hold.
    pub fn fire_seated(&mut self, lanes: &[Seated<'_>]) -> Result<Vec<Vec<f32>>> {
        self.fire_attached(lanes, &[])
    }

    /// Run one fire with guest programs at its boundaries (design §9).
    ///
    /// [`Shell::fire_seated`] is this with no attachment, and a fire with no
    /// attachment does exactly what it always did — not "almost", but the
    /// same instructions in the same order, because every line the
    /// attachments add is inside a loop over an empty slice.
    ///
    /// ```text
    /// gate       program_ready over EVERY attached instance   ← nothing launched
    /// prologue   Boundary::Prologue attachments, in order
    /// forward    steps 1..9 below
    /// bind       IntrinsicId::Logits -> this lane's readout ROW of the arena
    /// epilogue   Boundary::Epilogue attachments, in order
    /// ```
    ///
    /// **THE GATE IS THE WHOLE ARGUMENT FOR THE ORDER.** An epilogue fires
    /// after the forward has written the lane's KV. A readiness refusal
    /// discovered there would be a fire nobody can retry — the tokens are in
    /// the cache and the guest's pass never happened — so every attached
    /// instance is asked BEFORE anything launches, and a blocked one refuses
    /// the fire while refusing is still free. That refusal is
    /// [`Fault::Program`] naming the instance and the channel; the caller's
    /// contract layer is what turns it into a scheduling answer.
    ///
    /// **A PROLOGUE IS NOT HANDED A READOUT**, because before the graph there
    /// is none. A program that reads `logits` and is attached at
    /// [`Boundary::Prologue`] is refused by name inside
    /// [`Session::fire`](crate::program::Session::fire) — the same refusal
    /// that guards an unbound intrinsic anywhere else, and the reason it is a
    /// sentence rather than an address-zero dereference.
    ///
    /// # Errors
    ///
    /// As [`Shell::fire_seated`], plus [`Fault::Program`] for an attachment
    /// naming a lane this fire does not have, an instance that is blocked,
    /// declined or faulted, and whatever the guest launches said.
    pub fn fire_attached(
        &mut self,
        lanes: &[Seated<'_>],
        attachments: &[Attached],
    ) -> Result<Vec<Vec<f32>>> {
        let Shell {
            device,
            plan,
            baked,
            budgets,
            weights,
            arena,
            pools,
            inputs,
            spaces,
            masked,
            held,
            out,
            graphs,
            cache,
            // NAMED, NOT `..`: the guest-program plane is touched at the
            // fire's BOUNDARIES and nowhere between them, and spelling the
            // field out is what makes that a statement rather than an
            // omission — a `..` would absorb the next field somebody adds
            // without anyone deciding it belongs here.
            programs,
        } = self;
        let graphs = *graphs;

        // ── 0. THE GATE. Nothing has launched, so a refusal here is free. ──
        //
        // Every attachment, prologue and epilogue alike, before either runs:
        // an epilogue that discovered its rings were not ready AFTER the
        // forward would leave the lane's tokens in the cache with the guest's
        // pass unrun, which is a fire the caller cannot retry.
        for (index, attached) in attachments.iter().enumerate() {
            if attached.lane as usize >= lanes.len() {
                return Err(Fault::program(
                    "serve::fire",
                    format!(
                        "attachment {index} names lane {} of the {} this fire has",
                        attached.lane,
                        lanes.len()
                    ),
                ));
            }
            if attachments[..index]
                .iter()
                .any(|earlier| earlier.instance == attached.instance)
            {
                return Err(Fault::program(
                    "serve::fire",
                    format!(
                        "instance {} is attached twice to one fire, at attachment \
                         {index}; a program's stages are one pass with one commit, so \
                         firing it twice would gate against cursors the first pass \
                         already advanced",
                        attached.instance
                    ),
                ));
            }
            if let Some(channel) = programs.ready(attached.instance)? {
                return Err(Fault::program(
                    "serve::fire",
                    format!(
                        "instance {} is not ready to fire: channel {channel} does not \
                         meet its program's declared requirement, and an epilogue \
                         cannot block after the forward has written the lane's kv",
                        attached.instance
                    ),
                ));
            }
        }

        // ── 0b. THE DESCRIPTOR PORTS, read off the rings the gate just
        //    approved (`palo B3`, and [`crate::program::ports`] is the whole
        //    argument).
        //
        //    STILL NOTHING HAS LAUNCHED. A port read is `read_cell(channel,
        //    head)` — the committed front, which is the cell the guest's own
        //    pass takes this fire — so it is a four-byte copy off an
        //    allocation this shell owns, in front of a walk that has not
        //    started. It happens HERE, before the prologue, because a
        //    prologue is a pass with a commit and its cursors would move
        //    under the read.
        //
        //    A lane whose instance was bound `GeometryClass::Host` resolves
        //    `None` and the two lines below it never run: its fire reads the
        //    submission, exactly as it always did, byte for byte. That is
        //    what makes the host-carried fixture the parity leverage for the
        //    device-carried one — same program, same channels, one class
        //    apart.
        let mut envelope_of: Vec<Option<crate::program::Envelope>> = vec![None; lanes.len()];
        for attached in attachments {
            if let Some(envelope) = programs.envelope(attached.instance)? {
                envelope_of[attached.lane as usize] = Some(envelope);
            }
        }

        // ── The prologue. Channel reads, state, token prep — never the
        //    readout, which does not exist yet.
        for attached in attachments.iter().filter(|a| a.at == Boundary::Prologue) {
            let fired = programs.fire(device, attached.instance)?;
            committed_or(fired, attached, "prologue")?;
        }

        // 1. Lane words in. `compose` is arithmetic over a `Vec` of them:
        //    words to classes, classes to an order, counts to prefix sums.
        let submitted: Vec<FireLane> = lanes
            .iter()
            .map(|seated| FireLane::new(seated.lane.word, seated.lane.tokens.len() as u32))
            .collect();
        let composition = compose(baked, budgets, &submitted)?;
        let descriptor = FireDescriptor::of(&composition);
        let rows = composition.rows();
        let lane_count = composition.lane_count();

        // 2. The fire's own vectors, in fire order — which is the seriated
        //    order the composition chose, not the order the engine submitted.
        let mut seats: Vec<Seat> = Vec::with_capacity(lanes.len());
        let mut tables: Vec<&[u32]> = Vec::with_capacity(lanes.len());
        // THE MASKED AXIS, IN FIRE ORDER. One entry per lane, seriated with
        // the rest — the span table is indexed by the schedule's request
        // number, which is a position in the class order and not the order
        // the engine submitted.
        let mut masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(lanes.len());
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut slot_ids: Vec<i32> = Vec::with_capacity(lanes.len());
        for row in composition.lanes() {
            let seated = &lanes[row.source as usize];
            let lane = &seated.lane;
            // WHO KNOWS HOW LONG THE SEQUENCE IS depends on who owns its
            // pages. A shell-owned slot is one the shell opened and has been
            // counting ever since; a caller-owned one is a page table the
            // caller forked, trimmed or restored between fires, and its own
            // count is the only one that is right.
            let have = match seated.held {
                Some(held) => held,
                None => held.get(lane.slot as usize).copied().ok_or(Fault::Ceiling {
                    what: "slots",
                    need: u64::from(lane.slot) + 1,
                    have: held.len() as u64,
                })?,
            };
            debug_assert_eq!(
                row.row_offset as usize,
                tokens.len(),
                "a lane's rows stand where the composition placed them"
            );
            // A FRESH SEQUENCE ARRIVES WITH A ZEROED RECURRENT BANK, and
            // `have == 0` is the only place the contract says a sequence
            // begins.
            //
            // [`Shell::open`] says the same thing for a caller whose page
            // table is the SHELL's: it clears the slot's recurrent banks
            // because a linear-attention scan reads its whole state on its
            // first step, so a slot still holding the last sequence's
            // history would continue it. An engine that keeps its OWN page
            // table never calls `open` — the contract has no such verb, by
            // design — and until this line nothing else cleared the banks
            // either. The kv half was fine and stayed fine: `kv_len` says
            // nothing lives past the append, so a recycled page is
            // overwritten before it is read. The recurrent half has no
            // `kv_len`.
            //
            // The launch pattern that exposed it (`palo` build log 18, and
            // `tests/gpu/tests/cuda_launch_isolation`): THREE identical
            // greedy completions through ONE booted worker. The first was
            // right — the pools were `Buffer::zeroed` at load — and the
            // second and third answered echo-shaped garbage built out of the
            // prompt's own words, because their GDN layers were still
            // running the previous launch's sequence. Every other gate in
            // this tree launches once per boot, which is why it survived.
            //
            // Cost is one `cudaMemset` over one slot's banks on the FIRST
            // fire of a sequence and never again — a chunked prefill's
            // second chunk arrives with `have > 0` — and nothing at all for
            // a plan that declares no `CacheRow::State`.
            if have == 0 {
                pools.clear(lane.slot)?;
            }
            seats.push(Seat {
                slot: lane.slot,
                have,
                rows: row.rows,
            });
            tables.push(seated.pages);
            // THE WORD AND THE MASK, CHECKED AGAINST EACH OTHER, ONCE.
            // `compose` already refused a word this artifact has no class
            // for; what it cannot know is whether the class it resolved to
            // reads a mask. Both directions are a wrong answer that looks
            // like a right one, so both are refused (`Fault::MaskWord`
            // argues each).
            let runs_masked_arm = masked.contains(row.class as usize);
            if seated.mask.is_some() && masked.is_empty() {
                return Err(Fault::Maskless { lane: row.source });
            }
            if seated.mask.is_some() != runs_masked_arm {
                return Err(Fault::MaskWord {
                    lane: row.source,
                    word: lane.word,
                    runs_masked_arm,
                });
            }
            masks.push(crate::mask::LaneMask {
                mask: seated.mask,
                have,
                rows: row.rows,
            });
            slot_ids.push(lane.slot as i32);

            // WHERE THE TOKEN COMES FROM IS THE WHOLE OF `palo B3`. A
            // host-class lane's ids are in the submission, because the engine
            // folded them and stated them. A device-resolved lane's are the
            // cell the previous fire's epilogue wrote, which the engine could
            // not know and did not state — its `Lane::tokens` carries the row
            // COUNT and placeholders, and `tokens_for` refuses a port that
            // disagrees with the count the composition already carved for.
            let source = row.source as usize;
            let rows_here = lane.tokens.len();
            match envelope_of[source].as_ref() {
                Some(envelope) => {
                    envelope.check_extent(source, have.saturating_add(row.rows))?;
                    for &token in envelope.tokens_for(source, rows_here)? {
                        tokens.push(token as i32);
                    }
                    match envelope.positions_for(source, have, rows_here)? {
                        Some(stated) => positions.extend(stated.iter().map(|&p| p as i32)),
                        None => positions
                            .extend((0..rows_here).map(|at| narrow(u64::from(have) + at as u64))),
                    }
                }
                None => {
                    for (at, token) in lane.tokens.iter().enumerate() {
                        tokens.push(*token as i32);
                        positions.push(narrow(u64::from(have) + at as u64));
                    }
                }
            }
        }

        // 3. Page arithmetic, once per kv space. Every space is paged the
        //    same way in v1 — one page size, one block per slot — so the
        //    vectors coincide; the loop is per space because the geometry
        //    seat is, and a plan with two page sizes changes this call and
        //    nothing above it.
        let indptr_host = kv::indptr(&seats);
        let paging = pools.paging();
        let geometries = spaces
            .iter()
            .map(|_| kv::geometry_with(&paging, &seats, &tables))
            .collect::<Result<Vec<_>>>()?;
        let pages = geometries
            .first()
            .map_or(0, |geometry| geometry.indices.len() as u32);

        // 4. THE WINDOWS. Every region of the template, resolved against the
        //    class table this composition built: which rows and which lanes it
        //    runs over, deduplicated, each carrying the qo boundaries a ragged
        //    view inside it is cut by — rebased, because a sub-rectangle
        //    starts at its own zero. This is the whole of what makes a mixed
        //    fire legal, and `crate::window` is where it is argued.
        let mut windows = Windows::of(baked, composition.classes(), &indptr_host)?;
        let boundaries = windows.packed();

        // 4b. THE MASK BITS. A lane states its mask as runs over its own
        //    readable extent and `attention.masked` reads one bit per
        //    (query row, key position) pair with the causal bound already
        //    folded in, so the expansion happens here, once, off the same
        //    `have` and `rows` the page geometry was carved from
        //    (`crate::mask` argues every term of it). `None` is a fire no
        //    lane masked, and then no seat is bound at all.
        let staged = crate::mask::stage(&masks)?;

        // 5. Stage them onto the fire's stream, in front of the launches
        //    that read them.
        let handles = inputs.write(
            device.stream(),
            &crate::inputs::Fire {
                tokens: &tokens,
                positions: &positions,
                windows: &boundaries,
                slot_ids: &slot_ids,
                spaces: &geometries,
                mask: staged.as_ref(),
            },
        )?;
        windows.bind(handles.windows);

        // 6. The three tables a `Run` resolves through: the arena's
        //    rectangles at this fire's rows, the pools' storage under this
        //    fire's page tables, and the loader's weights, which never move.
        let slots = arena.slots(&baked.arena, u64::from(rows), u64::from(lane_count));
        let caches = pools.table(&inputs.seats(&handles, pages, rows, lane_count))?;

        // 7. The geometry seats, and their host twins. THE DUALITY: the IR
        //    names `kv_indptr` as a device input and the plan builders are
        //    host functions that walk its CONTENTS, so the same vector is
        //    bound twice — once as a handle for the launches, once as a
        //    `Vec<i32>` for `plan_decode`/`plan_prefill`.
        let mut geometry = Vec::with_capacity(spaces.len());
        for (space, facts) in spaces.iter().enumerate() {
            let seat = handles.spaces[space];
            let host = &geometries[space];
            geometry.push(CacheGeometry {
                indptr: Some(seat.indptr),
                indices: Some(seat.indices),
                seq_lens: None,
                last_page_len: Some(seat.last_page_len),
                kv_len: Some(seat.kv_len),
                row_valid: Some(handles.row_valid),
                request_of_token: None,
                write_page: Some(seat.write_page),
                write_offset: Some(seat.write_offset),
                // The custom-mask slab, bound whole: its entries are bits and
                // `Run::cut` excludes it for the same reason it excludes the
                // page-id list. Every space gets the same handle, because
                // every space of a v1 plan is paged over the same lanes with
                // the same extents — the day two spaces hold different
                // readable extents, this reads `staged` per space.
                mask: handles.mask,
                planning: facts.map(|facts| CachePlanning {
                    kv_indptr: host.indptr.clone(),
                    kv_len: host.kv_len.clone(),
                    // The FIRE's lanes; `Run::planning` narrows this to the
                    // asking node's window, which is the count a schedule is
                    // actually carved at.
                    shape: Shape {
                        num_requests: lane_count,
                        num_q_heads: facts.q_heads,
                        num_kv_heads: facts.kv_heads,
                        head_dim: facts.head_dim,
                        page_size: paging.page_size,
                        hnd_layout: false,
                    },
                    window: facts.window,
                    decode_workspace: Some(inputs.decode_grant()),
                    prefill_workspace: Some(inputs.prefill_grant()),
                    // No SKU this shell serves declares a latent space; a
                    // plan op that fired over one would panic naming the
                    // grant it never got, which is the honest answer to a
                    // binding hole.
                    mla_workspace: None,
                }),
            });
        }

        // 8. The walk. The prepare regions build and stage the attention
        //    schedules — one per window, so a mixed fire builds both — and
        //    the capture regions enqueue. The sink records nothing, as
        //    `EagerSink` would: in an eager fire the walk's own control flow
        //    IS the structure. What it does carry is the region number, which
        //    is how a `Run` knows whose window it is resolving in.
        let bindings = FireBindings {
            tokens: handles.tokens,
            positions: handles.positions,
            geometry,
            tables: FireTables {
                // Fire-wide going in and window-sliced coming out
                // (`Run::mask_indptr`): the plan-building arm takes its own
                // window's lanes, and the byte offsets inside stay absolute
                // because the slab they point into is not sliced.
                mask_indptr: handles.mask_indptr,
                pool_state: None,
            },
            device: device.device(),
            toggles: device.toggles(),
            // The shell's policy word going in: under a mode that records,
            // the builders carve graph-shaped, padded schedules, so that the
            // numbers a capture bakes into its launches are a function of the
            // fire's SHAPE and not of its contents.
            capture: graphs.shaped(),
        };
        // The one piece of state between the two halves of the walk: the sink
        // writes which region is running, the `Run` reads it to know which
        // window to resolve in. They cannot be one object — `walk` takes two
        // `&mut` — and this is the smallest thing that stands between them.
        let region = Cell::new(0u32);
        let mut run = Run::new(
            device.ctx(),
            &plan.values,
            weights.table(),
            &slots,
            &caches,
            bindings,
            &windows,
            &region,
        );
        // TWO MODES, ONE WALK (design §6, decision #11). Off and Shaped run
        // it whole; On splits it at the phase boundary — prepare on the open
        // stream, then the capture regions either replayed from this shape's
        // graph or run and recorded into one. Which is why `record::fire`
        // takes the same arguments `walk` does and answers the same errors:
        // it is not another path, it is the same one at two instants.
        if graphs.records() {
            cache.fire(
                &record::Fire {
                    plan,
                    baked,
                    descriptor: &descriptor,
                    stream: device.stream(),
                    key: record::Key::of(composition.classes()),
                },
                &mut run,
                &region,
            )?;
        } else {
            walk(
                plan,
                baked,
                &descriptor,
                &mut run,
                &mut Cursor::new(&region),
            )?;
        }
        drop(run);

        // 9. The one synchronization a fire has, and it is here because a
        //    caller asked for numbers — every entry below is enqueue-only by
        //    design (decision #15).
        device.synchronize()?;

        let logits = slots.0[out.0 as usize].ok_or_else(|| Fault::Unbound {
            what: format!("value {}, the out seam, which the carve gave no rectangle", out.0),
        })?;
        if logits.dtype != Dtype::Bf16 {
            return Err(Fault::Unbound {
                what: format!("an out seam landed as {:?}, which this shell cannot read back", logits.dtype),
            });
        }
        let width = logits.width as usize;
        let mut taken = vec![Vec::new(); lanes.len()];
        // Which ROW of the arena's logits rectangle each SUBMITTED lane reads
        // — the fire order is the seriated one, so a lane's row is a fact the
        // composition holds and nothing else does. It is what the readback
        // below indexes and what an epilogue's `logits` intrinsic is offset
        // by, and computing it twice is how the two would come to disagree.
        let mut last_row = vec![0u32; lanes.len()];
        let mut raw = vec![0u8; width * 2];
        for row in composition.lanes() {
            let last = row.row_offset + row.rows - 1;
            last_row[row.source as usize] = last;
            arena.read(logits.ptr + u64::from(last) * width as u64 * 2, &mut raw)?;
            taken[row.source as usize] = raw
                .chunks_exact(2)
                .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]])))
                .collect();
        }

        // ── The epilogue. The readout is back, so the intrinsic has
        //    something to point at: this lane's row of the arena rectangle,
        //    read where it lies rather than copied anywhere.
        //
        //    `INTRINSIC_STORAGE_RAW_BF16` and not a widened f32 buffer: the
        //    emitted kernel widens a bf16 column with `bits << 16`, which is
        //    the same arithmetic `bf16()` below does, so the guest reads
        //    exactly the f32 the caller is handed — bit for bit, which is
        //    what makes a parity diff against the host interpreter mean
        //    anything.
        let vocab = u32::try_from(width).unwrap_or(u32::MAX);
        for attached in attachments.iter().filter(|a| a.at == Boundary::Epilogue) {
            programs.bind_intrinsic(
                device,
                attached.instance,
                driver::tensor_ir::op::IntrinsicId::Logits,
                logits.ptr,
                INTRINSIC_STORAGE_RAW_BF16,
                vocab,
                vocab,
                last_row[attached.lane as usize],
            )?;
            let fired = programs.fire(device, attached.instance)?;
            committed_or(fired, attached, "epilogue")?;
        }

        // 10. Only now: the fire happened, so the sequences are longer. Only
        //     the slots this shell counts for — a caller that owns the page
        //     table owns the count too, and writing into `held` under its
        //     slot numbering would be writing into somebody else's table.
        for (seat, table) in seats.iter().zip(&tables) {
            if table.is_empty()
                && let Some(slot) = held.get_mut(seat.slot as usize)
            {
                *slot = seat.have + seat.rows;
            }
        }
        Ok(taken)
    }
}

/// A guest pass that ran, or the sentence for the one that did not.
///
/// **THREE VERDICTS ARE FAILURES HERE AND ONE IS NOT ELSEWHERE.** Fired on
/// its own, a [`Fired::Blocked`] program is a normal answer a caller retries
/// on. Attached to a model fire it is not: the gate already asked, before
/// anything launched, so a block at this point means the pass's own cursors
/// moved under it — which one attachment per instance is exactly the rule
/// that forbids. [`Fired::Declined`] is a stage clearing its commit slot and
/// [`Fired::Faulted`] is an instance that is unusable from now on; both leave
/// the guest's channels where they were, and both are the caller's to poison.
fn committed_or(fired: Fired, attached: &Attached, at: &str) -> Result<()> {
    match fired {
        Fired::Committed => Ok(()),
        Fired::Blocked(channel) => Err(Fault::program(
            "serve::fire",
            format!(
                "instance {}'s {at} blocked on channel {channel} AFTER the gate \
                 admitted it, so something advanced its cursors between the two",
                attached.instance
            ),
        )),
        Fired::Declined => Err(Fault::program(
            "serve::fire",
            format!(
                "instance {}'s {at} declined: a stage cleared its commit slot, so \
                 nothing the guest computed this fire is visible",
                attached.instance
            ),
        )),
        Fired::Faulted(why) => Err(Fault::program(
            "serve::fire",
            format!("instance {}'s {at} faulted and stays faulted: {why}", attached.instance),
        )),
    }
}

/// One bf16, widened.
///
/// The top sixteen bits of an f32 and nothing else — bf16 exists to make this
/// the whole conversion. Reading one as an f16 instead is the mistake the
/// loader's own docs name: same width, different exponent, and 0.0385 becomes
/// 1.6e-12 without crashing or warning.
fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}
