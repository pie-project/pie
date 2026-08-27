//! The engine's door: boot in call order, and one fire in call order.
//!
//! **THIS FILE HAS NO LOGIC AND THAT IS THE DESIGN** (§6: shells are thin
//! call-order crates). Every decision it looks like it makes was made
//! somewhere else and is being read back here: which windows run is
//! `driver::fire::walk`'s, where a rectangle lives is the compiler's carve,
//! which kernel answers an op is the dispatch arm's, which page a token
//! lands in is [`store::kv`](crate::store::kv)'s arithmetic. What is left —
//! and what a reader should be able to follow top to bottom — is the ORDER.
//!
//! ```text
//! load                                  fire
//! ----                                  ----
//! bind the device                       lane words  -> compose
//! compile(plan, budgets, profile)       regions     -> windows
//! read the kv spaces off the plan       seats       -> page geometry
//! land the checkpoint                   write the resident inputs
//! reserve arena, pools, inputs          carve the slot table
//! find the "out" seam                   build the cache table
//!                                       open one command buffer
//!                                       walk(plan, baked, desc, run, Cursor)
//!                                       commit + wait, read the last row
//! ```
//!
//! # There is no capture here, and that is §6's ruling rather than a gap
//!
//! The CUDA sibling has a `record.rs` and three modes; this shell has one.
//! Design §6 states it in the tree itself — *"no record.rs: dispatch is
//! encode-only (§15), so `EagerSink` per fire IS encoding"* — and the reason
//! is Metal's own shape. A CUDA launch is a call, so recording one into a
//! graph is a different act from performing it, and the whole capture
//! apparatus exists to buy back the per-launch host cost. A Metal dispatch
//! is already only an ENCODE: `dispatchThreads:` writes into a command
//! buffer and nothing runs until `commit`, so the fire path is a capture
//! that is submitted instead of replayed. What a Metal shell would still
//! gain from is an *indirect command buffer* — a reusable encoded pass — and
//! that is a future note, not this wave.
//!
//! One consequence worth naming: the eager walk of an artifact P6 baked with
//! fork groups is the SERIALIZATION of that DAG (build log 24's argument,
//! unchanged), because every fork edge runs from a lower region index to a
//! higher one. A metal `Cursor` no-ops `fork` and `join` and the answer is
//! the same answer.
//!
//! # The shell holds sequence state, and only this much of it
//!
//! A slot is a sequence's seat in the pools: its kv pages and its recurrent
//! banks. All the shell remembers about one is how many kv tokens it holds —
//! which is what the next fire's positions, page bounds and write
//! descriptors are all derived from. Everything else about a request (its
//! text, its sampler, its adapter) belongs to the engine.
//!
//! # What this plane refuses, and it refuses by name
//!
//! `kernels-metal` stamps one dtype (bf16) and stubs whole families: the MLA
//! ops, the indexer ops, the pooled-attention ops, the `hc.*` ops,
//! `norm.res_blend`, `attention.merge_lse`, every collective, and
//! `linear.lora_correct`. A plan that reaches one gets
//! `KernelError::Unsupported` carrying the op's own name, at the node that
//! needs it — never a silently-skipped launch. The consequence for this file
//! is that it seats no adapter routes and no draft or capture readout: those
//! axes have no arm to run on this plane, so a submission field for them
//! would be a promise the dispatch layer breaks one launch later.

use std::cell::Cell;
use std::path::Path;

use driver::fire::{FireDescriptor, Lane as FireLane, compose, walk};
use model_compiler::{Baked, Budgets, DeviceProfile, compile};
use model_ir::{Dtype, Plan, ValueId};
use model_loader::contract::ModelContract;

use crate::arena::Arena;
use crate::device::{Context, Handles, Pipelines};
use crate::encode::Sink;
use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use crate::run::{CacheGeometry, FireBindings, FireTables, Run};
use crate::store::Pools;
use crate::store::kv::{self, Paging, Seat};
use crate::weights::Weights;
use crate::window::{Cursor, Windows};

use driver::driver_api::fire::Mask;

/// The seam name the trunk's logits arrive under.
const OUT_SEAM: &str = "out";

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
    /// What the device charges. `None` takes the defaults.
    ///
    /// **THE FORK GROUPS ARE BAKED OFF, AND THE DEFAULT SAYS SO.** P6's
    /// side streams are a CUDA-stream mechanism the eager walk serializes
    /// anyway (see the module doc), so a metal load asks the compiler for an
    /// artifact with no fork group at all — which is byte for byte the
    /// artifact the compiler produced before P6 existed. A caller that
    /// states its own profile is free to say otherwise, and the walk will
    /// still serialize it.
    pub profile: Option<DeviceProfile>,
    /// Tokens per kv page.
    pub page_size: u32,
    /// The most tokens one sequence may hold.
    pub context: u32,
    /// How many sequences the pools seat at once.
    pub slots: u32,
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
    /// is what `compose` turns into a class and therefore into a window.
    pub word: u64,
    /// The token ids this fire feeds it — a prompt on the first fire, one
    /// token on every fire after.
    pub tokens: &'a [u32],
}

/// One request inside a fire, with the page table its caller owns.
///
/// **THE ONE THING [`Lane`] CANNOT SAY.** A `Lane` is a slot, a word and
/// some tokens, and everything else about where its kv lands is the shell's
/// own paging: a fixed block per slot, and a `held` count the shell keeps.
/// That is right for a deployment whose sequences are seats, and exactly
/// wrong for an engine with a real page allocator — so the contract states
/// both, and this is its shell-side shape.
#[derive(Debug, Clone, Copy)]
pub struct Seated<'a> {
    /// The request.
    pub lane: Lane<'a>,
    /// This lane's kv pages, in sequence order. Empty means the shell's.
    pub pages: &'a [u32],
    /// How many kv tokens the slot already holds. `None` asks the shell,
    /// which is the only honest answer when the shell owns the table.
    pub held: Option<u32>,
    /// An explicit attention mask over the lane's readable extent.
    ///
    /// **THIS PLANE STAGES NO MASK BITS YET, AND SAYS SO AT THE FIRE.** The
    /// metal sdpa shaders read a mask plane indexed by the LAUNCH's local
    /// row with a stated stride, which is a different ABI from the CUDA
    /// shell's per-lane packed runs plus an indptr; the expansion for it is
    /// unwritten. A lane that carries one is refused by name
    /// ([`Fault::Maskless`]) rather than served the unmasked continuation,
    /// which is the answer that would look right.
    pub mask: Option<&'a Mask>,
}

impl<'a> Seated<'a> {
    /// A lane whose pages and count are the shell's, carrying no mask.
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            lane,
            pages: &[],
            held: None,
            mask: None,
        }
    }
}

/// A loaded model, and the door a fire comes through.
pub struct Shell {
    device: Context,
    /// The compiled shader points, held for the life of the load. A steady
    /// stream of fires compiles nothing — [`Pipelines::compiled`] is the
    /// counter that makes the absence observable.
    pipelines: Pipelines,
    /// The handle table. Sealed after the weight rows are minted, rewound at
    /// the end of every fire.
    handles: Handles,
    plan: Plan,
    baked: Baked,
    budgets: Budgets,
    weights: Weights,
    arena: Arena,
    pools: Pools,
    inputs: Inputs,
    /// What the plan restates about its own caches: per cache ROW (the bytes
    /// one page holds) and per PLAN VALUE (the reading one schedule carves).
    ///
    /// Read at load to size the pools and to refuse a plan whose cache rows
    /// and attention readings disagree, and then held rather than dropped:
    /// the CUDA sibling's fire path builds a `ScheduleSeat` per plan value
    /// out of it every fire, and this plane's builders are pure carriers
    /// with no schedule to seat (`serve`'s step 7). Kept so that the day a
    /// metal schedule needs a workspace the fact is already in hand, and
    /// named here rather than deleted so the difference is visible.
    #[allow(dead_code)]
    facts: kv::Facts,
    /// How many kv geometry spaces the plan declares.
    spaces: usize,
    /// The classes whose window runs an `attention.masked` arm — read once
    /// off the bake, because a mask is only ever read by a lane the WORD put
    /// in one of them. Empty for an artifact that declares no masked arm.
    masked: model_ir::ClassSet,
    /// Per slot: how many kv tokens it holds.
    held: Vec<u32>,
    /// The trunk's logits, as the plan's `out` seam names them.
    out: ValueId,
    /// The guest-program plane (design §9). Empty until something registers
    /// a program, and never touched by [`Shell::fire_seated`] — a guest pass
    /// runs BESIDE a fire, at its boundaries, never inside it.
    programs: crate::program::Plane,
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
        let device = Context::bind()?;

        // Costs are input (design §6's `layout/` lineage row): the shell
        // hands numbers to a compiler that could equally have been run on a
        // laptop. `side_streams: 0` is the metal reading of P6 — see the
        // module doc — and it is set here rather than left to a default so
        // that a caller reading this file learns it.
        let profile = boot.profile.unwrap_or(DeviceProfile {
            sms: device.cores(),
            side_streams: 0,
            ..DeviceProfile::default()
        });
        let baked = compile(&boot.plan, &boot.budgets, &profile)?;

        // Heads, head widths and windows are on the ops, not on
        // `CacheRow::Kv`, so they are read off the plan rather than off a
        // config beside it — per cache ROW for the bytes a page holds, per
        // PLAN VALUE for the reading a schedule is carved at.
        let facts = kv::probe(&boot.plan)?;
        // The window argument's bake-time half, asked once: no attention
        // schedule may be carved over more classes than the arm consuming it
        // runs in. A per-fire check would be the same answer at a worse
        // instant — region masks are static — and the sentence names the
        // model text rather than the fire.
        crate::window::no_schedule_straddles_its_readers(&boot.plan, &baked)?;

        // Whether this artifact has anywhere for a mask to GO. Kept as a
        // CLASS SET rather than a boolean, because the question a fire asks
        // is per lane: does the class this lane's word resolved to run the
        // masked arm? The word and the mask are stamped at two instants by
        // two parties, and this set is what lets the shell check they agree.
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
        let handles = Handles::new();
        let weights =
            Weights::resident(&device, &handles, &boot.plan, boot.contract, boot.checkpoint)?;
        // **THE WEIGHT ROWS ARE THE LOAD-LIVED HANDLES, AND THIS IS THE
        // WATERMARK.** Everything minted after this line belongs to one fire
        // and is dropped at the end of it (`Handles::rewind`); everything
        // before it is read by every fire for the life of the load.
        handles.seal();

        let arena = Arena::reserve(&device, &baked.arena)?;
        let pools = Pools::reserve(&device, &boot.plan, paging, &facts)?;
        let spaces = boot
            .plan
            .caches
            .iter()
            .filter_map(|row| match row {
                model_ir::CacheRow::Kv { space, .. } => Some(*space as usize + 1),
                model_ir::CacheRow::State { .. } => None,
            })
            .max()
            .unwrap_or(0);
        let inputs = Inputs::reserve(
            &device,
            &boot.budgets,
            paging,
            spaces,
            baked.classes.classes.len(),
        )?;

        let out = boot
            .plan
            .seams
            .iter()
            .find(|seam| seam.seam == OUT_SEAM)
            .and_then(|seam| seam.values.first().copied())
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "no `{OUT_SEAM}` seam, so a fire would compute nothing a reader can take"
                ),
            })?;

        Ok(Shell {
            device,
            pipelines: Pipelines::new(),
            handles,
            plan: boot.plan,
            baked,
            budgets: boot.budgets,
            weights,
            arena,
            pools,
            inputs,
            facts,
            spaces,
            masked,
            held: vec![0; boot.slots as usize],
            out,
            programs: crate::program::Plane::new(),
        })
    }

    /// Open a slot for a fresh sequence.
    ///
    /// The kv pages need no clearing — `kv_len` says nothing before the
    /// append is live — but the recurrent banks do: a linear-attention scan
    /// reads its whole state on its first step, so a slot still holding the
    /// last sequence's history would continue it (palo build log 19).
    ///
    /// **A CALLER WITH ITS OWN PAGE TABLE NEVER CALLS THIS**, and says the
    /// same thing by other means: a lane arriving with `held == 0` is a
    /// sequence beginning, and [`Shell::fire_seated`] clears the slot's banks
    /// there for exactly the reason above.
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

    /// The artifact this shell walks.
    #[must_use]
    pub fn baked(&self) -> &Baked {
        &self.baked
    }

    /// The ceilings every fire is composed against.
    #[must_use]
    pub fn budgets(&self) -> &Budgets {
        &self.budgets
    }

    /// How the pools are paged.
    #[must_use]
    pub fn paging(&self) -> Paging {
        self.pools.paging()
    }

    /// The bound device's own name.
    #[must_use]
    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    /// The core count the cost model was handed — this plane's stand-in for
    /// the CUDA sibling's SM count. STATED rather than probed (Metal
    /// publishes no such number), which is why `api`'s `DeviceFacts` says so
    /// rather than presenting it as measured.
    #[must_use]
    pub fn cores(&self) -> u32 {
        self.device.cores()
    }

    /// One reservation's ceiling, as the device states it. What
    /// `Fault::Ceiling` is raised against when a carve will not fit a single
    /// `MTLBuffer`.
    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        self.device.max_buffer()
    }

    /// What the device says it will hold resident.
    #[must_use]
    pub fn working_set(&self) -> u64 {
        self.device.working_set()
    }

    /// The contract's thread-binding verb. Metal has no per-thread device
    /// state, so this is `Ok(())` and the reason is in [`Context::bind_thread`].
    ///
    /// # Errors
    ///
    /// Never; the signature matches the CUDA sibling's.
    pub fn bind_thread(&self) -> Result<()> {
        self.device.bind_thread()
    }

    /// How many shader points this load has compiled.
    ///
    /// The warm-cache observable: a steady stream of fires over one
    /// composition compiles nothing after the first, and an absence has no
    /// output unless something counts it.
    #[must_use]
    pub fn compiled(&self) -> u64 {
        self.pipelines.compiled()
    }

    /// The width of one readout row.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] when the carve gave the `out` seam no rectangle.
    pub fn out_width(&self) -> Result<u64> {
        let slots = self.arena.slots(
            &self.handles,
            &self.baked.arena,
            u64::from(self.budgets.max_tokens),
            u64::from(self.budgets.max_lanes),
        )?;
        slots.0[self.out.0 as usize]
            .map(|logits| u64::from(logits.width))
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, the out seam, which the carve gave no rectangle",
                    self.out.0
                ),
            })
    }

    /// What this load holds: weights, arena, pools, inputs — in bytes.
    #[must_use]
    pub fn footprint(&self) -> (u64, u64, u64, u64) {
        (
            self.weights.bytes(),
            self.arena.bytes(),
            self.pools.bytes(),
            self.inputs.bytes(),
        )
    }

    /// Register a guest program: adopt its package, compile every generated
    /// region this device will run.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] for a package that does not adopt, [`Fault::Compile`]
    /// for a region the Metal compiler refuses.
    pub fn register_program(
        &mut self,
        registration: &driver::driver_api::program::ProgramRegistration,
    ) -> Result<u64> {
        self.programs.register(&self.device, registration)
    }

    /// Bind an instance of `program_id`, answering its id. `seeds` are wire
    /// cells, one per `(channel, bytes)` pair.
    ///
    /// `extents` is what the program's symbolic value shapes resolve
    /// against, and it is an ARGUMENT because a guess zero-fills silently
    /// (build log 15): every stage's fire-path buffers are carved here, at
    /// bind, and one carved for a single readout row when the fire hands it
    /// four leaves three rows of zeroes that no launch faults on.
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
        self.programs
            .bind(&self.device, program_id, seeds, extents, geometry)
    }

    /// The first channel of `instance_id` whose declared requirement a fire
    /// right now would not meet, or `None` when it is ready.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance.
    pub fn program_ready(&self, instance_id: u64) -> Result<Option<u32>> {
        self.programs.ready(instance_id)
    }

    /// The session behind an instance id, for a caller that publishes into
    /// its channels or drains them.
    pub fn program_instance(&mut self, instance_id: u64) -> Option<&mut crate::program::Session> {
        self.programs.instance_mut(instance_id)
    }

    /// Drop an instance, freeing its rings and its stage buffers.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when there is no such instance.
    pub fn close_program_instance(&mut self, instance_id: u64) -> Result<()> {
        self.programs.close_instance(instance_id)
    }

    /// Run one instance's pass, on its own, beside no model fire.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, and whatever the launches
    /// said. A pass that blocks or refuses is a [`Fired`](crate::Fired), not
    /// an error.
    pub fn fire_program(&mut self, instance_id: u64) -> Result<crate::Fired> {
        self.programs.fire(&self.device, instance_id)
    }

    /// What the compile tiers have been doing.
    #[must_use]
    pub fn program_stats(&self) -> driver::CacheStats {
        self.programs.stats()
    }

    /// One fire over lanes whose pages and counts are the shell's.
    ///
    /// # Errors
    ///
    /// As [`Shell::fire_seated`].
    pub fn fire(&mut self, lanes: &[Lane<'_>]) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        self.fire_seated(&seated)
    }

    /// One fire, in call order.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] for a batch the artifact cannot describe or a
    /// dispatch this plane refuses, [`Fault::Maskless`]/[`Fault::MaskWord`]
    /// for a lane whose mask and word disagree, [`Fault::Ceiling`] for a
    /// count past a reservation, [`Fault::Device`] for a command buffer the
    /// GPU refused.
    #[allow(clippy::too_many_lines)]
    pub fn fire_seated(&mut self, lanes: &[Seated<'_>]) -> Result<Vec<Vec<f32>>> {
        let Shell {
            device,
            pipelines,
            handles,
            plan,
            baked,
            budgets,
            weights,
            arena,
            pools,
            inputs,
            facts: _,
            spaces,
            masked,
            held,
            out,
            // NAMED, NOT `..`: the guest-program plane is touched at the
            // fire's BOUNDARIES and nowhere between them, and spelling the
            // field out is what makes that a statement rather than an
            // omission — a `..` would absorb the next field somebody adds
            // without anyone deciding it belongs here.
            programs: _,
        } = self;

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
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut slot_ids: Vec<i32> = Vec::with_capacity(lanes.len());
        // WHICH LANE OWNS EACH TOKEN ROW, in fire row order — the vector the
        // metal sdpa entries index the page table through. The CUDA sibling
        // needs none: its plan builders walk the boundaries host-side. Built
        // here, from the composition, because the composition is the only
        // thing that knows a lane's fire POSITION (which is what a page
        // table is indexed by) as against its submission order.
        let mut request_of_token: Vec<i32> = Vec::with_capacity(rows as usize);
        // And the recurrent slot map, per ROW rather than per lane — see
        // `store::Seats::slot_of_row` for why this plane needs both shapes.
        let mut slot_of_row: Vec<i32> = Vec::with_capacity(rows as usize);
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
            // begins (palo build log 19, ported verbatim in spirit). The kv
            // half needs nothing — `kv_len` says nothing lives past the
            // append, so a recycled page is overwritten before it is read —
            // and the recurrent half has no `kv_len`: a linear-attention
            // scan reads its whole state on its first step, so a slot still
            // holding the last sequence's history would continue it. The
            // launch pattern that exposed it on the CUDA plane was three
            // identical completions through ONE boot; the second and third
            // answered echo-shaped garbage. This shell has the same banks
            // and the same exposure.
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
            // like a right one, so both are refused.
            let runs_masked_arm = masked.contains(row.class as usize);
            if seated.mask.is_some() {
                // This plane stages no mask bits (see `Seated::mask`), so
                // the refusal is unconditional and names the reason rather
                // than the class.
                return Err(Fault::Maskless { lane: row.source });
            }
            if runs_masked_arm {
                return Err(Fault::MaskWord {
                    lane: row.source,
                    word: lane.word,
                    runs_masked_arm,
                });
            }
            slot_ids.push(lane.slot as i32);
            let at_lane = slot_ids.len() as i32 - 1;
            for (at, token) in lane.tokens.iter().enumerate() {
                tokens.push(*token as i32);
                positions.push(narrow(u64::from(have) + at as u64));
                request_of_token.push(at_lane);
                slot_of_row.push(lane.slot as i32);
            }
        }

        // 3. Page arithmetic, once per kv space. Every space is paged the
        //    same way in v1 — one page size, one block per slot — so the
        //    vectors coincide; the loop is per space because the geometry
        //    seat is.
        let indptr_host = kv::indptr(&seats);
        let paging = pools.paging();
        let geometries = (0..*spaces)
            .map(|_| kv::geometry_with(&paging, &seats, &tables))
            .collect::<Result<Vec<_>>>()?;
        let pages = geometries
            .first()
            .map_or(0, |geometry| geometry.indices.len() as u32);

        // 4. THE WINDOWS. Every region of the template, resolved against the
        //    class table this composition built: which rows and which lanes
        //    it runs over, deduplicated, each carrying the qo boundaries a
        //    ragged view inside it is cut by — rebased, because a
        //    sub-rectangle starts at its own zero. This is the whole of what
        //    makes a mixed fire legal, and `crate::window` argues it.
        let mut windows = Windows::of(baked, composition.classes(), &indptr_host)?;
        let boundaries = windows.packed();

        // 5. Write the resident inputs. **THERE IS NO STAGING COPY ON THIS
        //    PLANE** — the reservations are `StorageModeShared`, so this is a
        //    memcpy into the same bytes the GPU will read. What still has to
        //    hold is the ORDER, and it is this line standing before the
        //    command buffer opens.
        let bound = inputs.write(
            handles,
            &crate::inputs::Fire {
                tokens: &tokens,
                positions: &positions,
                windows: &boundaries,
                slot_ids: &slot_ids,
                slot_of_row: &slot_of_row,
                request_of_token: &request_of_token,
                adapter_routes: None,
                spaces: &geometries,
                mask: None,
            },
        )?;
        windows.bind(handles, bound.windows)?;

        // 6. The three tables a `Run` resolves through: the arena's
        //    rectangles at this fire's rows, the pools' storage under this
        //    fire's page tables, and the loader's weights, which never move.
        let slots = arena.slots(
            handles,
            &baked.arena,
            u64::from(rows),
            u64::from(lane_count),
        )?;
        let caches = pools.table(
            handles,
            &inputs.seats(handles, &bound, pages, rows, lane_count),
        )?;

        // 7. The geometry seats. Metal's plan builders are pure carriers —
        //    they hold the tables the sdpa shaders read and compute no
        //    schedule at all — so there is no host twin of the page vectors
        //    here and no workspace grant, which is the whole of what the
        //    CUDA sibling's `CachePlanning` and `ScheduleSeat` carry.
        let mut geometry = Vec::with_capacity(*spaces);
        for space in 0..*spaces {
            let seat = bound.spaces[space];
            geometry.push(CacheGeometry {
                indptr: Some(seat.indptr),
                indices: Some(seat.indices),
                seq_lens: None,
                last_page_len: Some(seat.last_page_len),
                kv_len: Some(seat.kv_len),
                row_valid: Some(bound.row_valid),
                request_of_token: None,
                write_page: Some(seat.write_page),
                write_offset: Some(seat.write_offset),
            });
        }

        // 8. The walk, inside one command buffer and one compute pass. The
        //    pass is `MTLDispatchTypeSerial`, so every dispatch observes the
        //    writes of the one before it — which is exactly the ordering the
        //    walk assumes of a stream, and why nothing here speaks of
        //    barriers.
        let bindings = FireBindings {
            tokens: bound.tokens,
            positions: bound.positions,
            geometry,
            tables: FireTables {
                request_of_token: bound.request_of_token,
                mask: bound.mask,
                mask_enabled: bound.mask_enabled,
                mask_stride: bound.mask_stride,
            },
        };
        // The one piece of state between the two halves of the walk: the
        // sink writes which region is running, the `Run` reads it to know
        // which window to resolve in. They cannot be one object — `walk`
        // takes two `&mut` — and this is the smallest thing between them.
        let region = Cell::new(0u32);
        let frame = device.frame()?;
        {
            let sink = Sink::new(device, &frame, pipelines, handles);
            let mut run = Run::new(
                &sink,
                handles,
                &plan.values,
                weights.table(),
                &slots,
                &caches,
                bindings,
                &windows,
                &region,
            );
            walk(plan, baked, &descriptor, &mut run, &mut Cursor::new(&region))?;
        }

        // 9. The one synchronization a fire has, and it is here because a
        //    caller asked for numbers — every encode above is enqueue-only by
        //    design (decision #15).
        frame.commit()?;

        let logits = slots.0[out.0 as usize].ok_or_else(|| Fault::Unbound {
            what: format!(
                "value {}, the out seam, which the carve gave no rectangle",
                out.0
            ),
        })?;
        if logits.dtype != Dtype::Bf16 {
            return Err(Fault::Unbound {
                what: format!(
                    "an out seam landed as {:?}, which this shell cannot read back",
                    logits.dtype
                ),
            });
        }
        let width = logits.width as usize;
        let mut taken = vec![Vec::new(); lanes.len()];
        let mut raw = vec![0u8; width * 2];
        for row in composition.lanes() {
            let last = row.row_offset + row.rows - 1;
            arena.read_view(
                handles,
                logits.buf,
                u64::from(last) * width as u64 * 2,
                &mut raw,
            )?;
            taken[row.source as usize] = raw
                .chunks_exact(2)
                .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]])))
                .collect();
        }

        // 10. Only now: the fire happened, so the sequences are longer. Only
        //     the slots this shell counts for — a caller that owns the page
        //     table owns the count too.
        for (seat, table) in seats.iter().zip(&tables) {
            if table.is_empty()
                && let Some(slot) = held.get_mut(seat.slot as usize)
            {
                *slot = seat.have + seat.rows;
            }
        }

        // 11. And the fire's handles go with the fire. Everything minted
        //     since the load's seal — the arena's rectangles, the pools'
        //     views, the staged input vectors, every windowed cut — named
        //     bytes that this fire's carve placed and the next fire's carve
        //     will place differently. Keeping them would be a slow leak of
        //     retained buffers and a live handle resolvable against the
        //     wrong offset.
        handles.rewind();
        Ok(taken)
    }
}

/// One bf16, widened.
///
/// The top sixteen bits of an f32 and nothing else — bf16 exists to make
/// this the whole conversion. Reading one as an f16 instead is the mistake
/// the loader's own docs name: same width, different exponent, and 0.0385
/// becomes 1.6e-12 without crashing or warning.
fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}
