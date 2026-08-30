//! The guest-program plane, CUDA half: ETA at the fire's boundary.
//!
//! **WHAT A GUEST PROGRAM IS.** A tensor program the guest authored, planned
//! by `eta-compiler` on the host into a `LaunchPackage` plus a table of
//! emitted CUDA sources, and executed here against a set of channel rings.
//! Design §9 places it at the boundary of the model fire and nowhere else:
//! a prologue before the immutable graph, an epilogue after it, and never a
//! hook inside — guest code cannot enter a graph that is recorded once and
//! replayed forever.
//!
//! ```text
//! engine/src/program/        this module
//! -------------------        -----------
//! adopt_launch_package  ->   compile.rs   NVRTC, cubins, the three cache tiers
//! ChannelState, rings   ->   launch.rs    device cells, the lane-table bytes
//! readiness, step       ->   session.rs   one instance's lifetime and its fire
//! ```
//!
//! **THE HOST HALF IS THE GOLDEN AND IT IS NOT A MOCK.** `eta_exec` is
//! a complete interpreter of the same launch package: same ops, same channel
//! semantics, same pass-atomic commit. This half is the subject, and
//! `tests/program_parity.rs` runs both over the same programs with the same
//! inputs and demands byte-identical rings and identical outcomes, fire for
//! fire. Bit-for-bit reproducibility is the channel plane's first-class
//! contract, which is why `--fmad=false --prec-div=true --prec-sqrt=true` is
//! a determinism clause in [`compile`] rather than a tuning flag.
//!
//! **THE SEAM IS WIRED** (`palo B2`). A [`Step`]'s
//! [`Attachment`](engine::fire::Attachment)s name a lane and a
//! bound instance, and [`Shell::fire_attached`](crate::Shell::fire_attached)
//! runs them at the two points design §9 allows:
//!
//! ```text
//! gate       Plane::ready over EVERY attached instance      ← nothing has launched
//! prologue   Plane::fire, per Boundary::Prologue attachment
//! forward    compose -> windows -> walk / graph replay
//! bind       Plane::bind_intrinsic: the lane's logits ROW
//! epilogue   Plane::fire, per Boundary::Epilogue attachment
//! ```
//!
//! The gate is why [`Plane::ready`] exists. An epilogue fires after the
//! forward has written the lane's KV, so a readiness refusal discovered there
//! is a fire the caller cannot retry — the tokens are in the cache and the
//! guest's pass never happened. Asking every attached instance first turns a
//! blocked guest into a refusal the run-ahead retries for free.

pub mod compile;
/// The pinned mirror and counters of a channel with a host end — the
/// allocation that makes a guest round trip cost zero CUDA calls.
pub mod endpoint;
pub mod launch;
pub mod ports;
pub mod session;
pub mod wave;

use std::collections::BTreeMap;

use engine::program::ProgramRegistration;
use eta_exec::{Boundaries, ExecPlan, Extents, Versions, adopt_launch_package_with};
use eta_ir::registry::GeometryClass;

use crate::device::Context;
use crate::error::{Fault, Result};

pub use compile::{Cache, Compiled, Disk, Module, Region, Stage, Target};
pub use endpoint::Endpoint;
pub use launch::{ChannelShape, Cursor, Prepared, Rings, describe_values, scratch_bytes};
pub use ports::Envelope;
pub use session::{Fired, Launched, Session, seeds_of};
pub use wave::Wave;

/// One registered program: what the host planned, and what compiled from it.
#[derive(Debug)]
pub struct Program {
    /// The plane's own handle for it.
    pub id: u64,
    /// The host's content hash, which is what a re-registration is recognised
    /// by.
    pub hash: u64,
    /// The adopted launch package: the channel declarations, the stages, and
    /// the derived per-stage value indexes a fire reads.
    pub plan: ExecPlan,
    /// The compiled regions, one table per stage.
    pub compiled: Compiled,
    /// **THE LANE BATCHES**, one per [`Extents`] any instance of this program
    /// has bound at.
    ///
    /// The fire-path buffers used to be a session's, which is what made a
    /// boundary's sixty-four epilogues sixty-four one-block launches
    /// ([`Prepared`]'s own note, and the 25% of GPU time it names). They are
    /// the PROGRAM's now, so attachments that share a program and its extents
    /// share a launch with a block apiece.
    ///
    /// A `Vec` rather than a map because it holds one entry in every shape
    /// this workspace serves — an epilogue binds at `sampled_rows` and every
    /// other role at one — and two or three in the ones it does not.
    batches: Vec<Batch>,
}

/// **ONE PROGRAM'S FIRE-PATH BUFFERS FOR ONE SET OF EXTENTS**, one entry per
/// stage.
///
/// `None` for a stage with nothing to launch — the adapter prologue is
/// exactly that, its one region being a sink the model fire reads out of the
/// plan rather than a body anyone compiled.
#[derive(Debug)]
struct Batch {
    extents: Extents,
    stages: Vec<Option<Prepared>>,
}

impl Program {
    /// The batch cut for `extents`, cutting it if this program has never seen
    /// them.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when a value's shape does not resolve against
    /// `extents` or the scratch does not fit, and whatever the allocations
    /// said.
    fn batch(&mut self, extents: Extents, lanes: u32) -> Result<&mut Batch> {
        if let Some(at) = self
            .batches
            .iter()
            .position(|batch| batch.extents == extents)
        {
            return Ok(&mut self.batches[at]);
        }
        let shapes: Vec<ChannelShape> = self
            .plan
            .package
            .channels
            .iter()
            .map(ChannelShape::of)
            .collect();
        let mut stages = Vec::with_capacity(self.compiled.plans.len());
        for (index, stage_plan) in self.compiled.plans.iter().enumerate() {
            let launches = self
                .compiled
                .stages
                .get(index)
                .is_some_and(|stage| !stage.regions.is_empty());
            stages.push(if launches {
                Some(Prepared::build(stage_plan, &shapes, extents, lanes)?)
            } else {
                None
            });
        }
        self.batches.push(Batch { extents, stages });
        Ok(self.batches.last_mut().expect("just pushed"))
    }
}

/// The shell's guest-program plane: the compile cache, the registered
/// programs, and the bound instances.
///
/// **IT NEEDS A DEVICE, NOT A MODEL.** Everything here works against a bound
/// [`Context`] and nothing else — no weights, no arena, no `CompiledModel`. That is
/// what lets the parity test drive it with a context and a golden trace, and
/// it is also the honest shape: a guest program's channels are its own, and
/// the model fire meets it only at the two attachment points design §9 names.
#[derive(Debug)]
pub struct Plane {
    cache: Cache,
    programs: BTreeMap<u64, Program>,
    by_hash: BTreeMap<u64, u64>,
    instances: BTreeMap<u64, Bound>,
    next_program: u64,
    next_instance: u64,
    /// **THIS BOUNDARY'S STAGED FIRES**, `(program, extents, instance)` in
    /// staging order.
    ///
    /// The grouping key and the group, in one list: [`Plane::fly`] walks it
    /// once to decide which attachments ride which launch. Emptied at
    /// [`Plane::land`].
    staged: Vec<(u64, Extents, u64)>,
    /// **ONE BOUNDARY'S CONTROL PLANE, SHARED BY EVERY FIRE IN IT.**
    ///
    /// The plane's and not a session's, because the whole point of it is that
    /// the lanes of a boundary are CONTIGUOUS — see [`wave`]. Long-lived so
    /// that its device arena is allocated at a high-water mark rather than
    /// per boundary; empty between batches.
    wave: Wave,
}

impl Default for Plane {
    /// A plane that caches no cubin. Where they are kept is the DEPLOYMENT's
    /// answer, arriving on `Boot::program_cache_dir` (article 9: shells read
    /// no environment) — so a plane made without one stores nothing rather
    /// than going looking, and pays NVRTC instead. Every failure of that cache
    /// is a miss and never an error.
    fn default() -> Plane {
        Plane::new(Disk::disabled())
    }
}

impl Plane {
    /// A plane whose cubins are cached in `disk`.
    #[must_use]
    pub fn new(disk: Disk) -> Plane {
        Plane {
            cache: Cache::new(disk),
            programs: BTreeMap::new(),
            by_hash: BTreeMap::new(),
            instances: BTreeMap::new(),
            next_program: 1,
            next_instance: 1,
            staged: Vec::new(),
            wave: Wave::default(),
        }
    }

    /// What the compile tiers have been doing.
    #[must_use]
    pub const fn stats(&self) -> eta_exec::CacheStats {
        self.cache.stats()
    }

    /// Adopt a registration and compile its regions, answering the program id.
    ///
    /// A program already registered under the same hash answers its existing
    /// id and compiles nothing — the memory tier's whole job, and the one
    /// claim a cache test can make, since an absence has no output.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] when the package is not adoptable (no stages, or plans
    /// and stages that are not parallel), and [`Fault::Compile`] when a region
    /// does not compile here. The compile taxonomy is the point: a
    /// deterministic refusal is remembered and answers the next attempt
    /// immediately; a retryable one is not.
    pub fn register(
        &mut self,
        context: &Context,
        registration: &ProgramRegistration,
    ) -> Result<u64> {
        if let Some(&existing) = self.by_hash.get(&registration.program_hash) {
            return Ok(existing);
        }
        // `Boundaries::CUDA`, not the default: the vocabulary of semantic
        // library boundaries a backend implements is a fact about the backend,
        // and this one answers `envelope_dot`, `lora` and `attn_page_mask`
        // where the Metal half answers `metal.identity` and `metal.discard`.
        let plan = adopt_launch_package_with(registration.launch.clone(), Boundaries::CUDA)?;
        let compiled = self.cache.compile(
            registration.program_hash,
            &plan,
            &registration.emitted_kernels,
            Versions::from_compiler(registration.emitter_version),
            Target::of(context)?,
        )?;

        let id = self.next_program;
        self.next_program += 1;
        self.programs.insert(
            id,
            Program {
                id,
                hash: registration.program_hash,
                plan,
                compiled,
                batches: Vec::new(),
            },
        );
        self.by_hash.insert(registration.program_hash, id);
        Ok(id)
    }

    /// What was registered under `id`.
    #[must_use]
    pub fn program(&self, id: u64) -> Option<&Program> {
        self.programs.get(&id)
    }

    /// Bind an instance of `program_id`: allocate its rings, carve every
    /// stage's fire-path buffers, and seed the channels the guest seeded.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown program, a seed naming a channel the
    /// instance does not carry, or a seed of the wrong width; and whatever the
    /// allocations said.
    pub fn bind(
        &mut self,
        program_id: u64,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
        geometry: GeometryClass,
        adopted: &[Option<std::sync::Arc<Endpoint>>],
        ids: &[u64],
    ) -> Result<u64> {
        let program = self
            .programs
            .get_mut(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        // **THE FIRE-PATH BUFFERS ARE CUT HERE, AT BIND** (Build log 15), and
        // the extents are what they are cut against: a guess zero-fills
        // silently, so a batch that does not resolve is a refusal at the door
        // rather than three rows of zeroes at the first fire. One lane is
        // enough for the validation; a boundary grows it to what it brings.
        program.batch(extents, 1)?;
        let program = &*program;
        let endpoints = endpoints_for(&program.plan, adopted)?;
        let held = endpoints.clone();
        let session = match Session::bind(
            &program.compiled,
            &program.plan,
            seeds,
            extents,
            endpoints,
        ) {
            Ok(session) => session,
            Err(why) => {
                // Same rule as a refused `endpoints_for`: an instance that did
                // not bind holds no seat.
                release_seats(&held);
                return Err(why);
            }
        };
        let id = self.next_instance;
        self.next_instance += 1;
        self.instances.insert(
            id,
            Bound {
                program_id,
                session,
                endpoints: held,
                geometry,
                ids: ids.to_vec(),
            },
        );
        Ok(id)
    }

    /// What instance `id`'s descriptor ports resolve to right now, or `None`
    /// when its class says the host resolves them.
    ///
    /// **THE CLASS IS THE STATEMENT OF WHO RESOLVES, SO THE CLASS IS THE
    /// GATE.** A [`GeometryClass::Host`] instance's fire reads its token ids,
    /// its positions and its readable extent out of the SUBMISSION — the
    /// runtime folded them host-side and stated them — and reading the rings
    /// as well would be reading a second opinion about values that are
    /// already decided. A [`GeometryClass::DecodeEnvelope`] instance's
    /// submission carries placeholders for exactly those three, because the
    /// runtime could not know them, and this is where they come from.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, one whose program is gone,
    /// and whatever [`ports::resolve`] said.
    pub fn envelope(&self, id: u64) -> Result<Option<Envelope>> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        if bound.geometry == GeometryClass::Host {
            return Ok(None);
        }
        let program = self.programs.get(&bound.program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "instance {id} names program {}, which is gone",
                    bound.program_id
                ),
            )
        })?;
        bound
            .session
            .envelope(&program.plan, bound.geometry)
            .map(Some)
    }

    /// The class instance `id` was bound in.
    #[must_use]
    pub fn geometry_of(&self, id: u64) -> Option<GeometryClass> {
        self.instances.get(&id).map(|bound| bound.geometry)
    }

    /// One instance's rings and cursors, for publishing into and taking out of.
    #[must_use]
    pub fn instance(&self, id: u64) -> Option<&Session> {
        self.instances.get(&id).map(|bound| &bound.session)
    }

    /// One instance, mutably.
    pub fn instance_mut(&mut self, id: u64) -> Option<&mut Session> {
        self.instances.get_mut(&id).map(|bound| &mut bound.session)
    }

    /// Point one intrinsic of instance `id` at a device buffer.
    ///
    /// The runtime's attachment seam (design §9): a program that reads
    /// `IntrinsicId::Logits` is pointed at the readout buffer a model fire
    /// produced, and one that reads none never calls this.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance or an intrinsic past the
    /// side tables' pitch.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_intrinsic(
        &mut self,
        id: u64,
        intrinsic: eta_ir::op::IntrinsicId,
        base: u64,
        storage: u32,
        width: u32,
        row_stride: u32,
        row_offset: u32,
    ) -> Result<()> {
        self.instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?
            .session
            .bind_intrinsic(intrinsic, base, storage, width, row_stride, row_offset)
    }

    /// **HOW MANY SCORE PLANES INSTANCE `id` DECLARED**, or `None` for one
    /// that reads no score rectangle at all.
    ///
    /// The row count is the PROGRAM's — a ceiling it states the way
    /// `hidden(width)` states its width — and the shell is what refuses a
    /// claim larger than the load exports (`eta_ir::validate`'s type rule
    /// says so and can only check the width, because the plane count is not
    /// in the profile). This is the reading that lets it: a rectangle bound
    /// at the slab's pitch and read for more rows than a lane's block holds
    /// walks into the NEXT lane's mass, which is the one failure the whole
    /// per-lane addressing exists to prevent, and it is silent.
    #[must_use]
    pub fn declared_score_planes(&self, id: u64) -> Option<u32> {
        let bound = self.instances.get(&id)?;
        let program = self.programs.get(&bound.program_id)?;
        program
            .plan
            .package
            .values
            .iter()
            .filter(|value| value.intrinsic == Some(eta_ir::op::IntrinsicId::AttnScore))
            .filter_map(|value| value.shape.first().copied())
            .max()
    }

    /// The first channel of instance `id` whose declared requirement a fire
    /// right now would not meet, or `None` when it is ready.
    ///
    /// **THE GATE, ASKED BEFORE THE MODEL FIRE.** See
    /// [`Session::blocked_channel`]: an epilogue attachment is fired after the
    /// forward has written the lane's KV, so the caller has to know BEFORE it
    /// launches anything whether every attached instance can commit.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, or one whose program is
    /// gone.
    pub fn ready(&self, id: u64) -> Result<Option<u32>> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let program = self.programs.get(&bound.program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "instance {id} names program {}, which is gone",
                    bound.program_id
                ),
            )
        })?;
        Ok(bound.session.blocked_channel(&program.plan))
    }

    /// **THE CALLER'S PREDICTION AGAINST THIS ENGINE'S, IF THEY DIFFER**
    /// (alto design §1 article 3).
    ///
    /// A caller that mints tickets from its own monotone counters and an
    /// engine that mints them from its own are two counters that must agree,
    /// and F2b — where the fire stops synchronising and the caller's is the
    /// only one that can be right — is where they have to. This is the check
    /// that says so while both still exist: the caller states where it
    /// believes each channel stands, the engine compares it against the
    /// prediction its own fire would use, and a disagreement is named rather
    /// than silently resolved in the engine's favour.
    ///
    /// A prediction of [`Ticket::NONE`](engine::Ticket::NONE) is
    /// no claim about that end of the ring and is not compared. A channel the
    /// caller names that this instance does not carry IS a disagreement: it
    /// predicted about somebody else's ring.
    #[must_use]
    pub fn disagreeing_ticket(
        &self,
        id: u64,
        tickets: &[engine::Ticket],
    ) -> Option<String> {
        let bound = self.instances.get(&id)?;
        for ticket in tickets {
            let Some(dense) = bound.ids.iter().position(|held| *held == ticket.channel) else {
                return Some(format!(
                    "instance {id} predicted about channel {}, which it does not carry",
                    ticket.channel
                ));
            };
            let cursor = bound.session.cursor(dense as u32)?;
            let stated = |claim: u64, held: u64, end: &str| {
                (claim != engine::Ticket::NONE && claim != held).then(|| {
                    format!(
                        "instance {id}'s channel {} stands at {end} {held} and the caller \
                         predicted {claim}",
                        ticket.channel
                    )
                })
            };
            if let Some(why) = stated(ticket.expected_head, cursor.head, "head") {
                return Some(why);
            }
            if let Some(why) = stated(ticket.expected_tail, cursor.tail, "tail") {
                return Some(why);
            }
        }
        None
    }

    /// Fire instance `id` once: readiness, then every stage's regions, then
    /// one commit.
    ///
    /// **A BOUNDARY OF EXACTLY ONE LANE.** The three control kernels launch
    /// with `lane_count = 1` and the program's batch with `grid = 1`: a
    /// standalone fire is not a different protocol, only a smaller wave.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, and whatever the launches
    /// said. A program that blocks or refuses is a [`Fired`], not an error.
    pub fn fire(&mut self, context: &Context, id: u64) -> Result<Fired> {
        match self.stage(id)? {
            Launched::Airborne => {
                self.fly(context)?;
                self.land(context)?;
                context.synchronize()?;
                self.settle_launched(id)
            }
            // A refusal never reached the mint, so it staged no lane and
            // there is nothing to take back.
            Launched::Refused(fired) => Ok(fired),
        }
    }

    /// **THE STAGING HALF OF A FIRE**: mint instance `id`'s tickets, slot
    /// lists and three lane records into the plane's [`Wave`], and enqueue
    /// nothing at all.
    ///
    /// The program is looked up THROUGH the instance's own record rather than
    /// taken as an argument, so a caller cannot fire one program's stages
    /// against another's rings — which would index scratch, descriptors and a
    /// channel table sized for someone else, and would not fault.
    ///
    /// A [`Launched::Airborne`] has taken a lane of the wave and owes, in
    /// order: [`Plane::fly`], [`Plane::land`], a wait, and
    /// [`Plane::settle_launched`].
    ///
    /// # Errors
    ///
    /// As [`Plane::fire`].
    pub fn stage(&mut self, id: u64) -> Result<Launched> {
        let Plane {
            programs,
            instances,
            wave,
            staged,
            ..
        } = self;
        let bound = instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let program_id = bound.program_id;
        let program = programs.get(&program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!("instance {id} names program {program_id}, which is gone"),
            )
        })?;
        let launched = bound
            .session
            .stage(&program.compiled, &program.plan, wave)?;
        if matches!(launched, Launched::Airborne) {
            staged.push((program_id, bound.session.extents(), id));
        }
        Ok(launched)
    }

    /// **THE BOUNDARY, ON THE STREAM**: the wave's copy and its
    /// `pull_validate`, then every staged fire's regions, then the wave's
    /// `commit_bump` and `scatter_publish`.
    ///
    /// # The one thing this does that a per-fire launch could not
    ///
    /// The staged fires are grouped by `(program, extents)` — which is the
    /// key a [`Prepared`] batch is cut against — and each group's regions
    /// launch ONCE with a block per lane. Sixty-four decode samplers were
    /// sixty-four `cuLaunchKernel`s of one block each, 42.5 µs apiece and
    /// 25% of all GPU time in an nsys capture, every one of them reading half
    /// a megabyte of logits at one SM's share of the bandwidth while the
    /// other 141 idled. They are one launch of sixty-four blocks now.
    ///
    /// Order is preserved where it means something and nowhere else: within a
    /// group, a stage that reads what the previous one wrote is still a later
    /// launch on the same stream; between groups there is nothing to order,
    /// because two lanes of a boundary share no ring (a shared ring flushes
    /// the batch before it can be staged).
    ///
    /// Idempotent in the sense that matters: a boundary with nothing staged
    /// launches nothing.
    ///
    /// # Errors
    ///
    /// Whatever the allocations, the copies and the launches said.
    pub fn fly(&mut self, context: &Context) -> Result<()> {
        if self.staged.is_empty() {
            return Ok(());
        }
        self.wave.fly(context)?;
        let stream = context.stream();
        let Plane {
            programs,
            instances,
            staged,
            ..
        } = self;

        // Grouped in FIRST-SEEN order, so a boundary's launches come out in
        // the order its attachments were staged — which is what an error
        // message and a profile both read as "the order the shell asked for".
        let mut groups: Vec<(u64, Extents, Vec<u64>)> = Vec::new();
        for (program_id, extents, instance) in staged.iter() {
            match groups
                .iter_mut()
                .find(|(pid, ext, _)| pid == program_id && ext == extents)
            {
                Some((_, _, members)) => members.push(*instance),
                None => groups.push((*program_id, *extents, vec![*instance])),
            }
        }

        for (program_id, extents, members) in groups {
            let program = programs.get_mut(&program_id).ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("a staged fire names program {program_id}, which is gone"),
                )
            })?;
            // **A GROUP THAT DOES NOT FIT IS SPLIT, NOT REFUSED.** The
            // scratch ceiling is a wave's — 512 MiB over `lanes * stride` —
            // so a program whose stride is generous runs out of room before
            // the boundary does. Narrowing the launch costs one more kernel
            // and loses nothing; refusing would kill a boundary that has
            // already minted.
            let ceiling = program
                .batch(extents, 1)?
                .stages
                .iter()
                .flatten()
                .map(Prepared::lane_ceiling)
                .min()
                .unwrap_or(u32::MAX)
                .max(1) as usize;
            for chunk in members.chunks(ceiling) {
                Plane::fly_one(programs, instances, program_id, extents, chunk, stream)?;
            }
        }
        Ok(())
    }

    /// One launch's worth of a group: take every member's lane, stage the
    /// batch's tables, and launch each region once with a block per lane.
    fn fly_one(
        programs: &mut BTreeMap<u64, Program>,
        instances: &mut BTreeMap<u64, Bound>,
        program_id: u64,
        extents: Extents,
        members: &[u64],
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        {
            let program = programs.get_mut(&program_id).ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("a staged fire names program {program_id}, which is gone"),
                )
            })?;
            let lanes = u32::try_from(members.len()).unwrap_or(u32::MAX);
            let batch = program.batch(extents, lanes)?;
            for prepared in batch.stages.iter_mut().flatten() {
                prepared.begin(extents, lanes)?;
            }
            for instance in members {
                let bound = instances.get_mut(instance).ok_or_else(|| {
                    Fault::program(
                        "program::plane",
                        format!("a staged fire names instance {instance}, which is gone"),
                    )
                })?;
                bound.session.take_lane(&mut batch.stages)?;
            }
            for prepared in batch.stages.iter_mut().flatten() {
                prepared.commit_lanes(stream)?;
            }
        }
        {
            let program = programs.get(&program_id).ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("a staged fire names program {program_id}, which is gone"),
                )
            })?;
            // **THE STAGES LAUNCH UNCONDITIONALLY** (survey §7 I4). Every
            // stage launches whatever the pull decided: the emitted kernel's
            // first three lines read its lane's commit word and return when
            // it is clear (`fused_block1.cuh`), so a refused lane costs what a
            // running one does and writes only into the pending cell nobody
            // can address — and its neighbours in the same launch are
            // untouched, because the word is per lane.
            //
            // NO SYNCHRONIZE BETWEEN STAGES. One stream is the ordering.
            for (index, stage) in program.compiled.stages.iter().enumerate() {
                let Some(prepared) = program
                    .batches
                    .iter()
                    .find(|batch| batch.extents == extents)
                    .and_then(|batch| batch.stages.get(index))
                    .and_then(Option::as_ref)
                else {
                    // A stage with nothing to launch needs no buffers and no
                    // launch. Not an empty case: the adapter prologue is
                    // exactly this shape.
                    continue;
                };
                for region in stage.regions.iter() {
                    prepared.launch_region(region, stream)?;
                }
            }
        }
        Ok(())
    }

    /// **THE WAVE'S BUMP AND ITS PUBLICATION**, once for the whole batch —
    /// see [`wave::Wave::land`]. Clears the wave for the next boundary.
    ///
    /// # Errors
    ///
    /// Whatever the launches said.
    pub fn land(&mut self, context: &Context) -> Result<()> {
        self.staged.clear();
        self.wave.land(context)
    }

    /// **How many lanes are staged and not yet flown.**
    #[must_use]
    pub fn staged(&self) -> usize {
        self.staged.len().max(self.wave.staged())
    }

    /// **THROW AWAY A BATCH THAT WILL NEVER FLY.**
    ///
    /// The host half only, and there is no device half to throw: a staged
    /// lane has touched no stream. What this exists for is the one way lanes
    /// can be left in the wave — a fault raised BETWEEN a boundary's first
    /// stage and its landing, which unwinds past both — after which the
    /// sessions that staged them are wedged on their own pending mints
    /// anyway. Clearing keeps the next boundary from launching a control
    /// kernel over rows belonging to fires nobody will settle.
    pub fn abandon_wave(&mut self) {
        self.staged.clear();
        self.wave.clear();
    }


    /// **The verdict half** — see [`Session::settle_launched`].
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, or for a prediction the
    /// gate approved and the ring denied.
    pub fn settle_launched(&mut self, id: u64) -> Result<Fired> {
        let bound = self
            .instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        bound.session.settle_launched()
    }

    /// **Which shared rings instance `id` holds** — see
    /// [`Session::shared_rings`]. An unknown instance holds none.
    pub fn shared_rings(&self, id: u64) -> Vec<usize> {
        self.instances
            .get(&id)
            .map(|bound| bound.session.shared_rings().collect())
            .unwrap_or_default()
    }

    /// Drop instance `id`, freeing its rings and its stage buffers.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when there is no such instance — closing twice is a
    /// caller's bug, not a no-op.
    pub fn close_instance(&mut self, id: u64) -> Result<()> {
        let bound = self
            .instances
            .remove(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        // The seats go back; the RINGS do not necessarily go with them. A
        // shared ring is an `Arc` the engine's channel table also holds, so it
        // survives this drop and stays addressable by the attachments that are
        // still open — which is the whole point of a ring that belongs to the
        // channel (`release_seats`' own note).
        release_seats(&bound.endpoints);
        Ok(())
    }

    /// Drop program `id`, unloading this plane's share of its modules.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when there is no such program, or when instances are
    /// still bound to it: unloading a `CUmodule` under a live instance is a
    /// launch into freed machine code.
    pub fn close_program(&mut self, id: u64) -> Result<()> {
        let program = self
            .programs
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {id}")))?;
        let hash = program.hash;
        let bound = self
            .instances
            .values()
            .filter(|bound| bound.program_id == id)
            .count();
        if bound != 0 {
            return Err(Fault::program(
                "program::plane",
                format!(
                    "program {id} still has {bound} instance(s) bound, and unloading a \
                     CUmodule under a live one is a launch into freed machine code"
                ),
            ));
        }
        self.cache.forget(hash);
        self.by_hash.remove(&hash);
        self.programs.remove(&id);
        Ok(())
    }
}

/// One bound instance and the program it is an instance OF.
///
/// A struct rather than two maps: an instance whose program id is looked up
/// separately is an instance that can be fired against the wrong plan, and
/// nothing on the device would fault.
#[derive(Debug)]
struct Bound {
    program_id: u64,
    session: Session,
    /// **This instance's share of every ring it bound**, kept so that closing
    /// it gives back the seats its shared rings hold — see
    /// [`release_seats`]. Cloned `Arc`s, so holding them here is also what
    /// keeps a shared ring alive for exactly as long as some attachment can
    /// still address it.
    endpoints: Vec<Option<std::sync::Arc<Endpoint>>>,
    /// How much of the fire geometry this instance's descriptor resolves on
    /// the device — the caller's claim at bind, kept because it is what
    /// [`Plane::envelope`] gates on. A class the load does not serve never
    /// reaches here: [`Cuda::bind_instance`](crate::api::Cuda) refuses it by
    /// name.
    geometry: GeometryClass,
    /// This instance's channels by the CALLER's id, in the package's
    /// declaration order — `InstanceBinding::channels` verbatim. The only map
    /// between the ids a caller predicts about and the dense slots this plane
    /// counts, and what [`Plane::disagreeing_ticket`] reads.
    ids: Vec<u64>,
}

/// **EVERY CHANNEL'S HOST END, ADOPTED OR OPENED.**
///
/// A channel with no host end has no endpoint: its cells never leave the
/// device, so there is no pinned mirror for anyone to address and no
/// prediction anyone outside this shell could be wrong about.
///
/// A channel that DOES have one takes the caller's when the caller already
/// registered it — the runtime registers its channels before it binds the
/// instance that shares them, and **the mirror the guest writes through has
/// to be the mirror the pull reads**, or the guest's cells reach nobody — and
/// gets a fresh one otherwise, which is the shape a caller driving
/// `bind_instance` directly is in.
///
/// An adopted endpoint is checked against the declaration it is bound to. A
/// mirror cut for a different cell width or a different ring addresses the
/// wrong bytes on the first pull and nothing faults; it serves the wrong
/// token, quietly, forever.
///
/// # Errors
///
/// [`Fault::Program`] for an adopted endpoint that does not match the
/// declaration it is being bound to, and whatever the pinned allocation said.
fn endpoints_for(
    plan: &ExecPlan,
    adopted: &[Option<std::sync::Arc<Endpoint>>],
) -> Result<Vec<Option<std::sync::Arc<Endpoint>>>> {
    let mut seated: Vec<std::sync::Arc<Endpoint>> = Vec::new();
    match gather_endpoints(plan, adopted, &mut seated) {
        Ok(endpoints) => Ok(endpoints),
        Err(why) => {
            // A bind that does not happen holds no seat. The seats taken
            // before the refusal are given back here rather than left for the
            // eight-seat bound to trip over on the caller's next attempt.
            for endpoint in &seated {
                endpoint.detach();
            }
            Err(why)
        }
    }
}

fn gather_endpoints(
    plan: &ExecPlan,
    adopted: &[Option<std::sync::Arc<Endpoint>>],
    seated: &mut Vec<std::sync::Arc<Endpoint>>,
) -> Result<Vec<Option<std::sync::Arc<Endpoint>>>> {
    use eta_ir::container::HostRole;

    let mut endpoints = Vec::with_capacity(plan.package.channels.len());
    for (dense, declared) in plan.package.channels.iter().enumerate() {
        // **A `None` ROLE NO LONGER MEANS "NO ENDPOINT"** (design §5). It used
        // to: a channel whose cells never leave the device had its ring cut
        // inside this session, which is right for a ring one pass owns and
        // silently wrong for the one design §5 names — a device-only ring
        // shared by up to eight attachments, one putting and another taking.
        // Two sessions cut two slabs and the handoff crossed nothing.
        //
        // So the ring belongs to the channel now, for every role, and the
        // adoption below is the whole of it: a `None` channel this engine
        // registered offers a shared endpoint and every attachment takes the
        // same one. What stays session-local is the STAGING a host-visible
        // channel crosses through, which is `Rings::allocate`'s to cut.
        //
        // A `None` channel this engine never registered still gets `None` —
        // the caller driving `bind_instance` directly, whose ring nobody else
        // can name.
        if declared.host_role == HostRole::None
            && adopted.get(dense).and_then(Option::as_ref).is_none()
        {
            endpoints.push(None);
            continue;
        }
        let numel = declared
            .shape
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>()
            .max(1);
        // **A SHARED RING IS CUT AT ITS NATIVE WIDTH, NOT ITS WIRE WIDTH.**
        // `wire_bytes` sizes the pinned MIRROR, which is the width a guest
        // reads a cell at — bit-packed for a bool channel. A device-only ring
        // has no guest and no mirror; what its endpoint owns is the DEVICE
        // slab, which the emitted kernels index one byte per bool lane. Cut at
        // the wire width, a shared bool ring would be an eighth of the slab
        // every reader addresses.
        let wire = if declared.host_role == HostRole::None {
            super::program::launch::native_cell_bytes(
                eta_exec::concrete_dtype(declared.dtype),
                numel,
            )
        } else {
            eta_exec::wire_cell_bytes(eta_exec::concrete_dtype(declared.dtype), numel)
        };
        let wire_bytes = u32::try_from(wire).map_err(|_| {
            Fault::program(
                "program::plane",
                format!("channel {dense}'s wire cell is wider than a u32 counts"),
            )
        })?;
        let capacity = declared.capacity.max(1);
        match adopted.get(dense).and_then(Option::as_ref) {
            Some(endpoint) => {
                if endpoint.role() != declared.host_role
                    || endpoint.wire_bytes() != wire_bytes
                    || endpoint.cap1() != capacity + 1
                {
                    return Err(Fault::program(
                        "program::plane",
                        format!(
                            "channel {dense} is declared {:?} with a {wire_bytes}-byte cell \
                             and a ring of {}, and the endpoint offered for it is {:?} with \
                             a {}-byte cell and a ring of {}: a mirror cut for a different \
                             shape addresses the wrong bytes on the first pull and nothing \
                             faults",
                            declared.host_role,
                            capacity + 1,
                            endpoint.role(),
                            endpoint.wire_bytes(),
                            endpoint.cap1(),
                        ),
                    ));
                }
                // **ONE OF THE RING'S EIGHT SEATS** (design §5's bound). Only
                // a shared ring counts them: a host-visible endpoint is
                // adopted by one instance at a time by construction, and it is
                // the device-only ring that two passes deliberately share.
                if endpoint.role() == HostRole::None {
                    seated.push(endpoint.clone());
                    endpoint.attach()?;
                }
                endpoints.push(Some(endpoint.clone()));
            }
            None => endpoints.push(Some(std::sync::Arc::new(Endpoint::open(
                declared.host_role,
                wire_bytes,
                capacity,
            )?))),
        }
    }
    Ok(endpoints)
}

/// **Give back every seat this instance's channels hold** — the inverse of the
/// `attach` in [`endpoints_for`].
///
/// # Who owns a shared ring's lifetime (design §5)
///
/// The ring itself is an `Arc<Endpoint>` held by the engine's channel table
/// (`Cuda::channels`, keyed by the id its registration minted) and by every
/// session bound to it, so it is freed when the LAST holder drops — which is
/// after `close_channel` has removed the engine's entry AND every attachment
/// has been closed, in either order. That is the property a shared ring needs
/// and a per-session ring never had: it outlives any one instance's close, so
/// a pipeline may close its prefill pass while its decode passes are still
/// reading the ring the prefill filled.
///
/// The SEAT is separate bookkeeping and is given back here, so that a pipeline
/// which closes and rebuilds passes does not walk its ring up to the eight-seat
/// bound one rebuild at a time.
fn release_seats(endpoints: &[Option<std::sync::Arc<Endpoint>>]) {
    use eta_ir::container::HostRole;

    for endpoint in endpoints.iter().flatten() {
        if endpoint.role() == HostRole::None {
            endpoint.detach();
        }
    }
}
