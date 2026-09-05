//! The guest-program plane, CUDA half: a guest-authored tensor program,
//! planned by `eta-compiler`, executed against channel rings at the boundary
//! of the model fire only (a prologue before the graph, an epilogue after,
//! never a hook inside).

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
    /// The lane batches, one per [`Extents`] any instance of this program has
    /// bound at. Owned by the program (not the session) so attachments that
    /// share a program and its extents share one launch with a block apiece.
    /// A `Vec` rather than a map since it typically holds one or two entries.
    batches: Vec<Batch>,
}

/// One program's fire-path buffers for one set of extents, one entry per
/// stage. `None` for a stage with nothing to launch (e.g. the adapter
/// prologue).
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
/// Works against a bound [`Context`] and nothing else — no weights, no
/// arena, no `CompiledModel` — which is what lets the parity test drive it
/// with just a context and a golden trace.
#[derive(Debug)]
pub struct Plane {
    cache: Cache,
    programs: BTreeMap<u64, Program>,
    by_hash: BTreeMap<u64, u64>,
    instances: BTreeMap<u64, Bound>,
    next_program: u64,
    next_instance: u64,
    /// This boundary's staged fires, `(program, extents, instance)` in
    /// staging order. [`Plane::fly`] walks it once to group launches.
    /// Emptied at [`Plane::land`].
    staged: Vec<(u64, Extents, u64)>,
    /// One boundary's control plane, shared by every fire in it — see
    /// [`wave`]. Long-lived so its device arena is a high-water mark rather
    /// than per-boundary; empty between batches.
    wave: Wave,
    /// A tensor-parallel follower's plane: sessions bind as shadows of
    /// rank 0's (see [`Session`]).
    shadow: bool,
}

impl Default for Plane {
    /// A plane that caches no cubin: pays NVRTC instead. A cache failure is
    /// always a miss, never an error.
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
            shadow: false,
        }
    }

    /// Makes every session bound from now on a tensor-parallel follower's:
    /// its host-ended rings are rank 0's endpoints, read and never written.
    pub fn set_shadow(&mut self, shadow: bool) {
        self.shadow = shadow;
    }

    /// Every instance's predicted cursors, in instance order. The ranks of a
    /// tensor-parallel group compare these after a frame: a difference is a
    /// gate that decided differently on one rank.
    #[must_use]
    pub fn predictions(&self) -> Vec<(u64, Vec<Cursor>)> {
        self.instances
            .iter()
            .map(|(id, bound)| (*id, bound.session.predictions()))
            .collect()
    }

    /// What the compile tiers have been doing.
    #[must_use]
    pub const fn stats(&self) -> eta_exec::CacheStats {
        self.cache.stats()
    }

    /// Adopt a registration and compile its regions, answering the program id.
    ///
    /// A program already registered under the same hash answers its existing
    /// id and compiles nothing.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] when the package is not adoptable (no stages, or plans
    /// and stages that are not parallel), and [`Fault::Compile`] when a region
    /// does not compile here. A deterministic compile refusal is remembered
    /// and answers the next attempt immediately; a retryable one is not.
    pub fn register(
        &mut self,
        context: &Context,
        registration: &ProgramRegistration,
    ) -> Result<u64> {
        if let Some(&existing) = self.by_hash.get(&registration.program_hash) {
            return Ok(existing);
        }
        // `Boundaries::CUDA`, not the default: this backend answers
        // `envelope_dot`, `lora`, `attn_page_mask`.
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
        // Fire-path buffers are cut here at bind, against these extents, so
        // a shape that doesn't resolve refuses at the door rather than
        // zero-filling silently at the first fire.
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
            self.shadow,
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
    /// A [`GeometryClass::Host`] instance's fire reads token ids, positions
    /// and readable extent from the submission (folded host-side); a
    /// [`GeometryClass::DecodeEnvelope`] instance gets them from here
    /// instead, since the runtime couldn't know them at submission time.
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

    /// Point one intrinsic of instance `id` at a device buffer — e.g. a
    /// program reading `IntrinsicId::Logits` is pointed at the readout
    /// buffer a model fire produced.
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

    /// How many score planes instance `id` declared, or `None` for one that
    /// reads no score rectangle. A rectangle read for more rows than a
    /// lane's block holds would walk silently into the next lane's memory,
    /// which is what this lets the shell refuse against.
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
    /// right now would not meet, or `None` when it is ready. Asked before the
    /// model fire, since an epilogue attachment fires after the forward has
    /// already written the lane's KV. See [`Session::blocked_channel`].
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

    /// The caller's prediction against this engine's, if they differ. A
    /// prediction of [`Ticket::NONE`](engine::Ticket::NONE) makes no claim
    /// about that end of the ring and is not compared. A channel the caller
    /// names that this instance does not carry is itself a disagreement.
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
    /// one commit. A boundary of exactly one lane — not a different
    /// protocol, only a smaller wave.
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

    /// The staging half of a fire: mint instance `id`'s tickets, slot lists
    /// and lane records into the plane's [`Wave`]; enqueues nothing.
    ///
    /// The program is looked up through the instance's own record rather
    /// than taken as an argument, so a caller cannot fire one program's
    /// stages against another's rings.
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

    /// The boundary, on the stream: the wave's copy and its `pull_validate`,
    /// then every staged fire's regions, then the wave's `commit_bump` and
    /// `scatter_publish`.
    ///
    /// Staged fires are grouped by `(program, extents)` and each group's
    /// regions launch once with a block per lane, instead of one launch per
    /// lane. Order is preserved within a group (same stream); between groups
    /// there is nothing to order, since two lanes of a boundary share no
    /// ring. A boundary with nothing staged launches nothing.
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

        // First-seen order, so launches come out in staging order.
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
            // A group that doesn't fit the wave's scratch ceiling is split
            // into chunks, not refused.
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
            // Stages launch unconditionally; each kernel reads its lane's
            // per-lane commit word and returns early when refused. No
            // synchronize between stages — one stream is the ordering.
            for (index, stage) in program.compiled.stages.iter().enumerate() {
                let Some(prepared) = program
                    .batches
                    .iter()
                    .find(|batch| batch.extents == extents)
                    .and_then(|batch| batch.stages.get(index))
                    .and_then(Option::as_ref)
                else {
                    // A stage with nothing to launch needs no buffers.
                    continue;
                };
                for region in stage.regions.iter() {
                    prepared.launch_region(region, stream)?;
                }
            }
        }
        Ok(())
    }

    /// The wave's bump and its publication, once for the whole batch — see
    /// [`wave::Wave::land`]. Clears the wave for the next boundary.
    ///
    /// # Errors
    ///
    /// Whatever the launches said.
    pub fn land(&mut self, context: &Context) -> Result<()> {
        self.staged.clear();
        self.wave.land(context)
    }

    /// How many lanes are staged and not yet flown.
    #[must_use]
    pub fn staged(&self) -> usize {
        self.staged.len().max(self.wave.staged())
    }

    /// Throw away a batch that will never fly. Host half only — a staged
    /// lane has touched no stream. Used when a fault unwinds between a
    /// boundary's first stage and its landing, so the next boundary doesn't
    /// launch a control kernel over rows nobody will settle.
    pub fn abandon_wave(&mut self) {
        self.staged.clear();
        self.wave.clear();
    }


    /// The verdict half — see [`Session::settle_launched`].
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

    /// Which shared rings instance `id` holds — see
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
        // Seats go back; a shared ring's `Arc` is also held by the engine's
        // channel table, so it survives this drop.
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

/// One bound instance and the program it is an instance of.
#[derive(Debug)]
struct Bound {
    program_id: u64,
    session: Session,
    /// This instance's share of every ring it bound, kept so closing it
    /// gives back the seats its shared rings hold — see [`release_seats`].
    /// Cloned `Arc`s, so holding them keeps a shared ring alive as long as
    /// some attachment can address it.
    endpoints: Vec<Option<std::sync::Arc<Endpoint>>>,
    /// The caller's geometry claim at bind — what [`Plane::envelope`] gates
    /// on.
    geometry: GeometryClass,
    /// This instance's channels by the caller's id, in declaration order —
    /// what [`Plane::disagreeing_ticket`] reads.
    ids: Vec<u64>,
}

/// Every channel's host end, adopted or opened. A channel with no host end
/// has no endpoint. A channel that has one takes the caller's if already
/// registered (the guest's write mirror must be the pull's read mirror), or
/// gets a fresh one otherwise. An adopted endpoint is checked against its
/// declaration: a mismatched mirror addresses the wrong bytes and nothing
/// faults — it serves the wrong token, quietly, forever.
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
            // A bind that does not happen holds no seat; give back what was
            // taken before the refusal.
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
        // A `None` role does not mean "no endpoint": the ring belongs to the
        // channel, and a `None` channel this engine registered offers a
        // shared endpoint every attachment takes. A `None` channel never
        // registered still gets `None`.
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
        // A device-only ring is cut at its native width, not its wire width:
        // the emitted kernels index one byte per bool lane, so cutting at
        // the (bit-packed) wire width would undersize the slab.
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
                // Only a shared (device-only) ring counts seats; a
                // host-visible endpoint is adopted by one instance at a time.
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

/// Give back every seat this instance's channels hold — the inverse of the
/// `attach` in [`endpoints_for`].
///
/// A shared ring is an `Arc<Endpoint>` held by the engine's channel table
/// and by every bound session, freed only when the last holder drops — so
/// it outlives any one instance's close. The seat is separate bookkeeping,
/// given back here so repeated close/rebuild doesn't exhaust the seat bound.
fn release_seats(endpoints: &[Option<std::sync::Arc<Endpoint>>]) {
    use eta_ir::container::HostRole;

    for endpoint in endpoints.iter().flatten() {
        if endpoint.role() == HostRole::None {
            endpoint.detach();
        }
    }
}
