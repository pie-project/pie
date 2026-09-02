//! The guest-program plane, Metal half: ETA at the model fire's boundary
//! (prologue/epilogue, never mid-graph). `compile.rs`, `launch.rs` and
//! `session.rs` own compilation, device rings, and instance lifetime.

pub mod compile;
pub mod launch;
pub mod ports;
pub mod session;
pub mod shared;

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::program::ProgramRegistration;
use eta_exec::adopt_launch_package;
use eta_exec::{ExecPlan, Extents, Versions};
use eta_ir::registry::GeometryClass;

use crate::device::Context;
use crate::device::ctx::Frame;
use crate::error::{Fault, Result};

pub use compile::{Cache, Compiled, Module, Region, Stage, Target};
pub use launch::{ChannelShape, Cursor, Prepared, Rings};
pub use ports::Envelope;
pub use session::{Blocked, Fired, Launched, Session, seeds_of};
pub use shared::{MAX_ATTACHMENTS, SharedRing};

/// One registered program: what the host planned, and what compiled from it.
#[derive(Debug)]
pub struct Program {
    /// The plane's own handle for it.
    pub id: u64,
    /// The host's content hash a re-registration is recognised by.
    pub hash: u64,
    /// The adopted launch package: channel decls, stages, and derived
    /// per-stage value indexes.
    pub plan: ExecPlan,
    /// The compiled regions, one table per stage.
    pub compiled: Compiled,
}

/// The shell's guest-program plane: compile cache, registered programs,
/// and bound instances. Needs only a bound [`Context`] — no weights, no
/// arena, no `CompiledModel`.
#[derive(Debug)]
pub struct Plane {
    cache: Cache,
    /// Compiled pipelines every stage's regions are encoded with, owned
    /// by the plane (unlike CUDA's `CUfunction`) so guest and model
    /// shaders share one compile-once cache.
    pipelines: crate::device::Pipelines,
    programs: BTreeMap<u64, Program>,
    by_hash: BTreeMap<u64, u64>,
    instances: BTreeMap<u64, Bound>,
    /// Device-only rings keyed by the caller's channel id (a program
    /// names channels by dense slot, which differs per instance). Only
    /// `HostRole::None` channels are here.
    channels: BTreeMap<u64, Arc<SharedRing>>,
    next_program: u64,
    next_instance: u64,
}

impl Default for Plane {
    /// A plane with an empty compile cache; unlike CUDA there is no cubin
    /// directory (a compiled MSL library never leaves the process).
    fn default() -> Plane {
        Plane::new()
    }
}

impl Plane {
    /// An empty plane; unlike CUDA's `Disk`, this half has no persistent
    /// cache tier.
    #[must_use]
    pub fn new() -> Plane {
        Plane {
            cache: Cache::new(),
            pipelines: crate::device::Pipelines::new(),
            programs: BTreeMap::new(),
            by_hash: BTreeMap::new(),
            instances: BTreeMap::new(),
            channels: BTreeMap::new(),
            next_program: 1,
            next_instance: 1,
        }
    }

    /// What the compile tiers have been doing.
    #[must_use]
    pub const fn stats(&self) -> eta_exec::CacheStats {
        self.cache.stats()
    }

    /// Adopt a registration and compile its regions, answering the
    /// program id. A program already registered under the same hash
    /// answers its existing id and compiles nothing. Uses
    /// [`eta_exec::Boundaries::METAL`].
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] when the package is not adoptable, [`Fault::Compile`]
    /// when a region does not compile here — a deterministic refusal is
    /// remembered, a retryable one is not.
    pub fn register(
        &mut self,
        context: &Context,
        registration: &ProgramRegistration,
    ) -> Result<u64> {
        if let Some(&existing) = self.by_hash.get(&registration.program_hash) {
            return Ok(existing);
        }
        let plan = adopt_launch_package(registration.launch.clone())?;
        let compiled = self.cache.compile(
            context,
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
    /// stage's fire-path buffers, and seed the guest-seeded channels.
    /// `channels` names this instance's channels by global id in
    /// declaration order; a slot naming a registered channel adopts its
    /// shared ring, every other slot gets its own cut.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown program, a bad seed, or a shared
    /// ring past its [`MAX_ATTACHMENTS`] seats or wrong geometry; and
    /// whatever the allocations said.
    pub fn bind(
        &mut self,
        context: &Context,
        program_id: u64,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
        geometry: GeometryClass,
        channels: &[u64],
    ) -> Result<u64> {
        let program = self
            .programs
            .get(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        let adopted = self.seats_for(&program.plan, channels)?;
        let session = match Session::bind(
            context,
            &program.compiled,
            &program.plan,
            seeds,
            extents,
            &adopted,
        ) {
            Ok(session) => session,
            Err(why) => {
                // An instance that did not bind holds no seat.
                release_seats(&adopted);
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
                geometry,
                shared: adopted,
            },
        );
        Ok(id)
    }

    /// Cut the one ring a device-only channel owns, keyed by the
    /// caller's id. Every instance naming this id attaches to the same
    /// [`SharedRing`] rather than carving a copy; a second registration
    /// of the same id is refused.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an id already registered, and whatever the
    /// reservation said.
    pub fn register_channel(
        &mut self,
        context: &Context,
        id: u64,
        shape: ChannelShape,
    ) -> Result<()> {
        if self.channels.contains_key(&id) {
            return Err(Fault::program(
                "program::plane",
                format!(
                    "channel {id} is already registered, and a second ring under one \
                     name would leave the instances bound before it addressing \
                     different cells from the ones bound after"
                ),
            ));
        }
        self.channels
            .insert(id, Arc::new(SharedRing::open(context, shape)?));
        Ok(())
    }

    /// Forget channel `id`'s registration. The ring itself is an `Arc`
    /// every attached instance also holds, freed only when the last
    /// holder drops. Answers whether there was one.
    pub fn close_channel(&mut self, id: u64) -> bool {
        self.channels.remove(&id).is_some()
    }

    /// Which other instances share a ring with one of `instances` — the
    /// set a fence must widen to, since a shared ring's counters advance
    /// a frame after the fire that moved them.
    #[must_use]
    pub fn cohort(&self, instances: &[u64]) -> Vec<u64> {
        let held: Vec<&Arc<SharedRing>> = instances
            .iter()
            .filter_map(|id| self.instances.get(id))
            .flat_map(|bound| bound.shared.iter().flatten())
            .collect();
        if held.is_empty() {
            return Vec::new();
        }
        let mut cohort = Vec::new();
        for (id, bound) in &self.instances {
            if instances.contains(id) {
                continue;
            }
            // Identity, not equality: same allocation, via `Arc::ptr_eq`.
            if bound.shared.iter().flatten().any(|mine| {
                held.iter().any(|theirs| Arc::ptr_eq(mine, theirs))
            }) {
                cohort.push(*id);
            }
        }
        cohort
    }

    /// Take one seat on every shared ring this binding names, in the
    /// program's dense channel order. A refusal gives back the seats
    /// taken before it.
    fn seats_for(
        &self,
        plan: &ExecPlan,
        channels: &[u64],
    ) -> Result<Vec<Option<Arc<SharedRing>>>> {
        let mut adopted: Vec<Option<Arc<SharedRing>>> =
            Vec::with_capacity(plan.package.channels.len());
        for dense in 0..plan.package.channels.len() {
            // A channel this plane never registered gets `None` — its ring
            // is cut inside the session instead.
            let Some(ring) = channels.get(dense).and_then(|id| self.channels.get(id)) else {
                adopted.push(None);
                continue;
            };
            if let Err(why) = ring.attach() {
                release_seats(&adopted);
                return Err(why);
            }
            adopted.push(Some(Arc::clone(ring)));
        }
        Ok(adopted)
    }

    /// What instance `id`'s descriptor ports resolve to right now, or
    /// `None` when its class says the host resolves them. The class
    /// decides which of the nine ports are read (`ports::resolves`).
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, one whose program is
    /// gone, and whatever [`ports::resolve`] said.
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
    /// program reading `IntrinsicId::Logits` is pointed at a model
    /// fire's readout buffer. `width`/`dtype` are checked, not obeyed.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, and whatever the
    /// session's own bind said about the rectangle.
    pub fn bind_intrinsic(
        &mut self,
        id: u64,
        intrinsic: eta_ir::op::IntrinsicId,
        base: &crate::device::Buffer,
        offset: u64,
        width: u32,
        dtype: eta_ir::Dtype,
    ) -> Result<()> {
        self.instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?
            .session
            .bind_intrinsic(intrinsic, base, offset, width, dtype)
    }

    /// The first channel of instance `id` whose declared requirement a
    /// fire right now would not meet, or `None` when ready. Asked before
    /// launch, since an epilogue attachment fires after the forward has
    /// already written the lane's KV.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, or one whose program
    /// is gone.
    pub fn ready(&self, id: u64) -> Result<Option<session::Blocked>> {
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
        Ok(bound.session.readiness(&program.plan))
    }

    /// Fire instance `id` once: readiness, then every stage's regions,
    /// then one commit.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, and whatever the
    /// launches said. A blocked or refused program is a [`Fired`], not
    /// an error.
    pub fn fire(&mut self, context: &Context, id: u64) -> Result<Fired> {
        // Looked up through the instance's own record, so a caller can't
        // fire one program against another's rings.
        let bound = self
            .instances
            .get_mut(&id)
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
        bound
            .session
            .fire(context, &self.pipelines, &program.compiled, &program.plan)
    }

    /// Encode instance `id`'s whole pass into a command buffer someone
    /// else owns, without committing it — the attached spelling of
    /// [`Plane::fire`].
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance or one with a pass
    /// already airborne, and whatever the encode said.
    pub fn stage_into(&mut self, frame: &Frame, id: u64) -> Result<Launched> {
        let bound = self
            .instances
            .get_mut(&id)
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
        bound
            .session
            .stage_into(frame, &program.compiled, &program.plan)
    }

    /// Read the verdict of instance `id`'s airborne pass and commit its
    /// cursors. The caller owes proof the command buffer landed (see
    /// [`Session::settle_launched`]).
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance or one whose program is
    /// gone, and whatever the status reads said.
    pub fn settle_launched(&mut self, id: u64) -> Result<Fired> {
        let bound = self
            .instances
            .get_mut(&id)
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
        bound.session.settle_launched(&program.plan)
    }

    /// Drop instance `id`'s airborne mark without reading a verdict, for
    /// a staging whose command buffer will not be committed. No-op for
    /// an instance this plane does not carry.
    ///
    /// See [`Session::abandon_launched`] for what the caller owes.
    pub fn abandon_launched(&mut self, id: u64) {
        if let Some(bound) = self.instances.get_mut(&id) {
            bound.session.abandon_launched();
        }
    }

    /// Whether instance `id` has a pass in a command buffer that has not been
    /// settled. `false` for an instance this plane does not carry.
    #[must_use]
    pub fn is_airborne(&self, id: u64) -> bool {
        self.instances
            .get(&id)
            .is_some_and(|bound| bound.session.is_airborne())
    }

    /// Whether instance `id`'s program reads the draft column. A load
    /// whose model text declares no draft head has none to point at;
    /// `serve::prepare` refuses such an attachment by name.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, or one whose program
    /// is gone.
    pub fn needs_mtp_logits(&self, id: u64) -> Result<bool> {
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
        Ok(program.plan.needs_mtp_logits)
    }

    /// Whether instance `id`'s program reads the `mtp_drafts` intrinsic —
    /// the token plane, bound at its own rectangle beside the logits.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance or a program that is gone.
    pub fn needs_mtp_drafts(&self, id: u64) -> Result<bool> {
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
        Ok(program.plan.needs_mtp_drafts)
    }

    /// Whether instance `id`'s program reads the `attn_score` intrinsic.
    /// [`Plane::needs_mtp_logits`]'s twin.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an instance this plane does not carry, or
    /// one whose program is gone.
    pub fn needs_attn_scores(&self, id: u64) -> Result<bool> {
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
        Ok(program.plan.needs_attn_scores)
    }

    /// How many score planes instance `id` declared, or `None` for one
    /// that reads no score rectangle.
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
        // Seats go back; the rings do not necessarily go with them.
        release_seats(&bound.shared);
        Ok(())
    }

    /// Drop program `id`, dropping this plane's share of its libraries.
    /// Advisory, not protective (Metal has no unload verb; ARC keeps a
    /// bound [`Session`]'s [`Compiled`] alive), but stays since a caller
    /// closing a program with instances still bound has lost track of them.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when there is no such program, or instances
    /// are still bound to it.
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
                    "program {id} still has {bound} instance(s) bound; ARC would keep their \
                     pipelines alive, so this is a caller that has lost track of its \
                     instances rather than a launch into freed machine code"
                ),
            ));
        }
        self.cache.forget(hash);
        self.by_hash.remove(&hash);
        self.programs.remove(&id);
        Ok(())
    }
}

/// One bound instance and the program it is an instance of; a struct
/// (not two maps) so an instance can't be fired against the wrong plan.
#[derive(Debug)]
struct Bound {
    program_id: u64,
    session: Session,
    /// How much of the fire geometry this instance's descriptor resolves
    /// on the device — kept for [`Plane::envelope`] to gate on.
    geometry: GeometryClass,
    /// This instance's share of every ring belonging to a channel, dense
    /// declaration order, `None` for a channel with its own ring. Lets
    /// closing give seats back and [`Plane::cohort`] find who else moves
    /// a ring this instance reads.
    shared: Vec<Option<Arc<SharedRing>>>,
}

/// Give back every seat this instance's channels hold — the inverse of
/// `attach` in [`Plane::seats_for`].
fn release_seats(shared: &[Option<Arc<SharedRing>>]) {
    for ring in shared.iter().flatten() {
        ring.detach();
    }
}
