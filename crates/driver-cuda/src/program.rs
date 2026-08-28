//! The guest-program plane, CUDA half: PTIR at the fire's boundary.
//!
//! **WHAT A GUEST PROGRAM IS.** A tensor program the guest authored, planned
//! by `tensor-compiler` on the host into a `LaunchPackage` plus a table of
//! emitted CUDA sources, and executed here against a set of channel rings.
//! Design §9 places it at the boundary of the model fire and nowhere else:
//! a prologue before the immutable graph, an epilogue after it, and never a
//! hook inside — guest code cannot enter a graph that is recorded once and
//! replayed forever.
//!
//! ```text
//! driver/src/program/        this module
//! -------------------        -----------
//! adopt_launch_package  ->   compile.rs   NVRTC, cubins, the three cache tiers
//! ChannelState, rings   ->   launch.rs    device cells, the lane-table bytes
//! readiness, step       ->   session.rs   one instance's lifetime and its fire
//! ```
//!
//! **THE HOST HALF IS THE GOLDEN AND IT IS NOT A MOCK.** `driver::program` is
//! a complete interpreter of the same launch package: same ops, same channel
//! semantics, same pass-atomic commit. This half is the subject, and
//! `tests/program_parity.rs` runs both over the same programs with the same
//! inputs and demands byte-identical rings and identical outcomes, fire for
//! fire. Bit-for-bit reproducibility is the channel plane's first-class
//! contract, which is why `--fmad=false --prec-div=true --prec-sqrt=true` is
//! a determinism clause in [`compile`] rather than a tuning flag.
//!
//! **THE SEAM IS WIRED** (`palo B2`). A [`FireSubmission`]'s
//! [`Attachment`](driver::driver_api::fire::Attachment)s name a lane and a
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
pub mod launch;
pub mod ports;
pub mod session;

use std::collections::BTreeMap;

use driver::driver_api::program::ProgramRegistration;
use driver::tensor_ir::registry::GeometryClass;
use driver::{Boundaries, ExecPlan, Extents, Versions, adopt_launch_package_with};

use crate::device::Context;
use crate::error::{Fault, Result};

pub use compile::{Cache, Compiled, Disk, Module, Region, Stage, Target};
pub use launch::{ChannelShape, Cursor, Prepared, Rings};
pub use ports::Envelope;
pub use session::{Fired, Session, seeds_of};

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
}

impl Default for Plane {
    /// A plane whose cubins are cached wherever [`Disk::from_env`] points.
    fn default() -> Plane {
        Plane::new(Disk::from_env())
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
        }
    }

    /// What the compile tiers have been doing.
    #[must_use]
    pub const fn stats(&self) -> driver::CacheStats {
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
    ) -> Result<u64> {
        let program = self
            .programs
            .get(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        let session = Session::bind(&program.compiled, &program.plan, seeds, extents)?;
        let id = self.next_instance;
        self.next_instance += 1;
        self.instances.insert(
            id,
            Bound {
                program_id,
                session,
                geometry,
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
    /// engine folded them host-side and stated them — and reading the rings
    /// as well would be reading a second opinion about values that are
    /// already decided. A [`GeometryClass::DecodeEnvelope`] instance's
    /// submission carries placeholders for exactly those three, because the
    /// engine could not know them, and this is where they come from.
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
        bound.session.envelope(&program.plan).map(Some)
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
    /// The engine's attachment seam (design §9): a program that reads
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
        context: &Context,
        id: u64,
        intrinsic: driver::tensor_ir::op::IntrinsicId,
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
            .bind_intrinsic(
                context, intrinsic, base, storage, width, row_stride, row_offset,
            )
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

    /// Fire instance `id` once: readiness, then every stage's regions, then
    /// one commit.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, and whatever the launches
    /// said. A program that blocks or refuses is a [`Fired`], not an error.
    pub fn fire(&mut self, context: &Context, id: u64) -> Result<Fired> {
        // The program is looked up THROUGH the instance's own record rather
        // than taken as an argument, so a caller cannot fire one program's
        // stages against another's rings — which would index scratch,
        // descriptors and a channel table sized for someone else, and would
        // not fault.
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
            .fire(context, &program.compiled, &program.plan)
    }

    /// Drop instance `id`, freeing its rings and its stage buffers.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when there is no such instance — closing twice is a
    /// caller's bug, not a no-op.
    pub fn close_instance(&mut self, id: u64) -> Result<()> {
        self.instances
            .remove(&id)
            .map(|_| ())
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))
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
    /// How much of the fire geometry this instance's descriptor resolves on
    /// the device — the caller's claim at bind, kept because it is what
    /// [`Plane::envelope`] gates on. A class the load does not serve never
    /// reaches here: [`Cuda::bind_instance`](crate::api::Cuda) refuses it by
    /// name.
    geometry: GeometryClass,
}
