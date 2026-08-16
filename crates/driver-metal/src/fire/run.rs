//! Running a fire: the four calls, in one place.
//!
//! Everything under `model/` is a step of one walk:
//!
//! ```text
//! rows_of(step)          the frame's rows          model::frame
//! lower(plan, rows)      rows -> rectangles        model-compiler
//! plan(lowered, ..)      rectangles -> grids       model::dispatch
//! encode(dispatches)     grids -> a command buffer model::encode
//! ```
//!
//! Nothing here decides anything: it allocates the arena the lowering asked
//! for, stages the scalars the statements state, compiles the symbols they
//! name, and runs them in order.
//!
//! Deliberately NOT here: the KV pool, the paged translation and the
//! completion broker, which belong with the module that owns the buffers —
//! this one owns only the arena, the lowering's own number.

use crate::bind::encode::{Params, Pipelines, encode};
use crate::device::{Allocation, ArgumentTable, Context, Stepper, Timing};
use crate::error::Result;
use crate::layout::region::Region as _;
use crate::lowering::dispatch::{Dispatch, Geometry, Undispatchable, plan, table_width};
use crate::lowering::executor::{Frame, Resolver, Slice};
use crate::program::Compiler;
use model_compiler::lower::Lowered;

/// One fire's device state: the arena, the scalars, the pipelines, the table.
///
/// Held across fires by a caller that runs many — the pipelines especially,
/// since a model's symbol set is bounded by its text and recompiling per fire
/// would cost more than the fire.
pub struct Prepared {
    /// The activation arena, sized by [`Lowered::arena_bytes`].
    pub arena: Allocation,
    /// The scalars every statement states, staged.
    pub params: Params,
    /// The argument table, wide enough for the widest statement.
    pub table: ArgumentTable,
}

/// Allocate and stage everything one lowered fire needs.
///
/// Exists only for two tests in `device_real_weights.rs` that encode a
/// prefix by hand; every other caller composes this and `encode` through
/// [`run`], [`run_keeping_arena`] or [`submit`].
///
/// Callers must re-plan before reusing the returned arena: the dispatches
/// were planned against some frame, and the arena allocated here gives a
/// different one, so `Prepared::frame()` only agrees with them if the caller
/// re-plans. `submit` avoids the hazard by allocating before it plans.
///
/// # Errors
///
/// The arena allocation, the scalar staging, or the table.
pub fn prepare(
    context: &Context,
    lowered: &Lowered,
    dispatches: &[Dispatch<'_>],
) -> Result<Prepared> {
    // `.max(1)`: a fire whose values all live in named buffers needs no arena,
    // and a zero-length allocation has no address to bind.
    let arena = Allocation::new(
        context,
        (lowered.arena_bytes as u64).max(1),
        "activation arena",
    )?;
    let params = Params::stage(context, dispatches)?;
    let table = ArgumentTable::new(context, table_width(dispatches))?;
    Ok(Prepared {
        arena,
        params,
        table,
    })
}

impl Prepared {
    /// The frame the binder addresses.
    #[must_use]
    pub fn frame(&self) -> Frame {
        Frame {
            arena: Slice {
                address: self.arena.gpu_address(),
                bytes: self.arena.len(),
            },
        }
    }
}

/// Plan, prepare, compile and run one lowered fire; returns the [`Timing`]
/// the stepper measured.
///
/// # Errors
///
/// A statement that would not dispatch ([`Undispatchable`]), or any device
/// failure on the way.
pub fn run<R: Resolver>(
    context: &Context,
    compiler: &Compiler,
    pipelines: &mut Pipelines,
    lowered: &Lowered,
    geometry: Geometry,
    resolver: &mut R,
) -> Result<Timing> {
    run_keeping_arena(context, compiler, pipelines, lowered, geometry, resolver).map(|(t, _)| t)
}

/// [`run`], returning the ARENA beside the timing: reading it after a green
/// run catches three failure modes that a passing run alone would not — an
/// arena of zeros (projections no-opped), of NaNs (a norm handed a zero
/// epsilon), or of identical rows (a per-token axis that never reached the
/// kernels).
///
/// It IS the production path, with fresh state: a `Stepper`, `Scratch` and
/// `Regions` built per call cannot pipeline, reuse an address or record, so
/// this is the slowest way to run a fire. Use [`submit`] with a [`Machine`]
/// to serve.
///
/// # Errors
///
/// As [`run`].
pub fn run_keeping_arena<R: Resolver>(
    context: &Context,
    compiler: &Compiler,
    pipelines: &mut Pipelines,
    lowered: &Lowered,
    geometry: Geometry,
    resolver: &mut R,
) -> Result<(Timing, crate::fire::Lease)> {
    // This is `submit`, with per-call state instead of the caller's: no
    // pipelining (one stepper, one fire), no address reuse across calls, and
    // no recording -- properties of the STATE, which is what `Machine` holds.
    let mut stepper = Stepper::new(context)?;
    let scratch = crate::fire::Scratch::new();
    let mut regions = crate::device::Regions::new();

    let began = std::time::Instant::now();
    let fire = submit(
        &mut Machine {
            context,
            compiler,
            pipelines,
            stepper: &mut stepper,
            scratch: &scratch,
            regions: &mut regions,
            // Neither the weights nor the pool are registered here, so a
            // recording would allocate an ICB and fail to resolve, every call.
            recordings: None,
        },
        lowered,
        geometry,
        resolver,
    )?;
    // `fire` stays alive across the wait: its command buffer still addresses
    // the argument table and staged params, and dropping either early is a
    // use-after-free a green run would not show.
    let timing = stepper.wait_for_timing(fire.value, began.elapsed())?;
    Ok((timing, fire.arena))
}

/// A fire that has been COMMITTED and may still be running.
///
/// Everything the GPU still refers to, held together so a caller cannot drop
/// half of it: freeing the arena, table or params while the command buffer
/// executes is a use-after-free a green run will not show, since the bytes
/// are usually still there. Hands back only `arena`, and only after
/// [`Stepper::has_passed`] says the fire retired.
pub struct InFlight {
    /// The timeline value this fire signals.
    pub value: u64,
    /// Where its activations landed. Read it after the fire retires.
    ///
    /// A LEASE: the region goes back to the pool when this drops, which is
    /// what makes the next fire of this shape reuse the same address.
    pub arena: crate::fire::Lease,
    /// Held for the GPU, not for the caller.
    _table: ArgumentTable,
    _params: Params,
}

/// Everything a driver keeps ACROSS fires.
///
/// Five things, each wrong to rebuild per fire: `stepper` (the timeline and
/// allocator ring — fresh means no value to compare against, no pipelining),
/// `scratch` (the fire's regions — fresh leaks into the residency set
/// permanently and moves an address that should stay stable), `pipelines`
/// (the compile cache — rebuilding recompiles every shader), and `context`/
/// `compiler` (the device and its shader compiler).
///
/// Grouped rather than passed as five parameters because a caller that has
/// one has all of them, and separate parameters would exceed clippy's
/// argument-count lint.
#[derive(Debug)]
pub struct Machine<'c, 's> {
    /// The device.
    pub context: &'c Context,
    /// Its shader compiler.
    pub compiler: &'c Compiler,
    /// The compiled-pipeline cache.
    pub pipelines: &'c mut Pipelines,
    /// The command timeline and allocator ring.
    pub stepper: &'c mut Stepper<'s>,
    /// The reusable fire regions.
    pub scratch: &'c crate::fire::Scratch,
    /// Which buffer each address belongs to, for recording. `&mut` because a
    /// fire registers what it leases -- the caller registers the weights and
    /// the pool, and only `submit` knows the arena it took.
    pub regions: &'c mut crate::device::Regions,
    /// Fires already recorded, by what they are valid for.
    ///
    /// `None` means **do not try**: a recording binds buffers, so it can
    /// only be made once every region a fire's operands point into — the
    /// weights and the KV pool, owned by the caller — is registered.
    pub recordings: Option<&'c mut crate::fire::Recordings>,
}

/// Plan, encode and COMMIT one fire, without waiting for it — [`run`] and
/// [`run_keeping_arena`] are this function with per-call state, wrapped in a
/// wait; this is the one encode path in the module.
///
/// `stepper` is the caller's and must be the SAME one across fires: a fresh
/// `Stepper` per fire (what `run_keeping_arena` builds) has no value to
/// compare against and no allocator to alternate. [`Stepper::submit`] bounds
/// the depth by waiting for the step two back.
///
/// # Errors
///
/// As [`run`], plus a wedged stepper.
pub fn submit<R: Resolver>(
    machine: &mut Machine<'_, '_>,
    lowered: &Lowered,
    geometry: Geometry,
    resolver: &mut R,
) -> Result<InFlight> {
    let Machine {
        context,
        compiler,
        pipelines,
        stepper,
        scratch,
        regions,
        recordings,
    } = machine;
    // LEASED, and `scratch` must be the caller's -- the same one across
    // fires, like `stepper`: `ring::allocate` adds every buffer to the
    // residency set and nothing removes it, so a fresh region per fire leaks
    // permanently, and the arena's address is one of only three things that
    // differ between two fires of one shape, which is what recording a
    // command buffer once instead of re-encoding every dispatch relies on.
    let arena = scratch.take(
        context,
        (lowered.arena_bytes as u64).max(1),
        "activation arena",
    )?;
    // SAFETY: freshly allocated; nothing is encoded against it yet. Zeroed
    // for the reason `run_keeping_arena` states -- a slot no kernel writes
    // otherwise holds whatever the allocator handed over.
    unsafe { arena.zero(0, arena.len())? };
    let frame = Frame {
        arena: Slice {
            address: arena.gpu_address(),
            bytes: arena.len(),
        },
    };
    let dispatches = plan(lowered, frame, geometry, resolver).map_err(refusal)?;
    let params = Params::stage_in(context, scratch, &dispatches)?;
    // What this fire leased, so a recording can turn its operand addresses
    // back into buffers. Pooled, so re-registering is the same span and the
    // registry does not grow.
    regions.add(&arena);
    regions.add(params.region());
    let table = ArgumentTable::new(context, table_width(&dispatches))?;
    pipelines.ensure(context, compiler, &dispatches)?;
    // REPLAY if this exact fire has been recorded (roughly two orders of
    // magnitude cheaper than re-encoding every dispatch), else ENCODE.
    //
    // The fingerprint is the validity condition, checked rather than assumed:
    // a recording bakes each operand's buffer and offset, its grid and its
    // pipeline, so replaying one against a fire that differs in any of those
    // would silently run the wrong program. A fire whose digest is new gets
    // its own recording rather than rewriting one in flight.
    //
    // `Error::Unrecordable` (an operand in no registered allocation) falls
    // back to encoding on purpose; anything else `record` can fail at is a
    // bug and propagates.
    let recorded = match recordings.as_mut() {
        None => None,
        Some(recordings) => {
            match recordings.get_or_record(context, pipelines, &params, regions, &dispatches) {
                Ok(r) => Some((objc2::rc::Retained::from(r.buffer()), r.commands())),
                Err(crate::error::Error::Unrecordable { .. }) => None,
                Err(other) => return Err(other),
            }
        }
    };
    let value = match recorded {
        Some((icb, commands)) => stepper.submit(|encoder| {
            encoder.execute_commands(&icb, 0..commands)?;
            // The encode path closes its fire with one; a replay must too, or
            // whatever reads the arena next races the last recorded command.
            encoder.barrier(crate::device::Visibility::Device);
            Ok(())
        })?,
        None => {
            stepper.submit(|encoder| encode(encoder, &table, pipelines, &params, &dispatches))?
        }
    };
    Ok(InFlight {
        value,
        arena,
        _table: table,
        _params: params,
    })
}

/// A refusal, as this crate's error.
///
/// Rendered rather than wrapped: every variant is drift, and what a reader
/// needs is the symbol and the op, which the `Debug` already spells.
fn refusal(why: Undispatchable) -> crate::error::Error {
    crate::error::Error::Create {
        what: "fire",
        message: format!("{why:?}"),
    }
}
