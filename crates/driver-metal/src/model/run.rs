//! Running a fire: the four calls, in one place.
//!
//! Everything under `model/` is a step of one walk, and this is the walk:
//!
//! ```text
//! rows_of(step)          the frame's rows          model::frame
//! lower(plan, rows)      rows -> rectangles        model-compiler
//! plan(lowered, ..)      rectangles -> grids       model::dispatch
//! encode(dispatches)     grids -> a command buffer model::encode
//! ```
//!
//! Nothing here decides anything either. It allocates the arena the lowering
//! asked for, stages the scalars the statements state, compiles the symbols
//! they name, and runs them in order.
//!
//! # What is deliberately NOT here
//!
//! The KV pool, the paged translation and the completion broker. Those are the
//! frame's device half and they belong with the module that owns the buffers —
//! this one owns only the arena, which is the lowering's own number.

use crate::error::Result;
use crate::metal::{ArgumentTable, Compiler, Context, Handle, Stepper, Timing, allocate};
use crate::region::Region as _;
use kernels::KernelSig;

/// The signature table this backend dispatches against.
///
/// A function rather than a re-export so that the one place naming the table
/// is here: `model::dispatch::plan` takes the vocabulary as a parameter, which
/// keeps it testable against a table of its own, and this is the answer every
/// real caller gives it.
#[must_use]
pub fn table() -> &'static [KernelSig] {
    kernels_metal::KERNELS
}
use crate::model::dispatch::{Dispatch, Geometry, Undispatchable, plan};
use crate::model::encode::{Params, Pipelines, encode};
use crate::model::executor::{Frame, Resolver, Slice};
use model_compiler::lower::Lowered;

/// The widest operand count any statement of a fire binds, plus its scalars.
///
/// An argument table is created with a fixed bind count and a binding past it
/// is an error rather than a silent no-op — so the table has to be built for
/// the widest statement in the fire, not for a guess.
#[must_use]
pub fn table_width(dispatches: &[Dispatch<'_>]) -> usize {
    dispatches
        .iter()
        // One slot per operand, and ONE more for the packed params — the
        // scalars ride as a single struct, which is what every shader in the
        // tree takes (`constant RouterParams&` and its siblings).
        .map(|d| {
            let params = if d.params.is_empty() {
                0
            } else {
                d.param_slots.iter().map(|p| p.slot + 1).max().unwrap_or(0)
            };
            d.args.len().max(params)
        })
        .max()
        .unwrap_or(1)
        .max(1)
}

/// One fire's device state: the arena, the scalars, the pipelines, the table.
///
/// Held across fires by a caller that runs many — the pipelines especially,
/// since a model's symbol set is bounded by its text and recompiling per fire
/// would cost more than the fire.
pub struct Prepared {
    /// The activation arena, sized by [`Lowered::arena_bytes`].
    pub arena: Handle,
    /// The scalars every statement states, staged.
    pub params: Params,
    /// The argument table, wide enough for the widest statement.
    pub table: ArgumentTable,
}

/// Allocate and stage everything one lowered fire needs.
///
/// # Errors
///
/// The arena allocation, the scalar staging, or the table.
pub fn prepare(context: &Context, lowered: &Lowered, dispatches: &[Dispatch<'_>]) -> Result<Prepared> {
    // `.max(1)`: a fire whose values all live in named buffers needs no arena,
    // and a zero-length allocation has no address to bind.
    let arena = allocate(
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

/// Plan, prepare, compile and run one lowered fire.
///
/// Returns the [`Timing`] the stepper measured, so a caller can compare a fire
/// against the handwritten path it replaces without instrumenting anything.
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

/// [`run`], returning the ARENA beside the timing.
///
/// The arena is where every activation this fire produced landed, and reading
/// it is the difference between "the fire executed" and "the fire computed
/// something". `run` drops it, which is right for a caller that wants the next
/// fire and wrong for one that wants to look — and looking is how the three
/// failure modes that survive a green execution get caught: an arena of zeros
/// is a fire whose projections no-opped, an arena of NaNs is a norm handed a
/// zero epsilon, and an arena of identical rows is a per-token axis that never
/// reached the kernels.
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
) -> Result<(Timing, Handle)> {
    // The arena has to exist before the operands can be bound, and the
    // dispatches have to exist before the arena's width is known to be
    // enough — so the arena is allocated from the lowering's own number and
    // the binder checks every offset against it.
    let arena = allocate(
        context,
        (lowered.arena_bytes as u64).max(1),
        "activation arena",
    )?;
    // ZEROED, and it is not a formality. A fresh Metal buffer is usually zero
    // and nothing promises it, so a slot no kernel writes holds whatever the
    // allocator handed over -- which made two runs of the same fire over the
    // same weights produce different numbers, measured. Zeroing turns "a
    // region nobody wrote" from garbage that looks like saturation into a
    // zero, which is diagnosable.
    //
    // SAFETY: freshly allocated; nothing is encoded against it yet.
    unsafe { arena.zero(0, arena.len())? };
    let frame = Frame {
        arena: Slice {
            address: arena.gpu_address(),
            bytes: arena.len(),
        },
    };
    let dispatches =
        plan(lowered, table(), frame, geometry, resolver).map_err(refusal)?;
    let params = Params::stage(context, &dispatches)?;
    let table = ArgumentTable::new(context, table_width(&dispatches))?;
    pipelines.ensure(context, compiler, &dispatches)?;

    let mut stepper = Stepper::new(context)?;
    let timing =
        stepper.run(|encoder| encode(encoder, &table, pipelines, &params, &dispatches))?;
    Ok((timing, arena))
}

/// A fire that has been COMMITTED and may still be running.
///
/// Everything the GPU still refers to, held together so a caller cannot drop
/// half of it. The command buffer addresses the arena, reads its operands out
/// of the argument table and its scalars out of the staged params; freeing
/// any of the three while the buffer executes is a use-after-free that a
/// green run will not show, because the bytes are usually still there.
///
/// So this owns them and hands back only `arena`, and only after
/// [`Stepper::has_passed`] says the fire retired.
pub struct InFlight {
    /// The timeline value this fire signals.
    pub value: u64,
    /// Where its activations landed. Read it after the fire retires.
    ///
    /// A LEASE: the region goes back to the pool when this drops, which is
    /// what makes the next fire of this shape reuse the same address.
    pub arena: crate::metal::Lease,
    /// Held for the GPU, not for the caller.
    _table: ArgumentTable,
    _params: Params,
}

/// Everything a driver keeps ACROSS fires.
///
/// Five things, and what they have in common is the reason they are one
/// struct: each is wrong to rebuild per fire, and each was wrong in its own
/// way before it was held.
///
/// * `stepper` — the timeline and the allocator ring. A fresh one has no
///   value to compare against and no allocator to alternate, so it cannot
///   pipeline even in principle.
/// * `scratch` — the fire's regions. A fresh region per fire leaks into the
///   residency set permanently (nothing removes) and moves an address that
///   is one of only three differing between two fires of one shape.
/// * `pipelines` — the compile cache. Rebuilding it recompiles every shader.
/// * `context`, `compiler` — the device and its shader compiler.
///
/// Grouped rather than passed as five parameters because they travel
/// together and always will: a caller that has one has all of them, and the
/// list was already at clippy's argument limit when `scratch` joined it.
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
    pub scratch: &'c crate::metal::Scratch,
    /// Which buffer each address belongs to, for recording. `&mut` because a
    /// fire registers what it leases -- the caller registers the weights and
    /// the pool, and only `submit` knows the arena it took.
    pub regions: &'c mut crate::metal::Regions,
    /// Fires already recorded, by what they are valid for.
    pub recordings: &'c mut crate::metal::Recordings,
}



/// Plan, encode and COMMIT one fire, without waiting for it.
///
/// [`run`] and [`run_keeping_arena`] end in a wait, so a caller cannot queue
/// the next fire until this one has finished -- the call that would queue it
/// has not returned. That makes a dispatch depth a number the engine honours
/// and the driver serialises, which is what `.wiki/new-driver/next.md`
/// priority 1 is about.
///
/// `stepper` is the caller's and must be the SAME one across fires: the
/// timeline and the allocator ring live on it, and a fresh `Stepper` per fire
/// -- which is what `run_keeping_arena` does -- has neither a value to
/// compare against nor an allocator to alternate. [`Stepper::submit`] bounds
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
    // fires, like `stepper`. Two things follow from reuse, and both are
    // measured in `.wiki/driver/graph-metal.md`:
    //
    // * `ring::allocate` adds every buffer to the residency set and NOTHING
    //   removes it, so a fresh region per fire leaks permanently. Fifty
    //   allocate-and-drop cycles leave fifty allocations and 52 MB resident.
    //   A serving driver does three per fire.
    // * the arena's address is one of only three things that differ between
    //   two fires of one shape, and it is what stands between this driver and
    //   recording its command buffer once instead of re-encoding 424
    //   dispatches per fire -- 76.4% of a decode.
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
    let dispatches = plan(lowered, table(), frame, geometry, resolver).map_err(refusal)?;
    let params = Params::stage_in(context, scratch, &dispatches)?;
    // What this fire leased, so a recording can turn its operand addresses
    // back into buffers. Pooled, so re-registering is the same span and the
    // registry does not grow.
    regions.add(&arena);
    regions.add(params.region());
    let table = ArgumentTable::new(context, table_width(&dispatches))?;
    pipelines.ensure(context, compiler, &dispatches)?;
    // REPLAY if this exact fire has been recorded, encode if not.
    //
    // Measured: `llama_like`'s 424-dispatch decode costs **14.87 ms** to
    // encode and **39.8 us** to replay -- 374x, because encoding is about
    // 5 000 Objective-C messages and replaying is one. On decode that was
    // 76.4% of the step.
    //
    // The fingerprint is the validity condition, checked rather than assumed:
    // a recording bakes each operand's buffer and offset, its grid and its
    // pipeline, and replaying one against a fire that differs in any of those
    // runs the wrong program silently. A fire whose digest is new gets its own
    // recording -- nothing is ever rewritten in place, because a fire in
    // flight is executing out of its ICB.
    //
    // Falls back to encoding rather than failing when a recording cannot be
    // made: an operand in no registered allocation is a deployment that has
    // not registered its regions, and the encode path binds addresses and does
    // not care.
    let recorded = recordings
        .get_or_record(context, pipelines, &params, regions, &dispatches)
        .ok()
        .map(|r| (objc2::rc::Retained::from(r.buffer()), r.commands()));
    let value = match recorded {
        Some((icb, commands)) => stepper.submit(|encoder| encoder.execute_commands(&icb, 0..commands))?,
        None => stepper.submit(|encoder| encode(encoder, &table, pipelines, &params, &dispatches))?,
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
