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
