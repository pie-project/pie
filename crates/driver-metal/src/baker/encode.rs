//! The door a claim body fires through, and the only thing on this plane
//! that `kernels_metal::routine::Ctx` actually is.
//!
//! `Ctx<'a>` is `dyn Encode + 'a` — a TRAIT OBJECT, where cuda's `Ctx` is a
//! struct holding a JIT handle and a stream. That is the whole shape of the
//! difference between the two planes, and it is why `BoundOp::Plane` had to
//! grow `+ ?Sized`: what a shader fire talks to is an encoder the DRIVER
//! owns, and the kernel crate never names a concrete one.
//!
//! # It plans; it does not encode
//!
//! [`Encoder::fire`] turns a body's statement — *this entrypoint, over these
//! threads, with these arguments* — into a [`Dispatch`] and pushes it. It
//! does not touch a command encoder, and that is deliberate rather than
//! incidental: the device half wants the WHOLE fire before it starts, to size
//! one argument table for the widest statement in it, to batch-compile every
//! pipeline it names, and to fingerprint it for indirect-command-buffer
//! replay. A body that encoded as it went would take all three away.
//!
//! It is also what makes the walk testable. `tests/the_walk_is_the_program.rs`
//! hands the walk an `Encode` that records instead of planning, and nothing
//! in `baker::fire` or `baker::bound` can tell the difference — because
//! neither of them has ever seen a device.
//!
//! # What the legacy `Planner` carried that this does not
//!
//! `lowering::routine::Planner` was the same idea against the routine
//! registry, and it needed three things a claim body makes unnecessary:
//!
//! * A `&[Ty]` of the routine's DECLARED argument types, read for exactly one
//!   thing a value cannot say — an `Ty::InPacked` field arrives as a `U32`
//!   and is not one. No `#[claims]` body in `kernels-metal` states a packed
//!   field: a body writes its own argument list, so a struct's fields are
//!   scalars like every other scalar.
//! * A STAGED BLOCK — the `constant RmsParams&` pointer the packed run was
//!   addressed through — for the same reason, gone the same way.
//! * A live `Handles` cell, because a routine could ASK for staging mid-body
//!   through `ctx.ask` and mint handles after the planner's snapshot. A claim
//!   body's staging arrives on its `Cache` marks, minted before it runs; the
//!   one door left is [`Answered`](kernels::raises::Answered), which reads a
//!   view rather than binding a buffer.

use core::cell::{Cell, RefCell};

use kernels::plane::Refusal;
use kernels_metal::plane::{ArgValue, Encode, Fire};

use super::dispatch::{Dispatch, ParamSlot, Touches};
use super::marks::{Bindings, Bound, NOTHING};
use super::walk::Cursor;

/// What a fire's dispatches are accumulated into.
///
/// `Encode::fire` takes `&self` — `kernels::routine::Backend::Ctx` reaches a
/// body as a shared reference — so the accumulation is behind a [`RefCell`].
/// That is the whole of the interior mutability: one push per dispatch.
///
/// A BODY MAY STATE MORE THAN ONE, and this keeps them in the order it stated
/// them. Not hypothetical on this plane: `rope.full` is two neox launches and
/// `mla.latents_rope` on the sibling planes is two more, and an encoder that
/// could only carry one would push the second back into the lowering.
pub struct Encoder<'b> {
    /// What every handle a body was handed stands for. Shared with the walk,
    /// which is the thing that minted them.
    bindings: &'b RefCell<Bindings>,
    /// WHICH STATEMENT IS RUNNING, shared with the walk, which is the thing
    /// that knows.
    ///
    /// FROM THE STATEMENT, NOT FROM THE BODY: where in the plan a rectangle
    /// sits is a fact a claim body has no way to know and no business
    /// stating. `Ctx` is a `dyn Encode` with no room on it to be told, so the
    /// encoder reads the fire's own cursor — exactly as it reads the fire's
    /// own bindings, and neither is passed through the body.
    cursor: &'b Cell<Cursor>,
    /// What the bodies asked for, in the order they asked.
    out: RefCell<Vec<Dispatch>>,
}

impl<'b> Encoder<'b> {
    /// An encoder over one fire's binding list and cursor.
    #[must_use]
    pub fn over(bindings: &'b RefCell<Bindings>, cursor: &'b Cell<Cursor>) -> Self {
        Self {
            bindings,
            cursor,
            out: RefCell::new(Vec::new()),
        }
    }

    /// The dispatches the fire asked for, in the order it asked.
    #[must_use]
    pub fn finish(self) -> Vec<Dispatch> {
        self.out.into_inner()
    }
}

/// What a laid-out argument run IS: the regions a dispatch binds, the slots
/// its scalars are described by, and the words themselves.
///
/// Named because the tuple is three parallel statements about ONE run, and a
/// reader who meets it bare has to reconstruct which is which.
type LaidOut = (Vec<Bound>, Vec<ParamSlot>, Vec<u32>);

/// Lay one dispatch's arguments out as the argument table and the scalar run.
///
/// Metal's argument table has ONE SLOT PER OPERAND, buffers and scalars
/// alike: a buffer's slot holds its address, and a scalar's holds nothing
/// while its bits ride the staged run and a [`ParamSlot`] joins the two. So
/// this walks the body's own list once and the position in that list IS the
/// slot.
///
/// # Errors
///
/// [`Refusal::Absent`] for a handle this fire never minted — a body reaching
/// past its own statement, which is NOT answered with a zero address. On the
/// legacy table path that answer was a live defect:
/// `mxfp4_qmv_routed_bias` read an additive bias off a null pointer for every
/// expert logit and nothing in the path said a word.
///
/// [`Refusal::Unstated`] for a raised view handed back as an argument: a view
/// is host data a body READS, not a slot it binds, so a body that passed one
/// through has stated a slot nothing can fill.
fn lay_out(values: &[ArgValue], bindings: &Bindings) -> Result<LaidOut, Refusal> {
    let mut args = Vec::with_capacity(values.len());
    let mut params: Vec<u32> = Vec::new();
    let mut slots: Vec<ParamSlot> = Vec::new();
    let mut at = 0u32;

    for (slot, value) in values.iter().enumerate() {
        match *value {
            // A `Shaped` HANDLE IS STILL A HANDLE: it carries the rectangle
            // the statement gave the operand, which the marks read and an
            // encoder does not.
            ArgValue::Buffer(handle)
            | ArgValue::BufferMut(handle)
            | ArgValue::Shaped { handle, .. } => {
                args.push(bindings.at(handle).ok_or(Refusal::Absent {
                    what: "a buffer handle this fire did not mint",
                })?);
                continue;
            }
            // Every scalar lands the same way: a zero argument slot, its bits
            // appended to the staged run, and a `ParamSlot` joining the two.
            // The kinds differ only in WIDTH, which is the difference that
            // matters — see `ParamSlot::bytes`.
            ArgValue::I32(v) => scalar(
                &mut params,
                &mut slots,
                &mut at,
                slot,
                4,
                &[v.cast_unsigned()],
            ),
            ArgValue::U32(v) => scalar(&mut params, &mut slots, &mut at, slot, 4, &[v]),
            ArgValue::F32(v) => scalar(&mut params, &mut slots, &mut at, slot, 4, &[v.to_bits()]),
            ArgValue::Usize(v) => scalar(
                &mut params,
                &mut slots,
                &mut at,
                slot,
                8,
                // Low word first: `params` is a run of `u32` the stage copies
                // verbatim into a little-endian buffer, and every Apple GPU
                // is little-endian.
                &[(v & 0xffff_ffff) as u32, (v >> 32) as u32],
            ),
            ArgValue::Raised(_) => {
                return Err(Refusal::Unstated {
                    what: "a raised view in a dispatch argument list: a view is \
                           host data a body reads, not a slot it binds",
                });
            }
        }
        args.push(NOTHING);
    }
    Ok((args, slots, params))
}

/// Append one scalar's words and the slot that binds them.
///
/// `at` advances to the value's natural alignment FIRST, so an eight-byte
/// stride starts on an eight-byte boundary rather than wherever the previous
/// four-byte extent happened to end.
fn scalar(
    params: &mut Vec<u32>,
    slots: &mut Vec<ParamSlot>,
    at: &mut u32,
    slot: usize,
    bytes: u32,
    words: &[u32],
) {
    let first = u8::try_from(params.len()).unwrap_or(u8::MAX);
    params.extend_from_slice(words);
    *at = at.next_multiple_of(bytes);
    slots.push(ParamSlot {
        slot,
        at: *at,
        bytes,
        value: first,
    });
    *at += bytes;
}

/// What a dispatch reads and what it may write, OFF THE VALUES.
///
/// The direction is on the value and always was: a claim body spells a
/// written buffer `.arg_mut()` and a read one `.arg()`, so a launch's writes
/// are exactly the handles bound at [`ArgValue::BufferMut`] and every other
/// bound handle is a read.
///
/// The conservative answer this replaces — every operand as both — was honest
/// and expensive. `Touches` decides whether an encoder may run two launches
/// at once, so calling a read a write inserts a barrier between two dispatches
/// that never meet, and on a decode where every launch takes this path that
/// is a barrier per launch rather than per hazard.
///
/// It stays conservative in the direction that matters: a handle bound to
/// nothing meets nothing, and a scalar contributes neither.
fn directed(args: &[Bound], values: &[ArgValue]) -> Touches {
    let mut touches = Touches::default();
    for (i, arg) in args.iter().enumerate() {
        if arg.slice.is_nothing() {
            continue;
        }
        match values.get(i) {
            Some(ArgValue::BufferMut(_)) => touches.writes.push(arg.slice),
            _ => touches.reads.push(arg.slice),
        }
    }
    touches
}

impl Encode for Encoder<'_> {
    /// The `Asks` door, and this plane answers exactly one question through
    /// it.
    ///
    /// `Asks::absent()` resolves `(Ty::Buf, Source::Lit(Lit::Null))` and is
    /// what a body says when the entrypoint declares a buffer the point does
    /// not carry — `attn/kv_write.metal` names six that belong to a shared
    /// ring ABI this append does not use. That is a real slot with a real
    /// number, so it is answered with a real handle onto nothing: the
    /// encoder binds a zero-length region and a shader that reads it faults
    /// loudly rather than reading a neighbour.
    ///
    /// Everything else refuses. `Asks::param(n)` reaches for the STATEMENT's
    /// scalar run, which is the routine era's way of taking an operand a
    /// declaration does not state; a claim body is handed its scalars.
    fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
        match (ty, source) {
            (kernels::Ty::Buf, kernels::Source::Lit(kernels::Lit::Null)) => {
                Ok(ArgValue::Buffer(self.bindings.borrow_mut().take(NOTHING)))
            }
            _ => Err(Refusal::Unstated {
                what: "a value asked for off the fire: a claim body is handed \
                       every operand its point declares",
            }),
        }
    }

    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
        // A body with nothing to do should have refused already. A zero here
        // would become `dispatchThreads:` over an empty grid, which runs
        // nothing and reports success — so the buffer keeps whatever it held
        // and the model answers from stale bytes. The two are told apart on
        // purpose: `Refusal::Empty` is a body that noticed, and this is a
        // body that computed an extent and got zero.
        if fire.lanes.contains(&0) {
            return Err(Refusal::Grid {
                what: "the threads a claim asked for",
                at: 0,
            });
        }
        // Metal STATES its threadgroup rather than reflecting it — MSL
        // declares no workgroup size — so nothing else in the path would
        // catch a zero, and `threadsPerThreadgroup` of zero is undefined
        // rather than empty.
        if fire.group.contains(&0) {
            return Err(Refusal::Grid {
                what: "the threadgroup a claim asked for",
                at: 0,
            });
        }
        let (bound, param_slots, params) = lay_out(args, &self.bindings.borrow())?;
        let at = self.cursor.get();
        self.out.borrow_mut().push(Dispatch {
            symbol: fire.entrypoint,
            file: fire.file,
            stamp: fire.stamp,
            grid: fire.lanes,
            threadgroup: fire.group,
            touches: directed(&bound, args),
            args: bound,
            param_slots,
            params,
            layers: at.layer..at.layer + 1,
            op: at.op,
        });
        Ok(())
    }
}
