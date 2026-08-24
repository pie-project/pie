//! The door a claim body fires through, and the only thing on this plane that
//! `kernels_wgpu::routine::Ctx` actually is.
//!
//! `Ctx<'a>` is `dyn Encode + 'a` — a TRAIT OBJECT, where cuda's `Ctx` is a
//! struct holding a JIT handle and a stream. That is the whole shape of the
//! difference between the two planes, and it is why `BoundOp::Plane` carries
//! `+ ?Sized`: what a shader fire talks to is an encoder the DRIVER owns, and
//! the kernel crate never names a concrete one.
//!
//! # It plans; it does not encode
//!
//! [`Encoder::fire`] turns a body's statement — *this entrypoint, over these
//! invocations, with these arguments* — into a [`Dispatch`] and pushes it. It
//! does not touch a `wgpu::CommandEncoder`, and that is deliberate rather than
//! incidental: the device half wants the WHOLE fire before it starts, to
//! batch-compile every pipeline it names and to size one uniform staging buffer
//! for it. A body that encoded as it went would take both away.
//!
//! It is also what makes the walk testable. `tests/the_walk_is_the_program.rs`
//! hands the walk an `Encode` that records instead of planning, and nothing in
//! `baker::walk` or `baker::bound` can tell the difference — because neither of
//! them has ever seen an adapter.
//!
//! # The two divergences from `driver-metal/src/baker/encode.rs`
//!
//! Both come off the same fact — a WGSL module DECLARES what an MSL one leaves
//! to the driver — and both are refusals that had to be dropped rather than
//! ported:
//!
//! * **A zero threadgroup is not refused, because a body never states one.**
//!   Metal refuses `fire.group.contains(&0)` and must, since MSL declares no
//!   workgroup size and `threadsPerThreadgroup` of zero is undefined. Every
//!   `kernels-wgpu` body passes a bare `[u32; 3]` to `Fire::apply`, which sets
//!   `lanes` and leaves `group` at `[0, 0, 0]` — so porting metal's second
//!   guard would refuse every claimed point on this plane. The workgroup size
//!   is the MODULE's and is read off it by `src/encode.rs`.
//!
//! * **There is no `directed()` and no `Touches`.** Metal reads a dispatch's
//!   writes off `ArgValue::BufferMut`; this plane's `ArgValue` HAS NO SUCH
//!   VARIANT — `kernels_wgpu::routine`'s `buffer_at` and `buffer_mut_at` both
//!   produce `Shaped`, because on WebGPU the direction is the SHADER's
//!   (`var<storage, read>` against `read_write`) and not the binding's. There
//!   is nothing for a driver to say about it and nothing that would read the
//!   answer: see [`super::dispatch`] on why the barriers are not this driver's
//!   to place.
//!
//! # What the legacy `Planner` carried that this does not
//!
//! `lowering::routine::Planner` was the same idea against the routine registry.
//! What a claim body makes unnecessary is the live `Handles` cell it read
//! operands through: a routine could ASK for staging mid-body and mint handles
//! after the planner's snapshot, where a claim body's staging arrives on its
//! `Cache` marks, minted before it runs.

use core::cell::{Cell, RefCell};

use kernels::plane::Refusal;
use kernels_wgpu::plane::{ArgValue, Encode, Fire};

use super::dispatch::{Dispatch, ParamSlot};
use super::marks::{Bindings, Bound, NOTHING};
use super::walk::Cursor;

/// What a fire's dispatches are accumulated into.
///
/// `Encode::fire` takes `&self` — `kernels::routine::Backend::Ctx` reaches a
/// body as a shared reference — so the accumulation is behind a [`RefCell`].
/// That is the whole of the interior mutability: one push per dispatch.
///
/// A BODY MAY STATE MORE THAN ONE, and this keeps them in the order it stated
/// them. Not hypothetical on this plane: `rope.full` is two neox launches, and
/// an encoder that could only carry one would push the second back into the
/// lowering.
pub struct Encoder<'b> {
    /// What every handle a body was handed stands for. Shared with the walk,
    /// which is the thing that minted them.
    bindings: &'b RefCell<Bindings>,
    /// WHICH STATEMENT IS RUNNING, shared with the walk, which is the thing
    /// that knows.
    ///
    /// FROM THE STATEMENT, NOT FROM THE BODY: where in the plan a rectangle
    /// sits is a fact a claim body has no way to know and no business stating.
    /// `Ctx` is a `dyn Encode` with no room on it to be told, so the encoder
    /// reads the fire's own cursor — exactly as it reads the fire's own
    /// bindings, and neither is passed through the body.
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

/// What a laid-out argument run IS: the regions a dispatch binds, the fields
/// its scalars occupy in the uniform block, and the words themselves.
///
/// Named because the tuple is three parallel statements about ONE run, and a
/// reader who meets it bare has to reconstruct which is which.
type LaidOut = (Vec<Bound>, Vec<ParamSlot>, Vec<u32>);

/// Lay one dispatch's arguments out as its bind-group entries and its uniform
/// block.
///
/// THE TWO GROUPS ARE SPLIT HERE, and that split is this plane's whole argument
/// convention. Metal's argument table has one slot per operand, buffers and
/// scalars alike, and its `lay_out` walks one list where the position IS the
/// slot. WebGPU has no such table: a buffer becomes an entry of `@group(0)`
/// numbered by how many buffers came before it, and a scalar becomes a FIELD of
/// the one uniform block at `@group(1) @binding(0)`. So this walks the body's
/// list once and sorts each value into whichever of the two it belongs to —
/// which means the binding index of a buffer is its index among the BUFFERS,
/// not its index in the body's argument list.
///
/// `kernels_wgpu`'s shaders are written to exactly that convention, and
/// `norm/rms.wgsl` is the one to read: bindings 0, 1 and 2 are `x`, `w` and
/// `out_`, and `Params { eps, axis_size, w_stride, plus_one, gain }` is packed
/// by walking the fire's argument list in order — the shader's own comment
/// warns that "word 0 is the epsilon, word 4 is the gain, and a body that
/// reordered them would read a gain as an epsilon with no error anywhere".
///
/// # Errors
///
/// [`Refusal::Absent`] for a handle this fire never minted — a body reaching
/// past its own statement, which is NOT answered with an empty binding. On the
/// legacy table path that answer was a live defect: `mxfp4_qmv_routed_bias`
/// read an additive bias off a null pointer for every expert logit and nothing
/// in the path said a word.
///
/// [`Refusal::Unstated`] for a raised view handed back as an argument: a view
/// is host data a body READS, not a slot it binds, so a body that passed one
/// through has stated a slot nothing can fill.
fn lay_out(values: &[ArgValue], bindings: &Bindings) -> Result<LaidOut, Refusal> {
    let mut args = Vec::with_capacity(values.len());
    let mut params: Vec<u32> = Vec::new();
    let mut slots: Vec<ParamSlot> = Vec::new();
    let mut at = 0u32;

    for value in values {
        match *value {
            // A `Shaped` HANDLE IS STILL A HANDLE: it carries the rectangle
            // the statement gave the operand, which the marks read and an
            // encoder does not. There is no `BufferMut` on this plane — see
            // this module's header.
            ArgValue::Buffer(handle) | ArgValue::Shaped { handle, .. } => {
                args.push(bindings.at(handle).ok_or(Refusal::Absent {
                    what: "a buffer handle this fire did not mint",
                })?);
            }
            // Every scalar lands the same way: its bits appended to the block
            // and a `ParamSlot` saying where they went. The kinds differ only
            // in WIDTH, which is the difference that matters — see
            // `ParamSlot::bytes`.
            ArgValue::I32(v) => scalar(&mut params, &mut slots, &mut at, 4, &[v.cast_unsigned()]),
            ArgValue::U32(v) => scalar(&mut params, &mut slots, &mut at, 4, &[v]),
            ArgValue::F32(v) => scalar(&mut params, &mut slots, &mut at, 4, &[v.to_bits()]),
            ArgValue::Usize(v) => scalar(
                &mut params,
                &mut slots,
                &mut at,
                8,
                // Low word first: `Lang::USIZE` is `vec2<u32>` and `params` is
                // a run of `u32` the stage copies verbatim into a
                // little-endian buffer.
                &[(v & 0xffff_ffff) as u32, (v >> 32) as u32],
            ),
            ArgValue::Raised(_) => {
                return Err(Refusal::Unstated {
                    what: "a raised view in a dispatch argument list: a view is \
                           host data a body reads, not a slot it binds",
                });
            }
        }
    }
    Ok((args, slots, params))
}

/// Append one scalar's words and the field that holds them.
///
/// `at` advances to the value's natural alignment FIRST, so an eight-byte
/// `vec2<u32>` starts on an eight-byte boundary rather than wherever the
/// previous four-byte field happened to end. That is WGSL's own rule for a
/// uniform struct's members, so the offsets this computes are the ones
/// `src/reflect.rs` reads back off the module.
fn scalar(
    params: &mut Vec<u32>,
    slots: &mut Vec<ParamSlot>,
    at: &mut u32,
    bytes: u32,
    words: &[u32],
) {
    let first = u8::try_from(params.len()).unwrap_or(u8::MAX);
    params.extend_from_slice(words);
    *at = at.next_multiple_of(bytes);
    slots.push(ParamSlot {
        at: *at,
        bytes,
        value: first,
    });
    *at += bytes;
}

impl Encode for Encoder<'_> {
    /// The `Asks` door, and this plane answers exactly one question through it.
    ///
    /// `Asks::absent()` resolves `(Ty::Buf, Source::Lit(Lit::Null))` and is
    /// what a body says when the entrypoint declares a buffer the point does
    /// not carry — six of `kernels_wgpu::attn`'s sdpa arms say it, through that
    /// crate's own `points::absent` wrapper. That is a real binding with a real
    /// number, so it is answered with a real handle onto nothing: the encoder
    /// binds a zero-size binding and a shader that reads it reads nothing
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
        // would become `dispatch_workgroups(0, 1, 1)`, which `src/encode.rs`
        // already calls "legal WebGPU that runs nothing and reports success" —
        // so the buffer keeps whatever it held and the model answers from stale
        // bytes. The two are told apart on purpose: `Refusal::Empty` is a body
        // that noticed, and this is a body that computed an extent and got
        // zero.
        if fire.lanes.contains(&0) {
            return Err(Refusal::Grid {
                what: "the invocations a claim asked for",
                at: 0,
            });
        }
        // NO SECOND GUARD ON `fire.group`. Metal has one and needs it; here a
        // zero group is what every claim body states, because the workgroup
        // size is the MODULE's. See this file's header.
        let (bound, param_slots, params) = lay_out(args, &self.bindings.borrow())?;
        let at = self.cursor.get();
        self.out.borrow_mut().push(Dispatch {
            symbol: fire.entrypoint,
            file: fire.file,
            stamp: fire.stamp,
            lanes: fire.lanes,
            args: bound,
            param_slots,
            params,
            layer: at.layer,
            op: at.op,
        });
        Ok(())
    }
}
