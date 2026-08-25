//! The door a claim body fires through, and the only thing on this plane that
//! `kernels_vulkan::plane::Ctx` actually is.
//!
//! `Ctx<'a>` is `dyn Encode + 'a` — a TRAIT OBJECT, where cuda's `Ctx` is a
//! struct holding a JIT handle and a stream. That is the whole shape of the
//! difference between the two kinds of plane, and it is why
//! `kernels::bound::BoundOp::Plane` carries `+ ?Sized`: what a shader fire
//! talks to is an encoder the DRIVER owns, and the kernel crate never names a
//! concrete one.
//!
//! # It plans; it does not encode
//!
//! [`Encoder::fire`] turns a body's statement — *this entrypoint, over these
//! invocations, with these arguments* — into a [`Dispatch`] and pushes it. It
//! touches no `vk::CommandBuffer`, which is the crate header's own rule: *"A
//! fire is a few hundred rectangles and this driver plans all of them, then
//! records them into ONE command buffer with barriers only between the pairs
//! that touch the same bytes."* A body that recorded as it went would take the
//! barrier decision away, because the pairs are not knowable until the list is.
//!
//! It is also what makes the walk testable.
//! `tests/the_walk_is_the_program.rs` hands the walk an `Encode` that records
//! instead of planning, and nothing in [`crate::walk`] can tell the difference
//! — because none of it has ever seen an adapter.
//!
//! # It is NOT `crate::encode::Encoder`, and the two differ in what they KNOW
//!
//! That one is the legacy walk's and it holds a `Reflect`: it looks the module
//! up as the body fires, reads `OpExecutionMode LocalSize` back out, divides
//! the lanes, and asks `crate::binding::params_from` where the scalars go. All
//! three of those need the compiled SPIR-V, so all three make an encoder that
//! cannot run without one.
//!
//! This one knows none of it. A [`Dispatch`] here carries LANES undivided and a
//! WORD RUN unplaced, and the device half does the reflecting once it has the
//! whole fire — which is the same thing it wanted anyway, since it batch-builds
//! every pipeline the list names. What that buys is the portable half: this
//! module compiles with `default = []`, names no `ash` type, and is what a walk
//! test drives.
//!
//! # The two divergences from `driver-metal/src/baker/encode.rs`
//!
//! * **A zero workgroup is not refused, because a body never states one.**
//!   Metal refuses `fire.group.contains(&0)` and must, since MSL declares no
//!   workgroup size. Every `kernels-vulkan` body passes a bare `[u32; 3]` to
//!   `Fire::apply`, which sets `lanes` and leaves `group` at `[0, 0, 0]` — so
//!   porting metal's second guard would refuse every claim on this plane. The
//!   workgroup size is declared by `[numthreads]` in the Slang and recovered
//!   from the module.
//!
//! * **A zero LANE count IS refused, and this backend has paid for it.**
//!   `crate::encode` records the incident in one sentence: a zero would become
//!   `vkCmdDispatch(0, 1, 1)`, *"which is legal Vulkan that runs nothing and
//!   reports success over a buffer that kept its zeros. This backend has paid
//!   for that once: a shared expert's gate came back untouched and every routed
//!   token was combined under `sigmoid(0)`."*

use core::cell::{Cell, RefCell};

use kernels::plane::Refusal;
use kernels_vulkan::Capability;
use kernels_vulkan::plane::{ArgValue, Encode, Fire};

use super::dispatch::{Dispatch, Touches, merge};
use super::marks::{Bindings, Bound};
use super::walk::Cursor;

/// What a fire's dispatches are accumulated into.
///
/// `Encode::fire` takes `&self` — `kernels::plane::Backend::Ctx` reaches a body
/// as a shared reference — so the accumulation is behind a [`RefCell`]. That is
/// the whole of the interior mutability: one push per dispatch.
///
/// A BODY MAY STATE MORE THAN ONE, and this keeps them in the order it stated
/// them. Not hypothetical on this plane: `attention.decode`'s split arm is two
/// launches — the split pass and the fold — and an encoder that could only
/// carry one would push the second back into the lowering.
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
    /// THE TIER THIS ADAPTER ADVERTISES, and the one thing on this encoder that
    /// is a fact about a device.
    ///
    /// A body composes its module name from it (`module::path(entrypoint,
    /// self.best())`), which steps DOWN to the tier the build compiled. It is a
    /// plain value rather than a device query because that is all it ever was:
    /// a ceiling read once when the adapter was opened. A walk test states
    /// [`Capability::Baseline`] and gets the baseline artifacts, which is the
    /// honest answer for a process with no adapter in it.
    best: Capability,
    /// What the bodies asked for, in the order they asked.
    out: RefCell<Vec<Dispatch>>,
}

impl<'b> Encoder<'b> {
    /// An encoder over one fire's binding list and cursor, at a stated tier.
    #[must_use]
    pub fn over(
        bindings: &'b RefCell<Bindings>,
        cursor: &'b Cell<Cursor>,
        best: Capability,
    ) -> Self {
        Self {
            bindings,
            cursor,
            best,
            out: RefCell::new(Vec::new()),
        }
    }

    /// The dispatches the fire asked for, in the order it asked.
    #[must_use]
    pub fn finish(self) -> Vec<Dispatch> {
        self.out.into_inner()
    }
}

/// What a laid-out argument run IS: the regions a dispatch binds, the hazard
/// set they make, and the scalar words beside them.
///
/// Named because the tuple is three parallel statements about ONE run, and a
/// reader who meets it bare has to reconstruct which is which.
type LaidOut = (Vec<Bound>, Touches, Vec<u32>);

/// Lay one dispatch's arguments out as its descriptor run and its scalar words.
///
/// ONE PASS, TWO DESTINATIONS, sorted by VARIANT. `ArgValue::Buffer` takes a
/// descriptor and every other variant is a word of the scalar run — which is
/// `crate::encode`'s reading and not a new one: *"a routine hands over its
/// argument list in signature order and this reads the split off the
/// VARIANTS."* Whether the words end up as push constants or as a struct in a
/// storage buffer is the MODULE's decision, read by
/// `crate::binding::params_from` off the reflected declaration — and the
/// reachable symbols split almost evenly on it, so neither answer could be
/// assumed.
///
/// # The word convention is not this function's invention
///
/// One word per scalar in signature order, TWO for a `Usize`, low first, and
/// the pair aligned to an even word. `crate::encode::words` states the whole
/// argument; the short form is that nothing in this shader tree declares a
/// 64-bit integer, so an extent arrives as two `uint`s, and `PIE_STRIDE` is
/// `uint2` whose push-constant alignment is eight bytes.
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
    let mut touches = Touches::default();
    // Sized rather than grown: a `Usize` is two words and the rest are one, so
    // `len + 1` is the answer for every call this tree makes.
    let mut params: Vec<u32> = Vec::with_capacity(values.len() + 1);

    for value in values {
        match *value {
            ArgValue::Buffer { handle, writes, .. } => {
                let bound = bindings.at(handle).ok_or(Refusal::Absent {
                    what: "a buffer handle this fire did not mint",
                })?;
                // THE DIRECTION IS THE BINDING'S ON THIS PLANE, which is what
                // lets a hazard set exist at all: `.arg_mut()` produces
                // `writes: true` and `.arg()` produces `writes: false`. On
                // wgpu the direction is the SHADER's (`var<storage, read>`
                // against `read_write`) and there is nothing for a driver to
                // say about it — see `super::dispatch`.
                //
                // A WRITE IS ALSO A READ. A shader that writes through a
                // binding may read it first (`logit_softcap` does, in place),
                // and SPIR-V does not carry the `readonly` qualifier usefully
                // in this tree, so the conservative half of the pair is the
                // only honest one.
                merge(&mut touches.reads, bound.slice);
                if writes {
                    merge(&mut touches.writes, bound.slice);
                }
                args.push(bound);
            }
            ArgValue::I32(v) => params.push(v.cast_unsigned()),
            ArgValue::U32(v) => params.push(v),
            ArgValue::F32(v) => params.push(v.to_bits()),
            ArgValue::Usize(v) => {
                if params.len() % 2 == 1 {
                    params.push(0);
                }
                params.push((v & 0xffff_ffff) as u32);
                params.push((v >> 32) as u32);
            }
            ArgValue::Raised(_) => {
                return Err(Refusal::Unstated {
                    what: "a raised view in a dispatch argument list: a view is \
                           host data a body reads, not a slot it binds",
                });
            }
        }
    }
    Ok((args, touches, params))
}

impl Encode for Encoder<'_> {
    /// The tier a body composes its module name from. See [`Encoder::best`].
    fn best(&self) -> Capability {
        self.best
    }

    /// THE BY-NAME DOOR, AND THIS EXECUTOR HAS NO NAMES TO ANSWER WITH.
    ///
    /// `Staged::stream` asks for a tier-1 runtime plane by the name it is
    /// staged under — `"positions"`, `"request_of_token"`. The legacy encoder
    /// answers it off a `FireTable` it holds; the baker holds no such table,
    /// because a walk hands a claim body every operand its point declares and
    /// the per-fire planes ride the POOL ROW: `Plane::Pages` is
    /// `Struct<AttnFire>` here, so `positions` and `request_of_token` arrive
    /// as fields of the view a `Cache` mark already carries.
    ///
    /// So this is a refusal on purpose rather than a gap. A body that reaches
    /// for a name on this path is asking for the one crossing this executor
    /// was built to remove, and the refusal says which name it asked for.
    fn staged(&self, name: &'static str) -> Result<u32, Refusal> {
        let _ = name;
        Err(Refusal::Unstated {
            what: "a runtime plane asked for BY NAME: the walk hands a body \
                   every operand its point declares, and the per-fire planes \
                   ride the pool row",
        })
    }

    /// A window onto a handle this fire already minted.
    ///
    /// The arithmetic is the whole of it: a packed row's second half is the
    /// same allocation at an offset, and `Slice::span` refuses an offset that
    /// runs past the end rather than clamping — a clamped window is a shorter
    /// rectangle that computes and answers wrongly.
    ///
    /// WHAT IS NOT CHECKED HERE, and must be where a descriptor is written:
    /// the device's `minStorageBufferOffsetAlignment`. A window at an
    /// unaligned byte is a binding the card refuses, and this executor cannot
    /// reach a command buffer yet — the driver's device half does not compile.
    /// The check belongs beside the write, not beside the arithmetic, which is
    /// where `crate::encode`'s twin puts it.
    fn windowed(&self, of: u32, at: u64) -> Result<u32, Refusal> {
        let whole = self.bindings.borrow().at(of).ok_or(Refusal::Absent {
            what: "the allocation a window opens on: this fire minted no \
                       such handle",
        })?;
        let rest = whole
            .slice
            .bytes
            .checked_sub(at)
            .filter(|left| *left > 0)
            .ok_or(Refusal::Empty {
                what: "the range left past the byte a window opens at",
            })?;
        let slice = whole.slice.span(at, rest).ok_or(Refusal::Empty {
            what: "a window that runs past the allocation it opens on",
        })?;
        Ok(self.bindings.borrow_mut().take(Bound {
            slice,
            width: whole.width,
        }))
    }

    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
        // A body with nothing to do should have refused already. See this
        // module's header for what a zero cost this backend once.
        if fire.lanes.contains(&0) {
            return Err(Refusal::Grid {
                what: "the invocations a claim asked for",
                at: 0,
            });
        }
        // NO SECOND GUARD ON `fire.group`. Metal has one and needs it; here a
        // zero group is what every claim body states, because the workgroup
        // size is declared by `[numthreads]` in the Slang.
        let (args, touches, params) = lay_out(args, &self.bindings.borrow())?;
        let at = self.cursor.get();
        self.out.borrow_mut().push(Dispatch {
            symbol: fire.entrypoint,
            file: fire.file,
            stamp: fire.stamp,
            lanes: fire.lanes,
            args,
            touches,
            params,
            layer: at.layer,
            op: at.op,
        });
        Ok(())
    }
}
