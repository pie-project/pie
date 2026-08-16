//! This backend's instantiation of the `kernels` routine machinery.
//!
//! A routine is an ordinary `fn` that computes a launch and dispatches it.
//! Everything a row used to STATE — the launch rule, the grid parameter, the
//! head parameter, the operand list — is code in the body, and the table row
//! is derived from the signature by [`routine!`].
//!
//! This file is a PORT of `kernels-wgpu/src/routine.rs`, not a design. Two
//! backends with the same constraint answering it two ways is the drift this
//! whole refactor exists to remove, so where Vulkan has no reason to differ it
//! does not. `.wiki/kernel-x/vulkan-refactor.md` §4 and §5 record which of
//! these decisions are wgpu's and which are this backend's.
//!
//! # Why the context is a trait and not a struct
//!
//! `kernels-cuda-new`'s `Ctx` is a struct: it owns the NVRTC cache and the
//! cuBLAS handles, so the crate that declares the routines also launches them.
//!
//! This crate cannot be that. Its `[dependencies]` is `kernels` and nothing
//! else — `ash` is a dev-dependency, deliberately — while the `ash::Device`,
//! the command buffer being recorded, the descriptor pool and every
//! `vk::Pipeline` live in `driver-vulkan`. So the thing a body dispatches
//! through is a TRAIT the driver implements ([`Encode`]), and `Backend::Ctx`
//! is `dyn Encode`. The `?Sized` bound that makes this legal is already on
//! the shared machinery; `kernels-wgpu` needed it first, for this reason.
//!
//! # What is Vulkan's alone
//!
//! **The tier does not appear anywhere here.** One entrypoint compiles to up
//! to three `.spv` modules — `Baseline`, `Fp16`, `Coopmat` — and
//! [`crate::Capability::module`] keys them by entrypoint name plus a tag. A
//! body names the ENTRYPOINT; the driver walks `Capability::PREFERENCE` and
//! takes the best tier the device advertises. A body that could name a tier
//! could name one the device lacks, and this backend has already paid for
//! that once: an unadvertised cooperative-matrix shape is a SIGSEGV inside
//! `vkCreateComputePipelines` with the validation layer entirely silent.
//!
//! There is also no `module` field beside the entrypoint, which wgpu needs and
//! this backend does not: a WGSL file holds many entrypoints, while `build.rs`
//! here emits one `.spv` per entrypoint per tier, so the entrypoint alone is
//! the key.

#[cfg(test)]
use kernels::Ty;
#[cfg(test)]
use kernels::routine::Arg;
use kernels::routine::{Backend, Extent, Refusal};
use kernels::shader::ShaderValue;

/// This backend, as the machinery names it.
///
/// A marker: never constructed, carrying only the two concrete types the
/// `kernels` machinery is generic over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vulkan;

impl Backend for Vulkan {
    type Value = ArgValue;
    type Ctx<'a> = dyn Encode + 'a;

    // A region is `{address, rows, width}`; this backend's bound value is an
    // address alone. The other two already live per-launch in `Facts`
    // (`bind/table.rs:592`), so this refusal stays unreachable until a table
    // asks for a fat `In<N, _>` — which then fails at first fire, not compile time.
    fn region(_value: &ArgValue, _at: usize) -> Result<Extent, Refusal> {
        Err(Refusal::Absent { what: "a region's shape: the vulkan binder binds addresses only" })
    }
}

/// One value a caller supplies for one argument.
///
/// The scalar kinds are separate variants rather than one integer because the
/// widths differ and the check that matters is exactly the width one: a
/// [`Ty::Usize`] value handed to a [`Ty::I32`] argument is eight bytes going
/// into a four-byte slot, which either truncates or writes over its neighbour
/// depending on where in the push block it lands. Vulkan's push range is
/// packed by the driver from this list, so nothing downstream would report it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    /// A device allocation, named by whatever index the caller keys its own
    /// buffers on, and whether the launcher may WRITE through it.
    ///
    /// The handle is opaque on purpose: this crate cannot name a `vk::Buffer`
    /// and does not need to. The driver binds handles to its own buffers
    /// before it calls, and a routine only passes them through.
    ///
    /// The flag is Vulkan's alone -- `ShaderValue::buffer_mut` defaults to
    /// dropping it and the other two backends take the default. This driver
    /// puts a barrier between two dispatches only when they touch the same
    /// bytes and decides that from writability; barriering every neighbouring
    /// pair costs 8 microseconds each on this card, measured at 3.8 ms of a
    /// 7.2 ms decode. Under `kernel!` the fact came off the row's operand
    /// types, and a routine states it in the argument TYPE instead -- so the
    /// `Buf`/`BufMut` distinction has to survive the trip through a value,
    /// which is all the driver sees.
    Buffer {
        /// The caller's index for the allocation.
        handle: u32,
        /// Whether the shader may write through this binding.
        writes: bool,
    },
    /// A 32-bit signed scalar.
    I32(i32),
    /// A 32-bit unsigned scalar.
    U32(u32),
    /// A 32-bit float.
    F32(f32),
    /// A 64-bit stride or extent, which is what [`Ty::Usize`] means here.
    Usize(u64),
}

impl ArgValue {
    /// What this value is, for a refusal to name.
    #[must_use]
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Buffer { .. } => "a buffer",
            Self::I32(_) => "an i32",
            Self::U32(_) => "a u32",
            Self::F32(_) => "an f32",
            Self::Usize(_) => "a usize",
        }
    }
}

/// One dispatch, as a routine body states it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fire<'a> {
    /// The entrypoint, whole, as its `// pie:instantiate` line spells it.
    ///
    /// A body that varies over an instantiation axis picks the spelling
    /// itself; nothing pastes a suffix on afterwards, which is what `kernel!`
    /// did with `axes`. The TIER is not part of it — see the module docs.
    pub entrypoint: &'a str,
    /// LANES in each dimension — elements of work, not workgroups.
    ///
    /// The division into workgroups is the driver's, because the divisor is a
    /// property of the shader TEXT rather than of the launch: `[numthreads]`
    /// is declared in the `.slang`, lands in the SPIR-V as `OpExecutionMode
    /// LocalSize`, and `driver-vulkan/src/spirv.rs` already recovers it and
    /// already calls it "the divisor a grid is built with". A body that
    /// divided by it would carry a second copy of a number it cannot see.
    ///
    /// A body must not state a zero here. `vkCmdDispatch(0, 1, 1)` is legal
    /// Vulkan that runs nothing and reports success — the failure this whole
    /// surface exists to make impossible, and one this backend has met twice:
    /// a truncated group count left a shared expert's gate at its buffer's
    /// zeros, and every routed token was then combined under `sigmoid(0)`. A
    /// body with nothing to do returns [`Refusal::Empty`] instead.
    pub lanes: [u32; 3],
}

/// What a routine body dispatches through.
///
/// Implemented by `driver-vulkan`. The split is the crate boundary: a body
/// knows the entrypoint and the extent, and the driver knows what a buffer is,
/// which descriptor set a binding lands in, how the push block is packed, and
/// which capability tier this device can be given.
pub trait Encode {
    /// Run one dispatch.
    ///
    /// `args` is the routine's own argument list, in signature order — which
    /// is the SHADER's binding order, not the trace's. The two differ for
    /// 2,898 of this tree's 3,992 rectangles, and reconciling them is the
    /// caller's job, not this one's.
    ///
    /// The implementor separates buffers from scalars by variant, which is
    /// what makes the split derivable rather than stated twice.
    ///
    /// # Errors
    ///
    /// Whatever the device or the binding refused, as a [`Refusal`].
    fn dispatch(&self, fire: Fire<'_>, args: &[ArgValue]) -> Result<(), Refusal>;
}

/// This backend's value, as the shared operand types read it.
///
/// The ten operand types live in [`kernels::shader`], not per-backend, because
/// the vocabulary is closed and identical across metal/vulkan/wgpu. What is
/// NOT shared is the value, [`ArgValue`]; this impl is the whole of what a
/// shared operand type needs to know about it.
impl ShaderValue for ArgValue {
    fn as_buffer(self) -> Option<u32> {
        match self {
            Self::Buffer { handle, .. } => Some(handle),
            _ => None,
        }
    }
    fn as_i32(self) -> Option<i32> {
        match self {
            Self::I32(v) => Some(v),
            _ => None,
        }
    }
    fn as_u32(self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(v),
            _ => None,
        }
    }
    fn as_f32(self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }
    fn as_usize(self) -> Option<u64> {
        match self {
            Self::Usize(v) => Some(v),
            _ => None,
        }
    }
    fn buffer(handle: u32) -> Self {
        Self::Buffer {
            handle,
            writes: false,
        }
    }
    fn buffer_mut(handle: u32) -> Self {
        Self::Buffer {
            handle,
            writes: true,
        }
    }
    fn i32(v: i32) -> Self {
        Self::I32(v)
    }
    fn u32(v: u32) -> Self {
        Self::U32(v)
    }
    fn f32(v: f32) -> Self {
        Self::F32(v)
    }
    fn usize(v: u64) -> Self {
        Self::Usize(v)
    }
}

/// How Slang spells the twelve, as this tree's shaders declare them.
///
/// Every string is read out of `kernels/`, not invented; `PIE_ACT` is the
/// activation type the tier substitutes, which is why the two opaque
/// buffers name it rather than a concrete element type.
///
/// `USIZE` is empty on purpose: Slang has `uint64_t` under the shader-int64
/// capability, but nothing in this tree declares one — every live use is an
/// extent the driver resolves and narrows before it reaches a push block.
impl kernels::shader::Lang for Vulkan {
    const BUF: &'static str = "StructuredBuffer<PIE_ACT>";
    const BUF_MUT: &'static str = "RWStructuredBuffer<PIE_ACT>";
    const I32S: &'static str = "StructuredBuffer<int>";
    const U32S: &'static str = "StructuredBuffer<uint>";
    const U8S: &'static str = "StructuredBuffer<uint8_t>";
    const F32S: &'static str = "StructuredBuffer<float>";
    const F32S_MUT: &'static str = "RWStructuredBuffer<float>";
    const I32: &'static str = "int";
    const U32: &'static str = "uint";
    const F32: &'static str = "float";
    const USIZE: &'static str = "";
    const IN_PACKED: &'static str = "uint";
}

/// The operand vocabulary, from the crate that holds it once.
///
/// Re-exported rather than named through `kernels::shader` at every use, so a
/// body's signature reads as this backend's own and a family file imports one
/// module.
pub use kernels::shader::{Bind, Buf, BufMut, F32s, F32sMut, I32s, InPacked, U8s, U32s, Usize};

/// What a routine body dispatches through, spelled as a body writes it.
///
/// `dyn Encode + 'a`, and the lifetime is why [`Backend::Ctx`] is a generic
/// associated type rather than a plain one. A Vulkan `Encode` BORROWS
/// everything it needs — the `ash::Device`, the command buffer mid-recording,
/// the descriptor pool, the pipeline cache — all of which live in the
/// driver's frame, so an implementor is never `'static` and a bare
/// `dyn Encode` (which means `dyn Encode + 'static`) could not name one.
/// Bodies take `&Ctx<'_>`.
pub type Ctx<'a> = dyn Encode + 'a;

/// One routine, in this backend's instantiation of the machinery.
pub type Routine = kernels::routine::Routine<Vulkan>;

/// The backend's wrapper over [`kernels::routine!`], with [`Vulkan`] filled in
/// so a declaration names only the `fn`:
///
/// ```ignore
/// pub static ROUTINES: &[Routine] = &[
///     routine!(rms_single_row),
///     routine!(neox_decode, in_place = &[(0, 0)]),
/// ];
/// ```
#[macro_export]
macro_rules! routine {
    ($body:ident $(, $($fact:tt)*)?) => {
        ::kernels::routine!($crate::routine::Vulkan, $body $(, $($fact)*)?)
    };
}

/// `LaunchRule::Elementwise`: one lane per element of the whole rectangle, on
/// one axis.
///
/// The two launch rules this crate's elementwise kernels use are written here
/// once rather than in each family, because they are not per-family facts:
/// they are the two shapes `driver-vulkan`'s `geometry::lanes` has always
/// computed, and the point of a body stating its own grid is that the
/// statement is the same one, not a new one.
///
/// A lane count is in THREADS. The division into workgroups belongs to
/// whoever knows `[numthreads]`, which is the SPIR-V and so the driver, and
/// this crate deliberately does not reflect it -- see
/// `driver-vulkan::encode`, which does the `div_ceil`.
///
/// # Errors
///
/// [`Refusal::Empty`] when either extent is zero or negative, and
/// [`Refusal::Grid`] when the product does not fit the `u32` that
/// `vkCmdDispatch` takes.
///
/// Both are refusals rather than clamps because both fail SILENTLY otherwise.
/// `vkCmdDispatch(0, 1, 1)` runs nothing and returns success over a buffer
/// that keeps whatever it held; a product that wrapped covers a fraction of
/// the rectangle and also returns success. An extent of zero arrives here
/// honestly -- a routed expert that won no tokens has zero rows -- so this is
/// a value the caller reads, not a panic.
pub fn elementwise(width: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    let [w, r] = rectangle(width, rows)?;
    let n = u64::from(w) * u64::from(r);
    let n = u32::try_from(n).map_err(|_| Refusal::Grid {
        what: "width * rows",
        at: i64::try_from(n).unwrap_or(i64::MAX),
    })?;
    Ok([n, 1, 1])
}

/// `LaunchRule::ElementwiseRows`: the rows on their own axis.
///
/// # Errors
///
/// [`Refusal::Empty`], as [`elementwise`]. No `Grid` refusal is possible:
/// neither extent is multiplied, and each already fits a `u32`.
pub fn elementwise_rows(width: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    let [w, r] = rectangle(width, rows)?;
    Ok([w, r, 1])
}

/// The extents both rules share, checked once.
fn rectangle(width: i32, rows: i32) -> Result<[u32; 2], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([width.unsigned_abs(), rows.unsigned_abs()])
}

/// The provenance of an argument, re-exported so a routine signature can say
/// `Env<I32s>` without naming the machinery crate.
pub use kernels::routine::Env;

/// The slot marks, re-exported for the same reason: a routine signature states
/// `OutSlot<0, BufMut>` or `Weight<2, F32s>` so the operand slot the arm binds
/// this pointer from is a fact of the type, not of the hand-written arm alone.
pub use kernels::keys;
pub use kernels::routine::{Ask, Block, Else, Held, InSlot, Nth, Null, OutSlot, Over, Param, ParamF32, ParamOr, ParamOrLit, Reckoned, Say, Times, Weight};

/// Re-exported for the same reason.
pub use kernels::routine::Provenance as Supplier;

#[cfg(test)]
mod tests {
    use super::*;
    use kernels::routine::Provenance;

    /// Every argument kind refuses a value of another kind, by position (the
    /// width check described on [`ArgValue`]).
    #[test]
    fn an_argument_refuses_a_value_of_another_kind() {
        assert_eq!(
            <i32 as Arg<Vulkan>>::unpack(&ArgValue::Usize(1), 3),
            Err(Refusal::Kind {
                at: 3,
                want: Ty::I32
            })
        );
        assert_eq!(
            <Usize as Arg<Vulkan>>::unpack(&ArgValue::I32(1), 0),
            Err(Refusal::Kind {
                at: 0,
                want: Ty::Usize
            })
        );
        assert_eq!(
            <Buf as Arg<Vulkan>>::unpack(&ArgValue::F32(1.0), 7),
            Err(Refusal::Kind {
                at: 7,
                want: Ty::Buf
            })
        );
        // And a buffer handle is accepted by every buffer kind, because the
        // handle carries no element type -- the TYPE does, which is what makes
        // a body handed the wrong one fail to compile rather than at runtime.
        assert_eq!(
            <Buf as Arg<Vulkan>>::unpack(
                &ArgValue::Buffer {
                    handle: 9,
                    writes: false
                },
                0
            ),
            Ok(Buf(9))
        );
        assert_eq!(
            <U32s as Arg<Vulkan>>::unpack(
                &ArgValue::Buffer {
                    handle: 9,
                    writes: false
                },
                0
            ),
            Ok(U32s(9))
        );
    }

    /// `InPacked` takes a `u32`'s value and is not a `u32`.
    ///
    /// It is the one kind whose difference is not width but PLACE: the value
    /// rides a field of a struct an earlier buffer binds. A signature that
    /// spelled it `u32` would be one the driver cannot tell apart from a push
    /// block scalar.
    #[test]
    fn a_packed_field_is_a_u32_that_is_not_the_u32_kind() {
        assert_eq!(
            <InPacked as Arg<Vulkan>>::unpack(&ArgValue::U32(5), 0),
            Ok(InPacked(5))
        );
        assert_ne!(
            <InPacked as Arg<Vulkan>>::TY,
            <u32 as Arg<Vulkan>>::TY,
            "the two are the same width and different arguments"
        );
    }

    /// `Env` marks the supplier and changes nothing else.
    #[test]
    fn the_environment_wrapper_carries_the_type_and_states_the_supplier() {
        assert_eq!(<Env<I32s> as Arg<Vulkan>>::TY, Ty::I32s);
        assert_eq!(<Env<I32s> as Arg<Vulkan>>::PROV, Provenance::Env);
        assert_eq!(<I32s as Arg<Vulkan>>::PROV, Provenance::Trace);
    }

    /// The declared spellings are the shader tree's, not invented.
    ///
    /// Each of these was read out of `kernels/` before it was written here,
    /// and the count beside it is how many declarations use it. The point of
    /// `SPELLING` is a future generated cross-check against the real module,
    /// which is worth nothing if the strings were guessed.
    #[test]
    fn the_spellings_are_the_ones_the_slang_tree_declares() {
        assert_eq!(<Buf as Arg<Vulkan>>::SPELLING, "StructuredBuffer<PIE_ACT>");
        assert_eq!(
            <BufMut as Arg<Vulkan>>::SPELLING,
            "RWStructuredBuffer<PIE_ACT>"
        );
        assert_eq!(<I32s as Arg<Vulkan>>::SPELLING, "StructuredBuffer<int>");
        assert_eq!(<U32s as Arg<Vulkan>>::SPELLING, "StructuredBuffer<uint>");
        assert_eq!(<U8s as Arg<Vulkan>>::SPELLING, "StructuredBuffer<uint8_t>");
        // Empty, and deliberately: nothing in the tree declares a 64-bit
        // shader integer, so there is no spelling to record. A guess here
        // would be worse than the gap.
        assert_eq!(<Usize as Arg<Vulkan>>::SPELLING, "");
    }
}
