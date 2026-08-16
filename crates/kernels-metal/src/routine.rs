//! This backend's instantiation of the `kernels` routine machinery.
//!
//! A routine is an ordinary `fn` that computes a launch and dispatches it.
//! Everything a row used to STATE — the launch rule, the grid parameter, the
//! head parameter, the operand list — is code in the body, and the table row
//! is derived from the signature by [`routine!`].
//!
//! This file is a PORT of `kernels-vulkan/src/routine.rs`, which is itself a
//! port of `kernels-wgpu`'s, and that is deliberate:
//! `.wiki/kernel-x/refactor-bigplan.md` §0 measured the three shader tables as
//! **one table written down three times**, and three crates answering the same
//! constraint three ways is the drift this refactor exists to remove. Where
//! Metal has no reason to differ it does not, and the three places it does are
//! named below.
//!
//! # Why the context is a trait and not a struct
//!
//! `kernels-cuda-new`'s `Ctx` is a struct: it owns the NVRTC cache and the
//! cuBLAS handles, so the crate that declares the routines also launches them.
//!
//! This crate cannot be that. Its `[dependencies]` is `kernels` and nothing
//! else, while the `MTLDevice`, the command buffer being encoded, the argument
//! table and every compiled pipeline live in `driver-metal`. So the thing a
//! body dispatches through is a TRAIT the driver implements ([`Encode`]), and
//! `Backend::Ctx` is `dyn Encode`.
//!
//! # What is Metal's alone
//!
//! **A [`Fire`] names its FILE.** Metal has no ahead-of-time shader build:
//! `driver-metal` compiles from `(path, entry name)` through
//! `newLibraryWithSource` at run time, and the path is not derivable from the
//! entrypoint — `quant/qmm_t.metal` holds 54 of them. Vulkan needs no such
//! field because `build.rs` emits one `.spv` per entrypoint; wgpu carries a
//! `module` for the same reason this carries a file.
//!
//! **A [`Fire`] states its THREADGROUP.** MSL declares no workgroup size in
//! the shader text — there is no `[numthreads]` and no `@workgroup_size`, so
//! there is nothing for the driver to reflect and no divisor it could recover.
//! `dispatchThreads:threadsPerThreadgroup:` takes both numbers and the kernel's
//! own reductions depend on the second. This is the second of
//! `refactor-bigplan.md` §2's four legitimate divergences, and it is why
//! `kernels::routine`'s launch shape is *lanes plus an optional group* rather
//! than lanes alone.
//!
//! **There is no capability tier.** wgpu and vulkan pick `Subgroup`, `Fp16` or
//! `Baseline` per device and key their modules by it. Metal has no such axis:
//! one source compiles to one pipeline, and what varies is the entrypoint's
//! instantiation point, which a body spells itself.
//!
//! And one thing that is NOT Metal's, though it looks like it should be: a
//! buffer carries no writability. Vulkan's [`ArgValue`] does, because that
//! driver puts a barrier between two dispatches only when they touch the same
//! bytes. Metal's ordering comes from the encoder, so `ShaderValue::buffer_mut`
//! takes its default and `Buf` versus `BufMut` stays a property of the
//! signature's TYPE — which is where it belongs, because that is what makes a
//! body handed the wrong one fail to compile.

#[cfg(test)]
use kernels::Ty;
#[cfg(test)]
use kernels::routine::Arg;
use kernels::routine::{Backend, Refusal};
use kernels::shader::ShaderValue;

/// This backend, as the machinery names it.
///
/// A marker: never constructed, carrying only the two concrete types the
/// `kernels` machinery is generic over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Metal;

impl Backend for Metal {
    type Value = ArgValue;
    type Ctx<'a> = dyn Encode + 'a;
}

/// One value a caller supplies for one argument.
///
/// The scalar kinds are separate variants rather than one integer because the
/// widths differ and the check that matters is exactly the width one. Metal's
/// scalars ride a per-dispatch parameter run that `driver-metal` packs from
/// this list into `ParamSlot`s of a stated byte width; a [`Ty::Usize`] value
/// landing in a four-byte slot either truncates or writes over its neighbour,
/// and nothing downstream would report it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    /// A device allocation, named by whatever index the caller keys its own
    /// buffers on.
    ///
    /// The handle is opaque on purpose: this crate cannot name an `MTLBuffer`
    /// or a GPU address and does not need to. The driver resolves handles to
    /// its own `Slice`s before it calls, and a routine only passes them
    /// through.
    Buffer(u32),
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
            Self::Buffer(_) => "a buffer",
            Self::I32(_) => "an i32",
            Self::U32(_) => "a u32",
            Self::F32(_) => "an f32",
            Self::Usize(_) => "a usize",
        }
    }
}

/// One dispatch, as a routine body states it.
///
/// Not generic over a lifetime, unlike vulkan's, and that is the fourth thing
/// Metal does differently. Both of its strings are `&'static`: a Metal
/// `Dispatch` is **planned now and encoded later** — `fire::run` sizes the
/// argument table, batch-compiles pipelines and fingerprints the plan for
/// indirect-command-buffer replay between the two — so a borrowed entrypoint
/// would tie the plan's lifetime to the stack frame of the body that named it.
///
/// It costs nothing, because an entrypoint that is not a compile-time literal
/// is precisely the defect this plane exists to forbid. A body over 54
/// instantiations picks one of 54 spellings; it does not build a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fire {
    /// The entrypoint, whole, as the shader's `[[host_name(...)]]` spells it.
    ///
    /// A body that varies over an instantiation axis picks the spelling
    /// itself; nothing pastes a suffix on afterwards, which is what `kernel!`
    /// did with `axes`. That mattered: an entrypoint assembled by pasting an
    /// axis suffix onto a row name is how `neox_freqs_mb` came to name the
    /// DECODE symbol — a single-row kernel over a multi-row grid, which
    /// rotated row zero and left every row after it untouched.
    pub entrypoint: &'static str,
    /// The shader this entrypoint is compiled from, relative to the kernels
    /// directory — `"quant/qmm_t.metal"`.
    ///
    /// Metal's alone. See the module docs: there is no ahead-of-time build to
    /// key on, and the path is not derivable from the entrypoint.
    pub file: &'static str,
    /// Total THREADS in each dimension, not threadgroups.
    ///
    /// `MTLComputeCommandEncoder`'s `dispatchThreads:` takes a thread count
    /// and `cuLaunchKernel` takes a block count, so this is the number a Metal
    /// body computes and the number that must not be confused with the other
    /// one. Writing it the other way launches `n_heads` threads in total,
    /// which is not an error the hardware reports: the kernel's simd
    /// reductions just read lanes that were never dispatched.
    ///
    /// A body must not state a zero here. A dispatch of no threads runs
    /// nothing and reports success — the failure this whole surface exists to
    /// make impossible. A body with nothing to do returns [`Refusal::Empty`].
    pub lanes: [u32; 3],
    /// Threads per threadgroup.
    ///
    /// Stated, not reflected — see the module docs. The kernel's own
    /// reductions depend on it: `rms.metal` gives threadgroup `gid` the span
    /// `gid * axis_size`, and `qmm_t.metal`'s steel MMA is written for
    /// `WM * WN * SIMD_SIZE` threads exactly.
    pub group: [u32; 3],
}

/// What a routine body dispatches through.
///
/// Implemented by `driver-metal`. The split is the crate boundary: a body
/// knows the entrypoint, the file and the extent, and the driver knows what a
/// buffer is, which argument-table index a binding lands in, how the scalar
/// run is packed, and which pipelines a fire needs compiled.
///
/// # What the driver's implementation does, and why it is not an encode
///
/// `driver-metal` separates PLANNING from ENCODING — `Dispatch`'s own doc is
/// *"everything a command encoder needs, and nothing that needs a command
/// encoder to compute"* — and `fire::run` depends on the separation: it sizes
/// the argument table from the whole plan, compiles every pipeline the plan
/// names in one batch, and fingerprints the plan for indirect-command-buffer
/// replay, all before an encoder exists. So the driver's `dispatch` APPENDS a
/// `Dispatch` to the plan. A routine body replaces `plan_one`, not
/// `encode_one`, and the plan stays data.
pub trait Encode {
    /// Record one dispatch.
    ///
    /// `args` is the routine's own argument list, in signature order — which
    /// is the SHADER's binding order, not the trace's. The two differ for most
    /// of this tree's rectangles, and reconciling them is the caller's job,
    /// not this one's.
    ///
    /// The implementor separates buffers from scalars by variant, which is
    /// what makes the split derivable rather than stated twice.
    ///
    /// # Errors
    ///
    /// Whatever the binding refused, as a [`Refusal`].
    fn dispatch(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal>;
}

/// This backend's value, as the shared operand types read it.
///
/// The ten operand types live in [`kernels::shader`] because the vocabulary is
/// closed and identical in metal, vulkan and wgpu — measured over all 300
/// rows, `.wiki/kernel-x/refactor-bigplan.md` §7 Stage 1. What is NOT shared
/// is the value, which is [`ArgValue`], and this impl is the whole of what a
/// shared operand type needs to know about it.
impl ShaderValue for ArgValue {
    fn as_buffer(self) -> Option<u32> {
        match self {
            Self::Buffer(handle) => Some(handle),
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
        Self::Buffer(handle)
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

/// How MSL spells the twelve, as this tree's shaders declare them.
///
/// Every string was counted in `kernels/` before it was written here:
/// `const device T*` 248 declarations, `device T*` (without the `const`) 91,
/// `const device int*` 58, `const constant int&` 259, `const constant float&`
/// 48, `constant uint&` 2.
///
/// Two of the ten have TWO live spellings in the tree and the more common one
/// is recorded: `const device uint32_t*` (62) over `const device uint*` (44),
/// and `const device uint8_t*` (38) over `const device uchar*` (22). They are
/// the same MSL types, and a cross-check built on `SPELLING` will have to
/// accept both — which is a fact about the shader tree worth having written
/// down rather than smoothed over.
///
/// `USIZE` is empty on purpose, exactly as Slang's is. MSL has `ulong`, but
/// nothing in this tree declares one: every live use is an extent the driver
/// resolves on the host and narrows before it reaches a parameter slot. A
/// guess here would be worse than the gap.
impl kernels::shader::Lang for Metal {
    const BUF: &'static str = "const device T*";
    const BUF_MUT: &'static str = "device T*";
    const I32S: &'static str = "const device int*";
    const U32S: &'static str = "const device uint32_t*";
    const U8S: &'static str = "const device uint8_t*";
    const F32S: &'static str = "const device float*";
    const F32S_MUT: &'static str = "device float*";
    const I32: &'static str = "const constant int&";
    const U32: &'static str = "constant uint&";
    const F32: &'static str = "const constant float&";
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
/// associated type rather than a plain one. A Metal `Encode` BORROWS what it
/// appends to — the plan under construction, and the resolver the operands
/// were bound through — so an implementor is never `'static` and a bare
/// `dyn Encode` (which means `dyn Encode + 'static`) could not name one.
/// Bodies take `&Ctx<'_>`.
pub type Ctx<'a> = dyn Encode + 'a;

/// One routine, in this backend's instantiation of the machinery.
pub type Routine = kernels::routine::Routine<Metal>;

/// The backend's wrapper over [`kernels::routine!`], with [`Metal`] filled in
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
        ::kernels::routine!($crate::routine::Metal, $body $(, $($fact)*)?)
    };
}

/// `LaunchRule::Elementwise`: one lane per element of the whole rectangle, on
/// one axis.
///
/// The two launch rules this crate's elementwise kernels use are written here
/// once rather than in each family, because they are not per-family facts:
/// they are the two shapes `driver-metal`'s `geometry::lanes` has always
/// computed, and the point of a body stating its own grid is that the
/// statement is the same one, not a new one.
///
/// A lane count is in THREADS, which on Metal is also what the encoder takes
/// — `dispatchThreads:threadsPerThreadgroup:` — so unlike vulkan's there is no
/// `div_ceil` waiting in the driver, and unlike CUDA's there is no rounding
/// up to a whole group. A body that asks for `width * rows` threads gets
/// exactly `width * rows` threads, and that is why the elementwise bodies in
/// this crate carry no bounds guard: the grid IS the extent.
///
/// `width` and `rows` are `i32` because that is what the `Env` supplying them
/// carries and what vulkan's equivalent takes; the cross-backend gate compares
/// those types, so they match on purpose.
///
/// # Errors
///
/// [`Refusal::Empty`] when either extent is zero or negative, and
/// [`Refusal::Grid`] when the product does not fit a `u32`.
///
/// Both are refusals rather than clamps because both fail SILENTLY otherwise.
/// A dispatch of zero threads runs nothing and returns success over a buffer
/// that keeps whatever it held; a product that wrapped covers a fraction of
/// the rectangle and also returns success. An extent of zero arrives here
/// honestly -- a routed expert that won no tokens has zero rows -- so this is
/// a value the caller reads, not a panic.
pub fn elementwise(width: i32, rows: i32) -> Result<[u32; 3], kernels::routine::Refusal> {
    let [w, r] = rectangle(width, rows)?;
    let n = u64::from(w) * u64::from(r);
    let n = u32::try_from(n).map_err(|_| kernels::routine::Refusal::Grid {
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
pub fn elementwise_rows(width: i32, rows: i32) -> Result<[u32; 3], kernels::routine::Refusal> {
    let [w, r] = rectangle(width, rows)?;
    Ok([w, r, 1])
}

/// The extents both rules share, checked once.
fn rectangle(width: i32, rows: i32) -> Result<[u32; 2], kernels::routine::Refusal> {
    if width <= 0 {
        return Err(kernels::routine::Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(kernels::routine::Refusal::Empty { what: "rows" });
    }
    Ok([width.unsigned_abs(), rows.unsigned_abs()])
}

/// The provenance of an argument, re-exported so a routine signature can say
/// `Env<I32s>` without naming the machinery crate.
pub use kernels::routine::Env;

/// Re-exported for the same reason.
pub use kernels::routine::Provenance as Supplier;

#[cfg(test)]
mod tests {
    use super::*;
    use kernels::routine::Provenance;

    /// Every argument kind refuses a value of another kind, by position.
    ///
    /// The check that matters is the WIDTH one, and it is the reason the
    /// scalar kinds are separate variants at all: a `Usize` is eight bytes and
    /// an `I32` is four, so a value that crossed between them either truncates
    /// or writes over its neighbour depending on where in the parameter run it
    /// lands. `driver-metal` packs that run from this list — `ParamSlot`
    /// carries `at` and `bytes` — and would report neither.
    #[test]
    fn an_argument_refuses_a_value_of_another_kind() {
        assert_eq!(
            <i32 as Arg<Metal>>::unpack(&ArgValue::Usize(1), 3),
            Err(Refusal::Kind {
                at: 3,
                want: Ty::I32
            })
        );
        assert_eq!(
            <Usize as Arg<Metal>>::unpack(&ArgValue::I32(1), 0),
            Err(Refusal::Kind {
                at: 0,
                want: Ty::Usize
            })
        );
        assert_eq!(
            <Buf as Arg<Metal>>::unpack(&ArgValue::F32(1.0), 7),
            Err(Refusal::Kind {
                at: 7,
                want: Ty::Buf
            })
        );
        // And a buffer handle is accepted by every buffer kind, because the
        // handle carries no element type -- the TYPE does, which is what makes
        // a body handed the wrong one fail to compile rather than at runtime.
        assert_eq!(
            <Buf as Arg<Metal>>::unpack(&ArgValue::Buffer(9), 0),
            Ok(Buf(9))
        );
        assert_eq!(
            <U32s as Arg<Metal>>::unpack(&ArgValue::Buffer(9), 0),
            Ok(U32s(9))
        );
    }

    /// A writable buffer is the same VALUE as a read-only one.
    ///
    /// Vulkan's `ArgValue::Buffer` carries a `writes` flag because that driver
    /// decides barriers from it. Metal's ordering is the encoder's, so this
    /// backend takes `ShaderValue::buffer_mut`'s default and the distinction
    /// stays where it is checkable — in the signature's type.
    #[test]
    fn writability_is_a_type_and_not_a_value() {
        assert_eq!(
            <ArgValue as ShaderValue>::buffer_mut(4),
            <ArgValue as ShaderValue>::buffer(4)
        );
        assert_ne!(<Buf as Arg<Metal>>::TY, <BufMut as Arg<Metal>>::TY);
    }

    /// `InPacked` takes a `u32`'s value and is not a `u32`.
    ///
    /// It is the one kind whose difference is not width but PLACE: the value
    /// rides a field of a struct an earlier buffer binds.
    #[test]
    fn a_packed_field_is_a_u32_that_is_not_the_u32_kind() {
        assert_eq!(
            <InPacked as Arg<Metal>>::unpack(&ArgValue::U32(5), 0),
            Ok(InPacked(5))
        );
        assert_ne!(
            <InPacked as Arg<Metal>>::TY,
            <u32 as Arg<Metal>>::TY,
            "the two are the same width and different arguments"
        );
    }

    /// `Env` marks the supplier and changes nothing else.
    #[test]
    fn the_environment_wrapper_carries_the_type_and_states_the_supplier() {
        assert_eq!(<Env<I32s> as Arg<Metal>>::TY, Ty::I32s);
        assert_eq!(<Env<I32s> as Arg<Metal>>::PROV, Provenance::Env);
        assert_eq!(<I32s as Arg<Metal>>::PROV, Provenance::Trace);
    }

    /// The declared spellings are the shader tree's, not invented.
    ///
    /// Each of these was counted in `kernels/` before it was written here. The
    /// point of `SPELLING` is a future generated cross-check against the real
    /// shader, which is worth nothing if the strings were guessed.
    #[test]
    fn the_spellings_are_the_ones_the_msl_tree_declares() {
        assert_eq!(<Buf as Arg<Metal>>::SPELLING, "const device T*");
        assert_eq!(<BufMut as Arg<Metal>>::SPELLING, "device T*");
        assert_eq!(<I32s as Arg<Metal>>::SPELLING, "const device int*");
        assert_eq!(<U32s as Arg<Metal>>::SPELLING, "const device uint32_t*");
        assert_eq!(<U8s as Arg<Metal>>::SPELLING, "const device uint8_t*");
        // Empty, and deliberately: nothing in the tree declares a 64-bit
        // shader integer, so there is no spelling to record.
        assert_eq!(<Usize as Arg<Metal>>::SPELLING, "");
    }
}
