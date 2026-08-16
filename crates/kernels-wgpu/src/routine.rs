//! This backend's instantiation of the `kernels` routine machinery.
//!
//! A routine is an ordinary `fn` that computes a launch and dispatches it.
//! Everything a row used to STATE — the launch rule, the grid parameter, the
//! head parameter, the operand list — is code in the body, and the table row
//! is derived from the signature by [`macro@crate::routine`].
//!
//! # What differs from CUDA, and why the machinery had to move
//!
//! `kernels-cuda-new`'s `Ctx` is a struct: it owns the NVRTC cache and the
//! cuBLAS handles, so the crate that declares the routines is also the crate
//! that launches them.
//!
//! This crate cannot be. It depends on `kernels` and nothing else — no `wgpu`,
//! no adapter, no device — and that is its stated reason for existing: the
//! table and the shaders build on any machine that can build Rust. So the
//! thing a body dispatches through is a TRAIT the driver implements
//! ([`Encode`]), and `Backend::Ctx` is `dyn Encode`.
//!
//! That is the one change the second backend forced on the shared machinery:
//! `Backend::Ctx` is `?Sized` now. It costs nothing — the machinery only ever
//! names the context behind a reference — and without it this crate would have
//! had to take a `wgpu` dependency to satisfy a `Sized` bound.
//!
//! # What a body decides
//!
//! The entrypoint STRING and the workgroup counts. Both were data: the string
//! was a row's `name` plus an `axes` suffix pasted by the `kernel!` macro, and
//! the counts were `LaunchRule` plus `grid_param`/`head_param`, read by
//! `driver-wgpu`'s geometry pass. A body states both directly, which is what
//! makes a kernel whose grid does not follow any rule expressible without a
//! new rule.

use kernels::routine::{Backend, Refusal};
use kernels::shader::ShaderValue;

/// This backend, as the machinery names it.
///
/// A marker: never constructed, carrying only the two concrete types the
/// `kernels` machinery is generic over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wgpu;

impl Backend for Wgpu {
    type Value = ArgValue;
    type Ctx<'a> = dyn Encode + 'a;
}

/// One value a caller supplies for one argument.
///
/// The scalar kinds are separate variants rather than one integer because the
/// widths differ and the check that matters is exactly the width one: a
/// [`kernels::Ty::Usize`] value handed to a [`kernels::Ty::I32`] argument is eight bytes going
/// into a four-byte slot, which either truncates or writes over its neighbour
/// depending on where in the block it lands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    /// A device allocation, named by whatever index the caller keys its own
    /// buffers on. Opaque on purpose: this crate cannot name a `wgpu::Buffer`
    /// and does not need to. The driver binds handles to its own buffers
    /// before it calls, and a routine only passes them through.
    Buffer(u32),
    /// A 32-bit signed scalar.
    I32(i32),
    /// A 32-bit unsigned scalar.
    U32(u32),
    /// A 32-bit float.
    F32(f32),
    /// A 64-bit stride or extent, which is what [`kernels::Ty::Usize`] means here.
    ///
    /// WGSL has no 64-bit integer of any kind, so the shader reads this as a
    /// `vec2<u32>` — two words, low first. The width is why it is its own
    /// variant.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fire<'a> {
    /// The shader file the entrypoint lives in, as [`mod@crate::source`] keys it —
    /// `"rope/neox.wgsl"`.
    pub module: &'a str,
    /// The entrypoint, whole. A body that varies over an instantiation axis
    /// picks the spelling itself; nothing pastes a suffix on afterwards.
    pub entrypoint: &'a str,
    /// LANES in each dimension — elements of work, not workgroups.
    ///
    /// The division into workgroups is the driver's, because the divisor is a
    /// property of the shader TEXT rather than of the launch: `@workgroup_size`
    /// is declared in the WGSL and recovered by reflection, so a body that
    /// divided by it would carry a second copy of a number it cannot see.
    /// `driver-wgpu` already does exactly one `div_ceil` per axis for every
    /// rule it serves; stating lanes keeps that the only place it happens.
    ///
    /// A body must not state a zero here. `dispatch_workgroups(0, 1, 1)` is
    /// legal WebGPU that runs nothing and reports success, which is the one
    /// failure this whole surface exists to make impossible — so a body with
    /// nothing to do returns [`Refusal::Empty`] instead.
    pub lanes: [u32; 3],
}

/// What a routine body dispatches through.
///
/// Implemented by `driver-wgpu`. The split is the crate boundary: a body knows
/// the entrypoint and the grid, and the driver knows what a buffer is, which
/// bind group a scalar rides, and how to submit.
pub trait Encode {
    /// Run one dispatch.
    ///
    /// `args` is the routine's own argument list, in signature order. The
    /// implementor separates buffers from scalars by variant — which is what
    /// makes the split derivable rather than stated twice.
    ///
    /// # Errors
    ///
    /// Whatever the device or the binding refused, as a [`Refusal`].
    fn dispatch(&self, fire: Fire<'_>, args: &[ArgValue]) -> Result<(), Refusal>;
}

/// This backend's value, as the shared operand types read it.
///
/// The ten operand types live in [`kernels::shader`] because the vocabulary is
/// closed and identical in metal, vulkan and wgpu -- see
/// `.wiki/kernel-x/refactor-bigplan.md` §7. What is NOT shared is the value,
/// which is this enum, and this impl is the whole of what a shared type needs
/// to know about it.
impl ShaderValue for ArgValue {
    fn as_buffer(self) -> Option<u32> {
        match self {
            Self::Buffer(h) => Some(h),
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

/// How WGSL spells the twelve.
///
/// A storage binding's access mode for the opaque buffers, the element type
/// for the arrays, and `vec2<u32>` for a 64-bit extent, which WGSL has no
/// scalar for and these kernels read as two words, low first.
impl kernels::shader::Lang for Wgpu {
    const BUF: &'static str = "read";
    const BUF_MUT: &'static str = "read_write";
    const I32S: &'static str = "array<i32>";
    const U32S: &'static str = "array<u32>";
    const U8S: &'static str = "array<u8>";
    const F32S: &'static str = "array<f32>";
    const F32S_MUT: &'static str = "array<f32>";
    const I32: &'static str = "i32";
    const U32: &'static str = "u32";
    const F32: &'static str = "f32";
    const USIZE: &'static str = "vec2<u32>";
    const IN_PACKED: &'static str = "u32";
}

/// The operand vocabulary, from the crate that holds it once.
///
/// Re-exported rather than named through `kernels::shader` at every use, so a
/// body's signature reads as this backend's own and a family file imports one
/// module.
pub use kernels::shader::{Bind, Buf, BufMut, F32s, F32sMut, I32s, InPacked, U8s, U32s, Usize};

/// What a routine body dispatches through, spelled as a body writes it.
///
/// `dyn Encode + 'a`, and the lifetime is the whole reason `Backend::Ctx` is a
/// generic associated type. A wgpu `Encode` BORROWS: the device, the pipeline
/// cache and the fire's buffers all live in the caller's frame, so an
/// implementor is never `'static` and a plain `dyn Encode` could not name it.
/// CUDA's context is a struct and never noticed the difference; this is the
/// generalisation `refactor-plan.md` §7 said to write on the second consumer,
/// and it is the one place the second consumer actually needed one.
pub type Ctx<'a> = dyn Encode + 'a;

/// One routine, in this backend's instantiation of the machinery.
pub type Routine = kernels::routine::Routine<Wgpu>;

/// The backend's wrapper over [`kernels::routine!`], with [`Wgpu`] filled in
/// so a declaration names only the `fn`:
///
/// ```ignore
/// pub static ROUTINES: &[Routine] = &[
///     routine!(rms_single_row),
///     routine!(rope_neox_decode, in_place = &[(0, 0)]),
/// ];
/// ```
#[macro_export]
macro_rules! routine {
    ($body:ident $(, $($fact:tt)*)?) => {
        ::kernels::routine!($crate::routine::Wgpu, $body $(, $($fact)*)?)
    };
}

/// The provenance of an argument, re-exported so a routine signature can say
/// `Env<I32s>` without naming the machinery crate.
pub use kernels::routine::Env;

/// Re-exported for the same reason.
pub use kernels::routine::Provenance as Supplier;
