//! The shader backends' operand vocabulary, written once.
//!
//! `.wiki/kernel-x/refactor-bigplan.md` §7 Stage 1: the vocabulary is shared
//! and **identical in all three** shader backends. Measured over the 300
//! table statements: `Buf` 139, `I32` 73, `BufMut` 57, `I32s` 22, `F32` 20,
//! `U32s` 15, `U8s` 12, `Usize` 10, `U32` 8, `InPacked` 1.
//!
//! # It is not closed, and the reason is worth keeping
//!
//! That census said TEN, and this module said "ten types, no eleventh" until
//! `ssm` crossed and needed `F32s` and `F32sMut`. The census was not wrong —
//! it was complete over the rows that STATE their operands, and 285 of the 481
//! entrypoints have rows that state none. A family whose rows are bare
//! contributes nothing to an operand census and everything to the vocabulary
//! the moment somebody writes its signatures down.
//!
//! So the count is [`COUNT`], asserted rather than narrated, and it may rise
//! as the dark families cross. What stays true is the shared part: a type here
//! is one all three backends can use, and none of them has one the others
//! lack.
//!
//! So they are declared here rather than three times. What is NOT shared is
//! the value a backend binds — metal's, vulkan's and wgpu's `ArgValue` are
//! their own — so the impls are generic over [`ShaderValue`], which is the
//! little a shared operand type needs to know about a backend's value: how to
//! read a buffer handle or a scalar out of one, and how to make one.
//!
//! # Why not share the value too
//!
//! Because it is the one thing that is genuinely per-backend, and forcing it
//! would put a variant in every backend that only one of them uses. A metal
//! `ArgValue` may one day carry a function-constant; a vulkan one a
//! specialisation constant. `ShaderValue` asks for the ten kinds all three
//! already have and nothing else.
//!
//! # CUDA does not use this
//!
//! `kernels-cuda-new`'s vocabulary is not these ten — it has pointer arrays,
//! by-value aggregates, `MaybeConst`, and a `void*` ABI that drops the operand
//! type. Its `Arg` impls stay its own. This module is named `shader` and not
//! `arg` for that reason.

use crate::Ty;
use crate::routine::{Arg, Backend, Refusal};

/// How many operand types this module declares.
///
/// Asserted by `the_vocabulary_is_the_size_this_module_says_it_is`, so the
/// sentence above cannot drift from the list below — which it did once, from
/// ten to twelve, when `ssm` crossed and brought the float arrays with it.
pub const COUNT: usize = 12;

/// The little a shared operand type needs to know about a backend's value.
///
/// Implemented by each shader backend for its own `ArgValue`. Every method is
/// total: a value that is not of the asked-for kind answers `None`, and the
/// operand type turns that into [`Refusal::Kind`] naming the position.
pub trait ShaderValue: Copy {
    /// The buffer handle this value names, if it names one.
    fn as_buffer(self) -> Option<u32>;
    /// The 32-bit signed scalar this value is, if it is one.
    fn as_i32(self) -> Option<i32>;
    /// The 32-bit unsigned scalar this value is, if it is one.
    fn as_u32(self) -> Option<u32>;
    /// The 32-bit float this value is, if it is one.
    fn as_f32(self) -> Option<f32>;
    /// The 64-bit extent this value is, if it is one.
    fn as_usize(self) -> Option<u64>;

    /// A value naming a buffer the launcher only reads.
    fn buffer(handle: u32) -> Self;
    /// A value naming a buffer the launcher may WRITE through.
    ///
    /// Defaults to [`Self::buffer`], because for two of the three backends
    /// the distinction is not carried in the value: wgpu binds every storage
    /// entry the same way and metal's plan reads writability elsewhere.
    ///
    /// `driver-vulkan` overrides it, and the reason is measured. It puts a
    /// barrier between two dispatches only when they touch the same bytes,
    /// and it decides that from which operands a launch may WRITE. A fire is
    /// a few hundred rectangles over one arena; barriering every neighbouring
    /// pair costs 8 microseconds each on an RTX 4090, which was 3.8 ms of a
    /// 7.2 ms decode. Under `kernel!` that fact came off the row's operand
    /// types. A routine states it in the argument TYPE -- `BufMut` against
    /// `Buf` -- and this is how the type reaches the driver, which sees only
    /// values.
    ///
    /// A backend that does not override it loses nothing: every buffer reads
    /// as written, which is the coarse and SAFE direction.
    #[must_use]
    fn buffer_mut(handle: u32) -> Self {
        Self::buffer(handle)
    }
    /// A value carrying a 32-bit signed scalar.
    fn i32(v: i32) -> Self;
    /// A value carrying a 32-bit unsigned scalar.
    fn u32(v: u32) -> Self;
    /// A value carrying a 32-bit float.
    fn f32(v: f32) -> Self;
    /// A value carrying a 64-bit extent.
    fn usize(v: u64) -> Self;
}

/// How one shader LANGUAGE spells the twelve operand types.
///
/// The vocabulary is shared because it is one closed set of twelve; the
/// SPELLINGS are not, because there are three languages. WGSL writes a
/// read-only storage binding `read` and a signed array `array<i32>`; Slang
/// writes `StructuredBuffer<PIE_ACT>` and `StructuredBuffer<int>`; MSL writes
/// a pointer. One shared constant would have to be one of the three and
/// therefore wrong in the other two, which is the defect
/// `.wiki/kernel-x/refactor-bigplan.md` §1 is about — a fact recorded away
/// from the thing it is a fact about.
///
/// Implemented on the backend MARKER (`Vulkan`, `Wgpu`, `Metal`), not on its
/// value: the value is a runtime binding and the spelling is a property of the
/// text the kernel is written in.
pub trait Lang: Backend {
    /// A read-only opaque buffer.
    const BUF: &'static str;
    /// A writable opaque buffer.
    const BUF_MUT: &'static str;
    /// A read-only array of `i32`.
    const I32S: &'static str;
    /// A read-only array of `u32`.
    const U32S: &'static str;
    /// A read-only array of `u8`.
    const U8S: &'static str;
    /// A read-only array of `f32`.
    const F32S: &'static str;
    /// An array of `f32` the launcher may WRITE through.
    ///
    /// Two spellings and not one because two of the three languages really do
    /// have two: Slang writes `RWStructuredBuffer<float>` against
    /// `StructuredBuffer<float>` and MSL writes `device float*` against `const
    /// device float*`. WGSL does NOT -- the element type is `array<f32>` for
    /// both and the access lives in the `var<storage, read_write>` that
    /// declares the binding, not in the type. So `Wgpu` spells these two the
    /// same, and that is the language rather than an omission.
    const F32S_MUT: &'static str;
    /// A 32-bit signed scalar.
    const I32: &'static str;
    /// A 32-bit unsigned scalar.
    const U32: &'static str;
    /// A 32-bit float.
    const F32: &'static str;
    /// A 64-bit extent, as the shader receives it.
    ///
    /// Empty for a language whose kernels here declare no 64-bit integer and
    /// receive the value already narrowed — an empty spelling says "this
    /// backend has not written one down", which is what [`Arg::SPELLING`]'s
    /// own doc reserves it for, and is honest where a guess would not be.
    const USIZE: &'static str;
    /// A `u32` that rides a field of a bound struct.
    const IN_PACKED: &'static str;
}

/// A typed argument, back as the value a dispatch carries.
///
/// The inverse of [`Arg::unpack`]. A trait rather than a method per type so
/// that a body passes its arguments straight through — `x.v()` reads the same
/// for a buffer and a scalar, which is what keeps an argument list from having
/// to remember which is which.
pub trait Bind<V: ShaderValue>: Copy {
    /// This argument as a bound value.
    fn v(self) -> V;
}

/// Declare one buffer-shaped operand type.
macro_rules! buffer_arg {
    ($(#[$m:meta])* $name:ident, $ty:expr, $spelling:ident, $bind:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name(
            /// The driver's handle.
            pub u32,
        );

        impl<B: Lang> Arg<B> for $name
        where
            B::Value: ShaderValue,
        {
            const TY: Ty = $ty;
            const SPELLING: &'static str = B::$spelling;

            fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
                value
                    .as_buffer()
                    .map(Self)
                    .ok_or(Refusal::Kind { at, want: $ty })
            }
        }

        impl<V: ShaderValue> Bind<V> for $name {
            fn v(self) -> V {
                V::$bind(self.0)
            }
        }
    };
}

buffer_arg!(
    /// An opaque device buffer the launcher may WRITE through.
    BufMut,
    Ty::BufMut,
    BUF_MUT,
    buffer_mut
);
buffer_arg!(
    /// An opaque device buffer the launcher only reads.
    Buf,
    Ty::Buf,
    BUF,
    buffer
);
buffer_arg!(
    /// A read-only device array of `i32` — positions, and the like.
    I32s,
    Ty::I32s,
    I32S,
    buffer
);
buffer_arg!(
    /// A read-only device array of `u32` — the CSR/indptr family.
    U32s,
    Ty::U32s,
    U32S,
    buffer
);
buffer_arg!(
    /// A read-only device array of `u8` — the per-row validity masks.
    ///
    /// Neither WGSL nor Slang has an 8-bit storage type, so the shader reads
    /// this as packed 32-bit words. The DECLARED width is what makes a
    /// mismatch an error here rather than a stride bug on the device.
    U8s,
    Ty::U8s,
    U8S,
    buffer
);

buffer_arg!(
    /// A read-only device array of `f32` — recurrent state, staged scratch.
    F32s,
    Ty::F32s,
    F32S,
    buffer
);
buffer_arg!(
    /// A device array of `f32` the launcher may WRITE through.
    ///
    /// Distinct from [`BufMut`] in the ELEMENT type and from [`F32s`] in the
    /// access: `gdn`'s recurrent state is read and written by the same
    /// dispatch, and it is the `Mut` half that tells the driver a barrier is
    /// owed before whoever reads it next.
    F32sMut,
    Ty::F32sMut,
    F32S_MUT,
    buffer_mut
);

/// Declare one scalar operand over an existing Rust type.
macro_rules! scalar_arg {
    ($rust:ty, $ty:expr, $read:ident, $make:ident, $spelling:ident) => {
        impl<B: Lang> Arg<B> for $rust
        where
            B::Value: ShaderValue,
        {
            const TY: Ty = $ty;
            const SPELLING: &'static str = B::$spelling;

            fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
                value.$read().ok_or(Refusal::Kind { at, want: $ty })
            }
        }

        impl<V: ShaderValue> Bind<V> for $rust {
            fn v(self) -> V {
                V::$make(self)
            }
        }
    };
}

scalar_arg!(i32, Ty::I32, as_i32, i32, I32);
scalar_arg!(u32, Ty::U32, as_u32, u32, U32);
scalar_arg!(f32, Ty::F32, as_f32, f32, F32);

/// A 64-bit stride or extent.
///
/// Its own type rather than `u64` so the width is stated where the argument
/// is. Neither WGSL nor Slang has a 64-bit integer in these kernels, so the
/// shader reads it as two 32-bit words, low first — and a value that arrived
/// as eight bytes and was bound as four is the failure this distinguishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usize(
    /// The value.
    pub u64,
);

impl<B: Lang> Arg<B> for Usize
where
    B::Value: ShaderValue,
{
    const TY: Ty = Ty::Usize;
    const SPELLING: &'static str = B::USIZE;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        value.as_usize().map(Self).ok_or(Refusal::Kind {
            at,
            want: Ty::Usize,
        })
    }
}

impl<V: ShaderValue> Bind<V> for Usize {
    fn v(self) -> V {
        V::usize(self.0)
    }
}

/// A `u32` that rides a FIELD of a struct some earlier buffer binds, rather
/// than the scalar block.
///
/// The width is a `u32`'s; the difference is WHERE the value goes, which the
/// driver's binding decides and a routine does not. One row uses it, in all
/// three backends — `refactor-bigplan.md` §10 leaves open whether it should
/// stay a type at all once the ports are done.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InPacked(
    /// The value.
    pub u32,
);

impl<B: Lang> Arg<B> for InPacked
where
    B::Value: ShaderValue,
{
    const TY: Ty = Ty::InPacked;
    const SPELLING: &'static str = B::IN_PACKED;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        value.as_u32().map(Self).ok_or(Refusal::Kind {
            at,
            want: Ty::InPacked,
        })
    }
}

impl<V: ShaderValue> Bind<V> for InPacked {
    fn v(self) -> V {
        V::u32(self.0)
    }
}

impl<V: ShaderValue, T: Bind<V>> Bind<V> for crate::routine::Env<T> {
    fn v(self) -> V {
        self.0.v()
    }
}

/// `LaunchRule::Elementwise`: one lane per element of the whole rectangle.
///
/// The lane arithmetic is shared and the bodies are not, and that is not a
/// contradiction. `refactor-bigplan.md` §2 keeps the bodies separate because
/// the three backends have a right to differ about tiles, workgroup sizes and
/// tiers; §1.2 measured that the GRID arithmetic already agrees at the source
/// level, character for character. This is that half, written once.
///
/// # Errors
///
/// [`Refusal::Empty`] when either extent is zero or negative, and
/// [`Refusal::Grid`] when the product does not fit a `u32`.
///
/// Both are refusals rather than clamps because both fail SILENTLY otherwise.
/// A dispatch of zero runs nothing and reports success over a buffer that
/// keeps whatever it held; a product that WRAPPED covers a fraction of the
/// rectangle and also reports success -- which is `driver-metal`'s
/// quarter-prefill defect in miniature, and it is why this takes the product
/// in `u64` and refuses rather than multiplying two `u32`s and hoping. An
/// extent of zero arrives here honestly -- a routed expert that won no tokens
/// has zero rows -- so it is a value the caller reads, not a panic.
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

/// The extents both rules share, checked once, and named separately.
///
/// Which extent was empty, rather than that their product was: a caller
/// reading `nothing to launch: width * rows is zero` has to work out which of
/// the two it was.
fn rectangle(width: i32, rows: i32) -> Result<[u32; 2], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([width.unsigned_abs(), rows.unsigned_abs()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{Env, Provenance};

    /// The vocabulary is the size this module says it is.
    ///
    /// The count is prose otherwise, and this module's prose said "ten types,
    /// no eleventh" for as long as it took another backend to cross a family
    /// and need two more. A census that reads the declarations cannot say that.
    #[test]
    fn the_vocabulary_is_the_size_this_module_says_it_is() {
        let src = include_str!("shader.rs");
        let declared = src.matches("\nbuffer_arg!(").count()
            + src.matches("\nscalar_arg!(").count()
            // The ones written out by hand, because their unpacking is not
            // the shape either macro stamps. Matched on the `Arg<B> for` and
            // not on the bound: the bound was `Backend` when this was written
            // and is `Lang` now, and a census that names a trait it does not
            // own counts zero of them without saying so.
            + src
                .lines()
                .filter(|l| l.starts_with("impl") && l.contains("Arg<B> for "))
                .count();
        assert_eq!(
            declared, COUNT,
            "this module declares {declared} operand types and `COUNT` says \
             {COUNT}. Update it, and the sentence in the module docs that \
             quotes it -- a vocabulary that grew without either moving is how \
             the last count went stale."
        );
    }

    /// A rectangle whose product does not fit a `u32` is refused, not wrapped.
    ///
    /// This is the check the shared version exists for. `kernels-wgpu`'s
    /// first copy of this arithmetic multiplied two `u32`s -- which panics in
    /// debug and WRAPS in release, and a wrapped product is a grid covering a
    /// fraction of the rectangle that dispatches cleanly and reports success.
    /// That is `driver-metal`'s quarter-prefill defect arrived at by a
    /// different road, and nothing downstream reports either.
    #[test]
    fn a_rectangle_too_large_for_a_u32_is_refused_rather_than_wrapped() {
        // 65536 x 65536 is exactly 2^32: the smallest product that does not
        // fit, and the one that wraps to ZERO -- a grid that runs nothing.
        assert_eq!(
            elementwise(65536, 65536),
            Err(Refusal::Grid {
                what: "width * rows",
                at: 1 << 32
            })
        );
        // And the axis form cannot reach it, because it multiplies nothing.
        assert_eq!(elementwise_rows(65536, 65536), Ok([65536, 65536, 1]));
    }

    /// An empty rectangle names WHICH extent was empty.
    ///
    /// A caller told only that `width * rows` is zero has to work out which of
    /// the two it was, and one of them is ordinary -- a routed expert that won
    /// no tokens has zero rows and a caller wants `Ok`-shaped silence for it.
    #[test]
    fn an_empty_rectangle_names_the_extent_that_was_empty() {
        assert_eq!(elementwise(0, 8), Err(Refusal::Empty { what: "width" }));
        assert_eq!(elementwise(8, 0), Err(Refusal::Empty { what: "rows" }));
        assert_eq!(
            elementwise_rows(-1, 8),
            Err(Refusal::Empty { what: "width" })
        );
        assert_eq!(elementwise(64, 7), Ok([448, 1, 1]));
        assert_eq!(elementwise_rows(64, 7), Ok([64, 7, 1]));
    }

    /// A backend value, as small as [`ShaderValue`] allows.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum V {
        Buffer(u32),
        I32(i32),
        U32(u32),
        F32(f32),
        Usize(u64),
    }

    impl ShaderValue for V {
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

    #[derive(Clone, Copy, Debug)]
    struct Shader;

    impl Backend for Shader {
        type Value = V;
        type Ctx<'a> = ();
    }

    /// Spellings that are recognisably no real language's, so a test that
    /// asserted one by accident reads as obviously wrong.
    impl Lang for Shader {
        const BUF: &'static str = "buf";
        const BUF_MUT: &'static str = "buf_mut";
        const I32S: &'static str = "i32s";
        const U32S: &'static str = "u32s";
        const U8S: &'static str = "u8s";
        const F32S: &'static str = "f32s";
        const F32S_MUT: &'static str = "f32s_mut";
        const I32: &'static str = "i32";
        const U32: &'static str = "u32";
        const F32: &'static str = "f32";
        const USIZE: &'static str = "usize";
        const IN_PACKED: &'static str = "in_packed";
    }

    /// Every operand refuses a value of another kind, by position.
    ///
    /// The check that matters is the WIDTH one, and it is why the scalar kinds
    /// are separate at all: a `Usize` is eight bytes and an `I32` is four, so
    /// a value that crossed between them either truncates or writes over its
    /// neighbour depending on where in the block it lands. Nothing downstream
    /// would report either.
    #[test]
    fn an_operand_refuses_a_value_of_another_kind() {
        assert_eq!(
            <i32 as Arg<Shader>>::unpack(&V::Usize(1), 3),
            Err(Refusal::Kind {
                at: 3,
                want: Ty::I32
            })
        );
        assert_eq!(
            <Usize as Arg<Shader>>::unpack(&V::I32(1), 0),
            Err(Refusal::Kind {
                at: 0,
                want: Ty::Usize
            })
        );
        assert_eq!(
            <Buf as Arg<Shader>>::unpack(&V::F32(1.0), 7),
            Err(Refusal::Kind {
                at: 7,
                want: Ty::Buf
            })
        );
    }

    /// A handle carries no element type; the TYPE does.
    ///
    /// Which is what makes a body handed the wrong buffer kind fail to
    /// compile rather than at runtime — the handle would have been accepted.
    #[test]
    fn a_buffer_handle_is_accepted_by_every_buffer_kind() {
        assert_eq!(<Buf as Arg<Shader>>::unpack(&V::Buffer(9), 0), Ok(Buf(9)));
        assert_eq!(<U32s as Arg<Shader>>::unpack(&V::Buffer(9), 0), Ok(U32s(9)));
        assert_eq!(<U8s as Arg<Shader>>::unpack(&V::Buffer(9), 0), Ok(U8s(9)));
    }

    /// Binding and unpacking are inverse, for every kind.
    #[test]
    fn what_a_body_binds_is_what_a_signature_recovers() {
        assert_eq!(Bind::<V>::v(Buf(4)), V::Buffer(4));
        assert_eq!(Bind::<V>::v(BufMut(4)), V::Buffer(4));
        assert_eq!(Bind::<V>::v(-3i32), V::I32(-3));
        assert_eq!(Bind::<V>::v(3u32), V::U32(3));
        assert_eq!(Bind::<V>::v(0.5f32), V::F32(0.5));
        assert_eq!(Bind::<V>::v(Usize(1 << 40)), V::Usize(1 << 40));
        // `InPacked` binds as a `u32`: the width is a `u32`'s and only the
        // PLACE differs, which the driver decides.
        assert_eq!(Bind::<V>::v(InPacked(7)), V::U32(7));
    }

    /// `InPacked` takes a `u32`'s value and is not the `u32` operand.
    #[test]
    fn a_packed_field_is_a_u32_that_is_not_the_u32_operand() {
        assert_eq!(
            <InPacked as Arg<Shader>>::unpack(&V::U32(5), 0),
            Ok(InPacked(5))
        );
        assert_ne!(
            <InPacked as Arg<Shader>>::TY,
            <u32 as Arg<Shader>>::TY,
            "the two are the same width and different operands"
        );
    }

    /// `Env` marks the supplier and changes nothing else.
    #[test]
    fn the_environment_wrapper_carries_the_type_and_states_the_supplier() {
        assert_eq!(<Env<I32s> as Arg<Shader>>::TY, Ty::I32s);
        assert_eq!(<Env<I32s> as Arg<Shader>>::PROV, Provenance::Env);
        assert_eq!(<I32s as Arg<Shader>>::PROV, Provenance::Trace);
        assert_eq!(Bind::<V>::v(Env(Buf(2))), V::Buffer(2));
    }
}
