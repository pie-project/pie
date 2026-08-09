//! §3.2 — [`Abi`], the one place a crossing type is spelled.
//!
//! # The two-formats-one-width hazard
//!
//! `bf16` and `f16` are both sixteen bits, both `unsigned short` to any C
//! ABI, and mean entirely different numbers. Under the row world they were
//! two spellings of `Ty` (`Ty::Bf16s`, `Ty::F16s`) that a `void*` operand
//! (`Ty::BufMut`) erased, and the compiler could not tell them apart because
//! at the ABI they ARE the same. A row that named the wrong one produced a
//! kernel that ran, produced numbers, and produced wrong numbers.
//!
//! Here they are distinct Rust types. `*mut bf16` and `*mut f16` are not
//! interchangeable, so the hazard becomes the type system's DEFAULT rather
//! than a rule someone has to remember. That is the whole reason these two
//! unit structs exist.
//!
//! # An open set, not a closed enum
//!
//! `kernels::Ty` is an enum: adding a crossing type means adding a variant
//! and finding every `match`. [`Abi`] is a trait: adding a crossing type
//! means writing one impl, next to the kernel that needed it, and nothing
//! else in the tree changes. §3.2's word for this is "open set", and it is
//! why the typecheck emitter below can spell types the old
//! `emit_device_typecheck` had to refuse.
//!
//! # Nullable pointers
//!
//! A launcher that accepts a null and means something by it says so in its
//! Rust type: `Option<NonNull<T>>`, whose niche makes it the same word at
//! the ABI and whose absence is unmistakable at the call site. The row
//! world spelled this `Operand::nullable: bool` — a fact "not checkable by
//! the C++ compiler", as `kernels::Operand`'s own doc says, because every
//! pointer accepts null. It is checkable by this one.

use core::ffi::c_void;
use core::ptr::NonNull;

use kernels::Ty;

/// A type that crosses to the device, and its three spellings.
///
/// One impl per crossing type. The impl is the ONLY place the C++ spelling,
/// the marshalling tag and the nullability of that type are written, so
/// nothing can drift between the typecheck translation unit, the argument
/// buffer and the declaration.
pub trait Abi: Copy {
    /// How C++ spells this type, in the device text's own vocabulary.
    ///
    /// Fed verbatim into the typecheck translation unit §6.1 keeps: the
    /// generated `static_assert(std::is_same_v<...>)` over the real
    /// `__global__`'s type. This string is what makes the duplicated
    /// signature CHECKED rather than merely duplicated.
    const CPP: &'static str;

    /// The runtime's marshalling tag.
    ///
    /// `Args::bind` checks this per operand and picks the `ArgValue`
    /// variant from it. It survives until §5 step 9 retires the dynamic
    /// argument path; until then this is the second spelling, and it is on
    /// the same impl as the first so the two cannot disagree.
    const TY: Ty;

    /// The launcher accepts a null here and means something by it.
    const NULLABLE: bool = false;

    /// This value, as the runtime's dynamic argument.
    #[cfg(feature = "_cuda")]
    fn arg(self) -> crate::runtime::ArgValue;
}

/// `bf16` — brain float, the device text's own spelling.
///
/// A unit struct rather than an alias for `u16`: see this module's header
/// for why the distinction from [`f16`] is load-bearing. The `repr` is the
/// storage word so a `*mut bf16` is the same address a `__nv_bfloat16*` is.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct bf16(pub u16);

/// `f16` — IEEE half.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct f16(pub u16);

/// `__nv_fp8_e4m3` — the eight-bit float the quantized paths carry.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct fp8_e4m3(pub u8);

/// A `const T*` the launcher accepts a null for.
///
/// §3.2 names `Option<NonNull<T>>` for a nullable pointer, and that is the
/// spelling for a mutable one. A `const` parameter needs its own type
/// because the C++ spelling differs and the typecheck translation unit
/// compares whole parameter types — a dropped `const` is exactly the drift
/// it exists to catch.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaybeConst<T>(pub Option<NonNull<T>>);

impl<T> MaybeConst<T> {
    /// A possibly-null `const T*` from a raw pointer.
    #[must_use]
    pub fn new(p: *const T) -> Self {
        Self(NonNull::new(p.cast_mut()))
    }

    /// Absent.
    #[must_use]
    pub const fn none() -> Self {
        Self(None)
    }
}

macro_rules! scalar_abi {
    ($rust:ty, $cpp:literal, $ty:ident, $arg:ident) => {
        impl Abi for $rust {
            const CPP: &'static str = $cpp;
            const TY: Ty = Ty::$ty;
            #[cfg(feature = "_cuda")]
            fn arg(self) -> crate::runtime::ArgValue {
                crate::runtime::ArgValue::$arg(self)
            }
        }
    };
}

scalar_abi!(i32, "int", I32, I32);
scalar_abi!(u32, "unsigned int", U32, U32);
scalar_abi!(f32, "float", F32, F32);
scalar_abi!(bool, "bool", Bool, Bool);
scalar_abi!(i64, "long long", I64, I64);
scalar_abi!(u8, "unsigned char", U8, U8);
scalar_abi!(usize, "std::size_t", Usize, Usize);

/// One pointer impl: `*const T` and `*mut T` for one pointee.
///
/// `$cpp` is the WHOLE type including the `const` and the star, because the
/// typecheck TU compares whole parameter types and a `const` in the wrong
/// place is exactly the drift it is there to catch.
macro_rules! ptr_abi {
    ($pointee:ty, $const_cpp:literal, $const_ty:ident, $mut_cpp:literal, $mut_ty:ident) => {
        impl Abi for *const $pointee {
            const CPP: &'static str = $const_cpp;
            const TY: Ty = Ty::$const_ty;
            #[cfg(feature = "_cuda")]
            fn arg(self) -> crate::runtime::ArgValue {
                crate::runtime::ArgValue::Ptr(self as *mut c_void)
            }
        }
        impl Abi for *mut $pointee {
            const CPP: &'static str = $mut_cpp;
            const TY: Ty = Ty::$mut_ty;
            #[cfg(feature = "_cuda")]
            fn arg(self) -> crate::runtime::ArgValue {
                crate::runtime::ArgValue::Ptr(self.cast::<c_void>())
            }
        }
        /// A pointer the launcher accepts a null for.
        impl Abi for Option<NonNull<$pointee>> {
            const CPP: &'static str = $mut_cpp;
            const TY: Ty = Ty::$mut_ty;
            const NULLABLE: bool = true;
            #[cfg(feature = "_cuda")]
            fn arg(self) -> crate::runtime::ArgValue {
                crate::runtime::ArgValue::Ptr(
                    self.map_or(core::ptr::null_mut(), |p| p.as_ptr().cast::<c_void>()),
                )
            }
        }
        /// A `const` pointer the launcher accepts a null for.
        impl Abi for MaybeConst<$pointee> {
            const CPP: &'static str = $const_cpp;
            const TY: Ty = Ty::$const_ty;
            const NULLABLE: bool = true;
            #[cfg(feature = "_cuda")]
            fn arg(self) -> crate::runtime::ArgValue {
                crate::runtime::ArgValue::Ptr(
                    self.0.map_or(core::ptr::null_mut(), |p| p.as_ptr().cast::<c_void>()),
                )
            }
        }
    };
}

// The spellings are `kernels::Ty::cpp()`'s, so a row and a declaration
// describing the same parameter produce the same typecheck line.
ptr_abi!(
    bf16,
    "const ::pie_cuda_driver::kernels::device::bf16*",
    Bf16s,
    "::pie_cuda_driver::kernels::device::bf16*",
    BufMut
);
ptr_abi!(
    f16,
    "const ::pie_cuda_driver::kernels::device::f16*",
    F16s,
    "::pie_cuda_driver::kernels::device::f16*",
    BufMut
);
ptr_abi!(
    fp8_e4m3,
    "const ::pie_cuda_driver::kernels::device::fp8_e4m3*",
    Buf,
    "::pie_cuda_driver::kernels::device::fp8_e4m3*",
    BufMut
);
ptr_abi!(i32, "const ::std::int32_t*", I32s, "::std::int32_t*", I32sMut);
ptr_abi!(u32, "const ::std::uint32_t*", U32s, "::std::uint32_t*", U32sMut);
ptr_abi!(u8, "const ::std::uint8_t*", U8s, "::std::uint8_t*", U8sMut);
ptr_abi!(f32, "const float*", F32s, "float*", F32sMut);
ptr_abi!(c_void, "const void*", Buf, "void*", BufMut);

/// The stream, which is a parameter of every launcher and a value of none.
///
/// `cudaStream_t` is `void*` at the ABI, and the row world spelled it
/// `Ty::Stream` so that a generated binding could refuse to bind it from a
/// `Source`. In fn-world nothing binds it: the `fn` takes a stream and hands
/// it to the fire path, and it never appears in an argument list at all.
/// The impl exists so a declaration CAN name it where a launcher takes one
/// in the middle of its parameters.
#[derive(Clone, Copy, Debug)]
pub struct Stream(pub *mut c_void);

impl Abi for Stream {
    const CPP: &'static str = "cudaStream_t";
    const TY: Ty = Ty::Stream;
    #[cfg(feature = "_cuda")]
    fn arg(self) -> crate::runtime::ArgValue {
        crate::runtime::ArgValue::Ptr(self.0)
    }
}

/// A by-value aggregate over eight bytes.
///
/// The JIT argument path passes each value as one `u64` cell, which is every
/// scalar and every pointer and NOT the 200-byte plan structs some launchers
/// take whole. §3.2 asks for the byte-buffer variant; `runtime::ArgValue`
/// grew `Bytes` for it, and this is the declaration-side type that produces
/// one.
///
/// The bytes are BORROWED, not owned: `cuLaunchKernel` reads the argument
/// array during the call and the caller — a host `fn` with the aggregate on
/// its own stack — outlives it.
#[derive(Clone, Copy, Debug)]
pub struct Bytes<'a> {
    /// The aggregate's bytes, in the device's layout.
    pub bytes: &'a [u8],
    /// How C++ spells the aggregate.
    pub cpp: &'static str,
}

impl<'a> Bytes<'a> {
    /// This aggregate, as the runtime's dynamic argument.
    ///
    /// An inherent method and NOT an [`Abi`] impl, and the reason is the one
    /// fact about aggregates that matters: [`Abi::CPP`] is a `const`, one per
    /// type, and every by-value parameter block has a different spelling. A
    /// type whose C++ name is a FIELD cannot answer a `const`, which is the
    /// shape of the same argument [`crate::runtime::ArgValue::Bytes`] makes
    /// against a `Ty` variant for aggregates.
    ///
    /// `cpp` is read by the typecheck translation unit, which is where an
    /// aggregate's layout agreement is actually checked.
    #[cfg(feature = "_cuda")]
    #[must_use]
    pub fn arg(self) -> crate::runtime::ArgValue {
        crate::runtime::ArgValue::Bytes { ptr: self.bytes.as_ptr(), len: self.bytes.len() }
    }
}

/// The typecheck translation unit for one unit's rows.
///
/// # Why this exists next to `abi::emit_device_typecheck`
///
/// The old emitter spells a buffer operand with `device_cpp_ty(ty, storage)`,
/// where `storage` is the HEAD OF THE ROW'S `elem` — the template argument.
/// That works while `elem` is a type and fails when it is a value: it
/// returns `Err` for `device::i32(128)`, for `device::true_type::value`, for
/// a bare `128`, `true` or `false`. Seven of `rope`'s fourteen device rows
/// are value-instantiated, so seven of fourteen could not be typechecked at
/// all.
///
/// Here each parameter spells ITSELF, through [`Abi::CPP`], and the template
/// argument is only pasted between the brackets. A non-type template
/// argument stops being a problem because it was never a source of parameter
/// types in the first place — the old emitter's coupling was the bug.
///
/// The output is the same shape the old one produces and is compiled by the
/// same pass: one `static_assert` per row over a pointer-to-function
/// comparison, which is the strictest check C++ offers and fails on a
/// reordered pair of `int`s, a dropped `const`, or a `bf16` where the
/// `__global__` takes an `f16`.
///
/// `params` is `unit!`'s `PARAMS`, parallel to `unit.rows`.
///
/// # Panics
///
/// When `params` and `unit.rows` are different lengths, which is drift
/// between two things one macro writes and therefore cannot happen without
/// someone having edited generated output by hand.
#[must_use]
pub fn typecheck_tu(unit: &crate::unit::Unit, params: &[&[&str]], include: &str) -> String {
    assert_eq!(
        unit.rows.len(),
        params.len(),
        "{}: {} rows and {} parameter lists",
        unit.name,
        unit.rows.len(),
        params.len()
    );
    let mut out = String::new();
    out.push_str("// GENERATED by kernels-cuda-new::x::abi::typecheck_tu — do not edit.\n");
    out.push_str("//\n");
    out.push_str("// One assertion per declared instantiation. Each compares the type of\n");
    out.push_str("// the real `__global__` against a function pointer built from the\n");
    out.push_str("// declaration's parameter types, so a reordered pair, a dropped\n");
    out.push_str("// `const` or a bf16/f16 swap is a compile error naming the symbol.\n");
    out.push_str("#include <type_traits>\n#include \"");
    out.push_str(include);
    out.push_str("\"\n\nnamespace {\n");
    for (row, params) in unit.rows.iter().zip(params) {
        let tag = mangle(row.sig.symbol);
        out.push_str("\n// ");
        out.push_str(row.sig.symbol);
        out.push_str("\nusing fn_");
        out.push_str(&tag);
        out.push_str(" = void (*)(");
        for (i, p) in params.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(p);
        }
        out.push_str(");\nstatic_assert(::std::is_same_v<fn_");
        out.push_str(&tag);
        out.push_str(", decltype(&");
        out.push_str(&row.instantiation());
        out.push_str(")>,\n    \"");
        out.push_str(row.sig.symbol);
        out.push_str(": the declaration and the __global__ disagree\");\n");
    }
    out.push_str("\n}  // namespace\n");
    out
}

/// A symbol as a C++ identifier.
fn mangle(symbol: &str) -> String {
    symbol.replace([':', '<', '>', ',', ' ', '(', ')', '#'], "_")
}
