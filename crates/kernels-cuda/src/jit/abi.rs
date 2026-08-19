//! §3.2 — [`Abi`]: the one place a crossing type's C++ spelling, marshalling
//! tag, and nullability are written, so the typecheck translation unit, the
//! argument buffer, and the device declaration cannot drift from each other.
//!
//! `bf16` and `f16` are both sixteen-bit `unsigned short` at the C ABI but
//! mean different numbers; keeping them as distinct Rust types (`*mut bf16`
//! vs `*mut f16`) makes confusing them a type error instead of a kernel that
//! runs and silently produces wrong numbers.
//!
//! [`Abi`] is a trait, not an enum: adding a crossing type is one impl next
//! to the kernel that needs it, not a new variant plus every `match`.
//!
//! A nullable pointer is spelled `Option<NonNull<T>>` so "does this accept
//! null" is checked by the type, not a runtime `bool` flag.
//!
//! [`Abi::arg`] takes `&self`, not `self`: a by-value aggregate answers with
//! a pointer into the receiver, so the receiver must be the caller's own
//! binding and not a value moved into (and dangling after) `arg`'s frame.
//!
//! A by-value aggregate's layout is checked from both sides:
//! [`by_value!`](crate::by_value) asserts the Rust mirror's
//! `size_of`/`align_of`/`offset_of` against NVRTC-measured numbers in
//! `const` context (a Rust compile error on drift), and [`typecheck_tu`]
//! emits the same numbers as C++ `static_assert`s over the header's own
//! declaration (a C++ compile error on drift, `tests/typecheck_tu.rs`). Not
//! yet checked: a `__global__`'s whole parameter list against the launching
//! routine's [`Abi::CPP`] list — nothing pairs a routine with the template
//! instantiation it launches.

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
    ///
    /// `&self` and not `self`: a by-value aggregate answers with a pointer
    /// INTO the receiver, so the receiver must be the caller's binding and
    /// not a moved copy in this frame. See the module header.
    fn arg(&self) -> crate::jit::ArgValue;

    /// The same crossing, the other way: recover this type from a value bound
    /// at position `at`.
    ///
    /// This is what `call()` goes through. It is on the same impl as
    /// [`Abi::arg`] so the two directions cannot disagree about which
    /// `ArgValue` variant this type is.
    ///
    /// # Errors
    ///
    /// [`kernels::Refusal::Kind`] if the value is of another kind.
    fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal>;
}

/// The refusal every [`Abi::unpack`] gives for a value of the wrong kind.
const fn wrong_kind(at: usize, want: Ty) -> kernels::Refusal {
    kernels::Refusal::Kind { at, want }
}

/// [`Abi::unpack`] for a by-value aggregate: the bytes, read back as `T`.
///
/// The length is compared against the mirror's own `size_of` rather than
/// trusted, because a short read here would be a launch with a struct half
/// filled from whatever follows it.
///
/// # Errors
///
/// [`kernels::Refusal::Kind`] if the value is not bytes, or is the wrong
/// number of them.
pub fn unpack_aggregate<T: Copy>(
    value: &crate::jit::ArgValue,
    at: usize,
    want: Ty,
) -> Result<T, kernels::Refusal> {
    match value {
        crate::jit::ArgValue::Bytes { ptr, len } if *len == core::mem::size_of::<T>() => {
            // SAFETY: `ArgValue::Bytes` states `len` initialised bytes at
            // `ptr` for this call's duration, and `len` is `T`'s own size.
            // Read unaligned because the caller's buffer need not be aligned
            // for `T`.
            Ok(unsafe { ptr.cast::<T>().read_unaligned() })
        }
        _ => Err(wrong_kind(at, want)),
    }
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

/// A device element type, as a routine spells it into a template-id.
///
/// A body generic over `T: Inst` states its element type once; `ROUTINES`
/// names the instantiation. Every `CPP` is fully qualified (e.g.
/// `::pie::bf16`) because NVRTC resolves a name expression in the
/// translation unit's own empty scope, where a relative spelling has
/// nothing to be relative to.
pub trait Inst {
    /// How the device text spells this type.
    const CPP: &'static str;
}

// RENAMED FROM `Elem`, AND THE RENAME IS THE POINT. This trait answers *"how
// does the device text spell this type"* -- a C++ INSTANTIATION -- and two of
// its implementors, `ssm_f32` and `quant_f32`, are zero-sized structs that
// exist only to pick a template argument. Nothing points at them.
//
// `kernels::Elem` is the other half: what a `*const`/`*mut` may ADDRESS, with
// the `Ty` each direction binds as. F3 spends a wrapper's type parameter on
// that one, so it needed the name, and one word over two mechanisms is the
// `Weight`/`Bank` collision this tree already files as a defect.

impl Inst for bf16 {
    const CPP: &'static str = "::pie::bf16";
}

impl Inst for f16 {
    const CPP: &'static str = "::pie::f16";
}

impl Inst for fp8_e4m3 {
    const CPP: &'static str = "::pie::fp8_e4m3";
}

/// `u16` — a pure copy's element, which promises no arithmetic.
///
/// `layout`'s gather and its PLE relay are instantiated here and not at
/// [`bf16`]: both are byte moves, neither ever converts to float, and a tag
/// type that promises arithmetic nobody performs is a tag type that invites
/// it.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct u16_(pub u16);

impl Inst for u16_ {
    const CPP: &'static str = "::pie::u16";
}

/// A `MaybeConst<T>` and a `MaybeConst<U>` are one Rust type; the two element
/// markers below are not one C++ type, and `ssm` is where the difference
/// shows.
///
/// `ssm`'s scans carry their recurrent state in fp32 or in bf16, and the
/// choice is the kernel's own template parameter rather than the element type
/// of any buffer it takes. Both spell out of `pie::ssm` and not `pie`,
/// which is why they cannot be [`bf16`] and a hypothetical `f32` marker.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ssm_f32;

impl Inst for ssm_f32 {
    const CPP: &'static str = "::pie::ssm::f32";
}

/// See [`ssm_f32`].
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ssm_state_bf16;

impl Inst for ssm_state_bf16 {
    const CPP: &'static str = "::pie::ssm::state_bf16";
}

/// `pie::quant::f32` — the widening casts' target, which is `quant`'s own
/// alias and not `device`'s.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct quant_f32;

impl Inst for quant_f32 {
    const CPP: &'static str = "::pie::quant::f32";
}

/// A `const T*` the launcher accepts a null for.
///
/// §3.2 names `Option<NonNull<T>>` for a nullable pointer, and that is the
/// spelling for a mutable one. A `const` parameter needs its own type
/// because the C++ spelling differs and the typecheck translation unit
/// compares whole parameter types — a dropped `const` is exactly the drift
/// it exists to catch.
#[derive(Debug)]
#[repr(transparent)]
pub struct MaybeConst<T>(pub Option<NonNull<T>>);

// By hand, not derived: `NonNull<T>` is `Copy` for EVERY `T`, but a derive
// would demand `T: Copy` — which `c_void` cannot answer, and a pointer's
// copyability was never a fact about its pointee.
impl<T> Clone for MaybeConst<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for MaybeConst<T> {}

// A NULLABLE POINTER IS AN ELEMENT WITH ONE CARRIER — ITSELF.
//
// `MaybeConst<T>` is a `const T*` that may be absent, and the absence is the
// whole of what distinguishes it: there is no writing form, because the two
// launchers that take one (`ssm`'s conv bias, `gemm`'s LoRA bias) only read
// through it. So `Read` and `Write` are both `Self`, exactly as a shader
// plane's handle answers, and the `Ty` still splits because the table must say
// which way the launch drives it even where the value cannot.
impl<T: 'static> kernels::Elem for MaybeConst<T> {
    type Read = Self;
    type Write = Self;

    // A NULLABLE POINTER DOES NOT WINDOW. Every caller that advances an
    // operand bounds `start` against a rectangle first, and an absent bias has
    // no rectangle to bound against — answering with the value unchanged is
    // the honest reading of "advance a thing that may not be there".
    unsafe fn advance_read(read: Self::Read, _elems: usize) -> Self::Read {
        read
    }

    unsafe fn advance_write(write: Self::Write, _elems: usize) -> Self::Write {
        write
    }

    const CPP_CONST: &'static str = "const void*";
    const CPP_MUT: &'static str = "void*";
    const TY_CONST: Ty = Ty::Buf;
    const TY_MUT: Ty = Ty::BufMut;
}

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
            fn arg(&self) -> crate::jit::ArgValue {
                crate::jit::ArgValue::$arg(*self)
            }
            fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
                match value {
                    crate::jit::ArgValue::$arg(v) => Ok(*v),
                    _ => Err(wrong_kind(at, Ty::$ty)),
                }
            }
        }
        $crate::arg_via_abi!($rust);
    };
}

scalar_abi!(i32, "int", I32, I32);
scalar_abi!(u32, "unsigned int", U32, U32);
scalar_abi!(f32, "float", F32, F32);
scalar_abi!(bool, "bool", Bool, Bool);
scalar_abi!(i64, "long long", I64, I64);
scalar_abi!(usize, "std::size_t", Usize, Usize);

/// `Const<usize>`'s bound value, on every plane: `ConstRun for usize` (in
/// `kernels::routine`) fixes `Held = u64` so a checkpoint `usize` crosses the
/// ABI at one width regardless of the host's pointer size. This is the CUDA
/// reading of that width — the same `Ty::Usize`/`std::size_t` pair `usize`
/// itself wears, with the cast at the boundary because `ArgValue::Usize`
/// carries the Rust-native `usize` and not a `u64`. Not `scalar_abi!`: that
/// macro assumes the Rust type IS the `ArgValue` payload, which is true of
/// `usize` and not of this conversion.
impl Abi for u64 {
    const CPP: &'static str = "std::size_t";
    const TY: Ty = Ty::Usize;
    fn arg(&self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::Usize(*self as usize)
    }
    fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
        match value {
            crate::jit::ArgValue::Usize(v) => Ok(*v as u64),
            _ => Err(wrong_kind(at, Ty::Usize)),
        }
    }
}
crate::arg_via_abi!(u64);

/// `__nv_fp8_interpretation_t`, as the kernels take it.
///
/// A newtype and not `u32`: `scalar_abi!` requires the Rust type to *be* the
/// `ArgValue` payload, and here the payload is the field, not the wrapper.
/// `u32` would also compile and bind (`Ty::U32`/`Ty::Fp8Kind` both marshal
/// four bytes) — only [`typecheck_tu`], comparing `unsigned int` against
/// `::__nv_fp8_interpretation_t`, would catch the swap.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct fp8_kind(pub u32);

impl Abi for fp8_kind {
    const CPP: &'static str = "::__nv_fp8_interpretation_t";
    const TY: Ty = Ty::Fp8Kind;
    fn arg(&self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::U32(self.0)
    }
    fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
        match value {
            crate::jit::ArgValue::U32(v) => Ok(Self(*v)),
            _ => Err(wrong_kind(at, Ty::Fp8Kind)),
        }
    }
}

crate::arg_via_abi!(fp8_kind);

// No scalar `u8` impl: `Ty` has no general byte tag and no fn-world kernel
// needs one yet; the open set adds it with that kernel, not speculatively.

/// One pointer impl: `*const T` and `*mut T` for one pointee.
///
/// `$cpp` spells the WHOLE type, `const` and star included, because the
/// typecheck TU compares whole parameter types.
///
/// `unpack` accepts a [`Region`] as well as a bare [`Ptr`]: the binder mints
/// a `Region` (address plus shape) for every resolved operand, and a bare
/// pointer type here takes just the address, dropping the shape.
///
/// [`Region`]: crate::jit::ArgValue::Region
/// [`Ptr`]: crate::jit::ArgValue::Ptr
macro_rules! ptr_abi {
    ($pointee:ty, $const_cpp:literal, $const_ty:ident, $mut_cpp:literal, $mut_ty:ident) => {
        impl Abi for *const $pointee {
            const CPP: &'static str = $const_cpp;
            const TY: Ty = Ty::$const_ty;
            fn arg(&self) -> crate::jit::ArgValue {
                crate::jit::ArgValue::Ptr(*self as *mut c_void)
            }
            fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
                match value {
                    crate::jit::ArgValue::Ptr(p)
                    | crate::jit::ArgValue::Region { ptr: p, .. } => {
                        Ok(p.cast::<$pointee>().cast_const())
                    }
                    _ => Err(wrong_kind(at, Ty::$const_ty)),
                }
            }
        }
        impl Abi for *mut $pointee {
            const CPP: &'static str = $mut_cpp;
            const TY: Ty = Ty::$mut_ty;
            fn arg(&self) -> crate::jit::ArgValue {
                crate::jit::ArgValue::Ptr(self.cast::<c_void>())
            }
            fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
                match value {
                    crate::jit::ArgValue::Ptr(p)
                    | crate::jit::ArgValue::Region { ptr: p, .. } => Ok(p.cast::<$pointee>()),
                    _ => Err(wrong_kind(at, Ty::$mut_ty)),
                }
            }
        }
        /// A pointer the launcher accepts a null for.
        impl Abi for Option<NonNull<$pointee>> {
            const CPP: &'static str = $mut_cpp;
            const TY: Ty = Ty::$mut_ty;
            const NULLABLE: bool = true;
            fn arg(&self) -> crate::jit::ArgValue {
                crate::jit::ArgValue::Ptr(
                    self.map_or(core::ptr::null_mut(), |p| p.as_ptr().cast::<c_void>()),
                )
            }
            fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
                match value {
                    crate::jit::ArgValue::Ptr(p)
                    | crate::jit::ArgValue::Region { ptr: p, .. } => {
                        Ok(NonNull::new(p.cast::<$pointee>()))
                    }
                    _ => Err(wrong_kind(at, Ty::$mut_ty)),
                }
            }
        }
        /// A `const` pointer the launcher accepts a null for.
        impl Abi for MaybeConst<$pointee> {
            const CPP: &'static str = $const_cpp;
            const TY: Ty = Ty::$const_ty;
            const NULLABLE: bool = true;
            fn arg(&self) -> crate::jit::ArgValue {
                crate::jit::ArgValue::Ptr(
                    self.0.map_or(core::ptr::null_mut(), |p| p.as_ptr().cast::<c_void>()),
                )
            }
            fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
                match value {
                    crate::jit::ArgValue::Ptr(p)
                    | crate::jit::ArgValue::Region { ptr: p, .. } => {
                        Ok(MaybeConst(NonNull::new(p.cast::<$pointee>())))
                    }
                    _ => Err(wrong_kind(at, Ty::$const_ty)),
                }
            }
        }
        $crate::arg_via_abi!(
            *const $pointee,
            *mut $pointee,
            Option<NonNull<$pointee>>,
            MaybeConst<$pointee>,
        );
    };
}

/// `<*const E as Abi>::TY` and `<E as kernels::Elem>::TY_CONST` are the same
/// number, asserted where both are visible.
///
/// # Why a second macro and not a line inside `ptr_abi!`
///
/// THE ORPHAN RULE SPLIT THE MAPPING AND THIS IS WHAT REJOINS IT.
/// `impl kernels::Elem for f32` cannot be written in this crate -- neither
/// the trait nor the type is local -- so the primitive pointees are
/// implemented in `kernels/src/routine.rs` and this crate holds only their
/// `Abi`. Two copies of one mapping is the defect `Inst` exists to remove, so
/// the copies are made to disagree LOUDLY: every invocation below fails to
/// compile the moment a `ptr_abi!` line and its `Inst` impl name different
/// `Ty`s.
///
/// Not every `ptr_abi!` pointee has an `Inst`, which is why this is opt-in.
/// `*const c_void` is a pointee whose own pointee is a pointer -- the
/// `void**` shape -- and F3 is about what a `*const`/`*mut` addresses, not
/// about arrays of addresses. Those lines get `Abi` and no `Inst`, and a
/// signature cannot name one as an element.
///
/// `as u8` because `Ty` is not `PartialEq` in a const context; the
/// discriminants are what the ABI actually binds.
macro_rules! elem_agrees {
    ($($pointee:ty),* $(,)?) => {
        $(const _: () = {
            assert!(<*const $pointee as Abi>::TY as u8 == <$pointee as kernels::Elem>::TY_CONST as u8);
            assert!(<*mut $pointee as Abi>::TY as u8 == <$pointee as kernels::Elem>::TY_MUT as u8);
        };)*
    };
}

/// A DEVICE ARRAY OF `E`, as CUDA binds one.
///
/// The same written form as [`kernels::shader::Tensor`] and different innards,
/// which is what [`kernels::Elem`] requires: a shader plane's tensor holds a
/// binding index and this one holds nothing at all, because on CUDA the
/// carrier IS the pointer the element already names.
///
/// It exists so that one word says *"a plane of `E`"* on every plane. Once
/// [`kernels::Const`] stopped implying buffer-ness — `Const<Tensor<bf16>>` is a
/// weight and `Const<i32>` a scalar the statement carries — the carrier had to
/// say which, and `Tensor<E>` is that saying.
#[derive(Debug)]
pub struct Tensor<E>(core::marker::PhantomData<E>);

impl<E> Clone for Tensor<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E> Copy for Tensor<E> {}

// A TENSOR OF `E` BINDS EXACTLY AS `E` DOES. `#[repr]` does not come into it:
// the type is uninhabited at run time and forwards both carriers, so
// `In<Tensor<bf16>>` hands a body the same `*const bf16` that `In<bf16>` did.
impl<E: kernels::Elem> kernels::Elem for Tensor<E> {
    type Read = E::Read;
    type Write = E::Write;

    unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
        // SAFETY: the trait's obligation, forwarded to the element.
        unsafe { E::advance_read(read, elems) }
    }

    unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
        // SAFETY: as above.
        unsafe { E::advance_write(write, elems) }
    }

    const CPP_CONST: &'static str = E::CPP_CONST;
    const CPP_MUT: &'static str = E::CPP_MUT;
    const TY_CONST: Ty = E::TY_CONST;
    const TY_MUT: Ty = E::TY_MUT;
}

// A TENSOR IS THE WEIGHT RUN'S CARRIER, on this plane as on the other three.
// `Const<Tensor<bf16>>` claims the next weight and inherits the named-bank
// chain `Weight` had; `Const<i32>` claims the next slot of the params run.
impl<E: kernels::Elem> kernels::ConstRun for Tensor<E> {
    const RUN: kernels::routine::Claim = kernels::routine::Claim::Weight;
    const TY: Ty = E::TY_CONST;
    type Held = E::Read;
}

// THE LOCAL POINTEES. These four are this crate's own types, so the orphan
// rule allows the impl here -- beside the C++ spellings they already carry.
impl kernels::Elem for bf16 {
    // A POINTEE'S CARRIERS ARE ITS TWO POINTERS, as `prim_elem!` states for
    // the primitives in `kernels/src/routine.rs`. This is the same pair, for
    // the pointee this crate owns.
    type Read = *const bf16;
    type Write = *mut bf16;

    unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
        // SAFETY: the trait's obligation, forwarded to the caller.
        unsafe { read.add(elems) }
    }

    unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
        // SAFETY: as above.
        unsafe { write.add(elems) }
    }

    const CPP_CONST: &'static str = "const ::pie::bf16*";
    const CPP_MUT: &'static str = "::pie::bf16*";
    const TY_CONST: Ty = Ty::Bf16s;
    const TY_MUT: Ty = Ty::Bf16sMut;
}

impl kernels::Elem for f16 {
    // A POINTEE'S CARRIERS ARE ITS TWO POINTERS, as `prim_elem!` states for
    // the primitives in `kernels/src/routine.rs`. This is the same pair, for
    // the pointee this crate owns.
    type Read = *const f16;
    type Write = *mut f16;

    unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
        // SAFETY: the trait's obligation, forwarded to the caller.
        unsafe { read.add(elems) }
    }

    unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
        // SAFETY: as above.
        unsafe { write.add(elems) }
    }

    const CPP_CONST: &'static str = "const ::pie::f16*";
    const CPP_MUT: &'static str = "::pie::f16*";
    const TY_CONST: Ty = Ty::F16s;
    const TY_MUT: Ty = Ty::F16sMut;
}

ptr_abi!(
    bf16,
    "const ::pie::bf16*",
    Bf16s,
    "::pie::bf16*",
    Bf16sMut
);
ptr_abi!(
    f16,
    "const ::pie::f16*",
    F16s,
    "::pie::f16*",
    F16sMut
);
ptr_abi!(
    fp8_e4m3,
    "const ::pie::fp8_e4m3*",
    Buf,
    "::pie::fp8_e4m3*",
    BufMut
);
ptr_abi!(i32, "const ::std::int32_t*", I32s, "::std::int32_t*", I32sMut);
// `moe::hash_route_lookup`'s `tid2eid` needs `const int64_t*`. No
// `Ty::I64sMut` exists, so the mut spelling reuses `BufMut` until
// `kernels::Ty` gains one.
ptr_abi!(i64, "const ::std::int64_t*", I64s, "::std::int64_t*", BufMut);
// `moe::build_moe_ptrs_aligned`'s six operands are `const T**`/`T**`: the
// pointee is itself a pointer, so `CPP` must spell `const bf16**` etc, not
// the untyped `const void**` a bare `*const c_void` impl would give.
ptr_abi!(
    *const bf16,
    "const ::pie::bf16* const*",
    BufArrayOut,
    "const ::pie::bf16**",
    BufArrayOut
);
ptr_abi!(
    *mut bf16,
    "::pie::bf16* const*",
    BufArrayOutMut,
    "::pie::bf16**",
    BufArrayOutMut
);
ptr_abi!(
    *const u8,
    "const ::std::uint8_t* const*",
    BufArrayOut,
    "const ::std::uint8_t**",
    BufArrayOut
);
ptr_abi!(
    *const i32,
    "const ::std::int32_t* const*",
    BufArrayOut,
    "const ::std::int32_t**",
    BufArrayOut
);
ptr_abi!(i8, "const ::std::int8_t*", I8s, "::std::int8_t*", I8sMut);
ptr_abi!(u32, "const ::std::uint32_t*", U32s, "::std::uint32_t*", U32sMut);
ptr_abi!(u8, "const ::std::uint8_t*", U8s, "::std::uint8_t*", U8sMut);
// `*const u16` means the device parameter is literally `uint16_t*` — a
// width, not a format — distinct from the `u16_` `Inst` marker above, which
// names a template argument rather than a pointee.
ptr_abi!(u16, "const ::std::uint16_t*", U16s, "::std::uint16_t*", U16sMut);
ptr_abi!(f32, "const float*", F32s, "float*", F32sMut);
// No `u64` pointer impl: `pie::sample::lm_head_gemv_argmax_int8`'s
// `u64* partial_pairs` has no `Ty` word for it, so it crosses today as
// `*mut c_void` under `Ty::BufMut` (opaque to the host; only the tag's
// width is wrong, not the crossing). Closing it needs a `Ty::U64sMut` in
// `crates/kernels`, shared by every backend.
ptr_abi!(c_void, "const void*", Buf, "void*", BufMut);
// Pointer-to-pointer array kinds: outer const/mut is whether the launcher
// may move the cursor, inner is whether it may write through it. `ptr_abi!`
// produces all four from one invocation so a second invocation can't drift
// from the first:
//
//   *const *const c_void -> "const void* const*"  BufArray
//   *mut   *const c_void -> "const void**"        BufArrayOut
//   *const *mut   c_void -> "void* const*"        BufArrayMut
//   *mut   *mut   c_void -> "void**"              BufArrayOutMut
ptr_abi!(*const c_void, "const void* const*", BufArray, "const void**", BufArrayOut);
ptr_abi!(*mut c_void, "void* const*", BufArrayMut, "void**", BufArrayOutMut);

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
    fn arg(&self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::Ptr(self.0)
    }
    fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
        match value {
            crate::jit::ArgValue::Ptr(p) | crate::jit::ArgValue::Region { ptr: p, .. } => Ok(Self(*p)),
            _ => Err(wrong_kind(at, Ty::Stream)),
        }
    }
}

crate::arg_via_abi!(Stream);

/// A device address held as an opaque word.
///
/// Every pointer field of a by-value aggregate is one of these. It is a
/// `u64` and not a `*mut T` for three reasons, and they are the same three
/// `crate::attn::fa2::params::DevicePtr` gives: the host may never dereference it,
/// a raw pointer field would make the mirror `!Send`, and the device's
/// pointer is 64-bit regardless of what the host's is.
///
/// The floor spells this itself rather than importing `fa2`'s: a family may
/// depend on the floor and the floor may not depend on a family. The two
/// definitions are the same word and neither can drift into the other,
/// because `by_value!`'s `size_of`/`offset_of` assertions would fail on any
/// width but eight.
pub type DevicePtr = u64;

/// One by-value aggregate's measured C++ layout.
///
/// This is data with two reading consumers, which is why it is data at all:
/// [`typecheck_tu`] turns it into C++ `static_assert`s, and a human reading
/// a drift report needs to know which probe produced the numbers.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    /// How C++ spells the aggregate — the same string as [`Abi::CPP`].
    pub cpp: &'static str,
    /// `sizeof`, measured.
    pub size: usize,
    /// `alignof`, measured.
    pub align: usize,
    /// Each field's name and `offsetof`, measured, in declaration order.
    pub fields: &'static [(&'static str, usize)],
    /// The script that measured it, relative to the session's probe dir.
    ///
    /// Not decoration. Layout numbers are the one kind of constant in this
    /// tree that cannot be re-derived by reading, so the reproduction has to
    /// be named where the numbers are.
    pub probe: &'static str,
}

/// The measured layout of a by-value aggregate that crosses as bytes.
///
/// Implemented only by [`by_value!`](crate::by_value), never by hand.
pub trait ByValue: Abi {
    /// The measured layout.
    const LAYOUT: Layout;
}

/// Declare a Rust mirror of a C++ aggregate that a `__global__` takes by
/// value, with its measured layout.
///
/// # What this generates
///
/// 1. An [`Abi`] impl whose [`arg`](Abi::arg) is
///    [`ArgValue::Bytes`](crate::jit::ArgValue::Bytes) over the receiver, so
///    a [`unit!`](crate::unit) declaration names the aggregate exactly like
///    it names `f32`; nothing in `unit!`, `contract!` or `bind!` treats it
///    specially.
/// 2. A [`ByValue`] impl carrying the measured [`Layout`].
/// 3. `const` assertions on `size_of`, `align_of` and every named
///    `offset_of`. **These are the point.** A field inserted, widened or
///    reordered in the Rust mirror is a compile error; a mirror that drifts
///    from the header is caught by the same numbers asserted in C++ by
///    [`typecheck_tu`].
///
/// # The tag
///
/// `tag = ` names a [`Ty`]; the macro asserts **[`Ty::needs_mirror`] must be
/// true** in `const` context, since a `Ty` meaning "some aggregate" would
/// tag every aggregate alike — the exact hazard `ArgValue::Bytes` exists to
/// avoid. [`Ty::cpp`] for the chosen tag is NOT this type's C++ spelling
/// (that is always [`Abi::CPP`], via [`typecheck_tu`]) — the tag identifies
/// only the crossing kind.
///
/// [`Ty::needs_mirror`]: kernels::Ty::needs_mirror
///
/// # Example
///
/// ```ignore
/// by_value! {
///     KvCacheList as "KVCacheList<true>",
///     tag = KvCacheLayerView,
///     probe = "nvrtc-probes/xqa_kvcachelist.py",
///     size = 40, align = 8,
///     {
///         k_cache           @ 0  as "kCacheVLLM",
///         max_pages_per_seq @ 32 as "maxNbPagesPerSeq",
///     }
/// }
/// ```
#[macro_export]
macro_rules! by_value {
    (
        $rust:ident as $cpp:literal,
        tag = $tag:ident,
        probe = $probe:literal,
        size = $size:literal, align = $align:literal,
        { $($field:ident @ $at:literal as $cname:literal),* $(,)? }
    ) => {
        impl $crate::jit::Abi for $rust {
            const CPP: &'static str = $cpp;
            const TY: ::kernels::Ty = ::kernels::Ty::$tag;
            fn arg(&self) -> $crate::jit::ArgValue {
                // The borrow is of the caller's binding, which outlives the
                // `fire` call; the launch copies out of it before it returns.
                // See `abi`'s header for why this is `&self`.
                $crate::jit::ArgValue::Bytes {
                    ptr: ::core::ptr::from_ref::<$rust>(self).cast::<u8>(),
                    len: ::core::mem::size_of::<$rust>(),
                }
            }
            fn unpack(
                value: &$crate::jit::ArgValue,
                at: usize,
            ) -> ::core::result::Result<Self, ::kernels::Refusal> {
                $crate::jit::abi::unpack_aggregate::<$rust>(value, at, ::kernels::Ty::$tag)
            }
        }
        $crate::arg_via_abi!($rust);

        impl $crate::jit::ByValue for $rust {
            const LAYOUT: $crate::jit::Layout = $crate::jit::Layout {
                cpp: $cpp,
                size: $size,
                align: $align,
                fields: &[$(($cname, $at)),*],
                probe: $probe,
            };
        }

        const _: () = assert!(
            ::kernels::Ty::$tag.needs_mirror(),
            concat!(
                stringify!($rust), ": tag = ", stringify!($tag),
                " is a scalar or pointer kind. A by-value aggregate's tag must be \
                 a Ty whose needs_mirror() is true.",
            ),
        );
// 64-bit only: these assert a layout measured on a CUDA (64-bit) host
// against structs holding raw pointers, which cannot match on a 32-bit
// target; ungated they broke the `wasm32-unknown-unknown` build
// `driver-wgpu` gates.
        #[cfg(target_pointer_width = "64")]
        const _: () = assert!(
            ::core::mem::size_of::<$rust>() == $size,
            concat!(stringify!($rust), ": sizeof disagrees with the measured ", $cpp),
        );
        #[cfg(target_pointer_width = "64")]
        const _: () = assert!(
            ::core::mem::align_of::<$rust>() == $align,
            concat!(stringify!($rust), ": alignof disagrees with the measured ", $cpp),
        );
        $(
            #[cfg(target_pointer_width = "64")]
            const _: () = assert!(
                ::core::mem::offset_of!($rust, $field) == $at,
                concat!(
                    stringify!($rust), ".", stringify!($field),
                    ": offset disagrees with the measured ", $cpp, "::", $cname,
                ),
            );
        )*
    };

    // The untagged arm: some aggregates have no `Ty` that means them, and
    // `Ty::needs_mirror()` is a closed set the tagged arm cannot open for
    // them. Two rejected fixes: reusing a neighbouring tag (an
    // approximately-right tag reads as an assertion and checks nothing),
    // and minting a `Ty` per aggregate (reopens the closed enum this
    // file's header avoids). Instead `TY` here is `Ty::MlaPlanCache`, an
    // inert stand-in outside both `is_pointer` and
    // `bind::device::scalar`'s lists, so a future reader who does consult
    // it gets a named `ArgError::Unsupported`, not a silent wrong read.
    // Nothing consults it today: `Args::bind` short-circuits on
    // `ArgValue::Bytes`, and `typecheck_tu` spells parameters through
    // `Abi::CPP`. Otherwise identical to the tagged arm: same `ArgValue`,
    // same `Layout`, same size/align/offset assertions.
    (
        $rust:ident as $cpp:literal,
        untagged,
        probe = $probe:literal,
        size = $size:literal, align = $align:literal,
        { $($field:ident @ $at:literal as $cname:literal),* $(,)? }
    ) => {
        impl $crate::jit::Abi for $rust {
            const CPP: &'static str = $cpp;
            const TY: ::kernels::Ty = ::kernels::Ty::MlaPlanCache;
            fn arg(&self) -> $crate::jit::ArgValue {
                $crate::jit::ArgValue::Bytes {
                    ptr: ::core::ptr::from_ref::<$rust>(self).cast::<u8>(),
                    len: ::core::mem::size_of::<$rust>(),
                }
            }
            fn unpack(
                value: &$crate::jit::ArgValue,
                at: usize,
            ) -> ::core::result::Result<Self, ::kernels::Refusal> {
                $crate::jit::abi::unpack_aggregate::<$rust>(
                    value,
                    at,
                    ::kernels::Ty::MlaPlanCache,
                )
            }
        }
        $crate::arg_via_abi!($rust);

        impl $crate::jit::ByValue for $rust {
            const LAYOUT: $crate::jit::Layout = $crate::jit::Layout {
                cpp: $cpp,
                size: $size,
                align: $align,
                fields: &[$(($cname, $at)),*],
                probe: $probe,
            };
        }

        #[cfg(target_pointer_width = "64")]
        const _: () = assert!(
            ::core::mem::size_of::<$rust>() == $size,
            concat!(stringify!($rust), ": sizeof disagrees with the measured ", $cpp),
        );
        #[cfg(target_pointer_width = "64")]
        const _: () = assert!(
            ::core::mem::align_of::<$rust>() == $align,
            concat!(stringify!($rust), ": alignof disagrees with the measured ", $cpp),
        );
        $(
            #[cfg(target_pointer_width = "64")]
            const _: () = assert!(
                ::core::mem::offset_of!($rust, $field) == $at,
                concat!(
                    stringify!($rust), ".", stringify!($field),
                    ": offset disagrees with the measured ", $cpp, "::", $cname,
                ),
            );
        )*
    };
}

/// A by-value aggregate over eight bytes, spelled at the call site.
///
/// The JIT argument path passes each value as one `u64` cell, which is every
/// scalar and every pointer and NOT the 200-byte plan structs some launchers
/// take whole. §3.2 asks for the byte-buffer variant; `runtime::ArgValue`
/// grew `Bytes` for it, and this is the erased, one-off way to produce one.
///
/// The bytes are BORROWED, not owned: `cuLaunchKernel` reads the argument
/// array during the call and the caller — a host `fn` with the aggregate on
/// its own stack — outlives it.
///
/// # Prefer [`by_value!`](crate::by_value)
///
/// Use this only for an aggregate assembled on the spot with no named Rust
/// type — a `#[repr(C)]` local, a byte blob from elsewhere. A named mirror
/// should be declared with `by_value!`, which gives it a real [`Abi`] impl,
/// measured `offset_of` assertions and a [`Layout`] the typecheck TU can
/// assert in C++. This type has none of those: `cpp` is a runtime field, so
/// nothing checks that the bytes match the spelling.
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
    /// An inherent method and not an [`Abi`] impl, because [`Abi::CPP`] is a
    /// `const` — one spelling per type — and an erased carrier's spelling is
    /// a FIELD, different at every call site. A type whose C++ name varies
    /// per value cannot answer a `const`.
    ///
    /// That argument is about THIS type and not about aggregates in general;
    /// a named mirror has exactly one C++ spelling and so can and does answer
    /// the `const`. `by_value!` is that path and is the one to reach for.
    ///
    /// `cpp` is read by the typecheck translation unit, which is where an
    /// aggregate's layout agreement is actually checked.
    #[must_use]
    pub fn arg(self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::Bytes { ptr: self.bytes.as_ptr(), len: self.bytes.len() }
    }
}

/// The `__global__` [`typecheck_tu`] defines, so a compile of the unit has
/// something to be asked for by name.
pub const TYPECHECK_ENTRY: &str = "::pie::typecheck::probe";

/// One `static_assert`, wrapped the way the rest of the TU wraps them.
fn push_static_assert(out: &mut String, cond: &str, message: &str) {
    out.push_str("static_assert(");
    out.push_str(cond);
    out.push_str(",\n    \"");
    out.push_str(message);
    out.push_str("\");\n");
}

/// `root` with every one of `layouts` asserted against the header's own
/// declaration, as a translation unit NVRTC can be handed.
///
/// The numbers in a [`Layout`] were measured once, out of PTX, by the script
/// its `probe` field names. `by_value!` asserts them against the RUST mirror
/// in `const` context; this asserts the same numbers against the C++ the
/// mirror is a mirror OF. Neither alone is enough — the mirror and the header
/// can drift apart only in a direction the other one is looking.
///
/// The only thing the appended text DEFINES is [`TYPECHECK_ENTRY`], an empty
/// `__global__`. A `static_assert` produces no code, so without it the TU
/// would have nothing to lower and no name to ask for — and every compile in
/// this crate is asked for by name. The answer wanted is whether the unit
/// compiles at all; the entry point exists to make that question askable.
///
/// `__INTADDR__` and not `offsetof`: NVRTC's EDG front end rejects
/// `offsetof`/`__builtin_offsetof` ("type name is not allowed") and
/// pointer-difference forms ("must have a constant value"); this is the one
/// spelling that compiles, and it needs no extra `#include` beyond the
/// root's own.
#[must_use]
pub fn typecheck_tu(root: &str, layouts: &[Layout]) -> String {
    let mut out = String::with_capacity(root.len() + layouts.len() * 512);
    out.push_str(root);
    out.push_str(
        "\n\n// ── the typecheck unit's entry point ──\n\
         namespace pie::typecheck {\n\
         __global__ void probe() {}\n\
         }\n\
         \n// ── the measured layouts, asserted ──\n",
    );
    for layout in layouts {
        let cpp = layout.cpp;
        out.push_str(&format!("\n// {cpp}\n// measured by {}\n", layout.probe));
        push_static_assert(
            &mut out,
            &format!("sizeof({cpp}) == {}", layout.size),
            &format!("sizeof disagrees with the measurement in {}", layout.probe),
        );
        push_static_assert(
            &mut out,
            &format!("alignof({cpp}) == {}", layout.align),
            &format!("alignof disagrees with the measurement in {}", layout.probe),
        );
        for (field, at) in layout.fields {
            push_static_assert(
                &mut out,
                &format!("__INTADDR__(&((({cpp}*)0)->{field})) == {at}"),
                &format!("{field}'s offset disagrees with the measurement in {}", layout.probe),
            );
        }
    }
    out
}

// THE TWO COPIES OF THE POINTEE MAPPING, ASSERTED EQUAL.
//
// Seven primitives implemented in `kernels` and two local types implemented
// above; each has an `Abi` pair here whose `Ty`s were written on the
// `ptr_abi!` line. This is the whole reason the split is tolerable: the
// orphan rule forced a second copy, and a second copy that cannot silently
// disagree is a cross-check rather than a duplication.
//
// `i64` is the one worth reading twice -- its mutable direction is `BufMut`
// and not an `I64sMut`, because nothing in the tree declares an `int64_t*`
// parameter. An asymmetry stated in two files and asserted equal is an
// asymmetry that stays deliberate.
elem_agrees!(bf16, f16, i32, i64, i8, u32, u8, u16, f32, c_void);

// THE SHARED CONVERSION, over everything this file gives an ABI.
//
// `Abi::arg` and the shader planes' `Bind::v` were the same job under two
// names, which is one of the five things that told a CUDA body apart from a
// metal one at a glance. `Bind` is `kernels::routine`'s now, and `arg_via_abi!`
// stamps it beside the `Arg` impl for every crossing type -- a blanket cannot
// carry it here for the orphan reason `jit/arg.rs`'s header states.

/// A pointee this backend can both instantiate and bind.
///
/// ONE BOUND WHERE THERE WERE THREE. A generic routine used to carry
/// `T: Inst + kernels::Elem, <T as Elem>::Read: Abi, <T as Elem>::Write: Abi`
/// -- three lines naming two CUDA-only traits, on a signature whose shader
/// twin carried none. The facts are all implied by "this is a pointee CUDA
/// knows", so they are said once, here, and a routine writes `T: Pointee`.
///
/// # The bounds are on the associated types, not in a `where`
///
/// `trait Pointee: Elem where Self::Read: Abi` compiles and does NOT reach a
/// caller: a trait definition's `where` clause is checked at the impl and not
/// elaborated to the use site, so `T: Pointee` left `T::Read: Abi` unproven
/// and one hundred and twenty-eight bodies failed on a bound their signature
/// had just stated. `Elem<Read: Abi, Write: Abi>` is a SUPERTRAIT bound and
/// is elaborated, which is the whole difference.
/// # `Bind` rides along for the same reason
///
/// A body writes `x.arg()`, which is [`kernels::routine::Bind`], and a generic
/// one has only `T: Pointee` to prove it from. `Abi` alone used to be enough
/// because a blanket `impl<T: Abi + Copy> Bind<ArgValue> for T` supplied the
/// rest -- an impl the orphan rule refuses, so the conversion is stamped per
/// type by `arg_via_abi!` now and no longer follows from `Abi`. Naming it here
/// is what keeps a generic routine's signature at one bound.
pub trait Pointee:
    Inst
    + kernels::routine::Elem<
        Read: Abi + kernels::routine::Bind<crate::jit::ArgValue>,
        Write: Abi
            + kernels::routine::Bind<crate::jit::ArgValue>
            + kernels::routine::BindMut<crate::jit::ArgValue>,
    >
{
}

impl<T> Pointee for T where
    T: Inst
        + kernels::routine::Elem<
            Read: Abi + kernels::routine::Bind<crate::jit::ArgValue>,
            Write: Abi
            + kernels::routine::Bind<crate::jit::ArgValue>
            + kernels::routine::BindMut<crate::jit::ArgValue>,
        >
{
}
