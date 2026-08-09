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
//!
//! # By-value aggregates, and why [`Abi::arg`] takes `&self`
//!
//! A `__global__` that takes a struct **by value** — XQA's
//! `KVCacheList<true>` at `xqa/mha.cuh:2804`, FA2's `__grid_constant__`
//! params — crosses as [`ArgValue::Bytes`](crate::runtime::ArgValue::Bytes),
//! which is a borrowed `(ptr, len)`. That is the reason this trait's one
//! method takes `&self` rather than `self`:
//!
//! > `fn arg(self) -> ArgValue` moves the value into `arg`'s own frame,
//! > takes its address, and returns. The address is dangling before the
//! > caller sees it. For a scalar that is invisible because the value is
//! > copied into the cell; for an aggregate it is a launch reading a dead
//! > stack frame, which is the exact failure `northstar.md` §5.1 predicted
//! > this variant's first caller would meet — *"a wrong bypass is a launch
//! > with a garbage struct, not a type error"*.
//!
//! With `&self`, the borrow is of the `raw::` stub's own parameter binding,
//! which lives across the `fire` call, and `Args::bind` copies out of it
//! before returning. `Abi: Copy`, so `&self` costs a scalar impl nothing.
//!
//! **What a by-value aggregate is checked by**, since no `Ty` can name it:
//!
//! 1. [`typecheck_tu`] compares the whole `__global__` parameter list
//!    against [`Abi::CPP`], so an aggregate declared where the kernel takes
//!    a pointer — or the wrong aggregate — is a C++ compile error.
//! 2. [`by_value!`](crate::by_value) asserts the Rust mirror's `size_of`,
//!    `align_of` and every `offset_of` against numbers **measured out of
//!    NVRTC's PTX**, in `const` context, so a drifted mirror is a Rust
//!    compile error.
//! 3. [`typecheck_tu`] emits the same numbers as C++ `static_assert`s, so a
//!    drifted *header* is a C++ compile error.
//!
//! (2) and (3) are the same measurement asserted from both sides. That is
//! the whole defence, and it is stronger than the tag it replaces: `Ty`
//! could never have said anything about a field offset.

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
    #[cfg(feature = "_cuda")]
    fn arg(&self) -> crate::runtime::ArgValue;
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
            fn arg(&self) -> crate::runtime::ArgValue {
                crate::runtime::ArgValue::$arg(*self)
            }
        }
    };
}

scalar_abi!(i32, "int", I32, I32);
scalar_abi!(u32, "unsigned int", U32, U32);
scalar_abi!(f32, "float", F32, F32);
scalar_abi!(bool, "bool", Bool, Bool);
scalar_abi!(i64, "long long", I64, I64);
scalar_abi!(usize, "std::size_t", Usize, Usize);

/// `__nv_fp8_interpretation_t`, as the kernels take it.
///
/// **A newtype rather than `u32`, and `scalar_abi!` cannot spell it.** That
/// macro writes `ArgValue::$arg(*self)`, which requires the Rust type to *be*
/// the payload; here the payload is the field. Two lines of the macro's body
/// would have to become a conversion, and the macro is right not to have one:
/// every other scalar in this list is its own payload, and a macro that
/// admitted a conversion would stop proving that.
///
/// # Why not `u32`
///
/// `u32` compiles, binds, and launches. `Args::bind` passes because
/// `Ty::U32` and `Ty::Fp8Kind` both marshal four bytes into eight, and the
/// only thing that would notice is [`typecheck_tu`], which compares
/// `unsigned int` against `::__nv_fp8_interpretation_t` and is the one reader
/// that can tell them apart. **That is §3.2's bypass exactly** — two formats
/// at one width, distinguished by nothing the binder checks — and the whole
/// argument for `Abi::CPP` existing beside `Abi::TY` is that the spelling is
/// the check.
///
/// `kernels::Ty::cpp` has spelled it `::__nv_fp8_interpretation_t` since it
/// was written, and `kernels-cuda-new::abi:446` records the measurement that
/// makes the four bytes safe: the enum is four bytes wide, asserted in the
/// generated typecheck rather than assumed.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct fp8_kind(pub u32);

impl Abi for fp8_kind {
    const CPP: &'static str = "::__nv_fp8_interpretation_t";
    const TY: Ty = Ty::Fp8Kind;
    #[cfg(feature = "_cuda")]
    fn arg(&self) -> crate::runtime::ArgValue {
        crate::runtime::ArgValue::U32(self.0)
    }
}

// No scalar `u8`, and not an oversight: `Ty` has no general byte tag — the
// row world only ever crossed a scalar byte as a semantic enum (`KvScheme`,
// `KvDType`, both checked against `ArgValue::U8`) — and no fn-world kernel
// takes one. An open set adds the impl with its first kernel, under
// whichever tag is honest for it, rather than minting a near-miss here.

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
            fn arg(&self) -> crate::runtime::ArgValue {
                crate::runtime::ArgValue::Ptr(*self as *mut c_void)
            }
        }
        impl Abi for *mut $pointee {
            const CPP: &'static str = $mut_cpp;
            const TY: Ty = Ty::$mut_ty;
            #[cfg(feature = "_cuda")]
            fn arg(&self) -> crate::runtime::ArgValue {
                crate::runtime::ArgValue::Ptr(self.cast::<c_void>())
            }
        }
        /// A pointer the launcher accepts a null for.
        impl Abi for Option<NonNull<$pointee>> {
            const CPP: &'static str = $mut_cpp;
            const TY: Ty = Ty::$mut_ty;
            const NULLABLE: bool = true;
            #[cfg(feature = "_cuda")]
            fn arg(&self) -> crate::runtime::ArgValue {
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
            fn arg(&self) -> crate::runtime::ArgValue {
                crate::runtime::ArgValue::Ptr(
                    self.0.map_or(core::ptr::null_mut(), |p| p.as_ptr().cast::<c_void>()),
                )
            }
        }
    };
}

// The spellings are `kernels::Ty::cpp()`'s, so a row and a declaration
// describing the same parameter produce the same typecheck line.
//
// THAT SENTENCE BECAME TRUE OF THE TWO SIXTEEN-BIT FORMATS HERE, and it had
// been false of them since this file was written. `CPP` for `*mut bf16` has
// always been `::pie_cuda_driver::kernels::device::bf16*`, and the tag beside
// it said `Ty::BufMut`, whose `cpp()` is `void*` -- so a row and a declaration
// describing one destination produced two different lines, and
// `abi::self_describing` declined the `void*` one outright because every
// object pointer converts to it and an assertion against it holds for every
// possible kernel.
//
// Measured over `unit!`'s declarations at `d737aad29`, with each row's own
// `where [T = …]` substituted first: 266 fn-world rows at 2207 operand
// positions, of which **172 rows carry a written sixteen-bit destination at
// 269 positions** -- 252 bf16 (`*mut bf16` 245, `Option<NonNull<bf16>>` 7) and
// 17 f16 (`*mut f16` 11, `Option<NonNull<f16>>` 6). All 269 were `void*` and
// none was asserted. `tests/device_typecheck_types.rs`'s
// `the_written_sixteen_bit_positions_are_two_hundred_and_sixty_nine`
// re-derives that number from `unit::rows()` at run time rather than trusting
// this comment.
ptr_abi!(
    bf16,
    "const ::pie_cuda_driver::kernels::device::bf16*",
    Bf16s,
    "::pie_cuda_driver::kernels::device::bf16*",
    Bf16sMut
);
ptr_abi!(
    f16,
    "const ::pie_cuda_driver::kernels::device::f16*",
    F16s,
    "::pie_cuda_driver::kernels::device::f16*",
    F16sMut
);
ptr_abi!(
    fp8_e4m3,
    "const ::pie_cuda_driver::kernels::device::fp8_e4m3*",
    Buf,
    "::pie_cuda_driver::kernels::device::fp8_e4m3*",
    BufMut
);
ptr_abi!(i32, "const ::std::int32_t*", I32s, "::std::int32_t*", I32sMut);
// `moe::hash_route_lookup`'s `tid2eid`, a `[vocab, K]` `const int64_t*` table.
//
// `kernels::Ty::I64s` exists and `I64sMut` does not, so the mut spelling
// reuses `BufMut`. This used to read "exactly as `bf16` and `f16` do above",
// and that clause went with the tags: **`i64` is now the only pointee in this
// file whose READ half has a kind of its own and whose WRITTEN half does
// not.** That is a statement about `kernels::Ty`, not about this file --
// closing it is one variant plus its `cpp()`/`rust()`/`is_buffer` arms in the
// PORTABLE crate, which Metal, Vulkan, WGPU and CPU all read, so it is not an
// edit this file can make.
//
// `fp8_e4m3` above is a DIFFERENT shape and not a fourth case of this one:
// neither of its halves is named (`Buf`/`BufMut`), so it is a format with no
// vocabulary rather than a half missing from one.
ptr_abi!(i64, "const ::std::int64_t*", I64s, "::std::int64_t*", BufMut);
// `moe::build_moe_ptrs_aligned`'s six operands, `const T**` and `T**` at
// `moe_dispatch.cuh:1046-1051`. **The pointee is the POINTER**, which is why
// these are `ptr_abi!(*const bf16, …)` rather than an impl on
// `*mut *const c_void`: `CPP` is the DEVICE parameter's spelling, and the
// device parameter is `const bf16**`, where `Ty::BufArrayOut::cpp()` is
// `const void**` because that was the deleted C launcher's. `ptr_abi!(bf16, …)`
// above already carries exactly that split, so this is the established shape.
//
// Spelling any of the six `*const c_void` instead would compile, put `Ty::Buf`
// where `Args::bind` checks, and put `const void*` in the typecheck
// translation unit where the kernel says `const bf16*` — the bypass that
// reproduces the deleted rows' `Ty`s and loses the only thing the port added.
ptr_abi!(
    *const bf16,
    "const ::pie_cuda_driver::kernels::device::bf16* const*",
    BufArrayOut,
    "const ::pie_cuda_driver::kernels::device::bf16**",
    BufArrayOut
);
ptr_abi!(
    *mut bf16,
    "::pie_cuda_driver::kernels::device::bf16* const*",
    BufArrayOutMut,
    "::pie_cuda_driver::kernels::device::bf16**",
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
// The impl arrives with its first kernel, which is what an open set means
// (see the note above `scalar_abi!`). `i8`'s is
// `sample::lm_head_gemv_argmax_int8_bf16`, whose `const int8_t* __restrict__
// lm_head_weight` its row states as `I8s`. Spelling it `*const c_void`
// instead would compile, put `Ty::Buf` where `Args::bind` checks `I8s`, and
// put `const void*` in the typecheck TU where the kernel says `const
// int8_t*` — a bypass with no type error anywhere, which is the failure mode
// `x/xqa.rs`'s header names.
ptr_abi!(i8, "const ::std::int8_t*", I8s, "::std::int8_t*", I8sMut);
ptr_abi!(u32, "const ::std::uint32_t*", U32s, "::std::uint32_t*", U32sMut);
ptr_abi!(u8, "const ::std::uint8_t*", U8s, "::std::uint8_t*", U8sMut);
// `u16` is the WIDTH and nothing about the format, and that is the whole
// care this impl needs.
//
// `Ty::U16s`'s own doc: *"it says the WIDTH and nothing about the format —
// so it cannot stand in for either"* `Bf16s` or `F16s`. In fn-world that
// sentence becomes a rule with teeth: **`*const u16` here means the
// `__global__` literally takes `uint16_t*`**, not "some sixteen-bit thing".
// A `template <class T>` instantiated at `device::bf16` is spelled `*const
// bf16` — that is what §3.2's unit structs are for, and reaching for `u16`
// to avoid a generic parameter would re-open the two-formats-one-width
// hazard the whole of §3.2 exists to close.
//
// The honest case, and the impl's first caller: `layout/gather_rows.cuh`'s
// two kernels are instantiated at `device::u16` and NOT at `device::bf16`,
// and `families/layout.rs` says why — *"both are pure copies: neither ever
// converts to float, and the ahead-of-time launchers take `u16*` for exactly
// that reason. A tag type that promises arithmetic nobody performs is a tag
// type that invites it."* The device text says `uint16_t`, so the
// declaration says `u16`. That is the test to apply: read the
// instantiation, not the operand's name.
ptr_abi!(u16, "const ::std::uint16_t*", U16s, "::std::uint16_t*", U16sMut);
ptr_abi!(f32, "const float*", F32s, "float*", F32sMut);
// No `u64` pointer impl, and this one is a REFUSAL rather than an absence.
//
// `sample::device::lm_head_gemv_argmax_int8` takes `u64* partial_pairs` —
// one packed `(value, token)` per tile per row. `families/sample.rs` found
// that `kernels::Ty` has no word for it and refused to mint one on §10.5
// grounds; the port carried the compromise across unchanged, as `*mut
// c_void` under `Ty::BufMut`, which is what the row already said.
//
// `Abi` could add the impl on its own — `CPP` is `"::std::uint64_t*"` — but
// `TY` could not, and an impl whose `TY` lied would be exactly the bypass
// the `i8` note above exists to prevent. Closing it is `Ty::U64sMut` plus
// its `cpp()`/`rust()`/`ArgValue` arms in `crates/kernels`, a crate the
// row world still marshals every operand through. That is a step-9 change,
// when `Ty` retires or becomes fn-world's alone; doing it mid-sweep would
// add a variant to the dynamic path for one operand while thirteen families
// still depend on that path's stability. The buffer crosses correctly today
// — it is opaque to the host and only its width is wrong in the tag.
ptr_abi!(c_void, "const void*", Buf, "void*", BufMut);
// THE ARRAY KINDS, and the pleasing part is that two lines land all four.
//
// `ptr_abi!`'s two halves are `*const $pointee` and `*mut $pointee`, and
// with a pointer as the pointee that is exactly `Ty`'s own four-way split:
// the OUTER const/mut is whether the launcher may move the cursor, the
// INNER is whether it may write through it. `Ty::BufArray`'s doc says
// naming reads outside-in; these read outside-in too, and the C++ spellings
// are `kernels/src/lib.rs:1021-1024` verbatim.
//
//   *const *const c_void -> "const void* const*"  BufArray
//   *mut   *const c_void -> "const void**"        BufArrayOut
//   *const *mut   c_void -> "void* const*"        BufArrayMut
//   *mut   *mut   c_void -> "void**"              BufArrayOutMut
//
// `ssm` asked for the two `*const`-outer forms and gets four, because a
// macro that produced only the halves asked for would have to be invoked
// again with the same pointee to produce the others — and that is the shape
// where two spellings of one type drift apart.
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
    #[cfg(feature = "_cuda")]
    fn arg(&self) -> crate::runtime::ArgValue {
        crate::runtime::ArgValue::Ptr(self.0)
    }
}

/// A device address held as an opaque word.
///
/// Every pointer field of a by-value aggregate is one of these. It is a
/// `u64` and not a `*mut T` for three reasons, and they are the same three
/// `crate::fa2::params::DevicePtr` gives: the host may never dereference it,
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
///    [`ArgValue::Bytes`](crate::runtime::ArgValue::Bytes) over the
///    receiver. That makes the aggregate an ORDINARY `Abi` type: a
///    [`unit!`](crate::unit) declaration names it exactly the way it names
///    `f32`, and nothing in `unit!`, `contract!` or `bind!` knows the
///    difference. §3.2's "open set of impls, not a closed enum" is what buys
///    that; a `Ty` variant per aggregate would have been the forty-variant
///    `LaunchRule` mistake one level down.
/// 2. A [`ByValue`] impl carrying the measured [`Layout`].
/// 3. `const` assertions on `size_of`, `align_of` and every named
///    `offset_of`. **These are the point.** A field inserted, widened or
///    reordered in the Rust mirror is a compile error; a mirror that drifts
///    from the header is caught by the same numbers asserted in C++ by
///    [`typecheck_tu`].
///
/// # The tag
///
/// `tag = ` names a [`Ty`] and every choice is a near-miss, so the macro
/// makes the caller write one — and then checks it. `Ty` is a closed enum of
/// things the row world could bind from a `Source`; no variant of it can name
/// an arbitrary struct, and adding one that means "some aggregate" would tag
/// every aggregate alike — the exact hazard `ArgValue::Bytes`'s own comment
/// refuses, and per-struct variants would be the forty-variant `LaunchRule`
/// mistake in a second place.
///
/// The rule is **[`Ty::needs_mirror`] must be true**, and the macro asserts
/// it in `const` context. That predicate is `Ty`'s own answer to "does this
/// kind cross as a `#[repr(C)]` struct rather than as a primitive or a
/// pointer", which is precisely the question a by-value aggregate is asking.
/// It also happens to be the set `Args::bind` refuses — it accepts pointer
/// kinds and the ten scalar kinds and nothing else — so if some future walker
/// did consult the tag, the answer is a named `ArgError::Unsupported` and not
/// a silent accept of eight bytes where forty were meant. `bind` never gets
/// that far today: it short-circuits on `ArgValue::Bytes` before the tag
/// match, and every fn-world operand is `Source::Unbound` besides.
///
/// [`Ty::cpp`] for the chosen variant will not be this type's C++ spelling.
/// That does not drift, because no fn-world parameter is ever spelled
/// through `Ty::cpp` — [`typecheck_tu`] spells every one through
/// [`Abi::CPP`] — and it is one more reason `Abi::TY` dies with `Ty` at
/// step 9.
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
        impl $crate::x::Abi for $rust {
            const CPP: &'static str = $cpp;
            const TY: ::kernels::Ty = ::kernels::Ty::$tag;
            #[cfg(feature = "_cuda")]
            fn arg(&self) -> $crate::runtime::ArgValue {
                // The borrow is of the caller's binding, which outlives the
                // `fire` call; `Args::bind` copies out of it before it
                // returns. See `abi`'s header for why this is `&self`.
                $crate::runtime::ArgValue::Bytes {
                    ptr: ::core::ptr::from_ref::<$rust>(self).cast::<u8>(),
                    len: ::core::mem::size_of::<$rust>(),
                }
            }
        }

        impl $crate::x::ByValue for $rust {
            const LAYOUT: $crate::x::Layout = $crate::x::Layout {
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
        const _: () = assert!(
            ::core::mem::size_of::<$rust>() == $size,
            concat!(stringify!($rust), ": sizeof disagrees with the measured ", $cpp),
        );
        const _: () = assert!(
            ::core::mem::align_of::<$rust>() == $align,
            concat!(stringify!($rust), ": alignof disagrees with the measured ", $cpp),
        );
        $(
            const _: () = assert!(
                ::core::mem::offset_of!($rust, $field) == $at,
                concat!(
                    stringify!($rust), ".", stringify!($field),
                    ": offset disagrees with the measured ", $cpp, "::", $cname,
                ),
            );
        )*
    };

    // The untagged arm: an aggregate with no `Ty` that means it.
    //
    // The tagged arm above requires `Ty::$tag.needs_mirror()`, and that is a
    // CLOSED SET OF SIX in a crate three portable backends share. `Abi` is an
    // open set — any `#[repr(C)]` mirror can implement it — so the tagged arm
    // gates an open set behind a closed one, and the gate only opened for
    // `xqa`'s `KvCacheList` because `Ty::KvCacheLayerView` already existed and
    // already meant roughly the right thing. **Eleven families produced no
    // second `by_value!`**, and this is why.
    //
    // The two ways out were both refused, for reasons already written down:
    //
    //  * Borrow a neighbouring tag. `runtime::args`' own doc bars it — "the
    //    check would pass on a `MLAParams` bound where a `HopperParams` is
    //    declared and catch nothing." A tag that is approximately right is
    //    worse than no tag, because it reads as a statement.
    //  * Add a `Ty` per aggregate. This module's header bars it — "a `Ty`
    //    variant per aggregate would have been the forty-variant `LaunchRule`
    //    mistake one level down." And step 9 is measured at shrinking `Ty`,
    //    not growing it.
    //
    // So this arm states no tag at all. `TY` is `Ty::Unstated`'s honest
    // stand-in: `Ty::MlaPlanCache`, chosen because it is on NEITHER
    // `is_pointer`'s list NOR `bind::device::scalar`'s, so a walker that did
    // consult the tag gets `ArgError::Unsupported` — a named refusal, never a
    // silent accept of eight bytes where two hundred were meant.
    //
    // Nothing consults it today. `Args::bind` short-circuits on
    // `ArgValue::Bytes` before the tag match, every fn-world operand is
    // `Source::Unbound`, and `typecheck_tu` spells parameters through
    // `Abi::CPP` rather than `Ty::cpp`. `Abi::TY` dies with `Ty` at step 9,
    // and this arm is the reason it should: **the field was never carrying a
    // fact, it was carrying a permission.**
    //
    // Everything else is identical to the tagged arm — the same `ArgValue`,
    // the same `Layout`, the same size/align/offset assertions naming the
    // same probe. The measurement is the point of the macro and this arm
    // loses none of it.
    (
        $rust:ident as $cpp:literal,
        untagged,
        probe = $probe:literal,
        size = $size:literal, align = $align:literal,
        { $($field:ident @ $at:literal as $cname:literal),* $(,)? }
    ) => {
        impl $crate::x::Abi for $rust {
            const CPP: &'static str = $cpp;
            const TY: ::kernels::Ty = ::kernels::Ty::MlaPlanCache;
            #[cfg(feature = "_cuda")]
            fn arg(&self) -> $crate::runtime::ArgValue {
                $crate::runtime::ArgValue::Bytes {
                    ptr: ::core::ptr::from_ref::<$rust>(self).cast::<u8>(),
                    len: ::core::mem::size_of::<$rust>(),
                }
            }
        }

        impl $crate::x::ByValue for $rust {
            const LAYOUT: $crate::x::Layout = $crate::x::Layout {
                cpp: $cpp,
                size: $size,
                align: $align,
                fields: &[$(($cname, $at)),*],
                probe: $probe,
            };
        }

        const _: () = assert!(
            ::core::mem::size_of::<$rust>() == $size,
            concat!(stringify!($rust), ": sizeof disagrees with the measured ", $cpp),
        );
        const _: () = assert!(
            ::core::mem::align_of::<$rust>() == $align,
            concat!(stringify!($rust), ": alignof disagrees with the measured ", $cpp),
        );
        $(
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
/// `layouts` is the unit's by-value aggregates — every
/// [`ByValue::LAYOUT`] named by any row's parameter list. Each becomes
/// `static_assert`s on `sizeof`, `alignof` and every field's `offsetof`,
/// which is the SAME measurement `by_value!` asserts on the Rust side. That
/// is deliberate: the Rust assertions catch a drifted mirror, these catch a
/// drifted header, and only both together catch a rename that moved a field
/// in the header while someone updated the mirror to match the wrong
/// numbers. A unit with no by-value parameter passes `&[]`.
///
/// # Panics
///
/// When `params` and `unit.rows` are different lengths, which is drift
/// between two things one macro writes and therefore cannot happen without
/// someone having edited generated output by hand.
#[must_use]
pub fn typecheck_tu(
    unit: &crate::unit::Unit,
    params: &[&[&str]],
    layouts: &[Layout],
    include: &str,
) -> String {
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
    out.push_str("#include <cstddef>\n#include <type_traits>\n#include \"");
    out.push_str(include);
    out.push_str("\"\n\nnamespace {\n");
    for layout in layouts {
        out.push_str("\n// ");
        out.push_str(layout.cpp);
        out.push_str(" — by value. Measured by ");
        out.push_str(layout.probe);
        out.push_str(";\n// the same numbers are asserted on the Rust mirror by `by_value!`.\n");
        // `offsetof` and not the pointer-difference form: this TU is compiled
        // by the host compiler, where `offsetof` is the constant expression
        // and `(char*)&((T*)0)->f - (char*)(T*)0` is not. Under NVRTC it is
        // the other way round — `offsetof` is unavailable there and the
        // difference folds — which is why the probe uses the other spelling.
        push_static_assert(
            &mut out,
            &format!("sizeof({}) == {}", layout.cpp, layout.size),
            &format!("{}: sizeof moved; re-run {}", layout.cpp, layout.probe),
        );
        push_static_assert(
            &mut out,
            &format!("alignof({}) == {}", layout.cpp, layout.align),
            &format!("{}: alignof moved; re-run {}", layout.cpp, layout.probe),
        );
        for (field, at) in layout.fields {
            push_static_assert(
                &mut out,
                &format!("offsetof({}, {field}) == {at}", layout.cpp),
                &format!("{}::{field} moved; re-run {}", layout.cpp, layout.probe),
            );
        }
    }
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

/// One `static_assert`, wrapped the way the rest of the TU wraps them.
fn push_static_assert(out: &mut String, cond: &str, message: &str) {
    out.push_str("static_assert(");
    out.push_str(cond);
    out.push_str(",\n    \"");
    out.push_str(message);
    out.push_str("\");\n");
}

/// A symbol as a C++ identifier.
fn mangle(symbol: &str) -> String {
    symbol.replace([':', '<', '>', ',', ' ', '(', ')', '#'], "_")
}
