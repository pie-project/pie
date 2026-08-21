use core::ffi::c_void;
use core::ptr::NonNull;

use kernels::Ty;

pub trait Abi: Copy {
    const CPP: &'static str;

    const TY: Ty;

    const NULLABLE: bool = false;

    fn arg(&self) -> crate::jit::ArgValue;

    fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal>;
}

const fn wrong_kind(at: usize, want: Ty) -> kernels::Refusal {
    kernels::Refusal::Kind { at, want }
}

pub fn unpack_aggregate<T: Copy>(
    value: &crate::jit::ArgValue,
    at: usize,
    want: Ty,
) -> Result<T, kernels::Refusal> {
    match value {
        crate::jit::ArgValue::Bytes { ptr, len } if *len == core::mem::size_of::<T>() => {
            Ok(unsafe { ptr.cast::<T>().read_unaligned() })
        }
        _ => Err(wrong_kind(at, want)),
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct bf16(pub u16);

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct f16(pub u16);

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct fp8_e4m3(pub u8);

pub trait Inst {
    const CPP: &'static str;
}

impl Inst for bf16 {
    const CPP: &'static str = "::pie::bf16";
}

impl Inst for f16 {
    const CPP: &'static str = "::pie::f16";
}

impl Inst for fp8_e4m3 {
    const CPP: &'static str = "::pie::fp8_e4m3";
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct u16_(pub u16);

impl Inst for u16_ {
    const CPP: &'static str = "::pie::u16";
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ssm_f32;

impl Inst for ssm_f32 {
    const CPP: &'static str = "::pie::ssm::f32";
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ssm_state_bf16;

impl Inst for ssm_state_bf16 {
    const CPP: &'static str = "::pie::ssm::state_bf16";
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct quant_f32;

impl Inst for quant_f32 {
    const CPP: &'static str = "::pie::quant::f32";
}

#[derive(Debug)]
#[repr(transparent)]
pub struct MaybeConst<T>(pub Option<NonNull<T>>);

impl<T> Clone for MaybeConst<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for MaybeConst<T> {}

impl<T: 'static> kernels::Elem for MaybeConst<T> {
    type Read = Self;
    type Write = Self;

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
    #[must_use]
    pub fn new(p: *const T) -> Self {
        Self(NonNull::new(p.cast_mut()))
    }

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
                    crate::jit::ArgValue::Ptr(p) | crate::jit::ArgValue::Region { ptr: p, .. } => {
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
                    crate::jit::ArgValue::Ptr(p) | crate::jit::ArgValue::Region { ptr: p, .. } => {
                        Ok(p.cast::<$pointee>())
                    }
                    _ => Err(wrong_kind(at, Ty::$mut_ty)),
                }
            }
        }

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
                    crate::jit::ArgValue::Ptr(p) | crate::jit::ArgValue::Region { ptr: p, .. } => {
                        Ok(NonNull::new(p.cast::<$pointee>()))
                    }
                    _ => Err(wrong_kind(at, Ty::$mut_ty)),
                }
            }
        }

        impl Abi for MaybeConst<$pointee> {
            const CPP: &'static str = $const_cpp;
            const TY: Ty = Ty::$const_ty;
            const NULLABLE: bool = true;
            fn arg(&self) -> crate::jit::ArgValue {
                crate::jit::ArgValue::Ptr(
                    self.0
                        .map_or(core::ptr::null_mut(), |p| p.as_ptr().cast::<c_void>()),
                )
            }
            fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
                match value {
                    crate::jit::ArgValue::Ptr(p) | crate::jit::ArgValue::Region { ptr: p, .. } => {
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

macro_rules! elem_agrees {
    ($($pointee:ty),* $(,)?) => {
        $(const _: () = {
            assert!(<*const $pointee as Abi>::TY as u8 == <$pointee as kernels::Elem>::TY_CONST as u8);
            assert!(<*mut $pointee as Abi>::TY as u8 == <$pointee as kernels::Elem>::TY_MUT as u8);
        };)*
    };
}

#[derive(Debug)]
pub struct Tensor<E>(core::marker::PhantomData<E>);

impl<E> Clone for Tensor<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E> Copy for Tensor<E> {}

impl<E: kernels::Elem> kernels::Elem for Tensor<E> {
    type Read = E::Read;
    type Write = E::Write;

    unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
        unsafe { E::advance_read(read, elems) }
    }

    unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
        unsafe { E::advance_write(write, elems) }
    }

    const CPP_CONST: &'static str = E::CPP_CONST;
    const CPP_MUT: &'static str = E::CPP_MUT;
    const TY_CONST: Ty = E::TY_CONST;
    const TY_MUT: Ty = E::TY_MUT;
}

impl<E: kernels::Elem> kernels::ConstRun for Tensor<E> {
    const RUN: kernels::routine::Claim = kernels::routine::Claim::Weight;
    const TY: Ty = E::TY_CONST;
    type Held = E::Read;
}

impl kernels::Elem for bf16 {
    type Read = *const bf16;
    type Write = *mut bf16;

    unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
        unsafe { read.add(elems) }
    }

    unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
        unsafe { write.add(elems) }
    }

    const CPP_CONST: &'static str = "const ::pie::bf16*";
    const CPP_MUT: &'static str = "::pie::bf16*";
    const TY_CONST: Ty = Ty::Bf16s;
    const TY_MUT: Ty = Ty::Bf16sMut;
}

impl kernels::Elem for f16 {
    type Read = *const f16;
    type Write = *mut f16;

    unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
        unsafe { read.add(elems) }
    }

    unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
        unsafe { write.add(elems) }
    }

    const CPP_CONST: &'static str = "const ::pie::f16*";
    const CPP_MUT: &'static str = "::pie::f16*";
    const TY_CONST: Ty = Ty::F16s;
    const TY_MUT: Ty = Ty::F16sMut;
}

ptr_abi!(bf16, "const ::pie::bf16*", Bf16s, "::pie::bf16*", Bf16sMut);
ptr_abi!(f16, "const ::pie::f16*", F16s, "::pie::f16*", F16sMut);
ptr_abi!(
    fp8_e4m3,
    "const ::pie::fp8_e4m3*",
    Buf,
    "::pie::fp8_e4m3*",
    BufMut
);
ptr_abi!(
    i32,
    "const ::std::int32_t*",
    I32s,
    "::std::int32_t*",
    I32sMut
);

ptr_abi!(
    i64,
    "const ::std::int64_t*",
    I64s,
    "::std::int64_t*",
    BufMut
);

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
ptr_abi!(
    u32,
    "const ::std::uint32_t*",
    U32s,
    "::std::uint32_t*",
    U32sMut
);
ptr_abi!(u8, "const ::std::uint8_t*", U8s, "::std::uint8_t*", U8sMut);

ptr_abi!(
    u16,
    "const ::std::uint16_t*",
    U16s,
    "::std::uint16_t*",
    U16sMut
);
ptr_abi!(f32, "const float*", F32s, "float*", F32sMut);

ptr_abi!(c_void, "const void*", Buf, "void*", BufMut);

ptr_abi!(
    *const c_void,
    "const void* const*",
    BufArray,
    "const void**",
    BufArrayOut
);
ptr_abi!(
    *mut c_void,
    "void* const*",
    BufArrayMut,
    "void**",
    BufArrayOutMut
);

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
            crate::jit::ArgValue::Ptr(p) | crate::jit::ArgValue::Region { ptr: p, .. } => {
                Ok(Self(*p))
            }
            _ => Err(wrong_kind(at, Ty::Stream)),
        }
    }
}

crate::arg_via_abi!(Stream);

pub type DevicePtr = u64;

#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub cpp: &'static str,
    pub size: usize,
    pub align: usize,
    pub fields: &'static [(&'static str, usize)],
    pub probe: &'static str,
}

pub trait ByValue: Abi {
    const LAYOUT: Layout;
}

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

#[derive(Clone, Copy, Debug)]
pub struct Bytes<'a> {
    pub bytes: &'a [u8],
    pub cpp: &'static str,
}

impl<'a> Bytes<'a> {
    #[must_use]
    pub fn arg(self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::Bytes {
            ptr: self.bytes.as_ptr(),
            len: self.bytes.len(),
        }
    }
}

pub const TYPECHECK_ENTRY: &str = "::pie::typecheck::probe";

fn push_static_assert(out: &mut String, cond: &str, message: &str) {
    out.push_str("static_assert(");
    out.push_str(cond);
    out.push_str(",\n    \"");
    out.push_str(message);
    out.push_str("\");\n");
}

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
                &format!(
                    "{field}'s offset disagrees with the measurement in {}",
                    layout.probe
                ),
            );
        }
    }
    out
}

elem_agrees!(bf16, f16, i32, i64, i8, u32, u8, u16, f32, c_void);

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
