use core::ffi::c_void;
use core::ptr::NonNull;

pub trait Abi: Copy {
    const CPP: &'static str;

    const NULLABLE: bool = false;

    fn arg(&self) -> crate::jit::ArgValue;
}

#[macro_export]
macro_rules! bind_via_abi {
    ($($rust:ty),* $(,)?) => {
        $(
            impl ::kernels::plane::Bind<$crate::jit::ArgValue> for $rust {
                fn arg(self) -> $crate::jit::ArgValue {
                    <$rust as $crate::jit::Abi>::arg(&self)
                }
            }
        )*
    };
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

    const CPP: &'static str = "void";
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
    ($rust:ty, $cpp:literal, $arg:ident) => {
        impl Abi for $rust {
            const CPP: &'static str = $cpp;
            fn arg(&self) -> crate::jit::ArgValue {
                crate::jit::ArgValue::$arg(*self)
            }
        }
        $crate::bind_via_abi!($rust);
    };
}

scalar_abi!(i32, "int", I32);
scalar_abi!(u32, "unsigned int", U32);
scalar_abi!(f32, "float", F32);
scalar_abi!(bool, "bool", Bool);
scalar_abi!(i64, "long long", I64);
scalar_abi!(usize, "std::size_t", Usize);

impl Abi for u64 {
    const CPP: &'static str = "std::size_t";
    fn arg(&self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::Usize(*self as usize)
    }
}
crate::bind_via_abi!(u64);

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct fp8_kind(pub u32);

impl Abi for fp8_kind {
    const CPP: &'static str = "::__nv_fp8_interpretation_t";
    fn arg(&self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::U32(self.0)
    }
}

crate::bind_via_abi!(fp8_kind);

macro_rules! ptr_abi {
    ($pointee:ty, $const_cpp:literal, $mut_cpp:literal) => {
        impl Abi for *const $pointee {
            const CPP: &'static str = $const_cpp;
            fn arg(&self) -> crate::jit::ArgValue {
                crate::jit::ArgValue::Ptr(*self as *mut c_void)
            }
        }
        impl Abi for *mut $pointee {
            const CPP: &'static str = $mut_cpp;
            fn arg(&self) -> crate::jit::ArgValue {
                crate::jit::ArgValue::Ptr(self.cast::<c_void>())
            }
        }

        impl Abi for Option<NonNull<$pointee>> {
            const CPP: &'static str = $mut_cpp;
            const NULLABLE: bool = true;
            fn arg(&self) -> crate::jit::ArgValue {
                crate::jit::ArgValue::Ptr(
                    self.map_or(core::ptr::null_mut(), |p| p.as_ptr().cast::<c_void>()),
                )
            }
        }

        impl Abi for MaybeConst<$pointee> {
            const CPP: &'static str = $const_cpp;
            const NULLABLE: bool = true;
            fn arg(&self) -> crate::jit::ArgValue {
                crate::jit::ArgValue::Ptr(
                    self.0
                        .map_or(core::ptr::null_mut(), |p| p.as_ptr().cast::<c_void>()),
                )
            }
        }
        $crate::bind_via_abi!(Option<NonNull<$pointee>>, MaybeConst<$pointee>);
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

    const CPP: &'static str = E::CPP;
}

impl<E: kernels::Elem> kernels::ConstRun for Tensor<E> {
    type Held = E::Read;
}

#[derive(Debug)]
pub struct Bank<R>(core::marker::PhantomData<R>);

impl<R> Clone for Bank<R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R> Copy for Bank<R> {}

#[derive(Debug, Clone, Copy)]
pub struct Planes {
    pub codes: *const u8,

    pub scales: *const u8,
}

impl<R: kernels::points::Repr> kernels::ConstRun for Bank<R> {
    type Held = Planes;
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

    const CPP: &'static str = "::pie::bf16";
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

    const CPP: &'static str = "::pie::f16";
}

impl kernels::points::Scalar for bf16 {
    const KIND: kernels::points::ScalarKind = kernels::points::ScalarKind::Bf16;
}

impl kernels::points::Scalar for f16 {
    const KIND: kernels::points::ScalarKind = kernels::points::ScalarKind::F16;
}

ptr_abi!(bf16, "const ::pie::bf16*", "::pie::bf16*");
ptr_abi!(f16, "const ::pie::f16*", "::pie::f16*");
ptr_abi!(fp8_e4m3, "const ::pie::fp8_e4m3*", "::pie::fp8_e4m3*");
ptr_abi!(i32, "const ::std::int32_t*", "::std::int32_t*");

ptr_abi!(i64, "const ::std::int64_t*", "::std::int64_t*");

ptr_abi!(
    *const bf16,
    "const ::pie::bf16* const*",
    "const ::pie::bf16**"
);
ptr_abi!(*mut bf16, "::pie::bf16* const*", "::pie::bf16**");
ptr_abi!(
    *const u8,
    "const ::std::uint8_t* const*",
    "const ::std::uint8_t**"
);
ptr_abi!(
    *const i32,
    "const ::std::int32_t* const*",
    "const ::std::int32_t**"
);
ptr_abi!(i8, "const ::std::int8_t*", "::std::int8_t*");
ptr_abi!(u32, "const ::std::uint32_t*", "::std::uint32_t*");
ptr_abi!(u8, "const ::std::uint8_t*", "::std::uint8_t*");

ptr_abi!(u16, "const ::std::uint16_t*", "::std::uint16_t*");
ptr_abi!(f32, "const float*", "float*");

ptr_abi!(c_void, "const void*", "void*");

ptr_abi!(*const c_void, "const void* const*", "const void**");
ptr_abi!(*mut c_void, "void* const*", "void**");

#[derive(Clone, Copy, Debug)]
pub struct Stream(pub *mut c_void);

impl Abi for Stream {
    const CPP: &'static str = "cudaStream_t";
    fn arg(&self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::Ptr(self.0)
    }
}

crate::bind_via_abi!(Stream);

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
        probe = $probe:literal,
        size = $size:literal, align = $align:literal,
        { $($field:ident @ $at:literal as $cname:literal),* $(,)? }
    ) => {
        impl $crate::jit::Abi for $rust {
            const CPP: &'static str = $cpp;
            fn arg(&self) -> $crate::jit::ArgValue {
                $crate::jit::ArgValue::Bytes {
                    ptr: ::core::ptr::from_ref::<$rust>(self).cast::<u8>(),
                    len: ::core::mem::size_of::<$rust>(),
                }
            }
        }
        $crate::bind_via_abi!($rust);

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

pub trait Pointee:
    kernels::plane::Elem<
        Read: kernels::plane::Bind<crate::jit::ArgValue>,
        Write: kernels::plane::Bind<crate::jit::ArgValue>
                   + kernels::plane::BindMut<crate::jit::ArgValue>,
    >
{
}

impl<T> Pointee for T where
    T: kernels::plane::Elem<
            Read: kernels::plane::Bind<crate::jit::ArgValue>,
            Write: kernels::plane::Bind<crate::jit::ArgValue>
                       + kernels::plane::BindMut<crate::jit::ArgValue>,
        >
{
}
