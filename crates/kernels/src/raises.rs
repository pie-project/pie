use crate::Ty;
use crate::routine::Elem;

pub trait Raise: 'static {
    const KEY: &'static str;

    type Value: 'static;

    const RESIDENT: bool = false;
}

/// What answers a raise BY ITS KEY — the executor's staging, reached from a
/// `#[claims]` body through the plane's own context.
///
/// THE DOOR A GENERATED DISPATCH CANNOT OPEN. A `BoundOp` answers by COLUMN:
/// every accessor it has reads a slot the statement carries, and the whole
/// point of a raise is that no statement carries it. A routine took its
/// staging as an extra operand and the driver bound it positionally; a body
/// takes the same thing off `self`, and the only name it has for it is the
/// one the [`Raise`] declares. So the executor answers a `&'static str` and
/// nothing else: `driver-cuda/src/bind/views.rs::FireViews::raised` was
/// written to this shape before it had a caller, and this is the caller.
///
/// `None` IS A REFUSAL AND NEVER A NULL, with one exception the caller names
/// rather than the answerer: an object whose ABSENCE a kernel reads — the
/// row-validity plane, a mask nothing customised — is asked for through the
/// plane's optional door, and everything else refuses with the key in it.
/// An answerer that returns `Some(null)` has lied about staging something.
pub trait Answered {
    /// The staged object registered under `key`, or `None` when this fire
    /// staged none.
    fn raised(&self, key: &'static str) -> Option<*const core::ffi::c_void>;
}

pub struct Struct<T: Raise>(core::marker::PhantomData<T>);

impl<T: Raise> core::fmt::Debug for Struct<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Struct({})", T::KEY)
    }
}

impl<T: Raise> Elem for Struct<T> {
    type Read = *const T::Value;
    type Write = *mut T::Value;

    unsafe fn advance_read(read: Self::Read, _elems: usize) -> Self::Read {
        read
    }

    unsafe fn advance_write(write: Self::Write, _elems: usize) -> Self::Write {
        write
    }

    const CPP: &'static str = "";
    const CPP_CONST: &'static str = "";
    const CPP_MUT: &'static str = "";

    const TY_CONST: Ty = Ty::Raised;
    const TY_MUT: Ty = Ty::Raised;
}

#[macro_export]
macro_rules! raise {
    ($(#[$m:meta])* $name:ident = $key:literal => $value:ty) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name;

        impl $crate::raises::Raise for $name {
            const KEY: &'static str = $key;
            type Value = $value;
        }
    };
}

#[macro_export]
macro_rules! resident {
    ($(#[$m:meta])* $name:ident = $key:literal => $value:ty) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name;

        impl $crate::raises::Raise for $name {
            const KEY: &'static str = $key;
            type Value = $value;
            const RESIDENT: bool = true;
        }
    };
}
