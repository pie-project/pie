use crate::Ty;
use crate::routine::Elem;

pub trait Raise: 'static {
    const KEY: &'static str;

    type Value: 'static;

    const RESIDENT: bool = false;
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
