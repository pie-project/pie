use crate::Ty;
use crate::plane::Elem;

pub trait Raise: 'static {
    const KEY: &'static str;

    type Value: 'static;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Class {
    pub head_dim: u32,

    pub window: u32,
}

impl Class {
    pub const ANY: Self = Self {
        head_dim: 0,
        window: 0,
    };

    #[must_use]
    pub const fn attention(head_dim: u32, window: u32) -> Self {
        Self { head_dim, window }
    }

    #[must_use]
    pub const fn is_any(&self) -> bool {
        self.head_dim == 0 && self.window == 0
    }
}

pub trait Answered {
    fn raised(&self, key: &'static str, class: Class) -> Option<*const core::ffi::c_void>;
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
