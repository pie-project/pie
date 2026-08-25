use core::marker::PhantomData;

use kernels::plane::Refusal;
use kernels::shader::ShaderValue;

use crate::plane::{ArgValue, Ctx};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handle(pub u32);

impl<V: ShaderValue> kernels::Bind<V> for Handle {
    fn arg(self) -> V {
        V::buffer(self.0)
    }
}

impl<V: ShaderValue> kernels::BindMut<V> for Handle {
    fn arg_mut(self) -> V {
        V::buffer_mut(self.0)
    }
}

#[derive(Debug)]
pub struct Payload<T>(PhantomData<T>);

impl<T> Clone for Payload<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Payload<T> {}

impl<T: kernels::points::Scalar> kernels::Elem for Payload<T> {
    type Read = Handle;
    type Write = Handle;

    unsafe fn advance_read(read: Self::Read, _elems: usize) -> Self::Read {
        read
    }

    unsafe fn advance_write(write: Self::Write, _elems: usize) -> Self::Write {
        write
    }

    const CPP: &'static str = "";
}

impl<T: kernels::points::Scalar> kernels::ConstRun for Payload<T> {
    type Held = Handle;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankHandles {
    pub codes: Handle,

    pub scales: Handle,
}

#[derive(Debug)]
pub struct Bank<R>(PhantomData<R>);

impl<R> Clone for Bank<R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R> Copy for Bank<R> {}

impl<R: kernels::points::Repr> kernels::ConstRun for Bank<R> {
    type Held = BankHandles;
}

pub fn at_bf16<T: kernels::points::Scalar>(what: &'static str) -> Result<(), Refusal> {
    if T::KIND == kernels::points::ScalarKind::Bf16 {
        Ok(())
    } else {
        Err(Refusal::Absent { what })
    }
}

pub(crate) fn absent(ctx: &Ctx<'_>) -> Result<ArgValue, Refusal> {
    ctx.absent()
}

impl kernels::points::Plane for Ctx<'_> {
    type Tensor<T: kernels::points::Scalar> = Payload<T>;

    type Bank<R: kernels::points::Repr> = Bank<R>;

    type Recurrent = kernels::raises::Struct<crate::views::RecurrentState>;

    type Pages = kernels::raises::Struct<crate::views::AttnFire>;
}

#[kernels_macros::claims]
impl kernels::points::Dist for Ctx<'_> {}

#[kernels_macros::claims]
impl kernels::points::Mla for Ctx<'_> {}

#[kernels_macros::claims]
impl kernels::points::Index for Ctx<'_> {}

#[kernels_macros::claims]
impl kernels::points::Pool for Ctx<'_> {}

#[kernels_macros::claims]
impl kernels::points::Hc for Ctx<'_> {}

pub use kernels::shader::bf16;
