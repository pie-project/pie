use kernels::plane::Refusal;
use kernels::shader::ShaderValue;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    Buffer(u32),
    BufferMut(u32),
    I32(i32),
    U32(u32),
    F32(f32),
    Usize(u64),
}

impl ArgValue {
    #[must_use]
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Buffer(_) => "a buffer",
            Self::BufferMut(_) => "a writable buffer",
            Self::I32(_) => "an i32",
            Self::U32(_) => "a u32",
            Self::F32(_) => "an f32",
            Self::Usize(_) => "a usize",
        }
    }
}

pub trait Encode {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal>;

    fn absent(&self) -> Result<ArgValue, Refusal>;
}

impl ShaderValue for ArgValue {
    fn buffer_mut(handle: u32) -> Self {
        Self::BufferMut(handle)
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

pub use kernels::shader::{Bind, InPacked, Tensor, Usize, bf16};

pub type Ctx<'a> = dyn Encode + 'a;

pub use kernels::shader::{elementwise, elementwise_rows};

pub use kernels::plane::{Const, Fire, In, InOut, Out};
