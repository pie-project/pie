use kernels::plane::Refusal;
use kernels::shader::ShaderValue;

pub use crate::Capability;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vulkan;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    Buffer {
        handle: u32,
        writes: bool,
        rows: i32,
        width: i32,
    },
    I32(i32),
    U32(u32),
    F32(f32),
    Usize(u64),

    Raised(usize),
}

impl ArgValue {
    #[must_use]
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Buffer { .. } => "a buffer",
            Self::I32(_) => "an i32",
            Self::U32(_) => "a u32",
            Self::F32(_) => "an f32",
            Self::Usize(_) => "a usize",
            Self::Raised(_) => "a raised view",
        }
    }
}

pub trait Encode {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal>;

    fn staged(&self, name: &'static str) -> Result<u32, Refusal>;

    fn windowed(&self, of: u32, at: u64) -> Result<u32, Refusal>;

    fn best(&self) -> Capability {
        Capability::Baseline
    }
}

impl ShaderValue for ArgValue {
    fn buffer(handle: u32) -> Self {
        Self::Buffer {
            handle,
            writes: false,
            rows: 0,
            width: 0,
        }
    }
    fn buffer_mut(handle: u32) -> Self {
        Self::Buffer {
            handle,
            writes: true,
            rows: 0,
            width: 0,
        }
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

pub use kernels::shader::Bind;

pub type Ctx<'a> = dyn Encode + 'a;

pub use kernels::shader::{elementwise, elementwise_rows};

pub use kernels::plane::{Const, Fire, In, InOut, Out};

pub use crate::module::path as module_path;
