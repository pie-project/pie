use kernels::plane::{Backend, Extent, Refusal};
use kernels::shader::ShaderValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Metal;

impl Backend for Metal {
    type Value = ArgValue;
    type Ctx<'a> = dyn Encode + 'a;

    fn region(value: &ArgValue) -> Result<Extent, Refusal> {
        match *value {
            ArgValue::Shaped { rows, width, .. } => Ok(Extent { rows, width }),
            _ => Err(Refusal::Absent {
                what: "a region's shape: the bound value carries only a handle",
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    Buffer(u32),
    BufferMut(u32),
    I32(i32),
    U32(u32),
    F32(f32),
    Usize(u64),

    Raised(usize),
    Shaped { handle: u32, rows: i32, width: i32 },
}

impl kernels::plane::Absent for ArgValue {}

impl ArgValue {
    #[must_use]
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Buffer(_) => "a buffer",
            Self::Shaped { .. } => "a buffer",
            Self::BufferMut(_) => "a writable buffer",
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

    fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal>;
}

impl ShaderValue for ArgValue {
    fn as_buffer(self) -> Option<u32> {
        match self {
            Self::Shaped { handle, .. } => Some(handle),
            Self::Buffer(handle) | Self::BufferMut(handle) => Some(handle),
            _ => None,
        }
    }
    fn buffer_mut(handle: u32) -> Self {
        Self::BufferMut(handle)
    }
    fn as_i32(self) -> Option<i32> {
        match self {
            Self::I32(v) => Some(v),
            _ => None,
        }
    }
    fn as_u32(self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(v),
            _ => None,
        }
    }
    fn as_f32(self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }
    fn as_usize(self) -> Option<u64> {
        match self {
            Self::Usize(v) => Some(v),
            _ => None,
        }
    }
    fn as_raised(self) -> Option<usize> {
        match self {
            Self::Raised(a) => Some(a),
            _ => None,
        }
    }
    fn raised(addr: usize) -> Self {
        Self::Raised(addr)
    }
    fn as_extent(self) -> Option<(i32, i32)> {
        match self {
            Self::Shaped { rows, width, .. } => Some((rows, width)),
            _ => None,
        }
    }
    fn buffer_at(handle: u32, rows: i32, width: i32) -> Self {
        Self::Shaped {
            handle,
            rows,
            width,
        }
    }
    fn buffer_mut_at(handle: u32, rows: i32, width: i32) -> Self {
        Self::Shaped {
            handle,
            rows,
            width,
        }
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

impl kernels::plane::Answers<Metal> for Ctx<'_> {
    fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
        Encode::resolve(self, ty, source)
    }
}

pub use kernels::shader::{elementwise, elementwise_rows};

pub use kernels::plane::{Const, Fire, In, InOut, Out};

pub use kernels::plane::{Answers, Asks};
