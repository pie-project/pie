use crate::error::Error;
use crate::tensor::Comm;

pub const ABSENT: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    Buffer(u32),
    BufferMut(u32),
    I32(i32),
    U32(u32),
    F32(f32),
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
        }
    }
}

pub trait Arg: Copy {
    fn arg(self) -> ArgValue;
}

impl Arg for i32 {
    fn arg(self) -> ArgValue {
        ArgValue::I32(self)
    }
}

impl Arg for u32 {
    fn arg(self) -> ArgValue {
        ArgValue::U32(self)
    }
}

impl Arg for f32 {
    fn arg(self) -> ArgValue {
        ArgValue::F32(self)
    }
}

pub trait Encode {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error>;

    fn absent(&self) -> Result<ArgValue, Error>;

    fn comm(&self, op: &'static str) -> Result<Comm, Error> {
        Err(Error::Unsupported { op })
    }

    fn rendezvous(&self, _op: &'static str) -> Result<(), Error> {
        Ok(())
    }
}

pub type Ctx<'a> = dyn Encode + 'a;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fire {
    pub file: &'static str,

    pub entrypoint: &'static str,

    pub groups: [u32; 3],

    pub group: [u32; 3],
}

impl Fire {
    #[must_use]
    pub const fn at(file: &'static str, entrypoint: &'static str) -> Self {
        Self {
            file,
            entrypoint,
            groups: [1, 1, 1],
            group: [1, 1, 1],
        }
    }

    #[must_use]
    pub const fn groups(mut self, groups: [u32; 3]) -> Self {
        self.groups = groups;
        self
    }

    #[must_use]
    pub const fn group(mut self, group: [u32; 3]) -> Self {
        self.group = group;
        self
    }

    #[must_use]
    pub const fn threads(mut self, lanes: [u32; 3], group: [u32; 3]) -> Self {
        self.group = group;
        let mut i = 0;
        while i < 3 {
            let extent = if group[i] == 0 { 1 } else { group[i] };
            self.groups[i] = lanes[i].div_ceil(extent);
            i += 1;
        }
        self
    }

    #[must_use]
    pub fn apply<G: Geometry>(self, g: G) -> Self {
        g.apply_to(self)
    }
}

pub trait Geometry {
    #[must_use]
    fn apply_to(self, fire: Fire) -> Fire;
}

impl Geometry for [u32; 3] {
    fn apply_to(self, fire: Fire) -> Fire {
        fire.threads(self, fire.group)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub lanes: [u32; 3],

    pub group: [u32; 3],
}

impl Grid {
    #[must_use]
    pub const fn of(lanes: [u32; 3], group: [u32; 3]) -> Self {
        Self { lanes, group }
    }
}

impl Geometry for Grid {
    fn apply_to(self, fire: Fire) -> Fire {
        fire.threads(self.lanes, self.group)
    }
}

pub fn elementwise(op: &'static str, width: u32, rows: u32) -> Result<[u32; 3], Error> {
    nonzero(op, "width", width)?;
    nonzero(op, "rows", rows)?;
    let n = u64::from(width) * u64::from(rows);
    let n = u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("the grid will not launch: {width} x {rows} lanes"),
        )
    })?;
    Ok([n, 1, 1])
}

pub fn elementwise_rows(op: &'static str, width: u32, rows: u32) -> Result<[u32; 3], Error> {
    nonzero(op, "width", width)?;
    nonzero(op, "rows", rows)?;
    Ok([width, rows, 1])
}

pub fn head_grid(
    op: &'static str,
    head_dim: u32,
    heads: u32,
    depth: u32,
) -> Result<[u32; 3], Error> {
    Ok([
        nonzero(op, "the head width", head_dim)?,
        nonzero(op, "heads", heads)?,
        nonzero(op, "tokens", depth)?,
    ])
}

#[must_use]
pub const fn head_group(grid: [u32; 3]) -> [u32; 3] {
    [grid[0], 1, 1]
}

pub(crate) fn refuse(op: &'static str, detail: impl Into<String>) -> Error {
    Error::Backend {
        op,
        detail: detail.into(),
    }
}

pub(crate) fn nonzero(op: &'static str, axis: &'static str, v: u32) -> Result<u32, Error> {
    if v == 0 {
        return Err(refuse(op, format!("`{axis}` is zero")));
    }
    Ok(v)
}

#[allow(dead_code)]
pub(crate) fn stated(op: &'static str, v: u32) -> Result<i32, Error> {
    i32::try_from(v).map_err(|_| refuse(op, format!("{v} does not fit the shader's int")))
}

#[allow(unused_macros)]
macro_rules! dtype_dispatch {
    ($op:expr, $dtype:expr, { $($stamped:ident => $arm:expr),+ $(,)? }) => {
        match $dtype {
            $(::dtype::Dtype::$stamped => $arm,)+
            other => {
                return Err(crate::error::Error::DtypeUnsupported {
                    op: $op,
                    dtype: other,
                });
            }
        }
    };
}

#[allow(unused_imports)]
pub(crate) use dtype_dispatch;
