//! The encode plane: a kernel entry names a shader ([`Fire`]), marshals
//! arguments ([`ArgValue`]), and hands both to an [`Encode`] sink, which
//! resolves buffer handles and encodes the dispatch into the current command
//! buffer. Encode only, never sync.

use crate::error::Error;

/// One marshalled shader argument. Buffers travel as driver-scoped `u32`
/// handles; `Buffer` vs `BufferMut` is where read/write intent is recorded.
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

/// Scalar-to-argument marshalling.
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

impl Arg for u64 {
    fn arg(self) -> ArgValue {
        ArgValue::Usize(self)
    }
}

/// What a driver offers a kernel entry.
pub trait Encode {
    /// Encode one shader dispatch. Enqueue only — a returned `Ok` means the
    /// launch is in the command buffer, not that it ran.
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error>;

    /// The plane's stand-in for an optional buffer an op does not carry (a
    /// null binding). The attn/moe entries hold the optional slots.
    fn absent(&self) -> Result<ArgValue, Error>;
}

/// The context every kernel entry takes: any encode sink, behind `dyn` so
/// this crate never names a driver type.
pub type Ctx<'a> = dyn Encode + 'a;

/// One shader launch: the `.metal` file, the entrypoint, the dispatch
/// geometry, and the jit stamp a specialized point carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fire {
    pub file: &'static str,

    pub entrypoint: &'static str,

    /// Jit instantiation stamp for entrypoints the shader source does not
    /// spell out (see `quant`); empty for stamped-in-source points.
    pub stamp: &'static str,

    /// Total threads per axis.
    pub lanes: [u32; 3],

    /// Threadgroup extent per axis.
    pub group: [u32; 3],
}

impl Fire {
    #[must_use]
    pub const fn at(file: &'static str, entrypoint: &'static str) -> Self {
        Self {
            file,
            entrypoint,
            stamp: "",
            lanes: [0, 0, 0],
            group: [0, 0, 0],
        }
    }

    #[must_use]
    pub const fn stamp(mut self, stamp: &'static str) -> Self {
        self.stamp = stamp;
        self
    }

    #[must_use]
    pub const fn lanes(mut self, lanes: [u32; 3]) -> Self {
        self.lanes = lanes;
        self
    }

    #[must_use]
    pub const fn group(mut self, group: [u32; 3]) -> Self {
        self.group = group;
        self
    }

    #[must_use]
    pub fn apply<G: Geometry>(self, g: G) -> Self {
        g.apply_to(self)
    }
}

/// Anything that can finish a [`Fire`]'s geometry.
pub trait Geometry {
    #[must_use]
    fn apply_to(self, fire: Fire) -> Fire;
}

impl Geometry for [u32; 3] {
    fn apply_to(self, fire: Fire) -> Fire {
        fire.lanes(self)
    }
}

/// Lanes and threadgroup together — the shape most entries compute.
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
        fire.lanes(self.lanes).group(self.group)
    }
}

/// One thread per element, flattened: `[width * rows, 1, 1]`.
pub fn elementwise(op: &'static str, width: u32, rows: u32) -> Result<[u32; 3], Error> {
    nonzero(op, "width", width)?;
    nonzero(op, "rows", rows)?;
    let n = u64::from(width) * u64::from(rows);
    let n = u32::try_from(n)
        .map_err(|_| refuse(op, format!("the grid will not launch: {width} x {rows} lanes")))?;
    Ok([n, 1, 1])
}

/// One thread per element, rows kept on their own axis: `[width, rows, 1]`.
pub fn elementwise_rows(op: &'static str, width: u32, rows: u32) -> Result<[u32; 3], Error> {
    nonzero(op, "width", width)?;
    nonzero(op, "rows", rows)?;
    Ok([width, rows, 1])
}

/// One thread per element of a `[head_dim x heads]` slab per token:
/// `[head_dim, heads, tokens]`.
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

/// The threadgroup pairing [`head_grid`]: one head-row of threads.
#[must_use]
pub const fn head_group(grid: [u32; 3]) -> [u32; 3] {
    [grid[0], 1, 1]
}

/// An encode this backend cannot perform: degenerate or overflowing
/// geometry, an axis point no shader is stamped for. Reserved for facts only
/// the driver binds; cross-operand shape agreement is the validator's job.
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

/// An extent stated to a shader that reads it as `int`.
pub(crate) fn stated(op: &'static str, v: u32) -> Result<i32, Error> {
    i32::try_from(v).map_err(|_| refuse(op, format!("{v} does not fit the shader's int")))
}

/// Name the dtypes this entry is stamped for and get the named arm's value;
/// any other dtype returns [`Error::DtypeUnsupported`] from the enclosing
/// function.
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

pub(crate) use dtype_dispatch;
