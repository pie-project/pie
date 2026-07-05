//! Shape-typed primitive vocabulary shared across the PTIR layers.
//!
//! A value's type is [`ValueType`] `{ shape, dtype }` where [`Shape`] is a dim
//! list (`rank = shape.rank()`): scalar = `[]`, vector = `[n]`, matrix =
//! `[m, n]`. These leaf types ([`DType`], [`Shape`], [`ValueType`], [`Literal`],
//! [`Predicate`], [`RngKind`]) are what the PTIR op set ([`crate::op`]), the
//! trace container ([`crate::container`]), and the reference interpreter are
//! built from.

/// SSA value id.
pub type ValueId = u32;

/// Maximum tensor rank the IR represents inline. Scalar/vector/matrix need ≤ 2;
/// the headroom covers near-term batched shapes. A `list<u32>` shape lowers to
/// this; lowering rejects rank `> MAX_RANK`.
pub const MAX_RANK: usize = 4;

/// Element type of a value. Tag bytes are stable wire constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    F32 = 0,
    I32 = 1,
    U32 = 2,
    Bool = 3,
}

impl DType {
    pub fn is_float(self) -> bool {
        matches!(self, DType::F32)
    }
    pub fn is_int(self) -> bool {
        matches!(self, DType::I32 | DType::U32)
    }
    pub fn is_numeric(self) -> bool {
        matches!(self, DType::F32 | DType::I32 | DType::U32)
    }
}

/// A logical shape: an ordered list of dimension sizes, `rank = len`.
///
/// Stored inline (fixed capacity [`MAX_RANK`]) so the type stays `Copy`. The
/// **last axis** is the reduce/scan/argmax/pivot axis; a rank-2 `[m, n]` is `m`
/// rows of length `n` (per-row ops iterate axis 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [u32; MAX_RANK],
    rank: u8,
}

impl Shape {
    pub const SCALAR: Shape = Shape { dims: [0; MAX_RANK], rank: 0 };

    /// Build a shape from a dim slice. `None` if `dims.len() > MAX_RANK`.
    pub fn new(dims: &[u32]) -> Option<Shape> {
        if dims.len() > MAX_RANK {
            return None;
        }
        let mut d = [0u32; MAX_RANK];
        d[..dims.len()].copy_from_slice(dims);
        Some(Shape { dims: d, rank: dims.len() as u8 })
    }
    pub fn vector(n: u32) -> Shape {
        Shape::new(&[n]).unwrap()
    }
    pub fn matrix(m: u32, n: u32) -> Shape {
        Shape::new(&[m, n]).unwrap()
    }

    pub fn dims(&self) -> &[u32] {
        &self.dims[..self.rank as usize]
    }
    pub fn rank(&self) -> usize {
        self.rank as usize
    }
    pub fn is_scalar(&self) -> bool {
        self.rank == 0
    }
    pub fn numel(&self) -> u64 {
        self.dims().iter().map(|&d| d as u64).product()
    }
    pub fn last_len(&self) -> Option<u32> {
        self.dims().last().copied()
    }
    pub fn rows(&self) -> u32 {
        match self.rank as usize {
            0 | 1 => 1,
            r => self.dims[..r - 1].iter().product(),
        }
    }
    /// The shape with the last axis dropped (a reduction's result), or `None`
    /// for a scalar.
    pub fn drop_last(&self) -> Option<Shape> {
        if self.rank == 0 {
            return None;
        }
        Shape::new(&self.dims[..self.rank as usize - 1])
    }
}

/// A value's full type: [`Shape`] + [`DType`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueType {
    pub shape: Shape,
    pub dtype: DType,
}

impl ValueType {
    pub const fn new(shape: Shape, dtype: DType) -> Self {
        Self { shape, dtype }
    }
    pub fn scalar(dtype: DType) -> Self {
        Self { shape: Shape::SCALAR, dtype }
    }
    pub fn vector(n: u32, dtype: DType) -> Self {
        Self { shape: Shape::vector(n), dtype }
    }
}

/// Threshold predicate for the sort-free top-k / top-p / min-p pivot op. Each
/// variant carries the value id of its (host-supplied, de-hardwired) threshold,
/// so the program bytecode is threshold-invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predicate {
    /// top-k: keep the top `k` — a value id (a `U32` scalar, or a per-row
    /// `[rows]` `U32` vector for a matrix input).
    RankLe(ValueId),
    /// top-p: inclusive nucleus to mass `p` (a Scalar-F32 value id).
    CummassLe(ValueId),
    /// min-p: keep `>= thr` (a Scalar-F32 value id, e.g. `p·max_prob`).
    ProbGe(ValueId),
}

/// Distribution sampled by the noise op. Tag bytes are stable wire constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RngKind {
    Uniform = 0,
    Gumbel = 1,
}

/// A compile-time constant scalar (the payload of a `const` op).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Literal {
    F32(f32),
    I32(i32),
    U32(u32),
    Bool(bool),
}

impl Literal {
    pub fn dtype(self) -> DType {
        match self {
            Literal::F32(_) => DType::F32,
            Literal::I32(_) => DType::I32,
            Literal::U32(_) => DType::U32,
            Literal::Bool(_) => DType::Bool,
        }
    }
}
