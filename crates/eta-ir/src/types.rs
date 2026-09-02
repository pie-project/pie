//! Shape-typed primitive vocabulary shared across the ETA layers.

/// SSA value id.
pub type ValueId = u32;

/// Maximum tensor rank the IR represents inline; lowering rejects rank > MAX_RANK.
pub const MAX_RANK: usize = 4;

/// The element type of an ETA value, re-exported from `dtype::Dtype`.
///
/// ETA computes in four of its variants; see [`class_of`].
pub use dtype::Dtype;

/// What arithmetic a dtype admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DtypeClass {
    /// Approximate reals; comparisons and reductions follow IEEE-754.
    Float,
    /// Exact integers; division truncates.
    Int,
    /// Booleans: the only class the compare and logic ops produce.
    Logical,
}
/// The dtypes ETA computes in, in wire-tag order (indexed by wire byte).
pub const WIRE_ORDER: &[Dtype] = &[Dtype::F32, Dtype::I32, Dtype::U32, Dtype::Bool];

/// What arithmetic `d` admits in ETA, or `None` for a dtype ETA does not
/// compute in. [`supports`], [`is_float`], [`is_int`], [`is_numeric`] derive from it.
pub const fn class_of(d: Dtype) -> Option<DtypeClass> {
    match d {
        Dtype::F32 => Some(DtypeClass::Float),
        Dtype::I32 => Some(DtypeClass::Int),
        Dtype::U32 => Some(DtypeClass::Int),
        Dtype::Bool => Some(DtypeClass::Logical),

        // The rest of `dtype::Dtype`: ETA's op set has no arithmetic for
        // these (interpreter numerics are f32/i32/u32/bool).
        Dtype::F16
        | Dtype::Bf16
        | Dtype::E4m3
        | Dtype::E5m2
        | Dtype::E2m1
        | Dtype::Mxfp4
        | Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U4g64tiled
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128
        | Dtype::E8m0
        | Dtype::I64
        | Dtype::I16
        | Dtype::I8
        | Dtype::U64
        | Dtype::U16
        | Dtype::U8
        // Composite (quantized) formats: never materialize as traced values.
        | Dtype::Nvfp4
        | Dtype::U2g16k
        | Dtype::I3g16k
        | Dtype::U4g32k
        | Dtype::U5g32k
        | Dtype::I6g16k
        | Dtype::E4m3row
        | Dtype::E4m3tile128 => None,
    }
}

/// Lowercase wire name, used by the generated C header and diagnostics. `None`
/// for a dtype outside ETA's set — see [`class_of`].
pub const fn name(d: Dtype) -> Option<&'static str> {
    match d {
        Dtype::F32 => Some("f32"),
        Dtype::I32 => Some("i32"),
        Dtype::U32 => Some("u32"),
        Dtype::Bool => Some("bool"),
        _ => None,
    }
}

/// The wire byte naming `d` in the trace container, or `None` for a dtype
/// outside ETA's set. Not `dtype as u8`: `Dtype` has no `#[repr]`.
pub const fn to_wire(d: Dtype) -> Option<u8> {
    match d {
        Dtype::F32 => Some(0),
        Dtype::I32 => Some(1),
        Dtype::U32 => Some(2),
        Dtype::Bool => Some(3),
        _ => None,
    }
}

#[cfg(test)]
mod dtype_tests {
    use super::*;

    /// Pins the wire bytes themselves against the frozen container format.
    #[test]
    fn the_wire_bytes_are_the_ones_the_format_froze() {
        assert_eq!(to_wire(Dtype::F32), Some(0));
        assert_eq!(to_wire(Dtype::I32), Some(1));
        assert_eq!(to_wire(Dtype::U32), Some(2));
        assert_eq!(to_wire(Dtype::Bool), Some(3));
        assert_eq!(WIRE_ORDER.len(), 4);
    }

}

/// The dtype a wire byte names, or `None` if the byte names none.
pub const fn from_wire(byte: u8) -> Option<Dtype> {
    let index = byte as usize;
    if index < WIRE_ORDER.len() {
        Some(WIRE_ORDER[index])
    } else {
        None
    }
}

/// Whether ETA computes in `d` at all.
pub const fn supports(d: Dtype) -> bool {
    class_of(d).is_some()
}

/// Whether `d` is a [`DtypeClass::Float`].
pub const fn is_float(d: Dtype) -> bool {
    matches!(class_of(d), Some(DtypeClass::Float))
}
/// Whether `d` is a [`DtypeClass::Int`].
pub const fn is_int(d: Dtype) -> bool {
    matches!(class_of(d), Some(DtypeClass::Int))
}
/// Whether `d` admits arithmetic — float or int, but not
/// [`DtypeClass::Logical`], and not a dtype ETA does not compute in.
pub const fn is_numeric(d: Dtype) -> bool {
    matches!(class_of(d), Some(DtypeClass::Float | DtypeClass::Int))
}

/// [`name`] for a diagnostic, which has to print something either way.
pub const fn name_or_unknown(d: Dtype) -> &'static str {
    match name(d) {
        Some(n) => n,
        None => "<not an eta dtype>",
    }
}

/// The wire byte for a dtype that is about to be written into an encoding.
///
/// # Panics
///
/// If `d` is a dtype ETA does not compute in.
pub fn wire_dtype(d: Dtype) -> u8 {
    match to_wire(d) {
        Some(byte) => byte,
        None => panic!("dtype {d:?} is not one ETA computes in; it has no wire tag"),
    }
}

/// A logical shape: an ordered list of dimension sizes, `rank = len`. Stored
/// inline (fixed capacity [`MAX_RANK`]) so the type stays `Copy`. Last axis
/// is the reduce/scan/argmax/pivot axis; rank-2 `[m, n]` is `m` rows of `n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [u32; MAX_RANK],
    rank: u8,
}

impl Shape {
    /// The rank-0 shape: one element, no axes.
    pub const SCALAR: Shape = Shape {
        dims: [0; MAX_RANK],
        rank: 0,
    };

    /// Build a shape from a dim slice. `None` if `dims.len() > MAX_RANK`.
    pub fn new(dims: &[u32]) -> Option<Shape> {
        if dims.len() > MAX_RANK
            || dims.contains(&0)
            || dims
                .iter()
                .try_fold(1u64, |product, &dim| product.checked_mul(dim as u64))
                .is_none()
        {
            return None;
        }
        let mut d = [0u32; MAX_RANK];
        d[..dims.len()].copy_from_slice(dims);
        Some(Shape {
            dims: d,
            rank: u8::try_from(dims.len()).ok()?,
        })
    }
    /// The rank-1 shape `[n]`.
    ///
    /// # Panics
    ///
    /// If `n` is `0`; every extent must be at least 1. Use [`Shape::new`]
    /// for a length that came from outside the program.
    pub fn vector(n: u32) -> Shape {
        Shape::new(&[n]).unwrap()
    }
    /// The rank-2 shape `[m, n]`.
    ///
    /// # Panics
    ///
    /// If either extent is `0`, or if `m * n` overflows `u64`. Use
    /// [`Shape::new`] for extents that came from outside the program.
    pub fn matrix(m: u32, n: u32) -> Shape {
        Shape::new(&[m, n]).unwrap()
    }

    /// The extents, outermost first. Length is the rank.
    pub fn dims(&self) -> &[u32] {
        &self.dims[..self.rank as usize]
    }
    /// How many axes this shape has.
    pub fn rank(&self) -> usize {
        self.rank as usize
    }
    /// Whether this is the rank-0 shape.
    pub fn is_scalar(&self) -> bool {
        self.rank == 0
    }
    /// Total element count.
    ///
    /// Cannot overflow: [`Shape::new`] refuses a dim list whose `u64`
    /// product does not fit.
    pub fn numel(&self) -> u64 {
        self.dims().iter().map(|&d| d as u64).product()
    }
    /// The trailing extent — the axis reductions, scans and pivots run
    /// along — or `None` for a scalar.
    pub fn last_len(&self) -> Option<u32> {
        self.dims().last().copied()
    }
    /// The number of rows a rank-`n` shape has: the product of every axis but
    /// the last. Scalars and vectors are one row.
    ///
    /// `u64`, not `u32`: a leading-dim product can exceed `u32::MAX` even
    /// though the full dim product fits `u64` (e.g. `[65536, 65536, 2]`).
    pub fn rows(&self) -> u64 {
        match self.rank as usize {
            0 | 1 => 1,
            r => self.dims[..r - 1].iter().map(|&d| d as u64).product(),
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

/// A value's full type: [`Shape`] + [`Dtype`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueType {
    /// The value's extents.
    pub shape: Shape,
    /// The value's element type.
    pub dtype: Dtype,
}

impl ValueType {
    /// The type with the given shape and element type.
    pub const fn new(shape: Shape, dtype: Dtype) -> Self {
        Self { shape, dtype }
    }
    /// The rank-0 type of element type `dtype`.
    pub fn scalar(dtype: Dtype) -> Self {
        Self {
            shape: Shape::SCALAR,
            dtype,
        }
    }
    /// The `[n]` type of element type `dtype`.
    ///
    /// # Panics
    ///
    /// If `n` is `0`; see [`Shape::vector`].
    pub fn vector(n: u32, dtype: Dtype) -> Self {
        Self {
            shape: Shape::vector(n),
            dtype,
        }
    }
}

/// Threshold predicate for the sort-free top-k / top-p / min-p pivot op. Each
/// variant carries the value id of its (host-supplied) threshold, so the
/// program bytecode is threshold-invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Predicate {
    /// top-k: keep the top `k` — a value id (a `U32` scalar, or a per-row
    /// `[rows]` `U32` vector for a matrix input).
    RankLe(ValueId),
    /// top-p: inclusive nucleus to mass `p` (a Scalar-F32 value id).
    CummassLe(ValueId),
    /// min-p: keep `>= thr` (a Scalar-F32 value id, e.g. `p·max_prob`).
    ProbGe(ValueId),
}

impl Predicate {
    /// The value id this predicate thresholds on.
    pub fn value(self) -> ValueId {
        match self {
            Predicate::RankLe(value) | Predicate::CummassLe(value) | Predicate::ProbGe(value) => {
                value
            }
        }
    }

    /// The threshold value id, for rewriting it. See [`Predicate::value`].
    pub fn value_slot(&mut self) -> &mut ValueId {
        match self {
            Predicate::RankLe(value) | Predicate::CummassLe(value) | Predicate::ProbGe(value) => {
                value
            }
        }
    }
}

/// Distribution sampled by the noise op. Tag bytes are stable wire constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum RngKind {
    /// Draws in `(0, 1)`, exclusive at both ends.
    Uniform = 0,
    /// Standard Gumbel noise, for argmax-based categorical sampling.
    Gumbel = 1,
}

/// A compile-time constant scalar (the payload of a `const` op).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Literal {
    /// An [`Dtype::F32`] constant.
    F32(f32),
    /// An [`Dtype::I32`] constant.
    I32(i32),
    /// A [`Dtype::U32`] constant.
    U32(u32),
    /// A [`Dtype::Bool`] constant.
    Bool(bool),
}

impl Literal {
    /// This literal's element type.
    pub fn dtype(self) -> Dtype {
        match self {
            Literal::F32(_) => Dtype::F32,
            Literal::I32(_) => Dtype::I32,
            Literal::U32(_) => Dtype::U32,
            Literal::Bool(_) => Dtype::Bool,
        }
    }
}

