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

/// What arithmetic a dtype admits. Part of a dtype's declaration rather than a
/// predicate written after the fact: a new float type that nobody remembered to
/// add to `is_float` reads as "not a float" everywhere, silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DTypeClass {
    /// Approximate reals; comparisons and reductions follow IEEE-754.
    Float,
    /// Exact integers; division truncates.
    Int,
    /// Booleans: the only class the compare and logic ops produce.
    Logical,
}

/// Declares the scalar dtypes once and derives everything spelled per-dtype.
///
/// The wire byte, the lowercase name, and the arithmetic class appear on
/// exactly one line each, and [`DType::ALL`], [`DType::name`],
/// [`DType::from_wire`] and the `is_*` predicates all come from that line. The
/// alternative is what this replaced: four hand-kept lists, of which only
/// `name` was a `match` the compiler could check.
macro_rules! declare_dtypes {
    ($($variant:ident = $wire:literal, $name:literal, $class:ident;)*) => {
        /// Element type of a value. Tag bytes are stable wire constants.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[repr(u8)]
        pub enum DType {
            $(
                #[doc = concat!("The `", $name, "` element type.")]
                $variant = $wire,
            )*
        }

        impl DType {
            /// Every scalar dtype, in wire-tag order. See
            /// [`crate::registry::Stage::ALL`]. Channel decls may additionally
            /// carry the late-bound [`crate::container::ChanDType::Act`] tag,
            /// which is not a `DType`.
            pub const ALL: &'static [DType] = &[$(DType::$variant,)*];

            /// Lowercase wire name, used by the generated C header and diagnostics.
            pub fn name(self) -> &'static str {
                match self {
                    $(DType::$variant => $name,)*
                }
            }

            /// What arithmetic this dtype admits.
            pub fn class(self) -> DTypeClass {
                match self {
                    $(DType::$variant => DTypeClass::$class,)*
                }
            }
        }

        #[cfg(test)]
        mod dtype_tests {
            use super::*;

            /// `ALL` is indexed by wire byte, which is what makes
            /// [`DType::from_wire`] a lookup and what every `dtype > Bool as
            /// u8` bound in the decoders relies on. Listing a dtype out of
            /// wire order, or leaving a gap, silently breaks both.
            #[test]
            fn all_is_indexed_by_wire_byte() {
                let mut wire = 0u8;
                $(
                    assert_eq!(
                        DType::$variant as u8, wire,
                        "declare_dtypes! must list dtypes in wire order with no gaps"
                    );
                    assert_eq!(DType::ALL[usize::from(wire)], DType::$variant);
                    wire += 1;
                )*
                assert_eq!(DType::ALL.len(), usize::from(wire));
                assert!(DType::from_wire(wire).is_none());
                assert!(wire > 0, "declare_dtypes! declared nothing");
            }

            /// `is_numeric` is the one predicate that is not a restatement of
            /// its dtype's row.
            ///
            /// `is_float` and `is_int` compare `class()` against the class the
            /// row declares, so they cannot drift. `is_numeric` names *two of
            /// three* classes, and nothing about adding a fourth would edit it
            /// — a `Complex` class would silently read as non-numeric in every
            /// arithmetic rule in `infer` and `validate`. The walk forces the
            /// decision the way `Backend::ALL` is forced: a new class is a
            /// compile error here, and answering it means saying whether
            /// `is_numeric` covers it.
            #[test]
            fn a_new_dtype_class_has_to_answer_to_is_numeric() {
                fn numeric(class: DTypeClass) -> bool {
                    match class {
                        DTypeClass::Float | DTypeClass::Int => true,
                        DTypeClass::Logical => false,
                    }
                }
                let mut seen = 0usize;
                for dtype in DType::ALL {
                    assert_eq!(
                        dtype.is_numeric(),
                        numeric(dtype.class()),
                        "{} is {:?}",
                        dtype.name(),
                        dtype.class()
                    );
                    assert_eq!(dtype.is_float(), dtype.class() == DTypeClass::Float);
                    assert_eq!(dtype.is_int(), dtype.class() == DTypeClass::Int);
                    assert!(!dtype.name().is_empty());
                    seen += 1;
                }
                assert_eq!(seen, DType::ALL.len());
                let names: alloc::collections::BTreeSet<&str> =
                    DType::ALL.iter().map(|d| d.name()).collect();
                assert_eq!(names.len(), DType::ALL.len(), "two dtypes share a name");
            }
        }
    };
}

declare_dtypes! {
    F32 = 0, "f32", Float;
    I32 = 1, "i32", Int;
    U32 = 2, "u32", Int;
    Bool = 3, "bool", Logical;
}

impl DType {
    /// The dtype a wire byte names, or `None` if the byte names none.
    pub fn from_wire(byte: u8) -> Option<DType> {
        DType::ALL.get(usize::from(byte)).copied()
    }

    /// Whether this dtype is a [`DTypeClass::Float`].
    pub fn is_float(self) -> bool {
        self.class() == DTypeClass::Float
    }
    /// Whether this dtype is a [`DTypeClass::Int`].
    pub fn is_int(self) -> bool {
        self.class() == DTypeClass::Int
    }
    /// Whether this dtype admits arithmetic — float or int, but not
    /// [`DTypeClass::Logical`].
    pub fn is_numeric(self) -> bool {
        matches!(self.class(), DTypeClass::Float | DTypeClass::Int)
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
    /// product does not fit, so every constructible shape has one.
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
    /// `u64` rather than `u32`, matching [`Shape::numel`], because the only
    /// bound [`Shape::new`] enforces is that the *whole* dim product fits
    /// `u64`. A leading-dim product is smaller than that but still free to
    /// exceed `u32`: `[65536, 65536, 2]` is a shape `new` accepts, and its row
    /// count is `2^32`. Computing it in `u32` makes decoded — that is,
    /// untrusted — input panic in debug and silently answer `0` in release,
    /// which is a row loop that runs zero times rather than a rejection.
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

/// A value's full type: [`Shape`] + [`DType`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueType {
    /// The value's extents.
    pub shape: Shape,
    /// The value's element type.
    pub dtype: DType,
}

impl ValueType {
    /// The type with the given shape and element type.
    pub const fn new(shape: Shape, dtype: DType) -> Self {
        Self { shape, dtype }
    }
    /// The rank-0 type of element type `dtype`.
    pub fn scalar(dtype: DType) -> Self {
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
    pub fn vector(n: u32, dtype: DType) -> Self {
        Self {
            shape: Shape::vector(n),
            dtype,
        }
    }
}

/// Threshold predicate for the sort-free top-k / top-p / min-p pivot op. Each
/// variant carries the value id of its (host-supplied, de-hardwired) threshold,
/// so the program bytecode is threshold-invariant.
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
    ///
    /// Every variant carries exactly one, which is what lets
    /// [`Op::operands`](crate::op::Op::operands) treat `pivot_threshold` as a
    /// two-operand op without
    /// re-matching the predicate at each call site.
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
    /// An [`DType::F32`] constant.
    F32(f32),
    /// An [`DType::I32`] constant.
    I32(i32),
    /// A [`DType::U32`] constant.
    U32(u32),
    /// A [`DType::Bool`] constant.
    Bool(bool),
}

impl Literal {
    /// This literal's element type.
    pub fn dtype(self) -> DType {
        match self {
            Literal::F32(_) => DType::F32,
            Literal::I32(_) => DType::I32,
            Literal::U32(_) => DType::U32,
            Literal::Bool(_) => DType::Bool,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `rows` is the product of the leading axes, and [`Shape::new`] only
    /// bounds the product of *all* of them. A shape whose full product fits
    /// `u64` can still have more than `u32::MAX` rows, so the row count is
    /// computed in `u64` — narrower arithmetic here panics in debug and wraps
    /// to `0` in release on input the decoder accepts.
    #[test]
    fn rows_does_not_overflow_for_a_shape_new_accepts() {
        let shape = Shape::new(&[65_536, 65_536, 2]).expect("the u64 dim product fits");
        assert_eq!(shape.numel(), 8_589_934_592);
        assert_eq!(shape.rows(), 4_294_967_296);
        assert!(shape.rows() > u64::from(u32::MAX));
    }

    /// The identity the row count exists for, at a size where a `u32` product
    /// would have wrapped.
    #[test]
    fn rows_times_last_len_is_numel() {
        for dims in [
            &[7u32][..],
            &[3, 5],
            &[2, 3, 4],
            &[65_536, 65_536, 2],
            &[1 << 20, 1 << 20, 3],
        ] {
            let shape = Shape::new(dims).expect("dim product fits u64");
            let last = u64::from(shape.last_len().expect("non-scalar"));
            assert_eq!(shape.rows() * last, shape.numel(), "dims {dims:?}");
        }
    }
}
