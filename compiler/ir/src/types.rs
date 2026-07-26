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
    Float,
    Int,
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
        #[repr(u8)]
        pub enum DType {
            $($variant = $wire,)*
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
            /// [`DType::from_wire`] a lookup and what every `dtype > Bool as u8`
            /// bound in the decoders used to assume without saying.
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

    pub fn is_float(self) -> bool {
        self.class() == DTypeClass::Float
    }
    pub fn is_int(self) -> bool {
        self.class() == DTypeClass::Int
    }
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
            rank: dims.len() as u8,
        })
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
        Self {
            shape: Shape::SCALAR,
            dtype,
        }
    }
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
