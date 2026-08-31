//! Shape-typed primitive vocabulary shared across the ETA layers.
//!
//! A value's type is [`ValueType`] `{ shape, dtype }` where [`Shape`] is a dim
//! list (`rank = shape.rank()`): scalar = `[]`, vector = `[n]`, matrix =
//! `[m, n]`. These leaf types ([`Dtype`], [`Shape`], [`ValueType`], [`Literal`],
//! [`Predicate`], [`RngKind`]) are what the ETA op set ([`crate::op`]), the
//! trace container ([`crate::container`]), and the reference interpreter are
//! built from.
//!
//! [`Dtype`] is the odd one out: it is not this crate's, it is the tree's
//! ([`dtype::Dtype`]), and it names twelve storage formats ETA has no
//! arithmetic for alongside the four it computes in. [`class_of`] is where
//! that difference is decided, once.

/// SSA value id.
pub type ValueId = u32;

/// Maximum tensor rank the IR represents inline. Scalar/vector/matrix need ≤ 2;
/// the headroom covers near-term batched shapes. A `list<u32>` shape lowers to
/// this; lowering rejects rank `> MAX_RANK`.
pub const MAX_RANK: usize = 4;

/// The element type of an ETA value — [`dtype::Dtype`], the one enum the
/// loader, the kernels, the transfer contract and this IR all name.
///
/// ETA computes in **four** of its seventeen variants. Which four is
/// [`class_of`]'s answer and nothing else's; see that function for why the
/// question has exactly one place to be asked.
pub use dtype::Dtype;

/// What arithmetic a dtype admits. Part of a dtype's declaration rather than a
/// predicate written after the fact: a new float type that nobody remembered to
/// add to `is_float` reads as "not a float" everywhere, silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DtypeClass {
    /// Approximate reals; comparisons and reductions follow IEEE-754.
    Float,
    /// Exact integers; division truncates.
    Int,
    /// Booleans: the only class the compare and logic ops produce.
    Logical,
}
/// The dtypes ETA computes in, in wire-tag order.
///
/// Indexed by wire byte — that is what makes [`from_wire`] a lookup. Channel
/// decls may additionally carry the late-bound
/// [`crate::container::ChanDType::Act`] tag, which is not a `Dtype`.
pub const WIRE_ORDER: &[Dtype] = &[Dtype::F32, Dtype::I32, Dtype::U32, Dtype::Bool];

/// **The gate.** What arithmetic `d` admits in ETA, or `None` for a dtype ETA
/// does not compute in.
///
/// This is the only wildcard-free match over [`Dtype`] in the tree, and
/// everything else about ETA's dtype set is derived from it: [`supports`],
/// [`is_float`], [`is_int`], [`is_numeric`]. **Keep it that way.**
///
/// # Why the second arm lists thirteen names instead of saying `_`
///
/// [`Dtype`] is not this crate's. It is a leaf sixteen other consumers depend
/// on, and it grows for reasons that have nothing to do with ETA — a
/// checkpoint gains a quant format, and the enum gains a variant. That growth
/// has to land somewhere a human decides.
///
/// With the names written out it lands here, and nowhere else: adding `Fp6` to
/// `dtype::Dtype` fails to compile against this one match, and the author's
/// only way forward is to write `Fp6` into one arm or the other — which is the
/// question "does ETA compute in it", asked at the moment somebody can answer.
///
/// `_ => None` compiles today and compiles forever. It would move that
/// question from the build to a test nobody runs, and the next quant format
/// would join ETA's unsupported set without anybody having said so.
pub const fn class_of(d: Dtype) -> Option<DtypeClass> {
    match d {
        Dtype::F32 => Some(DtypeClass::Float),
        Dtype::I32 => Some(DtypeClass::Int),
        Dtype::U32 => Some(DtypeClass::Int),
        Dtype::Bool => Some(DtypeClass::Logical),

        // Everything else `dtype::Dtype` names. ETA's op set has no arithmetic
        // for these: the interpreter's numerics are f32/i32/u32/bool, the
        // container's literal payload is four bytes, and a traced value
        // materializes as one of the four above whatever a backend stores
        // underneath (see `ModelProfile::activation`). Moving a name up is a
        // change to the op set, the interpreter and the wire format at once.
        Dtype::F16
        | Dtype::Bf16
        | Dtype::Fp8E4m3
        | Dtype::Fp8E5m2
        | Dtype::Fp4
        | Dtype::Mxfp4
        | Dtype::MlxU4
        | Dtype::MlxU8
        | Dtype::MlxU4G32
        | Dtype::E8m0
        | Dtype::I64
        | Dtype::I16
        | Dtype::I8
        | Dtype::U64
        | Dtype::U16
        | Dtype::U8 => None,
    }
}

/// Lowercase wire name, used by the generated C header and diagnostics. `None`
/// for a dtype outside ETA's set — see [`class_of`].
///
/// The `_` arm is safe here in a way it would not be in [`class_of`]: a variant
/// added to [`Dtype`] cannot reach this function without first having failed to
/// compile against the gate, where somebody answered for it.
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
/// outside ETA's set — see [`class_of`].
///
/// The numbering lives here and not on [`Dtype`]. `Dtype` carries no `#[repr]`
/// and no explicit discriminants on purpose: it is a leaf sixteen consumers
/// depend on, and freezing a numbering into it for ETA's container format would
/// make the leaf carry one plane's wire format.
///
/// **This function is why `dtype as u8` is banned in this tree.** The enum used
/// to be `#[repr(u8)]` with the wire byte as its discriminant, and nine sites
/// cast to get the byte for free. A field-less enum casts whether or not it has
/// a `repr`, so every one of them still compiled after the merge — and answered
/// `dtype::Dtype`'s declaration order instead (`I32` → 10, not 1). Ask here.
///
/// The `_` arm is safe for the reason given on [`name`]: the gate is upstream.
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

    /// The four tables above are four hand-kept lists, and this is what keeps
    /// them one list.
    ///
    /// A macro generated them from single rows for a while, so they agreed by
    /// construction and this test only had to check them against the *format*.
    /// Written out, they can disagree — a fifth dtype added to [`WIRE_ORDER`]
    /// and forgotten in [`name`] is a compile-clean mistake. So the walk below
    /// is load-bearing rather than decorative, and it is the reason the wire
    /// bytes are also spelled out as literals in
    /// [`the_wire_bytes_are_the_ones_the_format_froze`]: a table that agrees
    /// with itself still has to agree with what the format froze.
    ///
    /// What did NOT move to a test is the thing that matters most: [`class_of`]
    /// is still wildcard-free, so `dtype::Dtype` growing a variant is still a
    /// build failure and not a test failure.
    #[test]
    fn wire_order_is_indexed_by_wire_byte() {
        for (wire, &d) in WIRE_ORDER.iter().enumerate() {
            let wire = u8::try_from(wire).expect("ETA declares fewer than 256 dtypes");
            assert_eq!(
                to_wire(d),
                Some(wire),
                "{d:?} is at index {wire} of WIRE_ORDER but does not answer that byte"
            );
            assert_eq!(from_wire(wire), Some(d));
            assert!(class_of(d).is_some(), "{d:?} is declared but has no class");
            assert!(
                name(d).is_some_and(|n| !n.is_empty()),
                "{d:?} is declared but has no name"
            );
        }
        let past_the_end = u8::try_from(WIRE_ORDER.len()).expect("fewer than 256");
        assert!(from_wire(past_the_end).is_none());
        assert!(!WIRE_ORDER.is_empty(), "ETA declares no dtype at all");
    }

    /// The wire bytes themselves, spelled out. A container written before the
    /// two dtype enums merged decodes against these four numbers, and nothing
    /// derived from the tables above can check that — only an assertion can.
    #[test]
    fn the_wire_bytes_are_the_ones_the_format_froze() {
        assert_eq!(to_wire(Dtype::F32), Some(0));
        assert_eq!(to_wire(Dtype::I32), Some(1));
        assert_eq!(to_wire(Dtype::U32), Some(2));
        assert_eq!(to_wire(Dtype::Bool), Some(3));
        assert_eq!(WIRE_ORDER.len(), 4);
    }

    /// Every dtype ETA supports is one ETA declares, in the other direction:
    /// nothing answers a class or a name without holding a place in
    /// [`WIRE_ORDER`].
    #[test]
    fn nothing_supported_is_missing_from_wire_order() {
        for d in [
            Dtype::F32,
            Dtype::F16,
            Dtype::Bf16,
            Dtype::Fp8E4m3,
            Dtype::Fp8E5m2,
            Dtype::Fp4,
            Dtype::Mxfp4,
            Dtype::MlxU4,
            Dtype::E8m0,
            Dtype::I64,
            Dtype::I32,
            Dtype::I16,
            Dtype::I8,
            Dtype::U64,
            Dtype::U32,
            Dtype::U16,
            Dtype::U8,
            Dtype::Bool,
        ] {
            assert_eq!(
                supports(d),
                WIRE_ORDER.contains(&d),
                "{d:?} disagrees between `class_of` and `WIRE_ORDER`"
            );
        }
    }

    /// A dtype outside ETA's set answers `None` to every question, rather than
    /// answering `F32` to one of them.
    #[test]
    fn a_dtype_eta_does_not_compute_in_answers_nothing() {
        for d in [Dtype::Bf16, Dtype::F16, Dtype::I8, Dtype::Mxfp4, Dtype::U64] {
            assert_eq!(class_of(d), None);
            assert!(!supports(d));
            assert_eq!(name(d), None);
            assert_eq!(to_wire(d), None);
            assert!(!is_float(d) && !is_int(d) && !is_numeric(d));
        }
    }

    /// `is_numeric` is the one predicate that is not a restatement of its
    /// dtype's class.
    ///
    /// `is_float` and `is_int` compare [`class_of`] against a class, so they
    /// cannot drift. `is_numeric` names *two of three* classes, and nothing
    /// about adding a fourth would edit it — a `Complex` class would silently
    /// read as non-numeric in every arithmetic rule in `infer` and `validate`.
    /// The local `numeric` below is wildcard-free for that reason: a new class
    /// is a compile error here, and answering it means saying whether
    /// `is_numeric` covers it.
    #[test]
    fn a_new_dtype_class_has_to_answer_to_is_numeric() {
        const fn numeric(class: DtypeClass) -> bool {
            match class {
                DtypeClass::Float | DtypeClass::Int => true,
                DtypeClass::Logical => false,
            }
        }
        for &d in WIRE_ORDER {
            let class = class_of(d).expect("a declared dtype has a class");
            assert_eq!(is_numeric(d), numeric(class), "{:?} is {class:?}", name(d));
            assert_eq!(is_float(d), class == DtypeClass::Float);
            assert_eq!(is_int(d), class == DtypeClass::Int);
        }
        let names: alloc::collections::BTreeSet<&str> =
            WIRE_ORDER.iter().filter_map(|&d| name(d)).collect();
        assert_eq!(names.len(), WIRE_ORDER.len(), "two dtypes share a name");
    }
}

/// The dtype a wire byte names, or `None` if the byte names none.
///
/// A lookup, not a match: [`WIRE_ORDER`] is indexed by wire byte. Inverse of
/// [`to_wire`] over ETA's set.
pub const fn from_wire(byte: u8) -> Option<Dtype> {
    let index = byte as usize;
    if index < WIRE_ORDER.len() {
        Some(WIRE_ORDER[index])
    } else {
        None
    }
}

/// Whether ETA computes in `d` at all. [`class_of`] is the only place that
/// answers; this reads its answer.
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
///
/// Derived from [`name`] and adding no match of its own — the gate is still
/// the only place ETA's set is decided. The stand-in reads as the outsider it
/// is rather than as a dtype, because a diagnostic that prints `f32` for a
/// `Bf16` is worse than no diagnostic.
pub const fn name_or_unknown(d: Dtype) -> &'static str {
    match name(d) {
        Some(n) => n,
        None => "<not an eta dtype>",
    }
}

/// The wire byte for a dtype that is about to be written into an encoding.
///
/// [`to_wire`]'s answer where the caller is an encoder and has no `None` to
/// return: the trace container, the compile-cache key, the descriptor word a
/// kernel reads. Every one of them once wrote `dtype as u8` and got the right
/// byte for free from a `#[repr(u8)]` four-variant enum. That cast still
/// compiles on [`Dtype`] — a field-less enum casts to an integer whether or
/// not it has a `repr` — and it now yields the leaf's *declaration* order,
/// which is not this numbering. So the cast is gone from the tree and this is
/// what replaced it.
///
/// # Panics
///
/// If `d` is a dtype ETA does not compute in. Encoding is the last place that
/// can notice, and there is nothing for it to write: the container format has
/// four dtype tags and no room for a fifth. Everything that reaches here has
/// passed either [`crate::container::decode`] (which only ever produces the four) or
/// [`crate::infer::body_types`] (which rejects the rest by name), so the panic
/// is for a container assembled in-process out of a dtype ETA has no
/// arithmetic for — a `Dtype::Bf16` written into a `ChannelDecl` by hand.
pub fn wire_dtype(d: Dtype) -> u8 {
    match to_wire(d) {
        Some(byte) => byte,
        None => panic!("dtype {d:?} is not one ETA computes in; it has no wire tag"),
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
