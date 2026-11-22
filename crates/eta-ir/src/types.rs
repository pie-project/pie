pub type ValueId = u32;

pub const MAX_RANK: usize = 4;

pub use dtype::Dtype;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DtypeClass {
    Float,
    Int,
    Logical,
}

// The dtypes ETA computes in, and how a guest program stores them.
//
// `f32`, `i32`, `u32` take a 32-bit word an element; `bool` a byte. `bf16`
// is a storage dtype for vocab-sized intermediates: two bytes an element
// (the upper half of an f32's bits), a value's elements packed two to a
// little-endian word, element `2w` in the low half of word `w`, `2w + 1` in
// the high half. A value starts on a word (every layout aligns a value past
// a word), so a word never holds two values' elements; the high half of an
// odd-length value's last word is padding that belongs to the value.
//
// The arithmetic rules for `bf16` (every backend implements exactly these):
//
// - Compute is f32. An op loads each `bf16` operand exactly (its bits
//   shifted up 16), computes as it would in f32, and rounds its result to
//   `bf16` on the store, round-to-nearest-even; a NaN stays a NaN (quiet).
// - The float operands of an elementwise op (map, compare, select) may mix
//   `f32` and `bf16`; the result is `bf16` when any float operand is
//   `bf16`, else `f32`. There is no `bf16` literal: a constant is an `f32`
//   the op mixes in, never rounded first. A scatter's values may be the
//   other float dtype than its base; the result is the base's dtype (a
//   `bf16` scatter-add sums in f32 and rounds once).
// - A reduction over the row (`reduce_sum` / `reduce_max` / `reduce_min`,
//   `matmul`'s inner products) accumulates in f32 and returns `f32`: its
//   result is the accumulator, never rounded to `bf16`. `reduce_argmax`
//   returns `i32` as for any input.
// - A scan (`cumsum` / `cumprod`) accumulates in f32 and returns the
//   input's dtype: each prefix is the f32 running value rounded once on its
//   store, never re-read rounded.
// - An order op (`sort_desc`, `top_k`) returns its values in the input's
//   dtype (each a copy of an input element, exact) and compares them as
//   f32; `pivot_threshold` reads a `bf16` row as f32 and accumulates its
//   mass in f32; its threshold operand is `f32` or `bf16`.
// - `cast` converts in and out: to `bf16` it rounds (as above), from `bf16`
//   it widens exactly to f32 first, then converts as from f32.
// - A sampling draw is never a bf16 row. `gumbel_argmax { logits, keep,
//   scale, state }` reduces each row to the index of its largest
//   `logits[i] * scale + gumbel(state, i)` (`-inf` where `keep` is
//   false), every lane computed in f32 from its operands as loaded and
//   never stored: rounding the perturbed lane to bf16 would collapse near
//   ties onto the lowest index and bias the draw. The noise is lane `i` of
//   `rng_keyed { state, shape: logits' shape, kind: Gumbel }`, generated
//   in the reduction (no noise row), so over f32 logits the draw is
//   bit-identical to `reduce_argmax(select(keep, logits * scale, -inf) +
//   rng_keyed(..))`. Ties and NaN go as `reduce_argmax`'s: the lowest index
//   of the largest non-NaN lane. A temperature is applied as one form, a
//   multiply by its reciprocal: `scale` is `1 / T` rounded once to f32 where
//   the temperature is known (the host, for a sampler's constant), and a
//   lane multiplies by it, a correctly rounded op on every backend, where
//   a divide is not (WGSL's and fast-math's are approximate).
// - Values that are ids, positions, indices, masks or rng state never take
//   `bf16`; `rng` returns `f32`. A model output (`output`) is the dtype its
//   declaration states.
//
// Two invocations never store the two halves of one word: a backend whose
// stores are words (WGSL) gives each invocation whole words of a `bf16`
// result, two adjacent elements each, and never reaches for atomics.

/// The wire tag of [`Dtype::Bf16`]. Tag 4 is a channel's `Act` dtype
/// (`container::DT_ACT`), so `bf16` takes 5.
pub const WIRE_BF16: u8 = 5;

pub const WIRE_ORDER: &[Dtype] = &[Dtype::F32, Dtype::I32, Dtype::U32, Dtype::Bool];

pub const fn class_of(d: Dtype) -> Option<DtypeClass> {
    match d {
        Dtype::F32 => Some(DtypeClass::Float),
        Dtype::Bf16 => Some(DtypeClass::Float),
        Dtype::I32 => Some(DtypeClass::Int),
        Dtype::U32 => Some(DtypeClass::Int),
        Dtype::Bool => Some(DtypeClass::Logical),

        Dtype::F16
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
        | Dtype::Nvfp4
        | Dtype::U2g16k
        | Dtype::I3g16k
        | Dtype::U4g32k
        | Dtype::U5g32k
        | Dtype::I6g16k
        | Dtype::E4m3row
        | Dtype::E4m3tile128
        | Dtype::E4m3g128
        | Dtype::E4m3tensor
        | Dtype::Fp8b128
        | Dtype::Fp8b32
        | Dtype::Mxfp8
        | Dtype::Ptq1_0
        | Dtype::KvU4 => None,
    }
}

pub const fn name(d: Dtype) -> Option<&'static str> {
    match d {
        Dtype::F32 => Some("f32"),
        Dtype::I32 => Some("i32"),
        Dtype::U32 => Some("u32"),
        Dtype::Bool => Some("bool"),
        Dtype::Bf16 => Some("bf16"),
        _ => None,
    }
}

pub const fn to_wire(d: Dtype) -> Option<u8> {
    match d {
        Dtype::F32 => Some(0),
        Dtype::I32 => Some(1),
        Dtype::U32 => Some(2),
        Dtype::Bool => Some(3),
        Dtype::Bf16 => Some(WIRE_BF16),
        _ => None,
    }
}

#[cfg(test)]
mod dtype_tests {
    use super::*;

    #[test]
    fn the_wire_bytes_are_the_ones_the_format_froze() {
        assert_eq!(to_wire(Dtype::F32), Some(0));
        assert_eq!(to_wire(Dtype::I32), Some(1));
        assert_eq!(to_wire(Dtype::U32), Some(2));
        assert_eq!(to_wire(Dtype::Bool), Some(3));
        assert_eq!(to_wire(Dtype::Bf16), Some(5));
        assert_eq!(WIRE_ORDER.len(), 4);
        assert_eq!(from_wire(4), None, "4 is a channel's Act, never a value's");
    }

    #[test]
    fn every_eta_dtype_round_trips_its_wire_tag() {
        for d in [Dtype::F32, Dtype::I32, Dtype::U32, Dtype::Bool, Dtype::Bf16] {
            assert!(supports(d), "{d:?}");
            assert_eq!(from_wire(wire_dtype(d)), Some(d), "{d:?}");
            assert!(name(d).is_some(), "{d:?}");
        }
        assert!(is_float(Dtype::Bf16) && is_numeric(Dtype::Bf16) && !is_int(Dtype::Bf16));
        assert_eq!(to_wire(Dtype::F16), None, "f16 stays out");
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(bf16_bits(1.0), 0x3F80);
        // 1 + 2^-8 is the tie between 1 and 1 + 2^-7: even keeps 1.
        assert_eq!(bf16_bits(f32::from_bits(0x3F80_8000)), 0x3F80);
        // 1 + 3 * 2^-8 ties between odd 0x3F81 and even 0x3F82.
        assert_eq!(bf16_bits(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert_eq!(bf16_bits(f32::from_bits(0x3F80_8001)), 0x3F81);
        assert_eq!(bf16_bits(f32::INFINITY), 0x7F80);
        assert_eq!(bf16_bits(f32::NEG_INFINITY), 0xFF80);
        assert_eq!(bf16_bits(f32::MAX), 0x7F80, "past the largest bf16 rounds to inf");
        assert!(bf16_to_f32(bf16_bits(f32::NAN)).is_nan());
        assert!(bf16_to_f32(bf16_bits(f32::from_bits(0x7F80_0001))).is_nan(), "a low NaN stays NaN");
        assert_eq!(bf16_bits(-0.0), 0x8000);
        for bits in [0u16, 0x3F80, 0xC2F7, 0x0001, 0x8001, 0x7F7F] {
            assert_eq!(bf16_bits(bf16_to_f32(bits)), bits, "{bits:#06x} is exact");
        }
    }
}

pub const fn from_wire(byte: u8) -> Option<Dtype> {
    let index = byte as usize;
    if index < WIRE_ORDER.len() {
        Some(WIRE_ORDER[index])
    } else if byte == WIRE_BF16 {
        Some(Dtype::Bf16)
    } else {
        None
    }
}

/// `x` rounded to bf16, round-to-nearest-even, as its 16 bits: the one
/// rounding every backend's `bf16` store does. A NaN stays a quiet NaN of
/// its sign; past the largest bf16 rounds to infinity.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub const fn bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// The f32 a bf16's bits name, exactly.
#[must_use]
pub const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// `x` rounded to bf16 and widened back: the f32 a `bf16` store keeps.
#[must_use]
pub const fn round_bf16(x: f32) -> f32 {
    bf16_to_f32(bf16_bits(x))
}

/// The dtype an elementwise op stores for float operands `a` and `b`
/// (see the rules above): `bf16` if either is, else `a`. `None` where the
/// two are not both floats and differ.
#[must_use]
pub fn float_result(a: Dtype, b: Dtype) -> Option<Dtype> {
    if a == b {
        return Some(a);
    }
    if is_float(a) && is_float(b) {
        return Some(Dtype::Bf16);
    }
    None
}

pub const fn supports(d: Dtype) -> bool {
    class_of(d).is_some()
}

pub const fn is_float(d: Dtype) -> bool {
    matches!(class_of(d), Some(DtypeClass::Float))
}
pub const fn is_int(d: Dtype) -> bool {
    matches!(class_of(d), Some(DtypeClass::Int))
}
pub const fn is_numeric(d: Dtype) -> bool {
    matches!(class_of(d), Some(DtypeClass::Float | DtypeClass::Int))
}

pub const fn name_or_unknown(d: Dtype) -> &'static str {
    match name(d) {
        Some(n) => n,
        None => "<not an eta dtype>",
    }
}

pub fn wire_dtype(d: Dtype) -> u8 {
    match to_wire(d) {
        Some(byte) => byte,
        None => panic!("dtype {d:?} is not one ETA computes in; it has no wire tag"),
    }
}

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
    pub fn rows(&self) -> u64 {
        match self.rank as usize {
            0 | 1 => 1,
            r => self.dims[..r - 1].iter().map(|&d| d as u64).product(),
        }
    }
    pub fn drop_last(&self) -> Option<Shape> {
        if self.rank == 0 {
            return None;
        }
        Shape::new(&self.dims[..self.rank as usize - 1])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueType {
    pub shape: Shape,
    pub dtype: Dtype,
}

impl ValueType {
    pub const fn new(shape: Shape, dtype: Dtype) -> Self {
        Self { shape, dtype }
    }
    pub fn scalar(dtype: Dtype) -> Self {
        Self {
            shape: Shape::SCALAR,
            dtype,
        }
    }
    pub fn vector(n: u32, dtype: Dtype) -> Self {
        Self {
            shape: Shape::vector(n),
            dtype,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Predicate {
    RankLe(ValueId),
    CummassLe(ValueId),
    ProbGe(ValueId),
}

impl Predicate {
    pub fn value(self) -> ValueId {
        match self {
            Predicate::RankLe(value) | Predicate::CummassLe(value) | Predicate::ProbGe(value) => {
                value
            }
        }
    }

    pub fn value_slot(&mut self) -> &mut ValueId {
        match self {
            Predicate::RankLe(value) | Predicate::CummassLe(value) | Predicate::ProbGe(value) => {
                value
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum RngKind {
    Uniform = 0,
    Gumbel = 1,
    Normal = 2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Literal {
    F32(f32),
    I32(i32),
    U32(u32),
    Bool(bool),
}

impl Literal {
    pub fn dtype(self) -> Dtype {
        match self {
            Literal::F32(_) => Dtype::F32,
            Literal::I32(_) => Dtype::I32,
            Literal::U32(_) => Dtype::U32,
            Literal::Bool(_) => Dtype::Bool,
        }
    }
}
