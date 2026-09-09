pub type ValueId = u32;

pub const MAX_RANK: usize = 4;

pub use dtype::Dtype;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DtypeClass {
    Float,
    Int,
    Logical,
}
pub const WIRE_ORDER: &[Dtype] = &[Dtype::F32, Dtype::I32, Dtype::U32, Dtype::Bool];

pub const fn class_of(d: Dtype) -> Option<DtypeClass> {
    match d {
        Dtype::F32 => Some(DtypeClass::Float),
        Dtype::I32 => Some(DtypeClass::Int),
        Dtype::U32 => Some(DtypeClass::Int),
        Dtype::Bool => Some(DtypeClass::Logical),

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

pub const fn name(d: Dtype) -> Option<&'static str> {
    match d {
        Dtype::F32 => Some("f32"),
        Dtype::I32 => Some("i32"),
        Dtype::U32 => Some("u32"),
        Dtype::Bool => Some("bool"),
        _ => None,
    }
}

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

    #[test]
    fn the_wire_bytes_are_the_ones_the_format_froze() {
        assert_eq!(to_wire(Dtype::F32), Some(0));
        assert_eq!(to_wire(Dtype::I32), Some(1));
        assert_eq!(to_wire(Dtype::U32), Some(2));
        assert_eq!(to_wire(Dtype::Bool), Some(3));
        assert_eq!(WIRE_ORDER.len(), 4);
    }
}

pub const fn from_wire(byte: u8) -> Option<Dtype> {
    let index = byte as usize;
    if index < WIRE_ORDER.len() {
        Some(WIRE_ORDER[index])
    } else {
        None
    }
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
