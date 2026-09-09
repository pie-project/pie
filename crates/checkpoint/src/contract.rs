use serde::{Deserialize, Serialize};

use crate::error::{Error, OrOverflow};
pub use crate::types::Visibility;
use crate::types::{Axis, DType, Encoding, QuantGranularity, RepackLayout, ScaleForm};

pub mod compile;
pub mod infer;
pub mod materialize;
pub mod rewrite;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Expr {
    Src(String),
    Out(String),
    Fill {
        value: u32,
        ty: TensorType,
    },
    Slice {
        src: Box<Expr>,
        axis: Axis,
        start: i64,
        len: i64,
    },
    Stride {
        src: Box<Expr>,
        axis: Axis,
        start: i64,
        len: i64,
        step: i64,
    },
    Gather {
        src: Box<Expr>,
        axis: Axis,
        indices: Vec<i64>,
    },
    Concat {
        axis: Axis,
        parts: Vec<Expr>,
    },
    Transmute {
        src: Box<Expr>,
        to: TensorType,
    },
    Shard {
        src: Box<Expr>,
        axis: Axis,
    },
    SrcIndexed(String),
    Select {
        src: Box<Expr>,
        axis: Axis,
        stride: i64,
        len: i64,
    },
    Repack {
        src: Box<Expr>,
        layout: RepackLayout,
        to: TensorType,
    },
    Cast {
        src: Box<Expr>,
        to: Encoding,
    },
    Scale {
        src: Box<Expr>,
        factor: ScaleFactor,
    },
    Bias {
        src: Box<Expr>,
        by: BiasBy,
    },
    Unary {
        src: Box<Expr>,
        op: UnaryOp,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    NegLn,
    Sqrt,
    Rsqrt,
}

impl UnaryOp {
    #[must_use]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Self::NegLn => (-x).ln(),
            Self::Sqrt => x.sqrt(),
            Self::Rsqrt => x.sqrt().recip(),
        }
    }

    #[must_use]
    pub fn defined_at(self, x: f64) -> bool {
        match self {
            Self::NegLn => x < 0.0,
            Self::Sqrt => x >= 0.0,
            Self::Rsqrt => x > 0.0,
        }
    }

    #[must_use]
    pub const fn domain(self) -> &'static str {
        match self {
            Self::NegLn => "strictly negative",
            Self::Sqrt => "non-negative",
            Self::Rsqrt => "strictly positive",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BiasBy {
    Uniform(u32),
    PerBlock { by: Box<Expr> },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleFactor {
    Uniform(u32),
    PerBlock { by: Box<Expr> },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorType {
    pub shape: Vec<i64>,
    pub encoding: Encoding,
}

impl TensorType {
    pub fn new(shape: Vec<i64>, encoding: Encoding) -> Self {
        Self { shape, encoding }
    }

    pub fn raw(shape: Vec<i64>, dtype: DType) -> Self {
        Self {
            shape,
            encoding: Encoding::Raw(dtype),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn element_count(&self) -> Result<i64, Error> {
        self.shape.iter().try_fold(1_i64, |acc, dim| {
            acc.checked_mul(*dim)
                .or_overflow(format!("shape {:?} overflows i64", self.shape))
        })
    }

    pub fn byte_size(&self) -> Result<u64, Error> {
        crate::types::encoding_nbytes(&self.shape, &self.encoding).ok_or_else(|| {
            Error::Contract(format!(
                "shape {:?} of {:?} has no whole-byte size",
                self.shape, self.encoding
            ))
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorContract {
    pub name: String,
    pub expr: Expr,
    pub shape: Option<Vec<i64>>,
    pub encoding: Encoding,
    pub scales: Option<Scales>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zero_points: Option<String>,
    #[serde(default)]
    pub visibility: Visibility,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scales {
    pub of: String,
    pub granularity: QuantGranularity,
    pub group_size: u32,
    pub channel_axis: u32,
    pub form: ScaleForm,
}

impl TensorContract {
    pub fn new(name: impl Into<String>, expr: Expr, shape: Vec<i64>, encoding: Encoding) -> Self {
        Self {
            name: name.into(),
            expr,
            shape: Some(shape),
            encoding,
            scales: None,
            zero_points: None,
            visibility: Visibility::Public,
        }
    }

    pub fn inferred(name: impl Into<String>, expr: Expr, encoding: Encoding) -> Self {
        Self {
            name: name.into(),
            expr,
            shape: None,
            encoding,
            scales: None,
            zero_points: None,
            visibility: Visibility::Public,
        }
    }

    pub fn internal(mut self) -> Self {
        self.visibility = Visibility::Internal;
        self
    }

    pub fn scaling(mut self, scales: Scales) -> Self {
        self.scales = Some(scales);
        self
    }

    pub fn offsetting(mut self, of: impl Into<String>) -> Self {
        self.zero_points = Some(of.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelContract {
    pub alignment: u32,
    pub tensors: Vec<TensorContract>,
    #[serde(default)]
    pub groups: Vec<GroupContract>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupContract {
    pub name: String,
    pub arity: u32,
    pub tensors: Vec<TensorContract>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Partition {
    pub rank: u32,
    pub world: u32,
}

impl Partition {
    pub const WHOLE: Self = Self { rank: 0, world: 1 };

    pub fn new(rank: u32, world: u32) -> Self {
        Self { rank, world }
    }
}

impl Default for Partition {
    fn default() -> Self {
        Self::WHOLE
    }
}

pub fn local_range(full: i64, world: u32, rank: u32, what: &str) -> Result<(i64, i64), Error> {
    let world = world.max(1);
    if rank >= world {
        return Err(Error::Shard(format!(
            "tp_rank {rank} is out of range for tp_size {world}; ranks are 0..{world}"
        )));
    }
    let world = i64::from(world);
    if full % world != 0 {
        return Err(Error::Shard(format!(
            "{what} is {full}, which tp_size {world} does not divide; use a \
             tp_size that divides it or run single-GPU"
        )));
    }
    let local = full / world;
    Ok((i64::from(rank) * local, local))
}

pub fn resolve_extents(requested: &[i64], total: i64) -> Result<Vec<i64>, Error> {
    if requested.is_empty() {
        return Err(Error::Contract(
            "Transmute needs at least one extent".to_string(),
        ));
    }
    let mut wildcard = None;
    let mut known = 1_i64;
    for (index, extent) in requested.iter().enumerate() {
        match *extent {
            -1 if wildcard.is_some() => {
                return Err(Error::Contract(
                    "Transmute allows at most one -1 extent".to_string(),
                ));
            }
            -1 => wildcard = Some(index),
            extent if extent < 1 => {
                return Err(Error::Contract(format!(
                    "Transmute extent {extent} must be >= 1 or -1"
                )));
            }
            extent => {
                known = known
                    .checked_mul(extent)
                    .or_overflow("Transmute extent overflows i64")?;
            }
        }
    }
    let mut shape = requested.to_vec();
    match wildcard {
        Some(index) if known > 0 && total % known == 0 => shape[index] = total / known,
        Some(_) => {
            return Err(Error::Contract(format!(
                "Transmute to {requested:?} does not divide {total} elements evenly"
            )));
        }
        None if known == total => {}
        None => {
            return Err(Error::Contract(format!(
                "Transmute to {requested:?} is {known} elements, not the {total} the operand's bytes hold"
            )));
        }
    }
    Ok(shape)
}

impl Expr {
    pub fn node_name(&self) -> &'static str {
        match self {
            Expr::Src(_) => "Src",
            Expr::Out(_) => "Out",
            Expr::Fill { .. } => "Fill",
            Expr::Slice { .. } => "Slice",
            Expr::Stride { .. } => "Stride",
            Expr::Gather { .. } => "Gather",
            Expr::Concat { .. } => "Concat",
            Expr::Transmute { .. } => "Transmute",
            Expr::Shard { .. } => "Shard",
            Expr::Repack { .. } => "Repack",
            Expr::Cast { .. } => "Cast",
            Expr::Scale { .. } => "Scale",
            Expr::Bias { .. } => "Bias",
            Expr::Unary { .. } => "Unary",
            Expr::SrcIndexed(_) => "SrcIndexed",
            Expr::Select { .. } => "Select",
        }
    }

    #[must_use]
    pub fn is_sharded(&self) -> bool {
        match self {
            Expr::Shard { .. } => true,
            Expr::Src(_) | Expr::Out(_) | Expr::Fill { .. } | Expr::SrcIndexed(_) => false,
            Expr::Slice { src, .. }
            | Expr::Stride { src, .. }
            | Expr::Gather { src, .. }
            | Expr::Select { src, .. }
            | Expr::Transmute { src, .. }
            | Expr::Repack { src, .. }
            | Expr::Unary { src, .. }
            | Expr::Cast { src, .. } => src.is_sharded(),
            Expr::Bias { src, by } => {
                src.is_sharded()
                    || match by {
                        BiasBy::Uniform(_) => false,
                        BiasBy::PerBlock { by } => by.is_sharded(),
                    }
            }
            Expr::Scale { src, factor } => {
                src.is_sharded()
                    || match factor {
                        ScaleFactor::Uniform(_) => false,
                        ScaleFactor::PerBlock { by } => by.is_sharded(),
                    }
            }
            Expr::Concat { parts, .. } => parts.iter().any(Expr::is_sharded),
        }
    }

    pub fn src(name: impl Into<String>) -> Self {
        Expr::Src(name.into())
    }

    pub fn out(name: impl Into<String>) -> Self {
        Expr::Out(name.into())
    }

    pub fn src_indexed(template: impl Into<String>) -> Self {
        Expr::SrcIndexed(template.into())
    }

    pub fn select(self, axis: u8, stride: i64, len: i64) -> Self {
        Expr::Select {
            src: Box::new(self),
            axis: Axis(axis),
            stride,
            len,
        }
    }

    pub fn slice(self, axis: u8, start: i64, len: i64) -> Self {
        Expr::Slice {
            src: Box::new(self),
            axis: Axis(axis),
            start,
            len,
        }
    }

    pub fn stride(self, axis: u8, start: i64, len: i64, step: i64) -> Self {
        Expr::Stride {
            src: Box::new(self),
            axis: Axis(axis),
            start,
            len,
            step,
        }
    }

    pub fn gather(self, axis: u8, indices: Vec<i64>) -> Self {
        Expr::Gather {
            src: Box::new(self),
            axis: Axis(axis),
            indices,
        }
    }

    pub fn concat(axis: u8, parts: Vec<Expr>) -> Self {
        Expr::Concat {
            axis: Axis(axis),
            parts,
        }
    }

    pub fn fill(value: f32, ty: TensorType) -> Self {
        Expr::Fill {
            value: value.to_bits(),
            ty,
        }
    }

    pub fn transmute(self, to: TensorType) -> Self {
        Expr::Transmute {
            src: Box::new(self),
            to,
        }
    }

    pub fn repack(self, layout: RepackLayout, to: TensorType) -> Self {
        Expr::Repack {
            src: Box::new(self),
            layout,
            to,
        }
    }

    pub fn cast(self, to: Encoding) -> Self {
        Expr::Cast {
            src: Box::new(self),
            to,
        }
    }

    pub fn scale(self, factor: f32) -> Self {
        Expr::Scale {
            src: Box::new(self),
            factor: ScaleFactor::Uniform(factor.to_bits()),
        }
    }

    pub fn unary(self, op: UnaryOp) -> Self {
        Expr::Unary {
            src: Box::new(self),
            op,
        }
    }

    pub fn bias(self, by: f32) -> Self {
        Expr::Bias {
            src: Box::new(self),
            by: BiasBy::Uniform(by.to_bits()),
        }
    }

    pub fn bias_per_block(self, by: Expr) -> Self {
        Expr::Bias {
            src: Box::new(self),
            by: BiasBy::PerBlock { by: Box::new(by) },
        }
    }

    pub fn scale_per_block(self, by: Expr) -> Self {
        Expr::Scale {
            src: Box::new(self),
            factor: ScaleFactor::PerBlock { by: Box::new(by) },
        }
    }

    pub fn shard(self, axis: u8) -> Self {
        Expr::Shard {
            src: Box::new(self),
            axis: Axis(axis),
        }
    }

    pub fn is_affine(&self) -> bool {
        match self {
            Expr::Src(_) | Expr::Out(_) | Expr::Fill { .. } | Expr::SrcIndexed(_) => true,
            Expr::Slice { src, .. }
            | Expr::Stride { src, .. }
            | Expr::Gather { src, .. }
            | Expr::Transmute { src, .. } => src.is_affine(),
            Expr::Concat { parts, .. } => parts.iter().all(Expr::is_affine),
            Expr::Shard { src, .. } | Expr::Select { src, .. } => src.is_affine(),
            Expr::Repack { .. }
            | Expr::Cast { .. }
            | Expr::Scale { .. }
            | Expr::Bias { .. }
            | Expr::Unary { .. } => false,
        }
    }

    pub fn sources(&self) -> Vec<&str> {
        let mut found = Vec::new();
        self.visit_sources(&mut found);
        found
    }

    fn visit_sources<'a>(&'a self, found: &mut Vec<&'a str>) {
        self.visit(&mut |expr| {
            if let Expr::Src(name) = expr {
                found.push(name.as_str());
            }
        });
    }

    pub fn outputs(&self) -> Vec<&str> {
        let mut found = Vec::new();
        self.visit(&mut |expr| {
            if let Expr::Out(name) = expr {
                found.push(name.as_str());
            }
        });
        found
    }

    pub fn visit<'a>(&'a self, seen: &mut impl FnMut(&'a Expr)) {
        seen(self);
        match self {
            Expr::Src(_) | Expr::Out(_) | Expr::Fill { .. } | Expr::SrcIndexed(_) => {}
            Expr::Slice { src, .. }
            | Expr::Stride { src, .. }
            | Expr::Gather { src, .. }
            | Expr::Transmute { src, .. }
            | Expr::Repack { src, .. }
            | Expr::Shard { src, .. }
            | Expr::Select { src, .. }
            | Expr::Unary { src, .. }
            | Expr::Cast { src, .. } => src.visit(seen),
            Expr::Bias { src, by } => {
                src.visit(seen);
                if let BiasBy::PerBlock { by } = by {
                    by.visit(seen);
                }
            }
            Expr::Scale { src, factor } => {
                src.visit(seen);
                if let ScaleFactor::PerBlock { by } = factor {
                    by.visit(seen);
                }
            }
            Expr::Concat { parts, .. } => {
                for part in parts {
                    part.visit(seen);
                }
            }
        }
    }

    pub fn map_children(
        self,
        mut f: impl FnMut(Expr) -> Result<Expr, Error>,
    ) -> Result<Expr, Error> {
        let mut boxed = |src: Box<Expr>| -> Result<Box<Expr>, Error> { Ok(Box::new(f(*src)?)) };
        Ok(match self {
            Expr::Src(_) | Expr::Out(_) | Expr::Fill { .. } | Expr::SrcIndexed(_) => self,
            Expr::Slice {
                src,
                axis,
                start,
                len,
            } => Expr::Slice {
                src: boxed(src)?,
                axis,
                start,
                len,
            },
            Expr::Stride {
                src,
                axis,
                start,
                len,
                step,
            } => Expr::Stride {
                src: boxed(src)?,
                axis,
                start,
                len,
                step,
            },
            Expr::Gather { src, axis, indices } => Expr::Gather {
                src: boxed(src)?,
                axis,
                indices,
            },
            Expr::Select {
                src,
                axis,
                stride,
                len,
            } => Expr::Select {
                src: boxed(src)?,
                axis,
                stride,
                len,
            },
            Expr::Transmute { src, to } => Expr::Transmute {
                src: boxed(src)?,
                to,
            },
            Expr::Repack { src, layout, to } => Expr::Repack {
                src: boxed(src)?,
                layout,
                to,
            },
            Expr::Unary { src, op } => Expr::Unary {
                src: boxed(src)?,
                op,
            },
            Expr::Cast { src, to } => Expr::Cast {
                src: boxed(src)?,
                to,
            },
            Expr::Scale { src, factor } => Expr::Scale {
                src: boxed(src)?,
                factor: match factor {
                    ScaleFactor::PerBlock { by } => ScaleFactor::PerBlock { by: boxed(by)? },
                    uniform => uniform,
                },
            },
            Expr::Bias { src, by } => Expr::Bias {
                src: boxed(src)?,
                by: match by {
                    BiasBy::PerBlock { by } => BiasBy::PerBlock { by: boxed(by)? },
                    uniform => uniform,
                },
            },
            Expr::Shard { src, axis } => Expr::Shard {
                src: boxed(src)?,
                axis,
            },
            Expr::Concat { axis, parts } => Expr::Concat {
                axis,
                parts: parts.into_iter().map(f).collect::<Result<_, _>>()?,
            },
        })
    }
}
