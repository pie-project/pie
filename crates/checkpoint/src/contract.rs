//! The model contract: tensor declarations as expressions over the
//! checkpoint's byte space. This module owns the declaration grammar;
//! [`infer`] types it, [`compile`] lowers it to byte spans, [`rewrite`] edits
//! a checked contract in place.

use serde::{Deserialize, Serialize};

use crate::error::{Error, OrOverflow};
pub use crate::types::Visibility;
use crate::types::{Axis, DType, Encoding, QuantGranularity, RepackLayout, ScaleForm};

pub mod compile;
pub mod infer;
pub mod materialize;
pub mod rewrite;

/// A tensor-valued expression. `Src`..`Shard` are the affine fragment
/// (placement only, compiles to byte spans, no kernel); `Repack`/`Cast`/
/// `Scale` are kernel escape hatches. No node permutes axes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Expr {
    /// A tensor read from the checkpoint, by its on-disk name.
    Src(String),
    /// A tensor produced by an earlier [`TensorContract`] in the same
    /// [`ModelContract`], by its declared name.
    Out(String),
    /// A constant tensor: every element is `value`, as [`f32::to_bits`].
    /// Realized by zeroing the destination, so `ty` must be a dtype whose
    /// zero byte denotes the value (excludes `E8m0` and quantized encodings).
    Fill { value: u32, ty: TensorType },
    /// A contiguous band: `out[.., i, ..] = src[.., start + i, ..]` for `i` in
    /// `0..len`. The placement that costs nothing: never breaks contiguity
    /// below this axis.
    Slice {
        src: Box<Expr>,
        axis: Axis,
        start: i64,
        len: i64,
    },
    /// A strided selection: `out[.., i, ..] = src[.., start + i * step, ..]`
    /// for `i` in `0..len`, with `step >= 2`. May not touch a quantized axis
    /// (breaks block alignment).
    Stride {
        src: Box<Expr>,
        axis: Axis,
        start: i64,
        len: i64,
        step: i64,
    },
    /// An explicit selection: `out[.., i, ..] = src[.., indices[i], ..]`.
    /// Indices are constants (keeps the algebra `Eq`/hashable). Duplicates
    /// allowed (broadcast); may not touch a quantized axis.
    Gather {
        src: Box<Expr>,
        axis: Axis,
        indices: Vec<i64>,
    },
    /// Concatenation along `axis`. Parts must agree on every other extent.
    Concat { axis: Axis, parts: Vec<Expr> },
    /// Renaming: the same bytes, a different type. Byte size preserved; a
    /// changed element width requires a whole-tensor operand (`Src`/`Out`);
    /// if both sides are quantized, the blocked-axis-onward shape must be
    /// unchanged. At most one extent may be `-1`.
    Transmute { src: Box<Expr>, to: TensorType },
    /// This rank's `1/world` partition of `src` along `axis`. Resolved into a
    /// concrete [`Expr::Slice`] before lowering; at [`Partition::WHOLE`] it
    /// types as its operand.
    Shard { src: Box<Expr>, axis: Axis },
    /// A checkpoint tensor whose name carries this group instance's index:
    /// `Src(template.replace("{}", index))`. Exactly one `{}`. Only for
    /// grids whose members are separate checkpoint tensors; a fused
    /// `[E, ..]` bank uses [`Expr::Select`].
    SrcIndexed(String),
    /// This group instance's band: [`Expr::Slice`] at `start = index *
    /// stride`. Affine in the index, so every instance is the same extent at
    /// a different offset. Type does not mention the index.
    Select {
        src: Box<Expr>,
        axis: Axis,
        stride: i64,
        len: i64,
    },
    /// Escape hatch: a backend-specific layout swizzle, opaque to the type
    /// checker. Element width unchanged between `src` and `to`; may add
    /// trailing zero-padding to fill a layout's tile quantum.
    Repack {
        src: Box<Expr>,
        layout: RepackLayout,
        to: TensorType,
    },
    /// Escape hatch: same values, different representation (dual of
    /// `Transmute`). Covers raw-to-raw cast, raw-to-quantized encode
    /// (publishes a scales tensor), quantized-to-raw decode. Shape
    /// preserved. Re-encoding one quantized scheme as another directly is
    /// refused; decode to `Internal` visibility and cast that instead.
    Cast { src: Box<Expr>, to: Encoding },
    /// Escape hatch: elementwise multiply, `out[i] = src[i] * factor[i]`.
    /// With a `PerBlock` factor this is dequantization. Only variant with
    /// two children: scales are outputs on encode, inputs on decode.
    Scale { src: Box<Expr>, factor: ScaleFactor },
    /// Escape hatch: elementwise add of one constant, as [`f32::to_bits`].
    /// Not a second field on `Scale`, to avoid ambiguous double-identity
    /// states. Per-block form is the zero-point half of an affine decode.
    Bias { src: Box<Expr>, by: BiasBy },
    /// Escape hatch: an elementwise function of one operand that `Scale` and
    /// `Bias` cannot express, because they are affine and it is not. Shape
    /// and dtype preserved. Import-time only: no device mask carries the
    /// transform, so a serving plan may not name one — what a converted
    /// artifact holds is the answer, not the recipe.
    Unary { src: Box<Expr>, op: UnaryOp },
}

/// Which elementwise function [`Expr::Unary`] applies. One variant per
/// function actually needed; each states the arithmetic and its domain,
/// because a value outside the domain is a wrong checkpoint rather than a
/// wrong number and must be refused as one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    /// `out = ln(-x)`, defined for `x < 0`. The inverse of `x -> -exp(x)`,
    /// which is how a decay rate is stored by a converter that keeps the
    /// rate itself where the model reads its logarithm.
    NegLn,
}

impl UnaryOp {
    /// The function, over one element.
    #[must_use]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Self::NegLn => (-x).ln(),
        }
    }

    /// Whether `x` is in this function's domain.
    #[must_use]
    pub fn defined_at(self, x: f64) -> bool {
        match self {
            Self::NegLn => x < 0.0,
        }
    }

    /// What the domain is, for a refusal that says why.
    #[must_use]
    pub const fn domain(self) -> &'static str {
        match self {
            Self::NegLn => "strictly negative",
        }
    }
}

/// What [`Expr::Bias`] adds; same shape as [`ScaleFactor`] (uniform or
/// per-block).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BiasBy {
    /// One compile-time constant for every element, as [`f32::to_bits`].
    Uniform(u32),
    /// One addend per block, from a declared tensor (must be [`Expr::Out`])
    /// — the zero-point half of an affine decode.
    PerBlock { by: Box<Expr> },
}

/// What [`Expr::Scale`] multiplies by: a uniform constant or a per-block
/// factor tensor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleFactor {
    /// One compile-time constant, as [`f32::to_bits`] (bit-exact so [`Expr`]
    /// stays `Eq`/hashable). `infer` rejects non-finite and zero factors
    /// (zero indicates a forgotten field, not a real scale).
    Uniform(u32),
    /// One factor per block, from a companion expression:
    /// `out[i0, .., ik] = src[i0, .., ik] * by[i0 / g0, .., ik / gk]` where
    /// `gj = src.shape[j] / by.shape[j]` — the grouping is the shape ratio,
    /// nothing else states it. The scales tensor takes the same
    /// [`Expr::Shard`]/[`Expr::Slice`]/[`Expr::Concat`] the weight takes,
    /// and `infer` checks it against the weight.
    PerBlock { by: Box<Expr> },
}

/// The type of a tensor-valued expression: logical shape plus how its elements
/// are encoded.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorType {
    /// Logical shape, in elements.
    pub shape: Vec<i64>,
    /// How those elements are encoded.
    pub encoding: Encoding,
}

impl TensorType {
    /// A type with the given shape and encoding.
    pub fn new(shape: Vec<i64>, encoding: Encoding) -> Self {
        Self { shape, encoding }
    }

    /// A raw (unquantized) type of `dtype`.
    pub fn raw(shape: Vec<i64>, dtype: DType) -> Self {
        Self {
            shape,
            encoding: Encoding::Raw(dtype),
        }
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Number of logical elements, or an error on overflow.
    pub fn element_count(&self) -> Result<i64, Error> {
        self.shape.iter().try_fold(1_i64, |acc, dim| {
            acc.checked_mul(*dim)
                .or_overflow(format!("shape {:?} overflows i64", self.shape))
        })
    }

    /// Size in bytes. Errors when a sub-byte encoding does not fill whole bytes.
    pub fn byte_size(&self) -> Result<u64, Error> {
        crate::types::encoding_nbytes(&self.shape, &self.encoding).ok_or_else(|| {
            Error::Contract(format!(
                "shape {:?} of {:?} has no whole-byte size",
                self.shape, self.encoding
            ))
        })
    }
}

/// One declared tensor. `encoding` is what the loader must produce; `shape`
/// is a prediction checked against the expression's actual type, for the
/// whole tensor (checked at [`Partition::WHOLE`] even though the plan
/// declares this rank's band). `shape: None` declines the prediction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorContract {
    /// Declared name of the tensor.
    pub name: String,
    /// How to build it from the checkpoint. See [`Expr`].
    pub expr: Expr,
    /// Declared logical shape of the whole tensor, or `None` where the
    /// contract makes no claim. Rank-independent: see this type's docs.
    pub shape: Option<Vec<i64>>,
    /// Declared encoding of its elements.
    pub encoding: Encoding,
    /// Set when this entry holds scales for another entry. See [`Scales`].
    pub scales: Option<Scales>,
    /// Set when this entry holds the zero points of another entry — the
    /// declared name of the weight it offsets. Only for zero points the
    /// checkpoint shipped. Defaults to `None` for pre-existing contracts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zero_points: Option<String>,
    /// Whether the engine binds this. See [`Visibility`]. Defaults to
    /// `Public` for contracts written before this field existed.
    #[serde(default)]
    pub visibility: Visibility,
}

/// What a scale tensor scales, stated explicitly rather than inferred from
/// name suffixes. Only for scales the checkpoint shipped; an encoded scales
/// plane is bound by name with no contract entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scales {
    /// Name of the tensor these scales belong to. Unlike [`Expr::Out`] this
    /// may name a *later* entry: it pairs two published tensors rather than
    /// feeding one into the other.
    pub of: String,
    /// How finely the scales divide `of`. See [`QuantGranularity`].
    pub granularity: QuantGranularity,
    /// Elements of `of` per scale entry, for [`QuantGranularity::PerGroup`].
    pub group_size: u32,
    /// The axis of `of` that per-channel scales index.
    pub channel_axis: u32,
    /// How the scale values themselves are stored. See [`ScaleForm`].
    pub form: ScaleForm,
}

impl TensorContract {
    /// A public entry named `name` and built by `expr`, declaring no scales.
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

    /// A declaration that states the encoding it wants and declines to predict
    /// the shape.
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

    /// Keep this declaration out of the engine's namespace. See
    /// [`Visibility::Internal`].
    pub fn internal(mut self) -> Self {
        self.visibility = Visibility::Internal;
        self
    }

    /// Declare that this entry holds the scales for `of`.
    pub fn scaling(mut self, scales: Scales) -> Self {
        self.scales = Some(scales);
        self
    }

    /// Declare that this entry holds the zero points for `of`. See
    /// [`TensorContract::zero_points`].
    pub fn offsetting(mut self, of: impl Into<String>) -> Self {
        self.zero_points = Some(of.into());
        self
    }
}

/// Everything one engine rank needs, as a name-resolved DAG. `tensors` is in
/// declaration order; [`Expr::Out`] may only name an earlier entry (DAG
/// acyclic by construction, so the checker runs in one pass).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelContract {
    /// Byte alignment every materialized buffer must satisfy (a target
    /// property).
    pub alignment: u32,
    /// Every tensor the contract declares.
    pub tensors: Vec<TensorContract>,
    /// Grids of interchangeable tensors, declared once and instantiated
    /// `arity` times. See [`GroupContract`].
    #[serde(default)]
    pub groups: Vec<GroupContract>,
}

/// A grid of interchangeable tensor sets, written once and instantiated
/// `arity` times (e.g. MoE expert weights). States that members are
/// same-size and interchangeable; not a residency decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupContract {
    /// Names this grid for diagnostics and for the engine's own bookkeeping.
    pub name: String,
    /// How many instances the grid has. The index runs `0..arity`.
    pub arity: u32,
    /// The tensors one instance is made of, as expressions that may mention
    /// the index through [`Expr::SrcIndexed`] and [`Expr::Select`].
    pub tensors: Vec<TensorContract>,
}

/// Which slice of a tensor-parallel world an expression is read for. Carried
/// as one value so the type checker and specializer can't be given
/// different answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Partition {
    /// Which rank of `world` this partition is.
    pub rank: u32,
    /// How many ranks share the tensor.
    pub world: u32,
}

impl Partition {
    /// The unsplit tensor: one rank owning everything; what every
    /// single-GPU load uses.
    pub const WHOLE: Self = Self { rank: 0, world: 1 };

    /// The partition `rank` of `world` denotes.
    pub fn new(rank: u32, world: u32) -> Self {
        Self { rank, world }
    }
}

impl Default for Partition {
    fn default() -> Self {
        Self::WHOLE
    }
}

/// The `[start, len)` band of a `full`-long axis that `rank` of `world`
/// owns. Both failure modes are [`Error::Shard`]: `world` not dividing
/// `full`, and `rank >= world`.
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

/// Resolves a [`Expr::Transmute`] shape against an operand of `total`
/// elements at the requested type, replacing a single `-1` with the extent
/// that fits. Counted in elements of the *output* type, so the wildcard
/// means the same byte span regardless of element width.
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
    /// The node's constructor name, for diagnostics.
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

    /// Whether an [`Expr::Shard`] appears anywhere in this expression.
    /// A name read through [`Expr::Out`] is not followed (its own shards
    /// were already resolved when it was built).
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

    /// A checkpoint tensor named `name`. See [`Expr::Src`].
    pub fn src(name: impl Into<String>) -> Self {
        Expr::Src(name.into())
    }

    /// A published tensor named `name`. See [`Expr::Out`].
    pub fn out(name: impl Into<String>) -> Self {
        Expr::Out(name.into())
    }

    /// A checkpoint tensor named by substituting the group index into
    /// `template`. See [`Expr::SrcIndexed`].
    pub fn src_indexed(template: impl Into<String>) -> Self {
        Expr::SrcIndexed(template.into())
    }

    /// This instance's band of a fused grid. See [`Expr::Select`].
    pub fn select(self, axis: u8, stride: i64, len: i64) -> Self {
        Expr::Select {
            src: Box::new(self),
            axis: Axis(axis),
            stride,
            len,
        }
    }

    /// The `[start, start + len)` run of `axis`. See [`Expr::Slice`].
    pub fn slice(self, axis: u8, start: i64, len: i64) -> Self {
        Expr::Slice {
            src: Box::new(self),
            axis: Axis(axis),
            start,
            len,
        }
    }

    /// Every `step`-th element of `axis` from `start`. See [`Expr::Stride`].
    pub fn stride(self, axis: u8, start: i64, len: i64, step: i64) -> Self {
        Expr::Stride {
            src: Box::new(self),
            axis: Axis(axis),
            start,
            len,
            step,
        }
    }

    /// Select `indices` along `axis`, in the order given. See [`Expr::Gather`].
    pub fn gather(self, axis: u8, indices: Vec<i64>) -> Self {
        Expr::Gather {
            src: Box::new(self),
            axis: Axis(axis),
            indices,
        }
    }

    /// `parts` joined along `axis`. See [`Expr::Concat`].
    pub fn concat(axis: u8, parts: Vec<Expr>) -> Self {
        Expr::Concat {
            axis: Axis(axis),
            parts,
        }
    }

    /// A constant tensor of `value` with type `ty`. See [`Expr::Fill`].
    pub fn fill(value: f32, ty: TensorType) -> Self {
        Expr::Fill {
            value: value.to_bits(),
            ty,
        }
    }

    /// The same bytes read as `to`. See [`Expr::Transmute`].
    pub fn transmute(self, to: TensorType) -> Self {
        Expr::Transmute {
            src: Box::new(self),
            to,
        }
    }

    /// The same values in `layout`, typed `to`. See [`Expr::Repack`].
    pub fn repack(self, layout: RepackLayout, to: TensorType) -> Self {
        Expr::Repack {
            src: Box::new(self),
            layout,
            to,
        }
    }

    /// The same values in `to`. See [`Expr::Cast`].
    pub fn cast(self, to: Encoding) -> Self {
        Expr::Cast {
            src: Box::new(self),
            to,
        }
    }

    /// Multiply every element by `factor`.
    pub fn scale(self, factor: f32) -> Self {
        Expr::Scale {
            src: Box::new(self),
            factor: ScaleFactor::Uniform(factor.to_bits()),
        }
    }

    /// Apply `op` to every element. See [`Expr::Unary`].
    pub fn unary(self, op: UnaryOp) -> Self {
        Expr::Unary {
            src: Box::new(self),
            op,
        }
    }

    /// Add `by` to every element.
    pub fn bias(self, by: f32) -> Self {
        Expr::Bias {
            src: Box::new(self),
            by: BiasBy::Uniform(by.to_bits()),
        }
    }

    /// Add `by`, one addend per block, blocked by the shape ratio — the
    /// zero-point half of an affine decode.
    pub fn bias_per_block(self, by: Expr) -> Self {
        Expr::Bias {
            src: Box::new(self),
            by: BiasBy::PerBlock { by: Box::new(by) },
        }
    }

    /// Multiply by `by`, one factor per block, blocked by the shape ratio.
    ///
    /// Over a quantized `self` this is dequantization, and the result is the
    /// scheme's logical dtype.
    pub fn scale_per_block(self, by: Expr) -> Self {
        Expr::Scale {
            src: Box::new(self),
            factor: ScaleFactor::PerBlock { by: Box::new(by) },
        }
    }

    /// This rank's share of `axis`. See [`Expr::Shard`].
    pub fn shard(self, axis: u8) -> Self {
        Expr::Shard {
            src: Box::new(self),
            axis: Axis(axis),
        }
    }

    /// Whether this expression lies entirely in the affine fragment, and so
    /// compiles to byte spans with no kernel and no intermediate buffer.
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

    /// Names of the checkpoint tensors this expression reads, in traversal
    /// order. Duplicates are preserved; the caller decides whether it cares.
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

    /// Names of the *earlier contracts* this expression reads, in traversal
    /// order. These are the edges of the contract DAG.
    pub fn outputs(&self) -> Vec<&str> {
        let mut found = Vec::new();
        self.visit(&mut |expr| {
            if let Expr::Out(name) = expr {
                found.push(name.as_str());
            }
        });
        found
    }

    /// Visits every node (self first, then operands in traversal order) —
    /// the one place the grammar's shape is enumerated for reads.
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

    /// Rebuilds this node with each immediate operand replaced by `f`. Only
    /// immediate operands: `f` decides whether to recurse.
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
