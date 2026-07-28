//! The model contract: a declaration of what the driver needs, as expressions
//! over the checkpoint's byte space.
//!
//! See `loader/spec.md` for the design rationale. In short: the driver declares
//! every tensor it wants and where its bytes come from; the loader decides how
//! to move them. Neither side needs to know the model family.
//!
//! This module owns the *declaration* — the grammar, the builders, and the
//! types a contract is written in. Everything that computes over one lives in a
//! child: [`infer`] says what an expression denotes and what it denotes at one
//! rank, [`compile`] solves it into byte rectangles, and [`rewrite`] edits a
//! checked contract in place.

use serde::{Deserialize, Serialize};

use crate::error::{Error, OrOverflow};
use crate::types::{Axis, DType, Encoding, QuantGranularity, QuantSpec, RepackSpec, ScaleForm};

pub mod compile;
pub mod infer;
pub mod rewrite;

/// A tensor-valued expression.
///
/// The first six variants form the **affine fragment**: each denotes a
/// piecewise-affine partial map from output coordinates to source coordinates,
/// and the fragment is closed under composition. Any expression built from them
/// alone compiles to a set of byte spans without materializing intermediates.
///
/// [`Expr::Repack`] and [`Expr::Quantize`] are the two escape hatches. They need
/// a kernel and are deliberately marked as such.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Expr {
    /// A tensor read from the checkpoint, by its on-disk name.
    Src(String),
    /// A tensor produced by an earlier [`TensorContract`] in the same
    /// [`ModelContract`], by its declared name.
    ///
    /// Split from [`Expr::Src`] on purpose: contracts routinely re-publish
    /// checkpoint names (a bank plus views under the original names), so a
    /// single namespace with a resolution order would make the difference
    /// between "the file's q_proj" and "my q_proj" depend on declaration order.
    Out(String),
    /// `out[.., i, ..] = src[.., start + i * step, ..]` for `i` in `0..len`.
    ///
    /// Covers tensor-parallel sharding, fused-tensor splitting, and strided
    /// selection (the even/odd row interleave of GPT-OSS gate/up).
    Slice {
        src: Box<Expr>,
        axis: Axis,
        start: i64,
        len: i64,
        step: i64,
    },
    /// Concatenation along `axis`. Parts must agree on every other extent.
    Cat { axis: Axis, parts: Vec<Expr> },
    /// Reinterpretation of the same elements under a new shape. Row-major, so
    /// this is a byte identity. At most one extent may be `-1`.
    Reshape { src: Box<Expr>, shape: Vec<i64> },
    /// Zero-extension along `axis`. Padded coordinates have no source.
    Pad {
        src: Box<Expr>,
        axis: Axis,
        before: i64,
        after: i64,
    },
    /// This rank's `1/world` partition of `src` along `axis`.
    ///
    /// The one node whose meaning depends on the target, and the reason
    /// compilation has a *specialization* stage: a contract is authored once
    /// and specialized once per rank, where [`Resolver::specialize`] rewrites
    /// each `Shard` into the concrete [`Expr::Slice`] that rank reads. Nothing
    /// below the frontend ever sees one.
    ///
    /// Stated as a node rather than a field beside the expression so that a
    /// contract stays rank-independent — the author writes the partition, not
    /// the arithmetic — and so that it composes: a shard of one leg of a
    /// [`Expr::Cat`] is expressible, which a whole-expression flag cannot say.
    ///
    /// The partition itself is [`local_range`].
    Shard { src: Box<Expr>, axis: Axis },
    /// Escape hatch: a backend-specific layout swizzle. Opaque to the type
    /// checker, so it must declare its own output type.
    Repack {
        src: Box<Expr>,
        spec: RepackSpec,
        out: TensorType,
    },
    /// Escape hatch: load-time quantization. Preserves logical shape.
    Quantize { src: Box<Expr>, spec: QuantSpec },
    /// Reinterpretation: the same bytes, under a different type.
    ///
    /// A checkpoint routinely stores sub-byte weights packed into a wider word
    /// (MLX ships 4-bit values eight to a `u32`). Nothing moves; the tensor is
    /// simply named for what it is. Total byte size must be preserved, and only
    /// a whole tensor may be reinterpreted — once the element width changes, an
    /// element offset into a partial view would no longer mean anything.
    Bitcast { src: Box<Expr>, out: TensorType },
}

/// The type of a tensor-valued expression: logical shape plus how its elements
/// are encoded.
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

/// One declared tensor.
///
/// `encoding` is what the driver wants the tensor to *be*, and the loader
/// inserts whatever cast, decode or encode reaches it. `shape` is different: it
/// is a *prediction*, checked against what the expression actually yields, so
/// that a driver whose model of the checkpoint is wrong fails to compile instead
/// of silently binding a plausible-looking buffer.
///
/// A prediction may be declined. `shape: None` says "I do not claim to know",
/// which is the honest answer for a packed quantized weight whose on-disk
/// extents are a property of the quantizer that produced the file rather than of
/// the model. Forcing a claim there is what produced `LogicalShape` in
/// `model_contracts.hpp`: a helper whose only job was to erase a shape the
/// driver had been made to state and could not stand behind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorContract {
    pub name: String,
    pub expr: Expr,
    pub shape: Option<Vec<i64>>,
    pub encoding: Encoding,
    /// Set when this entry holds scales for another entry. See [`Scales`].
    pub scales: Option<Scales>,
}

/// What a scale tensor scales, said by the entry that declares the scales.
///
/// A quantized weight and its scales are two runtime tensors, and the driver's
/// kernels need to know they belong together. The loader used to work that out
/// by matching name suffixes — `{name}_scale_inv`, then `{base}.scale` — with
/// the group size hardcoded to 128 beside them.
///
/// That guess had an author. `deepseek_v4_contract.hpp::dsv4_block_scales_to_fp32`
/// finds the pairing properly: it takes a `.scale` tensor, looks up
/// `<base>.weight`, and publishes the scale only if that companion is really
/// FP8-E4M3 — "guessing is how a scale tensor gets silently reinterpreted", as
/// its own comment puts it. Then it dropped the pair on the floor and the loader
/// re-derived it from strings. This field is the driver keeping what it found.
///
/// Only for scales the *checkpoint* shipped. When the loader quantizes a tensor
/// it creates the scale tensor itself and states the pairing from there, with no
/// name involved at all.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scales {
    /// Declared name of the tensor these scales belong to.
    ///
    /// Unlike [`Expr::Out`] this may name a *later* entry: it pairs two
    /// published tensors rather than feeding one into the other, and the one
    /// authoring site in the tree declares the scales first.
    pub of: String,
    pub granularity: QuantGranularity,
    /// Elements of `of` per scale entry, for [`QuantGranularity::PerGroup`].
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
        }
    }

    /// Declare that this entry holds the scales for `of`.
    pub fn scaling(mut self, scales: Scales) -> Self {
        self.scales = Some(scales);
        self
    }
}

/// Everything one driver rank needs, as a name-resolved DAG.
///
/// `tensors` is in declaration order; [`Expr::Out`] may only name an earlier
/// entry, which makes the DAG acyclic by construction and lets the checker run
/// in one pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelContract {
    /// Byte alignment every materialized buffer must satisfy. A target
    /// property; there is no reason to repeat it per tensor.
    pub alignment: u32,
    pub tensors: Vec<TensorContract>,
}

/// The `[start, len)` of a `full`-long axis that `rank` of `world` owns.///
/// The arithmetic [`Expr::Shard`] denotes, in one place, because two callers
/// need it: shape inference resolves a `Shard` node into a `Slice`, and the
/// rewriter decides whether a row-sharded read can be coalesced. Split between
/// them, the two answers were free to disagree — and did, on which
/// [`Error`](crate::error::Error) an indivisible axis produces.
///
/// `what` names the thing being split, because this is the message a user gets
/// for "tp_size does not divide this model". The driver used to pre-empt it with
/// its own per-family table of divisibility rules read off `config.json` — the
/// same fact checked twice, and only for the families someone had listed.
pub fn local_range(full: i64, world: u32, rank: u32, what: &str) -> Result<(i64, i64), Error> {
    let world = i64::from(world.max(1));
    if full % world != 0 {
        return Err(Error::Shard(format!(
            "{what} is {full}, which tp_size {world} does not divide; use a \
             tp_size that divides it or run single-GPU"
        )));
    }
    let local = full / world;
    Ok((i64::from(rank) * local, local))
}

/// Resolve a [`Expr::Reshape`] request against an operand holding `total`
/// elements, replacing a single `-1` with the extent that makes the count work.
///
/// The only resolver. The type checker needs it to state the output shape and
/// the byte-run compiler needs it to place the operand, and a second spelling
/// of "what does `-1` mean here" is a plan that disagrees with the type it was
/// checked against — silently, because both answers are plausible integers.
pub fn resolve_reshape(requested: &[i64], total: i64) -> Result<Vec<i64>, Error> {
    if requested.is_empty() {
        return Err(Error::Contract(
            "Reshape needs at least one extent".to_string(),
        ));
    }
    let mut wildcard = None;
    let mut known = 1_i64;
    for (index, extent) in requested.iter().enumerate() {
        match *extent {
            -1 if wildcard.is_some() => {
                return Err(Error::Contract(
                    "Reshape allows at most one -1 extent".to_string(),
                ));
            }
            -1 => wildcard = Some(index),
            extent if extent < 1 => {
                return Err(Error::Contract(format!(
                    "Reshape extent {extent} must be >= 1 or -1"
                )));
            }
            extent => {
                known = known
                    .checked_mul(extent)
                    .or_overflow("Reshape extent overflows i64")?;
            }
        }
    }
    let mut shape = requested.to_vec();
    match wildcard {
        Some(index) if known > 0 && total % known == 0 => shape[index] = total / known,
        Some(_) => {
            return Err(Error::Contract(format!(
                "Reshape to {requested:?} does not divide {total} elements evenly"
            )));
        }
        None if known == total => {}
        None => {
            return Err(Error::Contract(format!(
                "Reshape to {requested:?} ({known} elements) changes the element count from {total}"
            )));
        }
    }
    Ok(shape)
}

impl Expr {
    pub fn src(name: impl Into<String>) -> Self {
        Expr::Src(name.into())
    }

    pub fn out(name: impl Into<String>) -> Self {
        Expr::Out(name.into())
    }

    pub fn slice(self, axis: u8, start: i64, len: i64) -> Self {
        self.slice_step(axis, start, len, 1)
    }

    pub fn slice_step(self, axis: u8, start: i64, len: i64, step: i64) -> Self {
        Expr::Slice {
            src: Box::new(self),
            axis: Axis(axis),
            start,
            len,
            step,
        }
    }

    pub fn cat(axis: u8, parts: Vec<Expr>) -> Self {
        Expr::Cat {
            axis: Axis(axis),
            parts,
        }
    }

    pub fn reshape(self, shape: Vec<i64>) -> Self {
        Expr::Reshape {
            src: Box::new(self),
            shape,
        }
    }

    pub fn pad(self, axis: u8, before: i64, after: i64) -> Self {
        Expr::Pad {
            src: Box::new(self),
            axis: Axis(axis),
            before,
            after,
        }
    }

    pub fn repack(self, spec: RepackSpec, out: TensorType) -> Self {
        Expr::Repack {
            src: Box::new(self),
            spec,
            out,
        }
    }

    pub fn bitcast(self, out: TensorType) -> Self {
        Expr::Bitcast {
            src: Box::new(self),
            out,
        }
    }

    pub fn quantize(self, spec: QuantSpec) -> Self {
        Expr::Quantize {
            src: Box::new(self),
            spec,
        }
    }

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
            Expr::Src(_) | Expr::Out(_) => true,
            Expr::Slice { src, .. } | Expr::Reshape { src, .. } | Expr::Pad { src, .. } => {
                src.is_affine()
            }
            Expr::Cat { parts, .. } => parts.iter().all(Expr::is_affine),
            Expr::Bitcast { .. } => true,
            Expr::Shard { src, .. } => src.is_affine(),
            Expr::Repack { .. } | Expr::Quantize { .. } => false,
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

    fn visit<'a>(&'a self, seen: &mut impl FnMut(&'a Expr)) {
        seen(self);
        match self {
            Expr::Src(_) | Expr::Out(_) => {}
            Expr::Slice { src, .. }
            | Expr::Reshape { src, .. }
            | Expr::Pad { src, .. }
            | Expr::Repack { src, .. }
            | Expr::Bitcast { src, .. }
            | Expr::Shard { src, .. }
            | Expr::Quantize { src, .. } => src.visit(seen),
            Expr::Cat { parts, .. } => {
                for part in parts {
                    part.visit(seen);
                }
            }
        }
    }
}
