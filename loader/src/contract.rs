//! The model contract: a declaration of what the driver needs, as expressions
//! over the checkpoint's byte space.
//!
//! See `loader/spec.md` for the design rationale. In short: the driver declares
//! every tensor it wants and where its bytes come from; the loader decides how
//! to move them. Neither side needs to know the model family.
//!
//! This module owns the *declaration* and its type checker. Lowering an [`Expr`]
//! to a byte-span program is a separate concern (see `contract/compile.rs`).

use serde::{Deserialize, Serialize};

use crate::error::CompileError;
use crate::types::{Axis, DType, Encoding, QuantSpec, RepackSpec};

pub mod compile;

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
    pub fn element_count(&self) -> Result<i64, CompileError> {
        self.shape.iter().try_fold(1_i64, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| {
                CompileError::InvalidInput(format!("shape {:?} overflows i64", self.shape))
            })
        })
    }

    /// Size in bytes. Errors when a sub-byte encoding does not fill whole bytes.
    pub fn byte_size(&self) -> Result<u64, CompileError> {
        crate::types::encoding_nbytes(&self.shape, &self.encoding).ok_or_else(|| {
            CompileError::InvalidInput(format!(
                "shape {:?} of {:?} has no whole-byte size",
                self.shape, self.encoding
            ))
        })
    }
}

/// One declared tensor.
///
/// `shape` and `encoding` are derivable from `expr`; they are declared anyway
/// and checked, so that a contract is simultaneously a request and a proof
/// obligation. A driver whose model of the checkpoint is wrong fails to compile
/// instead of silently binding a plausible-looking buffer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorContract {
    pub name: String,
    pub expr: Expr,
    pub shape: Vec<i64>,
    pub encoding: Encoding,
}

impl TensorContract {
    pub fn new(name: impl Into<String>, expr: Expr, shape: Vec<i64>, encoding: Encoding) -> Self {
        Self {
            name: name.into(),
            expr,
            shape,
            encoding,
        }
    }

    /// The type this contract claims to have.
    pub fn declared(&self) -> TensorType {
        TensorType {
            shape: self.shape.clone(),
            encoding: self.encoding.clone(),
        }
    }
}

/// Everything one driver rank needs, as a name-resolved DAG.
///
/// `tensors` is in declaration order; [`Expr::Out`] may only name an earlier
/// entry, which makes the DAG acyclic by construction and lets the checker run
/// in one pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelContract {
    pub abi_version: u32,
    /// Byte alignment every materialized buffer must satisfy. A target
    /// property; there is no reason to repeat it per tensor.
    pub alignment: u32,
    pub tensors: Vec<TensorContract>,
}

impl ModelContract {
    /// Keep only the entries `retain` selects, plus everything they read.
    ///
    /// A contract is a DAG, so selecting an output selects its dependencies
    /// too; the closure is taken until it stops growing. This is how a caller
    /// asks for part of a model without authoring a second contract for it.
    pub fn retain_outputs(
        mut self,
        mut retain: impl FnMut(&str) -> bool,
    ) -> Result<Self, CompileError> {
        let mut selected = self
            .tensors
            .iter()
            .map(|contract| retain(&contract.name))
            .collect::<Vec<_>>();
        let by_name: std::collections::HashMap<&str, usize> = self
            .tensors
            .iter()
            .enumerate()
            .map(|(index, contract)| (contract.name.as_str(), index))
            .collect();
        loop {
            let mut changed = false;
            for index in 0..self.tensors.len() {
                if !selected[index] {
                    continue;
                }
                for name in self.tensors[index].expr.outputs() {
                    let dep = *by_name.get(name).ok_or_else(|| {
                        CompileError::InvalidInput(format!(
                            "contract {index} reads missing contract '{name}'"
                        ))
                    })?;
                    if !selected[dep] {
                        selected[dep] = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if !selected.iter().any(|selected| *selected) {
            return Err(CompileError::InvalidInput(
                "the component filter selected no tensors".to_string(),
            ));
        }
        let mut keep = selected.into_iter();
        self.tensors.retain(|_| keep.next().unwrap_or(false));
        Ok(self)
    }
}

/// Resolves [`Expr::Src`] names against a checkpoint.
pub trait CheckpointTypes {
    fn tensor_type(&self, name: &str) -> Option<TensorType>;
}

/// What resolving a contract's expressions turned up: the checkpoint tensors
/// they consulted, and the types the earlier entries published.
///
/// The compiler needs the same name→shape resolution the resolver just did, so
/// the resolver hands it over rather than making it look everything up twice.
#[derive(Clone, Debug, Default)]
pub struct Checked {
    /// Checkpoint tensors referenced by some [`Expr::Src`], by name.
    pub sources: std::collections::HashMap<String, TensorType>,
    /// Declared contracts, by name.
    pub outputs: std::collections::HashMap<String, TensorType>,
}

impl Checked {
    pub fn source(&self, name: &str) -> Option<&TensorType> {
        self.sources.get(name)
    }

    pub fn output(&self, name: &str) -> Option<&TensorType> {
        self.outputs.get(name)
    }
}

struct Scope<'a> {
    checkpoint: &'a dyn CheckpointTypes,
    resolved: Checked,
}

/// Type-check a standalone expression, with no prior contracts in scope.
///
/// Returns the inferred type alongside the resolution the compiler needs. Used
/// by builders that would rather derive a shape than declare it, and by tests.
pub fn infer_type(
    expr: &Expr,
    checkpoint: &dyn CheckpointTypes,
) -> Result<(TensorType, Checked), CompileError> {
    let mut resolver = Resolver::new(checkpoint);
    let ty = resolver.infer(expr)?;
    Ok((ty, resolver.into_checked()))
}

/// A scope built up one entry at a time.
///
/// [`ModelContract::check`] resolves a whole declaration in one call, which is
/// what a declared contract wants. A *compiler* wants to interleave: check an
/// expression, lower it, publish what it produced, then move to the next entry
/// with that name in scope. This exposes the same machinery in that order.
pub struct Resolver<'a> {
    scope: Scope<'a>,
}

impl<'a> Resolver<'a> {
    pub fn new(checkpoint: &'a dyn CheckpointTypes) -> Self {
        Self {
            scope: Scope {
                checkpoint,
                resolved: Checked::default(),
            },
        }
    }

    /// Infer `expr`'s type, resolving [`Expr::Out`] against what has been
    /// published so far.
    pub fn infer(&mut self, expr: &Expr) -> Result<TensorType, CompileError> {
        infer(expr, &mut self.scope)
    }

    /// Rewrite `expr` for one rank of a `world`-way tensor-parallel split,
    /// replacing every [`Expr::Shard`] with the slice that rank reads.
    ///
    /// `what` names the thing being specialized and appears in the divisibility
    /// error, which is what a user sees when a tp_size does not fit the model.
    pub fn specialize(
        &mut self,
        expr: Expr,
        rank: u32,
        world: u32,
        what: &str,
    ) -> Result<Expr, CompileError> {
        specialize(expr, &mut self.scope, rank, world, what)
    }

    /// Bring `name` into scope for later expressions.
    pub fn publish(&mut self, name: &str, ty: TensorType) {
        self.scope.resolved.outputs.insert(name.to_string(), ty);
    }

    pub fn checked(&self) -> &Checked {
        &self.scope.resolved
    }

    pub fn into_checked(self) -> Checked {
        self.scope.resolved
    }
}

/// Infer the type of `expr`, resolving names through `scope`.
fn infer(expr: &Expr, scope: &mut Scope<'_>) -> Result<TensorType, CompileError> {
    match expr {
        Expr::Src(name) => {
            let ty = scope.checkpoint.tensor_type(name).ok_or_else(|| {
                CompileError::InvalidInput(format!("checkpoint has no tensor named '{name}'"))
            })?;
            scope.resolved.sources.insert(name.clone(), ty.clone());
            Ok(ty)
        }
        Expr::Out(name) => scope.resolved.outputs.get(name).cloned().ok_or_else(|| {
            CompileError::InvalidInput(format!(
                "no contract named '{name}' is declared before this one"
            ))
        }),
        Expr::Slice {
            src,
            axis,
            start,
            len,
            step,
        } => {
            let ty = infer(src, scope)?;
            infer_slice(&ty, *axis, *start, *len, *step)
        }
        Expr::Cat { axis, parts } => {
            let mut types = Vec::with_capacity(parts.len());
            for part in parts {
                types.push(infer(part, scope)?);
            }
            infer_cat(*axis, &types)
        }
        Expr::Reshape { src, shape } => {
            let ty = infer(src, scope)?;
            infer_reshape(&ty, shape)
        }
        Expr::Pad {
            src,
            axis,
            before,
            after,
        } => {
            let ty = infer(src, scope)?;
            infer_pad(&ty, *axis, *before, *after)
        }
        Expr::Repack { src, out, .. } => {
            // Opaque by construction: we still evaluate the operand so that a
            // broken subtree is reported, but the layout transform declares its
            // own result.
            infer(src, scope)?;
            Ok(out.clone())
        }
        Expr::Quantize { src, spec } => {
            let ty = infer(src, scope)?;
            infer_quantize(&ty, spec)
        }
        Expr::Bitcast { src, out } => {
            let ty = infer(src, scope)?;
            let from = ty.byte_size()?;
            let to = out.byte_size()?;
            if from != to {
                return Err(CompileError::InvalidInput(format!(
                    "Bitcast changes the byte size, {from} -> {to}"
                )));
            }
            if !matches!(src.as_ref(), Expr::Src(_) | Expr::Out(_)) {
                return Err(CompileError::InvalidInput(
                    "Bitcast may only reinterpret a whole tensor".to_string(),
                ));
            }
            Ok(out.clone())
        }
        Expr::Shard { .. } => Err(CompileError::Internal(
            "Shard has no type until it is specialized against a target; call \
             Resolver::specialize first"
                .to_string(),
        )),
    }
}

/// Rewrite every [`Expr::Shard`] in `expr` into the concrete slice `rank` reads,
/// resolving extents through `scope`.
///
/// Children first, so that a shard over a shard — or over a [`Expr::Cat`] whose
/// legs are themselves sharded — sees an operand it can already type.
fn specialize(
    expr: Expr,
    scope: &mut Scope<'_>,
    rank: u32,
    world: u32,
    what: &str,
) -> Result<Expr, CompileError> {
    macro_rules! go {
        ($src:expr) => {
            Box::new(specialize(*$src, scope, rank, world, what)?)
        };
    }
    Ok(match expr {
        Expr::Src(_) | Expr::Out(_) => expr,
        Expr::Slice {
            src,
            axis,
            start,
            len,
            step,
        } => Expr::Slice {
            src: go!(src),
            axis,
            start,
            len,
            step,
        },
        Expr::Reshape { src, shape } => Expr::Reshape {
            src: go!(src),
            shape,
        },
        Expr::Pad {
            src,
            axis,
            before,
            after,
        } => Expr::Pad {
            src: go!(src),
            axis,
            before,
            after,
        },
        Expr::Repack { src, spec, out } => Expr::Repack {
            src: go!(src),
            spec,
            out,
        },
        Expr::Quantize { src, spec } => Expr::Quantize {
            src: go!(src),
            spec,
        },
        Expr::Bitcast { src, out } => Expr::Bitcast { src: go!(src), out },
        Expr::Cat { axis, parts } => Expr::Cat {
            axis,
            parts: parts
                .into_iter()
                .map(|part| specialize(part, scope, rank, world, what))
                .collect::<Result<_, _>>()?,
        },
        Expr::Shard { src, axis } => {
            let src = specialize(*src, scope, rank, world, what)?;
            if world <= 1 {
                // Not a degenerate one-rank slice but the operand itself, so
                // that a single-GPU plan is identical to one compiled from a
                // contract that never mentioned sharding.
                return Ok(src);
            }
            let ty = infer(&src, scope)?;
            let index = axis_index(axis, ty.shape.len(), what)?;
            let extent = ty.shape[index];
            if extent % i64::from(world) != 0 {
                // Named, because this is the message a user gets for "tp_size
                // does not divide this model". The driver used to pre-empt it
                // with its own per-family divisibility table over `config.json`,
                // which was the same fact checked twice and reachable only for
                // the families someone had listed.
                return Err(CompileError::InvalidInput(format!(
                    "'{what}' has {extent} along axis {index}, which tp_size {world} does not \
                     divide; use a tp_size that divides it or run single-GPU"
                )));
            }
            let len = extent / i64::from(world);
            src.slice(axis.0, i64::from(rank) * len, len)
        }
    })
}

/// Resolve an [`Axis`] against a rank, rejecting out-of-range axes.
fn axis_index(axis: Axis, rank: usize, what: &str) -> Result<usize, CompileError> {
    let index = usize::from(axis.0);
    if index >= rank {
        return Err(CompileError::InvalidInput(format!(
            "{what} names axis {index} of a rank-{rank} tensor"
        )));
    }
    Ok(index)
}

/// The group size along `axis`, when the encoding blocks that axis.
///
/// Quantized encodings pack `group_size` consecutive elements along
/// `channel_axis` under one scale, so any structural operation on that axis must
/// respect the block boundary or the scales stop lining up with the data.
fn block_granularity(encoding: &Encoding, axis: usize) -> Option<i64> {
    let Encoding::Quant(spec) = encoding else {
        return None;
    };
    let channel = spec.channel_axis?;
    if usize::from(channel.0) != axis {
        return None;
    }
    let group = i64::from(spec.normalized_group_size());
    (group > 1).then_some(group)
}

fn infer_slice(
    ty: &TensorType,
    axis: Axis,
    start: i64,
    len: i64,
    step: i64,
) -> Result<TensorType, CompileError> {
    let index = axis_index(axis, ty.rank(), "Slice")?;
    if step < 1 {
        return Err(CompileError::InvalidInput(format!(
            "Slice step must be >= 1, got {step}"
        )));
    }
    if len < 1 {
        return Err(CompileError::InvalidInput(format!(
            "Slice len must be >= 1, got {len}"
        )));
    }
    if start < 0 {
        return Err(CompileError::InvalidInput(format!(
            "Slice start must be >= 0, got {start}"
        )));
    }
    let extent = ty.shape[index];
    let last = start
        .checked_add(
            len.checked_sub(1)
                .and_then(|n| n.checked_mul(step))
                .ok_or_else(|| {
                    CompileError::InvalidInput("Slice extent overflows i64".to_string())
                })?,
        )
        .ok_or_else(|| CompileError::InvalidInput("Slice extent overflows i64".to_string()))?;
    if last >= extent {
        return Err(CompileError::InvalidInput(format!(
            "Slice reads index {last} of axis {index}, which has extent {extent}"
        )));
    }
    if let Some(group) = block_granularity(&ty.encoding, index) {
        if step != 1 {
            return Err(CompileError::InvalidInput(format!(
                "Slice with step {step} on quantized axis {index} would split its {group}-element groups"
            )));
        }
        if start % group != 0 || len % group != 0 {
            return Err(CompileError::InvalidInput(format!(
                "Slice [{start}, {len}) on quantized axis {index} is not aligned to its {group}-element groups"
            )));
        }
    }
    let mut shape = ty.shape.clone();
    shape[index] = len;
    Ok(TensorType {
        shape,
        encoding: ty.encoding.clone(),
    })
}

fn infer_cat(axis: Axis, parts: &[TensorType]) -> Result<TensorType, CompileError> {
    let Some((head, tail)) = parts.split_first() else {
        return Err(CompileError::InvalidInput(
            "Cat needs at least one part".to_string(),
        ));
    };
    let index = axis_index(axis, head.rank(), "Cat")?;
    let mut total = head.shape[index];
    for (offset, part) in tail.iter().enumerate() {
        if part.rank() != head.rank() {
            return Err(CompileError::InvalidInput(format!(
                "Cat part {} has rank {} but part 0 has rank {}",
                offset + 1,
                part.rank(),
                head.rank()
            )));
        }
        for (other, (lhs, rhs)) in head.shape.iter().zip(part.shape.iter()).enumerate() {
            if other != index && lhs != rhs {
                return Err(CompileError::InvalidInput(format!(
                    "Cat on axis {index}: part {} has shape {:?}, incompatible with part 0's {:?}",
                    offset + 1,
                    part.shape,
                    head.shape
                )));
            }
        }
        if crate::types::normalize_encoding(&part.encoding)
            != crate::types::normalize_encoding(&head.encoding)
        {
            return Err(CompileError::InvalidInput(format!(
                "Cat part {} is encoded as {:?}, incompatible with part 0's {:?}",
                offset + 1,
                part.encoding,
                head.encoding
            )));
        }
        total = total
            .checked_add(part.shape[index])
            .ok_or_else(|| CompileError::InvalidInput("Cat extent overflows i64".to_string()))?;
    }
    if let Some(group) = block_granularity(&head.encoding, index) {
        for (offset, part) in parts.iter().enumerate() {
            if part.shape[index] % group != 0 {
                return Err(CompileError::InvalidInput(format!(
                    "Cat part {offset} contributes {} elements to quantized axis {index}, which is not a multiple of its {group}-element groups",
                    part.shape[index]
                )));
            }
        }
    }
    let mut shape = head.shape.clone();
    shape[index] = total;
    Ok(TensorType {
        shape,
        encoding: head.encoding.clone(),
    })
}

fn infer_reshape(ty: &TensorType, requested: &[i64]) -> Result<TensorType, CompileError> {
    if requested.is_empty() {
        return Err(CompileError::InvalidInput(
            "Reshape needs at least one extent".to_string(),
        ));
    }
    if matches!(ty.encoding, Encoding::Quant(_)) {
        // A block-quantized tensor's element order is tied to its block
        // structure, so a row-major reinterpretation is not generally a byte
        // identity. No production case needs it; see spec.md §8.
        return Err(CompileError::InvalidInput(format!(
            "Reshape of a quantized tensor ({:?}) is not supported",
            ty.encoding
        )));
    }
    let total = ty.element_count()?;
    let mut wildcard = None;
    let mut known = 1_i64;
    for (index, extent) in requested.iter().enumerate() {
        match *extent {
            -1 if wildcard.is_some() => {
                return Err(CompileError::InvalidInput(
                    "Reshape allows at most one -1 extent".to_string(),
                ));
            }
            -1 => wildcard = Some(index),
            extent if extent < 1 => {
                return Err(CompileError::InvalidInput(format!(
                    "Reshape extent {extent} must be >= 1 or -1"
                )));
            }
            extent => {
                known = known.checked_mul(extent).ok_or_else(|| {
                    CompileError::InvalidInput("Reshape extent overflows i64".to_string())
                })?;
            }
        }
    }
    let mut shape = requested.to_vec();
    match wildcard {
        Some(index) if known > 0 && total % known == 0 => shape[index] = total / known,
        Some(_) => {
            return Err(CompileError::InvalidInput(format!(
                "Reshape of {:?} ({total} elements) to {requested:?} does not divide evenly",
                ty.shape
            )));
        }
        None if known == total => {}
        None => {
            return Err(CompileError::InvalidInput(format!(
                "Reshape of {:?} ({total} elements) to {requested:?} ({known} elements) changes the element count",
                ty.shape
            )));
        }
    }
    Ok(TensorType {
        shape,
        encoding: ty.encoding.clone(),
    })
}

fn infer_pad(
    ty: &TensorType,
    axis: Axis,
    before: i64,
    after: i64,
) -> Result<TensorType, CompileError> {
    let index = axis_index(axis, ty.rank(), "Pad")?;
    if before < 0 || after < 0 {
        return Err(CompileError::InvalidInput(format!(
            "Pad amounts must be >= 0, got before={before} after={after}"
        )));
    }
    if before == 0 && after == 0 {
        return Err(CompileError::InvalidInput(
            "Pad by zero on both sides has no effect; omit it".to_string(),
        ));
    }
    if let Some(group) = block_granularity(&ty.encoding, index)
        && (before % group != 0 || after % group != 0)
    {
        return Err(CompileError::InvalidInput(format!(
            "Pad by ({before}, {after}) on quantized axis {index} is not aligned to its {group}-element groups"
        )));
    }
    let extent = ty.shape[index]
        .checked_add(before)
        .and_then(|sum| sum.checked_add(after))
        .ok_or_else(|| CompileError::InvalidInput("Pad extent overflows i64".to_string()))?;
    let mut shape = ty.shape.clone();
    shape[index] = extent;
    Ok(TensorType {
        shape,
        encoding: ty.encoding.clone(),
    })
}

fn infer_quantize(ty: &TensorType, spec: &QuantSpec) -> Result<TensorType, CompileError> {
    if matches!(ty.encoding, Encoding::Quant(_)) {
        return Err(CompileError::InvalidInput(format!(
            "Quantize of an already-quantized tensor ({:?}) is not supported",
            ty.encoding
        )));
    }
    let spec = spec.clone().normalized();
    if let Some(channel) = spec.channel_axis {
        let index = axis_index(channel, ty.rank(), "Quantize channel_axis")?;
        let group = i64::from(spec.normalized_group_size());
        if group > 1 && ty.shape[index] % group != 0 {
            return Err(CompileError::InvalidInput(format!(
                "Quantize groups axis {index} by {group}, but its extent is {}",
                ty.shape[index]
            )));
        }
    }
    Ok(TensorType {
        shape: ty.shape.clone(),
        encoding: Encoding::Quant(spec),
    })
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

    /// Whether any part of this expression is partitioned across ranks, and so
    /// means something different on each of them.
    pub fn is_sharded(&self) -> bool {
        let mut found = false;
        self.visit(&mut |expr| found |= matches!(expr, Expr::Shard { .. }));
        found
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::QuantScheme;
    use std::collections::HashMap;

    struct FakeCheckpoint(HashMap<String, TensorType>);

    impl FakeCheckpoint {
        fn new(entries: &[(&str, &[i64], DType)]) -> Self {
            Self(
                entries
                    .iter()
                    .map(|(name, shape, dtype)| {
                        ((*name).to_string(), TensorType::raw(shape.to_vec(), *dtype))
                    })
                    .collect(),
            )
        }
    }

    impl CheckpointTypes for FakeCheckpoint {
        fn tensor_type(&self, name: &str) -> Option<TensorType> {
            self.0.get(name).cloned()
        }
    }

    /// Qwen3-1.7B attention projections, measured from the real checkpoint.
    fn qwen3() -> FakeCheckpoint {
        FakeCheckpoint::new(&[
            ("q_proj", &[2048, 2048], DType::BF16),
            ("k_proj", &[1024, 2048], DType::BF16),
            ("v_proj", &[1024, 2048], DType::BF16),
        ])
    }

    fn check_one(expr: Expr, checkpoint: &dyn CheckpointTypes) -> Result<TensorType, CompileError> {
        infer_type(&expr, checkpoint).map(|(ty, _)| ty)
    }

    fn fp8_per_row() -> QuantSpec {
        QuantSpec {
            scheme: QuantScheme::Fp8E4M3,
            logical_dtype: DType::F8E4M3,
            bits_per_element: 8,
            group_size: 1,
            channel_axis: Some(Axis(0)),
            scale_dtype: Some(DType::F32),
            zero_point_dtype: None,
            block_shape: Vec::new(),
        }
    }

    fn specialize_one(expr: Expr, rank: u32, world: u32) -> Result<Expr, CompileError> {
        let checkpoint = qwen3();
        Resolver::new(&checkpoint).specialize(expr, rank, world, "q")
    }

    #[test]
    fn a_component_filter_keeps_what_the_selection_reads() {
        let one = |name: &str, expr: Expr| {
            TensorContract::new(name, expr, vec![1], Encoding::Raw(DType::U8))
        };
        let contract = ModelContract {
            abi_version: 1,
            alignment: 1,
            tensors: vec![
                one("text.weight", Expr::src("text")),
                one("vision.base", Expr::src("vision")),
                one("vision.view", Expr::out("vision.base").slice(0, 0, 1)),
            ],
        };

        let filtered = contract
            .retain_outputs(|name| name == "vision.view")
            .unwrap();
        assert_eq!(filtered.tensors.len(), 2);
        assert_eq!(filtered.tensors[0].name, "vision.base");
        assert_eq!(filtered.tensors[1].name, "vision.view");
    }

    #[test]
    fn shard_has_no_type_until_it_is_specialized() {
        let err = check_one(Expr::src("q_proj").shard(0), &qwen3()).unwrap_err();
        assert!(format!("{err}").contains("specialize"), "{err}");
    }

    #[test]
    fn shard_becomes_this_ranks_slice() {
        let expr = specialize_one(Expr::src("q_proj").shard(0), 3, 4).unwrap();
        assert_eq!(expr, Expr::src("q_proj").slice(0, 1536, 512));
        assert_eq!(
            check_one(expr, &qwen3()).unwrap(),
            TensorType::raw(vec![512, 2048], DType::BF16)
        );
    }

    #[test]
    fn a_single_rank_shard_is_the_tensor_itself() {
        // Not a degenerate full-width slice: a one-GPU plan must be identical
        // to one compiled from a contract that never mentioned sharding.
        let expr = specialize_one(Expr::src("q_proj").shard(0), 0, 1).unwrap();
        assert_eq!(expr, Expr::src("q_proj"));
    }

    #[test]
    fn shard_composes_under_the_affine_fragment() {
        let expr = specialize_one(
            Expr::cat(
                0,
                vec![Expr::src("k_proj").shard(0), Expr::src("v_proj").shard(0)],
            ),
            1,
            2,
        )
        .unwrap();
        assert_eq!(
            check_one(expr, &qwen3()).unwrap(),
            TensorType::raw(vec![1024, 2048], DType::BF16)
        );
    }

    #[test]
    fn an_indivisible_extent_is_rejected_by_name() {
        let err = specialize_one(Expr::src("k_proj").shard(0), 0, 3).unwrap_err();
        let message = format!("{err}");
        assert!(message.contains("'q' has 1024 along axis 0"), "{message}");
        assert!(message.contains("tp_size 3"), "{message}");
    }

    #[test]
    fn src_resolves_to_checkpoint_type() {
        let ty = check_one(Expr::src("q_proj"), &qwen3()).unwrap();
        assert_eq!(ty, TensorType::raw(vec![2048, 2048], DType::BF16));
    }

    #[test]
    fn unknown_src_is_rejected() {
        let err = check_one(Expr::src("nope"), &qwen3()).unwrap_err();
        assert!(err.to_string().contains("no tensor named 'nope'"));
    }

    #[test]
    fn cat_sums_the_joined_axis() {
        let ty = check_one(
            Expr::cat(
                0,
                vec![
                    Expr::src("q_proj"),
                    Expr::src("k_proj"),
                    Expr::src("v_proj"),
                ],
            ),
            &qwen3(),
        )
        .unwrap();
        assert_eq!(ty.shape, vec![4096, 2048]);
    }

    #[test]
    fn cat_rejects_mismatched_other_axes() {
        let checkpoint =
            FakeCheckpoint::new(&[("a", &[4, 8], DType::BF16), ("b", &[4, 9], DType::BF16)]);
        let err = check_one(
            Expr::cat(0, vec![Expr::src("a"), Expr::src("b")]),
            &checkpoint,
        )
        .unwrap_err();
        assert!(err.to_string().contains("incompatible"));
    }

    #[test]
    fn cat_rejects_mismatched_encodings() {
        let checkpoint =
            FakeCheckpoint::new(&[("a", &[4, 8], DType::BF16), ("b", &[4, 8], DType::F32)]);
        let err = check_one(
            Expr::cat(0, vec![Expr::src("a"), Expr::src("b")]),
            &checkpoint,
        )
        .unwrap_err();
        assert!(err.to_string().contains("encoded as"));
    }

    #[test]
    fn tp_shard_then_fuse_composes() {
        // TP=2 rank 0: half of each projection, concatenated.
        let ty = check_one(
            Expr::cat(
                0,
                vec![
                    Expr::src("q_proj").slice(0, 0, 1024),
                    Expr::src("k_proj").slice(0, 0, 512),
                    Expr::src("v_proj").slice(0, 0, 512),
                ],
            ),
            &qwen3(),
        )
        .unwrap();
        assert_eq!(ty.shape, vec![2048, 2048]);
    }

    #[test]
    fn slice_rejects_reads_past_the_end() {
        let err = check_one(Expr::src("k_proj").slice(0, 512, 1024), &qwen3()).unwrap_err();
        assert!(err.to_string().contains("extent 1024"));
    }

    #[test]
    fn strided_slice_selects_every_other_row() {
        let checkpoint = FakeCheckpoint::new(&[("gate_up", &[512, 64], DType::BF16)]);
        let gate = check_one(Expr::src("gate_up").slice_step(0, 0, 256, 2), &checkpoint).unwrap();
        let up = check_one(Expr::src("gate_up").slice_step(0, 1, 256, 2), &checkpoint).unwrap();
        assert_eq!(gate.shape, vec![256, 64]);
        assert_eq!(up.shape, vec![256, 64]);
        // One past the last selectable odd row.
        assert!(check_one(Expr::src("gate_up").slice_step(0, 1, 257, 2), &checkpoint).is_err());
    }

    #[test]
    fn expert_stack_via_reshape_and_cat() {
        let checkpoint = FakeCheckpoint::new(&[
            ("e0.gate", &[6144, 2048], DType::BF16),
            ("e1.gate", &[6144, 2048], DType::BF16),
        ]);
        let ty = check_one(
            Expr::cat(
                0,
                vec![
                    Expr::src("e0.gate").reshape(vec![1, 6144, 2048]),
                    Expr::src("e1.gate").reshape(vec![1, 6144, 2048]),
                ],
            ),
            &checkpoint,
        )
        .unwrap();
        assert_eq!(ty.shape, vec![2, 6144, 2048]);
    }

    #[test]
    fn reshape_infers_one_wildcard() {
        let ty = check_one(Expr::src("q_proj").reshape(vec![16, 128, -1]), &qwen3()).unwrap();
        assert_eq!(ty.shape, vec![16, 128, 2048]);
    }

    #[test]
    fn reshape_rejects_element_count_changes() {
        let err = check_one(Expr::src("q_proj").reshape(vec![2048, 2049]), &qwen3()).unwrap_err();
        assert!(err.to_string().contains("changes the element count"));
    }

    #[test]
    fn pad_extends_the_axis() {
        let ty = check_one(Expr::src("k_proj").pad(0, 0, 128), &qwen3()).unwrap();
        assert_eq!(ty.shape, vec![1152, 2048]);
    }

    #[test]
    fn quantize_keeps_shape_and_replaces_encoding() {
        let ty = check_one(Expr::src("q_proj").quantize(fp8_per_row()), &qwen3()).unwrap();
        assert_eq!(ty.shape, vec![2048, 2048]);
        assert!(matches!(ty.encoding, Encoding::Quant(_)));
    }

    #[test]
    fn quantize_rejects_a_ragged_group_axis() {
        let checkpoint = FakeCheckpoint::new(&[("w", &[64, 100], DType::BF16)]);
        let mxfp4 = QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
            scale_dtype: Some(DType::U8),
            zero_point_dtype: None,
            block_shape: vec![32],
        };
        let err = check_one(Expr::src("w").quantize(mxfp4), &checkpoint).unwrap_err();
        assert!(err.to_string().contains("groups axis 1 by 32"));
    }

    #[test]
    fn slicing_across_a_quant_group_is_rejected() {
        let checkpoint = FakeCheckpoint::new(&[("w", &[64, 128], DType::BF16)]);
        let mxfp4 = QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
            scale_dtype: Some(DType::U8),
            zero_point_dtype: None,
            block_shape: vec![32],
        };
        let quantized = Expr::src("w").quantize(mxfp4);
        // Aligned to the 32-element groups: fine.
        assert!(check_one(quantized.clone().slice(1, 32, 64), &checkpoint).is_ok());
        // Straddles them: rejected.
        let err = check_one(quantized.slice(1, 16, 64), &checkpoint).unwrap_err();
        assert!(err.to_string().contains("not aligned"));
    }

    #[test]
    fn affine_fragment_is_recognized() {
        let fused = Expr::cat(0, vec![Expr::src("q_proj"), Expr::src("k_proj")]);
        assert!(fused.is_affine());
        assert!(!fused.quantize(fp8_per_row()).is_affine());
    }

    #[test]
    fn sources_are_reported_in_order() {
        let fused = Expr::cat(
            0,
            vec![
                Expr::src("q_proj").slice(0, 0, 1024),
                Expr::src("k_proj"),
                Expr::out("bank"),
            ],
        );
        assert_eq!(fused.sources(), vec!["q_proj", "k_proj"]);
    }

    fn qkv_contract(shape: Vec<i64>) -> TensorContract {
        TensorContract::new(
            "qkv.fused",
            Expr::cat(
                0,
                vec![
                    Expr::src("q_proj"),
                    Expr::src("k_proj"),
                    Expr::src("v_proj"),
                ],
            ),
            shape,
            Encoding::Raw(DType::BF16),
        )
    }

    fn model(tensors: Vec<TensorContract>) -> ModelContract {
        ModelContract {
            abi_version: 1,
            alignment: 256,
            tensors,
        }
    }

    /// Resolve a run of contracts in declaration order, exactly as the frontend
    /// does: each entry's type is published under its name so that later
    /// entries can read it through [`Expr::Out`].
    fn resolve(
        tensors: &[TensorContract],
        checkpoint: &dyn CheckpointTypes,
    ) -> Result<Vec<TensorType>, CompileError> {
        let mut resolver = Resolver::new(checkpoint);
        tensors
            .iter()
            .map(|tensor| {
                let ty = resolver.infer(&tensor.expr)?;
                resolver.publish(&tensor.name, ty.clone());
                Ok(ty)
            })
            .collect()
    }

    #[test]
    fn a_contract_resolves_to_what_it_declares() {
        let tensors = vec![qkv_contract(vec![4096, 2048])];
        let found = resolve(&tensors, &qwen3()).unwrap();
        assert_eq!(found[0], tensors[0].declared());
    }

    #[test]
    fn a_bank_publishes_views_by_name() {
        let tensors = vec![
            qkv_contract(vec![4096, 2048]),
            TensorContract::new(
                "q_proj",
                Expr::out("qkv.fused").slice(0, 0, 2048),
                vec![2048, 2048],
                Encoding::Raw(DType::BF16),
            ),
            TensorContract::new(
                "k_proj",
                Expr::out("qkv.fused").slice(0, 2048, 1024),
                vec![1024, 2048],
                Encoding::Raw(DType::BF16),
            ),
        ];
        let found = resolve(&tensors, &qwen3()).unwrap();
        assert_eq!(found[1].shape, vec![2048, 2048]);
        assert_eq!(found[2].shape, vec![1024, 2048]);
    }

    #[test]
    fn out_cannot_name_a_later_contract() {
        let err = resolve(
            &[
                TensorContract::new(
                    "early",
                    Expr::out("late"),
                    vec![2048, 2048],
                    Encoding::Raw(DType::BF16),
                ),
                qkv_contract(vec![4096, 2048]),
            ],
            &qwen3(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("declared before this one"));
    }

    #[test]
    fn out_and_src_do_not_collide() {
        // "q_proj" is both a checkpoint tensor and a declared contract; Src and
        // Out must keep pointing at different things.
        let found = resolve(
            &[
                TensorContract::new(
                    "q_proj",
                    Expr::src("q_proj").pad(0, 0, 64),
                    vec![2112, 2048],
                    Encoding::Raw(DType::BF16),
                ),
                TensorContract::new(
                    "q_proj.head0",
                    Expr::out("q_proj").slice(0, 0, 128),
                    vec![128, 2048],
                    Encoding::Raw(DType::BF16),
                ),
            ],
            &qwen3(),
        )
        .unwrap();
        assert_eq!(found[0].shape, vec![2112, 2048]);
        assert_eq!(found[1].shape, vec![128, 2048]);
    }

    #[test]
    fn qwen3_tp2_rank0_fp8_is_one_expression() {
        // The worked example from spec.md §5: shard, fuse, and quantize in one
        // declaration. Today's abi/fusion.rs refuses all three combinations.
        let expr = Expr::cat(
            0,
            vec![
                Expr::src("q_proj").slice(0, 0, 1024),
                Expr::src("k_proj").slice(0, 0, 512),
                Expr::src("v_proj").slice(0, 0, 512),
            ],
        )
        .quantize(fp8_per_row());
        let contract = model(vec![TensorContract::new(
            "model.layers.0.self_attn.qkv_proj.fused.weight",
            expr,
            vec![2048, 2048],
            Encoding::Quant(fp8_per_row()),
        )]);
        assert_eq!(
            resolve(&contract.tensors, &qwen3()).unwrap()[0],
            contract.tensors[0].declared()
        );
    }
}
