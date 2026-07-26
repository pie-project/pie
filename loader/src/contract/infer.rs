//! The type checker: what an [`Expr`] denotes, at one point in a
//! tensor-parallel split.
//!
//! Inference answers *"what shape and encoding does this expression produce"*,
//! and refuses the expressions that do not denote anything — a `Concat` whose parts
//! disagree off the joined axis, a `Slice` that cuts a quantization group in
//! half, a `Reshape` that changes the element count. It is *total*: every
//! [`Expr`] variant has a type, including [`Expr::Shard`], which reads its
//! extent off the [`Partition`] the resolver was built with.
//!
//! Specialization is the rewrite that eliminates [`Expr::Shard`], replacing each
//! one with the concrete slice this rank reads. Lowering requires it, because
//! byte offsets cannot be symbolic; typing does not. Keeping it a rewrite rather
//! than a typing precondition is what leaves the algebra with one grammar
//! instead of two.
//!
//! A rank enters the algebra through [`local_range`] and nowhere else;
//! `architecture.md` §6.3 rests on that being true. Both the checker and the
//! specializer reach it through the same [`shard_range`], so a rank they
//! disagreed about is not expressible.

use crate::error::{Error, OrOverflow};
use crate::types::{Axis, DType, Encoding, RepackLayout, RepackSpec};

use super::compile;
use super::{Expr, Partition, ScaleFactor, TensorType, local_range, resolve_extents};

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

    /// The type behind a lowering leaf, whichever namespace it names.
    pub fn type_of(&self, leaf: &compile::Leaf) -> Option<&TensorType> {
        match leaf {
            compile::Leaf::Checkpoint(name) => self.source(name),
            compile::Leaf::Contract(name) => self.output(name),
        }
    }
}

struct Scope<'a> {
    checkpoint: &'a dyn CheckpointTypes,
    resolved: Checked,
    partition: Partition,
    /// Which instance of a [`GroupContract`](crate::contract::GroupContract) is
    /// being resolved, or `None` outside a group.
    ///
    /// The group's [`Partition`]: carried on the scope for the same reason, so
    /// that the type checker and the specializer cannot be handed different
    /// answers about which instance this is. `None` is what makes an index node
    /// outside a group a contract error rather than a silent instance 0.
    instance: Option<u32>,
    /// What the caller is resolving, for the message an indivisible axis
    /// produces. This is the text a user sees when a `tp_size` does not fit the
    /// model, so it wants to be a tensor name rather than a shape.
    what: String,
}

impl Scope<'_> {
    /// This instance's index, or the error an index node outside a group earns.
    fn instance(&self, node: &str) -> Result<i64, Error> {
        self.instance.map(i64::from).ok_or_else(|| {
            Error::Contract(format!(
                "{node} names a group instance, but '{}' is not declared inside \
                 a group",
                self.what
            ))
        })
    }
}

/// Substitute `index` for the single `{}` in `template`.
///
/// The whole of the template language, and it is this short on purpose. A
/// contract exists so that the loader never has to *parse* a checkpoint name —
/// naming is the author's business — and a template with formatting options
/// would be a small language whose evaluator is exactly that parser. One
/// placeholder, decimal, and every other use of a brace is rejected here rather
/// than passed through to become a tensor nobody can find.
pub fn substitute_index(template: &str, index: u32) -> Result<String, Error> {
    let braces = template.matches('{').count();
    if braces != 1 || template.matches('}').count() != 1 || !template.contains("{}") {
        return Err(Error::Contract(format!(
            "indexed source template '{template}' must contain exactly one '{{}}' \
             and no other brace"
        )));
    }
    Ok(template.replace("{}", &index.to_string()))
}

/// Type-check a standalone expression against the unsplit tensor.
///
/// Returns the inferred type alongside the resolution the compiler needs. Used
/// by builders that would rather derive a shape than declare it, and by tests.
pub fn infer_type(
    expr: &Expr,
    checkpoint: &dyn CheckpointTypes,
) -> Result<(TensorType, Checked), Error> {
    let mut resolver = Resolver::new(checkpoint, Partition::WHOLE);
    let ty = resolver.infer(expr, "expression")?;
    Ok((ty, resolver.into_checked()))
}

/// A scope built up one entry at a time.
///
/// The only way to type-check a contract, and incremental on purpose: the
/// compiler interleaves. It checks an expression, lowers it, publishes what
/// that produced, then moves to the next entry with the new name in scope. A
/// whole-contract `check(&ModelContract)` would be the more obvious shape, but
/// it would have no caller — the pass that wants it is the same pass that has
/// to lower each entry before the next one can be resolved.
///
/// Built for one [`Partition`], because that is what makes typing total: a
/// resolver that did not know the split would have nothing to say about a
/// [`Expr::Shard`], and the hole would have to be paid for by every consumer.
pub struct Resolver<'a> {
    scope: Scope<'a>,
}

impl<'a> Resolver<'a> {
    pub fn new(checkpoint: &'a dyn CheckpointTypes, partition: Partition) -> Self {
        Self {
            scope: Scope {
                checkpoint,
                resolved: Checked::default(),
                partition,
                instance: None,
                what: String::new(),
            },
        }
    }

    /// The same resolver, bound to one instance of a
    /// [`GroupContract`](crate::contract::GroupContract).
    ///
    /// [`Expr::SrcIndexed`] and [`Expr::Select`] resolve against `instance` the
    /// way [`Expr::Shard`] resolves against the partition.
    pub fn for_instance(mut self, instance: u32) -> Self {
        self.scope.instance = Some(instance);
        self
    }

    /// Infer `expr`'s type, resolving [`Expr::Out`] against what has been
    /// published so far.
    ///
    /// `what` names the thing being resolved and appears in the divisibility
    /// error a [`Expr::Shard`] can raise.
    pub fn infer(&mut self, expr: &Expr, what: &str) -> Result<TensorType, Error> {
        self.scope.what.clear();
        self.scope.what.push_str(what);
        infer(expr, &mut self.scope)
    }

    /// Rewrite `expr` for this resolver's rank, replacing every
    /// [`Expr::Shard`] with the slice that rank reads.
    ///
    /// Lowering's precondition, not typing's: [`Resolver::infer`] types a
    /// `Shard` directly, but a byte offset cannot be symbolic.
    pub fn specialize(&mut self, expr: Expr, what: &str) -> Result<Expr, Error> {
        self.scope.what.clear();
        self.scope.what.push_str(what);
        specialize(expr, &mut self.scope)
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
fn infer(expr: &Expr, scope: &mut Scope<'_>) -> Result<TensorType, Error> {
    match expr {
        Expr::Src(name) => {
            let ty = scope.checkpoint.tensor_type(name).ok_or_else(|| {
                Error::Checkpoint(format!("checkpoint has no tensor named '{name}'"))
            })?;
            scope.resolved.sources.insert(name.clone(), ty.clone());
            Ok(ty)
        }
        Expr::Out(name) => scope.resolved.outputs.get(name).cloned().ok_or_else(|| {
            Error::Contract(format!(
                "no contract named '{name}' is declared before this one"
            ))
        }),
        Expr::SrcIndexed(template) => {
            let index = scope.instance("SrcIndexed")?;
            let name = substitute_index(template, index as u32)?;
            let ty = scope.checkpoint.tensor_type(&name).ok_or_else(|| {
                Error::Checkpoint(format!(
                    "checkpoint has no tensor named '{name}' (instance {index} of \
                     template '{template}')"
                ))
            })?;
            scope.resolved.sources.insert(name, ty.clone());
            Ok(ty)
        }
        Expr::Select {
            src,
            axis,
            stride,
            len,
        } => {
            let ty = infer(src, scope)?;
            // Typed at instance 0, because a `Select`'s type is the one thing
            // about it that does not depend on the index -- which is the
            // property that makes its instances interchangeable. The instance
            // that *is* being resolved is checked against the extent by
            // `specialize`, which knows the concrete start.
            let _ = stride;
            infer_slice(&ty, *axis, 0, *len)
        }
        Expr::Fill { value, ty } => infer_fill(*value, ty),
        Expr::Slice {
            src,
            axis,
            start,
            len,
        } => {
            let ty = infer(src, scope)?;
            infer_slice(&ty, *axis, *start, *len)
        }
        Expr::Stride {
            src,
            axis,
            start,
            len,
            step,
        } => {
            let ty = infer(src, scope)?;
            infer_stride(&ty, *axis, *start, *len, *step)
        }
        Expr::Gather { src, axis, indices } => {
            let ty = infer(src, scope)?;
            infer_gather(&ty, *axis, indices)
        }
        Expr::Concat { axis, parts } => {
            let mut types = Vec::with_capacity(parts.len());
            for part in parts {
                types.push(infer(part, scope)?);
            }
            infer_concat(*axis, &types)
        }
        Expr::Transmute { src, to } => {
            let ty = infer(src, scope)?;
            infer_transmute(&ty, to, src)
        }
        Expr::Repack { src, layout, to } => {
            let ty = infer(src, scope)?;
            repack_spec(&ty, *layout, to)?;
            Ok(to.clone())
        }
        Expr::Cast { src, to } => {
            let ty = infer(src, scope)?;
            infer_cast(&ty, to)
        }
        Expr::Scale { src, factor } => {
            let ty = infer(src, scope)?;
            match factor {
                ScaleFactor::Uniform(bits) => infer_scale(ty, *bits),
                ScaleFactor::PerBlock { by } => {
                    // The kernel reads the factors out of memory, so they have
                    // to be a tensor that exists rather than an expression the
                    // multiply would have to evaluate on its way past. Declare
                    // them -- `Visibility::Internal` if the driver has no use
                    // for them -- and scale by that name.
                    if !matches!(by.as_ref(), Expr::Out(_)) {
                        return Err(Error::Contract(
                            "Scale factors must be a declared tensor; declare \
                             them first and scale by that name"
                                .to_string(),
                        ));
                    }
                    let by = infer(by, scope)?;
                    infer_scale_per_block(ty, by)
                }
            }
        }
        Expr::Shard { src, axis } => {
            let ty = infer(src, scope)?;
            let (start, len) = shard_range(&ty, *axis, scope.partition, &scope.what)?;
            // Routed through `infer_slice` rather than rewriting the extent
            // directly, so that a shard is checked by everything a slice is —
            // in particular the quantization-group alignment that decides
            // whether this rank's band still lines up with its scales.
            infer_slice(&ty, *axis, start, len)
        }
    }
}

/// The band of `ty`'s `axis` that `partition` owns.
///
/// The one place [`Expr::Shard`] is given meaning. Both the checker and
/// [`specialize`] come through here, so the extent one states and the slice the
/// other emits are the same arithmetic by construction.
fn shard_range(
    ty: &TensorType,
    axis: Axis,
    partition: Partition,
    what: &str,
) -> Result<(i64, i64), Error> {
    let index = axis_index(axis, ty.rank(), what)?;
    local_range(
        ty.shape[index],
        partition.world,
        partition.rank,
        &format!("'{what}' along axis {index}"),
    )
}

/// Fill in everything about `expr` that only the target knows: which band a
/// [`Expr::Shard`] denotes, and what a [`Expr::Transmute`] wildcard stands for.
///
/// Both are extents the author declined to compute, and a byte offset cannot be
/// symbolic, so lowering sees neither. Doing it here rather than at lowering is
/// what lets the two agree by construction: the shape a `Transmute` is *checked*
/// against and the shape its bytes are *placed* under come from the same call.
///
/// Children first, so that a shard over a shard — or over a [`Expr::Concat`] whose
/// legs are themselves sharded — sees an operand it can already type.
fn specialize(expr: Expr, scope: &mut Scope<'_>) -> Result<Expr, Error> {
    if let Expr::Transmute { src, to } = expr {
        let src = specialize(*src, scope)?;
        let ty = infer(&src, scope)?;
        let to = infer_transmute(&ty, &to, &src)?;
        return Ok(src.transmute(to));
    }
    // The two group nodes resolve the same way `Shard` does, one step earlier:
    // a name and an offset both stop being symbolic here.
    if let Expr::SrcIndexed(template) = &expr {
        let index = scope.instance("SrcIndexed")?;
        return Ok(Expr::Src(substitute_index(template, index as u32)?));
    }
    if let Expr::Select {
        src,
        axis,
        stride,
        len,
    } = expr
    {
        let index = scope.instance("Select")?;
        let src = specialize(*src, scope)?;
        let start = index
            .checked_mul(stride)
            .or_overflow("a Select's start offset")?;
        let ty = infer(&src, scope)?;
        // Routed through `infer_slice` for the reason a `Shard` is: the band
        // this instance reads is checked by everything a `Slice` is checked by,
        // quantization-group alignment and extent included. An instance that
        // runs off the end of the grid is caught here, so declaring an `arity`
        // wider than the bank is a compile error rather than a slot of garbage.
        infer_slice(&ty, axis, start, len)?;
        return Ok(src.slice(axis.0, start, len));
    }
    // Every other variant is structural: its operands specialize and it puts
    // itself back together, which is what `map_children` says once.
    let Expr::Shard { src, axis } = expr else {
        return expr.map_children(|src| specialize(src, scope));
    };
    let src = specialize(*src, scope)?;
    if scope.partition.world <= 1 {
        // Not a degenerate one-rank slice but the operand itself, so that a
        // single-GPU plan is identical to one compiled from a contract that
        // never mentioned sharding.
        return Ok(src);
    }
    let ty = infer(&src, scope)?;
    let (start, len) = shard_range(&ty, axis, scope.partition, &scope.what)?;
    Ok(src.slice(axis.0, start, len))
}

/// Resolve an [`Axis`] against a rank, rejecting out-of-range axes.
fn axis_index(axis: Axis, rank: usize, what: &str) -> Result<usize, Error> {
    let index = usize::from(axis.0);
    if index >= rank {
        return Err(Error::Contract(format!(
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

/// Shared by [`Expr::Slice`] and [`Expr::Stride`]: both name `len` positions
/// `step` apart from `start`, and both must land inside the axis.
fn selected_axis(
    ty: &TensorType,
    axis: Axis,
    start: i64,
    len: i64,
    step: i64,
    what: &str,
) -> Result<usize, Error> {
    let index = axis_index(axis, ty.rank(), what)?;
    if len < 1 {
        return Err(Error::Contract(format!(
            "{what} len must be >= 1, got {len}"
        )));
    }
    if start < 0 {
        return Err(Error::Contract(format!(
            "{what} start must be >= 0, got {start}"
        )));
    }
    let extent = ty.shape[index];
    let last = start
        .checked_add(
            len.checked_sub(1)
                .and_then(|n| n.checked_mul(step))
                .or_overflow("selection overflows i64")?,
        )
        .or_overflow("selection overflows i64")?;
    if last >= extent {
        return Err(Error::Contract(format!(
            "{what} reads index {last} of axis {index}, which has extent {extent}"
        )));
    }
    Ok(index)
}

fn narrowed(ty: &TensorType, index: usize, len: i64) -> TensorType {
    let mut shape = ty.shape.clone();
    shape[index] = len;
    TensorType {
        shape,
        encoding: ty.encoding.clone(),
    }
}

/// The rule that stops one tensor from having two spellings.
///
/// A node that denotes exactly its operand is not a second way to say the
/// operand — it is a way to hide it, and every reader below has to carry a case
/// for the tensor that arrives unchanged. [`Expr::Cast`] and a [`Expr::Stride`]
/// of step 1 were already refused on this ground; this states the same refusal
/// once, for every node that can express it.
///
/// [`Expr::Shard`] is the one deliberate exception, and it is not an oversight:
/// it is the only node whose denotation is not a function of the expression
/// alone. At `world == 1` a shard *is* its operand, but refusing it there would
/// mean a contract could not be written without knowing how many ranks would
/// compile it — which is the property the node exists to provide.
/// [`Resolver::specialize`] canonicalizes it away instead, before anything
/// below the frontend sees it.
///
/// [`Expr::Cast`]: crate::contract::Expr::Cast
/// [`Expr::Stride`]: crate::contract::Expr::Stride
/// [`Expr::Shard`]: crate::contract::Expr::Shard
fn denotes_its_operand(node: &str, how: &str) -> Error {
    Error::Contract(format!(
        "{node} {how}, so it denotes its operand; say the operand instead"
    ))
}

fn infer_slice(ty: &TensorType, axis: Axis, start: i64, len: i64) -> Result<TensorType, Error> {
    let index = selected_axis(ty, axis, start, len, 1, "Slice")?;
    if start == 0 && len == ty.shape[index] {
        return Err(denotes_its_operand(
            "Slice",
            &format!("covers the whole of axis {index}"),
        ));
    }
    if let Some(group) = block_granularity(&ty.encoding, index)
        && (start % group != 0 || len % group != 0)
    {
        return Err(Error::Contract(format!(
            "Slice [{start}, {len}) on quantized axis {index} is not aligned to its {group}-element groups"
        )));
    }
    Ok(narrowed(ty, index, len))
}

fn infer_stride(
    ty: &TensorType,
    axis: Axis,
    start: i64,
    len: i64,
    step: i64,
) -> Result<TensorType, Error> {
    if step < 2 {
        return Err(Error::Contract(format!(
            "Stride step must be >= 2, got {step}; a contiguous run is a Slice"
        )));
    }
    // Same principle one rung down the cost hierarchy, and the same principle
    // `infer_gather` applies to a list of one: a progression with a single term
    // is a band, whatever its step claims.
    if len == 1 {
        return Err(Error::Contract(format!(
            "Stride of one position from {start} is a Slice, which costs less to \
             lower; say that instead"
        )));
    }
    let index = selected_axis(ty, axis, start, len, step, "Stride")?;
    // A band may land on a quantized axis if it lands on group boundaries. A
    // stride may not land on one at all: taking every other element of a block
    // leaves a block that no scale describes.
    if let Some(group) = block_granularity(&ty.encoding, index) {
        return Err(Error::Contract(format!(
            "Stride with step {step} on quantized axis {index} would split its {group}-element groups"
        )));
    }
    Ok(narrowed(ty, index, len))
}

/// The general placement, and the one the other two are special cases of.
///
/// Every rule here exists to keep the three nodes telling the truth about what
/// they cost. A list that is a run, or a run with a constant gap, denotes
/// something [`Expr::Slice`] or [`Expr::Stride`] says in constant space and
/// lowers to correspondingly fewer byte runs, so writing it as a list is not a
/// second way to say the same thing — it is a way to hide the cheaper one. The
/// same principle already refuses a `Stride` of step 1.
///
/// [`Expr::Slice`]: crate::contract::Expr::Slice
/// [`Expr::Stride`]: crate::contract::Expr::Stride
fn infer_gather(ty: &TensorType, axis: Axis, indices: &[i64]) -> Result<TensorType, Error> {
    let index = axis_index(axis, ty.rank(), "Gather")?;
    let extent = ty.shape[index];
    let Some((&first, rest)) = indices.split_first() else {
        return Err(Error::Contract(
            "Gather needs at least one index".to_string(),
        ));
    };
    for &i in indices {
        if i < 0 || i >= extent {
            return Err(Error::Contract(format!(
                "Gather reads index {i} of axis {index}, which has extent {extent}"
            )));
        }
    }
    // Same reason a stride may not: a permutation of the elements inside a
    // block leaves a block no scale describes. Permuting whole blocks is a
    // `Concat` of `Slice`s, which keeps every group intact and is checked as such.
    if let Some(group) = block_granularity(&ty.encoding, index) {
        return Err(Error::Contract(format!(
            "Gather on quantized axis {index} would split its {group}-element groups"
        )));
    }
    // A single index is a band of one, and every constant-gap list is one of
    // the two cheaper nodes. `step` is only a progression if it is positive:
    // a descending or repeating list is a genuine gather.
    let step = rest.first().map_or(1, |second| second - first);
    if step >= 1
        && rest
            .iter()
            .zip(indices)
            .all(|(next, prev)| next - prev == step)
    {
        let name = if step == 1 { "Slice" } else { "Stride" };
        return Err(Error::Contract(format!(
            "Gather of {} indices from {first} in steps of {step} is a {name}, \
             which costs less to lower; say that instead",
            indices.len()
        )));
    }
    Ok(narrowed(ty, index, indices.len() as i64))
}

fn infer_concat(axis: Axis, parts: &[TensorType]) -> Result<TensorType, Error> {
    let Some((head, tail)) = parts.split_first() else {
        return Err(Error::Contract(
            "Concat needs at least one part".to_string(),
        ));
    };
    if tail.is_empty() {
        return Err(denotes_its_operand("Concat", "has one part"));
    }
    let index = axis_index(axis, head.rank(), "Concat")?;
    let mut total = head.shape[index];
    for (offset, part) in tail.iter().enumerate() {
        if part.rank() != head.rank() {
            return Err(Error::Contract(format!(
                "Concat part {} has rank {} but part 0 has rank {}",
                offset + 1,
                part.rank(),
                head.rank()
            )));
        }
        for (other, (lhs, rhs)) in head.shape.iter().zip(part.shape.iter()).enumerate() {
            if other != index && lhs != rhs {
                return Err(Error::Contract(format!(
                    "Concat on axis {index}: part {} has shape {:?}, incompatible with part 0's {:?}",
                    offset + 1,
                    part.shape,
                    head.shape
                )));
            }
        }
        if crate::types::normalize_encoding(&part.encoding)
            != crate::types::normalize_encoding(&head.encoding)
        {
            return Err(Error::Contract(format!(
                "Concat part {} is encoded as {:?}, incompatible with part 0's {:?}",
                offset + 1,
                part.encoding,
                head.encoding
            )));
        }
        total = total
            .checked_add(part.shape[index])
            .or_overflow("Concat extent overflows i64")?;
    }
    if let Some(group) = block_granularity(&head.encoding, index) {
        for (offset, part) in parts.iter().enumerate() {
            if part.shape[index] % group != 0 {
                return Err(Error::Contract(format!(
                    "Concat part {offset} contributes {} elements to quantized axis {index}, which is not a multiple of its {group}-element groups",
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

/// Bits one element of `encoding` occupies. `None` when the encoding has no
/// fixed width.
fn element_bits(encoding: &Encoding) -> Option<u64> {
    match encoding {
        Encoding::Raw(dtype) => dtype.bytes().checked_mul(8),
        Encoding::Quant(spec) => Some(u64::from(spec.clone().normalized().normalized_bits())),
    }
}

/// The shape from the blocked axis onward, when the encoding blocks one.
///
/// What a rename may not touch: a quantized tensor's scales are laid out
/// against these extents, so regrouping them moves data out from under its
/// factors while the byte count still balances.
fn blocked_suffix(ty: &TensorType) -> Option<&[i64]> {
    let Encoding::Quant(spec) = &ty.encoding else {
        return None;
    };
    let channel = usize::from(spec.channel_axis?.0);
    ty.shape.get(channel..)
}

/// [`Expr::Transmute`]: the same bytes named differently.
///
/// `src` is passed for its *form* alone — whether it is a whole tensor — which
/// is the one thing a type cannot say and the one thing a change of element
/// width needs to know.
fn infer_transmute(ty: &TensorType, to: &TensorType, src: &Expr) -> Result<TensorType, Error> {
    let from_bytes = ty.byte_size()?;
    let bits = element_bits(&to.encoding).ok_or_else(|| {
        Error::Contract(format!(
            "Transmute to {:?} has no element width",
            to.encoding
        ))
    })?;
    let total_bits = from_bytes
        .checked_mul(8)
        .or_overflow("Transmute byte size")?;
    if !total_bits.is_multiple_of(bits) {
        return Err(Error::Contract(format!(
            "Transmute of {from_bytes} bytes does not divide into {bits}-bit elements"
        )));
    }
    let total = i64::try_from(total_bits / bits).or_overflow("Transmute element count")?;
    let resolved = TensorType {
        shape: resolve_extents(&to.shape, total)?,
        encoding: to.encoding.clone(),
    };

    let to_bytes = resolved.byte_size()?;
    if from_bytes != to_bytes {
        return Err(Error::Contract(format!(
            "Transmute changes the byte size, {from_bytes} -> {to_bytes}"
        )));
    }
    // Checked after `-1` is resolved, so that a target written with an inferred
    // extent is judged by what it turned out to mean rather than by how it was
    // spelled.
    if resolved == *ty {
        return Err(denotes_its_operand(
            "Transmute",
            "renames its operand to the type it already has",
        ));
    }
    if element_bits(&ty.encoding) != Some(bits) && !matches!(src, Expr::Src(_) | Expr::Out(_)) {
        return Err(Error::Contract(
            "Transmute changes the element width, so it may only rename a whole \
             tensor; publish the expression first and transmute that name"
                .to_string(),
        ));
    }
    if let (Some(from), Some(to)) = (blocked_suffix(ty), blocked_suffix(&resolved))
        && from != to
    {
        return Err(Error::Contract(format!(
            "Transmute regroups a quantized tensor from {from:?} to {to:?} at and \
             below its blocked axis; only leading axes may regroup"
        )));
    }
    Ok(resolved)
}

/// A fill is a leaf, so it is its own declared type -- with three conditions,
/// all of which come from the fact that the plan realizes it by zeroing the
/// destination and never writing there.
fn infer_fill(value: u32, ty: &TensorType) -> Result<TensorType, Error> {
    if ty.shape.is_empty() {
        return Err(Error::Contract(
            "Fill has no extents; a zeroed FFI node reads as a rank-0 fill of \
             zero, so the rank is what tells one apart from a forgotten field"
                .to_string(),
        ));
    }
    if let Some(bad) = ty.shape.iter().find(|extent| **extent < 1) {
        return Err(Error::Contract(format!(
            "Fill shape {:?} has a non-positive extent {bad}; a fill has no \
             operand to solve a wildcard against, so every extent must be given",
            ty.shape
        )));
    }
    let Encoding::Raw(dtype) = &ty.encoding else {
        return Err(Error::Contract(format!(
            "Fill into {:?} is not a constant: a quantized code word means \
             nothing without the block scale beside it, and which code reads as \
             zero is scheme-specific",
            ty.encoding
        )));
    };
    if value != 0.0_f32.to_bits() || *dtype == DType::E8M0 {
        return Err(Error::Contract(format!(
            "Fill of {} as {dtype:?} is not a run of zero bytes, which is all \
             the zeroing can write",
            f32::from_bits(value)
        )));
    }
    Ok(ty.clone())
}

/// The geometry a repack kernel needs, derived rather than restated.
///
/// A repack is opaque to the checker in one direction only. What the swizzle
/// does to a byte is a kernel's business, and this cannot see through it --
/// which is why `to` is declared rather than derived. But *how many* bytes
/// there are on each side is not opaque at all: the operand's type says one
/// side and `to` says the other, so a [`RepackSpec`] is a function of the two
/// and a contract that also stated it could disagree with itself.
///
/// Deriving it is also what makes the source selection expressible. The fields
/// this used to carry -- a row offset, a valid-row count, a column offset and
/// stride, an even/odd row map -- were [`Expr::Slice`], [`Expr::Shard`] and
/// [`Expr::Stride`] spelled as integers a kernel reads, which is why a repack
/// could not be sharded like anything else: there was nowhere to put the node,
/// so the rank had to be resolved before the contract existed.
///
/// Leaving the two sides unrelated was a memory-safety hole, not a stylistic
/// one. The destination buffer is sized from `to` and the kernel writes
/// `batch * target_rows * target_cols` elements; a declaration that understated
/// it produced a device-side overrun with nothing between the author and the
/// fault. A target *larger* than the source is allowed and is the padding a
/// tile quantum needs -- the kernel zero-fills the tail.
pub(crate) fn repack_spec(
    ty: &TensorType,
    layout: RepackLayout,
    to: &TensorType,
) -> Result<RepackSpec, Error> {
    // What each layout's operand looks like, and how many columns that is. The
    // column count is the logical one -- MXFP4 groups of 32 for a weight -- so
    // that a target's padding is comparable to it.
    let (want_rank, cols) = match layout {
        RepackLayout::MarlinMxfp4Weight => {
            if ty.rank() != 4 || ty.shape[3] != 16 {
                return Err(Error::Contract(format!(
                    "MarlinMxfp4Weight Repack operand must be [B, R, K/32, 16], got {:?}",
                    ty.shape
                )));
            }
            (4, ty.shape[2].checked_mul(32).unwrap_or(i64::MAX))
        }
        RepackLayout::MarlinMxfp4Scale => {
            if ty.rank() != 3 {
                return Err(Error::Contract(format!(
                    "MarlinMxfp4Scale Repack operand must be [B, R, groups], got {:?}",
                    ty.shape
                )));
            }
            (3, ty.shape[2])
        }
    };
    debug_assert_eq!(ty.rank(), want_rank);
    if to.rank() != 3 {
        return Err(Error::Contract(format!(
            "Repack declares {:?}; a repack produces [batch, rows, cols]",
            to.shape
        )));
    }
    let (batch, rows) = (ty.shape[0], ty.shape[1]);
    if to.shape[0] != batch {
        return Err(Error::Contract(format!(
            "Repack operand has batch {batch} but declares {:?}",
            to.shape
        )));
    }
    // Padding only. A target smaller than its source is a truncation the
    // algebra can say -- `Expr::Slice` on the operand -- so a kernel doing it
    // silently would be the same fact stated twice, and wrong once.
    if to.shape[1] < rows || to.shape[2] < cols {
        return Err(Error::Contract(format!(
            "Repack declares {:?}, smaller than the [{batch}, {rows}, {cols}] it \
             reads; narrow the operand instead",
            to.shape
        )));
    }
    // An element is the same number of bits on both sides. `Repack` changes
    // where a byte sits and nothing else -- it is the kernel-priced member of
    // the *placement* family, so it may no more reinterpret an element than a
    // `Slice` may. Stated per element rather than per row or per tensor
    // because padding is the one thing a repack may add, and padding changes
    // both of those.
    //
    // This is `Expr::Transmute`'s invariant seen from the other side, and it
    // closes the same hole: the destination buffer is sized from `to`, whose
    // shape was checked just above and whose element width was not. A repack
    // naming a wider encoding over-allocated in silence; one naming a narrower
    // encoding under-allocated and the kernel wrote past the end.
    let source_bits = row_bits(&ty.shape[2..], &ty.encoding, "Repack operand")?;
    let target_bits = element_bits(&to.encoding).ok_or_else(|| {
        Error::Contract(format!(
            "Repack declares {:?}, which has no fixed element width",
            to.encoding
        ))
    })?;
    let cols_u64 = u64::try_from(cols).unwrap_or(u64::MAX);
    if source_bits != target_bits.saturating_mul(cols_u64) {
        return Err(Error::Contract(format!(
            "Repack reads {cols} columns of {:?} as {source_bits} bits and \
             writes them as {target_bits}-bit {:?}; a repack moves bytes, it \
             does not reinterpret them",
            ty.encoding, to.encoding
        )));
    }
    Ok(RepackSpec {
        layout,
        batch: dim_u32(batch, "Repack batch")?,
        source_rows: dim_u32(rows, "Repack source rows")?,
        target_rows: dim_u32(to.shape[1], "Repack target rows")?,
        source_cols: dim_u32(cols, "Repack source columns")?,
        target_cols: dim_u32(to.shape[2], "Repack target columns")?,
    })
}

/// The bits one row of `trailing` extents occupies at `encoding`.
fn row_bits(trailing: &[i64], encoding: &Encoding, what: &str) -> Result<u64, Error> {
    let bits = element_bits(encoding)
        .ok_or_else(|| Error::Contract(format!("{what} has no fixed element width")))?;
    let mut total = bits;
    for extent in trailing {
        let extent = u64::try_from(*extent)
            .map_err(|_| Error::Contract(format!("{what} has a negative extent {extent}")))?;
        total = total
            .checked_mul(extent)
            .ok_or_else(|| Error::Contract(format!("{what} row size overflow")))?;
    }
    Ok(total)
}

fn dim_u32(value: i64, what: &str) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::Contract(format!("{what} {value} does not fit in u32")))
}

/// A cast keeps the shape and replaces the representation.
///
/// Which of the three directions this is falls out of the pair of encodings, so
/// there is nothing for an author to choose beyond naming the destination --
/// and nothing for the plan builder to derive from a coincidence of types.
///
/// The one pair with no meaning is quantized to quantized. There is no kernel
/// for it in either backend, and there is no obvious one either: the scales of
/// the destination scheme are not a function of the source's, so it is a decode
/// and an encode however it is spelled. Saying that here makes the two-step
/// visible instead of hiding it behind a declaration.
fn infer_cast(ty: &TensorType, to: &Encoding) -> Result<TensorType, Error> {
    if *to == ty.encoding {
        return Err(denotes_its_operand(
            "Cast",
            &format!("re-encodes its operand as the {to:?} it already is"),
        ));
    }
    match (&ty.encoding, to) {
        (Encoding::Quant(from), Encoding::Quant(into)) => {
            return Err(Error::Contract(format!(
                "Cast re-encodes {:?} as {:?}; no kernel does that in one step, \
                 so cast to a raw type first and cast that",
                from.scheme, into.scheme
            )));
        }
        (_, Encoding::Quant(spec)) => {
            // The blocked axis has to divide, or the last group of every row is
            // short and the scales stop describing the payload.
            if let Some(channel) = spec.channel_axis {
                let index = axis_index(channel, ty.rank(), "Cast channel_axis")?;
                let group = i64::from(spec.normalized_group_size());
                if group > 1 && ty.shape[index] % group != 0 {
                    return Err(Error::Contract(format!(
                        "Cast groups axis {index} by {group}, but its extent is {}",
                        ty.shape[index]
                    )));
                }
            }
        }
        _ => {}
    }
    Ok(TensorType {
        shape: ty.shape.clone(),
        encoding: match to {
            Encoding::Quant(spec) => Encoding::Quant(spec.clone().normalized()),
            raw => raw.clone(),
        },
    })
}
/// A uniform `Scale` preserves both shape and encoding; only the values move.
///
/// Restricted to raw floating-point elements, and the restriction is the useful
/// part. A uniform factor over a `Quant` encoding would have to mean one of two
/// different things — decode, multiply and re-encode, or multiply the stored
/// factors and leave the payload — and a family that wrote it would get
/// whichever the executor happened to implement. Scaling an integer tensor is a
/// rounding rule nobody stated. Both are refused here, where the message can
/// name the tensor, rather than reaching a kernel that has no way to signal it
/// did something other than what was asked.
///
/// A quantized operand is not refused for lack of meaning in general — it is
/// refused for lack of meaning *with one constant*. Say the scales and it is
/// dequantization; see [`ScaleFactor::PerGroup`].
fn infer_scale(ty: TensorType, factor_bits: u32) -> Result<TensorType, Error> {
    let dtype = match ty.encoding {
        Encoding::Raw(dtype) => dtype,
        Encoding::Quant(_) => {
            return Err(Error::Contract(format!(
                "Scale of a quantized tensor ({:?}) is not supported",
                ty.encoding
            )));
        }
    };
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(Error::Contract(format!(
            "Scale requires F32, F16 or BF16 elements, got {dtype:?}"
        )));
    }
    // A non-finite constant scales every element to the same non-finite value,
    // which is never what a contract meant to say and is invisible until the
    // model produces garbage.
    let factor = f32::from_bits(factor_bits);
    if !factor.is_finite() {
        return Err(Error::Contract(format!(
            "Scale factor must be finite, got {factor}"
        )));
    }
    // Zero is rejected for a different reason than the value being useless: it
    // is what an all-zero `PieLoaderExprNode` carries, so a C++ author who
    // builds a `Scale` node without setting `scale_factor_bits` gets exactly
    // this. Allowing it would turn that omission into a tensor of zeros that
    // loads, caches and runs. `-0.0` is the same bit-pattern hazard with the
    // sign bit set, so it goes too.
    if factor == 0.0 {
        return Err(Error::Contract(
            "Scale factor is zero, which is also what an unset factor field \
             reads as; state the constant the contract meant"
                .to_string(),
        ));
    }
    if factor == 1.0 {
        return Err(denotes_its_operand("Scale", "multiplies by one"));
    }
    Ok(ty)
}

/// A per-group `Scale` yields the logical type of what it read.
///
/// Over `Raw` that is the input type unchanged, as with a uniform factor. Over
/// `Quant` it is the scheme's `logical_dtype`, and the operation is
/// dequantization: the stored elements mean nothing without their scales, so
/// multiplying by them is exactly what turns the payload back into numbers.
/// That is why this is the one place the algebra unpacks a quantized tensor,
/// and why it needs no `Dequantize` sibling.
///
/// The check that earns its keep is the last one. `by` must be `src` with
/// `axis` divided by `group`, so a partition applied to the weight and not to
/// the scales — or applied to both on different axes — is a compile error that
/// names both shapes. Recovering that pairing from a name convention instead,
/// below the contract, is unfalsifiable by construction.
fn infer_scale_per_block(ty: TensorType, by: TensorType) -> Result<TensorType, Error> {
    let out_dtype = match &ty.encoding {
        Encoding::Raw(dtype) => *dtype,
        Encoding::Quant(spec) => spec.logical_dtype,
    };
    if !matches!(out_dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(Error::Contract(format!(
            "Scale requires F32, F16 or BF16 elements, got {out_dtype:?}"
        )));
    }
    // The factors are read as numbers, so they must be a type that denotes one.
    // `E8M0` is here because that is what a block-scaled checkpoint stores: a
    // bare exponent, no sign and no mantissa.
    match &by.encoding {
        Encoding::Raw(DType::F32 | DType::F16 | DType::BF16 | DType::E8M0) => {}
        other => {
            return Err(Error::Contract(format!(
                "Scale factors must be raw F32, F16, BF16 or E8M0 elements, got {other:?}"
            )));
        }
    }
    // Rank first, because everything below indexes both shapes together. Equal
    // rank is what makes the ratio per axis meaningful at all: a factor tensor
    // of a different rank is not a coarser view of this one, it is a different
    // tensor the author paired by mistake.
    if by.shape.len() != ty.shape.len() {
        return Err(Error::Contract(format!(
            "Scale factors have shape {:?}, which is not a blocking of {:?} \
             -- a factor tensor states its block size by having the same rank \
             and dividing each axis",
            by.shape, ty.shape
        )));
    }
    let mut blocked = false;
    for (axis, (&extent, &factors)) in ty.shape.iter().zip(by.shape.iter()).enumerate() {
        if factors <= 0 {
            return Err(Error::Contract(format!(
                "Scale factors have extent {factors} on axis {axis}, so no \
                 block size divides it"
            )));
        }
        if extent % factors != 0 {
            return Err(Error::Contract(format!(
                "Scale factors have shape {:?}, but axis {axis} of {:?} is not \
                 a whole number of blocks of {}",
                by.shape,
                ty.shape,
                extent / factors
            )));
        }
        if factors != extent {
            blocked = true;
        }
    }
    // Symmetry rule A: a node may not denote exactly its operand. Factors
    // shaped like the weight are one number per element, which is a plain
    // elementwise product and not a blocking -- and, read the other way, a
    // `Uniform` scale is the rank-0 case this would shadow. Rejecting it keeps
    // the two spellings of "no blocks" from both being legal.
    if !blocked && !ty.shape.is_empty() {
        return Err(Error::Contract(format!(
            "Scale factors have the operand's own shape {:?}, so they group \
             nothing; a factor per element is an elementwise product, not a \
             block scale",
            ty.shape
        )));
    }
    Ok(TensorType {
        shape: ty.shape,
        encoding: Encoding::Raw(out_dtype),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{ModelContract, TensorContract};
    use crate::types::{DType, QuantScheme, QuantSpec};
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

    fn check_one(expr: Expr, checkpoint: &dyn CheckpointTypes) -> Result<TensorType, Error> {
        infer_type(&expr, checkpoint).map(|(ty, _)| ty)
    }

    fn fp8_per_row() -> QuantSpec {
        QuantSpec {
            scheme: QuantScheme::Fp8E4M3,
            logical_dtype: DType::F8E4M3,
            bits_per_element: 8,
            group_size: 1,
            channel_axis: Some(Axis(0)),
        }
    }

    fn specialize_one(expr: Expr, rank: u32, world: u32) -> Result<Expr, Error> {
        let checkpoint = qwen3();
        Resolver::new(&checkpoint, Partition::new(rank, world)).specialize(expr, "q")
    }

    fn check_at(expr: &Expr, rank: u32, world: u32) -> Result<TensorType, Error> {
        let checkpoint = qwen3();
        Resolver::new(&checkpoint, Partition::new(rank, world)).infer(expr, "q")
    }

    #[test]
    fn shard_types_as_this_ranks_band() {
        // Typing is total: a `Shard` reads its extent off the resolver's
        // partition instead of being a variant the checker has no answer for.
        let ty = check_at(&Expr::src("q_proj").shard(0), 3, 4).unwrap();
        assert_eq!(ty, TensorType::raw(vec![512, 2048], DType::BF16));
    }

    #[test]
    fn a_shard_and_its_specialization_have_the_same_type() {
        // The invariant that lets specialization be a lowering rewrite rather
        // than a typing precondition. Both answers come from `shard_range`, so
        // a disagreement is not expressible -- this pins that they still do.
        let shard = Expr::concat(
            0,
            vec![Expr::src("k_proj").shard(0), Expr::src("v_proj").shard(0)],
        );
        let specialized = specialize_one(shard.clone(), 1, 2).unwrap();
        assert_eq!(
            check_at(&shard, 1, 2).unwrap(),
            check_at(&specialized, 1, 2).unwrap()
        );
    }

    #[test]
    fn a_shard_is_checked_by_everything_a_slice_is() {
        // Routing through `infer_slice` is what makes this a rejection. Read
        // as bare extent arithmetic, a shard that halves a 128-element group
        // types fine and hands the rank a band its scales no longer describe.
        let checkpoint = FakeCheckpoint(HashMap::from([(
            "w".to_string(),
            TensorType {
                shape: vec![256, 2048],
                encoding: Encoding::Quant(QuantSpec {
                    scheme: QuantScheme::Mxfp4E2M1E8M0,
                    logical_dtype: DType::BF16,
                    bits_per_element: 4,
                    group_size: 128,
                    channel_axis: Some(Axis(0)),
                }),
            },
        )]));
        let expr = Expr::src("w").shard(0);
        let mut ok = Resolver::new(&checkpoint, Partition::new(0, 2));
        assert!(ok.infer(&expr, "w").is_ok());
        let mut split = Resolver::new(&checkpoint, Partition::new(0, 4));
        let err = split.infer(&expr, "w").unwrap_err();
        assert!(format!("{err}").contains("128-element groups"), "{err}");
    }

    /// A checkpoint whose only tensor is blocked along axis 1, so the two
    /// selection nodes can be asked the same question about the same axis.
    fn blocked() -> FakeCheckpoint {
        FakeCheckpoint(HashMap::from([(
            "w".to_string(),
            TensorType {
                shape: vec![64, 256],
                encoding: Encoding::Quant(QuantSpec {
                    scheme: QuantScheme::Mxfp4E2M1E8M0,
                    logical_dtype: DType::BF16,
                    bits_per_element: 4,
                    group_size: 32,
                    channel_axis: Some(Axis(1)),
                }),
            },
        )]))
    }

    /// The two nodes exist because one is cheap and the other is not, so the
    /// cheap one must not be spellable as the expensive one.
    #[test]
    fn a_contiguous_run_is_a_slice_not_a_stride() {
        let err = check_one(Expr::src("q_proj").stride(0, 0, 512, 1), &qwen3()).unwrap_err();
        assert!(
            err.to_string().contains("a contiguous run is a Slice"),
            "{err}"
        );
    }

    /// The rule that distinguishes the two on a blocked axis: a band can land
    /// on group boundaries, a stride never can.
    #[test]
    fn a_band_may_cross_a_blocked_axis_where_a_stride_may_not() {
        let ty = check_one(Expr::src("w").slice(1, 32, 64), &blocked()).unwrap();
        assert_eq!(ty.shape, vec![64, 64]);

        let err = check_one(Expr::src("w").stride(1, 0, 64, 2), &blocked()).unwrap_err();
        assert!(
            err.to_string()
                .contains("would split its 32-element groups"),
            "{err}"
        );
    }

    /// `start + (len - 1) * step` is the last index read, and it is the bound
    /// that matters — `start + len` would admit a stride that runs off the end.
    #[test]
    fn a_stride_is_bounded_by_the_last_index_it_reads() {
        let ok = check_one(Expr::src("q_proj").stride(0, 0, 1024, 2), &qwen3()).unwrap();
        assert_eq!(ok.shape, vec![1024, 2048]);

        // `start + len` is 1026, comfortably inside the axis; only the strided
        // last index reveals that the selection runs off the end.
        let err = check_one(Expr::src("q_proj").stride(0, 2, 1024, 2), &qwen3()).unwrap_err();
        assert!(err.to_string().contains("reads index 2048"), "{err}");
    }

    /// The cost hierarchy, stated in the type checker: a list that either of
    /// the cheaper nodes could express is refused with that node named. This is
    /// the same rule that refuses a `Stride` of step 1, one rung further up.
    #[test]
    fn a_gather_must_not_be_expressible_as_a_slice_or_a_stride() {
        let run = check_one(Expr::src("q_proj").gather(0, vec![4, 5, 6, 7]), &qwen3()).unwrap_err();
        assert!(run.to_string().contains("is a Slice"), "{run}");

        let progression =
            check_one(Expr::src("q_proj").gather(0, vec![1, 4, 7, 10]), &qwen3()).unwrap_err();
        assert!(
            progression.to_string().contains("is a Stride"),
            "{progression}"
        );

        // A single index is a band of one, so it is a `Slice` too.
        let one = check_one(Expr::src("q_proj").gather(0, vec![9]), &qwen3()).unwrap_err();
        assert!(one.to_string().contains("is a Slice"), "{one}");
    }

    /// The cost hierarchy is a total order, so the demotion rule has to hold on
    /// every edge of it. `Gather` was already held to both; this is the edge
    /// between the two cheaper nodes, which a progression of one term crosses.
    #[test]
    fn a_stride_of_one_position_is_a_slice() {
        let err = check_one(Expr::src("q_proj").stride(0, 3, 1, 2), &qwen3()).unwrap_err();
        assert!(err.to_string().contains("is a Slice"), "{err}");
    }

    /// One rule, five nodes: an expression that denotes exactly its operand is
    /// a second spelling of that operand, and two spellings of one tensor is
    /// what the whole algebra is arranged to avoid.
    ///
    /// `Expr::Cast` has been refused on this ground since it was written and
    /// `Expr::Shard` is exempt on purpose — see [`denotes_its_operand`].
    #[test]
    fn no_node_may_denote_exactly_its_operand() {
        let whole_axis = check_one(Expr::src("q_proj").slice(0, 0, 2048), &qwen3()).unwrap_err();
        assert!(
            whole_axis
                .to_string()
                .contains("covers the whole of axis 0"),
            "{whole_axis}"
        );
        // A band that stops one row short is still a band.
        assert!(check_one(Expr::src("q_proj").slice(0, 0, 2047), &qwen3()).is_ok());

        let one_part = check_one(Expr::concat(0, vec![Expr::src("q_proj")]), &qwen3()).unwrap_err();
        assert!(one_part.to_string().contains("has one part"), "{one_part}");

        let rename = check_one(
            Expr::src("q_proj").transmute(TensorType::raw(vec![2048, 2048], DType::BF16)),
            &qwen3(),
        )
        .unwrap_err();
        assert!(
            rename.to_string().contains("the type it already has"),
            "{rename}"
        );
        // The check is on what `-1` resolved to, not on how it was written.
        let inferred = check_one(
            Expr::src("q_proj").transmute(TensorType::raw(vec![2048, -1], DType::BF16)),
            &qwen3(),
        )
        .unwrap_err();
        assert!(
            inferred.to_string().contains("the type it already has"),
            "{inferred}"
        );

        let unit = check_one(Expr::src("q_proj").scale(1.0), &qwen3()).unwrap_err();
        assert!(unit.to_string().contains("multiplies by one"), "{unit}");
        // Negation is a real multiply, and so is anything else.
        assert!(check_one(Expr::src("q_proj").scale(-1.0), &qwen3()).is_ok());
    }

    /// What is left once the two cheaper nodes are excluded: an order nobody
    /// can compute. Descending and repeating lists are genuine gathers, not
    /// progressions in disguise.
    #[test]
    fn a_gather_takes_the_orders_nothing_cheaper_can_say() {
        let permuted =
            check_one(Expr::src("q_proj").gather(0, vec![3, 0, 2, 1]), &qwen3()).unwrap();
        assert_eq!(permuted.shape, vec![4, 2048]);

        let descending = check_one(Expr::src("q_proj").gather(0, vec![9, 6, 3]), &qwen3()).unwrap();
        assert_eq!(descending.shape, vec![3, 2048]);

        // Reading one row twice is a broadcast, and a well-defined thing to ask
        // for -- nothing in a placement node requires the map to be injective.
        let repeated = check_one(Expr::src("q_proj").gather(0, vec![7, 7, 7]), &qwen3()).unwrap();
        assert_eq!(repeated.shape, vec![3, 2048]);
    }

    #[test]
    fn a_gather_is_bounded_by_every_index_it_reads() {
        let empty = check_one(Expr::src("q_proj").gather(0, vec![]), &qwen3()).unwrap_err();
        assert!(empty.to_string().contains("at least one index"), "{empty}");

        let past =
            check_one(Expr::src("q_proj").gather(0, vec![0, 2048, 1]), &qwen3()).unwrap_err();
        assert!(past.to_string().contains("reads index 2048"), "{past}");

        let negative = check_one(Expr::src("q_proj").gather(0, vec![2, -1]), &qwen3()).unwrap_err();
        assert!(
            negative.to_string().contains("reads index -1"),
            "{negative}"
        );
    }

    /// The same rule a stride is held to, and for the same reason: a
    /// permutation *inside* a block leaves a block no scale describes. Whole
    /// blocks may be permuted, and that is a `Concat` of `Slice`s.
    #[test]
    fn a_gather_may_not_touch_a_quantized_axis() {
        let err = check_one(Expr::src("w").gather(1, vec![3, 0, 2, 1]), &blocked()).unwrap_err();
        assert!(
            err.to_string()
                .contains("would split its 32-element groups"),
            "{err}"
        );

        let by_block = check_one(
            Expr::concat(
                1,
                vec![
                    Expr::src("w").slice(1, 32, 32),
                    Expr::src("w").slice(1, 0, 32),
                ],
            ),
            &blocked(),
        )
        .unwrap();
        assert_eq!(by_block.shape, vec![64, 64]);
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
            Expr::concat(
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
        assert!(message.contains("'q' along axis 0 is 1024"), "{message}");
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
    fn concat_sums_the_joined_axis() {
        let ty = check_one(
            Expr::concat(
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
    fn concat_rejects_mismatched_other_axes() {
        let checkpoint =
            FakeCheckpoint::new(&[("a", &[4, 8], DType::BF16), ("b", &[4, 9], DType::BF16)]);
        let err = check_one(
            Expr::concat(0, vec![Expr::src("a"), Expr::src("b")]),
            &checkpoint,
        )
        .unwrap_err();
        assert!(err.to_string().contains("incompatible"));
    }

    #[test]
    fn concat_rejects_mismatched_encodings() {
        let checkpoint =
            FakeCheckpoint::new(&[("a", &[4, 8], DType::BF16), ("b", &[4, 8], DType::F32)]);
        let err = check_one(
            Expr::concat(0, vec![Expr::src("a"), Expr::src("b")]),
            &checkpoint,
        )
        .unwrap_err();
        assert!(err.to_string().contains("encoded as"));
    }

    #[test]
    fn tp_shard_then_fuse_composes() {
        // TP=2 rank 0: half of each projection, concatenated.
        let ty = check_one(
            Expr::concat(
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
        let gate = check_one(Expr::src("gate_up").stride(0, 0, 256, 2), &checkpoint).unwrap();
        let up = check_one(Expr::src("gate_up").stride(0, 1, 256, 2), &checkpoint).unwrap();
        assert_eq!(gate.shape, vec![256, 64]);
        assert_eq!(up.shape, vec![256, 64]);
        // One past the last selectable odd row.
        assert!(check_one(Expr::src("gate_up").stride(0, 1, 257, 2), &checkpoint).is_err());
    }

    #[test]
    fn expert_stack_via_transmute_and_concat() {
        let checkpoint = FakeCheckpoint::new(&[
            ("e0.gate", &[6144, 2048], DType::BF16),
            ("e1.gate", &[6144, 2048], DType::BF16),
        ]);
        let ty = check_one(
            Expr::concat(
                0,
                vec![
                    Expr::src("e0.gate")
                        .transmute(TensorType::raw(vec![1, 6144, 2048], DType::BF16)),
                    Expr::src("e1.gate")
                        .transmute(TensorType::raw(vec![1, 6144, 2048], DType::BF16)),
                ],
            ),
            &checkpoint,
        )
        .unwrap();
        assert_eq!(ty.shape, vec![2, 6144, 2048]);
    }

    #[test]
    fn transmute_infers_one_wildcard() {
        let ty = check_one(
            Expr::src("q_proj").transmute(TensorType::raw(vec![16, 128, -1], DType::BF16)),
            &qwen3(),
        )
        .unwrap();
        assert_eq!(ty.shape, vec![16, 128, 2048]);
    }

    #[test]
    fn transmute_rejects_a_shape_the_bytes_do_not_fill() {
        let err = check_one(
            Expr::src("q_proj").transmute(TensorType::raw(vec![2048, 2049], DType::BF16)),
            &qwen3(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("not the"), "{err}");
    }

    /// The wildcard counts elements of the *output* type, so a rename that also
    /// halves the element width doubles what `-1` stands for.
    #[test]
    fn a_wildcard_is_measured_in_the_type_being_named() {
        let ty = check_one(
            Expr::src("q_proj").transmute(TensorType::raw(vec![2048, -1], DType::F8E4M3)),
            &qwen3(),
        )
        .unwrap();
        assert_eq!(ty.shape, vec![2048, 4096]);
    }

    /// The rule `Bitcast` had and `Reshape` did not, now stated once: a partial
    /// view's element offsets stop meaning anything when the element size
    /// changes under them.
    #[test]
    fn a_change_of_element_width_needs_a_whole_tensor() {
        let err = check_one(
            Expr::src("q_proj")
                .slice(0, 0, 1024)
                .transmute(TensorType::raw(vec![1024, 4096], DType::F8E4M3)),
            &qwen3(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("whole"), "{err}");
        // The same rename of the whole tensor is fine.
        check_one(
            Expr::src("q_proj").transmute(TensorType::raw(vec![2048, 4096], DType::F8E4M3)),
            &qwen3(),
        )
        .unwrap();
    }

    /// The other half of the merge: `Reshape` refused every quantized operand,
    /// so a stack of quantized experts had to fold its rank lift into a
    /// `Bitcast` of the whole packed tensor. Leading axes may now regroup.
    #[test]
    fn a_quantized_tensor_may_gain_a_leading_axis() {
        let checkpoint = FakeCheckpoint::new(&[("packed", &[128, 64], DType::U8)]);
        let mxfp4 = Encoding::Quant(QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        });
        let packed = Expr::src("packed").transmute(TensorType {
            shape: vec![128, 128],
            encoding: mxfp4.clone(),
        });
        let ty = check_one(
            packed.clone().transmute(TensorType {
                shape: vec![1, 128, 128],
                encoding: Encoding::Quant(QuantSpec {
                    channel_axis: Some(Axis(2)),
                    ..match &mxfp4 {
                        Encoding::Quant(spec) => spec.clone(),
                        _ => unreachable!(),
                    }
                }),
            }),
            &checkpoint,
        )
        .unwrap();
        assert_eq!(ty.shape, vec![1, 128, 128]);

        // Reblocking is not a rename: the byte count balances, but every scale
        // would then cover a different 32 elements.
        let err = check_one(
            packed.transmute(TensorType {
                shape: vec![256, 64],
                encoding: mxfp4,
            }),
            &checkpoint,
        )
        .unwrap_err();
        assert!(err.to_string().contains("only leading axes"), "{err}");
    }

    /// The node that replaced `Pad`: the padding is a leg of a `Concat` with a
    /// type of its own, which is what lets `Concat`'s existing check see it.
    #[test]
    fn a_fill_extends_an_axis_as_a_concat_leg() {
        let ty = check_one(
            Expr::concat(
                0,
                vec![
                    Expr::src("k_proj"),
                    Expr::fill(0.0, TensorType::raw(vec![128, 2048], DType::BF16)),
                ],
            ),
            &qwen3(),
        )
        .unwrap();
        assert_eq!(ty.shape, vec![1152, 2048]);

        // And a leg whose other extents disagree is now a `Concat` error, where a
        // `Pad` could not have been wrong about them at all.
        let err = check_one(
            Expr::concat(
                0,
                vec![
                    Expr::src("k_proj"),
                    Expr::fill(0.0, TensorType::raw(vec![128, 2049], DType::BF16)),
                ],
            ),
            &qwen3(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("Concat"), "{err}");
    }

    #[test]
    fn a_fill_must_name_a_value_the_zeroing_can_write() {
        for (value, encoding, why) in [
            (
                1.0_f32,
                Encoding::Raw(DType::BF16),
                "a nonzero constant is not a run of zero bytes",
            ),
            (
                0.0,
                Encoding::Raw(DType::E8M0),
                "E8M0's zero byte is 2^-127, not zero",
            ),
            (
                0.0,
                Encoding::Quant(fp8_per_row()),
                "a code word means nothing without its scale",
            ),
        ] {
            let err = check_one(
                Expr::fill(
                    value,
                    TensorType {
                        shape: vec![4, 4],
                        encoding,
                    },
                ),
                &qwen3(),
            )
            .unwrap_err();
            assert!(err.to_string().contains("Fill"), "{why}: {err}");
        }
    }

    /// A fill has no operand, so there is nothing for a wildcard to be solved
    /// against and nothing for a zeroed FFI node to be mistaken for.
    #[test]
    fn a_fill_must_give_every_extent() {
        for shape in [vec![4, -1], vec![4, 0], vec![]] {
            let err = check_one(
                Expr::fill(0.0, TensorType::raw(shape.clone(), DType::BF16)),
                &qwen3(),
            )
            .unwrap_err();
            assert!(
                err.to_string().contains("extent") || err.to_string().contains("rank"),
                "{shape:?}: {err}"
            );
        }
    }

    /// A GPT-OSS expert block: `[experts, rows, groups, 16]` of packed nibbles.
    fn mxfp4_blocks() -> FakeCheckpoint {
        FakeCheckpoint::new(&[
            ("blocks", &[2, 16, 2, 16], DType::U8),
            ("scales", &[2, 16, 2], DType::U8),
        ])
    }

    /// A Marlin-swizzled MXFP4 weight: the operand's packed nibbles, moved.
    ///
    /// The target names MXFP4 because that is what the bytes are — a repack
    /// preserves the element width, so a `raw(BF16)` target over packed
    /// nibbles is the mis-sizing `repack_spec` refuses.
    fn marlin_target(rows: i64, cols: i64) -> TensorType {
        TensorType {
            shape: vec![2, rows, cols],
            encoding: Encoding::Quant(QuantSpec {
                scheme: QuantScheme::Mxfp4E2M1E8M0,
                logical_dtype: DType::BF16,
                bits_per_element: 4,
                group_size: 32,
                channel_axis: Some(Axis(1)),
            }),
        }
    }

    fn marlin_weight(src: Expr, rows: i64, cols: i64) -> Expr {
        src.repack(RepackLayout::MarlinMxfp4Weight, marlin_target(rows, cols))
    }

    /// What `Repack` stopped restating.
    ///
    /// The spec used to name the operand's own geometry back to the checker --
    /// `batch`, `source_rows`, `source_cols` -- so a contract could disagree
    /// with the tensor it was reading and nothing would notice until a kernel
    /// ran. There is no way to say it wrong now: the numbers *are* the operand's
    /// type, and the author writes only the layout and the destination.
    #[test]
    fn a_repack_derives_its_geometry_from_its_operand() {
        let checkpoint = mxfp4_blocks();
        let mut resolver = Resolver::new(&checkpoint, Partition::default());
        let ty = resolver
            .infer(&marlin_weight(Expr::src("blocks"), 32, 64), "w")
            .unwrap();
        assert_eq!(ty.shape, vec![2, 32, 64]);

        let operand = resolver.infer(&Expr::src("blocks"), "w").unwrap();
        let spec = repack_spec(
            &operand,
            RepackLayout::MarlinMxfp4Weight,
            &marlin_target(32, 64),
        )
        .unwrap();
        assert_eq!(spec.batch, 2);
        assert_eq!(spec.source_rows, 16);
        assert_eq!(spec.target_rows, 32);
        // Two groups of 32 packed elements is 64 logical columns.
        assert_eq!(spec.source_cols, 64);
        assert_eq!(spec.target_cols, 64);
    }

    /// The selection the escape hatch gave back.
    ///
    /// `Repack` used to require a bare `Expr::Src`, because the spec narrowed in
    /// checkpoint coordinates and a composed operand would have narrowed twice.
    /// With the narrowing gone the operand is free, which is what lets a repack
    /// be sharded by `Expr::Shard` like every other node instead of by a rank
    /// resolved into an integer before the contract is written.
    #[test]
    fn a_repack_selects_in_its_operand() {
        let checkpoint = mxfp4_blocks();
        let mut resolver = Resolver::new(&checkpoint, Partition::new(1, 2));

        // A shard, then the even rows of it: GPT-OSS's gate half, exactly.
        let half = Expr::src("blocks").shard(1).stride(1, 0, 4, 2);
        let ty = resolver
            .infer(&marlin_weight(half.clone(), 8, 64), "w")
            .unwrap();
        assert_eq!(ty.shape, vec![2, 8, 64]);

        let spec = repack_spec(
            &resolver.infer(&half, "w").unwrap(),
            RepackLayout::MarlinMxfp4Weight,
            &marlin_target(8, 64),
        )
        .unwrap();
        assert_eq!(
            spec.source_rows, 4,
            "the shard's half, then every other row"
        );
    }

    /// The invariant `Repack` was missing. It is opaque in one direction only:
    /// what the swizzle does to a byte is the kernel's business, but how many
    /// bytes come out is stated twice -- once in the declaration the buffer is
    /// sized from, once in the geometry the kernel writes according to -- and
    /// nothing made the two agree.
    ///
    /// Padding is the kernel's, so a target may be wider than the operand; a
    /// target that is *narrower* is a truncation, which is `Slice`'s job and
    /// not something a swizzle should be trusted to do.
    #[test]
    fn a_repack_target_may_pad_but_not_truncate() {
        let checkpoint = mxfp4_blocks();
        let mut resolver = Resolver::new(&checkpoint, Partition::default());
        resolver
            .infer(&marlin_weight(Expr::src("blocks"), 128, 128), "w")
            .expect("padding both axes is the kernel's zero fill");

        for (rows, cols) in [(8, 64), (32, 32)] {
            let err = resolver
                .infer(&marlin_weight(Expr::src("blocks"), rows, cols), "w")
                .unwrap_err();
            assert!(
                err.to_string().contains("smaller than"),
                "{rows}x{cols}: {err}"
            );
        }
    }

    /// The other factor of the same product. `to`'s *shape* was checked against
    /// the operand and its *encoding* was not, while the destination buffer is
    /// sized from both -- so a repack could name any element width it liked.
    ///
    /// Wider over-allocates in silence. Narrower under-allocates and the kernel
    /// writes past the end, which is the memory-safety fault the shape check
    /// was written to prevent, reached through the factor it did not cover.
    ///
    /// The rule is the algebra's own table: `Repack` is the kernel-priced
    /// member of the *placement* family, so it preserves type and value and
    /// moves only bytes. Padding columns is still allowed -- that adds
    /// elements, it does not resize them.
    #[test]
    fn a_repack_may_not_reinterpret_its_elements() {
        let checkpoint = mxfp4_blocks();
        let mut resolver = Resolver::new(&checkpoint, Partition::default());

        // [2, 16, 2, 16] U8 is 256 bits a row, 64 logical MXFP4 columns.
        for (encoding, why) in [
            (Encoding::Raw(DType::BF16), "four times too wide"),
            (Encoding::Raw(DType::U8), "twice too wide"),
        ] {
            let err = resolver
                .infer(
                    &Expr::src("blocks").repack(
                        RepackLayout::MarlinMxfp4Weight,
                        TensorType {
                            shape: vec![2, 32, 64],
                            encoding: encoding.clone(),
                        },
                    ),
                    "w",
                )
                .unwrap_err();
            assert!(
                err.to_string().contains("does not reinterpret them"),
                "{why}: {err}"
            );
        }

        // Padding the column axis leaves the element width alone, so it stays
        // legal -- the pad is more elements, not bigger ones.
        resolver
            .infer(&marlin_weight(Expr::src("blocks"), 32, 128), "w")
            .expect("a padded column count is not a reinterpretation");
    }

    /// Each layout reads one shape. Naming it in the type is what replaced the
    /// spec's `source_rows`/`source_cols`, so a mismatch is now a type error
    /// rather than a number nobody checked.
    #[test]
    fn a_repack_operand_must_have_the_shape_its_layout_reads() {
        let checkpoint = mxfp4_blocks();
        let mut resolver = Resolver::new(&checkpoint, Partition::default());

        // The scale tensor is rank 3; the weight layout reads rank 4.
        let err = resolver
            .infer(&marlin_weight(Expr::src("scales"), 32, 64), "w")
            .unwrap_err();
        assert!(err.to_string().contains("[B, R, K/32, 16]"), "{err}");

        let err = resolver
            .infer(
                &Expr::src("blocks").repack(
                    RepackLayout::MarlinMxfp4Scale,
                    TensorType::raw(vec![2, 32, 2], DType::U8),
                ),
                "s",
            )
            .unwrap_err();
        assert!(err.to_string().contains("[B, R, groups]"), "{err}");
    }

    /// A repack that names no kernel is no longer expressible in Rust, so the
    /// rule now lives where a zeroed field can still arrive:
    /// `a_repack_that_names_no_kernel_is_refused_at_the_boundary` in
    /// `ffi::tests`.
    #[test]
    fn a_cast_keeps_shape_and_replaces_encoding() {
        let ty = check_one(
            Expr::src("q_proj").cast(Encoding::Quant(fp8_per_row())),
            &qwen3(),
        )
        .unwrap();
        assert_eq!(ty.shape, vec![2048, 2048]);
        assert!(matches!(ty.encoding, Encoding::Quant(_)));
    }

    /// Both directions, and the pair matters: one node covers encoding and
    /// decoding because the destination is the whole of the question.
    #[test]
    fn a_cast_decodes_as_readily_as_it_encodes() {
        let quantized = Expr::src("q_proj").cast(Encoding::Quant(fp8_per_row()));
        let decoded = quantized.cast(Encoding::Raw(DType::BF16));
        let ty = check_one(decoded, &qwen3()).unwrap();
        assert_eq!(ty.shape, vec![2048, 2048]);
        assert_eq!(ty.encoding, Encoding::Raw(DType::BF16));
    }

    /// A cast to what the operand already is would be a kernel that does
    /// nothing, and it is more likely a mistake than a request.
    #[test]
    fn a_cast_to_the_encoding_already_held_is_rejected() {
        let err = check_one(
            Expr::src("q_proj").cast(Encoding::Raw(DType::BF16)),
            &qwen3(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("already is"), "{err}");
    }

    /// There is no kernel that goes scheme to scheme, and there is no obvious
    /// one either -- the destination's scales are not a function of the
    /// source's. Refused here so the two-step is written down.
    #[test]
    fn a_cast_between_quantized_schemes_is_rejected() {
        let mxfp4 = QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        };
        let err = check_one(
            Expr::src("q_proj")
                .cast(Encoding::Quant(fp8_per_row()))
                .cast(Encoding::Quant(mxfp4)),
            &qwen3(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("re-encodes Fp8E4M3"), "{err}");
        assert!(
            err.to_string().contains("cast to a raw type first"),
            "{err}"
        );
    }

    #[test]
    fn a_cast_needs_a_kernel() {
        assert!(
            !Expr::src("q_proj")
                .cast(Encoding::Raw(DType::F32))
                .is_affine()
        );
    }

    #[test]
    fn a_cast_rejects_a_ragged_group_axis() {
        let checkpoint = FakeCheckpoint::new(&[("w", &[64, 100], DType::BF16)]);
        let mxfp4 = QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        };
        let err = check_one(Expr::src("w").cast(Encoding::Quant(mxfp4)), &checkpoint).unwrap_err();
        assert!(err.to_string().contains("Cast groups axis 1 by 32"));
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
        };
        let quantized = Expr::src("w").cast(Encoding::Quant(mxfp4));
        // Aligned to the 32-element groups: fine.
        assert!(check_one(quantized.clone().slice(1, 32, 64), &checkpoint).is_ok());
        // Straddles them: rejected.
        let err = check_one(quantized.slice(1, 16, 64), &checkpoint).unwrap_err();
        assert!(err.to_string().contains("not aligned"));
    }

    #[test]
    fn affine_fragment_is_recognized() {
        let fused = Expr::concat(0, vec![Expr::src("q_proj"), Expr::src("k_proj")]);
        assert!(fused.is_affine());
        assert!(!fused.cast(Encoding::Quant(fp8_per_row())).is_affine());
    }

    #[test]
    fn scale_keeps_the_type_it_was_given() {
        let ty = check_one(Expr::src("q_proj").scale(0.25), &qwen3()).unwrap();
        assert_eq!(ty.shape, vec![2048, 2048]);
        assert_eq!(ty.encoding, Encoding::Raw(DType::BF16));
        // Multiplying a value is not a coordinate map, so it leaves the
        // fragment that compiles to byte runs.
        assert!(!Expr::src("q_proj").scale(0.25).is_affine());
    }

    #[test]
    fn scale_rejects_a_quantized_operand() {
        let scaled = Expr::src("q_proj")
            .cast(Encoding::Quant(fp8_per_row()))
            .scale(0.25);
        let err = check_one(scaled, &qwen3()).unwrap_err();
        assert!(err.to_string().contains("Scale of a quantized tensor"));
    }

    #[test]
    fn scale_rejects_an_integer_operand() {
        let checkpoint = FakeCheckpoint::new(&[("ids", &[16], DType::I32)]);
        let err = check_one(Expr::src("ids").scale(2.0), &checkpoint).unwrap_err();
        assert!(err.to_string().contains("I32"));
    }

    /// A NaN or infinite factor would silently poison every element, and the
    /// bit-pattern encoding means it survives the cache key and the FFI
    /// unchanged. Rejecting it here is the only place it can be caught.
    #[test]
    fn scale_rejects_a_factor_that_is_not_finite() {
        for factor in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = check_one(Expr::src("q_proj").scale(factor), &qwen3()).unwrap_err();
            assert!(err.to_string().contains("finite"), "{factor} was accepted");
        }
    }

    /// Zero is the resting value of the FFI's `scale_factor_bits`, so a C++
    /// author who forgets to set it must not get a tensor of zeros that loads
    /// and runs.
    #[test]
    fn scale_rejects_the_factor_an_unset_field_reads_as() {
        for factor in [0.0f32, -0.0f32] {
            let err = check_one(Expr::src("q_proj").scale(factor), &qwen3()).unwrap_err();
            assert!(err.to_string().contains("zero"), "{factor} was accepted");
        }
    }

    #[test]
    fn sources_are_reported_in_order() {
        let fused = Expr::concat(
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
            Expr::concat(
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
            alignment: 256,
            tensors,
            groups: Vec::new(),
        }
    }

    /// Resolve a run of contracts in declaration order, exactly as the frontend
    /// does: each entry's type is published under its name so that later
    /// entries can read it through [`Expr::Out`].
    fn resolve(
        tensors: &[TensorContract],
        checkpoint: &dyn CheckpointTypes,
    ) -> Result<Vec<TensorType>, Error> {
        let mut resolver = Resolver::new(checkpoint, Partition::WHOLE);
        tensors
            .iter()
            .map(|tensor| {
                let ty = resolver.infer(&tensor.expr, &tensor.name)?;
                resolver.publish(&tensor.name, ty.clone());
                Ok(ty)
            })
            .collect()
    }

    /// The type a contract that declares its shape claims to have.
    fn declared(contract: &TensorContract) -> TensorType {
        TensorType {
            shape: contract
                .shape
                .clone()
                .expect("test contract declares a shape"),
            encoding: contract.encoding.clone(),
        }
    }

    #[test]
    fn a_contract_resolves_to_what_it_declares() {
        let tensors = vec![qkv_contract(vec![4096, 2048])];
        let found = resolve(&tensors, &qwen3()).unwrap();
        assert_eq!(found[0], declared(&tensors[0]));
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
                    Expr::concat(
                        0,
                        vec![
                            Expr::src("q_proj"),
                            Expr::fill(0.0, TensorType::raw(vec![64, 2048], DType::BF16)),
                        ],
                    ),
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
        // declaration. The hand-written per-model fusion pass this replaced
        // refused all three in combination.
        let expr = Expr::concat(
            0,
            vec![
                Expr::src("q_proj").slice(0, 0, 1024),
                Expr::src("k_proj").slice(0, 0, 512),
                Expr::src("v_proj").slice(0, 0, 512),
            ],
        )
        .cast(Encoding::Quant(fp8_per_row()));
        let contract = model(vec![TensorContract::new(
            "model.layers.0.self_attn.qkv_proj.fused.weight",
            expr,
            vec![2048, 2048],
            Encoding::Quant(fp8_per_row()),
        )]);
        assert_eq!(
            resolve(&contract.tensors, &qwen3()).unwrap()[0],
            declared(&contract.tensors[0])
        );
    }
}
