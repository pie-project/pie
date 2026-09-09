use crate::error::{Error, OrOverflow};
use crate::types::{Axis, DType, Encoding, RepackLayout, RepackSpec, TILED_BAND, TILED_STEP};

use super::compile;
use super::{
    BiasBy, Expr, Partition, ScaleFactor, TensorType, UnaryOp, local_range, resolve_extents,
};

pub trait CheckpointTypes {
    fn tensor_type(&self, name: &str) -> Option<TensorType>;
}

#[derive(Clone, Debug, Default)]
pub struct Checked {
    pub sources: std::collections::HashMap<String, TensorType>,
    pub outputs: std::collections::HashMap<String, TensorType>,
}

impl Checked {
    pub fn source(&self, name: &str) -> Option<&TensorType> {
        self.sources.get(name)
    }

    pub fn output(&self, name: &str) -> Option<&TensorType> {
        self.outputs.get(name)
    }

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
    instance: Option<u32>,
    what: String,
}

impl Scope<'_> {
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

pub fn infer_type(
    expr: &Expr,
    checkpoint: &dyn CheckpointTypes,
) -> Result<(TensorType, Checked), Error> {
    let mut resolver = Resolver::new(checkpoint, Partition::WHOLE);
    let ty = resolver.infer(expr, "expression")?;
    Ok((ty, resolver.into_checked()))
}

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

    pub fn for_instance(mut self, instance: u32) -> Self {
        self.scope.instance = Some(instance);
        self
    }

    pub fn infer(&mut self, expr: &Expr, what: &str) -> Result<TensorType, Error> {
        self.scope.what.clear();
        self.scope.what.push_str(what);
        infer(expr, &mut self.scope)
    }

    pub fn infer_whole(&mut self, expr: &Expr, what: &str) -> Result<TensorType, Error> {
        let partition = std::mem::replace(&mut self.scope.partition, Partition::WHOLE);
        let ty = self.infer(expr, what);
        self.scope.partition = partition;
        ty
    }

    pub fn specialize(&mut self, expr: Expr, what: &str) -> Result<Expr, Error> {
        self.scope.what.clear();
        self.scope.what.push_str(what);
        specialize(expr, &mut self.scope)
    }

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
        Expr::Unary { src, op } => {
            let ty = infer(src, scope)?;
            infer_unary(ty, *op)
        }
        Expr::Bias { src, by } => {
            let ty = infer(src, scope)?;
            match by {
                BiasBy::Uniform(bits) => infer_bias(ty, *bits),
                BiasBy::PerBlock { by } => {
                    if !matches!(by.as_ref(), Expr::Out(_)) {
                        return Err(Error::Contract(
                            "Bias addends must be a declared tensor; declare \
                             them first and bias by that name"
                                .to_string(),
                        ));
                    }
                    let by_ty = infer(by, scope)?;
                    infer_bias_per_block(ty, by_ty)
                }
            }
        }
        Expr::Scale { src, factor } => {
            let ty = infer(src, scope)?;
            match factor {
                ScaleFactor::Uniform(bits) => infer_scale(ty, *bits),
                ScaleFactor::PerBlock { by } => {
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
            if scope.partition.world <= 1 {
                return Ok(ty);
            }
            infer_slice(&ty, *axis, start, len)
        }
    }
}

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

fn specialize(expr: Expr, scope: &mut Scope<'_>) -> Result<Expr, Error> {
    if let Expr::Transmute { src, to } = expr {
        let src = specialize(*src, scope)?;
        let ty = infer(&src, scope)?;
        let to = infer_transmute(&ty, &to, &src)?;
        return Ok(src.transmute(to));
    }
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
        infer_slice(&ty, axis, start, len)?;
        return Ok(src.slice(axis.0, start, len));
    }
    let Expr::Shard { src, axis } = expr else {
        return expr.map_children(|src| specialize(src, scope));
    };
    let src = specialize(*src, scope)?;
    if scope.partition.world <= 1 {
        return Ok(src);
    }
    let ty = infer(&src, scope)?;
    let (start, len) = shard_range(&ty, axis, scope.partition, &scope.what)?;
    Ok(src.slice(axis.0, start, len))
}

fn axis_index(axis: Axis, rank: usize, what: &str) -> Result<usize, Error> {
    let index = usize::from(axis.0);
    if index >= rank {
        return Err(Error::Contract(format!(
            "{what} names axis {index} of a rank-{rank} tensor"
        )));
    }
    Ok(index)
}

fn blocked_axis(ty: &TensorType) -> Option<(usize, i64)> {
    let Encoding::Quant(spec) = &ty.encoding else {
        return None;
    };
    if let Some((elems, _)) = spec.block_layout() {
        let last = ty.rank().checked_sub(1)?;
        let group = i64::try_from(elems).ok()?;
        return (group > 1).then_some((last, group));
    }
    let channel = usize::from(spec.channel_axis?.0);
    let group = i64::from(spec.normalized_group_size());
    (group > 1).then_some((channel, group))
}

fn block_granularity(ty: &TensorType, axis: usize) -> Option<i64> {
    blocked_axis(ty)
        .filter(|&(index, _)| index == axis)
        .map(|(_, group)| group)
}

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
    if let Some(group) = block_granularity(ty, index)
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
    if len == 1 {
        return Err(Error::Contract(format!(
            "Stride of one position from {start} is a Slice, which costs less to \
             lower; say that instead"
        )));
    }
    let index = selected_axis(ty, axis, start, len, step, "Stride")?;
    if let Some(group) = block_granularity(ty, index) {
        return Err(Error::Contract(format!(
            "Stride with step {step} on quantized axis {index} would split its {group}-element groups"
        )));
    }
    Ok(narrowed(ty, index, len))
}

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
    if let Some(group) = block_granularity(ty, index) {
        return Err(Error::Contract(format!(
            "Gather on quantized axis {index} would split its {group}-element groups"
        )));
    }
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
    if let Some(group) = block_granularity(head, index) {
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

fn element_bits(encoding: &Encoding) -> Option<u64> {
    match encoding {
        Encoding::Raw(dtype) => dtype.bytes_ceil().checked_mul(8),
        Encoding::Quant(spec) => Some(u64::from(spec.clone().normalized().normalized_bits())),
    }
}

fn blocked_suffix(ty: &TensorType) -> Option<&[i64]> {
    let (channel, _) = blocked_axis(ty)?;
    ty.shape.get(channel..)
}

fn elements_in(bytes: u64, encoding: &Encoding) -> Result<i64, Error> {
    if let Encoding::Quant(spec) = encoding
        && let Some((elems, block)) = spec.block_layout()
    {
        if block == 0 || !bytes.is_multiple_of(block) {
            return Err(Error::Contract(format!(
                "Transmute of {bytes} bytes is not a whole number of {block}-byte blocks"
            )));
        }
        return i64::try_from(bytes / block * elems).or_overflow("Transmute element count");
    }
    let bits = element_bits(encoding).ok_or_else(|| {
        Error::Contract(format!("Transmute to {encoding:?} has no element width"))
    })?;
    let total_bits = bytes.checked_mul(8).or_overflow("Transmute byte size")?;
    if !total_bits.is_multiple_of(bits) {
        return Err(Error::Contract(format!(
            "Transmute of {bytes} bytes does not divide into {bits}-bit elements"
        )));
    }
    i64::try_from(total_bits / bits).or_overflow("Transmute element count")
}

fn infer_transmute(ty: &TensorType, to: &TensorType, src: &Expr) -> Result<TensorType, Error> {
    let from_bytes = ty.byte_size()?;
    let total = elements_in(from_bytes, &to.encoding)?;
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
    if resolved == *ty {
        return Err(denotes_its_operand(
            "Transmute",
            "renames its operand to the type it already has",
        ));
    }
    if element_bits(&ty.encoding) != element_bits(&to.encoding)
        && !matches!(src, Expr::Src(_) | Expr::Out(_))
    {
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
    if value != 0.0_f32.to_bits() || *dtype == DType::E8m0 {
        return Err(Error::Contract(format!(
            "Fill of {} as {dtype:?} is not a run of zero bytes, which is all \
             the zeroing can write",
            f32::from_bits(value)
        )));
    }
    Ok(ty.clone())
}

pub(crate) fn repack_spec(
    ty: &TensorType,
    layout: RepackLayout,
    to: &TensorType,
) -> Result<RepackSpec, Error> {
    let to_rank: usize = match layout {
        RepackLayout::MarlinMxfp4Weight | RepackLayout::MarlinMxfp4Scale => 3,
        RepackLayout::TiledAffineU4Weight | RepackLayout::TiledAffineFactor => 2,
    };
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
        RepackLayout::TiledAffineU4Weight => {
            if ty.rank() != 2 {
                return Err(Error::Contract(format!(
                    "TiledAffineU4Weight Repack operand must be [rows, k], got {:?}",
                    ty.shape
                )));
            }
            if ty.shape[1] % i64::from(TILED_STEP) != 0 {
                return Err(Error::Contract(format!(
                    "TiledAffineU4Weight Repack reads a {}-wide row, which is not a whole \
                     number of {TILED_STEP}-wide contraction steps",
                    ty.shape[1]
                )));
            }
            (2, ty.shape[1])
        }
        RepackLayout::TiledAffineFactor => {
            if ty.rank() != 2 {
                return Err(Error::Contract(format!(
                    "TiledAffineFactor Repack operand must be [rows, groups], got {:?}",
                    ty.shape
                )));
            }
            (2, ty.shape[1])
        }
    };
    debug_assert_eq!(ty.rank(), want_rank);
    if to.rank() != to_rank {
        return Err(Error::Contract(format!(
            "Repack declares {:?}; a {layout:?} repack produces {}",
            to.shape,
            if to_rank == 3 {
                "[batch, rows, cols]"
            } else {
                "[rows, cols]"
            }
        )));
    }
    let (batch, rows) = if to_rank == 3 {
        (ty.shape[0], ty.shape[1])
    } else {
        (1, ty.shape[0])
    };
    if to_rank == 3 && to.shape[0] != batch {
        return Err(Error::Contract(format!(
            "Repack operand has batch {batch} but declares {:?}",
            to.shape
        )));
    }
    let (to_rows, to_cols) = (to.shape[to_rank - 2], to.shape[to_rank - 1]);
    if to_rank == 2 {
        let banded = rows
            .checked_add(i64::from(TILED_BAND) - 1)
            .map_or(rows, |up| {
                up / i64::from(TILED_BAND) * i64::from(TILED_BAND)
            });
        if to_rows != banded {
            return Err(Error::Contract(format!(
                "{layout:?} Repack of {rows} rows lands {banded} -- the next whole \
                 {TILED_BAND}-column band -- and declares {to_rows}"
            )));
        }
    }
    if to_rows < rows || to_cols < cols {
        return Err(Error::Contract(format!(
            "Repack declares {:?}, smaller than the [{batch}, {rows}, {cols}] it \
             reads; narrow the operand instead",
            to.shape
        )));
    }
    let source_bits = row_bits(&ty.shape[to_rank - 1..], &ty.encoding, "Repack operand")?;
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
        target_rows: dim_u32(to_rows, "Repack target rows")?,
        source_cols: dim_u32(cols, "Repack source columns")?,
        target_cols: dim_u32(to_cols, "Repack target columns")?,
    })
}

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
    if !matches!(dtype, DType::F32 | DType::F16 | DType::Bf16) {
        return Err(Error::Contract(format!(
            "Scale requires F32, F16 or BF16 elements, got {dtype:?}"
        )));
    }
    let factor = f32::from_bits(factor_bits);
    if !factor.is_finite() {
        return Err(Error::Contract(format!(
            "Scale factor must be finite, got {factor}"
        )));
    }
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

fn infer_unary(ty: TensorType, op: UnaryOp) -> Result<TensorType, Error> {
    let dtype = match ty.encoding {
        Encoding::Raw(dtype) => dtype,
        Encoding::Quant(_) => {
            return Err(Error::Contract(format!(
                "{op:?} of a quantized tensor ({:?}) is not supported; a code \
                 word means nothing to apply a function to until its scales \
                 are named",
                ty.encoding
            )));
        }
    };
    if !matches!(dtype, DType::F32 | DType::F16 | DType::Bf16) {
        return Err(Error::Contract(format!(
            "{op:?} requires F32, F16 or BF16 elements, got {dtype:?}"
        )));
    }
    Ok(ty)
}

fn infer_bias(ty: TensorType, by_bits: u32) -> Result<TensorType, Error> {
    let dtype = match ty.encoding {
        Encoding::Raw(dtype) => dtype,
        Encoding::Quant(_) => {
            return Err(Error::Contract(format!(
                "Bias of a quantized tensor ({:?}) is not supported; a code \
                 word means nothing to add to until its scales are named",
                ty.encoding
            )));
        }
    };
    if !matches!(dtype, DType::F32 | DType::F16 | DType::Bf16) {
        return Err(Error::Contract(format!(
            "Bias requires F32, F16 or BF16 elements, got {dtype:?}"
        )));
    }
    let by = f32::from_bits(by_bits);
    if !by.is_finite() {
        return Err(Error::Contract(format!("Bias must be finite, got {by}")));
    }
    if by == 0.0 {
        return Err(Error::Contract(
            "Bias is zero, which is also what an unset constant field reads \
             as; state the constant the contract meant"
                .to_string(),
        ));
    }
    Ok(ty)
}

fn infer_bias_per_block(ty: TensorType, by: TensorType) -> Result<TensorType, Error> {
    let dtype = match &ty.encoding {
        Encoding::Raw(dtype) => *dtype,
        Encoding::Quant(_) => {
            return Err(Error::Contract(format!(
                "Bias of a quantized tensor ({:?}) is not supported; scale it \
                 per block first -- the scale is what turns codes into numbers \
                 an addend can reach",
                ty.encoding
            )));
        }
    };
    if !matches!(dtype, DType::F32 | DType::F16 | DType::Bf16) {
        return Err(Error::Contract(format!(
            "Bias requires F32, F16 or BF16 elements, got {dtype:?}"
        )));
    }
    match &by.encoding {
        Encoding::Raw(DType::F32 | DType::F16 | DType::Bf16) => {}
        other => {
            return Err(Error::Contract(format!(
                "Bias addends must be raw F32, F16 or BF16 elements, got {other:?}"
            )));
        }
    }
    if by.shape.len() != ty.shape.len() {
        return Err(Error::Contract(format!(
            "Bias addends have shape {:?}, which is not a blocking of {:?} \
             -- an addend tensor states its block size by having the same rank \
             and dividing each axis",
            by.shape, ty.shape
        )));
    }
    let mut blocked = false;
    for (axis, (&extent, &addends)) in ty.shape.iter().zip(by.shape.iter()).enumerate() {
        if addends <= 0 {
            return Err(Error::Contract(format!(
                "Bias addends have extent {addends} on axis {axis}, so no \
                 block size divides it"
            )));
        }
        if extent % addends != 0 {
            return Err(Error::Contract(format!(
                "Bias addends have shape {:?}, but axis {axis} of {:?} is not \
                 a whole number of blocks of {}",
                by.shape,
                ty.shape,
                extent / addends
            )));
        }
        if addends != extent {
            blocked = true;
        }
    }
    if !blocked && !ty.shape.is_empty() {
        return Err(Error::Contract(format!(
            "Bias addends have the operand's own shape {:?}, so they group \
             nothing; an addend per element is an elementwise sum, not a \
             block bias",
            ty.shape
        )));
    }
    Ok(ty)
}

fn infer_scale_per_block(ty: TensorType, by: TensorType) -> Result<TensorType, Error> {
    let out_dtype = match &ty.encoding {
        Encoding::Raw(dtype) => *dtype,
        Encoding::Quant(spec) => spec.logical_dtype,
    };
    if !matches!(out_dtype, DType::F32 | DType::F16 | DType::Bf16) {
        return Err(Error::Contract(format!(
            "Scale requires F32, F16 or BF16 elements, got {out_dtype:?}"
        )));
    }
    match &by.encoding {
        Encoding::Raw(DType::F32 | DType::F16 | DType::Bf16 | DType::E8m0) => {}
        other => {
            return Err(Error::Contract(format!(
                "Scale factors must be raw F32, F16, BF16 or E8M0 elements, got {other:?}"
            )));
        }
    }
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
