use std::collections::HashMap;

use crate::contract::compile::{Leaf, Lowering, RunSource};
use crate::contract::infer::Resolver;
use crate::contract::{
    Expr, ModelContract, Partition, ScaleFactor, TensorContract, Visibility, local_range,
};
use crate::error::{Error, OrOverflow};
use crate::file::{File, Metadata, RawTensor};
use crate::plan::StorageTarget;
use crate::types::{Axis, CheckpointFormat, Encoding, FileId, TensorId};

pub fn coalesce_direct_row_shards(
    contract: &ModelContract,
    metadata: &Metadata,
    target: &StorageTarget,
) -> Result<ModelContract, Error> {
    if target.tp_size <= 1 {
        return Ok(contract.clone());
    }

    const MIN_GROUP_TENSORS: usize = 16;
    const DEFAULT_MAX_BANK_BYTES: u64 = 4 * 1024 * 1024 * 1024;
    let max_bank_bytes = DEFAULT_MAX_BANK_BYTES;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct GroupKey {
        shape: Vec<i64>,
        encoding: Encoding,
    }

    let mut buckets: Vec<(GroupKey, Vec<usize>)> = Vec::new();
    let mut local_bytes_by_index = vec![0_u64; contract.tensors.len()];
    for (index, tensor) in contract.tensors.iter().enumerate() {
        let Expr::Shard {
            src,
            axis: Axis(0),
            replicas: 1,
        } = &tensor.expr
        else {
            continue;
        };
        let Expr::Src(name) = src.as_ref() else {
            continue;
        };
        let Some(raw) = metadata.tensor_by_name(name) else {
            continue;
        };
        if raw.shape.len() != 2 || raw.shape[0] <= 0 || raw.shape[1] <= 0 {
            continue;
        }
        if raw.encoding != tensor.encoding {
            continue;
        }
        let elem = match dense_element_bytes(raw, "direct row shard coalescing") {
            Ok(elem) => elem,
            Err(_) => continue,
        };
        let (_, local_rows) = local_range(
            raw.shape[0],
            target.tp_size,
            target.tp_rank,
            &format!("the row count of '{}'", tensor.name),
        )?;
        if tensor.shape.as_deref() != Some(&raw.shape[..]) {
            continue;
        }
        let row_bytes =
            checked_mul_i64(raw.shape[1], elem, "direct row shard coalescing row bytes")?;
        local_bytes_by_index[index] = checked_mul_i64(
            local_rows,
            row_bytes,
            "direct row shard coalescing local bytes",
        )?;
        let key = GroupKey {
            shape: raw.shape.clone(),
            encoding: tensor.encoding.clone(),
        };
        if let Some((_, indices)) = buckets.iter_mut().find(|(candidate, _)| *candidate == key) {
            indices.push(index);
        } else {
            buckets.push((key, vec![index]));
        }
    }

    let mut group_for = vec![None; contract.tensors.len()];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (_, indices) in &buckets {
        let mut chunk = Vec::new();
        let mut chunk_bytes = 0_u64;
        for &index in indices {
            let tensor_bytes = local_bytes_by_index[index];
            if !chunk.is_empty()
                && chunk_bytes.saturating_add(tensor_bytes) > max_bank_bytes
                && chunk.len() >= MIN_GROUP_TENSORS
            {
                let group_id = groups.len();
                for &member in &chunk {
                    group_for[member] = Some(group_id);
                }
                groups.push(std::mem::take(&mut chunk));
                chunk_bytes = 0;
            }
            chunk.push(index);
            chunk_bytes = chunk_bytes.saturating_add(tensor_bytes);
        }
        if chunk.len() >= MIN_GROUP_TENSORS {
            let group_id = groups.len();
            for &member in &chunk {
                group_for[member] = Some(group_id);
            }
            groups.push(chunk);
        }
    }

    if groups.is_empty() {
        return Ok(contract.clone());
    }

    let mut emitted_groups = vec![false; groups.len()];
    let mut new_tensors = Vec::with_capacity(contract.tensors.len() + groups.len());

    for (tensor, group) in contract.tensors.iter().zip(&group_for) {
        if let Some(group_id) = *group {
            if emitted_groups[group_id] {
                continue;
            }
            emitted_groups[group_id] = true;
            emit_row_shard_bank(
                contract,
                metadata,
                target,
                group_id,
                &groups[group_id],
                &mut new_tensors,
            )?;
            continue;
        }

        new_tensors.push(tensor.clone());
    }

    Ok(ModelContract {
        tensors: new_tensors,
        ..contract.clone()
    })
}

fn emit_row_shard_bank(
    contract: &ModelContract,
    metadata: &Metadata,
    target: &StorageTarget,
    group_id: usize,
    indices: &[usize],
    new_tensors: &mut Vec<TensorContract>,
) -> Result<(), Error> {
    let first = &contract.tensors[indices[0]];
    let first_raw = direct_raw(metadata, first)?;
    let rows = first_raw.shape[0];
    let cols = first_raw.shape[1];
    dense_element_bytes(first_raw, "direct row shard coalescing")?;
    let (local_start, local_rows) = local_range(
        rows,
        target.tp_size,
        target.tp_rank,
        &format!("the row count of '{}'", first.name),
    )?;

    let mut parts = Vec::with_capacity(indices.len());
    for &old_index in indices {
        let raw = direct_raw(metadata, &contract.tensors[old_index])?;
        parts.push(Expr::src(raw.name.clone()).slice(0, local_start, local_rows));
    }

    // The bank is the compiler's own staging buffer: no plan names it, so it
    // must never be published (an engine's sink refuses a name it does not
    // index — "the load contract publishes `__pie.row_shard_bank.0`, which
    // this plan does not name"). The members keep everything the author
    // declared besides their expression and shape: a scale plane that loses
    // its `scaling(...)` binds no quantized weight any more.
    let bank_name = format!("__pie.row_shard_bank.{group_id}");
    new_tensors.push(
        TensorContract::new(
            bank_name.clone(),
            Expr::concat(0, parts),
            vec![local_rows * indices.len() as i64, cols],
            first.encoding.clone(),
        )
        .internal(),
    );

    for (slot, &old_index) in indices.iter().enumerate() {
        let original = &contract.tensors[old_index];
        new_tensors.push(TensorContract {
            expr: Expr::out(bank_name.clone()).slice(0, slot as i64 * local_rows, local_rows),
            shape: Some(vec![local_rows, cols]),
            ..original.clone()
        });
    }
    Ok(())
}

fn direct_raw<'a>(
    metadata: &'a Metadata,
    contract: &TensorContract,
) -> Result<&'a RawTensor, Error> {
    let Some(name) = direct_src(&contract.expr) else {
        return Err(Error::Contract(format!(
            "runtime tensor '{}' is not a direct tensor",
            contract.name
        )));
    };
    metadata.tensor_by_name(name).ok_or_else(|| {
        Error::Contract(format!(
            "runtime tensor '{}' references missing source tensor '{name}'",
            contract.name
        ))
    })
}

fn direct_src(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Src(name) => Some(name.as_str()),
        Expr::Shard { src, .. } => direct_src(src),
        _ => None,
    }
}

fn dense_element_bytes(raw: &RawTensor, context: &str) -> Result<u64, Error> {
    match &raw.encoding {
        Encoding::Raw(dtype) => Ok(dtype.bytes_ceil()),
        Encoding::Quant(spec) => spec.dense_element_bytes().ok_or_else(|| {
            Error::Contract(format!(
                "{context} '{}' has non-affine packed encoding",
                raw.name
            ))
        }),
    }
}

fn checked_mul_i64(lhs: i64, rhs: u64, context: &str) -> Result<u64, Error> {
    let lhs =
        u64::try_from(lhs).map_err(|_| Error::Contract(format!("{context}: negative value")))?;
    lhs.checked_mul(rhs)
        .or_overflow(format!("{context}: byte overflow"))
}

/// A block-scaled FP8 plane is read out through the factors beside it only
/// when its Cast reads it straight from the checkpoint; a Cast over a Concat
/// of such planes (a gate and up leg fused) would read the joined codes with
/// no factors at all. Each leg is cast on its own, into a tensor of its own,
/// and the legs are joined after; a Cast this cannot split is refused.
pub fn cast_block_scaled_fp8_legs_apart(
    contract: &ModelContract,
    metadata: &Metadata,
) -> Result<ModelContract, Error> {
    let sources = crate::file::Sources::new(metadata);
    let scaled = |expr: &Expr| {
        expr.sources().into_iter().any(|name| {
            sources
                .by_name(name)
                .is_some_and(|raw| sources.fp8_block_scale(raw).is_some())
        })
    };
    if !contract.tensors.iter().any(|tensor| scaled(&tensor.expr)) {
        return Ok(contract.clone());
    }
    let mut tensors = Vec::with_capacity(contract.tensors.len());
    for tensor in &contract.tensors {
        let mut legs = Vec::new();
        let expr = split_casts(tensor.expr.clone(), &tensor.name, &scaled, &mut legs)?;
        tensors.extend(legs);
        tensors.push(TensorContract {
            expr,
            ..tensor.clone()
        });
    }
    Ok(ModelContract {
        tensors,
        ..contract.clone()
    })
}

fn split_casts(
    expr: Expr,
    name: &str,
    scaled: &impl Fn(&Expr) -> bool,
    legs: &mut Vec<TensorContract>,
) -> Result<Expr, Error> {
    let expr = expr.map_children(|child| split_casts(child, name, scaled, legs))?;
    let Expr::Cast { src, to, chunks } = expr else {
        return Ok(expr);
    };
    let mut joined_inside = false;
    src.visit(&mut |inner| {
        if let Expr::Concat { parts, .. } = inner {
            joined_inside |= parts.iter().any(scaled);
        }
    });
    if !joined_inside {
        return Ok(Expr::Cast { src, to, chunks });
    }
    let Expr::Concat { axis, parts } = *src else {
        return Err(Error::Contract(format!(
            "'{name}' casts block-scaled FP8 planes joined under another \
             operation, which would read their codes with no factors"
        )));
    };
    if chunks != 0 {
        return Err(Error::Contract(format!(
            "'{name}' casts block-scaled FP8 planes joined and cut into {chunks} \
             shards at once; state the cut on each plane"
        )));
    }
    // an encode reads the joined plane at its logical float, each leg read
    // out to that float first
    let leg_to = match &to {
        Encoding::Quant(spec) => Encoding::Raw(spec.logical_dtype),
        raw => raw.clone(),
    };
    let mut joined = Vec::with_capacity(parts.len());
    for part in parts {
        let own = scaled(&part);
        let cast = Expr::Cast {
            src: Box::new(part),
            to: leg_to.clone(),
            chunks: 0,
        };
        if !own {
            joined.push(cast);
            continue;
        }
        let leg = format!("{name}.fp8_leg{}", legs.len());
        legs.push(TensorContract::inferred(leg.clone(), cast, leg_to.clone()).internal());
        joined.push(Expr::Out(leg));
    }
    let joined = Expr::Concat {
        axis,
        parts: joined,
    };
    Ok(if leg_to == to {
        joined
    } else {
        Expr::Cast {
            src: Box::new(joined),
            to,
            chunks: 0,
        }
    })
}

/// Pushes outer affine operators (`Slice`, `Shard`, `Stride`, `Gather`) inward
/// through `Expr::Scale` via scale equivariance:
/// `A(Scale_G(e_q, e_s)) -> Scale_G(A(e_q), A_{/G}(e_s))`
/// whenever `A` is congruent with the scale block grid `G`.
pub fn push_affine_into_scale(
    contract: &ModelContract,
    metadata: &Metadata,
    target: &StorageTarget,
) -> Result<ModelContract, Error> {
    if !contract
        .tensors
        .iter()
        .any(|t| needs_scale_rewrite(&t.expr))
    {
        return Ok(contract.clone());
    }
    let sources = crate::file::Sources::new(metadata);
    let mut resolver = Resolver::new(&sources, Partition::WHOLE);
    let mut declared: HashMap<String, TensorContract> = HashMap::new();
    let mut tensors = Vec::with_capacity(contract.tensors.len());

    for tensor in &contract.tensors {
        let mut legs = Vec::new();
        let expr = rewrite_scale_expr(
            tensor.expr.clone(),
            &tensor.name,
            target,
            &mut resolver,
            &mut declared,
            &mut legs,
        )?;
        tensors.extend(legs);
        let updated = TensorContract {
            expr,
            ..tensor.clone()
        };
        if let Ok(ty) = resolver.infer_whole(&updated.expr, &updated.name) {
            resolver.publish(&updated.name, ty);
        }
        declared.insert(updated.name.clone(), updated.clone());
        tensors.push(updated);
    }
    Ok(ModelContract {
        tensors,
        ..contract.clone()
    })
}

fn needs_scale_rewrite(expr: &Expr) -> bool {
    match expr {
        Expr::Scale { src, factor } => {
            contains_scale(src)
                || matches!(
                    factor,
                    ScaleFactor::PerBlock { by } if !matches!(by.as_ref(), Expr::Out(_))
                )
        }
        other => contains_scale(other),
    }
}

fn contains_scale(expr: &Expr) -> bool {
    let mut found = false;
    expr.visit(&mut |node| {
        if matches!(node, Expr::Scale { .. }) {
            found = true;
        }
    });
    found
}

fn rewrite_scale_expr(
    expr: Expr,
    name: &str,
    target: &StorageTarget,
    resolver: &mut Resolver<'_>,
    declared: &mut HashMap<String, TensorContract>,
    legs: &mut Vec<TensorContract>,
) -> Result<Expr, Error> {
    let expr = expr.map_children(|child| {
        rewrite_scale_expr(child, name, target, resolver, declared, legs)
    })?;

    if let Expr::Scale {
        src,
        factor: ScaleFactor::PerBlock { by },
    } = expr
    {
        if !matches!(by.as_ref(), Expr::Out(_)) {
            let by_ty = resolver.infer_whole(&by, name)?;
            let leg_name = format!("{name}.scale_factor{}", legs.len());
            let leg =
                TensorContract::inferred(leg_name.clone(), *by, by_ty.encoding.clone()).internal();
            resolver.publish(&leg_name, by_ty);
            declared.insert(leg_name.clone(), leg.clone());
            legs.push(leg);
            return Ok(Expr::Scale {
                src,
                factor: ScaleFactor::PerBlock {
                    by: Box::new(Expr::Out(leg_name)),
                },
            });
        }
        return Ok(Expr::Scale {
            src,
            factor: ScaleFactor::PerBlock { by },
        });
    }

    match expr {
        Expr::Slice {
            src,
            axis,
            start,
            len,
        } if matches!(src.as_ref(), Expr::Scale { .. }) => {
            let Expr::Scale {
                src: q_src,
                factor,
            } = *src
            else {
                unreachable!()
            };
            push_one_affine(
                *q_src,
                factor,
                axis,
                AffineStep::Slice { start, len },
                name,
                target,
                resolver,
                declared,
                legs,
            )
        }
        Expr::Shard {
            src,
            axis,
            replicas,
        } if matches!(src.as_ref(), Expr::Scale { .. }) => {
            let Expr::Scale {
                src: q_src,
                factor,
            } = *src
            else {
                unreachable!()
            };
            push_one_affine(
                *q_src,
                factor,
                axis,
                AffineStep::Shard { replicas },
                name,
                target,
                resolver,
                declared,
                legs,
            )
        }
        Expr::Stride {
            src,
            axis,
            start,
            len,
            step,
        } if matches!(src.as_ref(), Expr::Scale { .. }) => {
            let Expr::Scale {
                src: q_src,
                factor,
            } = *src
            else {
                unreachable!()
            };
            push_one_affine(
                *q_src,
                factor,
                axis,
                AffineStep::Stride { start, len, step },
                name,
                target,
                resolver,
                declared,
                legs,
            )
        }
        Expr::Gather { src, axis, indices } if matches!(src.as_ref(), Expr::Scale { .. }) => {
            let Expr::Scale {
                src: q_src,
                factor,
            } = *src
            else {
                unreachable!()
            };
            push_one_affine(
                *q_src,
                factor,
                axis,
                AffineStep::Gather { indices },
                name,
                target,
                resolver,
                declared,
                legs,
            )
        }
        other => Ok(other),
    }
}

enum AffineStep {
    Slice { start: i64, len: i64 },
    Shard { replicas: u32 },
    Stride { start: i64, len: i64, step: i64 },
    Gather { indices: Vec<i64> },
}

#[allow(clippy::too_many_arguments)]
fn push_one_affine(
    q_src: Expr,
    factor: ScaleFactor,
    axis: Axis,
    step: AffineStep,
    name: &str,
    target: &StorageTarget,
    resolver: &mut Resolver<'_>,
    declared: &mut HashMap<String, TensorContract>,
    legs: &mut Vec<TensorContract>,
) -> Result<Expr, Error> {
    let ScaleFactor::PerBlock { by } = factor else {
        let new_q = match step {
            AffineStep::Slice { start, len } => q_src.slice(axis.0, start, len),
            AffineStep::Shard { replicas } => q_src.shard_among(axis.0, replicas),
            AffineStep::Stride { start, len, step } => q_src.stride(axis.0, start, len, step),
            AffineStep::Gather { indices } => q_src.gather(axis.0, indices),
        };
        return Ok(Expr::Scale {
            src: Box::new(new_q),
            factor,
        });
    };

    let q_ty = resolver.infer_whole(&q_src, name)?;
    let s_ty = resolver.infer_whole(&by, name)?;
    if q_ty.rank() != s_ty.rank() {
        return Err(Error::Contract(format!(
            "'{name}': Scale factors have shape {:?}, which is not a blocking of {:?}",
            s_ty.shape, q_ty.shape
        )));
    }
    let axis_idx = usize::from(axis.0);
    if axis_idx >= q_ty.rank() {
        return Err(Error::Contract(format!(
            "'{name}': axis {} out of bounds for rank-{} tensor",
            axis.0,
            q_ty.rank()
        )));
    }
    let q_dim = q_ty.shape[axis_idx];
    let s_dim = s_ty.shape[axis_idx];
    if s_dim <= 0 || q_dim % s_dim != 0 {
        return Err(Error::Contract(format!(
            "'{name}': Scale factors have shape {:?}, which does not divide axis {} of {:?}",
            s_ty.shape, axis.0, q_ty.shape
        )));
    }
    let g_a = q_dim / s_dim;

    let s_base = match by.as_ref() {
        Expr::Out(s_name) => declared
            .get(s_name)
            .filter(|c| c.expr.is_affine())
            .map(|c| c.expr.clone())
            .unwrap_or_else(|| (*by).clone()),
        _ => (*by).clone(),
    };

    let (new_q, new_s) = match step {
        AffineStep::Slice { start, len } => {
            if start % g_a != 0 || len % g_a != 0 {
                return Err(Error::Contract(format!(
                    "'{name}': Slice [{start}, {}) along axis {} splits {g_a}-element scale blocks",
                    start + len,
                    axis.0
                )));
            }
            let s_start = start / g_a;
            let s_len = len / g_a;
            let new_s = if s_start == 0 && s_len == s_dim {
                s_base
            } else {
                s_base.slice(axis.0, s_start, s_len)
            };
            (q_src.slice(axis.0, start, len), new_s)
        }
        AffineStep::Shard { replicas } => {
            let partition = Partition::new(target.tp_rank, target.tp_size).grouped(replicas)?;
            if partition.world > 1 {
                let (start, len) = local_range(
                    q_dim,
                    partition.world,
                    partition.rank,
                    &format!("'{name}' along axis {}", axis.0),
                )?;
                if s_dim > 1 && (start % g_a != 0 || len % g_a != 0) {
                    return Err(Error::Contract(format!(
                        "'{name}': Shard along axis {} (tp_size={}) splits {g_a}-element scale blocks",
                        axis.0, target.tp_size
                    )));
                }
            }
            let new_s = if s_dim == 1 {
                s_base
            } else {
                s_base.shard_among(axis.0, replicas)
            };
            (q_src.shard_among(axis.0, replicas), new_s)
        }
        AffineStep::Stride { start, len, step } => {
            if g_a > 1 {
                return Err(Error::Contract(format!(
                    "'{name}': Stride with step {step} along axis {} splits {g_a}-element scale blocks",
                    axis.0
                )));
            }
            (
                q_src.stride(axis.0, start, len, step),
                s_base.stride(axis.0, start, len, step),
            )
        }
        AffineStep::Gather { indices } => {
            if g_a == 1 {
                let new_s = simplify_index_selection(s_base, axis.0, &indices, s_dim);
                (q_src.gather(axis.0, indices), new_s)
            } else {
                let g_usize = usize::try_from(g_a).unwrap_or(1);
                if indices.is_empty() || !indices.len().is_multiple_of(g_usize) {
                    return Err(Error::Contract(format!(
                        "'{name}': Gather of {} indices along axis {} is not a multiple of {g_a}-element scale blocks",
                        indices.len(),
                        axis.0
                    )));
                }
                let mut block_indices = Vec::with_capacity(indices.len() / g_usize);
                for chunk in indices.chunks_exact(g_usize) {
                    let first = chunk[0];
                    if first < 0 || first % g_a != 0 {
                        return Err(Error::Contract(format!(
                            "'{name}': Gather along axis {} scrambles elements across {g_a}-element scale blocks",
                            axis.0
                        )));
                    }
                    for (u, &idx) in chunk.iter().enumerate() {
                        if idx != first + u as i64 {
                            return Err(Error::Contract(format!(
                                "'{name}': Gather along axis {} scrambles elements across {g_a}-element scale blocks",
                                axis.0
                            )));
                        }
                    }
                    block_indices.push(first / g_a);
                }
                let new_s = simplify_index_selection(s_base, axis.0, &block_indices, s_dim);
                (q_src.gather(axis.0, indices), new_s)
            }
        }
    };

    let leg_name = format!("{name}.scale_factor{}", legs.len());
    let new_s_ty = resolver.infer_whole(&new_s, &leg_name)?;
    let leg = TensorContract::inferred(leg_name.clone(), new_s, new_s_ty.encoding.clone()).internal();
    resolver.publish(&leg_name, new_s_ty);
    declared.insert(leg_name.clone(), leg.clone());
    legs.push(leg);

    Ok(Expr::Scale {
        src: Box::new(new_q),
        factor: ScaleFactor::PerBlock {
            by: Box::new(Expr::Out(leg_name)),
        },
    })
}

fn simplify_index_selection(base: Expr, axis: u8, indices: &[i64], full_extent: i64) -> Expr {
    if indices.len() as i64 == full_extent
        && indices.iter().enumerate().all(|(i, &idx)| idx == i as i64)
    {
        return base;
    }
    let Some((&first, rest)) = indices.split_first() else {
        return base;
    };
    if rest.is_empty() {
        return base.slice(axis, first, 1);
    }
    let step = rest[0] - first;
    if step >= 1 && rest.iter().zip(indices).all(|(next, prev)| next - prev == step) {
        if step == 1 {
            return base.slice(axis, first, indices.len() as i64);
        }
        return base.stride(axis, first, indices.len() as i64, step);
    }
    base.gather(axis, indices.to_vec())
}

#[derive(Clone, Debug)]
struct InverseRowRun {
    src_row: i64,
    row_len: i64,
    train_tensor: String,
    dst_row: i64,
    train_total_rows: i64,
}

/// Synthesizes the direct cross-contract bridge `C_infer o C_train^-1` from
/// a materialized trainer layout (`train_contract` across `train_tp_size`
/// ranks) to a target inference layout (`infer_contract`).
///
/// Returns `(trainer_metadata, bridge_contract)` where `trainer_metadata`
/// describes the trainer's materialized in-memory buffers at `virtual_path`
/// and `bridge_contract` expresses `infer_contract` directly over the
/// trainer's materialized tensors.
pub fn compose_bridge(
    canonical_metadata: &Metadata,
    train_contract: &ModelContract,
    infer_contract: &ModelContract,
    train_tp_size: u32,
    virtual_path: &str,
) -> Result<(Metadata, ModelContract), Error> {
    let train_world = train_tp_size.max(1);
    let canonical_sources = crate::file::Sources::new(canonical_metadata);
    let align = u64::from(train_contract.alignment.max(256));

    let mut trainer_tensors = Vec::new();
    let mut inverse_runs: HashMap<String, Vec<InverseRowRun>> = HashMap::new();
    let mut next_tensor_id = 0_u32;
    let mut file_offset = 0_u64;

    for rank in 0..train_world {
        let partition = Partition::new(rank, train_world);
        let mut resolver = Resolver::new(&canonical_sources, partition);
        for tensor in &train_contract.tensors {
            let spec_expr = resolver.specialize(tensor.expr.clone(), &tensor.name)?;
            let ty = resolver.infer(&spec_expr, &tensor.name)?;
            resolver.publish(&tensor.name, ty.clone());
            if tensor.visibility != Visibility::Public {
                continue;
            }
            let span_bytes = ty.byte_size()?;
            let rem = file_offset % align;
            if rem != 0 {
                file_offset = file_offset.saturating_add(align - rem);
            }
            let train_name = if train_world == 1 {
                tensor.name.clone()
            } else {
                format!("rank{rank}.{}", tensor.name)
            };
            let tid = TensorId(next_tensor_id);
            next_tensor_id += 1;
            trainer_tensors.push(RawTensor {
                id: tid,
                name: train_name.clone(),
                file_id: FileId(0),
                file_offset,
                span_bytes,
                shape: ty.shape.clone(),
                encoding: ty.encoding.clone(),
            });
            file_offset = file_offset.saturating_add(span_bytes);

            if spec_expr.is_affine() {
                let lowering =
                    crate::contract::compile::compile(&spec_expr, resolver.checked(), 1 << 20)?;
                let train_total_rows = ty.shape.first().copied().unwrap_or(1);
                match lowering {
                    Lowering::Copy(copies) => {
                        for run in &copies.runs {
                            let RunSource::Leaf { leaf, src_elem } = run.source else {
                                continue;
                            };
                            let Leaf::Checkpoint(src_name) = &copies.leaves[leaf] else {
                                continue;
                            };
                            let Some(raw) = canonical_metadata.tensor_by_name(src_name) else {
                                continue;
                            };
                            let row_elems: i64 = raw.shape.iter().skip(1).product::<i64>().max(1);
                            if src_elem % row_elems != 0
                                || run.dst_elem % row_elems != 0
                                || run.len % row_elems != 0
                            {
                                return Err(Error::Contract(format!(
                                    "trainer tensor '{train_name}' slices '{src_name}' at sub-row granularity"
                                )));
                            }
                            inverse_runs
                                .entry(src_name.clone())
                                .or_default()
                                .push(InverseRowRun {
                                    src_row: src_elem / row_elems,
                                    row_len: run.len / row_elems,
                                    train_tensor: train_name.clone(),
                                    dst_row: run.dst_elem / row_elems,
                                    train_total_rows,
                                });
                        }
                    }
                    Lowering::Gather(gather) => {
                        let Leaf::Checkpoint(src_name) = &gather.leaves[gather.leaf] else {
                            continue;
                        };
                        let Some(raw) = canonical_metadata.tensor_by_name(src_name) else {
                            continue;
                        };
                        if gather.rows == 1 {
                            for (dst_idx, &src_idx) in gather.indices.iter().enumerate() {
                                inverse_runs.entry(src_name.clone()).or_default().push(
                                    InverseRowRun {
                                        src_row: src_idx,
                                        row_len: 1,
                                        train_tensor: train_name.clone(),
                                        dst_row: dst_idx as i64,
                                        train_total_rows: raw.shape.first().copied().unwrap_or(1),
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    let trainer_metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: virtual_path.to_string(),
            size_bytes: file_offset,
            format: CheckpointFormat::Zt,
        }],
        tensors: trainer_tensors,
    };

    let mut inverse_exprs: HashMap<String, Expr> = HashMap::new();
    for tensor in &infer_contract.tensors {
        for src_name in tensor.expr.sources() {
            if inverse_exprs.contains_key(src_name) {
                continue;
            }
            let raw = canonical_metadata.tensor_by_name(src_name).ok_or_else(|| {
                Error::Checkpoint(format!("missing canonical source tensor '{src_name}'"))
            })?;
            let total_rows = raw.shape.first().copied().unwrap_or(1);
            let mut runs = inverse_runs.get(src_name).cloned().ok_or_else(|| {
                Error::Contract(format!(
                    "canonical source tensor '{src_name}' is not materialized by the trainer contract"
                ))
            })?;
            runs.sort_by_key(|r| (r.src_row, r.dst_row));

            let mut merged: Vec<InverseRowRun> = Vec::new();
            let mut cursor = 0_i64;
            for run in runs {
                if run.src_row + run.row_len <= cursor {
                    continue;
                }
                if run.src_row > cursor {
                    return Err(Error::Contract(format!(
                        "canonical source tensor '{src_name}' has unmapped row interval [{cursor}, {}) in trainer contract",
                        run.src_row
                    )));
                }
                let trim = cursor - run.src_row;
                let trimmed = InverseRowRun {
                    src_row: cursor,
                    row_len: run.row_len - trim,
                    train_tensor: run.train_tensor,
                    dst_row: run.dst_row + trim,
                    train_total_rows: run.train_total_rows,
                };
                cursor = trimmed.src_row + trimmed.row_len;
                if let Some(last) = merged.last_mut()
                    && last.train_tensor == trimmed.train_tensor
                    && last.dst_row + last.row_len == trimmed.dst_row
                {
                    last.row_len += trimmed.row_len;
                } else {
                    merged.push(trimmed);
                }
            }
            if cursor < total_rows {
                return Err(Error::Contract(format!(
                    "canonical source tensor '{src_name}' has unmapped trailing rows [{cursor}, {total_rows}) in trainer contract"
                )));
            }

            let mut parts: Vec<Expr> = merged
                .into_iter()
                .map(|r| {
                    if r.dst_row == 0 && r.row_len == r.train_total_rows {
                        Expr::src(r.train_tensor)
                    } else {
                        Expr::src(r.train_tensor).slice(0, r.dst_row, r.row_len)
                    }
                })
                .collect();
            let inv_expr = if parts.len() == 1 {
                parts.remove(0)
            } else {
                Expr::concat(0, parts)
            };
            inverse_exprs.insert(src_name.to_string(), inv_expr);
        }
    }

    let trainer_sources = crate::file::Sources::new(&trainer_metadata);
    let mut bridge_tensors = Vec::with_capacity(infer_contract.tensors.len());
    for tensor in &infer_contract.tensors {
        let substituted = substitute_sources(tensor.expr.clone(), &inverse_exprs, &trainer_sources)?;
        bridge_tensors.push(TensorContract {
            expr: substituted,
            ..tensor.clone()
        });
    }

    Ok((
        trainer_metadata,
        ModelContract {
            tensors: bridge_tensors,
            ..infer_contract.clone()
        },
    ))
}

fn substitute_sources(
    expr: Expr,
    inverse_exprs: &HashMap<String, Expr>,
    trainer_sources: &crate::file::Sources<'_>,
) -> Result<Expr, Error> {
    let expr = expr.map_children(|child| substitute_sources(child, inverse_exprs, trainer_sources))?;
    match expr {
        Expr::Src(name) => inverse_exprs.get(&name).cloned().ok_or_else(|| {
            Error::Contract(format!(
                "missing inverse mapping for source tensor '{name}'"
            ))
        }),
        Expr::Slice {
            src,
            axis: Axis(0),
            start,
            len,
        } => {
            if let Expr::Slice {
                src: inner_src,
                axis: Axis(0),
                start: inner_start,
                len: _inner_len,
            } = *src
            {
                let combined_start = inner_start + start;
                if let Expr::Src(ref train_name) = *inner_src
                    && let Some(raw) = trainer_sources.by_name(train_name)
                    && combined_start == 0
                    && len == raw.shape[0]
                {
                    return Ok(*inner_src);
                }
                return Ok(inner_src.slice(0, combined_start, len));
            }
            if let Expr::Src(ref train_name) = *src
                && let Some(raw) = trainer_sources.by_name(train_name)
                && start == 0
                && len == raw.shape[0]
            {
                return Ok(*src);
            }
            Ok(Expr::Slice {
                src,
                axis: Axis(0),
                start,
                len,
            })
        }
        other => Ok(other),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use super::*;
    use crate::contract::{AxisLeg, TensorType};
    use crate::executor::Execution;
    use crate::plan::{DestExtent, Extent, StorageInstr, compile};
    use crate::types::{BufferId, DType, InstrId, QuantScheme, QuantSpec};
    use crate::verify::{ContractView, verify_plan};

    fn raw_tensor(
        id: u32,
        name: &str,
        offset: u64,
        shape: &[i64],
        dtype: DType,
    ) -> RawTensor {
        let elems: u64 = shape.iter().map(|&d| d as u64).product();
        RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: offset,
            span_bytes: elems * dtype.bytes_ceil(),
            shape: shape.to_vec(),
            encoding: Encoding::Raw(dtype),
        }
    }

    #[test]
    fn scale_equivariance_pushes_slice_shard_and_block_congruent_gather_inward() {
        let vpath = "/memfs/test_scale_equivariance.bin";
        ztensor::memfs::mount_chunks(vpath, vec![Arc::from(vec![0u8; 4096])]);
        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: vpath.to_string(),
                size_bytes: 4096,
                format: CheckpointFormat::Safetensors,
            }],
            tensors: vec![
                raw_tensor(0, "w", 0, &[8, 16], DType::U8),
                raw_tensor(1, "s", 256, &[4, 1], DType::U8),
            ],
        };
        let quant = Encoding::Quant(QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::Bf16,
            group_size: 32,
            bits_per_element: 4,
            channel_axis: Some(Axis(1)),
        });

        // 1. Valid Shard pushed inward through Scale_G (g_0 = 2, g_1 = 32) at tp_size = 2 (each rank gets 4 rows = 2 blocks)
        let contract_ok = ModelContract {
            alignment: 256,
            tensors: vec![
                TensorContract::new(
                    "scales",
                    Expr::src("s").transmute(TensorType::new(vec![4, 1], Encoding::Raw(DType::E8m0))),
                    vec![4, 1],
                    Encoding::Raw(DType::E8m0),
                )
                .internal(),
                TensorContract::new(
                    "w_sharded",
                    Expr::src("w")
                        .transmute(TensorType::new(vec![8, 32], quant.clone()))
                        .scale_per_block(Expr::out("scales"))
                        .shard(0),
                    vec![8, 32],
                    Encoding::Raw(DType::Bf16),
                ),
            ],
            groups: Vec::new(),
        };
        let target_tp2 = StorageTarget {
            tp_size: 2,
            tp_rank: 1,
            ..StorageTarget::default()
        };
        let plan = compile(&metadata, &contract_ok, target_tp2).expect("congruent Shard(Scale) should compile");
        verify_plan(&plan, Some(&ContractView::of(&contract_ok))).expect("plan should verify");

        // 2. Invalid Slice splitting a 2-row scale block on axis 0 (start = 1, len = 4)
        let contract_bad_slice = ModelContract {
            alignment: 256,
            tensors: vec![
                TensorContract::new(
                    "scales",
                    Expr::src("s").transmute(TensorType::new(vec![4, 1], Encoding::Raw(DType::E8m0))),
                    vec![4, 1],
                    Encoding::Raw(DType::E8m0),
                )
                .internal(),
                TensorContract::new(
                    "w_sliced",
                    Expr::src("w")
                        .transmute(TensorType::new(vec![8, 32], quant.clone()))
                        .scale_per_block(Expr::out("scales"))
                        .slice(0, 1, 4),
                    vec![4, 32],
                    Encoding::Raw(DType::Bf16),
                ),
            ],
            groups: Vec::new(),
        };
        let err = compile(&metadata, &contract_bad_slice, StorageTarget::default()).unwrap_err();
        assert!(err.to_string().contains("splits 2-element scale blocks"), "{err}");

        // 3. Valid block-congruent Gather on axis 0 (swapping 2-row blocks: [2, 3, 0, 1, 6, 7, 4, 5])
        let contract_good_gather = ModelContract {
            alignment: 256,
            tensors: vec![
                TensorContract::new(
                    "scales",
                    Expr::src("s").transmute(TensorType::new(vec![4, 1], Encoding::Raw(DType::E8m0))),
                    vec![4, 1],
                    Encoding::Raw(DType::E8m0),
                )
                .internal(),
                TensorContract::new(
                    "w_gathered",
                    Expr::src("w")
                        .transmute(TensorType::new(vec![8, 32], quant.clone()))
                        .scale_per_block(Expr::out("scales"))
                        .gather(0, vec![2, 3, 0, 1, 6, 7, 4, 5]),
                    vec![8, 32],
                    Encoding::Raw(DType::Bf16),
                ),
            ],
            groups: Vec::new(),
        };
        let plan_g = compile(&metadata, &contract_good_gather, StorageTarget::default())
            .expect("block-congruent Gather(Scale) should compile");
        verify_plan(&plan_g, Some(&ContractView::of(&contract_good_gather))).expect("plan should verify");

        // 4. Invalid interleaved Gather on axis 0 ([0, 2, 4, 6, 1, 3, 5, 7]) scrambling 2-row scale blocks
        let contract_bad_gather = ModelContract {
            alignment: 256,
            tensors: vec![
                TensorContract::new(
                    "scales",
                    Expr::src("s").transmute(TensorType::new(vec![4, 1], Encoding::Raw(DType::E8m0))),
                    vec![4, 1],
                    Encoding::Raw(DType::E8m0),
                )
                .internal(),
                TensorContract::new(
                    "w_bad_gather",
                    Expr::src("w")
                        .transmute(TensorType::new(vec![8, 32], quant))
                        .scale_per_block(Expr::out("scales"))
                        .gather(0, vec![0, 2, 4, 6, 1, 3, 5, 7]),
                    vec![8, 32],
                    Encoding::Raw(DType::Bf16),
                ),
            ],
            groups: Vec::new(),
        };
        let err_g = compile(&metadata, &contract_bad_gather, StorageTarget::default()).unwrap_err();
        assert!(err_g.to_string().contains("scrambles elements across 2-element scale blocks"), "{err_g}");
    }

    #[test]
    fn composite_sum_axis_rejects_flat_shard_even_when_divisible() {
        let vpath = "/memfs/test_composite_qkv.bin";
        ztensor::memfs::mount_chunks(vpath, vec![Arc::from(vec![0u8; 4096])]);
        // Q = [16, 8], K = [4, 8], V = [4, 8]; total rows = 24, which IS divisible by tp_size = 8!
        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: vpath.to_string(),
                size_bytes: 4096,
                format: CheckpointFormat::Safetensors,
            }],
            tensors: vec![
                raw_tensor(0, "q", 0, &[16, 8], DType::Bf16),
                raw_tensor(1, "k", 256, &[4, 8], DType::Bf16),
                raw_tensor(2, "v", 320, &[4, 8], DType::Bf16),
                raw_tensor(3, "fused_qkv", 384, &[24, 8], DType::Bf16),
            ],
        };
        let target = StorageTarget {
            tp_size: 8,
            tp_rank: 0,
            ..StorageTarget::default()
        };

        // Case A: Flat Shard wrapping heterogeneous Concat([16, 4, 4]) at tp_size = 8 (24 % 8 == 0)
        let bad_concat_shard = ModelContract {
            alignment: 256,
            tensors: vec![TensorContract::new(
                "qkv",
                Expr::concat(0, vec![Expr::src("q"), Expr::src("k"), Expr::src("v")]).shard(0),
                vec![24, 8],
                Encoding::Raw(DType::Bf16),
            )],
            groups: Vec::new(),
        };
        let err = compile(&metadata, &bad_concat_shard, target.clone()).unwrap_err();
        assert!(err.to_string().contains("composite Concat legs"), "{err}");

        // Case B: Pre-fused checkpoint tensor with declared composite sum-axis type 16^(r=1) + 4^(r=2) + 4^(r=2)
        let bad_prefused = ModelContract {
            alignment: 256,
            tensors: vec![
                TensorContract::new(
                    "qkv",
                    Expr::src("fused_qkv").shard(0),
                    vec![24, 8],
                    Encoding::Raw(DType::Bf16),
                )
                .with_composite_axis(
                    0,
                    vec![
                        AxisLeg::new(16, 1),
                        AxisLeg::new(4, 2),
                        AxisLeg::new(4, 2),
                    ],
                ),
            ],
            groups: Vec::new(),
        };
        let err2 = compile(&metadata, &bad_prefused, target).unwrap_err();
        assert!(err2.to_string().contains("composite sum-axis type"), "{err2}");
    }

    #[test]
    fn invertible_bridge_c_infer_after_c_train_inv_matches_cold_load_bit_for_bit() {
        // Canonical checkpoint: q=[16, 4], k=[8, 4], v=[8, 4], gate=[16, 4], up=[16, 4] (F32)
        let canonical_path = "/memfs/canonical_bridge_test.bin";
        let mut raw_bytes = Vec::with_capacity(4096);
        for i in 0..1024u32 {
            raw_bytes.extend_from_slice(&((i as f32) * 0.25 + 1.0).to_le_bytes());
        }
        ztensor::memfs::mount_chunks(canonical_path, vec![Arc::from(raw_bytes.clone())]);

        let canonical_meta = Metadata {
            files: vec![File {
                id: FileId(0),
                path: canonical_path.to_string(),
                size_bytes: raw_bytes.len() as u64,
                format: CheckpointFormat::Safetensors,
            }],
            tensors: vec![
                raw_tensor(0, "q", 0, &[16, 4], DType::F32),
                raw_tensor(1, "k", 256, &[8, 4], DType::F32),
                raw_tensor(2, "v", 384, &[8, 4], DType::F32),
                raw_tensor(3, "gate", 512, &[16, 4], DType::F32),
                raw_tensor(4, "up", 768, &[16, 4], DType::F32),
            ],
        };

        // Trainer contract C_train (at TP_train = 2): unfused q, k, v, gate, up sharded across 2 trainer ranks
        let f32_enc = Encoding::Raw(DType::F32);
        let train_contract = ModelContract {
            alignment: 256,
            tensors: vec![
                TensorContract::new("train_q", Expr::src("q").shard(0), vec![16, 4], f32_enc.clone()),
                TensorContract::new("train_k", Expr::src("k").shard(0), vec![8, 4], f32_enc.clone()),
                TensorContract::new("train_v", Expr::src("v").shard(0), vec![8, 4], f32_enc.clone()),
                TensorContract::new("train_gate", Expr::src("gate").shard(0), vec![16, 4], f32_enc.clone()),
                TensorContract::new("train_up", Expr::src("up").shard(0), vec![16, 4], f32_enc.clone()),
            ],
            groups: Vec::new(),
        };

        // Inference contract C_infer (at TP_infer = 4): fused QKV (with GQA r_kv = 2 on K, V!) and fused gate_up
        let infer_contract = ModelContract {
            alignment: 256,
            tensors: vec![
                TensorContract::new(
                    "infer_qkv",
                    Expr::concat(
                        0,
                        vec![
                            Expr::src("q").shard_among(0, 1),
                            Expr::src("k").shard_among(0, 2),
                            Expr::src("v").shard_among(0, 2),
                        ],
                    ),
                    vec![32, 4],
                    f32_enc.clone(),
                ),
                TensorContract::new(
                    "infer_gate_up",
                    Expr::concat(
                        0,
                        vec![
                            Expr::src("gate").shard_among(0, 1),
                            Expr::src("up").shard_among(0, 1),
                        ],
                    ),
                    vec![32, 4],
                    f32_enc,
                ),
            ],
            groups: Vec::new(),
        };

        // Materialize trainer ranks 0 and 1 into a contiguous trainer HBM buffer
        let trainer_vpath = "/memfs/trainer_hbm_state.bin";
        let (trainer_meta, bridge_contract) = compose_bridge(
            &canonical_meta,
            &train_contract,
            &infer_contract,
            2,
            trainer_vpath,
        )
        .expect("compose_bridge should succeed");

        let mut trainer_hbm = vec![0u8; trainer_meta.files[0].size_bytes as usize];
        for train_rank in 0..2u32 {
            let t_target = StorageTarget {
                tp_size: 2,
                tp_rank: train_rank,
                ..StorageTarget::default()
            };
            let t_plan = compile(&canonical_meta, &train_contract, t_target).unwrap();
            let t_exec = Execution::new(&t_plan, Path::new("/")).run().unwrap();
            for (t_name, t_bytes) in &t_exec.tensors {
                let full_name = format!("rank{train_rank}.{t_name}");
                let raw = trainer_meta.tensor_by_name(&full_name).unwrap();
                let start = raw.file_offset as usize;
                trainer_hbm[start..start + t_bytes.len()].copy_from_slice(t_bytes);
            }
        }
        ztensor::memfs::mount_chunks(trainer_vpath, vec![Arc::from(trainer_hbm)]);

        // Now verify that for EVERY inference rank 0..4, executing the synthesized bridge C_infer o C_train^-1
        // produces bit-for-bit identical weights to cold-loading C_infer from canonical_meta!
        for infer_rank in 0..4u32 {
            let i_target = StorageTarget {
                tp_size: 4,
                tp_rank: infer_rank,
                ..StorageTarget::default()
            };
            let cold_plan = compile(&canonical_meta, &infer_contract, i_target.clone()).unwrap();
            let bridge_plan = compile(&trainer_meta, &bridge_contract, i_target).unwrap();

            verify_plan(&bridge_plan, Some(&ContractView::of(&bridge_contract)))
                .expect("synthesized bridge plan must pass validator");

            let cold_out = Execution::new(&cold_plan, Path::new("/")).run().unwrap();
            let bridge_out = Execution::new(&bridge_plan, Path::new("/")).run().unwrap();
            assert_eq!(
                bridge_out.tensors, cold_out.tensors,
                "bridge output mismatch on infer_rank {infer_rank}"
            );
            assert_eq!(
                bridge_out.arena, cold_out.arena,
                "bridge arena mismatch on infer_rank {infer_rank}"
            );
        }

        // Also test byte-hole detection in verify_plan on a corrupted sub-tensor write interval
        let mut hole_plan = compile(
            &trainer_meta,
            &bridge_contract,
            StorageTarget {
                tp_size: 4,
                tp_rank: 0,
                ..StorageTarget::default()
            },
        )
        .unwrap();
        // Shrink one ExtentWrite destination extent so a 16-byte hole remains in the finalized buffer
        for instr in &mut hole_plan.instrs {
            if let StorageInstr::ExtentWrite { dest, .. } = instr {
                dest.stride = Extent::byte_run(16);
                break;
            }
        }
        let violations = verify_plan(&hole_plan, Some(&ContractView::of(&bridge_contract)))
            .expect_err("validator must reject sub-tensor byte hole");
        assert!(
            violations.iter().any(|v| v.message.contains("unwritten byte hole")),
            "expected unwritten byte hole violation, got: {violations:#?}"
        );
        let _ = (BufferId(0), InstrId(0), DestExtent { buffer: BufferId(0), offset: 0, stride: Extent::byte_run(0) });
    }
}
