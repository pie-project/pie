//! Contract-to-contract rewrites that run before the plan is built.
//!
//! These are optimizations, not semantics: each one returns a contract that
//! declares exactly the same tensors as the one it was given. They live in the
//! planner because they reason about *shape and cost*, never about which model
//! a tensor belongs to — a rewrite that needed the model's name would be the
//! driver's business, not the compiler's.

use crate::checkpoint::{CheckpointMetadata, RawTensor};
use crate::contract::{Expr, ModelContract, TensorContract, local_range};
use crate::error::{Error, OrOverflow};
use crate::plan::StorageTarget;
use crate::types::{Axis, Encoding};

/// Replace many equally-shaped row shards with one bank plus views of it.
///
/// A rank holding the same row band of a hundred identically-shaped weights
/// reads a hundred small strided copies; stated as one `Concat` of those bands
/// plus a `Slice` per member, it reads one. Purely an optimization: the
/// contract it returns declares exactly the same tensors.
pub fn coalesce_direct_row_shards(
    contract: &ModelContract,
    metadata: &CheckpointMetadata,
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
        // One pattern says everything the old flag-plus-match pair said:
        // this tensor is a whole checkpoint tensor, split by row.
        let Expr::Shard { src, axis: Axis(0) } = &tensor.expr else {
            continue;
        };
        let Expr::Src(name) = src.as_ref() else {
            continue;
        };
        let Some(raw) = metadata.tensor_by_name(name) else {
            continue;
        };
        // Extents come off the checkpoint, because the contract's shape is
        // already this rank's band and cannot say how wide the whole is.
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
        if tensor.shape.as_deref() != Some(&[local_rows, raw.shape[1]][..]) {
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
    metadata: &CheckpointMetadata,
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

    // The bank is the local row band of every member, end to end. Stated as
    // an expression the offsets are the compiler's problem, not this pass's.
    let mut parts = Vec::with_capacity(indices.len());
    for &old_index in indices {
        let raw = direct_raw(metadata, &contract.tensors[old_index])?;
        parts.push(Expr::src(raw.name.clone()).slice(0, local_start, local_rows));
    }

    let bank_name = format!("__pie.row_shard_bank.{group_id}");
    new_tensors.push(TensorContract::new(
        bank_name.clone(),
        Expr::concat(0, parts),
        vec![local_rows * indices.len() as i64, cols],
        first.encoding.clone(),
    ));

    for (slot, &old_index) in indices.iter().enumerate() {
        let original = &contract.tensors[old_index];
        let mut rebuilt = TensorContract::new(
            original.name.clone(),
            Expr::out(bank_name.clone()).slice(0, slot as i64 * local_rows, local_rows),
            vec![local_rows, cols],
            original.encoding.clone(),
        );
        // "Declares exactly the same tensors" is this module's whole license,
        // and a declaration is more than a name and a shape. Rebuilding a
        // member without its `scales` silently unpaired every coalesced
        // block-scale tensor from its FP8 weight — 230 of Qwen3.6-27B-FP8's
        // 263 quant attachments gone at tp 2, and every affected GEMM threw
        // "quant scale data is null" — and dropping `visibility` promoted
        // internal factors into the bind table.
        rebuilt.scales = original.scales.clone();
        rebuilt.visibility = original.visibility;
        new_tensors.push(rebuilt);
    }
    Ok(())
}

fn direct_raw<'a>(
    metadata: &'a CheckpointMetadata,
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

/// The checkpoint tensor a direct contract reads, seeing through the partition
/// a sharded one wraps it in — a rank's band of a tensor is still that tensor.
fn direct_src(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Src(name) => Some(name.as_str()),
        Expr::Shard { src, .. } => direct_src(src),
        _ => None,
    }
}

fn dense_element_bytes(raw: &RawTensor, context: &str) -> Result<u64, Error> {
    match &raw.encoding {
        Encoding::Raw(dtype) => Ok(dtype.bytes()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{CheckpointFile, RawTensor};
    use crate::contract::Scales;
    use crate::types::{
        CheckpointFormat, DType, FileId, QuantGranularity, ScaleForm, TensorId, Visibility,
    };

    /// A coalesced member is still the declaration it was: its `scales`
    /// pairing and its `visibility` ride along into the bank view. Losing the
    /// first silently unpaired 230 of Qwen3.6-27B-FP8's 263 block-scale
    /// attachments at tp 2 ("quant scale data is null" in every affected
    /// GEMM); losing the second promoted internal factors into the bind table.
    #[test]
    fn a_coalesced_member_keeps_its_scales_and_visibility() {
        let mut tensors = Vec::new();
        let mut contract_tensors = Vec::new();
        let mut offset = 0u64;
        // 16 members: `coalesce_direct_row_shards`'s MIN_GROUP_TENSORS.
        for index in 0..16 {
            let name = format!("layer.{index}.weight_scale_inv");
            let span = 8 * 4 * 2;
            tensors.push(RawTensor {
                id: TensorId(index as u32),
                name: name.clone(),
                file_id: FileId(0),
                file_offset: offset,
                span_bytes: span,
                shape: vec![8, 4],
                encoding: Encoding::Raw(DType::BF16),
            });
            offset += span;
            let mut entry = TensorContract::new(
                name.clone(),
                Expr::src(&name).shard(0),
                vec![4, 4],
                Encoding::Raw(DType::BF16),
            );
            entry.scales = Some(Scales {
                of: format!("layer.{index}.weight"),
                granularity: QuantGranularity::PerGroup,
                group_size: 128,
                channel_axis: 0,
                form: ScaleForm::F32Factors,
            });
            entry.visibility = Visibility::Internal;
            contract_tensors.push(entry);
        }
        let metadata = CheckpointMetadata {
            files: vec![CheckpointFile {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes: offset,
                format: CheckpointFormat::Safetensors,
            }],
            tensors,
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: contract_tensors,
            groups: Vec::new(),
        };
        let target = StorageTarget {
            tp_rank: 0,
            tp_size: 2,
            ..StorageTarget::default()
        };

        let rewritten = coalesce_direct_row_shards(&contract, &metadata, &target).unwrap();
        assert!(
            rewritten
                .tensors
                .iter()
                .any(|tensor| tensor.name.starts_with("__pie.row_shard_bank.")),
            "the fixture must actually trigger the coalesce"
        );
        for original in &contract.tensors {
            let member = rewritten
                .tensors
                .iter()
                .find(|tensor| tensor.name == original.name)
                .expect("every member is still declared");
            assert_eq!(member.scales, original.scales, "{}", original.name);
            assert_eq!(member.visibility, original.visibility, "{}", original.name);
        }
    }
}
