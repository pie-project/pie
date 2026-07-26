use crate::checkpoint::{CheckpointMetadata, RawTensor};
use crate::config::ModelConfig;
use crate::contract::{Expr, ModelContract, TensorContract};
use crate::error::CompileError;
use crate::load_plan::StorageTarget;
use crate::types::{
    Axis, BackendKind, DType, Encoding, Mxfp4MoePolicy, QuantScheme, QuantSpec, RepackLayout,
    RepackSpec, RowMap, TensorId,
};
use std::collections::HashSet;

mod fusion;
mod gpt_oss;
mod metal;
mod nemotron;
mod phi3;
mod policy;
mod profile;
mod qwen_moe;

use policy::{runtime_quant_model_supported, runtime_quantizable_name};
use profile::{ArchProfile, arch_profile};

/// Infer a contract for `cfg`'s family, from the tensors `metadata` names.
///
/// The one function in the loader that knows what a model *is* — everything
/// below it works from the contract alone. It exists so that a driver which has
/// not yet been taught to author its own contract still gets one.
pub fn default_contract(
    metadata: &CheckpointMetadata,
    cfg: &ModelConfig,
    target: &StorageTarget,
) -> Result<ModelContract, CompileError> {
    let mut builder = ContractBuilder {
        metadata,
        cfg,
        target,
        consumed: HashSet::new(),
        tensors: Vec::new(),
    };
    builder.build()?;
    let sharded = builder
        .tensors
        .iter()
        .filter(|contract| contract.expr.is_sharded())
        .count();
    let (total_bytes, sharded_bytes) =
        builder
            .tensors
            .iter()
            .fold((0_u64, 0_u64), |(total, sharded), contract| {
                let bytes = match contract.expr.sources().as_slice() {
                    [name] => metadata
                        .tensor_by_name(name)
                        .map(|raw| raw.span_bytes)
                        .unwrap_or(0),
                    _ => 0,
                };
                (
                    total.saturating_add(bytes),
                    sharded.saturating_add(if contract.expr.is_sharded() { bytes } else { 0 }),
                )
            });
    if crate::planner_debug_enabled() {
        eprintln!(
            "[pie-loader] default ABI model_type={} tp={}/{} tensors={} sharded={} bytes={} sharded_bytes={}",
            cfg.model_type,
            target.tp_rank,
            target.tp_size,
            builder.tensors.len(),
            sharded,
            total_bytes,
            sharded_bytes
        );
    }
    Ok(ModelContract {
        abi_version: 1,
        alignment: target.preferred_alignment.max(1),
        tensors: builder.tensors,
    })
}

/// Replace many equally-shaped row shards with one bank plus views of it.
///
/// A rank holding the same row band of a hundred identically-shaped weights
/// reads a hundred small strided copies; stated as one `Cat` of those bands
/// plus a `Slice` per member, it reads one. Purely an optimization: the
/// contract it returns declares exactly the same tensors.
pub fn coalesce_direct_row_shards(
    contract: &ModelContract,
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
) -> Result<ModelContract, CompileError> {
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
            target,
            &format!("the row count of '{}'", tensor.name),
        )?;
        if tensor.shape != [local_rows, raw.shape[1]] {
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

    if crate::planner_debug_enabled() {
        let coalesced = groups.iter().map(Vec::len).sum::<usize>();
        eprintln!(
            "[pie-loader] row-shard coalescing groups={} tensors={} max_bank_bytes={}",
            groups.len(),
            coalesced,
            max_bank_bytes
        );
    }

    let mut emitted_groups = vec![false; groups.len()];
    let mut old_to_new = vec![usize::MAX; contract.tensors.len()];
    let mut new_tensors = Vec::with_capacity(contract.tensors.len() + groups.len());

    for old_index in 0..contract.tensors.len() {
        if let Some(group_id) = group_for[old_index] {
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
                &mut old_to_new,
                &mut new_tensors,
            )?;
            continue;
        }

        old_to_new[old_index] = new_tensors.len();
        new_tensors.push(contract.tensors[old_index].clone());
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
    old_to_new: &mut [usize],
    new_tensors: &mut Vec<TensorContract>,
) -> Result<(), CompileError> {
    let first = &contract.tensors[indices[0]];
    let first_raw = direct_raw(metadata, first)?;
    let rows = first_raw.shape[0];
    let cols = first_raw.shape[1];
    dense_element_bytes(first_raw, "direct row shard coalescing")?;
    let (local_start, local_rows) =
        local_range(rows, target, &format!("the row count of '{}'", first.name))?;

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
        Expr::cat(0, parts),
        vec![local_rows * indices.len() as i64, cols],
        first.encoding.clone(),
    ));

    for (slot, &old_index) in indices.iter().enumerate() {
        let original = &contract.tensors[old_index];
        old_to_new[old_index] = new_tensors.len();
        new_tensors.push(TensorContract::new(
            original.name.clone(),
            Expr::out(bank_name.clone()).slice(0, slot as i64 * local_rows, local_rows),
            vec![local_rows, cols],
            original.encoding.clone(),
        ));
    }
    Ok(())
}

fn direct_raw<'a>(
    metadata: &'a CheckpointMetadata,
    contract: &TensorContract,
) -> Result<&'a RawTensor, CompileError> {
    let Some(name) = direct_src(&contract.expr) else {
        return Err(CompileError::InvalidInput(format!(
            "runtime tensor '{}' is not a direct tensor",
            contract.name
        )));
    };
    metadata.tensor_by_name(name).ok_or_else(|| {
        CompileError::InvalidInput(format!(
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

struct ContractBuilder<'a> {
    metadata: &'a CheckpointMetadata,
    cfg: &'a ModelConfig,
    target: &'a StorageTarget,
    consumed: HashSet<TensorId>,
    tensors: Vec<TensorContract>,
}

impl ContractBuilder<'_> {
    fn profile(&self) -> ArchProfile {
        arch_profile(&self.cfg.model_type)
    }

    fn build(&mut self) -> Result<(), CompileError> {
        if self.target.backend == BackendKind::Metal {
            if self.profile().metal_qwen35 {
                return self.add_metal_qwen35_contracts();
            }
            return Err(CompileError::InvalidInput(format!(
                "Metal storage schema does not support model_type='{}'",
                self.cfg.model_type
            )));
        }
        let runtime_quant = self.runtime_quant_scheme()?;
        self.add_phi3_fused_splits()?;
        self.add_gpt_oss_mxfp4_groups()?;
        self.add_fused_moe_gate_up_tp_slices()?;
        self.add_qwen_moe_expert_stacks()?;
        self.add_nemotron_h_packed_expert_views()?;
        self.add_dense_fused_projection_joins(runtime_quant.is_some())?;
        self.add_mla_fused_projection_joins()?;
        for raw in &self.metadata.tensors {
            if self.consumed.contains(&raw.id) {
                continue;
            }
            if !self.source_name_allowed(&raw.name) {
                continue;
            }
            if let Some(scheme) = runtime_quant
                && runtime_quantizable_name(&raw.name, scheme)
            {
                self.push_runtime_quant(raw, raw.name.clone(), scheme)?;
            } else {
                self.push_direct(raw, self.output_name(&raw.name), self.shard_axis(&raw.name));
            }
        }
        Ok(())
    }

    fn source_name_allowed(&self, raw_name: &str) -> bool {
        if let Some(prefix) = self.primary_source_prefix() {
            return raw_name.starts_with(prefix);
        }
        true
    }

    fn primary_source_prefix(&self) -> Option<&'static str> {
        self.profile().source_prefix
    }

    fn output_name(&self, raw_name: &str) -> String {
        if let Some(prefix) = self.profile().source_prefix
            && let Some(stripped) = raw_name.strip_prefix(prefix)
        {
            return stripped.to_string();
        }
        raw_name.to_string()
    }

    fn dtype(&self, raw: &RawTensor) -> DType {
        dtype_for_encoding(&raw.encoding)
    }

    fn push_direct(&mut self, raw: &RawTensor, output_name: String, shard_axis: Option<Axis>) {
        let (expr, shape) = self.shard(Expr::src(raw.name.clone()), raw.shape.clone(), shard_axis);
        self.tensors.push(TensorContract::new(
            output_name,
            expr,
            shape,
            raw.encoding.clone(),
        ));
    }

    /// Partition an expression and its shape across ranks along `axis`.
    ///
    /// The expression records *that* the tensor is split; the shape records what
    /// this rank ends up holding. Both are left alone when there is nothing to
    /// split, so a single-GPU build is identical to one authored without
    /// sharding, and a non-divisible extent is left for the frontend to reject
    /// with a message that names the tensor.
    pub(super) fn shard(
        &self,
        expr: Expr,
        mut shape: Vec<i64>,
        axis: Option<Axis>,
    ) -> (Expr, Vec<i64>) {
        let world = i64::from(self.target.tp_size);
        let Some(axis) = axis.filter(|_| world > 1) else {
            return (expr, shape);
        };
        if let Some(dim) = shape.get_mut(usize::from(axis.0))
            && *dim % world == 0
        {
            *dim /= world;
        }
        (expr.shard(axis.0), shape)
    }

    fn push_runtime_quant(
        &mut self,
        raw: &RawTensor,
        output_name: String,
        scheme: QuantScheme,
    ) -> Result<(), CompileError> {
        if raw.shape.len() != 2 {
            return Err(CompileError::InvalidInput(format!(
                "runtime_quant source '{}' must be 2-D",
                raw.name
            )));
        }
        // Allowed sources:
        //   * BF16/FP16/FP32 raw  — handled by the executor's bf16 cast path.
        //   * FP8 (E4M3) raw     — used by GLM-5.1 routed experts: weights ship
        //                          quantized; the executor dequants them to bf16
        //                          using a sibling `_scale_inv` tensor at
        //                          materialize time, then re-encodes to the
        //                          target scheme. Only meaningful when the
        //                          target is a *smaller* scheme (e.g. MXFP4).
        let source_dtype_ok = matches!(
            raw.encoding,
            Encoding::Raw(DType::BF16 | DType::F16 | DType::F32 | DType::F8E4M3)
        );
        if !source_dtype_ok {
            return Err(CompileError::InvalidInput(format!(
                "runtime_quant source '{}' must be BF16/FP16/FP32/F8E4M3",
                raw.name
            )));
        }
        let spec = match scheme {
            QuantScheme::Fp8E4M3 => QuantSpec {
                scheme,
                logical_dtype: DType::F8E4M3,
                bits_per_element: 8,
                group_size: 1,
                channel_axis: Some(Axis(0)),
                scale_dtype: Some(DType::F32),
                zero_point_dtype: None,
                block_shape: Vec::new(),
            }
            .normalized(),
            QuantScheme::Int8Symmetric => QuantSpec {
                scheme,
                logical_dtype: DType::I8,
                bits_per_element: 8,
                group_size: 1,
                channel_axis: Some(Axis(0)),
                scale_dtype: Some(DType::F32),
                zero_point_dtype: None,
                block_shape: Vec::new(),
            }
            .normalized(),
            QuantScheme::Mxfp4E2M1E8M0 => {
                // K dimension (columns for 2-D weight) must be 32-multiple
                // because the MXFP4 block scale covers 32 contiguous elements.
                if raw.shape[1] % 32 != 0 {
                    return Err(CompileError::InvalidInput(format!(
                        "runtime_quant Mxfp4 source '{}' cols {} must be a multiple of 32",
                        raw.name, raw.shape[1]
                    )));
                }
                QuantSpec {
                    scheme,
                    logical_dtype: DType::BF16,
                    bits_per_element: 4,
                    group_size: 32,
                    channel_axis: Some(Axis(1)),
                    scale_dtype: Some(DType::U8),
                    zero_point_dtype: None,
                    block_shape: vec![32],
                }
                .normalized()
            }
            _ => {
                return Err(CompileError::InvalidInput(format!(
                    "unsupported runtime_quant scheme {:?}",
                    scheme
                )));
            }
        };
        let (expr, shape) = self.shard(
            Expr::src(raw.name.clone()),
            raw.shape.clone(),
            self.shard_axis(&raw.name),
        );
        self.tensors.push(TensorContract::new(
            output_name,
            expr,
            shape,
            Encoding::Quant(spec),
        ));
        Ok(())
    }

    fn push_expr(&mut self, output_name: String, raw: &RawTensor, shape: Vec<i64>, expr: Expr) {
        self.tensors.push(TensorContract::new(
            output_name,
            expr,
            shape,
            Encoding::Raw(self.dtype(raw)),
        ));
        self.consumed.insert(raw.id);
    }

    fn push_repack(
        &mut self,
        output_name: String,
        raw: &RawTensor,
        encoding: Encoding,
        shape: Vec<i64>,
        spec: RepackSpec,
    ) {
        let out = crate::contract::TensorType {
            shape: shape.clone(),
            encoding: encoding.clone(),
        };
        self.tensors.push(TensorContract::new(
            output_name,
            Expr::src(raw.name.clone()).repack(spec, out),
            shape,
            encoding,
        ));
    }

    fn shard_axis(&self, name: &str) -> Option<Axis> {
        if self.target.tp_size <= 1 {
            return None;
        }
        let profile = self.profile();
        // Shard embed_tokens on axis 0 to save per-rank memory (Kimi, GLM-5.1).
        if profile.shard_embed_tokens && name.ends_with(".embed_tokens.weight") {
            return Some(Axis(0));
        }
        // Replicate lm_head (Kimi K2.6): avoids requiring TP greedy argmax for
        // logits emission — ~1.7GB/rank extra but simplifies the logits path.
        if profile.replicate_lm_head && name.ends_with(".lm_head.weight") {
            return None;
        }
        (profile.shard_axis_fn)(name)
    }

    fn runtime_quant_scheme(&self) -> Result<Option<QuantScheme>, CompileError> {
        let mode = self.cfg.runtime_quant.as_str();
        if mode.is_empty() {
            return Ok(None);
        }
        let scheme = match mode {
            "fp8" => QuantScheme::Fp8E4M3,
            "int8" => QuantScheme::Int8Symmetric,
            "fp4" | "mxfp4" => QuantScheme::Mxfp4E2M1E8M0,
            other => {
                return Err(CompileError::InvalidInput(format!(
                    "unsupported runtime_quant '{other}'; expected 'fp8', 'int8', or 'fp4'"
                )));
            }
        };
        // For FP4 we accept a pre-quantized checkpoint (GLM-5.1 ships FP8
        // experts). For FP8/INT8 the legacy gate stays — we only re-quant
        // BF16 weights, never re-quant an already-quantized checkpoint.
        if !self.cfg.quant_method.is_empty() && scheme != QuantScheme::Mxfp4E2M1E8M0 {
            return Ok(None);
        }
        if !runtime_quant_model_supported(&self.cfg.model_type, scheme) {
            return Err(CompileError::InvalidInput(format!(
                "runtime_quant={} is not supported for model_type='{}'",
                mode, self.cfg.model_type
            )));
        }
        Ok(Some(scheme))
    }
}

fn dtype_for_encoding(encoding: &Encoding) -> DType {
    match encoding {
        Encoding::Raw(dtype) => *dtype,
        Encoding::Quant(spec) => spec.logical_dtype,
    }
}

fn mxfp4_encoding(channel_axis: Axis) -> Encoding {
    Encoding::Quant(
        QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(channel_axis),
            scale_dtype: Some(DType::U8),
            zero_point_dtype: None,
            block_shape: vec![32],
        }
        .normalized(),
    )
}

fn align_up_i64(value: i64, alignment: i64) -> Result<i64, CompileError> {
    if value < 0 || alignment <= 0 {
        return Err(CompileError::InvalidInput(
            "align_up_i64 requires non-negative value and positive alignment".to_string(),
        ));
    }
    value
        .checked_add(alignment - 1)
        .and_then(|v| v.checked_div(alignment))
        .and_then(|v| v.checked_mul(alignment))
        .ok_or_else(|| CompileError::InvalidInput("alignment overflow".to_string()))
}

fn u32_dim(value: i64, context: &str) -> Result<u32, CompileError> {
    u32::try_from(value).map_err(|_| {
        CompileError::InvalidInput(format!("{context}: dimension {value} does not fit u32"))
    })
}

fn dense_element_bytes(raw: &RawTensor, context: &str) -> Result<u64, CompileError> {
    match &raw.encoding {
        Encoding::Raw(dtype) => Ok(dtype.bytes()),
        Encoding::Quant(spec) => spec.dense_element_bytes().ok_or_else(|| {
            CompileError::InvalidInput(format!(
                "{context} '{}' has non-affine packed encoding",
                raw.name
            ))
        }),
    }
}

/// The `[start, len)` this rank owns of a `full`-long axis.
///
/// `what` names the thing being split, because this is the message a user gets
/// for "tp_size does not divide this model". The driver used to pre-empt it with
/// its own per-family table of divisibility rules read off `config.json` — the
/// same fact checked twice, and only for the families someone had listed.
fn local_range(full: i64, target: &StorageTarget, what: &str) -> Result<(i64, i64), CompileError> {
    let world = i64::from(target.tp_size.max(1));
    let rank = i64::from(target.tp_rank);
    if full % world != 0 {
        return Err(CompileError::InvalidInput(format!(
            "{what} is {full}, which tp_size {} does not divide; use a tp_size \
             that divides it or run single-GPU",
            target.tp_size
        )));
    }
    let local = full / world;
    Ok((rank * local, local))
}

fn checked_mul_i64(lhs: i64, rhs: u64, context: &str) -> Result<u64, CompileError> {
    let lhs = u64::try_from(lhs)
        .map_err(|_| CompileError::InvalidInput(format!("{context}: negative value")))?;
    checked_mul_u64(lhs, rhs, context)
}

fn checked_mul_u64(lhs: u64, rhs: u64, context: &str) -> Result<u64, CompileError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| CompileError::InvalidInput(format!("{context}: byte overflow")))
}
