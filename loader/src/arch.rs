use crate::checkpoint::{CheckpointMetadata, RawTensor};
use crate::config::ModelConfig;
use crate::contract::Expr;
use crate::error::CompileError;
use crate::load_plan::StorageTarget;
use crate::types::{
    Axis, BackendKind, DType, Encoding, Layout, Mxfp4MoePolicy, QuantScheme, QuantSpec,
    RepackLayout, RepackSpec, RowMap, TensorId,
};
use std::collections::{HashMap, HashSet};

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeAbi {
    pub name: String,
    pub version: u32,
    pub tensors: Vec<RuntimeTensorContract>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeTensorContract {
    pub output_name: String,
    /// What this tensor *is*, in the algebra of [`crate::contract`].
    pub expr: Expr,
    pub encoding: Encoding,
    pub shape: Vec<i64>,
    pub layout: Layout,
    pub alignment: u32,
    pub shard_axis: Option<Axis>,
}

impl RuntimeTensorContract {
    /// The shape a driver will actually find, which is not always `shape`.
    ///
    /// `shape` is the tensor as the *model* describes it. When `shard_axis` is
    /// set, each rank holds only its slice of that axis, so the declared shape
    /// and the delivered shape differ by a factor of the world size. Keeping
    /// both is what lets a pass describe a tensor once and have it sharded
    /// consistently; this is the function that says which is which.
    pub fn runtime_shape(&self, tp_size: u32) -> Vec<i64> {
        let mut shape = self.shape.clone();
        if tp_size > 1
            && let Some(axis) = self.shard_axis
            && let Some(dim) = shape.get_mut(usize::from(axis.0))
            && *dim % i64::from(tp_size) == 0
        {
            *dim /= i64::from(tp_size);
        }
        shape
    }
}

impl RuntimeAbi {
    pub fn default_for_target(
        metadata: &CheckpointMetadata,
        cfg: &ModelConfig,
        target: &StorageTarget,
    ) -> Result<Self, CompileError> {
        let mut builder = DefaultAbiBuilder {
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
            .filter(|contract| contract.shard_axis.is_some())
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
                        sharded.saturating_add(if contract.shard_axis.is_some() {
                            bytes
                        } else {
                            0
                        }),
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
        Ok(Self {
            name: match target.backend {
                crate::types::BackendKind::Cuda => "pie-cuda".to_string(),
                crate::types::BackendKind::Metal => "pie-metal".to_string(),
                crate::types::BackendKind::Unknown => "pie".to_string(),
            },
            version: 1,
            tensors: builder.tensors,
        })
    }

    pub fn retain_outputs(
        mut self,
        mut retain: impl FnMut(&str) -> bool,
    ) -> Result<Self, CompileError> {
        let mut selected = self
            .tensors
            .iter()
            .map(|contract| retain(&contract.output_name))
            .collect::<Vec<_>>();
        let by_name: HashMap<&str, usize> = self
            .tensors
            .iter()
            .enumerate()
            .map(|(index, contract)| (contract.output_name.as_str(), index))
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
                            "runtime ABI contract {index} reads missing contract '{name}'"
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
                "runtime ABI component filter selected no tensors".to_string(),
            ));
        }

        // References are by name, so dropping tensors needs no renumbering.
        let mut tensors = Vec::new();
        for (contract, selected) in self.tensors.into_iter().zip(selected) {
            if selected {
                tensors.push(contract);
            }
        }
        self.tensors = tensors;
        Ok(self)
    }

    pub fn coalesce_direct_row_shards(
        &self,
        metadata: &CheckpointMetadata,
        target: &StorageTarget,
    ) -> Result<Self, CompileError> {
        if target.tp_size <= 1 {
            return Ok(self.clone());
        }

        const MIN_GROUP_TENSORS: usize = 16;
        const DEFAULT_MAX_BANK_BYTES: u64 = 4 * 1024 * 1024 * 1024;
        let max_bank_bytes = DEFAULT_MAX_BANK_BYTES;

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct GroupKey {
            shape: Vec<i64>,
            encoding: Encoding,
            layout: Layout,
            alignment: u32,
        }

        let mut buckets: Vec<(GroupKey, Vec<usize>)> = Vec::new();
        let mut local_bytes_by_index = vec![0_u64; self.tensors.len()];
        for (index, contract) in self.tensors.iter().enumerate() {
            if contract.shard_axis != Some(Axis(0))
                || contract.shape.len() != 2
                || contract.shape[0] <= 0
                || contract.shape[1] <= 0
            {
                continue;
            }
            let Expr::Src(name) = &contract.expr else {
                continue;
            };
            let Some(raw) = metadata.tensor_by_name(name) else {
                continue;
            };
            if raw.shape != contract.shape || raw.encoding != contract.encoding {
                continue;
            }
            let elem = match dense_element_bytes(raw, "direct row shard coalescing") {
                Ok(elem) => elem,
                Err(_) => continue,
            };
            let (_, local_rows) = local_range(
                contract.shape[0],
                target,
                &format!("the row count of '{}'", contract.output_name),
            )?;
            let row_bytes = checked_mul_i64(
                contract.shape[1],
                elem,
                "direct row shard coalescing row bytes",
            )?;
            local_bytes_by_index[index] = checked_mul_i64(
                local_rows,
                row_bytes,
                "direct row shard coalescing local bytes",
            )?;
            let key = GroupKey {
                shape: contract.shape.clone(),
                encoding: contract.encoding.clone(),
                layout: contract.layout.clone(),
                alignment: contract.alignment,
            };
            if let Some((_, indices)) = buckets.iter_mut().find(|(candidate, _)| *candidate == key)
            {
                indices.push(index);
            } else {
                buckets.push((key, vec![index]));
            }
        }

        let mut group_for = vec![None; self.tensors.len()];
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
            return Ok(self.clone());
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
        let mut old_to_new = vec![usize::MAX; self.tensors.len()];
        let mut new_tensors = Vec::with_capacity(self.tensors.len() + groups.len());

        for old_index in 0..self.tensors.len() {
            if let Some(group_id) = group_for[old_index] {
                if emitted_groups[group_id] {
                    continue;
                }
                emitted_groups[group_id] = true;
                self.emit_row_shard_bank(
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
            new_tensors.push(self.tensors[old_index].clone());
        }

        Ok(Self {
            name: self.name.clone(),
            version: self.version,
            tensors: new_tensors,
        })
    }

    fn emit_row_shard_bank(
        &self,
        metadata: &CheckpointMetadata,
        target: &StorageTarget,
        group_id: usize,
        indices: &[usize],
        old_to_new: &mut [usize],
        new_tensors: &mut Vec<RuntimeTensorContract>,
    ) -> Result<(), CompileError> {
        let first = &self.tensors[indices[0]];
        let rows = first.shape[0];
        let cols = first.shape[1];
        let first_raw = direct_raw(metadata, first)?;
        dense_element_bytes(first_raw, "direct row shard coalescing")?;
        let (local_start, local_rows) = local_range(
            rows,
            target,
            &format!("the row count of '{}'", first.output_name),
        )?;

        // The bank is the local row band of every member, end to end. Stated as
        // an expression the offsets are the compiler's problem, not this pass's.
        let mut parts = Vec::with_capacity(indices.len());
        for &old_index in indices {
            let raw = direct_raw(metadata, &self.tensors[old_index])?;
            parts.push(Expr::src(raw.name.clone()).slice(0, local_start, local_rows));
        }

        let bank_name = format!("__pie.row_shard_bank.{group_id}");
        new_tensors.push(RuntimeTensorContract {
            output_name: bank_name.clone(),
            expr: Expr::cat(0, parts),
            encoding: first.encoding.clone(),
            shape: vec![local_rows * indices.len() as i64, cols],
            layout: first.layout.clone(),
            alignment: first.alignment,
            shard_axis: None,
        });

        for (slot, &old_index) in indices.iter().enumerate() {
            let original = &self.tensors[old_index];
            old_to_new[old_index] = new_tensors.len();
            new_tensors.push(RuntimeTensorContract {
                output_name: original.output_name.clone(),
                expr: Expr::out(bank_name.clone()).slice(0, slot as i64 * local_rows, local_rows),
                encoding: original.encoding.clone(),
                shape: vec![local_rows, cols],
                layout: original.layout.clone(),
                alignment: original.alignment,
                shard_axis: None,
            });
        }
        Ok(())
    }
}

fn direct_raw<'a>(
    metadata: &'a CheckpointMetadata,
    contract: &RuntimeTensorContract,
) -> Result<&'a RawTensor, CompileError> {
    let Expr::Src(name) = &contract.expr else {
        return Err(CompileError::InvalidInput(format!(
            "runtime tensor '{}' is not a direct tensor",
            contract.output_name
        )));
    };
    metadata.tensor_by_name(name).ok_or_else(|| {
        CompileError::InvalidInput(format!(
            "runtime tensor '{}' references missing source tensor '{name}'",
            contract.output_name
        ))
    })
}

struct DefaultAbiBuilder<'a> {
    metadata: &'a CheckpointMetadata,
    cfg: &'a ModelConfig,
    target: &'a StorageTarget,
    consumed: HashSet<TensorId>,
    tensors: Vec<RuntimeTensorContract>,
}

impl DefaultAbiBuilder<'_> {
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

    fn alignment(&self) -> u32 {
        self.target.preferred_alignment.max(1)
    }

    fn dtype(&self, raw: &RawTensor) -> DType {
        dtype_for_encoding(&raw.encoding)
    }

    fn push_direct(&mut self, raw: &RawTensor, output_name: String, shard_axis: Option<Axis>) {
        self.tensors.push(RuntimeTensorContract {
            output_name,
            expr: Expr::src(raw.name.clone()),
            encoding: raw.encoding.clone(),
            shape: raw.shape.clone(),
            layout: Layout::dense(self.alignment()),
            alignment: self.alignment(),
            shard_axis,
        });
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
        self.tensors.push(RuntimeTensorContract {
            output_name,
            expr: Expr::src(raw.name.clone()),
            encoding: Encoding::Quant(spec),
            shape: raw.shape.clone(),
            layout: Layout::dense(self.alignment()),
            alignment: self.alignment(),
            shard_axis: self.shard_axis(&raw.name),
        });
        Ok(())
    }

    fn push_expr(&mut self, output_name: String, raw: &RawTensor, shape: Vec<i64>, expr: Expr) {
        self.tensors.push(RuntimeTensorContract {
            output_name,
            expr,
            encoding: Encoding::Raw(self.dtype(raw)),
            shape,
            layout: Layout::dense(self.alignment()),
            alignment: self.alignment(),
            shard_axis: None,
        });
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
        self.tensors.push(RuntimeTensorContract {
            output_name,
            expr: Expr::src(raw.name.clone()).repack(spec, out),
            encoding,
            shape,
            layout: Layout::dense(self.alignment()),
            alignment: self.alignment(),
            shard_axis: None,
        });
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

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(name: &str, expr: Expr) -> RuntimeTensorContract {
        RuntimeTensorContract {
            output_name: name.to_string(),
            expr,
            encoding: Encoding::Raw(DType::U8),
            shape: vec![1],
            layout: Layout::dense(1),
            alignment: 1,
            shard_axis: None,
        }
    }

    #[test]
    fn component_filter_retains_selected_dependencies() {
        let abi = RuntimeAbi {
            name: "test".to_string(),
            version: 1,
            tensors: vec![
                contract("text.weight", Expr::src("text")),
                contract("vision.base", Expr::src("vision")),
                contract("vision.view", Expr::out("vision.base").slice(0, 0, 1)),
            ],
        };

        let filtered = abi.retain_outputs(|name| name == "vision.view").unwrap();
        assert_eq!(filtered.tensors.len(), 2);
        assert_eq!(filtered.tensors[0].output_name, "vision.base");
        assert_eq!(filtered.tensors[1].output_name, "vision.view");
        assert_eq!(filtered.tensors[1].expr.outputs(), ["vision.base"]);
    }
}
