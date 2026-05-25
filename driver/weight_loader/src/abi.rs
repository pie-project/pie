use crate::config::ModelConfig;
use crate::error::CompileError;
use crate::ffi_types::{
    PieLoaderRuntimeAbiView, PieLoaderRuntimeByteSpanSlice, PieLoaderRuntimeSourceKind,
    PieLoaderRuntimeTensorContractView, PieLoaderSemanticRole,
};
use crate::semantic::SemanticRole;
use crate::source::{
    CheckpointMetadata, RawTensor, ffi_dtype, ffi_i64_slice, ffi_optional_axis, ffi_quant_scheme,
    ffi_string,
};
use crate::storage::StorageTarget;
use crate::types::{
    Axis, BackendKind, DType, Encoding, Layout, Mxfp4MoePolicy, QuantScheme, QuantSpec,
    RepackLayout, RepackSpec, RowMap, Sharding, TensorId,
};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeAbi {
    pub name: String,
    pub version: u32,
    pub tensors: Vec<RuntimeTensorContract>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeTensorContract {
    pub output_name: String,
    pub source: RuntimeTensorSource,
    pub metadata: Vec<TensorId>,
    pub dtype: DType,
    pub encoding: Encoding,
    pub shape: Vec<i64>,
    pub layout: Layout,
    pub sharding: Sharding,
    pub alignment: u32,
    pub shard_axis: Option<Axis>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeTensorSource {
    DirectTensor(TensorId),
    Semantic {
        role: SemanticRole,
        layer: Option<u32>,
        expert: Option<u32>,
    },
    ByteSpans(Vec<RuntimeByteSpan>),
    Join {
        tensors: Vec<TensorId>,
        axis: Axis,
    },
    SelectContract {
        contract: usize,
        axis: Axis,
        start: i64,
        length: i64,
    },
    Repack {
        tensor: TensorId,
        spec: RepackSpec,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeByteSpan {
    pub tensor: TensorId,
    pub source_offset_bytes: u64,
    pub dest_offset_bytes: u64,
    pub span_bytes: u64,
}

impl RuntimeAbi {
    pub fn from_ffi(view: &PieLoaderRuntimeAbiView) -> Result<Self, CompileError> {
        let contracts = if view.tensors.len == 0 {
            &[][..]
        } else {
            if view.tensors.ptr.is_null() {
                return Err(CompileError::NullArgument("runtime_abi.tensors"));
            }
            unsafe { std::slice::from_raw_parts(view.tensors.ptr, view.tensors.len) }
        };
        Ok(Self {
            name: ffi_string(view.name, "runtime_abi.name")?,
            version: view.version,
            tensors: contracts
                .iter()
                .map(RuntimeTensorContract::from_ffi)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

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
        Ok(Self {
            name: match target.backend {
                crate::types::BackendKind::Cuda => "pie-cuda".to_string(),
                crate::types::BackendKind::Portable => "pie-portable".to_string(),
                crate::types::BackendKind::Unknown => "pie".to_string(),
            },
            version: 1,
            tensors: builder.tensors,
        })
    }
}

struct DefaultAbiBuilder<'a> {
    metadata: &'a CheckpointMetadata,
    cfg: &'a ModelConfig,
    target: &'a StorageTarget,
    consumed: HashSet<TensorId>,
    tensors: Vec<RuntimeTensorContract>,
}

struct FusedProjectionCandidate {
    output_name: String,
    tensors: Vec<TensorId>,
    rows: i64,
    cols: i64,
    bytes: u64,
}

impl DefaultAbiBuilder<'_> {
    fn build(&mut self) -> Result<(), CompileError> {
        let runtime_quant = self.runtime_quant_scheme()?;
        self.add_phi3_fused_splits()?;
        self.add_gpt_oss_mxfp4_groups()?;
        self.add_fused_moe_gate_up_tp_slices()?;
        self.add_nemotron_h_packed_expert_views()?;
        self.add_dense_fused_projection_joins(runtime_quant.is_some())?;
        for raw in &self.metadata.tensors {
            if self.consumed.contains(&raw.id) {
                continue;
            }
            if let Some(scheme) = runtime_quant
                && runtime_quantizable_name(&raw.name)
            {
                self.push_runtime_quant(raw, raw.name.clone(), scheme)?;
            } else {
                self.push_direct(raw, raw.name.clone(), self.shard_axis(&raw.name));
            }
        }
        Ok(())
    }

    fn add_dense_fused_projection_joins(
        &mut self,
        runtime_quant_enabled: bool,
    ) -> Result<(), CompileError> {
        if self.target.backend != BackendKind::Cuda
            || self.target.tp_size != 1
            || runtime_quant_enabled
        {
            return Ok(());
        }

        let mut qkv_candidates = Vec::new();
        let mut gate_up_candidates = Vec::new();
        let mut qkv_bytes = 0u64;
        let mut gate_up_bytes = 0u64;

        for layer in 0..self.cfg.num_hidden_layers {
            let p = format!("model.layers.{layer}.");
            if let Some(candidate) = self.fused_join_candidate(
                &(p.clone() + "self_attn.qkv_proj.fused.weight"),
                &[
                    p.clone() + "self_attn.q_proj.weight",
                    p.clone() + "self_attn.k_proj.weight",
                    p.clone() + "self_attn.v_proj.weight",
                ],
            )? {
                qkv_bytes = qkv_bytes.checked_add(candidate.bytes).ok_or_else(|| {
                    CompileError::InvalidInput("fused projection byte budget overflow".to_string())
                })?;
                qkv_candidates.push(candidate);
            }
            if let Some(candidate) = self.fused_join_candidate(
                &(p.clone() + "mlp.gate_up_proj.fused.weight"),
                &[
                    p.clone() + "mlp.gate_proj.weight",
                    p.clone() + "mlp.up_proj.weight",
                ],
            )? {
                gate_up_bytes = gate_up_bytes.checked_add(candidate.bytes).ok_or_else(|| {
                    CompileError::InvalidInput("fused projection byte budget overflow".to_string())
                })?;
                gate_up_candidates.push(candidate);
            }
        }

        if qkv_candidates.is_empty() && gate_up_candidates.is_empty() {
            return Ok(());
        }

        let budget = dense_fused_projection_budget_bytes();
        let total_bytes = qkv_bytes.checked_add(gate_up_bytes).ok_or_else(|| {
            CompileError::InvalidInput("fused projection byte budget overflow".to_string())
        })?;
        let mut candidates = Vec::new();
        if total_bytes <= budget {
            candidates.extend(qkv_candidates);
            candidates.extend(gate_up_candidates);
        } else {
            // Prefer QKV fusion when the full duplicate set is too large. It
            // is much smaller than gate/up on Qwen-style models and enables
            // the fused decode postprocess without giving up large-model KV
            // capacity. Gate/up is only enabled as a complete model-wide set
            // when it also fits the remaining budget.
            let mut used = 0u64;
            if qkv_bytes <= budget {
                used = qkv_bytes;
                candidates.extend(qkv_candidates);
            }
            if gate_up_bytes <= budget.saturating_sub(used) {
                candidates.extend(gate_up_candidates);
            }
        }
        if candidates.is_empty() {
            return Ok(());
        }

        for candidate in candidates {
            for tensor in &candidate.tensors {
                self.consumed.insert(*tensor);
            }
            self.tensors.push(RuntimeTensorContract {
                output_name: candidate.output_name,
                source: RuntimeTensorSource::Join {
                    tensors: candidate.tensors,
                    axis: Axis(0),
                },
                metadata: Vec::new(),
                dtype: DType::BF16,
                encoding: Encoding::Raw(DType::BF16),
                shape: vec![candidate.rows, candidate.cols],
                layout: Layout::dense(self.alignment()),
                sharding: Sharding::replicated(),
                alignment: self.alignment(),
                shard_axis: None,
            });
        }

        Ok(())
    }

    fn fused_join_candidate(
        &self,
        output_name: &str,
        input_names: &[String],
    ) -> Result<Option<FusedProjectionCandidate>, CompileError> {
        if self
            .metadata
            .tensors
            .iter()
            .any(|raw| raw.name == output_name)
        {
            return Ok(None);
        }

        let mut tensors = Vec::with_capacity(input_names.len());
        let mut rows = 0i64;
        let mut cols: Option<i64> = None;
        let mut bytes = 0u64;

        for name in input_names {
            let Some(raw) = self.metadata.tensors.iter().find(|raw| raw.name == *name) else {
                return Ok(None);
            };
            if raw.shape.len() != 2 || raw.encoding != Encoding::Raw(DType::BF16) {
                return Ok(None);
            }
            let current_cols = raw.shape[1];
            if let Some(expected) = cols {
                if current_cols != expected {
                    return Ok(None);
                }
            } else {
                cols = Some(current_cols);
            }
            tensors.push(raw.id);
            rows = rows.checked_add(raw.shape[0]).ok_or_else(|| {
                CompileError::InvalidInput(format!(
                    "fused projection '{output_name}' row count overflow"
                ))
            })?;
            bytes = bytes.checked_add(raw.span_bytes).ok_or_else(|| {
                CompileError::InvalidInput(format!(
                    "fused projection '{output_name}' byte count overflow"
                ))
            })?;
        }

        Ok(Some(FusedProjectionCandidate {
            output_name: output_name.to_string(),
            tensors,
            rows,
            cols: cols.unwrap_or(0),
            bytes,
        }))
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
            source: RuntimeTensorSource::DirectTensor(raw.id),
            metadata: Vec::new(),
            dtype: self.dtype(raw),
            encoding: raw.encoding.clone(),
            shape: raw.shape.clone(),
            layout: Layout::dense(self.alignment()),
            sharding: Sharding::replicated(),
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
        if !matches!(
            raw.encoding,
            Encoding::Raw(DType::BF16 | DType::F16 | DType::F32)
        ) {
            return Err(CompileError::InvalidInput(format!(
                "runtime_quant source '{}' must be BF16/FP16/FP32",
                raw.name
            )));
        }
        let dtype = match scheme {
            QuantScheme::Fp8E4M3 => DType::F8E4M3,
            QuantScheme::Int8Symmetric => DType::I8,
            _ => {
                return Err(CompileError::InvalidInput(format!(
                    "unsupported runtime_quant scheme {:?}",
                    scheme
                )));
            }
        };
        self.tensors.push(RuntimeTensorContract {
            output_name,
            source: RuntimeTensorSource::DirectTensor(raw.id),
            metadata: Vec::new(),
            dtype,
            encoding: Encoding::Quant(
                QuantSpec {
                    scheme,
                    logical_dtype: dtype,
                    bits_per_element: 8,
                    group_size: 1,
                    channel_axis: Some(Axis(0)),
                    scale_dtype: Some(DType::F32),
                    zero_point_dtype: None,
                    block_shape: Vec::new(),
                }
                .normalized(),
            ),
            shape: raw.shape.clone(),
            layout: Layout::dense(self.alignment()),
            sharding: Sharding::replicated(),
            alignment: self.alignment(),
            shard_axis: self.shard_axis(&raw.name),
        });
        Ok(())
    }

    fn push_byte_spans(
        &mut self,
        output_name: String,
        raw: &RawTensor,
        dtype: DType,
        shape: Vec<i64>,
        spans: Vec<RuntimeByteSpan>,
    ) {
        self.tensors.push(RuntimeTensorContract {
            output_name,
            source: RuntimeTensorSource::ByteSpans(spans),
            metadata: Vec::new(),
            dtype,
            encoding: Encoding::Raw(dtype),
            shape,
            layout: Layout::dense(self.alignment()),
            sharding: Sharding::replicated(),
            alignment: self.alignment(),
            shard_axis: None,
        });
        self.consumed.insert(raw.id);
    }

    fn push_repack(
        &mut self,
        output_name: String,
        raw: &RawTensor,
        dtype: DType,
        encoding: Encoding,
        shape: Vec<i64>,
        spec: RepackSpec,
    ) {
        self.tensors.push(RuntimeTensorContract {
            output_name,
            source: RuntimeTensorSource::Repack {
                tensor: raw.id,
                spec,
            },
            metadata: Vec::new(),
            dtype,
            encoding,
            shape,
            layout: Layout::dense(self.alignment()),
            sharding: Sharding::replicated(),
            alignment: self.alignment(),
            shard_axis: None,
        });
    }

    fn add_phi3_fused_splits(&mut self) -> Result<(), CompileError> {
        if !self.cfg.model_type.eq_ignore_ascii_case("phi3") {
            return Ok(());
        }
        for raw in &self.metadata.tensors {
            if raw.name.ends_with(".self_attn.qkv_proj.weight") {
                self.add_phi3_qkv_split(raw)?;
            } else if raw.name.ends_with(".mlp.gate_up_proj.weight") {
                self.add_phi3_gate_up_split(raw)?;
            }
        }
        Ok(())
    }

    fn add_phi3_qkv_split(&mut self, raw: &RawTensor) -> Result<(), CompileError> {
        if raw.shape.len() != 2 {
            return Err(CompileError::InvalidInput(format!(
                "Phi-3 fused QKV '{}' must be 2-D",
                raw.name
            )));
        }
        let dtype = self.dtype(raw);
        let elem = dense_element_bytes(raw, "Phi-3 fused QKV")?;
        let q_rows = raw.shape[1];
        let kv_rows = (raw.shape[0] - q_rows) / 2;
        if q_rows <= 0 || kv_rows <= 0 || q_rows + 2 * kv_rows != raw.shape[0] {
            return Err(CompileError::InvalidInput(format!(
                "Phi-3 fused QKV '{}' has unsupported shape {:?}",
                raw.name, raw.shape
            )));
        }
        let cols = raw.shape[1];
        let base = raw.name.trim_end_matches(".self_attn.qkv_proj.weight");
        let specs = [
            ("q_proj", 0_i64, q_rows),
            ("k_proj", q_rows, kv_rows),
            ("v_proj", q_rows + kv_rows, kv_rows),
        ];
        for (proj, start, rows) in specs {
            let (local_start, local_rows) = local_range(rows, self.target)?;
            let source_rows = start + local_start;
            let span = span_for_rows(local_rows, cols, elem)?;
            self.push_byte_spans(
                format!("{base}.self_attn.{proj}.weight"),
                raw,
                dtype,
                vec![local_rows, cols],
                vec![RuntimeByteSpan {
                    tensor: raw.id,
                    source_offset_bytes: span_for_rows(source_rows, cols, elem)?,
                    dest_offset_bytes: 0,
                    span_bytes: span,
                }],
            );
        }
        Ok(())
    }

    fn add_phi3_gate_up_split(&mut self, raw: &RawTensor) -> Result<(), CompileError> {
        if raw.shape.len() != 2 || raw.shape[0] % 2 != 0 {
            return Err(CompileError::InvalidInput(format!(
                "Phi-3 fused gate/up '{}' has unsupported shape {:?}",
                raw.name, raw.shape
            )));
        }
        let dtype = self.dtype(raw);
        let elem = dense_element_bytes(raw, "Phi-3 fused gate/up")?;
        let half_rows = raw.shape[0] / 2;
        let cols = raw.shape[1];
        let base = raw.name.trim_end_matches(".mlp.gate_up_proj.weight");
        let specs = [("gate_proj", 0_i64), ("up_proj", half_rows)];
        for (proj, start) in specs {
            let (local_start, local_rows) = local_range(half_rows, self.target)?;
            self.push_byte_spans(
                format!("{base}.mlp.{proj}.weight"),
                raw,
                dtype,
                vec![local_rows, cols],
                vec![RuntimeByteSpan {
                    tensor: raw.id,
                    source_offset_bytes: span_for_rows(start + local_start, cols, elem)?,
                    dest_offset_bytes: 0,
                    span_bytes: span_for_rows(local_rows, cols, elem)?,
                }],
            );
        }
        Ok(())
    }

    fn add_fused_moe_gate_up_tp_slices(&mut self) -> Result<(), CompileError> {
        if self.target.tp_size <= 1 {
            return Ok(());
        }
        let candidates: Vec<RawTensor> = self
            .metadata
            .tensors
            .iter()
            .filter(|raw| {
                raw.name.ends_with(".experts.gate_up_proj")
                    || raw.name.ends_with(".mlp.experts.gate_up_proj")
            })
            .cloned()
            .collect();
        for raw in &candidates {
            if raw.shape.len() != 3 || raw.shape[1] % 2 != 0 {
                continue;
            }
            let dtype = self.dtype(raw);
            let elem = dense_element_bytes(raw, "fused MoE gate/up")?;
            let experts = raw.shape[0];
            let full_i = raw.shape[1] / 2;
            let hidden = raw.shape[2];
            let (local_start, local_i) = local_range(full_i, self.target)?;
            let row_bytes = checked_mul_i64(hidden, elem, "MoE gate/up row bytes")?;
            let full_expert_bytes = checked_mul_i64(raw.shape[1], row_bytes, "MoE expert bytes")?;
            let local_half_bytes = checked_mul_i64(local_i, row_bytes, "MoE local half bytes")?;
            let local_expert_bytes =
                checked_mul_u64(2, local_half_bytes, "MoE local expert bytes")?;
            let mut spans = Vec::with_capacity(experts as usize * 2);
            for expert in 0..experts {
                let expert_base =
                    checked_mul_i64(expert, full_expert_bytes, "MoE source expert offset")?;
                let dest_base =
                    checked_mul_i64(expert, local_expert_bytes, "MoE dest expert offset")?;
                spans.push(RuntimeByteSpan {
                    tensor: raw.id,
                    source_offset_bytes: expert_base
                        + checked_mul_i64(local_start, row_bytes, "MoE gate source offset")?,
                    dest_offset_bytes: dest_base,
                    span_bytes: local_half_bytes,
                });
                spans.push(RuntimeByteSpan {
                    tensor: raw.id,
                    source_offset_bytes: expert_base
                        + checked_mul_i64(full_i + local_start, row_bytes, "MoE up source offset")?,
                    dest_offset_bytes: dest_base + local_half_bytes,
                    span_bytes: local_half_bytes,
                });
            }
            self.push_byte_spans(
                raw.name.clone(),
                raw,
                dtype,
                vec![experts, 2 * local_i, hidden],
                spans,
            );
        }
        Ok(())
    }

    fn add_nemotron_h_packed_expert_views(&mut self) -> Result<(), CompileError> {
        if self.target.backend != BackendKind::Cuda
            || !self.cfg.model_type.eq_ignore_ascii_case("nemotron_h")
            || self.cfg.num_experts == 0
        {
            return Ok(());
        }

        for layer in 0..self.cfg.num_hidden_layers {
            let base = format!("language_model.backbone.layers.{layer}.mixer.experts");
            let up_name = format!("{base}.up_proj.packed.weight");
            let down_name = format!("{base}.down_proj.packed.weight");
            if self.metadata.tensors.iter().any(|raw| raw.name == up_name)
                || self
                    .metadata
                    .tensors
                    .iter()
                    .any(|raw| raw.name == down_name)
            {
                continue;
            }

            let mut up = Vec::with_capacity(self.cfg.num_experts as usize);
            let mut down = Vec::with_capacity(self.cfg.num_experts as usize);
            let mut complete = true;
            for expert in 0..self.cfg.num_experts {
                let prefix = format!("{base}.{expert}.");
                let Some(up_raw) = self
                    .metadata
                    .tensors
                    .iter()
                    .find(|raw| raw.name == prefix.clone() + "up_proj.weight")
                else {
                    complete = false;
                    break;
                };
                let Some(down_raw) = self
                    .metadata
                    .tensors
                    .iter()
                    .find(|raw| raw.name == prefix.clone() + "down_proj.weight")
                else {
                    complete = false;
                    break;
                };
                up.push(up_raw);
                down.push(down_raw);
            }
            if !complete {
                continue;
            }

            self.add_nemotron_h_layer_packed_experts(&base, &up, &down)?;
        }
        Ok(())
    }

    fn add_nemotron_h_layer_packed_experts(
        &mut self,
        base: &str,
        up: &[&RawTensor],
        down: &[&RawTensor],
    ) -> Result<(), CompileError> {
        let Some(first_up) = up.first().copied() else {
            return Ok(());
        };
        let Some(first_down) = down.first().copied() else {
            return Ok(());
        };
        if first_up.shape.len() != 2
            || first_down.shape.len() != 2
            || first_up.encoding != Encoding::Raw(DType::BF16)
            || first_down.encoding != Encoding::Raw(DType::BF16)
        {
            return Ok(());
        }

        let full_intermediate = first_up.shape[0];
        let hidden = first_up.shape[1];
        if first_down.shape[0] != hidden || first_down.shape[1] != full_intermediate {
            return Ok(());
        }
        for raw in up {
            if raw.shape != first_up.shape || raw.encoding != first_up.encoding {
                return Ok(());
            }
        }
        for raw in down {
            if raw.shape != first_down.shape || raw.encoding != first_down.encoding {
                return Ok(());
            }
        }

        let elem = dense_element_bytes(first_up, "Nemotron-H expert")?;
        let (local_start, local_intermediate) = local_range(full_intermediate, self.target)?;
        let local_up_bytes = span_for_rows(local_intermediate, hidden, elem)?;
        let expert_count = i64::try_from(up.len()).map_err(|_| {
            CompileError::InvalidInput("Nemotron-H expert count does not fit i64".to_string())
        })?;

        let up_contract = self.tensors.len();
        let mut up_spans = Vec::with_capacity(up.len());
        for (expert, raw) in up.iter().enumerate() {
            let expert = i64::try_from(expert).map_err(|_| {
                CompileError::InvalidInput("Nemotron-H expert index does not fit i64".to_string())
            })?;
            up_spans.push(RuntimeByteSpan {
                tensor: raw.id,
                source_offset_bytes: span_for_rows(local_start, hidden, elem)?,
                dest_offset_bytes: checked_mul_i64(
                    expert,
                    local_up_bytes,
                    "Nemotron-H packed up destination offset",
                )?,
                span_bytes: local_up_bytes,
            });
        }
        self.tensors.push(RuntimeTensorContract {
            output_name: format!("{base}.up_proj.packed.weight"),
            source: RuntimeTensorSource::ByteSpans(up_spans),
            metadata: Vec::new(),
            dtype: DType::BF16,
            encoding: Encoding::Raw(DType::BF16),
            shape: vec![expert_count * local_intermediate, hidden],
            layout: Layout::dense(self.alignment()),
            sharding: Sharding::replicated(),
            alignment: self.alignment(),
            shard_axis: None,
        });

        let down_contract = self.tensors.len();
        self.tensors.push(RuntimeTensorContract {
            output_name: format!("{base}.down_proj.packed.weight"),
            source: RuntimeTensorSource::Join {
                tensors: down.iter().map(|raw| raw.id).collect(),
                axis: Axis(0),
            },
            metadata: Vec::new(),
            dtype: DType::BF16,
            encoding: Encoding::Raw(DType::BF16),
            shape: vec![expert_count * hidden, full_intermediate],
            layout: Layout::dense(self.alignment()),
            sharding: Sharding::replicated(),
            alignment: self.alignment(),
            shard_axis: (self.target.tp_size > 1).then_some(Axis(1)),
        });

        for (expert, raw) in up.iter().enumerate() {
            let expert = i64::try_from(expert).map_err(|_| {
                CompileError::InvalidInput("Nemotron-H expert index does not fit i64".to_string())
            })?;
            self.tensors.push(RuntimeTensorContract {
                output_name: raw.name.clone(),
                source: RuntimeTensorSource::SelectContract {
                    contract: up_contract,
                    axis: Axis(0),
                    start: expert * local_intermediate,
                    length: local_intermediate,
                },
                metadata: Vec::new(),
                dtype: DType::BF16,
                encoding: Encoding::Raw(DType::BF16),
                shape: vec![local_intermediate, hidden],
                layout: Layout::dense(self.alignment()),
                sharding: Sharding::replicated(),
                alignment: self.alignment(),
                shard_axis: None,
            });
            self.consumed.insert(raw.id);
        }

        for (expert, raw) in down.iter().enumerate() {
            let expert = i64::try_from(expert).map_err(|_| {
                CompileError::InvalidInput("Nemotron-H expert index does not fit i64".to_string())
            })?;
            let sharding = if self.target.tp_size > 1 {
                Sharding {
                    axis: Some(Axis(1)),
                    world: self.target.tp_size,
                    rank: self.target.tp_rank,
                }
            } else {
                Sharding::replicated()
            };
            self.tensors.push(RuntimeTensorContract {
                output_name: raw.name.clone(),
                source: RuntimeTensorSource::SelectContract {
                    contract: down_contract,
                    axis: Axis(0),
                    start: expert * hidden,
                    length: hidden,
                },
                metadata: Vec::new(),
                dtype: DType::BF16,
                encoding: Encoding::Raw(DType::BF16),
                shape: vec![hidden, local_intermediate],
                layout: Layout::dense(self.alignment()),
                sharding,
                alignment: self.alignment(),
                shard_axis: None,
            });
            self.consumed.insert(raw.id);
        }

        Ok(())
    }

    fn add_gpt_oss_mxfp4_groups(&mut self) -> Result<(), CompileError> {
        if !self.cfg.model_type.eq_ignore_ascii_case("gpt_oss")
            && !self.cfg.model_type.eq_ignore_ascii_case("gpt-oss")
            && !self.cfg.model_type.eq_ignore_ascii_case("gptoss")
        {
            return Ok(());
        }
        let native = self.target.mxfp4_moe == Mxfp4MoePolicy::NativeGemm;
        if native && !self.target.native_mxfp4_moe {
            return Err(CompileError::InvalidInput(
                "GPT-OSS native MXFP4 requested, but target does not support native MXFP4 MoE"
                    .to_string(),
            ));
        }
        let blocks: Vec<RawTensor> = self
            .metadata
            .tensors
            .iter()
            .filter(|raw| raw.name.ends_with("_blocks"))
            .cloned()
            .collect();
        for block in &blocks {
            let base = block.name.trim_end_matches("_blocks");
            let scale_name = format!("{base}_scales");
            let bias_name = format!("{base}_bias");
            let Some(scale) = self
                .metadata
                .tensors
                .iter()
                .find(|raw| raw.name == scale_name)
            else {
                continue;
            };
            let Some(bias) = self
                .metadata
                .tensors
                .iter()
                .find(|raw| raw.name == bias_name)
            else {
                continue;
            };
            if native {
                self.add_gpt_oss_native_mxfp4_group(block, scale, bias, base)?;
            } else {
                self.push_direct(block, format!("{base}.weight"), None);
                self.push_direct(scale, format!("{base}.weight_scale"), None);
                self.push_direct(bias, format!("{base}.bias"), None);
            }
            self.consumed.insert(block.id);
            self.consumed.insert(scale.id);
            self.consumed.insert(bias.id);
        }
        Ok(())
    }

    fn add_gpt_oss_native_mxfp4_group(
        &mut self,
        block: &RawTensor,
        scale: &RawTensor,
        bias: &RawTensor,
        base: &str,
    ) -> Result<(), CompileError> {
        if base.ends_with("gate_up_proj") {
            self.add_gpt_oss_native_gate_up(block, scale, bias, base)
        } else if base.ends_with("down_proj") {
            self.add_gpt_oss_native_down(block, scale, bias, base)
        } else {
            Err(CompileError::InvalidInput(format!(
                "GPT-OSS MXFP4 tensor '{}' is not gate_up_proj or down_proj",
                block.name
            )))
        }
    }

    fn add_gpt_oss_native_gate_up(
        &mut self,
        block: &RawTensor,
        scale: &RawTensor,
        bias: &RawTensor,
        base: &str,
    ) -> Result<(), CompileError> {
        if block.shape.len() != 4 || scale.shape.len() != 3 || bias.shape.len() != 2 {
            return Err(CompileError::InvalidInput(format!(
                "GPT-OSS native gate/up '{}' has unsupported block/scale/bias rank",
                base
            )));
        }
        let experts = block.shape[0];
        let fused_rows = block.shape[1];
        let groups = block.shape[2];
        let lanes = block.shape[3];
        if fused_rows % 2 != 0 || lanes != 16 {
            return Err(CompileError::InvalidInput(format!(
                "GPT-OSS native gate/up '{}' expected [E, 2I, H/32, 16], got {:?}",
                base, block.shape
            )));
        }
        if scale.shape != vec![experts, fused_rows, groups]
            || bias.shape != vec![experts, fused_rows]
        {
            return Err(CompileError::InvalidInput(format!(
                "GPT-OSS native gate/up '{}' scale/bias shape mismatch",
                base
            )));
        }
        let full_intermediate = fused_rows / 2;
        let hidden = checked_mul_i64(groups, 32, "GPT-OSS hidden size")? as i64;
        let (local_start, local_intermediate) = local_range(full_intermediate, self.target)?;
        let intermediate_native = align_up_i64(local_intermediate, 128)?;
        let prefix = base.trim_end_matches("gate_up_proj");
        for (name, row_map) in [("gate_proj", RowMap::Even), ("up_proj", RowMap::Odd)] {
            let out_base = format!("{prefix}{name}");
            self.push_repack(
                format!("{out_base}.weight"),
                block,
                DType::BF16,
                mxfp4_encoding(Axis(1)),
                vec![experts, intermediate_native, hidden],
                RepackSpec {
                    layout: RepackLayout::MarlinMxfp4Weight,
                    row_map,
                    batch: u32_dim(experts, "GPT-OSS experts")?,
                    source_rows: u32_dim(fused_rows, "GPT-OSS gate/up source rows")?,
                    source_row_offset: u32_dim(local_start, "GPT-OSS gate/up source row offset")?,
                    target_rows: u32_dim(intermediate_native, "GPT-OSS gate/up target rows")?,
                    valid_rows: u32_dim(local_intermediate, "GPT-OSS gate/up valid rows")?,
                    source_stride_cols: u32_dim(hidden, "GPT-OSS hidden stride")?,
                    source_col_offset: 0,
                    source_cols: u32_dim(hidden, "GPT-OSS hidden size")?,
                    target_cols: u32_dim(hidden, "GPT-OSS hidden size")?,
                },
            );
            self.push_repack(
                format!("{out_base}.weight_scale"),
                scale,
                DType::U8,
                Encoding::Raw(DType::U8),
                vec![experts, intermediate_native, groups],
                RepackSpec {
                    layout: RepackLayout::MarlinMxfp4Scale,
                    row_map,
                    batch: u32_dim(experts, "GPT-OSS experts")?,
                    source_rows: u32_dim(fused_rows, "GPT-OSS gate/up source rows")?,
                    source_row_offset: u32_dim(local_start, "GPT-OSS gate/up source row offset")?,
                    target_rows: u32_dim(intermediate_native, "GPT-OSS gate/up target rows")?,
                    valid_rows: u32_dim(local_intermediate, "GPT-OSS gate/up valid rows")?,
                    source_stride_cols: u32_dim(groups, "GPT-OSS hidden group stride")?,
                    source_col_offset: 0,
                    source_cols: u32_dim(groups, "GPT-OSS hidden groups")?,
                    target_cols: u32_dim(groups, "GPT-OSS hidden groups")?,
                },
            );
            self.push_repack(
                format!("{out_base}.bias"),
                bias,
                DType::BF16,
                Encoding::Raw(DType::BF16),
                vec![experts, local_intermediate],
                RepackSpec {
                    layout: RepackLayout::DenseRowGather,
                    row_map,
                    batch: u32_dim(experts, "GPT-OSS experts")?,
                    source_rows: u32_dim(fused_rows, "GPT-OSS gate/up bias rows")?,
                    source_row_offset: u32_dim(
                        local_start,
                        "GPT-OSS gate/up bias source row offset",
                    )?,
                    target_rows: u32_dim(local_intermediate, "GPT-OSS gate/up bias target rows")?,
                    valid_rows: u32_dim(local_intermediate, "GPT-OSS gate/up bias valid rows")?,
                    source_stride_cols: 1,
                    source_col_offset: 0,
                    source_cols: 1,
                    target_cols: 1,
                },
            );
        }
        Ok(())
    }

    fn add_gpt_oss_native_down(
        &mut self,
        block: &RawTensor,
        scale: &RawTensor,
        bias: &RawTensor,
        base: &str,
    ) -> Result<(), CompileError> {
        if block.shape.len() != 4 || scale.shape.len() != 3 || bias.shape.len() != 2 {
            return Err(CompileError::InvalidInput(format!(
                "GPT-OSS native down '{}' has unsupported block/scale/bias rank",
                base
            )));
        }
        let experts = block.shape[0];
        let hidden = block.shape[1];
        let groups = block.shape[2];
        let lanes = block.shape[3];
        if lanes != 16 {
            return Err(CompileError::InvalidInput(format!(
                "GPT-OSS native down '{}' expected [E, H, I/32, 16], got {:?}",
                base, block.shape
            )));
        }
        if scale.shape != vec![experts, hidden, groups] || bias.shape != vec![experts, hidden] {
            return Err(CompileError::InvalidInput(format!(
                "GPT-OSS native down '{}' scale/bias shape mismatch",
                base
            )));
        }
        let full_intermediate = checked_mul_i64(groups, 32, "GPT-OSS intermediate size")? as i64;
        let (local_start, local_intermediate) = local_range(full_intermediate, self.target)?;
        if local_start % 32 != 0 || local_intermediate % 32 != 0 {
            return Err(CompileError::InvalidInput(format!(
                "GPT-OSS native down '{}' TP shard must align to MXFP4 32-wide groups",
                base
            )));
        }
        let local_groups = local_intermediate / 32;
        let source_group_offset = local_start / 32;
        let intermediate_native = align_up_i64(local_intermediate, 128)?;
        self.push_repack(
            format!("{base}.weight"),
            block,
            DType::BF16,
            mxfp4_encoding(Axis(2)),
            vec![experts, hidden, intermediate_native],
            RepackSpec {
                layout: RepackLayout::MarlinMxfp4Weight,
                row_map: RowMap::Identity,
                batch: u32_dim(experts, "GPT-OSS experts")?,
                source_rows: u32_dim(hidden, "GPT-OSS down source rows")?,
                source_row_offset: 0,
                target_rows: u32_dim(hidden, "GPT-OSS down target rows")?,
                valid_rows: u32_dim(hidden, "GPT-OSS down valid rows")?,
                source_stride_cols: u32_dim(full_intermediate, "GPT-OSS down source stride")?,
                source_col_offset: u32_dim(local_start, "GPT-OSS down source column offset")?,
                source_cols: u32_dim(local_intermediate, "GPT-OSS intermediate size")?,
                target_cols: u32_dim(intermediate_native, "GPT-OSS padded intermediate size")?,
            },
        );
        self.push_repack(
            format!("{base}.weight_scale"),
            scale,
            DType::U8,
            Encoding::Raw(DType::U8),
            vec![experts, hidden, intermediate_native / 32],
            RepackSpec {
                layout: RepackLayout::MarlinMxfp4Scale,
                row_map: RowMap::Identity,
                batch: u32_dim(experts, "GPT-OSS experts")?,
                source_rows: u32_dim(hidden, "GPT-OSS down source rows")?,
                source_row_offset: 0,
                target_rows: u32_dim(hidden, "GPT-OSS down target rows")?,
                valid_rows: u32_dim(hidden, "GPT-OSS down valid rows")?,
                source_stride_cols: u32_dim(groups, "GPT-OSS down source group stride")?,
                source_col_offset: u32_dim(
                    source_group_offset,
                    "GPT-OSS down source group offset",
                )?,
                source_cols: u32_dim(local_groups, "GPT-OSS down source groups")?,
                target_cols: u32_dim(intermediate_native / 32, "GPT-OSS down target groups")?,
            },
        );
        self.push_direct(bias, format!("{base}.bias"), None);
        Ok(())
    }

    fn shard_axis(&self, name: &str) -> Option<Axis> {
        if self.target.tp_size <= 1 {
            return None;
        }
        llama_like_shard_axis(name)
    }

    fn runtime_quant_scheme(&self) -> Result<Option<QuantScheme>, CompileError> {
        let mode = self.cfg.runtime_quant.as_str();
        if mode.is_empty() {
            return Ok(None);
        }
        let scheme = match mode {
            "fp8" => QuantScheme::Fp8E4M3,
            "int8" => QuantScheme::Int8Symmetric,
            other => {
                return Err(CompileError::InvalidInput(format!(
                    "unsupported runtime_quant '{other}'; expected 'fp8' or 'int8'"
                )));
            }
        };
        if !self.cfg.quant_method.is_empty() {
            return Ok(None);
        }
        if !runtime_quant_model_supported(&self.cfg.model_type) {
            return Err(CompileError::InvalidInput(format!(
                "runtime_quant={} is supported for qwen2/qwen3/qwen3_5/llama/mistral-style dense models, got '{}'",
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

fn dense_fused_projection_budget_bytes() -> u64 {
    // Fused dense projections replace the original TP1 BF16 tensors. The
    // unfused fallback binds non-owning views into the fused buffer, so this is
    // no longer a persistent duplicate-memory budget. The threshold now
    // selects which groups get a fused GEMM: all groups through 8B-class Qwen
    // models, and QKV-only above that where gate/up fusion has regressed.
    const DEFAULT_BUDGET: u64 = 10 * 1024 * 1024 * 1024;
    std::env::var("PIE_CUDA_FUSED_PROJECTION_BUDGET_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_BUDGET)
}

fn local_range(full: i64, target: &StorageTarget) -> Result<(i64, i64), CompileError> {
    let world = i64::from(target.tp_size.max(1));
    let rank = i64::from(target.tp_rank);
    if full % world != 0 {
        return Err(CompileError::InvalidInput(format!(
            "dimension {full} is not divisible by tp_size {}",
            target.tp_size
        )));
    }
    let local = full / world;
    Ok((rank * local, local))
}

fn span_for_rows(rows: i64, cols: i64, element_bytes: u64) -> Result<u64, CompileError> {
    checked_mul_i64(
        rows,
        checked_mul_i64(cols, element_bytes, "row byte size")?,
        "row span",
    )
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

fn llama_like_shard_axis(name: &str) -> Option<Axis> {
    if ends_with_any(
        name,
        &[
            ".q_proj.weight",
            ".q_proj.bias",
            ".k_proj.weight",
            ".k_proj.bias",
            ".v_proj.weight",
            ".v_proj.bias",
            ".gate_proj.weight",
            ".up_proj.weight",
            ".sinks",
            ".w1.weight",
            ".w3.weight",
            ".w1.bias",
            ".w3.bias",
            ".q_proj.weight_scale",
            ".q_proj.weight_scale_inv",
            ".k_proj.weight_scale",
            ".k_proj.weight_scale_inv",
            ".v_proj.weight_scale",
            ".v_proj.weight_scale_inv",
            ".gate_proj.weight_scale",
            ".gate_proj.weight_scale_inv",
            ".up_proj.weight_scale",
            ".up_proj.weight_scale_inv",
            ".linear_attn.in_proj_z.weight",
            ".linear_attn.in_proj_b.weight",
            ".linear_attn.in_proj_a.weight",
            ".linear_attn.dt_bias",
            ".linear_attn.A_log",
        ],
    ) {
        Some(Axis(0))
    } else if ends_with_any(
        name,
        &[
            ".o_proj.weight",
            ".down_proj.weight",
            ".w2.weight",
            ".linear_attn.out_proj.weight",
        ],
    ) {
        Some(Axis(1))
    } else if name.ends_with(".experts.down_proj") || name.ends_with(".mlp.experts.down_proj") {
        Some(Axis(2))
    } else {
        None
    }
}

fn runtime_quant_model_supported(model_type: &str) -> bool {
    matches!(
        model_type,
        "qwen3" | "qwen2" | "llama" | "llama3" | "mistral" | "qwen3_5" | "qwen3_5_text"
    )
}

fn runtime_quantizable_name(name: &str) -> bool {
    ends_with_any(
        name,
        &[
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
        ],
    )
}

fn ends_with_any(value: &str, suffixes: &[&str]) -> bool {
    suffixes.iter().any(|suffix| value.ends_with(suffix))
}

impl RuntimeTensorContract {
    fn from_ffi(view: &PieLoaderRuntimeTensorContractView) -> Result<Self, CompileError> {
        let dtype = ffi_dtype(view.dtype);
        let encoding = match view.encoding_kind {
            crate::ffi_types::PieLoaderEncodingKind::Raw => Encoding::Raw(dtype),
            crate::ffi_types::PieLoaderEncodingKind::Quant => Encoding::Quant(
                QuantSpec {
                    scheme: ffi_quant_scheme(view.quant_scheme),
                    logical_dtype: dtype,
                    bits_per_element: view.quant_bits_per_element,
                    group_size: view.quant_group_size,
                    channel_axis: ffi_optional_axis(view.quant_channel_axis)?,
                    scale_dtype: view
                        .quant_has_scale_dtype
                        .then_some(ffi_dtype(view.quant_scale_dtype)),
                    zero_point_dtype: view
                        .quant_has_zero_point_dtype
                        .then_some(ffi_dtype(view.quant_zero_point_dtype)),
                    block_shape: ffi_i64_slice(
                        view.quant_block_shape,
                        "runtime_tensor.quant_block_shape",
                    )?,
                }
                .normalized(),
            ),
        };
        let shape = ffi_i64_slice(view.shape, "runtime_tensor.shape")?;
        for dim in &shape {
            if *dim < 0 {
                return Err(CompileError::InvalidInput(format!(
                    "runtime tensor '{}' has negative dimension {}",
                    ffi_string(view.output_name, "runtime_tensor.output_name")?,
                    dim
                )));
            }
        }
        let output_name = ffi_string(view.output_name, "runtime_tensor.output_name")?;
        let shard_axis = if view.shard_axis < 0 {
            None
        } else {
            let axis = u8::try_from(view.shard_axis).map_err(|_| {
                CompileError::InvalidInput(format!(
                    "runtime tensor '{}' shard_axis {} is out of range",
                    output_name, view.shard_axis
                ))
            })?;
            Some(Axis(axis))
        };
        Ok(Self {
            output_name: output_name.clone(),
            source: runtime_source(view)?,
            metadata: ffi_u32_slice(
                view.metadata_tensor_ids,
                "runtime_tensor.metadata_tensor_ids",
            )?
            .into_iter()
            .map(TensorId)
            .collect(),
            dtype,
            encoding,
            shape,
            layout: Layout::dense(view.alignment.max(1)),
            sharding: Sharding {
                axis: shard_axis,
                world: 1,
                rank: 0,
            },
            alignment: view.alignment,
            shard_axis,
        })
    }
}

fn runtime_source(
    view: &PieLoaderRuntimeTensorContractView,
) -> Result<RuntimeTensorSource, CompileError> {
    match view.source_kind {
        PieLoaderRuntimeSourceKind::DirectTensor => {
            return Ok(RuntimeTensorSource::DirectTensor(TensorId(
                view.source_tensor_id,
            )));
        }
        PieLoaderRuntimeSourceKind::Semantic => {}
        PieLoaderRuntimeSourceKind::Join => {
            let ids = ffi_u32_slice(view.source_tensor_ids, "runtime_tensor.source_tensor_ids")?;
            if ids.is_empty() {
                return Err(CompileError::InvalidInput(
                    "Join runtime tensor source has no inputs".to_string(),
                ));
            }
            return Ok(RuntimeTensorSource::Join {
                tensors: ids.into_iter().map(TensorId).collect(),
                axis: ffi_axis(view.axis, "runtime_tensor.axis")?,
            });
        }
        PieLoaderRuntimeSourceKind::ByteSpans => {
            let spans = ffi_byte_spans(view.byte_spans, "runtime_tensor.byte_spans")?;
            if spans.is_empty() {
                return Err(CompileError::InvalidInput(
                    "ByteSpans runtime tensor source has no spans".to_string(),
                ));
            }
            return Ok(RuntimeTensorSource::ByteSpans(spans));
        }
        PieLoaderRuntimeSourceKind::Select => {
            if view.length <= 0 {
                return Err(CompileError::InvalidInput(
                    "Select runtime tensor source must have positive length".to_string(),
                ));
            }
            return Ok(RuntimeTensorSource::SelectContract {
                contract: usize::try_from(view.source_contract_id).map_err(|_| {
                    CompileError::InvalidInput(format!(
                        "Select source_contract_id {} is out of range",
                        view.source_contract_id
                    ))
                })?,
                axis: ffi_axis(view.axis, "runtime_tensor.axis")?,
                start: view.start,
                length: view.length,
            });
        }
    }
    Ok(RuntimeTensorSource::Semantic {
        role: semantic_role(view.semantic_role)?,
        layer: view.has_layer.then_some(view.layer),
        expert: view.has_expert.then_some(view.expert),
    })
}

fn ffi_byte_spans(
    slice: PieLoaderRuntimeByteSpanSlice,
    name: &'static str,
) -> Result<Vec<RuntimeByteSpan>, CompileError> {
    if slice.len == 0 {
        return Ok(Vec::new());
    }
    if slice.ptr.is_null() {
        return Err(CompileError::NullArgument(name));
    }
    unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) }
        .iter()
        .enumerate()
        .map(|(index, span)| {
            if span.span_bytes == 0 {
                return Err(CompileError::InvalidInput(format!(
                    "{name}[{index}] has zero span"
                )));
            }
            Ok(RuntimeByteSpan {
                tensor: TensorId(span.source_tensor_id),
                source_offset_bytes: span.source_offset_bytes,
                dest_offset_bytes: span.dest_offset_bytes,
                span_bytes: span.span_bytes,
            })
        })
        .collect()
}

fn ffi_u32_slice(
    slice: crate::ffi_types::PieLoaderU32Slice,
    name: &'static str,
) -> Result<Vec<u32>, CompileError> {
    if slice.len == 0 {
        return Ok(Vec::new());
    }
    if slice.ptr.is_null() {
        return Err(CompileError::NullArgument(name));
    }
    Ok(unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) }.to_vec())
}

fn ffi_axis(axis: i32, name: &'static str) -> Result<Axis, CompileError> {
    if axis < 0 {
        return Err(CompileError::InvalidInput(format!(
            "{name} must be non-negative"
        )));
    }
    Ok(Axis(u8::try_from(axis).map_err(|_| {
        CompileError::InvalidInput(format!("{name} {axis} is out of range"))
    })?))
}

fn semantic_role(role: PieLoaderSemanticRole) -> Result<SemanticRole, CompileError> {
    Ok(match role {
        PieLoaderSemanticRole::DirectTensor => {
            return Err(CompileError::InvalidInput(
                "DirectTensor is not a semantic role".to_string(),
            ));
        }
        PieLoaderSemanticRole::TokenEmbedding => SemanticRole::TokenEmbedding,
        PieLoaderSemanticRole::OutputEmbedding => SemanticRole::OutputEmbedding,
        PieLoaderSemanticRole::AttentionQ => SemanticRole::AttentionQ,
        PieLoaderSemanticRole::AttentionK => SemanticRole::AttentionK,
        PieLoaderSemanticRole::AttentionV => SemanticRole::AttentionV,
        PieLoaderSemanticRole::AttentionO => SemanticRole::AttentionO,
        PieLoaderSemanticRole::MlpGate => SemanticRole::MlpGate,
        PieLoaderSemanticRole::MlpUp => SemanticRole::MlpUp,
        PieLoaderSemanticRole::MlpDown => SemanticRole::MlpDown,
        PieLoaderSemanticRole::ExpertRouter => SemanticRole::ExpertRouter,
        PieLoaderSemanticRole::ExpertGate => SemanticRole::ExpertGate,
        PieLoaderSemanticRole::ExpertUp => SemanticRole::ExpertUp,
        PieLoaderSemanticRole::ExpertDown => SemanticRole::ExpertDown,
        PieLoaderSemanticRole::ExpertBias => SemanticRole::ExpertBias,
        PieLoaderSemanticRole::Norm => SemanticRole::Norm,
        PieLoaderSemanticRole::QuantData => SemanticRole::QuantData,
        PieLoaderSemanticRole::QuantScale => SemanticRole::QuantScale,
        PieLoaderSemanticRole::QuantZeroPoint => SemanticRole::QuantZeroPoint,
        PieLoaderSemanticRole::QuantGroupIndex => SemanticRole::QuantGroupIndex,
    })
}
