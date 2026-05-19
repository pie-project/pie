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
use crate::types::{Axis, DType, Encoding, Layout, QuantSpec, Sharding, TensorId};
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

impl DefaultAbiBuilder<'_> {
    fn build(&mut self) -> Result<(), CompileError> {
        self.add_phi3_fused_splits()?;
        self.add_gpt_oss_mxfp4_groups();
        self.add_fused_moe_gate_up_tp_slices()?;
        for raw in &self.metadata.tensors {
            if self.consumed.contains(&raw.id) {
                continue;
            }
            self.push_direct(raw, raw.name.clone(), self.shard_axis(&raw.name));
        }
        Ok(())
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

    fn add_gpt_oss_mxfp4_groups(&mut self) {
        if !self.cfg.model_type.eq_ignore_ascii_case("gpt_oss")
            && !self.cfg.model_type.eq_ignore_ascii_case("gpt-oss")
            && !self.cfg.model_type.eq_ignore_ascii_case("gptoss")
        {
            return;
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
            self.push_direct(block, format!("{base}.weight"), None);
            self.push_direct(scale, format!("{base}.weight_scale"), None);
            self.push_direct(bias, format!("{base}.bias"), None);
            self.consumed.insert(block.id);
            self.consumed.insert(scale.id);
            self.consumed.insert(bias.id);
        }
    }

    fn shard_axis(&self, name: &str) -> Option<Axis> {
        if self.target.tp_size <= 1 {
            return None;
        }
        llama_like_shard_axis(name)
    }
}

fn dtype_for_encoding(encoding: &Encoding) -> DType {
    match encoding {
        Encoding::Raw(dtype) => *dtype,
        Encoding::Quant(spec) => spec.logical_dtype,
    }
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
