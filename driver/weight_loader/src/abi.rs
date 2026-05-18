use crate::error::CompileError;
use crate::ffi_types::{
    PieLoaderRuntimeAbiView, PieLoaderRuntimeByteSpanSlice, PieLoaderRuntimeSourceKind,
    PieLoaderRuntimeTensorContractView, PieLoaderSemanticRole,
};
use crate::semantic::SemanticRole;
use crate::source::{ffi_dtype, ffi_i64_slice, ffi_quant_scheme, ffi_string};
use crate::types::{Axis, DType, Encoding, Layout, QuantSpec, Sharding, TensorId};

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
}

impl RuntimeTensorContract {
    fn from_ffi(view: &PieLoaderRuntimeTensorContractView) -> Result<Self, CompileError> {
        let dtype = ffi_dtype(view.dtype);
        let encoding = match view.encoding_kind {
            crate::ffi_types::PieLoaderEncodingKind::Raw => Encoding::Raw(dtype),
            crate::ffi_types::PieLoaderEncodingKind::Quant => Encoding::Quant(QuantSpec {
                scheme: ffi_quant_scheme(view.quant_scheme),
                logical_dtype: dtype,
                bits_per_element: 0,
                group_size: 0,
                channel_axis: None,
                scale_dtype: None,
                zero_point_dtype: None,
                block_shape: Vec::new(),
            }),
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
