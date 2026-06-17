use crate::abi::{RuntimeAbi, RuntimeTensorContract, RuntimeTensorSource};
use crate::error::CompileError;
use crate::ir::{ByteSpan, LayoutExpr, LayoutPlan};
use crate::semantic::SemanticGraph;
use crate::source::{CheckpointMetadata, RawTensor};
use crate::storage::StorageTarget;
use crate::types::{
    DType, Encoding, QuantScheme, QuantSpec, Sharding, TensorDecl, TensorId, encoding_nbytes,
};

pub fn plan_from_semantics(
    metadata: &CheckpointMetadata,
    graph: &SemanticGraph,
    abi: &RuntimeAbi,
    target: &StorageTarget,
) -> Result<LayoutPlan, CompileError> {
    let mut plan = LayoutPlan::new();
    let mut contract_values = Vec::with_capacity(abi.tensors.len());
    let mut next_generated_tensor = abi.tensors.len() as u32;
    for (output_index, contract) in abi.tensors.iter().enumerate() {
        let (value_id, output_id) = lower_contract(
            metadata,
            graph,
            contract,
            target,
            &mut plan,
            &contract_values,
            TensorId(output_index as u32),
            &mut next_generated_tensor,
        )?;
        contract_values.push(value_id);
        plan.outputs.push(output_id);
    }
    Ok(plan)
}

fn lower_contract(
    metadata: &CheckpointMetadata,
    graph: &SemanticGraph,
    contract: &RuntimeTensorContract,
    target: &StorageTarget,
    plan: &mut LayoutPlan,
    contract_values: &[crate::types::ExprId],
    output_id: TensorId,
    next_generated_tensor: &mut u32,
) -> Result<(crate::types::ExprId, crate::types::ExprId), CompileError> {
    let (mut current, mut current_decl) =
        lower_contract_source(metadata, graph, contract, plan, contract_values, output_id)?;
    let metadata_values = lower_metadata_sources(metadata, contract, plan)?;
    // Shapes must match for an identity load. Leading singleton dims are a
    // no-op for the byte layout (same numel, same row-major order), but ggml's
    // ne array can't preserve a leading-1 source dim (it becomes a trailing ggml
    // dim and is trimmed) — e.g. a Conv1d weight [1, C, K] (out_channels=1)
    // declares as runtime shape [C, K]. Treat shapes equal-after-stripping-
    // leading-ones as a match so such conv weights load.
    let strip_leading_ones = |s: &[i64]| -> Vec<i64> {
        let first = s.iter().position(|&d| d != 1).unwrap_or(s.len());
        s[first..].to_vec()
    };
    if current_decl.shape != contract.shape
        && strip_leading_ones(&current_decl.shape) != strip_leading_ones(&contract.shape)
    {
        return Err(CompileError::InvalidInput(format!(
            "runtime tensor '{}' shape {:?} does not match source shape {:?}",
            contract.output_name, contract.shape, current_decl.shape
        )));
    }

    if let Some(axis) = contract.shard_axis.filter(|_| target.tp_size > 1) {
        let axis_index = axis.0 as usize;
        if axis_index >= current_decl.shape.len()
            || current_decl.shape[axis_index] % i64::from(target.tp_size) != 0
        {
            return Err(CompileError::InvalidInput(format!(
                "runtime tensor '{}' cannot be partitioned on axis {} into {} ranks",
                contract.output_name, axis.0, target.tp_size
            )));
        }
        let mut decl = current_decl.clone();
        decl.shape[axis_index] /= i64::from(target.tp_size);
        decl.sharding = Sharding {
            axis: Some(axis),
            world: target.tp_size,
            rank: target.tp_rank,
        };
        current = plan.push(LayoutExpr::Partition {
            input: current,
            axis,
            parts: target.tp_size,
            index: target.tp_rank,
            decl: decl.clone(),
        });
        current_decl = decl;
    }

    current = lower_encoding_change(
        plan,
        current,
        &metadata_values,
        &mut current_decl,
        contract,
        output_id,
        next_generated_tensor,
    )?;

    let target_layout = contract.layout.clone();
    if current_decl.layout != target_layout || current_decl.alignment != contract.alignment {
        let mut decl = current_decl.clone();
        decl.layout = target_layout.clone();
        decl.alignment = contract.alignment;
        current = plan.push(LayoutExpr::View {
            input: current,
            layout: target_layout.clone(),
            axis: None,
            start: 0,
            length: 0,
            decl: decl.clone(),
        });
        current_decl = decl;
    }

    let realized_decl = TensorDecl {
        id: output_id,
        name: contract.output_name.clone(),
        shape: current_decl.shape.clone(),
        encoding: contract.encoding.clone(),
        layout: target_layout,
        sharding: if target.tp_size > 1 && contract.shard_axis.is_some() {
            current_decl.sharding
        } else {
            contract.sharding
        },
        alignment: contract.alignment,
    };
    current_decl = realized_decl.clone();

    let value = current;
    let realized = plan.push(LayoutExpr::Realize {
        input: current,
        runtime_name: contract.output_name.clone(),
        decl: current_decl,
    });
    Ok((value, realized))
}

fn lower_metadata_sources(
    metadata: &CheckpointMetadata,
    contract: &RuntimeTensorContract,
    plan: &mut LayoutPlan,
) -> Result<Vec<crate::types::ExprId>, CompileError> {
    let mut values = Vec::with_capacity(contract.metadata.len());
    for tensor_id in &contract.metadata {
        let raw = metadata.tensor(*tensor_id).ok_or_else(|| {
            CompileError::InvalidInput(format!(
                "runtime tensor '{}' references missing metadata tensor {}",
                contract.output_name, tensor_id.0
            ))
        })?;
        values.push(plan.push(LayoutExpr::Source {
            tensor: raw.id,
            decl: source_decl(raw),
        }));
    }
    Ok(values)
}

fn lower_contract_source(
    metadata: &CheckpointMetadata,
    graph: &SemanticGraph,
    contract: &RuntimeTensorContract,
    plan: &mut LayoutPlan,
    contract_values: &[crate::types::ExprId],
    output_id: TensorId,
) -> Result<(crate::types::ExprId, TensorDecl), CompileError> {
    match &contract.source {
        RuntimeTensorSource::DirectTensor(_) | RuntimeTensorSource::Semantic { .. } => {
            let raw = resolve_raw_tensor(metadata, graph, contract)?;
            let source_decl = source_decl(raw);
            let source = plan.push(LayoutExpr::Source {
                tensor: raw.id,
                decl: source_decl.clone(),
            });
            Ok((source, source_decl))
        }
        RuntimeTensorSource::ByteSpans(spans) => {
            let decl = contract_decl(contract, output_id);
            for span in spans {
                let raw = metadata.tensor(span.tensor).ok_or_else(|| {
                    CompileError::InvalidInput(format!(
                        "runtime tensor '{}' references missing ByteSpans source tensor {}",
                        contract.output_name, span.tensor.0
                    ))
                })?;
                let end = span
                    .source_offset_bytes
                    .checked_add(span.span_bytes)
                    .ok_or_else(|| {
                        CompileError::InvalidInput(format!(
                            "runtime tensor '{}' ByteSpans source offset overflow",
                            contract.output_name
                        ))
                    })?;
                if end > raw.span_bytes {
                    return Err(CompileError::InvalidInput(format!(
                        "runtime tensor '{}' ByteSpans source range exceeds '{}'",
                        contract.output_name, raw.name
                    )));
                }
            }
            let value = plan.push(LayoutExpr::ByteSpans {
                spans: spans
                    .iter()
                    .map(|span| ByteSpan {
                        tensor: span.tensor,
                        source_offset_bytes: span.source_offset_bytes,
                        dest_offset_bytes: span.dest_offset_bytes,
                        span_bytes: span.span_bytes,
                    })
                    .collect(),
                decl: decl.clone(),
            });
            Ok((value, decl))
        }
        RuntimeTensorSource::Join { tensors, axis } => {
            let mut inputs = Vec::with_capacity(tensors.len());
            for tensor_id in tensors {
                let raw = metadata.tensor(*tensor_id).ok_or_else(|| {
                    CompileError::InvalidInput(format!(
                        "runtime tensor '{}' references missing Join source tensor {}",
                        contract.output_name, tensor_id.0
                    ))
                })?;
                let mut decl = source_decl(raw);
                decl.layout = contract.layout.clone();
                decl.alignment = contract.alignment;
                inputs.push(plan.push(LayoutExpr::Source {
                    tensor: raw.id,
                    decl,
                }));
            }
            let decl = contract_decl(contract, output_id);
            let joined = plan.push(LayoutExpr::Join {
                inputs,
                axis: *axis,
                decl: decl.clone(),
            });
            Ok((joined, decl))
        }
        RuntimeTensorSource::SelectContract {
            contract: source_contract,
            axis,
            start,
            length,
        } => {
            let input = contract_values
                .get(*source_contract)
                .copied()
                .ok_or_else(|| {
                    CompileError::InvalidInput(format!(
                        "runtime tensor '{}' references source contract {} before it exists",
                        contract.output_name, source_contract
                    ))
                })?;
            let decl = contract_decl(contract, output_id);
            let selected = plan.push(LayoutExpr::View {
                input,
                layout: decl.layout.clone(),
                axis: Some(*axis),
                start: *start,
                length: *length,
                decl: decl.clone(),
            });
            Ok((selected, decl))
        }
        RuntimeTensorSource::Repack { tensor, spec } => {
            let raw = metadata.tensor(*tensor).ok_or_else(|| {
                CompileError::InvalidInput(format!(
                    "runtime tensor '{}' references missing Repack source tensor {}",
                    contract.output_name, tensor.0
                ))
            })?;
            let source_decl = source_decl(raw);
            let source = plan.push(LayoutExpr::Source {
                tensor: raw.id,
                decl: source_decl,
            });
            let decl = contract_decl(contract, output_id);
            let repacked = plan.push(LayoutExpr::Repack {
                input: source,
                spec: *spec,
                decl: decl.clone(),
            });
            Ok((repacked, decl))
        }
    }
}

fn resolve_raw_tensor<'a>(
    metadata: &'a CheckpointMetadata,
    graph: &SemanticGraph,
    contract: &RuntimeTensorContract,
) -> Result<&'a RawTensor, CompileError> {
    let source_id = match &contract.source {
        RuntimeTensorSource::DirectTensor(id) => *id,
        RuntimeTensorSource::Semantic {
            role,
            layer,
            expert,
        } => {
            let mut matches = graph.tensors.iter().filter(|tensor| {
                tensor.role == *role
                    && layer.is_none_or(|layer| tensor.layer == Some(layer))
                    && expert.is_none_or(|expert| tensor.expert == Some(expert))
            });
            let first = matches.next().ok_or_else(|| {
                CompileError::InvalidInput(format!(
                    "runtime tensor '{}' could not match semantic role {:?}",
                    contract.output_name, role
                ))
            })?;
            if matches.next().is_some() {
                return Err(CompileError::InvalidInput(format!(
                    "runtime tensor '{}' semantic role {:?} is ambiguous",
                    contract.output_name, role
                )));
            }
            first.raw
        }
        RuntimeTensorSource::ByteSpans(_)
        | RuntimeTensorSource::Join { .. }
        | RuntimeTensorSource::Repack { .. }
        | RuntimeTensorSource::SelectContract { .. } => {
            return Err(CompileError::InvalidInput(format!(
                "runtime tensor '{}' has no single raw source",
                contract.output_name
            )));
        }
    };
    metadata.tensor(source_id).ok_or_else(|| {
        CompileError::InvalidInput(format!(
            "runtime tensor '{}' references missing source tensor {}",
            contract.output_name, source_id.0
        ))
    })
}

fn contract_decl(contract: &RuntimeTensorContract, output_id: TensorId) -> TensorDecl {
    TensorDecl {
        id: output_id,
        name: contract.output_name.clone(),
        shape: contract.shape.clone(),
        encoding: contract.encoding.clone(),
        layout: contract.layout.clone(),
        sharding: contract.sharding,
        alignment: contract.alignment,
    }
}

fn lower_encoding_change(
    plan: &mut LayoutPlan,
    input: crate::types::ExprId,
    metadata: &[crate::types::ExprId],
    current_decl: &mut TensorDecl,
    contract: &RuntimeTensorContract,
    output_id: TensorId,
    next_generated_tensor: &mut u32,
) -> Result<crate::types::ExprId, CompileError> {
    if current_decl.encoding == contract.encoding {
        return Ok(input);
    }
    match (current_decl.encoding.clone(), contract.encoding.clone()) {
        (Encoding::Raw(_), Encoding::Raw(dtype)) => {
            let mut decl = current_decl.clone();
            decl.encoding = Encoding::Raw(dtype);
            *current_decl = decl.clone();
            Ok(plan.push(LayoutExpr::Cast { input, dtype, decl }))
        }
        (Encoding::Quant(source), Encoding::Raw(dtype)) => {
            if source.logical_dtype != dtype {
                return Err(CompileError::InvalidInput(format!(
                    "runtime tensor '{}' requests raw {:?} from quantized {:?}",
                    contract.output_name, dtype, source.logical_dtype
                )));
            }
            let mut decl = current_decl.clone();
            decl.encoding = Encoding::Raw(dtype);
            *current_decl = decl.clone();
            Ok(plan.push(LayoutExpr::Decode {
                scheme: source.scheme,
                data: input,
                metadata: metadata.to_vec(),
                decl,
            }))
        }
        (Encoding::Raw(_), Encoding::Quant(target)) => {
            let mut decl = current_decl.clone();
            decl.id = output_id;
            decl.name = contract.output_name.clone();
            decl.encoding = Encoding::Quant(target.clone());
            *current_decl = decl.clone();
            let metadata_outputs =
                runtime_quant_metadata_outputs(contract, &decl, next_generated_tensor);
            Ok(plan.push(LayoutExpr::Encode {
                scheme: target.scheme,
                input,
                metadata_outputs,
                decl,
            }))
        }
        (Encoding::Quant(source), Encoding::Quant(target)) => {
            let mut decl = current_decl.clone();
            decl.encoding = Encoding::Quant(target.clone());
            *current_decl = decl.clone();
            Ok(plan.push(LayoutExpr::Transcode {
                from: source.scheme,
                to: target.scheme,
                data: input,
                metadata: metadata.to_vec(),
                metadata_outputs: Vec::new(),
                decl,
            }))
        }
    }
}

fn runtime_quant_metadata_outputs(
    contract: &RuntimeTensorContract,
    quant_decl: &TensorDecl,
    next_generated_tensor: &mut u32,
) -> Vec<TensorDecl> {
    let Encoding::Quant(spec) = &quant_decl.encoding else {
        return Vec::new();
    };
    if quant_decl.shape.len() != 2 {
        return Vec::new();
    }
    let (name, shape, encoding) = match spec.scheme {
        QuantScheme::Fp8E4M3 | QuantScheme::Int8Symmetric => (
            format!("{}_scale_inv", contract.output_name),
            vec![quant_decl.shape[0]],
            Encoding::Raw(DType::F32),
        ),
        QuantScheme::Mxfp4E2M1E8M0 => {
            // E8M0 block scale: 1 uint8 byte per 32-element block along the
            // K (column) dimension. The encode-tile kernel writes a row-major
            // `[rows, cols/32]` byte tensor. Name matches GPT-OSS naming so
            // downstream paths that look up `*.weight_scale` find it.
            let cols = quant_decl.shape[1];
            if cols % 32 != 0 {
                return Vec::new();
            }
            (
                format!("{}_scale", contract.output_name),
                vec![quant_decl.shape[0], cols / 32],
                Encoding::Raw(DType::U8),
            )
        }
        _ => return Vec::new(),
    };
    let id = TensorId(*next_generated_tensor);
    *next_generated_tensor = next_generated_tensor.saturating_add(1);
    vec![TensorDecl {
        id,
        name,
        shape,
        encoding,
        layout: contract.layout.clone(),
        sharding: quant_decl.sharding,
        alignment: contract.alignment,
    }]
}

fn source_decl(raw: &RawTensor) -> TensorDecl {
    let alignment = raw.layout.alignment.max(1);
    let encoding = crate::types::normalize_encoding(&raw.encoding);
    TensorDecl {
        id: raw.id,
        name: raw.name.clone(),
        shape: raw.shape.clone(),
        encoding,
        layout: raw.layout.clone(),
        sharding: Sharding::replicated(),
        alignment,
    }
}

pub fn runtime_bytes(shape: &[i64], encoding: &Encoding) -> Result<u64, CompileError> {
    encoding_nbytes(shape, encoding)
        .ok_or_else(|| CompileError::InvalidInput("runtime tensor byte size overflow".to_string()))
}

#[allow(dead_code)]
fn _raw_quant_spec(dtype: DType) -> QuantSpec {
    QuantSpec {
        scheme: crate::types::QuantScheme::None,
        logical_dtype: dtype,
        bits_per_element: 0,
        group_size: 0,
        channel_axis: None,
        scale_dtype: None,
        zero_point_dtype: None,
        block_shape: Vec::new(),
    }
}
