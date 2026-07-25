//! Metal Qwen3.5 storage schema and its MLX affine-U4 contracts.

use super::*;

pub(super) fn metal_qwen35_runtime_name(raw_name: &str) -> Result<Option<String>, CompileError> {
    if raw_name.starts_with("model.visual.") || raw_name.starts_with("mtp.") {
        return Ok(None);
    }
    if let Some(suffix) = raw_name.strip_prefix("lm_head.") {
        return Ok(Some(format!("shared_embedding.{suffix}")));
    }
    if raw_name == "model.language_model.norm.weight" {
        return Ok(Some("final_norm.weight".to_string()));
    }
    let Some(rest) = raw_name.strip_prefix("model.language_model.layers.") else {
        return Err(CompileError::InvalidInput(format!(
            "Metal Qwen3.5 schema has no declared mapping or skip for '{raw_name}'"
        )));
    };
    let Some((layer, suffix)) = rest.split_once('.') else {
        return Err(CompileError::InvalidInput(format!(
            "Metal Qwen3.5 layer tensor '{raw_name}' is malformed"
        )));
    };
    if layer.is_empty() || !layer.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(CompileError::InvalidInput(format!(
            "Metal Qwen3.5 layer tensor '{raw_name}' has an invalid layer index"
        )));
    }
    Ok(Some(format!("layers.{layer}.{suffix}")))
}

impl DefaultAbiBuilder<'_> {
    pub(super) fn add_metal_qwen35_contracts(&mut self) -> Result<(), CompileError> {
        let tensors = self.metadata.tensors.clone();
        for raw in &tensors {
            let Some(output_name) = metal_qwen35_runtime_name(&raw.name)? else {
                continue;
            };
            if raw.name.ends_with(".weight") && raw.encoding == Encoding::Raw(DType::U32) {
                let base = raw.name.strip_suffix(".weight").expect("suffix checked");
                let (Some(scales), Some(biases)) = (
                    tensors
                        .iter()
                        .find(|candidate| candidate.name == format!("{base}.scales")),
                    tensors
                        .iter()
                        .find(|candidate| candidate.name == format!("{base}.biases")),
                ) else {
                    return Err(CompileError::InvalidInput(format!(
                        "Metal affine-U4 weight '{}' is missing scales or biases",
                        raw.name
                    )));
                };
                self.push_mlx_affine_u4(raw, scales, biases, output_name)?;
            } else {
                self.push_direct(raw, output_name, None);
            }
        }
        if self.tensors.is_empty() {
            return Err(CompileError::InvalidInput(
                "Metal Qwen3.5 schema found no text-decoder tensors".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn push_mlx_affine_u4(
        &mut self,
        raw: &RawTensor,
        scales: &RawTensor,
        biases: &RawTensor,
        output_name: String,
    ) -> Result<(), CompileError> {
        if raw.shape.len() != 2 || scales.shape.len() != 2 || biases.shape != scales.shape {
            return Err(CompileError::InvalidInput(format!(
                "MLX affine-U4 triplet '{}' has incompatible shapes: weight={:?} scales={:?} biases={:?}",
                raw.name, raw.shape, scales.shape, biases.shape
            )));
        }
        let rows = raw.shape[0];
        let logical_cols = raw.shape[1].checked_mul(8).ok_or_else(|| {
            CompileError::InvalidInput("MLX U4 logical shape overflow".to_string())
        })?;
        if rows != scales.shape[0] || scales.shape[1] <= 0 || logical_cols % scales.shape[1] != 0 {
            return Err(CompileError::InvalidInput(format!(
                "MLX affine-U4 triplet '{}' cannot derive a group size from weight={:?} scales={:?}",
                raw.name, raw.shape, scales.shape
            )));
        }
        let group_size = u32::try_from(logical_cols / scales.shape[1]).map_err(|_| {
            CompileError::InvalidInput("MLX affine-U4 group size does not fit u32".to_string())
        })?;
        let scale_dtype = dtype_for_encoding(&scales.encoding);
        let bias_dtype = dtype_for_encoding(&biases.encoding);
        self.tensors.push(RuntimeTensorContract {
            output_name,
            source: RuntimeTensorSource::ByteSpans(vec![RuntimeByteSpan {
                tensor: raw.id,
                source_offset_bytes: 0,
                dest_offset_bytes: 0,
                span_bytes: raw.span_bytes,
            }]),
            metadata: Vec::new(),
            dtype: DType::BF16,
            encoding: Encoding::Quant(
                QuantSpec {
                    scheme: QuantScheme::MlxAffineU4,
                    logical_dtype: DType::BF16,
                    bits_per_element: 4,
                    group_size,
                    channel_axis: Some(Axis(1)),
                    scale_dtype: Some(scale_dtype),
                    zero_point_dtype: Some(bias_dtype),
                    block_shape: vec![i64::from(group_size)],
                }
                .normalized(),
            ),
            shape: vec![rows, logical_cols],
            layout: Layout::dense(self.alignment()),
            sharding: Sharding::replicated(),
            alignment: self.alignment(),
            shard_axis: None,
        });
        Ok(())
    }
}
