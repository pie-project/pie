//! Nemotron-H packed-expert views.

use super::*;

impl DefaultAbiBuilder<'_> {
    pub(super) fn add_nemotron_h_packed_expert_views(&mut self) -> Result<(), CompileError> {
        if self.target.backend != BackendKind::Cuda
            || !self.profile().nemotron_packed_experts
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

    pub(super) fn add_nemotron_h_layer_packed_experts(
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
}
