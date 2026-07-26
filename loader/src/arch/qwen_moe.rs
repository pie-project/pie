//! Qwen3-MoE expert stacking and fused gate-up TP slicing.

use super::*;

impl DefaultAbiBuilder<'_> {
    pub(super) fn add_fused_moe_gate_up_tp_slices(&mut self) -> Result<(), CompileError> {
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
            dense_element_bytes(raw, "fused MoE gate/up")?;
            let experts = raw.shape[0];
            let full_i = raw.shape[1] / 2;
            let hidden = raw.shape[2];
            let (local_start, local_i) = local_range(
                full_i,
                self.target,
                &format!("the intermediate size of '{}'", raw.name),
            )?;
            // Gate rows [0, I) and up rows [I, 2I) are sharded independently
            // and re-joined, so the local halves stay adjacent per expert.
            let src = Expr::src(raw.name.clone());
            self.push_expr(
                raw.name.clone(),
                raw,
                vec![experts, 2 * local_i, hidden],
                Expr::cat(
                    1,
                    vec![
                        src.clone().slice(1, local_start, local_i),
                        src.slice(1, full_i + local_start, local_i),
                    ],
                ),
            );
        }
        Ok(())
    }

    /// Stack per-expert MoE weights into the fused 3-D tensors the qwen3_5_moe
    /// forward consumes. HF Qwen3-MoE source layout: per expert `e`,
    /// `mlp.experts.{e}.gate_proj.weight` / `.up_proj.weight` are `[I, H]` and
    /// `.down_proj.weight` is `[H, I]`. Output:
    ///   `mlp.experts.gate_up_proj` -> `[E, 2I, H]` (rows `[0,I)`=gate, `[I,2I)`=up)
    ///   `mlp.experts.down_proj`    -> `[E, H, I]`  (per-expert stack)
    /// Built as multi-source byte spans (no GPU-side duplicate). No-op when the
    /// checkpoint already ships the fused tensors (qwen3_5_moe pre-fused case).
    pub(super) fn add_qwen_moe_expert_stacks(&mut self) -> Result<(), CompileError> {
        if !self.profile().stack_per_expert_moe || self.target.tp_size != 1 {
            // TP>1 per-expert sharding is not wired yet; the bind then fails
            // loudly on the missing fused tensor rather than loading silently wrong.
            return Ok(());
        }
        let num_experts = self.cfg.num_experts as i64;
        if num_experts <= 0 {
            return Ok(());
        }

        struct LayerStack {
            gate_up_name: String,
            down_name: String,
            gate_up_shape: Vec<i64>,
            down_shape: Vec<i64>,
            gate_up: Expr,
            down: Expr,
            dtype: DType,
            consumed: Vec<TensorId>,
        }

        // Index sources by name once: the lookups below are O(layers × experts),
        // and a linear scan per lookup would make the whole pass quadratic.
        let by_name: std::collections::HashMap<&str, &RawTensor> = self
            .metadata
            .tensors
            .iter()
            .map(|raw| (raw.name.as_str(), raw))
            .collect();

        let mut stacks: Vec<LayerStack> = Vec::new();
        for layer in 0..self.cfg.num_hidden_layers {
            let prefix = format!("model.layers.{layer}.mlp.experts.");
            let gate_up_name = format!("{prefix}gate_up_proj");
            if by_name.contains_key(gate_up_name.as_str()) {
                continue; // already pre-fused; leave to the direct/TP-slice paths.
            }
            let Some(gate0) = by_name
                .get(format!("{prefix}0.gate_proj.weight").as_str())
                .copied()
            else {
                continue; // not a per-expert checkpoint at this layer.
            };
            if gate0.shape.len() != 2 {
                return Err(CompileError::InvalidInput(format!(
                    "qwen moe expert stack: '{}' expected 2-D, got {:?}",
                    gate0.name, gate0.shape
                )));
            }
            let inter = gate0.shape[0];
            let hidden = gate0.shape[1];
            let dtype = self.dtype(gate0);
            dense_element_bytes(gate0, "qwen moe expert stack")?;

            let mut gate_up_parts = Vec::with_capacity(num_experts as usize);
            let mut down_parts = Vec::with_capacity(num_experts as usize);
            let mut consumed = Vec::with_capacity((num_experts * 3) as usize);
            for e in 0..num_experts {
                let g = by_name
                    .get(format!("{prefix}{e}.gate_proj.weight").as_str())
                    .copied();
                let u = by_name
                    .get(format!("{prefix}{e}.up_proj.weight").as_str())
                    .copied();
                let d = by_name
                    .get(format!("{prefix}{e}.down_proj.weight").as_str())
                    .copied();
                let (Some(g), Some(u), Some(d)) = (g, u, d) else {
                    return Err(CompileError::InvalidInput(format!(
                        "qwen moe expert stack: layer {layer} expert {e} missing gate/up/down"
                    )));
                };
                if g.shape != [inter, hidden]
                    || u.shape != [inter, hidden]
                    || d.shape != [hidden, inter]
                {
                    return Err(CompileError::InvalidInput(format!(
                        "qwen moe expert stack: layer {layer} expert {e} shape mismatch \
                         (gate {:?} up {:?} down {:?}, expected gate/up [{inter},{hidden}] \
                         down [{hidden},{inter}])",
                        g.shape, u.shape, d.shape
                    )));
                }
                if self.dtype(g) != dtype || self.dtype(u) != dtype || self.dtype(d) != dtype {
                    return Err(CompileError::InvalidInput(format!(
                        "qwen moe expert stack: layer {layer} expert {e} dtype mismatch"
                    )));
                }
                // One expert slab is gate over up; the leading 1 makes the
                // per-expert concatenation a stack.
                gate_up_parts.push(
                    Expr::cat(
                        0,
                        vec![Expr::src(g.name.clone()), Expr::src(u.name.clone())],
                    )
                    .reshape(vec![1, 2 * inter, hidden]),
                );
                down_parts.push(Expr::src(d.name.clone()).reshape(vec![1, hidden, inter]));
                consumed.push(g.id);
                consumed.push(u.id);
                consumed.push(d.id);
            }

            stacks.push(LayerStack {
                dtype,
                gate_up_name,
                down_name: format!("{prefix}down_proj"),
                gate_up_shape: vec![num_experts, 2 * inter, hidden],
                down_shape: vec![num_experts, hidden, inter],
                gate_up: Expr::cat(0, gate_up_parts),
                down: Expr::cat(0, down_parts),
                consumed,
            });
        }

        let align = self.alignment();
        for s in stacks {
            self.tensors.push(RuntimeTensorContract {
                output_name: s.gate_up_name,
                expr: s.gate_up,
                encoding: Encoding::Raw(s.dtype),
                shape: s.gate_up_shape,
                layout: Layout::dense(align),
                alignment: align,
                shard_axis: None,
            });
            self.tensors.push(RuntimeTensorContract {
                output_name: s.down_name,
                expr: s.down,
                encoding: Encoding::Raw(s.dtype),
                shape: s.down_shape,
                layout: Layout::dense(align),
                alignment: align,
                shard_axis: None,
            });
            for id in s.consumed {
                self.consumed.insert(id);
            }
        }
        Ok(())
    }
}
