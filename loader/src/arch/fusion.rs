//! Fused projection joins (dense QKV / gate-up and MLA q_a+kv_a).

use super::*;

pub(super) struct FusedProjectionCandidate {
    output_name: String,
    tensors: Vec<TensorId>,
    names: Vec<String>,
    rows: i64,
    cols: i64,
    bytes: u64,
}

pub(super) fn dense_fused_projection_budget_bytes() -> u64 {
    // Fused dense projections replace the original TP1 BF16 tensors. The
    // unfused fallback binds non-owning views into the fused buffer, so this is
    // no longer a persistent duplicate-memory budget. The threshold now
    // selects which groups get a fused GEMM: all groups through 8B-class Qwen
    // models, and QKV-only above that where gate/up fusion has regressed.
    const DEFAULT_BUDGET: u64 = 10 * 1024 * 1024 * 1024;
    DEFAULT_BUDGET
}

impl DefaultAbiBuilder<'_> {
    pub(super) fn add_dense_fused_projection_joins(
        &mut self,
        runtime_quant_enabled: bool,
    ) -> Result<(), CompileError> {
        if self.target.backend != BackendKind::Cuda
            || self.target.tp_size != 1
            || runtime_quant_enabled
            || self.profile().skip_dense_qkv_fusion
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
                expr: Expr::cat(0, candidate.names.iter().cloned().map(Expr::src).collect()),
                encoding: Encoding::Raw(DType::BF16),
                shape: vec![candidate.rows, candidate.cols],
                layout: Layout::dense(self.alignment()),
                alignment: self.alignment(),
            });
        }

        Ok(())
    }

    pub(super) fn add_mla_fused_projection_joins(&mut self) -> Result<(), CompileError> {
        if self.target.backend != BackendKind::Cuda {
            return Ok(());
        }
        if !self.profile().mla_fused_joins {
            return Ok(());
        }

        let mut candidates = Vec::new();
        for layer in 0..self.cfg.num_hidden_layers {
            let p = format!("model.layers.{layer}.");
            // Fuse q_a_proj + kv_a_proj_with_mqa (same input: norm_x, unsharded)
            if let Some(c) = self.fused_join_candidate(
                &(p.clone() + "self_attn.q_kv_a_proj.fused.weight"),
                &[
                    p.clone() + "self_attn.q_a_proj.weight",
                    p.clone() + "self_attn.kv_a_proj_with_mqa.weight",
                ],
            )? {
                candidates.push(c);
            }
            // Fuse shared gate + up (same input: norm_y)
            if let Some(c) = self.fused_join_candidate(
                &(p.clone() + "mlp.shared_experts.gate_up_proj.fused.weight"),
                &[
                    p.clone() + "mlp.shared_experts.gate_proj.weight",
                    p.clone() + "mlp.shared_experts.up_proj.weight",
                ],
            )? {
                candidates.push(c);
            }
        }

        for candidate in candidates {
            for tensor in &candidate.tensors {
                self.consumed.insert(*tensor);
            }
            self.tensors.push(RuntimeTensorContract {
                output_name: candidate.output_name,
                expr: Expr::cat(0, candidate.names.iter().cloned().map(Expr::src).collect()),
                encoding: Encoding::Raw(DType::BF16),
                shape: vec![candidate.rows, candidate.cols],
                layout: Layout::dense(self.alignment()),
                alignment: self.alignment(),
            });
        }
        Ok(())
    }

    pub(super) fn fused_join_candidate(
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
        let mut names = Vec::with_capacity(input_names.len());
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
            names.push(raw.name.clone());
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
            names,
            rows,
            cols: cols.unwrap_or(0),
            bytes,
        }))
    }
}
