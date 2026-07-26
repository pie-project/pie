//! Phi-3 fused qkv / gate_up splits.

use super::*;

impl ContractBuilder<'_> {
    pub(super) fn add_phi3_fused_splits(&mut self) -> Result<(), CompileError> {
        if !self.profile().phi3_fused_splits {
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

    pub(super) fn add_phi3_qkv_split(&mut self, raw: &RawTensor) -> Result<(), CompileError> {
        if raw.shape.len() != 2 {
            return Err(CompileError::InvalidInput(format!(
                "Phi-3 fused QKV '{}' must be 2-D",
                raw.name
            )));
        }
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
            let (local_start, local_rows) = local_range(
                rows,
                self.target,
                &format!("the row count of '{}'", raw.name),
            )?;
            self.push_expr(
                format!("{base}.self_attn.{proj}.weight"),
                raw,
                vec![local_rows, cols],
                Expr::src(raw.name.clone()).slice(0, start + local_start, local_rows),
            );
        }
        Ok(())
    }

    pub(super) fn add_phi3_gate_up_split(&mut self, raw: &RawTensor) -> Result<(), CompileError> {
        if raw.shape.len() != 2 || raw.shape[0] % 2 != 0 {
            return Err(CompileError::InvalidInput(format!(
                "Phi-3 fused gate/up '{}' has unsupported shape {:?}",
                raw.name, raw.shape
            )));
        }
        let half_rows = raw.shape[0] / 2;
        let cols = raw.shape[1];
        let base = raw.name.trim_end_matches(".mlp.gate_up_proj.weight");
        let specs = [("gate_proj", 0_i64), ("up_proj", half_rows)];
        for (proj, start) in specs {
            let (local_start, local_rows) = local_range(
                half_rows,
                self.target,
                &format!("half the row count of '{}'", raw.name),
            )?;
            self.push_expr(
                format!("{base}.mlp.{proj}.weight"),
                raw,
                vec![local_rows, cols],
                Expr::src(raw.name.clone()).slice(0, start + local_start, local_rows),
            );
        }
        Ok(())
    }
}
