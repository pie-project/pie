//! GPT-OSS MXFP4 expert groups (routed decode and native GEMM).

use super::*;

impl DefaultAbiBuilder<'_> {
    pub(super) fn add_gpt_oss_mxfp4_groups(&mut self) -> Result<(), CompileError> {
        if !self.profile().gpt_oss_mxfp4_groups {
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

    pub(super) fn add_gpt_oss_native_mxfp4_group(
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

    pub(super) fn add_gpt_oss_native_gate_up(
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
        let (local_start, local_intermediate) = local_range(
            full_intermediate,
            self.target,
            &format!("the intermediate size of '{base}'"),
        )?;
        let intermediate_native = align_up_i64(local_intermediate, 128)?;
        let prefix = base.trim_end_matches("gate_up_proj");
        for (name, row_map) in [("gate_proj", RowMap::Even), ("up_proj", RowMap::Odd)] {
            let out_base = format!("{prefix}{name}");
            self.push_repack(
                format!("{out_base}.weight"),
                block,
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

    pub(super) fn add_gpt_oss_native_down(
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
        let (local_start, local_intermediate) = local_range(
            full_intermediate,
            self.target,
            &format!("the intermediate size of '{base}'"),
        )?;
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
}
