use model_dsl::{Dtype, Weight};
use model_loader::contract::{Expr, ModelContract, TensorContract, TensorType};
use model_loader::types::{DType, Encoding};

use super::model::Model;
use crate::contract::{self, ModelError};
use crate::encoding;

const BANK_ROWS: u8 = 1;

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        assert!(
            self.tp == 1,
            "an import states the whole checkpoint; build the model at tp = 1"
        );
        let mut tensors = vec![
            contract::copy(src, &self.embed, "model.embed_tokens.weight")?,
            contract::copy(src, &self.final_norm, "model.norm.weight")?,
            contract::copy(src, &self.head, "lm_head.weight")?,
        ];
        for (l, layer) in self.layers.iter().enumerate() {
            let ck = |what: &str| format!("model.layers.{l}.{what}");
            let attn = &layer.attn;
            let mlp = &layer.mlp;

            tensors.push(contract::copy(
                src,
                &layer.attn_norm,
                ck("input_layernorm.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &layer.mlp_norm,
                ck("post_attention_layernorm.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.q_proj,
                ck("self_attn.q_proj.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.q_bias,
                ck("self_attn.q_proj.bias"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.k_proj,
                ck("self_attn.k_proj.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.k_bias,
                ck("self_attn.k_proj.bias"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.v_proj,
                ck("self_attn.v_proj.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.v_bias,
                ck("self_attn.v_proj.bias"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.o_proj,
                ck("self_attn.o_proj.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.o_bias,
                ck("self_attn.o_proj.bias"),
            )?);
            tensors.push(contract::copy(src, &attn.sinks, ck("self_attn.sinks"))?);
            tensors.push(contract::copy(src, &mlp.router, ck("mlp.router.weight"))?);
            tensors.push(contract::copy(
                src,
                &mlp.router_bias,
                ck("mlp.router.bias"),
            )?);

            let rows = i64::from(mlp.inter);
            tensors.extend(banked_interleaved(
                src,
                &mlp.gate_up,
                ck("mlp.experts.gate_up_proj_blocks"),
                ck("mlp.experts.gate_up_proj_scales"),
                rows,
            )?);
            tensors.push(contract::declare(
                src,
                &mlp.gate_up_bias,
                deinterleaved(Expr::src(ck("mlp.experts.gate_up_proj_bias")), rows),
            )?);

            tensors.extend(banked(
                src,
                &mlp.down,
                ck("mlp.experts.down_proj_blocks"),
                ck("mlp.experts.down_proj_scales"),
            )?);
            tensors.push(contract::copy(
                src,
                &mlp.down_bias,
                ck("mlp.experts.down_proj_bias"),
            )?);
        }
        Ok(ModelContract {
            alignment: contract::ALIGNMENT,
            tensors,

            groups: Vec::new(),
        })
    }
}

fn banked(
    src: &ztensor::Source,
    w: &Weight,
    blocks: String,
    scales: String,
) -> Result<Vec<TensorContract>, ModelError> {
    bank_planes(src, w, blocks, scales, |expr| expr)
}

fn banked_interleaved(
    src: &ztensor::Source,
    w: &Weight,
    blocks: String,
    scales: String,
    rows: i64,
) -> Result<Vec<TensorContract>, ModelError> {
    bank_planes(src, w, blocks, scales, |expr| deinterleaved(expr, rows))
}

fn bank_planes(
    src: &ztensor::Source,
    w: &Weight,
    blocks: String,
    scales: String,
    lay: impl Fn(Expr) -> Expr,
) -> Result<Vec<TensorContract>, ModelError> {
    for plane in [blocks.as_str(), scales.as_str()] {
        let stored = contract::stored_encoding(src, plane)?;
        if stored != Encoding::Raw(DType::U8) {
            return Err(ModelError::Illegible {
                name: w.name.clone(),
                detail: format!(
                    "`{plane}` is stored {stored:?}, and a bank plane is read \
                     as raw u8 bytes"
                ),
            });
        }
    }
    let codes = bank_codes(w);
    let scaled = bank_scales(w);
    Ok(vec![
        TensorContract::inferred(
            w.name.clone(),
            lay(Expr::src(blocks).transmute(codes.clone())),
            codes.encoding,
        ),
        TensorContract::new(
            model_dsl::scales_name(&w.name),
            lay(Expr::src(scales).transmute(scaled.clone())),
            scaled.shape,
            scaled.encoding,
        )
        .scaling(contract::scaling(w)),
    ])
}

fn deinterleaved(src: Expr, rows: i64) -> Expr {
    Expr::concat(
        BANK_ROWS,
        vec![
            src.clone().stride(BANK_ROWS, 0, rows, 2),
            src.stride(BANK_ROWS, 1, rows, 2),
        ],
    )
}

fn bank_codes(w: &Weight) -> TensorType {
    TensorType::new(contract::extents(w), contract::grouped(w))
}

fn bank_scales(w: &Weight) -> TensorType {
    let shape = contract::extents(w);
    let axis = u32::from(contract::channel_axis(w));
    TensorType::new(
        contract::divided(&shape, axis, &w.name),
        encoding(Dtype::E8m0),
    )
}
