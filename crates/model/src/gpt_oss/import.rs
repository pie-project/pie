use model_dsl::{Dtype, Weight};
use model_loader::contract::{Expr, ModelContract, Scales, TensorContract, TensorType};
use model_loader::types::{Axis, Encoding, QuantGranularity, QuantSpec, ScaleForm};

use super::model::Model;
use crate::contract::{self, ModelError};
use crate::encoding;

const FUSED: i64 = 2;

const ROW_AXIS: u8 = 1;

const GROUP: u32 = 32;

const ALIGNMENT: u32 = 256;

const EMBED: &str = "model.embed_tokens.weight";

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        match src.get(EMBED) {
            Some(_) => self.import_from_huggingface(src),
            None => Err(ModelError::Missing(EMBED.to_string())),
        }
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, ModelError> {
        let tp = self.tp;
        let mut tensors = vec![
            contract::copy(src, &self.embed, tp, EMBED)?,
            contract::copy(src, &self.final_norm, tp, "model.norm.weight")?,
            contract::copy(src, &self.head, tp, "lm_head.weight")?,
        ];
        for (l, layer) in self.layers.iter().enumerate() {
            let ck = |what: &str| format!("model.layers.{l}.{what}");
            let attn = &layer.attn;
            let mlp = &layer.mlp;

            tensors.push(contract::copy(
                src,
                &layer.attn_norm,
                tp,
                ck("input_layernorm.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &layer.mlp_norm,
                tp,
                ck("post_attention_layernorm.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.q_proj,
                tp,
                ck("self_attn.q_proj.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.q_bias,
                tp,
                ck("self_attn.q_proj.bias"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.k_proj,
                tp,
                ck("self_attn.k_proj.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.k_bias,
                tp,
                ck("self_attn.k_proj.bias"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.v_proj,
                tp,
                ck("self_attn.v_proj.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.v_bias,
                tp,
                ck("self_attn.v_proj.bias"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.o_proj,
                tp,
                ck("self_attn.o_proj.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.o_bias,
                tp,
                ck("self_attn.o_proj.bias"),
            )?);
            tensors.push(contract::copy(src, &attn.sinks, tp, ck("self_attn.sinks"))?);
            tensors.push(contract::copy(
                src,
                &mlp.router,
                tp,
                ck("mlp.router.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &mlp.router_bias,
                tp,
                ck("mlp.router.bias"),
            )?);

            let rows = i64::from(mlp.inter);
            tensors.extend(banked(
                &mlp.gate_up,
                ck("mlp.experts.gate_up_proj_blocks"),
                ck("mlp.experts.gate_up_proj_scales"),
                Some(rows),
            ));
            tensors.push(contract::declare(
                src,
                &mlp.gate_up_bias,
                tp,
                deinterleaved(Expr::src(ck("mlp.experts.gate_up_proj_bias")), rows),
            )?);

            tensors.extend(banked(
                &mlp.down,
                ck("mlp.experts.down_proj_blocks"),
                ck("mlp.experts.down_proj_scales"),
                None,
            ));
            tensors.push(contract::copy(
                src,
                &mlp.down_bias,
                tp,
                ck("mlp.experts.down_proj_bias"),
            )?);
        }
        Ok(ModelContract {
            alignment: ALIGNMENT,
            tensors,

            groups: Vec::new(),
        })
    }
}

fn banked(
    w: &Weight,
    blocks: String,
    scales: String,
    interleaved_by: Option<i64>,
) -> Vec<TensorContract> {
    let laid = |expr: Expr| match interleaved_by {
        Some(rows) => deinterleaved(expr, rows),
        None => expr,
    };
    let codes = bank_codes(w);
    let scaled = bank_scales(w);
    vec![
        TensorContract::inferred(
            w.name.clone(),
            laid(Expr::src(blocks).transmute(codes.clone())),
            codes.encoding,
        ),
        TensorContract::new(
            format!("{}.scales", w.name),
            laid(Expr::src(scales).transmute(scaled.clone())),
            scaled.shape,
            scaled.encoding,
        )
        .scaling(scaling(w)),
    ]
}

fn deinterleaved(src: Expr, rows: i64) -> Expr {
    Expr::concat(
        ROW_AXIS,
        vec![
            src.clone().stride(ROW_AXIS, 0, rows, FUSED),
            src.stride(ROW_AXIS, 1, rows, FUSED),
        ],
    )
}

fn bank_codes(w: &Weight) -> TensorType {
    TensorType::new(extents(w), grouped_along(w, block_axis(w)))
}

fn bank_scales(w: &Weight) -> TensorType {
    let mut shape = extents(w);
    let last = shape
        .len()
        .checked_sub(1)
        .unwrap_or_else(|| panic!("`{}` is a bank and has no contracted axis", w.name));
    let k = shape[last];
    let group = i64::from(GROUP);
    assert!(
        k % group == 0,
        "`{}` contracts over {k}, which is not a whole number of {GROUP}-code blocks",
        w.name,
    );
    shape[last] = k / group;
    TensorType::new(shape, encoding(Dtype::E8m0))
}

fn scaling(w: &Weight) -> Scales {
    Scales {
        of: w.name.clone(),
        granularity: QuantGranularity::PerGroup,
        group_size: GROUP,
        channel_axis: u32::from(block_axis(w)),
        form: ScaleForm::RawE8M0,
    }
}

fn block_axis(w: &Weight) -> u8 {
    let last = w
        .shape
        .len()
        .checked_sub(1)
        .unwrap_or_else(|| panic!("`{}` is a bank and has no contracted axis", w.name));
    u8::try_from(last).expect("an axis inside a shape")
}

fn grouped_along(w: &Weight, axis: u8) -> Encoding {
    match encoding(w.dtype) {
        Encoding::Quant(spec) => Encoding::Quant(QuantSpec {
            channel_axis: Some(Axis(axis)),
            ..spec
        }),
        Encoding::Raw(dtype) => panic!(
            "`{}` is {dtype:?}, which groups nothing; only a quantized bank \
             has a blocked axis",
            w.name
        ),
    }
}

fn extents(w: &Weight) -> Vec<i64> {
    w.shape
        .iter()
        .map(|&extent| i64::try_from(extent).expect("an extent no i64 holds"))
        .collect()
}
