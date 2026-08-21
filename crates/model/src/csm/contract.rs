use model_loader::contract::Expr;
use model_loader::error::Error;
use model_loader::types::{DType, Encoding};

use crate::shared::builder::{Builder, is_raw};

pub fn author_csm(b: &mut Builder<'_>) -> Result<(), Error> {
    bf16_weights(b)?;

    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

fn bf16_weights(b: &mut Builder<'_>) -> Result<(), Error> {
    for raw in b.tensors().to_vec() {
        if !is_raw(&raw.encoding, DType::F32) {
            continue;
        }
        let axis = b.shard_axis(&raw.name)?;
        let (expr, local) = b.shard(Expr::src(&raw.name), raw.shape.clone(), axis);
        let bf16 = Encoding::Raw(DType::BF16);
        b.define(
            b.output_name(&raw.name),
            expr.cast(bf16.clone()),
            bf16,
            Some(local),
        );
        b.consume(raw.id);
    }
    Ok(())
}
