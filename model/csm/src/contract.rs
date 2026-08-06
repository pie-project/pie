//! What CSM binds.
//!
//! Ported from `driver/cuda/src/model/csm/csm_contract.hpp`. One family-wide
//! fact: every kernel in the backbone, the depth decoder and the Mimi codec
//! reads bf16, and the published checkpoints ship fp32.

use pie_loader::contract::Expr;
use pie_loader::error::Error;
use pie_loader::types::{DType, Encoding};

use pie_model_common::builder::{Builder, is_raw};

/// CSM: a dense backbone plus a depth decoder and the Mimi codec, all bf16.
pub fn author_csm(b: &mut Builder<'_>) -> Result<(), Error> {
    bf16_weights(b)?;
    // The dense tail, stated rather than bundled: a family's contract is
    // its pass sequence, and hiding three of them behind a helper meant
    // six families' contracts could not be read where they live.
    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

/// Ask for bf16 everywhere, because that is the only thing CSM's kernels
/// read.
///
/// The published checkpoints are fp32 throughout, and every tensor the bind
/// resolves goes through the same accessor — there is no CSM weight that
/// wants to stay fp32. So the narrowing is a property of the family, and the
/// right place to say it is once, here.
///
/// The narrowing is written as a `cast`, which is what it is: the values are
/// preserved and their representation is not, so it costs a kernel. It also
/// means the fp32 bytes are never resident — the bind used to allocate a
/// bf16 copy of every weight while the fp32 original stayed loaded, so peak
/// memory was three times the bf16 model.
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
