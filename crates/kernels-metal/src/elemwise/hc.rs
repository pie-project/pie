//! `Hc`: hyper-connections — residual streams expanded, mixed by learned
//! gates, and folded back layer by layer. The metal plane never claimed any
//! of these points (the old file held an empty claims impl beside the
//! norms), so every entry is a typed refusal and the driver arm stays dumb.

use kernels::KernelError;

use crate::encode::Ctx;
use crate::tensor::Tensor;

pub fn expand(_ctx: &Ctx<'_>, _x: Tensor, _streams: u32, _y: Tensor) -> Result<(), KernelError> {
    Err(KernelError::Unsupported { op: "hc.expand" })
}

pub fn rmsnorm_f32(
    _ctx: &Ctx<'_>,
    _streams: Tensor,
    _eps: f32,
    _y: Tensor,
) -> Result<(), KernelError> {
    Err(KernelError::Unsupported {
        op: "hc.rmsnorm_f32",
    })
}

#[allow(clippy::too_many_arguments)]
pub fn gates(
    _ctx: &Ctx<'_>,
    _normed: Tensor,
    _streams: Tensor,
    _scale: Tensor,
    _base: Tensor,
    _stream_count: u32,
    _gate_eps: f32,
    _alpha: f32,
    _sinkhorn: u32,
    _x: Tensor,
    _post_mix: Tensor,
    _comb_mix: Tensor,
) -> Result<(), KernelError> {
    Err(KernelError::Unsupported { op: "hc.gates" })
}

pub fn fold(
    _ctx: &Ctx<'_>,
    _x: Tensor,
    _streams: Tensor,
    _post_mix: Tensor,
    _comb_mix: Tensor,
    _y: Tensor,
) -> Result<(), KernelError> {
    Err(KernelError::Unsupported { op: "hc.fold" })
}

// `collapse` went with `Hc::Collapse`: no plane could fire it honestly (review R5).
