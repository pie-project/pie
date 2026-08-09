//! MLX affine quantization: a scale and a bias per group.
//!
//! Unlike the MX schemes this one stores an offset, so a group's minimum is
//! representable exactly and the zero point need not be a codepoint.

/// One group's affine scale and zero point, by MLX's rule.
///
/// Transcribed from `mlx/backend/cpu/quantized.cpp::quantize`, because the
/// scheme is MLX's and a checkpoint this produces is meant to be
/// interchangeable with one `mlx_lm convert` produced. Three details are worth
/// naming, since none is what an independently written affine quantizer would
/// do:
///
///  * **The scale is usually negative.** It is negated unless the group's
///    minimum is the larger in magnitude, which puts code 0 on whichever end
///    dominates. Dequantization is `code * scale + zero` either way, so a
///    consumer never sees the difference -- but a producer that "fixed" the sign
///    would place the codes on the other end of the group's range.
///  * **The endpoint is snapped, not the scale.** `scale` is recomputed as
///    `edge / round(edge / scale)` so that the dominant endpoint lands exactly on
///    a code, which is what keeps the largest magnitude in the group exact.
///  * **`w_max` starts at zero**, not at negative infinity, so a group whose
///    values are all negative is quantized over the range up to zero rather
///    than up to its own largest element. Nothing about the arithmetic suggests
///    this; it is simply what MLX does.
///  * **The rounding is half AWAY FROM ZERO**, not half to even. That one
///    choice was the whole of an 8.2% disagreement with `mx.quantize` on an
///    MXFP4-derived expert bank, whose values sit on half-integers by
///    construction, and it is why every mismatch was by exactly one.
///  * **`eps` floors the scale** so a constant group -- every expert bias row
///    that is all zeros, for one -- divides by `1e-7` instead of by zero.
pub fn mlx_affine_group_params(values: &[f64]) -> (f32, f32) {
    const N_BINS: f32 = 15.0;
    const EPS: f32 = 1e-7;
    let mut w_min = f32::INFINITY;
    let mut w_max = 0.0f32;
    for &value in values {
        let value = value as f32;
        w_min = w_min.min(value);
        w_max = w_max.max(value);
    }
    let mask = w_min.abs() > w_max.abs();
    let mut scale = ((w_max - w_min) / N_BINS).max(EPS);
    if !mask {
        scale = -scale;
    }
    let edge = if mask { w_min } else { w_max };
    let q0 = (edge / scale).round();
    let mut bias = 0.0f32;
    if q0 != 0.0 {
        scale = edge / q0;
        bias = edge;
    }
    (scale, bias)
}
