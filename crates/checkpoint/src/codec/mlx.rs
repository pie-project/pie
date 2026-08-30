//! MLX affine quantization: a scale and a bias per group. Unlike the MX
//! schemes this one stores an offset, so a group's minimum is
//! representable exactly and the zero point need not be a codepoint.

/// One group's affine scale and zero point, by MLX's rule (transcribed
/// from `mlx/backend/cpu/quantized.cpp::quantize`, so a checkpoint this
/// produces is bit-compatible with one `mlx_lm convert` produced). Five
/// details, since none is what an independently written affine quantizer
/// would do:
///
///  * **The scale is usually negative** -- negated unless the group's
///    minimum is larger in magnitude, putting code 0 on whichever end
///    dominates. A producer that "fixed" the sign would misplace the codes.
///  * **The endpoint is snapped, not the scale**: recomputed as
///    `edge / round(edge / scale)`, keeping the largest magnitude exact.
///  * **`w_max` starts at zero**, not negative infinity, so an all-negative
///    group quantizes over the range up to zero, not up to its own max.
///  * **The rounding is half AWAY FROM ZERO**, not half to even -- the
///    source of an 8.2% disagreement with `mx.quantize` on an
///    MXFP4-derived expert bank, whose values sit on half-integers.
///  * **`eps` floors the scale**, so a constant group (all-zero bias row) divides by `1e-7` instead of by zero.
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
