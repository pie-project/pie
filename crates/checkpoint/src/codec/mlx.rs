pub fn mlx_affine_group_params_bits(values: &[f64], bits: u32) -> (f32, f32) {
    #[allow(clippy::cast_precision_loss)]
    let n_bins = ((1u32 << bits) - 1) as f32;
    const EPS: f32 = 1e-7;
    let mut w_min = f32::INFINITY;
    let mut w_max = 0.0f32;
    for &value in values {
        let value = value as f32;
        w_min = w_min.min(value);
        w_max = w_max.max(value);
    }
    let mask = w_min.abs() > w_max.abs();
    let mut scale = ((w_max - w_min) / n_bins).max(EPS);
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

pub fn decode_mlx_affine_codes(bytes: &[u8], bits: u32) -> Vec<f64> {
    match bits {
        4 => {
            let mut values = Vec::with_capacity(bytes.len() * 2);
            for byte in bytes {
                values.push(f64::from(byte & 0xF));
                values.push(f64::from(byte >> 4));
            }
            values
        }
        2 => {
            let mut values = Vec::with_capacity(bytes.len() * 4);
            for byte in bytes {
                for shift in [0, 2, 4, 6] {
                    values.push(f64::from((byte >> shift) & 0x3));
                }
            }
            values
        }
        8 => bytes.iter().map(|byte| f64::from(*byte)).collect(),
        other => unreachable!("an MLX affine code is 2, 4 or 8 bits, not {other}"),
    }
}
