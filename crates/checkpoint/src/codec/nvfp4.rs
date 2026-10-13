//! NVFP4 as modelopt writes it: e2m1 codes two to a byte (low nibble first),
//! one e4m3 scale per 16 codes, and one f32 tensor scale (`weight_scale_2`)
//! under every e4m3 scale of a row. A weight is
//! `e2m1(code) · e4m3(scale) · tensor_scale`.

use super::fp8::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};
use super::invalid;
use super::mxfp4::encode_fp4_e2m1;
use crate::error::Error;

pub const GROUP: usize = 16;

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// The largest magnitude a code holds.
const E2M1_MAX: f32 = 6.0;

/// The largest magnitude an e4m3 scale holds.
const E4M3_MAX: f32 = 448.0;

/// `rows` rows of `k` weights, decoded: `codes` is `rows · k / 2` bytes,
/// `scales` `rows · k / 16` e4m3 bytes, `tensor` one f32 a row.
pub fn decode(
    codes: &[u8],
    scales: &[u8],
    tensor: &[f32],
    rows: usize,
    k: usize,
) -> Result<Vec<f32>, Error> {
    if k == 0 || !k.is_multiple_of(GROUP) {
        return Err(invalid(format!(
            "an NVFP4 row is whole {GROUP}-code groups, and this one is {k} wide"
        )));
    }
    let groups = k / GROUP;
    if codes.len() != rows * k / 2 || scales.len() != rows * groups || tensor.len() != rows {
        return Err(invalid(format!(
            "{rows} NVFP4 rows of {k} are {} code bytes, {} scale bytes and {rows} tensor \
             scales, and these are {}, {} and {}",
            rows * k / 2,
            rows * groups,
            codes.len(),
            scales.len(),
            tensor.len()
        )));
    }
    let mut out = Vec::with_capacity(rows * k);
    for row in 0..rows {
        for g in 0..groups {
            let gain = fp8_e4m3_to_f32(scales[row * groups + g]) * tensor[row];
            let at = (row * k + g * GROUP) / 2;
            for byte in &codes[at..at + GROUP / 2] {
                out.push(E2M1[usize::from(byte & 0xF)] * gain);
                out.push(E2M1[usize::from(byte >> 4)] * gain);
            }
        }
    }
    Ok(out)
}

/// An NVFP4 weight's three planes.
pub struct Planes {
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    pub tensor: Vec<f32>,
}

/// The planes `decode` reads back, quantized the way modelopt quantizes a
/// weight: the tensor scale maps the row's largest magnitude to the largest
/// e4m3 times the largest e2m1, each group's scale its own largest magnitude
/// to the largest e2m1, and each code rounds to the nearest e2m1.
pub fn encode(values: &[f32], rows: usize, k: usize) -> Result<Planes, Error> {
    if k == 0 || !k.is_multiple_of(GROUP) || values.len() != rows * k {
        return Err(invalid(format!(
            "{} values are not {rows} NVFP4 rows of whole {GROUP}-code groups, {k} wide",
            values.len()
        )));
    }
    let groups = k / GROUP;
    let mut codes = vec![0u8; rows * k / 2];
    let mut scales = vec![0u8; rows * groups];
    let mut tensor = vec![0f32; rows];
    for row in 0..rows {
        let line = &values[row * k..(row + 1) * k];
        let amax = line.iter().fold(0f32, |m, v| m.max(v.abs()));
        let global = if amax > 0.0 {
            amax / (E2M1_MAX * E4M3_MAX)
        } else {
            1.0
        };
        tensor[row] = global;
        for g in 0..groups {
            let group = &line[g * GROUP..(g + 1) * GROUP];
            let gmax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
            let byte = f32_to_fp8_e4m3(gmax / E2M1_MAX / global);
            scales[row * groups + g] = byte;
            let gain = fp8_e4m3_to_f32(byte) * global;
            let at = (row * k + g * GROUP) / 2;
            for (pair, out) in group
                .as_chunks::<2>()
                .0
                .iter()
                .zip(&mut codes[at..at + GROUP / 2])
            {
                let code = |v: f32| {
                    if gain > 0.0 {
                        encode_fp4_e2m1(v / gain)
                    } else {
                        0
                    }
                };
                *out = code(pair[0]) | (code(pair[1]) << 4);
            }
        }
    }
    Ok(Planes {
        codes,
        scales,
        tensor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvfp4_every_case() {
        a_code_is_its_e2m1_under_both_scales();
        the_low_nibble_is_the_first_code();
        an_encode_decodes_within_half_a_step();
        a_short_plane_is_refused();
    }

    fn a_code_is_its_e2m1_under_both_scales() {
        let codes = vec![0x71u8; 8];
        let scales = [0x38u8];
        let out = decode(&codes, &scales, &[0.25], 1, 16).expect("one row decodes");
        assert_eq!(out[0], 0.5 * 1.0 * 0.25);
        assert_eq!(out[1], 6.0 * 1.0 * 0.25);
    }

    fn the_low_nibble_is_the_first_code() {
        let mut codes = vec![0u8; 8];
        codes[0] = 0x0F;
        let out = decode(&codes, &[0x40], &[1.0], 1, 16).expect("one row decodes");
        assert_eq!(out[0], -6.0 * 2.0);
        assert_eq!(out[1], 0.0);
    }

    fn an_encode_decodes_within_half_a_step() {
        let (rows, k) = (3, 64);
        let values: Vec<f32> = (0..rows * k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013 * (1 + i / k) as f32)
            .collect();
        let Planes {
            codes,
            scales,
            tensor,
        } = encode(&values, rows, k).expect("encodes");
        let back = decode(&codes, &scales, &tensor, rows, k).expect("decodes");
        for row in 0..rows {
            for g in 0..k / GROUP {
                let gain = fp8_e4m3_to_f32(scales[row * (k / GROUP) + g]) * tensor[row];
                for i in 0..GROUP {
                    let at = row * k + g * GROUP + i;
                    let err = (back[at] - values[at]).abs();
                    assert!(
                        err <= gain + 1e-6,
                        "row {row} group {g}: {} decodes to {}, more than a step ({gain}) away",
                        values[at],
                        back[at]
                    );
                }
            }
        }
    }

    fn a_short_plane_is_refused() {
        assert!(decode(&[0u8; 7], &[0], &[1.0], 1, 16).is_err());
        assert!(decode(&[0u8; 8], &[0], &[1.0], 1, 15).is_err());
        assert!(encode(&[0.0; 15], 1, 15).is_err());
    }
}
