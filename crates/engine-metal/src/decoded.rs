//! Banks decoded to bf16 at load, for the ops that read a weight as one
//! dense plane and have no quantized arm: the MLA absorbs' `kv_b`. The CUDA
//! shell decodes the same bank into fire scratch on every launch
//! (`dense_or_decoded`); this shell does it once, since its memory is the
//! host's and the decode is a few hundred MiB of arithmetic at boot.
//!
//! The codec is MLX's affine one, as `mlx_quantized_block.metal` reads it:
//! `w = code * scale + bias`, codes packed least-significant-first inside a
//! byte, one scale and one bias per `group` codes along the row.

use std::collections::BTreeSet;

use model_ir::{Attention, Def, Operation, Trace};

/// The weight rows the trace's MLA absorbs read as dense `kv_b` planes.
pub(crate) fn absorbed_weights(trace: &Trace) -> BTreeSet<usize> {
    let mut rows = BTreeSet::new();
    for node in &trace.nodes {
        let kv_b = match &node.op {
            Operation::Attention(Attention::MlaAbsorbQ { kv_b, .. })
            | Operation::Attention(Attention::MlaAbsorbOut { kv_b, .. }) => *kv_b,
            _ => continue,
        };
        if let Some(Def::Weight(w)) = trace.values.get(kv_b.0 as usize).map(|v| &v.def) {
            rows.insert(*w as usize);
        }
    }
    rows
}

/// A bf16 bit pattern's value.
pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// The nearest bf16 (round to nearest even, as the device converts).
pub(crate) fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(round)) >> 16) as u16
}

/// One `[n, k]` affine bank decoded row-major to bf16 bytes. `codes` is
/// `n * k * bits / 8` bytes, `scales` and `biases` (`None` for a symmetric
/// scheme) are `n * k / group` bf16 each.
///
/// # Errors
///
/// A refused geometry: bits outside {2, 4, 8}, a row not a whole number of
/// groups, or planes shorter than the geometry says.
pub(crate) fn decode_affine(
    codes: &[u8],
    scales: &[u8],
    biases: Option<&[u8]>,
    n: usize,
    k: usize,
    group: usize,
    bits: usize,
) -> Result<Vec<u8>, String> {
    if !matches!(bits, 2 | 4 | 8) {
        return Err(format!("{bits}-bit codes: this decoder unpacks 2, 4 and 8"));
    }
    if group == 0 || k % group != 0 {
        return Err(format!("a {k}-wide row is not a whole number of {group}-code groups"));
    }
    let per_byte = 8 / bits;
    if k % per_byte != 0 {
        return Err(format!("a {k}-wide row does not pack into whole bytes at {bits} bits"));
    }
    let row_bytes = k / per_byte;
    let groups = k / group;
    if codes.len() < n * row_bytes {
        return Err(format!("codes plane holds {} bytes, {} needed", codes.len(), n * row_bytes));
    }
    if scales.len() < n * groups * 2 || biases.is_some_and(|b| b.len() < n * groups * 2) {
        return Err(format!("scale/bias planes shorter than {n} x {groups} bf16"));
    }
    let mask = ((1u16 << bits) - 1) as u8;
    let mut out = vec![0u8; n * k * 2];
    for row in 0..n {
        let codes = &codes[row * row_bytes..(row + 1) * row_bytes];
        for g in 0..groups {
            let at = (row * groups + g) * 2;
            let scale = bf16_to_f32(u16::from_le_bytes([scales[at], scales[at + 1]]));
            let bias = biases.map_or(0.0, |b| bf16_to_f32(u16::from_le_bytes([b[at], b[at + 1]])));
            for i in 0..group {
                let col = g * group + i;
                let byte = codes[col / per_byte];
                let code = (byte >> ((col % per_byte) * bits)) & mask;
                let value = f32::from(code) * scale + bias;
                let o = (row * k + col) * 2;
                out[o..o + 2].copy_from_slice(&f32_to_bf16(value).to_le_bytes());
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf(v: f32) -> [u8; 2] {
        f32_to_bf16(v).to_le_bytes()
    }

    #[test]
    fn eight_bit_codes_are_bytes() {
        let codes = [1u8, 2, 3, 4];
        let scales = bf(0.5);
        let biases = bf(-1.0);
        let out = decode_affine(&codes, &scales, Some(&biases), 1, 4, 4, 8).unwrap();
        let got: Vec<f32> = out
            .chunks(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        assert_eq!(got, vec![-0.5, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn sub_byte_codes_unpack_least_significant_first() {
        // 4-bit: 0xBA = low nibble 0xA first, then 0xB. 2-bit: 0b11100100 = 0,1,2,3.
        let out = decode_affine(&[0xBA], &bf(1.0), None, 1, 2, 2, 4).unwrap();
        let got: Vec<f32> = out
            .chunks(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        assert_eq!(got, vec![10.0, 11.0]);
        let out = decode_affine(&[0b1110_0100], &bf(2.0), Some(&bf(1.0)), 1, 4, 4, 2).unwrap();
        let got: Vec<f32> = out
            .chunks(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        assert_eq!(got, vec![1.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn bf16_round_trips() {
        for v in [0.0f32, 1.0, -2.5, 3.0e-3, 1.0e30] {
            assert_eq!(bf16_to_f32(f32_to_bf16(v)), bf16_to_f32(f32_to_bf16(v)));
            assert!((bf16_to_f32(f32_to_bf16(v)) - v).abs() <= v.abs() / 128.0);
        }
    }
}
