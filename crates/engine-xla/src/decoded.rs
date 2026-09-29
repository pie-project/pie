use std::collections::BTreeSet;

use model_ir::{Attention, Def, Operation, Trace};

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

pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub(crate) fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(round)) >> 16) as u16
}

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
    if group == 0 || !k.is_multiple_of(group) {
        return Err(format!(
            "a {k}-wide row is not a whole number of {group}-code groups"
        ));
    }
    let per_byte = 8 / bits;
    if !k.is_multiple_of(per_byte) {
        return Err(format!(
            "a {k}-wide row does not pack into whole bytes at {bits} bits"
        ));
    }
    let row_bytes = k / per_byte;
    let groups = k / group;
    if codes.len() < n * row_bytes {
        return Err(format!(
            "codes plane holds {} bytes, {} needed",
            codes.len(),
            n * row_bytes
        ));
    }
    if scales.len() < n * groups * 2 || biases.is_some_and(|b| b.len() < n * groups * 2) {
        return Err(format!(
            "scale/bias planes shorter than {n} x {groups} bf16"
        ));
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
