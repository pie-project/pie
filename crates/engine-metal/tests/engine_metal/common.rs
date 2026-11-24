//! Helpers several tests share: bf16 conversion, deterministic data and the
//! byte views kernels read.

#![allow(dead_code)]

use eta_ir::container::{PortBinding, PortSource};
use eta_ir::registry::Port;
use eta_ir::types::{Dtype as EtaDtype, Shape};

pub(crate) fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    [(bits >> 16) as u8, (bits >> 24) as u8]
}

pub(crate) fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .map(|f| ((f.to_bits() + 0x7fff + ((f.to_bits() >> 16) & 1)) >> 16) as u16)
        .flat_map(u16::to_le_bytes)
        .collect()
}

pub(crate) fn bf16_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

pub(crate) fn noise(at: u64) -> u8 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 40) as u8
}

pub(crate) fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(((bits + rounding) >> 16) << 16)
}

/// A row's mask words, one u32 a row (its enable flag, its boundary 0).
pub(crate) fn mask_words(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

pub(crate) fn vals(n: usize, s: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = (i as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ s;
            x ^= x >> 33;
            x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
            bf(((x >> 40) as f32 / 16777216. - 0.5) * 4.)
        })
        .collect()
}

pub(crate) fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn f32s(v: &[u8]) -> Vec<f32> {
    v.as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
        .collect()
}

pub(crate) fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

pub(crate) fn bf(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff0000)
}

/// bf16 store/load: bf16 is the top 16 bits of f32 with round-to-nearest-even.
pub(crate) fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

pub(crate) fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Decode an IEEE binary16 bit pattern to f32 (exact — every f16 is an f32).
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if (bits >> 15) & 1 == 1 { -1.0f32 } else { 1.0 };
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    if exp == 0 {
        sign * f32::from(mant) * 2f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + f32::from(mant) / 1024.0) * 2f32.powi(i32::from(exp) - 15)
    }
}

pub(crate) fn const_port(port: Port, words: &[u32]) -> PortBinding {
    PortBinding {
        port,
        source: PortSource::Const {
            dtype: EtaDtype::U32,
            shape: Shape::vector(words.len() as u32),
            data: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        },
    }
}
