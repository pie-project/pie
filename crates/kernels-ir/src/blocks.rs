//! Block-scaled cache rows: a row's leading columns coded a block at a time
//! under one scale each, its trailing columns kept in bf16, as DeepSeek's
//! caches hold their KV ([`BlockRows::V4`], [`BlockRows::MX8`],
//! [`BlockRows::NV4`]). The rule is exact: [`BlockRows::encode`] lands the
//! bytes a backend's store must, and [`BlockRows::decode`] reads them back.
//!
//! A row's values are first rounded to bf16 (the row a result's own plane
//! would hold). Each block's scale is taken of its largest magnitude
//! `amax`, floored: `q = max(amax, floor) / max`, `max` the code's largest
//! value, and either
//! - [`BlockScale::Pow2`]: `s = 2^ceil(log2 q)`, stored as its biased
//!   exponent (ue8m0), or
//! - [`BlockScale::E4m3`]: `s = e4m3(q)`, stored as its e4m3 bits;
//!
//! each code is then `code(x / s)` (f32 division, nearest, ties to even,
//! saturating at the code's `max`, the sign kept on a zero).
//!
//! A row's data bytes are its codes (e4m3 one a byte; e2m1 two a byte, the
//! lower column in the low nibble), then its tail's bf16 (little-endian);
//! its scale bytes are its blocks' scales, then zeros. A page of `p` rows
//! holds `p` data rows, then `p` scale rows ([`BlockRows::at`]).

use crate::skeleton::{round_bf16, round_e4m3};

/// The element a block's codes are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Code {
    /// e4m3: one a byte, at most 448.
    E4m3,
    /// e2m1: two a byte (the lower column in the low nibble), at most 6.
    E2m1,
}

impl Code {
    /// The largest value a code holds.
    #[must_use]
    pub const fn max(self) -> f32 {
        match self {
            Code::E4m3 => 448.0,
            Code::E2m1 => 6.0,
        }
    }

    /// The bits a code takes.
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Code::E4m3 => 8,
            Code::E2m1 => 4,
        }
    }

    /// `x`'s code: the nearest (ties to even), saturating, the sign kept.
    #[must_use]
    pub fn encode(self, x: f32) -> u8 {
        match self {
            Code::E4m3 => e4m3_bits(x),
            Code::E2m1 => e2m1_bits(x),
        }
    }

    /// A code's value.
    #[must_use]
    pub fn value(self, code: u8) -> f32 {
        match self {
            Code::E4m3 => e4m3_value(code),
            Code::E2m1 => e2m1_value(code),
        }
    }
}

/// How a block's scale is taken of its floored largest magnitude.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockScale {
    /// The power of two at or above `q`, stored as its biased exponent.
    Pow2,
    /// `q` rounded to e4m3, stored as its bits.
    E4m3,
}

/// A cache row's block-scaled layout (see the module's rule).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockRows {
    /// The element of the coded columns.
    pub code: Code,
    /// The coded columns a scale.
    pub block: u32,
    /// How a block's scale is taken and stored.
    pub scale: BlockScale,
    /// The least `amax` a scale is taken of.
    pub floor: f32,
    /// The leading columns coded.
    pub coded: u32,
    /// The trailing columns kept in bf16.
    pub tail: u32,
    /// A row's scale bytes: its blocks' scales, then zeros.
    pub scale_bytes: u32,
}

impl BlockRows {
    /// DeepSeek-V4's rows (FlashMLA's `fp8_ds_mla`): 448 e4m3 codes under
    /// one power-of-two scale a 64, then 64 bf16 columns; 8 scale bytes.
    pub const V4: BlockRows = BlockRows {
        code: Code::E4m3,
        block: 64,
        scale: BlockScale::Pow2,
        floor: 1e-4,
        coded: 448,
        tail: 64,
        scale_bytes: 8,
    };

    /// DeepSeek-V4.1's window rows (MXFP8): 512 e4m3 codes under one
    /// power-of-two scale a 32.
    pub const MX8: BlockRows = BlockRows {
        code: Code::E4m3,
        block: 32,
        scale: BlockScale::Pow2,
        floor: 1e-4,
        coded: 512,
        tail: 0,
        scale_bytes: 16,
    };

    /// DeepSeek-V4.1's compressed rows (NVFP4): 512 e2m1 codes under one
    /// e4m3 scale a 16, the scale floored at the least e4m3 (`6 · 2^-9`
    /// over 6).
    pub const NV4: BlockRows = BlockRows {
        code: Code::E2m1,
        block: 16,
        scale: BlockScale::E4m3,
        floor: 6.0 * 0.001_953_125,
        coded: 512,
        tail: 0,
        scale_bytes: 32,
    };

    /// The columns a row holds.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.coded + self.tail
    }

    /// The scales a row holds.
    #[must_use]
    pub const fn scales(&self) -> u32 {
        self.coded / self.block
    }

    /// A row's data bytes: its codes, then its tail's bf16.
    #[must_use]
    pub const fn data_bytes(&self) -> u32 {
        self.coded * self.code.bits() / 8 + 2 * self.tail
    }

    /// A row's bytes: its data and its scales.
    #[must_use]
    pub const fn row_bytes(&self) -> u32 {
        self.data_bytes() + self.scale_bytes
    }

    /// Whether the layout is one the rule lands: whole blocks of whole
    /// bytes, a scale byte each, and a positive, finite floor.
    #[must_use]
    pub fn verify(&self) -> bool {
        self.block > 0
            && self.coded.is_multiple_of(self.block)
            && (self.block * self.code.bits()).is_multiple_of(8)
            && self.scales() <= self.scale_bytes
            && self.floor.is_finite()
            && self.floor > 0.0
    }

    /// The byte offsets of row `slot` of page `page`'s data and its scales,
    /// a page of `page_size` rows: its data rows, then its scale rows.
    #[must_use]
    pub const fn at(&self, page: u64, slot: u64, page_size: u64) -> (u64, u64) {
        let (data, scales) = (self.data_bytes() as u64, self.scale_bytes as u64);
        let base = page * page_size * (data + scales);
        (base + slot * data, base + page_size * data + slot * scales)
    }

    /// The scale of a block of largest magnitude `amax` and its stored
    /// byte.
    #[must_use]
    pub fn block_scale(&self, amax: f32) -> (f32, u8) {
        let q = amax.max(self.floor) / self.code.max();
        match self.scale {
            BlockScale::Pow2 => {
                let bits = q.to_bits();
                let biased = ((bits >> 23) & 0xff) + u32::from(bits & 0x7f_ffff != 0);
                let biased = biased.min(254);
                (f32::from_bits(biased << 23), biased as u8)
            }
            BlockScale::E4m3 => {
                let byte = e4m3_bits(q);
                (e4m3_value(byte), byte)
            }
        }
    }

    /// `row` (its [`width`](Self::width) values) as the cache stores it:
    /// its data bytes and its scale bytes.
    ///
    /// # Panics
    ///
    /// Where `row` is not a row's width.
    #[must_use]
    pub fn encode(&self, row: &[f32]) -> (Vec<u8>, Vec<u8>) {
        assert_eq!(row.len(), self.width() as usize, "a row's width");
        let row: Vec<f32> = row.iter().map(|x| round_bf16(*x)).collect();
        let (coded, block) = (self.coded as usize, self.block as usize);
        let mut codes = Vec::with_capacity(coded);
        let mut scales = vec![0u8; self.scale_bytes as usize];
        for (b, run) in row[..coded].chunks(block).enumerate() {
            let amax = run.iter().fold(0f32, |m, x| m.max(x.abs()));
            let (s, byte) = self.block_scale(amax);
            scales[b] = byte;
            codes.extend(run.iter().map(|x| self.code.encode(x / s)));
        }
        let mut data = match self.code {
            Code::E4m3 => codes,
            Code::E2m1 => codes
                .chunks(2)
                .map(|pair| (pair[0] & 0xf) | (pair[1] << 4))
                .collect(),
        };
        for x in &row[coded..] {
            let bits = (x.to_bits() >> 16) as u16;
            data.extend(bits.to_le_bytes());
        }
        (data, scales)
    }

    /// The row a cache's data and scale bytes hold, read back.
    ///
    /// # Panics
    ///
    /// Where the bytes are not a row's.
    #[must_use]
    pub fn decode(&self, data: &[u8], scales: &[u8]) -> Vec<f32> {
        assert_eq!(data.len(), self.data_bytes() as usize, "a row's data bytes");
        assert_eq!(
            scales.len(),
            self.scale_bytes as usize,
            "a row's scale bytes"
        );
        let (coded, block) = (self.coded as usize, self.block as usize);
        let code = |c: usize| match self.code {
            Code::E4m3 => data[c],
            Code::E2m1 => (data[c / 2] >> (4 * (c % 2))) & 0xf,
        };
        let scale = |b: usize| match self.scale {
            BlockScale::Pow2 => f32::from_bits(u32::from(scales[b]) << 23),
            BlockScale::E4m3 => e4m3_value(scales[b]),
        };
        let codes = coded * self.code.bits() as usize / 8;
        (0..coded)
            .map(|c| self.code.value(code(c)) * scale(c / block))
            .chain(
                data[codes..]
                    .chunks(2)
                    .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)),
            )
            .collect()
    }

    /// `row` as the cache stores it, read back.
    #[must_use]
    pub fn stored(&self, row: &[f32]) -> Vec<f32> {
        let (data, scales) = self.encode(row);
        self.decode(&data, &scales)
    }
}

/// `x`'s e4m3 bits: the nearest (ties to even), saturating at ±448, the
/// sign kept on a zero, as CUDA's `satfinite` conversion lands it.
#[must_use]
pub fn e4m3_bits(x: f32) -> u8 {
    let sign = u8::from(x.is_sign_negative()) << 7;
    if x.is_nan() {
        return sign | 0x7f;
    }
    let a = round_e4m3(x).abs();
    if a == 0.0 {
        return sign;
    }
    let exponent = ((a.to_bits() >> 23) & 0xff) as i32 - 127;
    if exponent < -6 {
        // a subnormal: its multiple of 2^-9
        return sign | (a / 0.001_953_125) as u8;
    }
    let mantissa = ((a.to_bits() >> 20) & 0x7) as u8;
    sign | (((exponent + 7) as u8) << 3) | mantissa
}

/// An e4m3 code's value.
#[must_use]
pub fn e4m3_value(code: u8) -> f32 {
    let (e, m) = (i32::from((code >> 3) & 0xf), f32::from(code & 0x7));
    let v = if e == 0 {
        m * 0.001_953_125
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if code & 0x80 != 0 { -v } else { v }
}

/// `x`'s e2m1 code: the nearest of 0, 0.5, 1, 1.5, 2, 3, 4 and 6 (ties to
/// the even code), saturating at ±6, the sign kept on a zero.
#[must_use]
pub fn e2m1_bits(x: f32) -> u8 {
    let a = x.abs();
    let m = u8::from(a > 0.25)
        + u8::from(a >= 0.75)
        + u8::from(a > 1.25)
        + u8::from(a >= 1.75)
        + u8::from(a > 2.5)
        + u8::from(a >= 3.5)
        + u8::from(a > 5.0);
    if x.is_sign_negative() { m | 8 } else { m }
}

/// An e2m1 code's value.
#[must_use]
pub fn e2m1_value(code: u8) -> f32 {
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = VALUES[usize::from(code & 7)];
    if code & 8 != 0 { -v } else { v }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_codes_round_to_the_nearest_and_saturate() {
        assert_eq!(e4m3_bits(448.0), 0x7e);
        assert_eq!(e4m3_bits(1e9), 0x7e);
        assert_eq!(e4m3_bits(-1e9), 0xfe);
        assert_eq!(e4m3_bits(1.0), 0x38);
        assert_eq!(e4m3_bits(0.001_953_125), 0x01);
        assert_eq!(e4m3_bits(-0.0), 0x80);
        for code in 0u8..=0xff {
            if code & 0x7f == 0x7f {
                continue;
            }
            assert_eq!(e4m3_bits(e4m3_value(code)), code, "{code:#x}");
        }
        // ties to the even code
        assert_eq!(e2m1_bits(0.25), 0);
        assert_eq!(e2m1_bits(0.75), 2);
        assert_eq!(e2m1_bits(2.5), 4);
        assert_eq!(e2m1_bits(5.0), 6);
        assert_eq!(e2m1_bits(-7.0), 0xf);
        for code in 0u8..16 {
            assert_eq!(e2m1_bits(e2m1_value(code)), code, "{code:#x}");
        }
    }

    #[test]
    fn each_format_lays_out_deepseeks_bytes() {
        for (rows, bytes) in [
            (BlockRows::V4, 584),
            (BlockRows::MX8, 528),
            (BlockRows::NV4, 288),
        ] {
            assert!(rows.verify());
            assert_eq!(rows.width(), 512);
            assert_eq!(rows.row_bytes(), bytes);
        }
        assert_eq!(
            BlockRows::V4.at(2, 3, 64),
            (2 * 64 * 584 + 3 * 576, 2 * 64 * 584 + 64 * 576 + 3 * 8)
        );
    }

    #[test]
    fn a_rows_blocks_scale_by_their_largest_and_read_back_within_a_code() {
        let row: Vec<f32> = (0..512)
            .map(|i| round_bf16(((i * 37 % 101) as f32 - 50.0) / 7.0 * (1.0 + (i / 64) as f32)))
            .collect();
        for rows in [BlockRows::V4, BlockRows::MX8, BlockRows::NV4] {
            let (data, scales) = rows.encode(&row);
            assert_eq!(data.len() as u32, rows.data_bytes());
            let back = rows.decode(&data, &scales);
            for (b, run) in row[..rows.coded as usize]
                .chunks(rows.block as usize)
                .enumerate()
            {
                let amax = run.iter().fold(0f32, |m, x| m.max(x.abs()));
                let (s, byte) = rows.block_scale(amax);
                assert_eq!(scales[b], byte);
                // the block's largest lands within the code's max, or (an
                // e4m3 scale rounded down) past it by the scale's own step
                let past = if rows.scale == BlockScale::E4m3 {
                    1.0 + 1.0 / 16.0
                } else {
                    1.0
                };
                assert!(amax / s <= rows.code.max() * past, "{rows:?} block {b}");
                for (c, x) in run.iter().enumerate() {
                    let got = back[b * rows.block as usize + c];
                    // half the code's widest gap, and what saturation cuts
                    let step = s * match rows.code {
                        Code::E4m3 => 16.0,
                        Code::E2m1 => 1.0 + 6.0 / 16.0,
                    };
                    assert!((got - x).abs() <= step, "{rows:?} {b}:{c}: {got} vs {x}");
                }
            }
            assert_eq!(
                &back[rows.coded as usize..],
                &row[rows.coded as usize..],
                "the tail"
            );
            assert!(scales[rows.scales() as usize..].iter().all(|b| *b == 0));
        }
        // the floor: an all-zero row's scale
        let zero = vec![0f32; 512];
        let (_, scales) = BlockRows::V4.encode(&zero);
        assert_eq!(scales[0], BlockRows::V4.block_scale(0.0).1);
        assert_eq!(f32::from_bits(u32::from(scales[0]) << 23), 2f32.powi(-22));
        let (data, scales) = BlockRows::NV4.encode(&zero);
        assert!(data.iter().all(|b| *b == 0));
        assert_eq!(scales[0], 0x01);
    }
}
