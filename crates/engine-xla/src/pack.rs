//! The host side of the storage contract kernels-xla's `pack` declares:
//! the device type a bank's codes plane lands as, and the transforms that
//! land a checkpoint plane's bytes in that form (PJRT's host layout).

use dtype::Dtype;
use kernels_xla::hlo::Ty;
use kernels_xla::pack::{code_bits, code_elem, codes_per_row};

/// The device type of a codes plane `rows` x `width` (as declared), or
/// `None` for a dtype that is not a bank's codes.
#[must_use]
pub fn storage(dtype: Dtype, rows: u32, width: u32) -> Option<Ty> {
    let elem = code_elem(dtype)?;
    let codes = codes_per_row(dtype, u64::from(width))?;
    Some(Ty::new(
        elem,
        &[i64::from(rows), i64::try_from(codes).ok()?],
    ))
}

/// The checkpoint's packed rows of a codes plane, one byte per code, low
/// bits first (PJRT's host layout for a sub-byte type is one element per
/// byte). `None` for a dtype that is not a bank's codes.
#[must_use]
pub fn land(dtype: Dtype, packed: &[u8]) -> Option<Vec<u8>> {
    let bits = code_bits(dtype)?;
    if bits == 8 {
        return Some(packed.to_vec());
    }
    let per = (8 / bits) as usize;
    let mask = (1u8 << bits) - 1;
    let mut out = vec![0u8; packed.len() * per];
    par_chunks(&mut out, packed, per, |dst, src| {
        for (d, &b) in dst.chunks_exact_mut(per).zip(src) {
            for (j, slot) in d.iter_mut().enumerate() {
                *slot = (b >> (j as u32 * bits)) & mask;
            }
        }
    });
    Some(out)
}

/// The e5m2 byte of `v` when it is exactly representable.
fn e5m2_exact(v: f64) -> Option<u8> {
    if v.is_nan() {
        return Some(0x7E);
    }
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let a = v.abs();
    if a == 0.0 {
        return Some(sign);
    }
    let e = a.log2().floor() as i32;
    // Guard log2 rounding at exact powers of two.
    let e = if 2f64.powi(e) > a {
        e - 1
    } else if 2f64.powi(e + 1) <= a {
        e + 1
    } else {
        e
    };
    if (-14..=15).contains(&e) {
        let m = (a / 2f64.powi(e) - 1.0) * 4.0;
        if m.fract() != 0.0 {
            return None;
        }
        Some(sign | (((e + 15) as u8) << 2) | m as u8)
    } else if e < -14 {
        let m = a / 2f64.powi(-16);
        if m.fract() != 0.0 || m < 1.0 || m > 3.0 {
            return None;
        }
        Some(sign | m as u8)
    } else {
        None
    }
}

const E2M1: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// An mxfp4 bank's weights, one e5m2 byte each, from its packed codes (two
/// per byte, low nibble first) and e8m0 scales (one per 32 codes, in the
/// same order); `None` when a block's scale puts a value outside what e5m2
/// holds exactly.
#[must_use]
pub fn mxfp4_e5m2(codes: &[u8], scales: &[u8]) -> Option<Vec<u8>> {
    if codes.len() != scales.len() * 16 {
        return None;
    }
    // lut[scale][code]: 0x100 marks inexact.
    let mut lut = vec![[0u16; 16]; 256];
    for (s, row) in lut.iter_mut().enumerate() {
        let f = if s == 0xFF {
            f64::NAN
        } else {
            2f64.powi(s as i32 - 127)
        };
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = e5m2_exact(E2M1[c] * f).map_or(0x100, u16::from);
        }
    }
    let mut out = vec![0u8; codes.len() * 2];
    let bad = std::sync::atomic::AtomicBool::new(false);
    // One block is 16 code bytes and one scale; chunk by blocks.
    par_blocks(&mut out, codes, scales, |dst, src, sc| {
        let (dst, _) = dst.as_chunks_mut::<32>();
        let (src, _) = src.as_chunks::<16>();
        for ((d, s), &e) in dst.iter_mut().zip(src).zip(sc) {
            let row = &lut[e as usize];
            let mut miss = 0u16;
            let (d, _) = d.as_chunks_mut::<2>();
            for (pair, &b) in d.iter_mut().zip(s) {
                let lo = row[(b & 0xF) as usize];
                let hi = row[(b >> 4) as usize];
                miss |= (lo | hi) & 0x100;
                pair[0] = lo as u8;
                pair[1] = hi as u8;
            }
            if miss != 0 {
                bad.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    });
    if bad.into_inner() { None } else { Some(out) }
}

fn threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(32)
}

/// Runs `f(dst, src)` over matching chunks (`dst` holds `per` bytes per
/// `src` byte) on the available cores.
fn par_chunks(dst: &mut [u8], src: &[u8], per: usize, f: impl Fn(&mut [u8], &[u8]) + Sync) {
    let n = threads();
    let step = src.len().div_ceil(n).max(1 << 16);
    std::thread::scope(|scope| {
        for (d, s) in dst.chunks_mut(step * per).zip(src.chunks(step)) {
            let f = &f;
            scope.spawn(move || f(d, s));
        }
    });
}

/// [`par_chunks`] over mxfp4 blocks: 32 output bytes, 16 code bytes and one
/// scale each.
fn par_blocks(
    dst: &mut [u8],
    codes: &[u8],
    scales: &[u8],
    f: impl Fn(&mut [u8], &[u8], &[u8]) + Sync,
) {
    let n = threads();
    let blocks = scales.len().div_ceil(n).max(1 << 12);
    std::thread::scope(|scope| {
        for ((d, c), s) in dst
            .chunks_mut(blocks * 32)
            .zip(codes.chunks(blocks * 16))
            .zip(scales.chunks(blocks))
        {
            let f = &f;
            scope.spawn(move || f(d, c, s));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_unpack_low_bits_first() {
        assert_eq!(
            land(Dtype::U4g64, &[0x21, 0xF0]).unwrap(),
            vec![1, 2, 0, 15]
        );
        assert_eq!(
            land(Dtype::U2g32, &[0b1110_0100]).unwrap(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(land(Dtype::U8g64, &[7, 9]).unwrap(), vec![7, 9]);
        assert!(land(Dtype::Bf16, &[0]).is_none());
    }

    #[test]
    fn mxfp4_prescales_exactly_or_refuses() {
        // Scale 2^0: codes 1 (0.5), 7 (6.0), 9 (-0.5), 8 (-0.0).
        let codes = [0x71u8, 0x89].repeat(8);
        let got = mxfp4_e5m2(&codes, &[127]).unwrap();
        // 0.5 = 2^-1: e=14 → 0x38; 6 = 1.5·2^2: e=17, m=2 → 0x46.
        assert_eq!(&got[..4], &[0x38, 0x46, 0xB8, 0x80]);
        // 2^-30 puts 0.5 far below e5m2's smallest subnormal.
        assert!(mxfp4_e5m2(&codes, &[97]).is_none());
        // Subnormal edge: 0.5·2^-15 = 2^-16, the smallest e5m2 subnormal.
        let got = mxfp4_e5m2(&[0x01u8; 16], &[112]).unwrap();
        assert_eq!(got[0], 0x01);
    }
}
