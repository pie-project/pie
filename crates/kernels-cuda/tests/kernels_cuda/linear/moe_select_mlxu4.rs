//! The grouped affine select on the CUDA cores
//! (`moe_matmul_select_mlxu4_grouped`), off the WMMA arm two ways (f16
//! rows, and `kernel.moe.affine_wmma` off), bit for bit against the host
//! walking its sum as the kernel does, and against the host's f64 dot of the
//! dequantized weights.

use crate::common::linear_ref::{Bench, TOL, affine, check, dot, t, tab};
use crate::common::{from_bf16, to_bf16};
use dtype::Dtype::{self, Bf16, F16, I32, U8};
use kernels_cuda::linear::moe::{self, GroupSeat};
use kernels_cuda::tuning::Tuning;

/// The kernel's K tile: a padded tile's tail adds `0 * 0`.
const TILE_K: usize = 64;

/// f32 to f16, to nearest even, through the subnormals.
fn to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | u16::from(mant != 0) << 9;
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    let (m, shift) = if e <= 0 {
        if e < -10 {
            return sign;
        }
        (mant | 0x80_0000, (14 - e) as u32)
    } else {
        (mant, 13)
    };
    let half = (m >> shift) as u16 | if e > 0 { (e as u16) << 10 } else { 0 };
    let (rem, mid) = (m & ((1 << shift) - 1), 1 << (shift - 1));
    sign | (half + u16::from(rem > mid || (rem == mid && half & 1 == 1)))
}

fn from_f16(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let (exp, mant) = (u32::from(h >> 10) & 0x1f, u32::from(h & 0x3ff));
    match exp {
        0 => {
            let magnitude = mant as f32 * 2f32.powi(-24);
            if sign == 0 { magnitude } else { -magnitude }
        }
        31 => f32::from_bits(sign | 0x7f80_0000 | mant << 13),
        _ => f32::from_bits(sign | (exp + 112) << 23 | mant << 13),
    }
}

/// One case: `[experts, tokens, top, n, k]`, the code width, the group,
/// whether a route reads its token's row, and the activation's dtype.
type Case = ([usize; 5], usize, usize, bool, Dtype);

fn grouped(tuning: Tuning, ([ex, tokens, top, n, k], bits, group, by_token, dtype): Case) {
    let what = format!("{dtype:?} {ex}x{tokens}x{top} {n}x{k} u{bits} g{group} {by_token}");
    let mut b = Bench::new(0x3a1 ^ (tokens * 13 + n * 7 + k) as u64);
    let count = tokens * top;
    // A token's routes spread over the experts, some left unrouted.
    let routes = tab(count, |r| {
        ((r / top * 7 + r % top * 3 + r / top / 5) % ex) as i32
    });
    let (rows_x, div) = if by_token { (tokens, top) } else { (count, 1) };
    let (to, from) = if dtype == F16 {
        (to_f16 as fn(f32) -> u16, from_f16 as fn(u16) -> f32)
    } else {
        (to_bf16 as fn(f32) -> u16, from_bf16 as fn(u16) -> f32)
    };
    let raw = tab(rows_x * k, |_| to(b.lcg.unit() * 2.0));
    let a = tab(raw.len(), |i| from(raw[i]));
    let (s, factors, zeros) = affine(&mut b.lcg, [ex * n, k, group, bits]);
    let per = 8 / bits;
    let code = |i: usize| (u16::from(s.bytes[i / per]) >> (bits * (i % per))) & ((1 << bits) - 1);
    // The weight as the kernel widens it: one fused step, in f32.
    let w = |i: usize| {
        let (f, z) = (from_bf16(factors[i / group]), from_bf16(zeros[i / group]));
        f32::from(code(i)).mul_add(f, z)
    };
    let x = t(b.up(&raw), rows_x, k, dtype);
    let codes = t(b.up(&s.bytes), ex, n * k / per, U8);
    let planes = |b: &mut Bench, v: &[u16]| t(b.up(v), ex, n * k / group * 2, U8);
    let (scales, biases) = (planes(&mut b, &factors), planes(&mut b, &zeros));
    let r = t(b.up(&routes), tokens, top, I32);
    let mut y = t(b.up(&vec![0x7c01u16; count * n]), count, n, dtype);
    let ctx = b.ctx().with_tuning(tuning);
    let seat = GroupSeat::RESIDENT;
    moe::matmul_select_quant(&ctx, x, codes, scales, Some(biases), r, &mut y, seat)
        .unwrap_or_else(|why| panic!("{what}: the grouped select declined: {why}"));
    b.sync();
    let got: Vec<u16> = b.down(y.ptr, count * n);
    let row = |r: usize| r / div;
    let (mut differ, mut first) = (0, None);
    for r in 0..count {
        let e = routes[r] as usize;
        for c in 0..n {
            // A tile's K in order, its tail past K adding zero.
            let at = (e * n + c) * k;
            let mut acc = 0f32;
            for i in 0..k.next_multiple_of(TILE_K) {
                let (wv, xv) = if i < k {
                    (w(at + i), a[row(r) * k + i])
                } else {
                    (0.0, 0.0)
                };
                acc = wv.mul_add(xv, acc);
            }
            let (want, have) = (to(acc), got[r * n + c]);
            if have != want {
                differ += 1;
                first.get_or_insert_with(|| {
                    format!(
                        "route {r} row {c}: {have:#06x} ({}), the walk {want:#06x} ({})",
                        from(have),
                        from(want)
                    )
                });
            }
        }
    }
    assert_eq!(
        differ, 0,
        "{what}: {differ} values off the walk, first {first:?}"
    );
    // The walk is the dequantized weights' dot.
    let d = dot(
        [count, n, k],
        |r, i| f64::from(a[row(r) * k + i]),
        |r, c, i| f64::from(w((routes[r] as usize * n + c) * k + i)),
    );
    let got = tab(count * n, |i| f64::from(from(got[i])));
    check(&got, &d, TOL, &what);
}

#[test]
fn the_grouped_affine_select_lands_its_walk_off_the_wmma_arm() {
    let d = Tuning::DEFAULT;
    // Every fire grouped, from one route; the tcgen05 arm off.
    #[rustfmt::skip]
    let grouped_from_one = Tuning { affine_grouped: true, affine_grouped_from: 1, moe_tc5: false, ..d };
    #[rustfmt::skip]
    let no_wmma = Tuning { affine_wmma: false, ..grouped_from_one };
    let cases: &[Case] = &[
        // One token, one route.
        ([1, 1, 1, 64, 64], 4, 64, true, F16),
        ([8, 1, 2, 128, 256], 4, 32, true, Bf16),
        // K off the 128 tile, and off the 64 one.
        ([8, 5, 2, 96, 96], 4, 32, true, F16),
        ([16, 7, 4, 200, 160], 4, 32, false, Bf16),
        ([8, 9, 2, 128, 2880], 4, 64, true, Bf16),
        ([32, 64, 4, 256, 384], 4, 128, true, F16),
        // Experts no route reaches, a fan-out of eight.
        ([128, 16, 8, 128, 512], 4, 64, true, Bf16),
        // Two- and eight-bit codes.
        ([8, 12, 2, 128, 256], 2, 32, true, F16),
        ([8, 12, 2, 128, 256], 8, 128, false, Bf16),
        // Many routes an expert, past a batch of 16.
        ([4, 1024, 4, 64, 320], 4, 32, true, F16),
        ([64, 2048, 2, 64, 128], 4, 128, true, Bf16),
    ];
    for &case in cases {
        // f16 rows never take the WMMA arm; bf16 ones do unless it is off.
        if case.4 == F16 {
            grouped(grouped_from_one, case);
        }
        grouped(no_wmma, case);
    }
}
