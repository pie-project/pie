//! The tiled four-bit decode point on the CUDA cores and on the tensor
//! cores, at every row count it serves, against the host's dot.

use crate::common::SM100;
use crate::common::linear_ref::{Bench, TOL, affine, dot, t};
use dtype::Dtype::{Bf16, U8};
use kernels_cuda::linear::affine::{matmul_gemv, repack, repacked_rows};
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::tuning::Tuning;

#[test]
fn the_tiled_decode_point_lands_alike_on_the_tensor_cores() {
    // A column count off the band, so the padded band's columns are cut.
    let (n, k) = (200usize, 1024usize);
    let np = repacked_rows(n as u32) as usize;
    for rows in 1..=16 {
        // From 2 rows on the tensor cores; never (17).
        for from in [2u32, 17] {
            let mut b = Bench::new(0x71e + rows as u64);
            let (x, xv) = b.bf16(rows * k);
            let (s, f, z) = affine(&mut b.lcg, [n, k, 64, 4]);
            let (c, f, z) = (b.up(&s.bytes), b.up(&f), b.up(&z));
            let (tc, tf, tz) = (
                b.zeros(np * k / 2),
                b.zeros(np * k / 32),
                b.zeros(np * k / 32),
            );
            let ctx = b.ctx().with_tuning(Tuning {
                qmm_gemv_mma_from: from,
                ..SM100
            });
            let (mut oc, mut of, mut oz) = (
                t(tc, np, k / 2, U8),
                t(tf, np, k / 32, U8),
                t(tz, np, k / 32, U8),
            );
            let (c, f, z) = (t(c, n, k / 2, U8), t(f, n, k / 32, U8), t(z, n, k / 32, U8));
            repack(&ctx, c, f, z, &mut oc, &mut of, &mut oz).expect("repacks");
            let mut y = b.out(rows, n);
            matmul_gemv(
                &ctx,
                t(x, rows, k, Bf16),
                oc,
                of,
                oz,
                &mut y,
                GroupSeat::RESIDENT,
            )
            .expect("fires");
            let want = dot(
                [rows, n, k],
                |r, i| xv[r * k + i],
                |_, c, i| f64::from(s.values[c * k + i]),
            );
            b.check(
                y.ptr,
                &want,
                TOL,
                &format!("{rows} rows, tensor cores from {from}"),
            );
        }
    }
}
