//! The dense projections (bf16, affine, NVFP4, FP8 and the quantizing
//! products) against the host's dot of their decoded operands.

use crate::common::linear_ref::{Bench, TOL, Want, affine, check, dot, fp8_bank, glu_want};
use crate::common::linear_ref::{pick, t, tab};
use crate::common::{SM100, from_bf16, to_bf16};
use dtype::Dtype::{Bf16, F32, I32, U8};
use kernels_cuda::linear::act_quant::{self, Format, Source, Weight};
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::linear::tc5::{Elem, Glu, atom_bytes, supported};
use kernels_cuda::tuning::Tuning;

#[rustfmt::skip]
const GEMV: Tuning = Tuning { gemm_fp8_gemv_until: 8, ..SM100 };

/// The dot of `x` (`[m][k]`) and `w` (`[n][k]`), with `bias[c]` added.
fn dense(x: &[f64], w: &[f64], [m, n, k]: [usize; 3], bias: Option<&[f64]>) -> Vec<Want> {
    let want = dot([m, n, k], |r, i| x[r * k + i], |_, c, i| w[c * k + i]);
    let add = |i: usize| bias.map_or(0.0, |b| b[i % n]);
    tab(want.len(), |i| (want[i].0 + add(i), want[i].1))
}

#[test]
fn the_few_rows_and_narrow_gemms_land_the_dot() {
    use kernels_cuda::linear::{gemm_narrow_bf16, gemm_rows_bf16};
    // (few rows, m, n, k, bias): the rows kernel at each K split (8, 4, 2,
    // 1 warps a row) and over a cluster (a router's few columns), the narrow
    // one at column counts of its template.
    #[rustfmt::skip]
    let cases = [(true, 8, 512, 4096, false), (true, 5, 300, 96, true), (true, 2, 64, 48, true),
        (true, 3, 7, 1032, false), (true, 8, 16, 8192, true), (false, 64, 4, 10240, false), (false, 7, 1, 264, true)];
    for (rows, m, n, k, bias) in cases {
        let mut b = Bench::new((m * 31 + n * 7 + k) as u64);
        let ((x, xv), (w, wv), (bias_at, bv)) = (b.bf16(m * k), b.bf16(n * k), b.bf16(n));
        let (bias_at, y, ctx) = (if bias { bias_at } else { 0 }, b.zeros(m * n * 2), b.ctx());
        let s = (m as i32, n as i32, k as i32);
        let fire = || match rows {
            true => gemm_rows_bf16(&ctx, w, x, bias_at, y, s.0, s.1, s.2, false),
            false => gemm_narrow_bf16(&ctx, w, x, bias_at, y, s.0, s.1, s.2),
        };
        fire().expect("fires");
        let want = dense(&xv, &wv, [m, n, k], bias.then_some(&bv[..]));
        b.check(y, &want, TOL, &format!("{m}x{n}x{k}"));
        // The slices add in a fixed order: a second launch lands the same bits.
        let first = b.read(y, m * n);
        fire().expect("fires again");
        assert_eq!(b.read(y, m * n), first, "{m}x{n}x{k} lands again");
    }
}

#[test]
fn the_skinny_gemm_lands_the_dot_and_its_epilogue() {
    use kernels_cuda::linear::skinny::{Epilogue, ROWS, covers, skinny_bf16};
    // Each epilogue at a ragged row count, and a full tile.
    #[rustfmt::skip]
    let cases = [(Epilogue::Store, 5, 192, 128), (Epilogue::Store, 64, 640, 2560),
        (Epilogue::Softcap(30.0), 64, 256, 256), (Epilogue::Geglu, 5, 96, 256)];
    for (epi, m, n, k) in cases {
        let rows = pick(epi == Epilogue::Geglu, 2 * n, n);
        let mut b = Bench::new(0x51 ^ (m * n) as u64);
        let ((w, wv), (x, xv), (y, yv)) = (b.bf16(rows * k), b.bf16(m * k), b.bf16((m + 1) * n));
        let s = (m as i32, n as i32, k as i32);
        skinny_bf16(&b.ctx(), w, x, y, s.0, s.1, s.2, epi).expect("the skinny GEMM fires");
        let d = dense(&xv, &wv, [m, rows, k], None);
        let gu = |i: usize| (d[i / n * rows + i % n], d[i / n * rows + n + i % n]);
        let want = tab(m * n, |i| match epi {
            Epilogue::Softcap(c) => (f64::from(c) * (d[i].0 / f64::from(c)).tanh(), d[i].1),
            Epilogue::Geglu => glu_want(Glu::GegluTanh, gu(i).0, gu(i).1),
            Epilogue::Store => d[i],
        });
        b.check(y, &want, TOL, &format!("{epi:?} {m}x{n}x{k}"));
        let past = b.read(y, (m + 1) * n)[m * n..] == yv[m * n..];
        assert!(past, "{epi:?}: the row past m moved");
    }
    // Shapes off the tile are refused, without a launch.
    let store = Epilogue::Store;
    assert!(!covers(1, 96, 128, store, ROWS) && !covers(65, 64, 128, store, ROWS));
    assert!(!covers(1, 64, 192, store, ROWS) && !covers(1, 48, 128, Epilogue::Geglu, ROWS));
    let mut b = Bench::new(1);
    let (w, a, y) = (b.zeros(96 * 256), b.zeros(256), b.zeros(192));
    assert!(skinny_bf16(&b.ctx(), w, a, y, 1, 96, 128, store).is_err());
}

#[test]
fn the_biased_projection_lands_the_dot_and_its_bias() {
    use kernels_cuda::linear::gemm::matmul_bias;
    // The bias folded into the gemv's store, and on cuBLASLt's epilogue.
    #[rustfmt::skip]
    let cases = [(1, 512, 256, false), (3, 4104, 256, false), (64, 640, 2880, true)];
    for (m, n, k, lt) in cases {
        let mut b = Bench::new(0x5b ^ n as u64);
        let ((w, wv), (x, xv), (bias, bv)) = (b.bf16(n * k), b.bf16(m * k), b.bf16(n));
        let mut y = b.out(m, n);
        let mut tuning = b.ctx().tuning();
        tuning.gemm_bias_epilogue = lt;
        let ctx = b.ctx().with_tuning(tuning);
        let (xt, wt, bt) = (t(x, m, k, Bf16), t(w, n, k, Bf16), t(bias, 1, n, Bf16));
        // Twice: the first call picks the shape's tactic, the second runs it.
        for _ in 0..2 {
            matmul_bias(&ctx, xt, wt, bt, &mut y).expect("the biased projection fires");
        }
        let want = dense(&xv, &wv, [m, n, k], Some(&bv));
        b.check(y.ptr, &want, TOL, &format!("{m}x{n}x{k}"));
    }
}

#[test]
fn the_lane_projection_and_the_scalar_multiply_land_each_dtype() {
    use kernels_cuda::elemwise::norm::mul_scalar;
    use kernels_cuda::linear::lane_gemm::act_x_wt;
    let (rows, k, n, s) = (5usize, 96usize, 40usize, 1000.0f32 / 3.0);
    for dtype in [F32, Bf16] {
        let mut b = Bench::new(11);
        let ((x, xv), (w, wv)) = (b.bf16(rows * k), b.bf16(n * k));
        let f32s = tab(xv.len(), |i| xv[i] as f32);
        let (f, y) = (dtype == F32, b.zeros(rows * n * 4));
        let x = pick(f, b.up(&f32s), x);
        let read = |b: &Bench, at: u64, count: usize| -> Vec<f64> {
            let v: Vec<f32> = b.down(at, pick(f, count, 0));
            pick(f, tab(v.len(), |i| f64::from(v[i])), b.read(at, count))
        };
        let ctx = b.ctx();
        let (xt, wt, mut yt) = (t(x, rows, k, dtype), t(w, n, k, Bf16), t(y, rows, n, dtype));
        act_x_wt(&ctx, "linear.matmul", xt, wt, &mut yt).expect("the lane projection fires");
        // The scalar multiply scales the plane in place, to the rounded product.
        mul_scalar(&ctx, s, &mut t(x, rows, k, dtype)).expect("the multiply fires");
        b.sync();
        let (want, got) = (dense(&xv, &wv, [rows, n, k], None), read(&b, y, rows * n));
        check(&got, &want, pick(f, 1e-5, TOL), &format!("{dtype:?}"));
        let round = |v: f32| pick(f, v, from_bf16(to_bf16(v)));
        let scaled = tab(rows * k, |i| f64::from(round(f32s[i] * round(s))));
        let same = read(&b, x, rows * k) == scaled;
        assert!(same, "{dtype:?}: the scalar product");
    }
}

#[test]
fn the_strided_grouped_projection_answers_each_group() {
    use kernels_cuda::linear::moe::matmul_grouped_strided;
    let (rows, groups, k, n) = (37usize, 4usize, 256usize, 64usize);
    let mut b = Bench::new(0x9a7);
    let ((x, xv), (w, wv)) = (b.bf16(rows * groups * k), b.bf16(groups * n * k));
    let mut y = b.out(rows, groups * n);
    let (xt, wt) = (t(x, rows, groups * k, Bf16), t(w, groups * n, k, Bf16));
    matmul_grouped_strided(&b.ctx(), xt, wt, &mut y, groups as u32).expect("fires");
    // Row r's group g against plane g: rows of `n` columns, `groups` a row.
    let w = |r: usize, c: usize, i: usize| wv[((r % groups) * n + c) * k + i];
    let want = dot([rows * groups, n, k], |r, i| xv[r * k + i], w);
    b.check(y.ptr, &want, TOL, "strided");
}

#[test]
fn the_quantized_dense_projections_answer_the_dequantized_dot() {
    use kernels_cuda::linear::{nvfp4, quant};
    // (NVFP4, bits, rows): the affine kernel at eight bits and four; NVFP4
    // by route at one row, through a decoded bf16 tile past it.
    for (nv, bits, rows) in [(false, 8, 5), (false, 4, 26), (true, 4, 1), (true, 4, 7)] {
        let (n, k, seat) = (320usize, 1024usize, GroupSeat::RESIDENT);
        let mut b = Bench::new(0x61 + rows as u64);
        let (x, xv) = b.bf16(rows * k);
        let (xt, mut y, ctx) = (t(x, rows, k, Bf16), b.out(rows, n), b.ctx());
        let w: Vec<f64> = if nv {
            let s = b.side(Elem::E2m1, Some(16), n, k);
            let ts = tab(n, |r| 0.5 + (r % 9) as f32 / 8.0);
            let (c, sc, tt) = (b.up(&s.bytes), b.up(&s.codes), b.up(&ts));
            let (c, sc, tt) = (t(c, n, k / 2, U8), t(sc, n, k / 16, U8), t(tt, n, 4, U8));
            nvfp4::matmul(&ctx, "linear.matmul", xt, c, sc, tt, &mut y, seat).expect("fires");
            tab(n * k, |i| f64::from(s.value(i / k, k, i % k) * ts[i / k]))
        } else {
            let (s, f, z) = affine(&mut b.lcg, [n, k, 64, bits]);
            let (c, f, z) = (b.up(&s.bytes), b.up(&f), b.up(&z));
            let c = t(c, n, k * bits / 8, U8);
            let (f, z) = (t(f, n, k / 32, U8), t(z, n, k / 32, U8));
            let post = quant::OffsetKind::Post;
            quant::matmul(&ctx, xt, c, f, post, Some(z), Bf16, &mut y, seat).expect("fires");
            tab(n * k, |i| f64::from(s.values[i]))
        };
        let want = dense(&xv, &w, [rows, n, k], None);
        let what = format!("NVFP4 {nv}, {bits} bits, {rows} rows");
        b.check(y.ptr, &want, TOL, &what);
    }
}

#[test]
fn the_fp8_block_projections_answer_the_dot() {
    use kernels_cuda::linear::moe::{dense_fp8_shape, matmul_grouped_fp8, matmul_grouped_fp8_with};
    use kernels_cuda::linear::moe_route::group_routes;
    // (arm, rows, groups, n, k): 0, the mma.sync kernel in each CTA shape
    // the dense pick lands; 1, its grouped projection; 2, a row's groups as
    // its routes on the grouped tcgen05 GEMM; 3, the GEMV over bf16 rows (K
    // split across warps, then rows padded to four).
    #[rustfmt::skip]
    let cases = [(0, 2, 1, 4096, 2048), (0, 4, 1, 8192, 1024), (0, 4, 1, 32768, 1024),
        (0, 8, 1, 1024, 4096), (1, 3, 8, 256, 1024), (2, 37, 4, 256, 1024), (3, 1, 1, 640, 6144),
        (3, 3, 1, 1024, 1536)];
    for (arm, rows, groups, n, k) in cases {
        if !supported() && arm >= 2 {
            continue;
        }
        let mut b = Bench::new(0xf8 + rows as u64);
        let (x, xv) = b.bf16(rows * groups * k);
        // One case's rows each their own factors (`e4m3g128`).
        let per = if (arm, rows, n) == (0, 8, 1024) {
            1
        } else {
            128
        };
        let (codes, atoms, wv) = fp8_bank(&mut b, groups * n, k, per, 0x51 + n as u64);
        let (mut y, back) = (b.out(rows, groups * n), b.zeros(rows * groups * k * 2));
        let ctx = b.ctx().with_tuning(GEMV);
        let (xt, fmt) = (t(x, rows, groups * k, Bf16), Format::E4m3 { gran: 128 });
        act_quant::read_back(&ctx, "test", fmt, Source::rows(xt, None), back).expect("reads back");
        let (rg, n32, k32) = (rows * groups, n as u32, k as u32);
        match arm {
            0 => {
                let shape = dense_fp8_shape(rows as u32, n32, k32).expect("a shape");
                matmul_grouped_fp8_with(&ctx, xt, codes, atoms, &mut y, 1, shape)
            }
            1 => matmul_grouped_fp8(&ctx, xt, codes, atoms, &mut y, groups as u32),
            2 => {
                let mut routes = t(b.zeros(rg * 4), rows, groups, I32);
                group_routes(&ctx, groups as u32, &mut routes).expect("the routes lay");
                let (xr, mut yr) = (t(x, rg, k, Bf16), t(y.ptr, rg, n, Bf16));
                let (experts, elem, gran, alpha_from) = (groups as u32, Elem::E4m3, 128, None);
                #[rustfmt::skip]
                let bank = act_quant::Bank { codes, atoms, experts, n: n32, k: k32, elem, gran, alpha_e: 0, alpha_from };
                act_quant::select(&ctx, "test", fmt, xr, routes, None, bank, &mut yr)
            }
            _ => act_quant::fp8_block_gemv(&ctx, xt, codes, atoms, &mut y).map(|took| assert!(took)),
        }
        .expect("the fp8 projection fires");
        // The quantizing arms read e4m3 rows, the GEMV the bf16 ones.
        let a = if arm == 3 { xv } else { b.read(back, rg * k) };
        let w = |r: usize, c: usize, i: usize| f64::from(wv[((r % groups) * n + c) * k + i]);
        let want = dot([rg, n, k], |r, i| a[r * k + i], w);
        let what = format!("arm {arm}: {rows}x{groups}x{n}x{k}");
        b.check(y.ptr, &want, TOL, &what);
    }
}

#[test]
fn the_quantizing_products_answer_the_read_back_dot() {
    use kernels_cuda::linear::act_quant::{held_planes, matmul, matmul_held, quantize, read_back};
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    // (activation block, weight, its block, m, n, k): W4A8 (MXFP4 under
    // row-major scales) and W8A8 (e4m3 under laid atoms), in V4's 128s and
    // V4.1's 32s.
    #[rustfmt::skip]
    let cases = [(128, Elem::E2m1, 32, 7, 256, 512), (32, Elem::E2m1, 32, 64, 128, 256),
        (128, Elem::E4m3, 128, 5, 384, 640), (32, Elem::E4m3, 32, 130, 200, 1024)];
    for (gran, elem, wg, m, n, k) in cases {
        let mut b = Bench::new(0x48 ^ (m * n + k) as u64);
        let (x, _) = b.bf16(m * k);
        let s = b.side(elem, Some(wg), n, k);
        let laid = elem == Elem::E4m3;
        let atoms = &s.scales.as_ref().expect("scaled").0;
        let scales = pick(laid, atoms, &s.codes);
        let (codes, scales) = (b.up(&s.bytes), b.up(scales));
        let (mut y, back, ctx) = (b.out(m, n), b.zeros(m * k * 2), b.ctx());
        let (fmt, src) = (Format::E4m3 { gran }, Source::rows(t(x, m, k, Bf16), None));
        let (n32, k32, wg) = (n as u32, k as u32, wg as u32);
        #[rustfmt::skip]
        let w = Weight { codes, scales, n: n32, k: k32, elem, gran: wg, laid };
        matmul(&ctx, "test", fmt, src, w, &mut y, 1.0).expect("the product fires");
        read_back(&ctx, "test", fmt, src, back).expect("the read-back fires");
        let a = b.read(back, m * k);
        let w_of = |_, c, i| f64::from(s.value(c, k, i));
        let want = dot([m, n, k], |r, i| a[r * k + i], w_of);
        let what = format!("{elem:?} in {gran}s, {m}x{n}x{k}");
        b.check(y.ptr, &want, TOL, &what);
        if !(laid && gran == 128) {
            continue;
        }
        // Codes a producer landed in the held planes, read with no quantize
        // launch; and under `kernel.pdl`, behind a product of other rows.
        let planes = held_planes(&ctx, "test", m as u32, k32).expect("the held planes");
        let atoms = atom_bytes(m, k, 128);
        quantize(&ctx, "test", fmt, src, planes[0], planes[1], atoms).expect("lands");
        let (mut held, mut again, other) = (b.out(m, n), b.out(m, n), b.bf16(m * k).0);
        matmul_held(&ctx, "test", planes, m as u32, w, &mut held, 1.0).expect("fires");
        let base = ctx.tuning();
        let pdl = b.ctx().with_tuning(Tuning { pdl: true, ..base });
        let first = Source::rows(t(other, m, k, Bf16), None);
        matmul(&pdl, "test", fmt, first, w, &mut b.out(m, n), 1.0).expect("fires");
        matmul(&pdl, "test", fmt, src, w, &mut again, 1.0).expect("fires");
        let got = b.read(y.ptr, m * n);
        let (held, again) = (b.read(held.ptr, m * n), b.read(again.ptr, m * n));
        let same = held == got && again == got;
        assert!(same, "{what}: the held and dependent products");
    }
}
