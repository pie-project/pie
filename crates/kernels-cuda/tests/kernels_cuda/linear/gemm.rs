//! The dense projections (bf16, affine, NVFP4, FP8 and the quantizing
//! products) against the host's dot of their decoded operands.

use crate::common::linear_ref::{Bench, TOL, Want, affine, check, dot, fp8_bank};
use crate::common::linear_ref::{pick, t, tab};
use crate::common::{SM100, from_bf16, to_bf16};
use dtype::Dtype::{Bf16, F32, I32, Nvfp4, U8};
use kernels_cuda::linear::act_quant::{self, Format, Source, Weight};
use kernels_cuda::linear::gemm_sm100::{Elem, atom_bytes, supported};
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::tuning::Tuning;

#[rustfmt::skip]
const GEMV: Tuning = Tuning { gemm_fp8_gemv_until: 8, ..SM100 };

/// The dot of `x` (`[m][k]`) and `w` (`[n][k]`).
fn dense(x: &[f64], w: &[f64], [m, n, k]: [usize; 3]) -> Vec<Want> {
    dot([m, n, k], |r, i| x[r * k + i], |_, c, i| w[c * k + i])
}

#[test]
fn the_few_rows_and_narrow_gemms_land_the_dot() {
    use kernels_cuda::linear::gemv::{gemm_narrow_bf16, gemm_rows_bf16};
    // (few rows, m, n, k): the rows kernel at each K split (8, 4, 2, 1
    // warps a row) and over a cluster (a router's few columns), the narrow
    // one at column counts of its template.
    #[rustfmt::skip]
    let cases = [(true, 8, 512, 4096), (true, 5, 300, 96), (true, 2, 64, 48),
        (true, 3, 7, 1032), (true, 8, 16, 8192), (false, 64, 4, 10240), (false, 7, 1, 264)];
    for (rows, m, n, k) in cases {
        let mut b = Bench::new((m * 31 + n * 7 + k) as u64);
        let ((x, xv), (w, wv)) = (b.bf16(m * k), b.bf16(n * k));
        let (y, ctx) = (b.zeros(m * n * 2), b.ctx());
        let s = (m as i32, n as i32, k as i32);
        let fire = || match rows {
            true => gemm_rows_bf16(&ctx, w, x, y, s.0, s.1, s.2, false),
            false => gemm_narrow_bf16(&ctx, w, x, y, s.0, s.1, s.2),
        };
        fire().expect("fires");
        let want = dense(&xv, &wv, [m, n, k]);
        b.check(y, &want, TOL, &format!("{m}x{n}x{k}"));
        // The slices add in a fixed order: a second launch lands the same bits.
        let first = b.read(y, m * n);
        fire().expect("fires again");
        assert_eq!(b.read(y, m * n), first, "{m}x{n}x{k} lands again");
    }
}

#[test]
fn the_skinny_skeleton_refuses_shapes_off_its_tile() {
    use kernels_cuda::linear::skeleton::Epilogue;
    use kernels_cuda::linear::skinny::{ROWS, covers, skinny};
    // Shapes off the tile are refused, without a launch.
    assert!(!covers(1, 96, 128, 1, ROWS) && !covers(65, 64, 128, 1, ROWS));
    assert!(!covers(1, 64, 192, 1, ROWS) && !covers(1, 48, 128, 2, ROWS));
    let mut b = Bench::new(1);
    let (w, a, y) = (b.zeros(96 * 256), b.zeros(256), b.zeros(192));
    let fired = skinny(
        &b.ctx(),
        Epilogue::store(),
        w,
        a,
        &[],
        &[y],
        None,
        1,
        96,
        128,
    );
    assert!(fired.is_err());
}

#[test]
fn the_lane_projection_and_the_scalar_multiply_land_each_dtype() {
    use kernels_cuda::elemwise::pointwise::mul_scalar;
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
        let (want, got) = (dense(&xv, &wv, [rows, n, k]), read(&b, y, rows * n));
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
    // A decode's one row on the grouped GEMV (`kernel.gemm.grouped_gemv`).
    let mut y1 = b.out(1, groups * n);
    let ctx = b.ctx().with_tuning(Tuning {
        grouped_gemv: true,
        ..Tuning::DEFAULT
    });
    let xt1 = t(x, 1, groups * k, Bf16);
    matmul_grouped_strided(&ctx, xt1, wt, &mut y1, groups as u32).expect("fires");
    let want1 = dot([groups, n, k], |r, i| xv[r * k + i], w);
    b.check(y1.ptr, &want1, TOL, "strided, one row");
}

#[test]
fn the_quantized_dense_projections_answer_the_dequantized_dot() {
    use kernels_cuda::linear::{affine, dequant};
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
            let (c, sc, tt) = (t(c, n, k / 2, Nvfp4), t(sc, n, k / 16, U8), t(tt, n, 4, U8));
            dequant::nvfp4_matmul(&ctx, "linear.matmul", xt, c, sc, tt, &mut y, seat)
                .expect("fires");
            tab(n * k, |i| f64::from(s.value(i / k, k, i % k) * ts[i / k]))
        } else {
            let (s, f, z) = affine(&mut b.lcg, [n, k, 64, bits]);
            let (c, f, z) = (b.up(&s.bytes), b.up(&f), b.up(&z));
            let c = t(c, n, k * bits / 8, U8);
            let (f, z) = (t(f, n, k / 32, U8), t(z, n, k / 32, U8));
            let post = affine::OffsetKind::Post;
            affine::matmul(&ctx, xt, c, f, post, Some(z), Bf16, &mut y, seat).expect("fires");
            tab(n * k, |i| f64::from(s.values[i]))
        };
        let want = dense(&xv, &w, [rows, n, k]);
        let what = format!("NVFP4 {nv}, {bits} bits, {rows} rows");
        b.check(y.ptr, &want, TOL, &what);
    }
}

/// A decode's few rows of a 4-bit MLX plane on the mma.sync GEMV
/// (`kernel.gemm.affine_mma`): each row against the dequantized dot, one
/// tile's rows and a part of one, a plane whose columns end mid-block.
#[test]
fn the_affine_gemv_on_the_tensor_cores_answers_the_dequantized_dot() {
    use kernels_cuda::linear::affine;
    for (rows, n, k) in [
        (1usize, 320usize, 1024usize),
        (2, 4096, 4096),
        (9, 1000, 2048),
        (16, 512, 14336),
    ] {
        let seat = GroupSeat::RESIDENT;
        let mut b = Bench::new(0x4a + rows as u64);
        let (x, xv) = b.bf16(rows * k);
        let (xt, mut y) = (t(x, rows, k, Bf16), b.out(rows, n));
        let ctx = b.ctx().with_tuning(Tuning {
            affine_mma: true,
            ..b.ctx().tuning()
        });
        let (s, f, z) = affine(&mut b.lcg, [n, k, 64, 4]);
        let (c, f, z) = (b.up(&s.bytes), b.up(&f), b.up(&z));
        let c = t(c, n, k / 2, U8);
        let (f, z) = (t(f, n, k / 32, U8), t(z, n, k / 32, U8));
        let post = affine::OffsetKind::Post;
        affine::matmul(&ctx, xt, c, f, post, Some(z), Bf16, &mut y, seat).expect("fires");
        let w = tab(n * k, |i| f64::from(s.values[i]));
        let want = dense(&xv, &w, [rows, n, k]);
        b.check(y.ptr, &want, TOL, &format!("{rows} rows, {n} x {k}"));
    }
}

#[test]
fn the_fp8_block_projections_answer_the_dot() {
    use kernels_cuda::linear::moe::group_routes;
    use kernels_cuda::linear::moe::{dense_fp8_shape, matmul_grouped_fp8, matmul_grouped_fp8_with};
    use kernels_cuda::linear::moe::{matmul_fp8_w8a8, select_fp8_w8a8};
    // (arm, rows, groups, n, k): 0, the mma.sync kernel in each CTA shape
    // the dense pick lands; 1, its grouped projection; 2, a row's groups as
    // its routes on the grouped tcgen05 GEMM; 3, the GEMV over bf16 rows (K
    // split across warps, then rows padded to four); 4, the W8A8 matmul
    // (narrow planes K split, wide ones whole); 5, a row's groups as its
    // routes on the W8A8 select; 6, cuBLASLt's block-scaled FP8 GEMM (rows
    // padded to fours).
    #[rustfmt::skip]
    let cases = [(0, 2, 1, 4096, 2048), (0, 4, 1, 8192, 1024), (0, 4, 1, 32768, 1024),
        (0, 8, 1, 1024, 4096), (1, 3, 8, 256, 1024), (2, 37, 4, 256, 1024), (3, 1, 1, 640, 6144),
        (3, 3, 1, 1024, 1536), (4, 5, 1, 384, 512), (4, 13, 1, 256, 1024), (4, 30, 1, 128, 2048),
        (4, 130, 1, 640, 1536), (4, 40, 1, 16384, 512), (4, 130, 1, 32768, 512),
        (5, 37, 4, 256, 1024), (5, 9, 8, 128, 512), (5, 200, 3, 512, 1024), (5, 130, 2, 768, 512),
        (6, 5, 1, 384, 512), (6, 64, 1, 640, 1536),
        (6, 37, 1, 16384, 512), (6, 130, 1, 256, 2048)];
    // The mma.sync fp8 kernels (arms 0, 1, 4 and 5) are sm_89's: an Ada's e4m3 MMA.
    let fp8_mma = kernels_cuda::jit::cache::arch()
        .and_then(|a| {
            a.trim_start_matches("sm_")
                .trim_end_matches(char::is_alphabetic)
                .parse::<u32>()
                .ok()
        })
        .is_some_and(|cc| cc >= 89);
    // cuBLASLt's 128x128-block float-scaled FP8 GEMM (arm 6) is Hopper's: a
    // Blackwell's cuBLASLt scales FP8 in 32-wide UE8M0 vectors instead.
    let hopper = kernels_cuda::jit::cache::arch().is_some_and(|a| a.starts_with("sm_90"));
    for (arm, rows, groups, n, k) in cases {
        if (!supported() && arm == 2)
            || (!fp8_mma && matches!(arm, 0 | 1 | 4 | 5))
            || (!hopper && arm == 6)
        {
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
            3 => kernels_cuda::linear::gemv_fp8::fp8_block_gemv(&ctx, xt, codes, atoms, 128, &mut y).map(|took| assert!(took)),
            4 => matmul_fp8_w8a8(&ctx, xt, codes, atoms, &mut y),
            6 => kernels_cuda::linear::gemm::cublas_lt_fp8::matmul(&ctx, "test", xt, codes, atoms, &mut y)
                .map(|took| assert!(took, "cuBLASLt takes {rows}x{n}x{k}")),
            _ => {
                let mut routes = t(b.zeros(rg * 4), rows, groups, I32);
                group_routes(&ctx, groups as u32, &mut routes).expect("the routes lay");
                let (xr, mut yr) = (t(x, rg, k, Bf16), t(y.ptr, rg, n, Bf16));
                select_fp8_w8a8(&ctx, xr, codes, atoms, groups as u32, routes, &mut yr)
            }
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

#[test]
fn the_w8a16_projections_answer_the_dot() {
    use kernels_cuda::linear::w8a16;
    // (block, m, n, k): the mma.sync kernel at each CTA shape (16, 32 and 64
    // rows, a part tile of each), and wider fires on the widened plane.
    #[rustfmt::skip]
    let cases = [(128, 1, 384, 1024), (128, 5, 200, 640), (128, 16, 512, 2048), (128, 23, 256, 768),
        (128, 64, 136, 512), (128, 100, 256, 1024), (32, 1, 256, 1024), (32, 37, 200, 512),
        (32, 130, 200, 1024), (32, 64, 1280, 5120), (32, 50, 256, 1152), (32, 64, 520, 1536),
        (128, 48, 384, 2048), (32, 24, 640, 1024), (128, 20, 384, 2048), (128, 256, 384, 1024), (128, 300, 136, 512), (32, 1024, 256, 2048), (128, 777, 520, 640)];
    for (gran, m, n, k) in cases {
        let mut b = Bench::new(0xa16 ^ (m * n + k) as u64);
        let (x, xv) = b.bf16(m * k);
        let s = b.side(Elem::E4m3, Some(gran), n, k);
        let atoms = &s.scales.as_ref().expect("scaled").0;
        let (codes, atoms) = (b.up(&s.bytes), b.up(atoms));
        let (mut y, ctx) = (b.out(m, n), b.ctx());
        let took = w8a16::matmul(
            &ctx,
            "test",
            t(x, m, k, Bf16),
            codes,
            atoms,
            gran as u32,
            &mut y,
        )
        .expect("the W8A16 product fires");
        assert!(took, "{m}x{n}x{k} in {gran}s is a shape it reads");
        let want = dot(
            [m, n, k],
            |r, i| xv[r * k + i],
            |_, c, i| f64::from(s.value(c, k, i)),
        );
        b.check(y.ptr, &want, TOL, &format!("W8A16 in {gran}s, {m}x{n}x{k}"));
    } // The GEMV over bf16 rows at a decode's few rows, in either block width.
    for (gran, m, n, k) in [
        (32, 1, 1280, 5120),
        (32, 3, 512, 1024),
        (128, 2, 640, 2048),
        (32, 1, 4096, 1280),
        (128, 2, 256, 640),
    ] {
        let mut b = Bench::new(0x6e3 ^ (m * n + k) as u64);
        let (x, xv) = b.bf16(m * k);
        let s = b.side(Elem::E4m3, Some(gran), n, k);
        let atoms = &s.scales.as_ref().expect("scaled").0;
        let (codes, atoms) = (b.up(&s.bytes), b.up(atoms));
        let (mut y, ctx) = (b.out(m, n), b.ctx().with_tuning(GEMV));
        let took = kernels_cuda::linear::gemv_fp8::fp8_block_gemv(
            &ctx,
            t(x, m, k, Bf16),
            codes,
            atoms,
            gran as u32,
            &mut y,
        )
        .expect("the GEMV fires");
        assert!(took, "{m}x{n}x{k} in {gran}s is a shape the GEMV reads");
        let want = dot(
            [m, n, k],
            |r, i| xv[r * k + i],
            |_, c, i| f64::from(s.value(c, k, i)),
        );
        b.check(
            y.ptr,
            &want,
            TOL,
            &format!("the GEMV in {gran}s, {m}x{n}x{k}"),
        );
    }
}

#[test]
fn an_mxfp8_release_laid_a_row_a_32_answers_the_dot() {
    use kernels_cuda::linear::act_quant::{Bank, read_back, scales_to_atoms, select};
    use kernels_cuda::linear::moe::select_fp8_w8a16;
    use kernels_cuda::linear::w8a16;
    // (m, experts, n, k): an MXFP8 release's plane as it ships, e4m3 codes
    // beside one e8m0 exponent a 1x32 block of each row (`[rows, k / 32]`),
    // its exponents laid into atoms a row a row (as the load lays them); a
    // projection (one expert) and a bank, each W8A8 against e4m3 in 1x32
    // blocks and W8A16, under the default tuning and B200's.
    #[rustfmt::skip]
    let cases = [(1, 1, 384, 1024), (37, 1, 256, 6144), (130, 1, 200, 2048), (3, 4, 256, 512),
        (64, 8, 128, 1024)];
    for (m, ex, n, k) in cases {
        for tuning in [Tuning::DEFAULT, SM100] {
            let mut b = Bench::new(0x3c8 ^ (m * n + k + ex) as u64);
            let rows = ex * n;
            let s = b.side(Elem::E4m3, Some(32), rows, k);
            let host = &s.scales.as_ref().expect("scaled").0;
            let (codes, plane) = (b.up(&s.bytes), b.up(&s.codes));
            let atoms = b.zeros(atom_bytes(rows, k, 32));
            let ctx = b.ctx().with_tuning(tuning);
            let (r32, k32, n32) = (rows as u32, k as u32, n as u32);
            scales_to_atoms(&ctx, "test", plane, atoms, r32, k32 / 32, 1).expect("lays");
            b.sync();
            let laid: Vec<u8> = b.down(atoms, host.len());
            assert!(
                laid == *host,
                "{rows}x{k}: the exponents land where the GEMM reads them"
            );
            let what = |arm: &str| format!("{arm} {m}x{ex}x{n}x{k}, sm100 {}", tuning.moe_tc5);
            // Each row routed to its own expert, in turn.
            let routes = tab(m, |r| (r % ex) as i32);
            let r = t(b.up(&routes), m, 1, I32);
            let w =
                |r: usize, c: usize, i: usize| f64::from(s.value(routes[r] as usize * n + c, k, i));
            let (x, xv) = b.bf16(m * k);
            let x = t(x, m, k, Bf16);
            // W8A16: the bf16 rows against the codes widened in registers.
            let mut y = b.out(m, n);
            if ex == 1 {
                let took = w8a16::matmul(&ctx, "test", x, codes, atoms, 32, &mut y).expect("fires");
                assert!(took, "{}", what("W8A16"));
            } else {
                select_fp8_w8a16(
                    &ctx, "test", x, codes, atoms, ex as u32, 32, r, &mut y, None,
                )
                .expect("the W8A16 select fires");
            }
            b.check(
                y.ptr,
                &dot([m, n, k], |r, i| xv[r * k + i], w),
                TOL,
                &what("W8A16"),
            );
            if !supported() {
                continue;
            }
            // W8A8: the rows quantized to e4m3 in 1x32 blocks (read back),
            // against the laid atoms on the tcgen05 GEMM.
            let (fmt, src) = (Format::E4m3 { gran: 32 }, Source::rows(x, None));
            let back = b.zeros(m * k * 2);
            read_back(&ctx, "test", fmt, src, back).expect("the read-back fires");
            let mut y = b.out(m, n);
            if ex == 1 {
                #[rustfmt::skip]
                let weight = Weight { codes, scales: atoms, n: n32, k: k32, elem: Elem::E4m3, gran: 32, laid: true };
                act_quant::matmul(&ctx, "test", fmt, src, weight, &mut y, 1.0).expect("fires");
            } else {
                #[rustfmt::skip]
                let bank = Bank { codes, atoms, experts: ex as u32, n: n32, k: k32, elem: Elem::E4m3, gran: 32, alpha_e: 0, alpha_from: None };
                select(&ctx, "test", fmt, x, r, None, bank, &mut y).expect("the W8A8 select fires");
            }
            let a = b.read(back, m * k);
            b.check(
                y.ptr,
                &dot([m, n, k], |r, i| a[r * k + i], w),
                TOL,
                &what("W8A8"),
            );
        }
    }
}
