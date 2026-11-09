//! Mosaic kernels built in Rust (`kernels_xla::mosaic`), called from our
//! StableHLO through `stablehlo.custom_call @tpu_custom_call`, compiled by
//! the PJRT plugin and run on the TPU.

use engine_xla::bench::{Bench, assert_close};
use kernels_xla::hlo::Elem;
use kernels_xla::mosaic::{Body, Grid, ICmp, Kernel, MTy, Operand, Sem, Window};

/// `z = x + y` over a `[rows, 256]` f32 array, one `[8, 256]` block per
/// grid step (block `i` read at step `rows/8 - 1 - i`: the index map is
/// honoured, not just the identity).
#[test]
fn a_rust_built_mosaic_add_runs_on_the_device() {
    let rows = 40u32;
    let steps = i64::from(rows / 8);
    let block = move || Operand {
        array: vec![i64::from(rows), 256],
        elem: Elem::F32,
        block: vec![8, 256],
        window: Window::Blocked,
        index: Box::new(move |b: &mut Body, g: &Grid| {
            let last = b.i32(steps - 1);
            let i = b.subi(last, g.ids[0])?;
            Ok(vec![i, b.i32(0)])
        }),
    };
    let kernel = Kernel {
        name: "add".into(),
        grid: vec![steps],
        semantics: vec![Sem::Parallel],
        prefetch: vec![],
        ins: vec![block(), block()],
        outs: vec![block()],
        vmem_limit: None,
        scratch: vec![],
    };
    let compiled = kernel
        .build(|b, blk| {
            let x = b.vload_all(blk.ins[0])?;
            let y = b.vload_all(blk.ins[1])?;
            let z = b.addf(x, y)?;
            b.vstore_all(z, blk.outs[0])
        })
        .unwrap();

    let n = (rows * 256) as usize;
    let xs: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let ys: Vec<f32> = (0..n).map(|i| 1.0 - (i % 7) as f32).collect();
    let mut b = Bench::new();
    let x = b.f32(rows, 256, &xs);
    let y = b.f32(rows, 256, &ys);
    let z = b.zeros(dtype::Dtype::F32, rows, 256);
    let ran = b
        .run(|ctx| {
            ctx.emit(&mut |cx| {
                let xv = cx.read(x)?;
                let yv = cx.read(y)?;
                let out = compiled.call(cx, &[xv, yv])?;
                cx.write(z, out[0])
            })
        })
        .unwrap_or_else(|e| panic!("{e}\n--- mosaic ---\n{}", compiled.module));
    if !ran {
        return;
    }
    let want: Vec<f32> = xs.iter().zip(&ys).map(|(a, b)| a + b).collect();
    assert_close(&b.read_f32(z), &want, 0.0, 0.0);
}

/// One `[rows, cols]` f32 operand or result read whole by a one-step grid.
fn whole(rows: i64, cols: i64, elem: Elem) -> Operand {
    Operand {
        array: vec![rows, cols],
        elem,
        block: vec![rows, cols],
        window: Window::Blocked,
        index: Box::new(|b: &mut Body, _g: &Grid| Ok(vec![b.i32(0), b.i32(0)])),
    }
}

/// Runs a one-step kernel over `ins` (f32 `[rows, cols]` each) into one f32
/// `[orows, ocols]` result; `None` when no device ran it.
fn run_whole(
    name: &str,
    ins: &[(i64, i64, Vec<f32>)],
    out: (i64, i64),
    body: impl FnOnce(&mut Body, &kernels_xla::mosaic::Blocks) -> kernels_xla::hlo::Built<()>,
) -> Option<Vec<f32>> {
    let kernel = Kernel {
        name: name.into(),
        grid: vec![1],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![],
        ins: ins.iter().map(|(r, c, _)| whole(*r, *c, Elem::F32)).collect(),
        outs: vec![whole(out.0, out.1, Elem::F32)],
        vmem_limit: None,
        scratch: vec![],
    };
    let compiled = kernel.build(body).unwrap();
    let mut b = Bench::new();
    let hs: Vec<_> = ins
        .iter()
        .map(|(r, c, v)| b.f32(*r as u32, *c as u32, v))
        .collect();
    let z = b.zeros(dtype::Dtype::F32, out.0 as u32, out.1 as u32);
    let ran = b
        .run(|ctx| {
            ctx.emit(&mut |cx| {
                let vs = hs.iter().map(|&h| cx.read(h)).collect::<Result<Vec<_>, _>>()?;
                let o = compiled.call(cx, &vs)?;
                cx.write(z, o[0])
            })
        })
        .unwrap_or_else(|e| panic!("{name}: {e}\n--- mosaic ---\n{}", compiled.module));
    ran.then(|| b.read_f32(z))
}

/// A row of 128 turned into a column (`[1, 128]` → `[128, 1]`) and spread
/// across the lanes: `out[i][j] = k[i]`.
#[test]
fn a_row_relaid_as_a_column_spreads_across_lanes() {
    let k: Vec<f32> = (0..8 * 128).map(|i| if i < 128 { i as f32 * 0.25 } else { 0.0 }).collect();
    let Some(got) = run_whole("colbcast", &[(8, 128, k.clone())], (128, 128), |b, blk| {
        let z = b.index(0);
        let k8 = b.vload(blk.ins[0], &[z, z], &[8, 128])?;
        let row = b.rows(k8, 0, 1)?;
        let col = b.shape_cast(row, &[128, 1])?;
        let sq = b.broadcast(col, &[128, 128])?;
        b.vstore(sq, blk.outs[0], &[z, z])
    }) else {
        return;
    };
    let want: Vec<f32> = (0..128 * 128).map(|i| k[i / 128]).collect();
    assert_close(&got, &want, 0.0, 0.0);
}

/// `kᵀ·d` for two rows through the MXU reading its left operand transposed:
/// the outer product `out[i][j] = k[i] d[j]` (f32 operands; the tolerance
/// tells how they are rounded).
#[test]
fn an_lhs_transposed_matmul_forms_an_outer_product() {
    let k: Vec<f32> = (0..8 * 128).map(|i| if i < 128 { 1.0 + i as f32 / 97.0 } else { 0.0 }).collect();
    let d: Vec<f32> = (0..8 * 128).map(|i| if i < 128 { 0.5 - i as f32 / 211.0 } else { 0.0 }).collect();
    let Some(got) = run_whole("outer", &[(8, 128, k.clone()), (8, 128, d.clone())], (128, 128), |b, blk| {
        let z = b.index(0);
        let k8 = b.vload(blk.ins[0], &[z, z], &[8, 128])?;
        let d8 = b.vload(blk.ins[1], &[z, z], &[8, 128])?;
        let acc = b.splat_f(&[128, 128], Elem::F32, 0.0)?;
        let o = b.matmul_tl(k8, d8, acc)?;
        b.vstore(o, blk.outs[0], &[z, z])
    }) else {
        return;
    };
    let want: Vec<f32> = (0..128 * 128).map(|i| k[i / 128] * d[i % 128]).collect();
    let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs() / b.abs().max(1e-3)).fold(0.0f32, f32::max);
    eprintln!("outer product max relative error {err:e}");
    assert_close(&got, &want, 1e-2, 1e-2);
}

/// An f32 `[8, 128]·[128, 128]` on the MXU: how close to the f32 product.
#[test]
fn an_f32_matmul_reports_its_rounding() {
    let x: Vec<f32> = (0..8 * 128).map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5).collect();
    let w: Vec<f32> = (0..128 * 128).map(|i| ((i * 53 % 89) as f32 / 89.0) - 0.5).collect();
    let Some(got) = run_whole("mm32", &[(8, 128, x.clone()), (128, 128, w.clone())], (8, 128), |b, blk| {
        let z = b.index(0);
        let xv = b.vload(blk.ins[0], &[z, z], &[8, 128])?;
        let wv = b.vload(blk.ins[1], &[z, z], &[128, 128])?;
        let acc = b.splat_f(&[8, 128], Elem::F32, 0.0)?;
        let o = b.matmul(xv, wv, acc, false)?;
        b.vstore(o, blk.outs[0], &[z, z])
    }) else {
        return;
    };
    let mut want = vec![0f32; 8 * 128];
    for r in 0..8 {
        for c in 0..128 {
            want[r * 128 + c] = (0..128).map(|k| x[r * 128 + k] as f64 * w[k * 128 + c] as f64).sum::<f64>() as f32;
        }
    }
    let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("f32 matmul max abs error {err:e}");
    assert_close(&got, &want, 0.5, 0.5);
}

/// A column sum (`vector.multi_reduction` over rows) of `[128, 128]`.
#[test]
fn a_column_sum_reduces_the_rows() {
    let x: Vec<f32> = (0..128 * 128).map(|i| (i % 13) as f32 - 6.0 + (i / 128) as f32 * 0.01).collect();
    let Some(got) = run_whole("colsum", &[(128, 128, x.clone())], (8, 128), |b, blk| {
        let z = b.index(0);
        let xv = b.vload(blk.ins[0], &[z, z], &[128, 128])?;
        let s = b.sum_rows(xv)?;
        let s8 = b.broadcast(s, &[8, 128])?;
        b.vstore(s8, blk.outs[0], &[z, z])
    }) else {
        return;
    };
    let mut want = vec![0f32; 8 * 128];
    for c in 0..128 {
        let s: f32 = (0..128).map(|r| x[r * 128 + c]).sum();
        for r in 0..8 {
            want[r * 128 + c] = s;
        }
    }
    assert_close(&got, &want, 1e-4, 1e-4);
}

/// iota, integer and float compares as masks, select, exp, concatenate and a
/// strided slice, in one kernel: `out = [where(r >= c, exp(x), 0) ; x[8..24]
/// where x > 0.5 else -1 ; (r + c) as f32]` over a `[32, 128]` x.
#[test]
fn masks_exp_concat_and_slices_lower_and_run() {
    let x: Vec<f32> = (0..32 * 128).map(|i| ((i * 31 % 97) as f32 / 97.0) - 0.3).collect();
    let Some(got) = run_whole("vecops", &[(32, 128, x.clone())], (64, 128), |b, blk| {
        let z = b.index(0);
        let xv = b.vload(blk.ins[0], &[z, z], &[32, 128])?;
        let r = b.iota(&[32, 128], 0);
        let c = b.iota(&[32, 128], 1);
        let lower = b.cmpi(ICmp::Sge, r, c)?;
        let e = b.exp(xv)?;
        let zero = b.splat_f(&[32, 128], Elem::F32, 0.0)?;
        let tri = b.select(lower, e, zero)?;
        let mid = b.slice(xv, &[8, 0], &[16, 128])?;
        let half = b.splat_f(&[16, 128], Elem::F32, 0.5)?;
        let big = b.cmpf(true, mid, half)?;
        let neg = b.splat_f(&[16, 128], Elem::F32, -1.0)?;
        let picked = b.select(big, mid, neg)?;
        let rc = b.addi(r, c)?;
        let rc = b.slice(rc, &[0, 0], &[16, 128])?;
        let rcf = b.op("arith.sitofp", &[rc], "", &[MTy::Vector(vec![16, 128], Elem::F32)])[0];
        let out = b.concat(&[tri, picked, rcf], 0)?;
        b.vstore(out, blk.outs[0], &[z, z])
    }) else {
        return;
    };
    let mut want = vec![0f32; 64 * 128];
    for r in 0..32 {
        for c in 0..128 {
            want[r * 128 + c] = if r >= c { x[r * 128 + c].exp() } else { 0.0 };
        }
    }
    for r in 0..16 {
        for c in 0..128 {
            let v = x[(r + 8) * 128 + c];
            want[(32 + r) * 128 + c] = if v > 0.5 { v } else { -1.0 };
            want[(48 + r) * 128 + c] = (r + c) as f32;
        }
    }
    assert_close(&got, &want, 1e-5, 1e-5);
}

/// Runs a one-step kernel over 3-d f32 inputs into one 3-d f32 result (the
/// arrays flattened row-major on the host); `None` when no device ran it.
fn run_whole3(
    name: &str,
    ins: &[(Vec<i64>, Vec<f32>)],
    out: &[i64],
    body: impl FnOnce(&mut Body, &kernels_xla::mosaic::Blocks) -> kernels_xla::hlo::Built<()>,
) -> Option<Vec<f32>> {
    let op = |d: &[i64]| Operand {
        array: d.to_vec(),
        elem: Elem::F32,
        block: d.to_vec(),
        window: Window::Blocked,
        index: Box::new(|b: &mut Body, _g: &Grid| Ok(vec![b.i32(0), b.i32(0), b.i32(0)])),
    };
    let kernel = Kernel {
        name: name.into(),
        grid: vec![1],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![],
        ins: ins.iter().map(|(d, _)| op(d)).collect(),
        outs: vec![op(out)],
        vmem_limit: None,
        scratch: vec![],
    };
    let compiled = kernel.build(body).unwrap();
    let mut b = Bench::new();
    let rows = |d: &[i64]| (d[0] * d[1]) as u32;
    let hs: Vec<_> = ins.iter().map(|(d, v)| b.f32(rows(d), d[2] as u32, v)).collect();
    let z = b.zeros(dtype::Dtype::F32, rows(out), out[2] as u32);
    let ran = b
        .run(|ctx| {
            ctx.emit(&mut |cx| {
                let vs: Vec<_> = hs
                    .iter()
                    .zip(ins)
                    .map(|(&h, (d, _))| cx.read(h).and_then(|v| cx.reshape(v, d).map_err(Into::into)))
                    .collect::<Result<_, _>>()?;
                let o = compiled.call(cx, &vs)?;
                let o = cx.reshape(o[0], &[rows(out) as i64, out[2]])?;
                cx.write(z, o)
            })
        })
        .unwrap_or_else(|e| panic!("{name}: {e}\n--- mosaic ---\n{}", compiled.module));
    ran.then(|| b.read_f32(z))
}

/// Batched matmuls over `[4, 32, 128]`-shaped operands in the three
/// contraction forms the GDN chunk kernel uses, and an f32-precision one.
#[test]
fn batched_matmuls_contract_the_axes_they_name() {
    let (h, c, d) = (4i64, 32i64, 128i64);
    let mk = |seed: usize, n: usize| -> Vec<f32> { (0..n).map(|i| (((i * 37 + seed) % 101) as f32 / 101.0) - 0.5).collect() };
    let a = mk(1, (h * c * d) as usize); // [h, c, d]
    let bb = mk(5, (h * d * d) as usize); // [h, d, d]
    let at = |i: i64, r: i64, k: i64| a[((i * c + r) * d + k) as usize];
    let bt = |i: i64, k: i64, n: i64| bb[((i * d + k) * d + n) as usize];
    // 1. [h,c,d] x [h,d,d] -> [h,c,d]
    if let Some(got) = run_whole3("bmm", &[(vec![h, c, d], a.clone()), (vec![h, d, d], bb.clone())], &[h, c, d], |b, blk| {
        let x = b.vload_all(blk.ins[0])?;
        let w = b.vload_all(blk.ins[1])?;
        let acc = b.splat_f(&[h, c, d], Elem::F32, 0.0)?;
        let o = b.matmul_g(x, w, acc, 2, 1, false)?;
        b.vstore_all(o, blk.outs[0])
    }) {
        let mut want = vec![0f32; (h * c * d) as usize];
        for i in 0..h { for r in 0..c { for n in 0..d {
            want[((i * c + r) * d + n) as usize] = (0..d).map(|k| at(i, r, k) as f64 * bt(i, k, n) as f64).sum::<f64>() as f32;
        }}}
        assert_close(&got, &want, 0.1, 0.1);
    } else {
        return;
    }
    // 2. [h,c,d] x [h,c,d]^T (contract 2 with 2) -> [h,c,c]; and 3. contract 1 with 1 -> [h,d,d]
    let a2 = mk(9, (h * c * d) as usize);
    let a2t = |i: i64, r: i64, k: i64| a2[((i * c + r) * d + k) as usize];
    let got = run_whole3("bmm_t", &[(vec![h, c, d], a.clone()), (vec![h, c, d], a2.clone())], &[h, c, c], |b, blk| {
        let x = b.vload_all(blk.ins[0])?;
        let y = b.vload_all(blk.ins[1])?;
        let acc = b.splat_f(&[h, c, c], Elem::F32, 0.0)?;
        let o = b.matmul_g(x, y, acc, 2, 2, true)?;
        b.vstore_all(o, blk.outs[0])
    }).unwrap();
    let mut want = vec![0f32; (h * c * c) as usize];
    for i in 0..h { for r in 0..c { for s in 0..c {
        want[((i * c + r) * c + s) as usize] = (0..d).map(|k| at(i, r, k) as f64 * a2t(i, s, k) as f64).sum::<f64>() as f32;
    }}}
    let err = got.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    eprintln!("f32-precision batched matmul max abs error {err:e}");
    assert_close(&got, &want, 1e-3, 1e-3);
    let got = run_whole3("bmm_tl", &[(vec![h, c, d], a.clone()), (vec![h, c, d], a2.clone())], &[h, d, d], |b, blk| {
        let x = b.vload_all(blk.ins[0])?;
        let y = b.vload_all(blk.ins[1])?;
        let acc = b.splat_f(&[h, d, d], Elem::F32, 0.0)?;
        let o = b.matmul_g(x, y, acc, 1, 1, false)?;
        b.vstore_all(o, blk.outs[0])
    }).unwrap();
    let mut want = vec![0f32; (h * d * d) as usize];
    for i in 0..h { for k in 0..d { for n in 0..d {
        want[((i * d + k) * d + n) as usize] = (0..c).map(|r| at(i, r, k) as f64 * a2t(i, r, n) as f64).sum::<f64>() as f32;
    }}}
    assert_close(&got, &want, 0.1, 0.1);
}

/// 3-d iota masks pick the lower triangle of every head, and a `[h, 1, c]`
/// row and `[h, c, c]` lane slice broadcast as the chunk kernel reads them.
#[test]
fn three_d_masks_and_broadcasts_follow_their_axes() {
    let (h, c) = (4i64, 32i64);
    let x: Vec<f32> = (0..h * c * 128).map(|i| (i % 7) as f32).collect();
    let r: Vec<f32> = (0..h * c).map(|i| (i % 11) as f32 * 0.5).collect(); // [h, 1, c]
    let Some(got) = run_whole3("mask3", &[(vec![h, c, 128], x.clone()), (vec![h, 1, c], r.clone())], &[h, c, c], |b, blk| {
        let xv = b.vload_all(blk.ins[0])?;
        let rv = b.vload_all(blk.ins[1])?;
        let sq = [h, c, c];
        let ri = b.iota(&sq, 1);
        let cj = b.iota(&sq, 2);
        let lower = b.cmpi(ICmp::Sge, ri, cj)?;
        let col = b.slice(xv, &[0, 0, 0], &[h, c, c])?;
        let row = b.broadcast(rv, &sq)?;
        let d = b.subf(col, row)?;
        let zero = b.splat_f(&sq, Elem::F32, 0.0)?;
        let o = b.select(lower, d, zero)?;
        b.vstore_all(o, blk.outs[0])
    }) else {
        return;
    };
    let mut want = vec![0f32; (h * c * c) as usize];
    for i in 0..h { for a in 0..c { for j in 0..c {
        if a >= j {
            want[((i * c + a) * c + j) as usize] = x[((i * c + a) * 128 + j) as usize] - r[(i * c + j) as usize];
        }
    }}}
    assert_close(&got, &want, 0.0, 0.0);
}

/// A `[32, 48]` plane transposed to `[48, 32]`, and its columns (a head's
/// value per token) stretched along the lanes as `[48, 32, 128]`.
#[test]
fn a_transposed_plane_spreads_its_columns_along_the_lanes() {
    let (c, h) = (32i64, 48i64);
    let x: Vec<f32> = (0..c * h).map(|i| ((i * 13 % 59) as f32) - 29.0).collect();
    let Some(got) = run_whole3("tr", &[(vec![1, c, h], x.clone())], &[h, c, 128], |b, blk| {
        let xv = b.vload_all(blk.ins[0])?;
        let xv = b.shape_cast(xv, &[c, h])?;
        let xt = b.transpose(xv)?;
        let col = b.shape_cast(xt, &[h, c, 1])?;
        let wide = b.broadcast(col, &[h, c, 128])?;
        b.vstore_all(wide, blk.outs[0])
    }) else {
        return;
    };
    let want: Vec<f32> = (0..h * c * 128).map(|i| x[((i / 128) % c * h + i / (c * 128)) as usize]).collect();
    assert_close(&got, &want, 0.0, 0.0);
}
