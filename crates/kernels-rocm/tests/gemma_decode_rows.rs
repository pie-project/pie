//! The non-GEMM kernels of a Gemma-4 31B decode step at its shapes (hidden
//! 5376, gate_up 2·21504, sliding layers of 32 query heads over 16 kv heads
//! of 256, global layers of 32 over 4 of 512 with a quarter rotated), at 1,
//! 7, 128 and 512 rows (and a prefill chunk of 1500), and under a staged
//! window that pads the launch past its live rows (rows past the window
//! untouched).

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_abs, close_bf16, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::elemwise::norm;
use kernels_rocm::linear::mlp;
use kernels_rocm::{KvPool, Tensor};

const HIDDEN: u32 = 5376;
const INTER: u32 = 21504;

fn bf(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    values(n, seed, scale).into_iter().map(round_bf16).collect()
}

/// The four window words {rows, row_offset, lanes, lane_offset}.
fn window(gpu: &Gpu, rows: u32, offset: u32) -> Tensor {
    gpu.ints(&[rows as i32, offset as i32, rows as i32, offset as i32])
}

/// The (rows, live rows) cases: whole launches, then padded buckets whose
/// window holds a run of rows.
const CASES: [(u32, Option<(u32, u32)>); 7] = [
    (1, None),
    (7, None),
    (128, None),
    (512, None),
    (128, Some((100, 7))),
    (512, Some((128, 300))),
    // A prefill chunk's rows in a padded bucket.
    (2048, Some((1500, 300))),
];

fn live_rows(rows: u32, win: Option<(u32, u32)>) -> std::ops::Range<usize> {
    win.map_or(0..rows as usize, |(n, at)| at as usize..(at + n) as usize)
}

fn gelu_tanh(g: f32) -> f32 {
    let (a, b) = (0.797_884_6_f32, 0.044_715_f32);
    0.5 * g * (1.0 + (a * (g + b * g * g * g)).tanh())
}

#[test]
fn geglu_tanh_packed_at_decode_rows() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    // Gemma's width (the packs) and one the packs do not divide.
    for inter in [INTER, 1001] {
        for (rows, win) in CASES {
            let p = bf((rows * 2 * inter) as usize, 1, 3.0);
            let packed = gpu.tensor(&p, rows, 2 * inter, Dtype::Bf16);
            let mut y = gpu.zeros(rows, inter, Dtype::Bf16);
            if let Some((n, at)) = win {
                ctx.arm_stage(window(&gpu, n, at).ptr);
            }
            mlp::geglu_tanh_packed(&ctx, packed, inter, 1, &mut y).unwrap();
            ctx.disarm_stage();
            let got = gpu.read(&y);
            let live = live_rows(rows, win);
            let i = inter as usize;
            let want: Vec<f32> = (0..rows as usize * i)
                .map(|e| {
                    let (r, c) = (e / i, e % i);
                    if live.contains(&r) {
                        gelu_tanh(p[r * 2 * i + c]) * p[r * 2 * i + i + c]
                    } else {
                        0.0
                    }
                })
                .collect();
            close_bf16(
                &format!("geglu_tanh_packed {inter} rows{rows} win {win:?}"),
                &got,
                &want,
                1.0,
            );
            for (e, &v) in got.iter().enumerate() {
                if !live.contains(&(e / i)) {
                    assert_eq!(
                        v, 0.0,
                        "geglu {inter} rows{rows} wrote past its window at {e}"
                    );
                }
            }
        }
    }
}

#[test]
fn rmsnorm_residual_add_at_decode_rows() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let inv = |x: &[f32], eps: f32| {
        1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt()
    };
    for scale in [false, true] {
        for (rows, win) in CASES {
            let h = HIDDEN as usize;
            let n = rows as usize * h;
            let (xs, ys) = (bf(n, 2, 2.0), bf(n, 3, 2.0));
            let (w0s, w1s) = (bf(h, 4, 1.0), bf(h, 5, 1.0));
            let sv = 0.75f32;
            let x = gpu.tensor(&xs, rows, HIDDEN, Dtype::Bf16);
            let mut y = gpu.tensor(&ys, rows, HIDDEN, Dtype::Bf16);
            let mut t = gpu.zeros(rows, HIDDEN, Dtype::Bf16);
            let mut out = gpu.zeros(rows, HIDDEN, Dtype::Bf16);
            let mut scaled = gpu.zeros(rows, HIDDEN, Dtype::Bf16);
            let s = gpu.tensor(&[sv], 1, 1, Dtype::Bf16);
            let (w0, w1) = (
                gpu.tensor(&w0s, 1, HIDDEN, Dtype::Bf16),
                gpu.tensor(&w1s, 1, HIDDEN, Dtype::Bf16),
            );
            if let Some((n, at)) = win {
                ctx.arm_stage(window(&gpu, n, at).ptr);
            }
            norm::rmsnorm_residual_add(
                &ctx,
                x,
                w0,
                1e-6,
                &mut t,
                &mut y,
                scale.then_some((s, &mut scaled)),
                Some(norm::PostNorm {
                    weight: w1,
                    plus_one: false,
                    eps: 1e-6,
                    out: &mut out,
                }),
            )
            .unwrap();
            ctx.disarm_stage();
            let (gt, gy, go) = (gpu.read(&t), gpu.read(&y), gpu.read(&out));
            let gs = gpu.read(&scaled);
            let live = live_rows(rows, win);
            let tag = format!("rmsnorm_residual_add rows{rows} win {win:?} scale={scale}");
            for r in 0..rows as usize {
                let at = r * h..(r + 1) * h;
                if !live.contains(&r) {
                    assert_eq!(&gy[at.clone()], &ys[at.clone()], "{tag}: dead row {r} y");
                    assert!(
                        gt[at.clone()].iter().all(|&v| v == 0.0),
                        "{tag}: dead row {r} t"
                    );
                    assert!(go[at].iter().all(|&v| v == 0.0), "{tag}: dead row {r} out");
                    continue;
                }
                // The chain as the device rounds it: each step stored, the
                // next reading the stored value (an ulp of the norm's sum
                // order in one step must not fail the next).
                let xr = &xs[at.clone()];
                let i0 = inv(xr, 1e-6);
                let tr: Vec<f32> = xr.iter().zip(&w0s).map(|(v, w)| v * i0 * w).collect();
                let yr: Vec<f32> = ys[at.clone()]
                    .iter()
                    .zip(&gt[at.clone()])
                    .map(|(a, b)| round_bf16(a + b))
                    .collect();
                let last: Vec<f32> = if scale {
                    gy[at.clone()].iter().map(|v| round_bf16(v * sv)).collect()
                } else {
                    gy[at.clone()].to_vec()
                };
                let i1 = inv(&last, 1e-6);
                let or: Vec<f32> = last.iter().zip(&w1s).map(|(v, w)| v * i1 * w).collect();
                close_bf16(&format!("{tag} t row {r}"), &gt[at.clone()], &tr, 1.0);
                close_bf16(&format!("{tag} y row {r}"), &gy[at.clone()], &yr, 1.0);
                if scale {
                    close_bf16(
                        &format!("{tag} scaled row {r}"),
                        &gs[at.clone()],
                        &last,
                        1.0,
                    );
                }
                close_bf16(&format!("{tag} out row {r}"), &go[at], &or, 2.0);
            }
        }
    }
}

#[test]
fn qkv_fused_at_decode_rows() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (theta, eps, ps) = (1e6f32, 1e-6f32, 16u32);
    for (hq, hk, d, rotary) in [(32u32, 16u32, 256u32, 256u32), (32, 4, 512, 128)] {
        for (rows, win) in CASES {
            let width = (hq + 2 * hk) * d;
            let packed_s = bf((rows * width) as usize, 6, 2.0);
            let qw = bf(d as usize, 7, 1.0);
            let kw = bf(d as usize, 8, 1.0);
            let positions: Vec<i32> = (0..rows as i32).map(|r| (r * 977 + 3) % 9000).collect();
            // Row r lands in page r / 2 (reversed), slot 3 + r % 2.
            let pages = rows.div_ceil(2) as usize + 1;
            let w_page: Vec<i32> = (0..rows as i32).map(|r| pages as i32 - 1 - r / 2).collect();
            let w_off: Vec<i32> = (0..rows as i32).map(|r| 3 + r % 2).collect();
            let slots = pages * ps as usize;
            let row = (hk * d) as usize;
            let pool = KvPool {
                keys: gpu.zeros(slots as u32, row as u32, Dtype::Bf16),
                values: gpu.zeros(slots as u32, row as u32, Dtype::Bf16),
                bf16_keys: Tensor::ABSENT,
                bf16_values: Tensor::ABSENT,
                key_scales: Tensor::ABSENT,
                value_scales: Tensor::ABSENT,
                page_indices: Tensor::ABSENT,
                page_indptr: Tensor::ABSENT,
                last_page_lens: Tensor::ABSENT,
                row_valid: Tensor::ABSENT,
                env_min: Tensor::ABSENT,
                env_max: Tensor::ABSENT,
                has_envelopes: false,
                page_size: ps as i32,
                seq_stride: row as i64,
                head_stride: i64::from(d),
                layout: 0,
                scheme_byte: 0,
                block_size: 0,
                max_pages_per_request: 0,
                pages_in_batch: 0,
            };
            let mut q = gpu.zeros(rows, hq * d, Dtype::Bf16);
            if let Some((n, at)) = win {
                ctx.arm_stage(window(&gpu, n, at).ptr);
            }
            kernels_rocm::custom::qkv_fused_qknorm_rope_vnorm_write(
                &ctx,
                gpu.tensor(&packed_s, rows, width, Dtype::Bf16),
                gpu.column(&positions),
                gpu.tensor(&qw, 1, d, Dtype::Bf16),
                gpu.tensor(&kw, 1, d, Dtype::Bf16),
                eps,
                &pool,
                gpu.column(&w_page),
                gpu.column(&w_off),
                hk,
                d,
                theta,
                rotary,
                &mut q,
            )
            .unwrap();
            ctx.disarm_stage();
            let (q, k, v) = (gpu.read(&q), gpu.read(&pool.keys), gpu.read(&pool.values));

            let d_ = d as usize;
            let normed = |x: &[f32], w: Option<&[f32]>| -> Vec<f32> {
                let ss: f32 = x.iter().map(|v| v * v).sum();
                let inv = 1.0 / (ss / d as f32 + eps).sqrt();
                x.iter()
                    .enumerate()
                    .map(|(i, &v)| w.map_or(v * inv, |w| v * inv * w[i]))
                    .collect()
            };
            let roped = |x: Vec<f32>, pos: i32| -> Vec<f32> {
                let half = d_ / 2;
                let mut y = x.clone();
                for i in 0..half {
                    let (c, s) = if i < rotary as usize / 2 {
                        let f = f64::from(theta).powf(-2.0 * i as f64 / f64::from(d));
                        let a = f64::from(pos) * f;
                        (a.cos() as f32, a.sin() as f32)
                    } else {
                        (1.0, 0.0)
                    };
                    y[i] = x[i] * c - x[i + half] * s;
                    y[i + half] = x[i + half] * c + x[i] * s;
                }
                y
            };
            let live = live_rows(rows, win);
            let mut written = vec![false; slots];
            for r in 0..rows as usize {
                let src = &packed_s[r * width as usize..(r + 1) * width as usize];
                let tag = format!("qkv d{d} rows{rows} win {win:?} row {r}");
                let qrow = &q[r * hq as usize * d_..(r + 1) * hq as usize * d_];
                if !live.contains(&r) {
                    assert!(
                        qrow.iter().all(|&x| x == 0.0),
                        "{tag}: a dead row is untouched"
                    );
                    continue;
                }
                for h in 0..hq as usize {
                    let want = roped(normed(&src[h * d_..(h + 1) * d_], Some(&qw)), positions[r]);
                    close_abs(
                        &format!("{tag} q head {h}"),
                        &qrow[h * d_..(h + 1) * d_],
                        &want,
                        2e-2,
                    );
                }
                let slot = w_page[r] as usize * ps as usize + w_off[r] as usize;
                written[slot] = true;
                for h in 0..hk as usize {
                    let at = slot * row + h * d_;
                    let ks = &src[(hq as usize + h) * d_..(hq as usize + h + 1) * d_];
                    let vs = &src[((hq + hk) as usize + h) * d_..((hq + hk) as usize + h + 1) * d_];
                    close_abs(
                        &format!("{tag} k head {h}"),
                        &k[at..at + d_],
                        &roped(normed(ks, Some(&kw)), positions[r]),
                        2e-2,
                    );
                    close_bf16(
                        &format!("{tag} v head {h}"),
                        &v[at..at + d_],
                        &normed(vs, None),
                        1.0,
                    );
                }
            }
            for (slot, _) in written.iter().enumerate().filter(|(_, w)| !**w) {
                assert!(
                    k[slot * row..(slot + 1) * row].iter().all(|&x| x == 0.0)
                        && v[slot * row..(slot + 1) * row].iter().all(|&x| x == 0.0),
                    "qkv d{d} rows{rows} win {win:?}: slot {slot} written by no live row"
                );
            }
        }
    }
}
