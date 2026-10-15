#![cfg(feature = "cuda")]

//! `kernel.elemwise.hc_tile` swaps the hyper-connection projection for its
//! shared-memory tiles, the fold and the gates' pre-mix for their 16-byte
//! forms and the scale-free per-head norm for a warp a head: the fold and
//! the gates land bit for bit what the forms before land, the projection
//! the host dot, the norm within a bf16 ulp; on the staged window's rows and
//! nowhere else.

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::{hc, norm};
use kernels_cuda::tensor::Tensor;
use kernels_cuda::tuning::Tuning;

const M: usize = 4;

fn tuning(tile: bool) -> Tuning {
    Tuning {
        hc_tile: tile,
        ..Tuning::DEFAULT
    }
}

/// The projection's mixes, `tile` choosing the form; `window` is
/// `(live, base)` over `rows` staged rows.
fn project(
    tile: bool,
    rows: usize,
    mix_hc: usize,
    window: Option<(u32, u32)>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden = 4096;
    let fan = M * hidden;
    let planes = rows + window.map_or(0, |(_, base)| base as usize);
    let mut lcg = Lcg::seeded(0x7113 + rows as u64 + mix_hc as u64);
    let normed: Vec<f32> = (0..planes * fan).map(|_| lcg.unit()).collect();
    let w: Vec<f32> = (0..mix_hc * fan).map(|_| lcg.unit() * 0.01).collect();
    let mut gpu = Gpu::open();
    let n_at = gpu.up(&normed);
    let w_at = gpu.up(&w);
    let m_at = gpu.zeros(planes * mix_hc * 4);
    let ctx = gpu.ctx().with_tuning(tuning(tile));
    if let Some((live, base)) = window {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    let mut mixes = Tensor::new(m_at, rows as u32, mix_hc as u32, Dtype::F32);
    hc::project(
        &ctx,
        Tensor::new(n_at, rows as u32, fan as u32, Dtype::F32),
        Tensor::new(w_at, mix_hc as u32, fan as u32, Dtype::F32),
        M as u32,
        &mut mixes,
    )
    .expect("the projection fires");
    gpu.sync();
    (gpu.down(m_at, planes * mix_hc), normed, w)
}

fn check_project(rows: usize, mix_hc: usize, window: Option<(u32, u32)>) {
    let (got, normed, w) = project(true, rows, mix_hc, window);
    let fan = M * 4096;
    let (live, base) = window.unwrap_or((rows as u32, 0));
    for r in 0..rows + base as usize {
        let inside = r >= base as usize && r < (base + live) as usize;
        for o in 0..mix_hc {
            let g = f64::from(got[r * mix_hc + o]);
            if !inside {
                assert_eq!(g, 0.0, "row {r} lies outside the window and is left alone");
                continue;
            }
            let want: f64 = (0..fan)
                .map(|d| f64::from(normed[r * fan + d]) * f64::from(w[o * fan + d]))
                .sum();
            let mag: f64 = (0..fan)
                .map(|d| (f64::from(normed[r * fan + d]) * f64::from(w[o * fan + d])).abs())
                .sum();
            assert!(
                (g - want).abs() <= 1e-5 * mag.max(1e-3),
                "{rows} rows, {mix_hc} columns: row {r} column {o} is {g}, the dot {want}"
            );
        }
    }
}

#[test]
fn the_tiled_projection_answers_the_dot() {
    check_project(8, 24, None);
    check_project(70, 24, None);
    check_project(300, 24, None);
    check_project(3, 8, None);
    check_project(5, 4, None);
    check_project(40, 24, Some((29, 6)));
    check_project(130, 8, None);
    check_project(129, 4, None);
    check_project(200, 24, Some((150, 6)));
}

/// The fold, the gates and the scale-free norm under one form.
fn rows_pass(tile: bool, window: Option<(u32, u32, u32)>) -> Vec<Vec<u16>> {
    let (rows, live, base) = window.unwrap_or((37, 37, 0));
    let planes = (base + rows) as usize;
    let h = 4096usize;
    let head_dim = 512usize;
    let heads = 3usize;
    let mix_hc = 2 * M + M * M;
    let mut lcg = Lcg::seeded(0xf01d);
    let x: Vec<u16> = (0..planes * h).map(|_| to_bf16(lcg.unit())).collect();
    let streams: Vec<u16> = (0..planes * M * h).map(|_| to_bf16(lcg.unit())).collect();
    let post: Vec<f32> = (0..planes * M).map(|_| lcg.unit()).collect();
    let comb: Vec<f32> = (0..planes * M * M).map(|_| lcg.unit()).collect();
    let mixes: Vec<f32> = (0..planes * mix_hc).map(|_| lcg.unit() * 3.0).collect();
    let scale: Vec<f32> = vec![0.7, 1.3, 0.9];
    let bias: Vec<f32> = (0..mix_hc).map(|_| lcg.unit()).collect();
    let q: Vec<u16> = (0..planes * heads * head_dim)
        .map(|_| to_bf16(lcg.unit() * 4.0))
        .collect();

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let s_at = gpu.up(&streams);
    let post_at = gpu.up(&post);
    let comb_at = gpu.up(&comb);
    let mixes_at = gpu.up(&mixes);
    let scale_at = gpu.up(&scale);
    let bias_at = gpu.up(&bias);
    let q_at = gpu.up(&q);
    let folded = gpu.zeros(planes * M * h * 2);
    let gated = gpu.zeros(planes * h * 2);
    let post_out = gpu.zeros(planes * M * 4);
    let comb_out = gpu.zeros(planes * M * M * 4);
    let normed = gpu.zeros(planes * heads * head_dim * 2);
    let ctx = gpu.ctx().with_tuning(tuning(tile));
    if window.is_some() {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    let r = rows;
    hc::fold(
        &ctx,
        Tensor::new(x_at, r, h as u32, Dtype::Bf16),
        Tensor::new(s_at, r, (M * h) as u32, Dtype::Bf16),
        Tensor::new(post_at, r, M as u32, Dtype::F32),
        Tensor::new(comb_at, r, (M * M) as u32, Dtype::F32),
        &mut Tensor::new(folded, r, (M * h) as u32, Dtype::Bf16),
    )
    .expect("the fold fires");
    hc::gates(
        &ctx,
        Tensor::new(mixes_at, r, mix_hc as u32, Dtype::F32),
        Tensor::new(s_at, r, (M * h) as u32, Dtype::Bf16),
        Tensor::new(scale_at, 1, 3, Dtype::F32),
        Tensor::new(bias_at, 1, mix_hc as u32, Dtype::F32),
        M as u32,
        1e-6,
        2.0,
        20,
        true,
        &mut Tensor::new(gated, r, h as u32, Dtype::Bf16),
        &mut Tensor::new(post_out, r, M as u32, Dtype::F32),
        &mut Tensor::new(comb_out, r, (M * M) as u32, Dtype::F32),
    )
    .expect("the gates fire");
    norm::rmsnorm_no_scale(
        &ctx,
        Tensor::new(q_at, r, (heads * head_dim) as u32, Dtype::Bf16),
        head_dim as u32,
        1e-6,
        &mut Tensor::new(normed, r, (heads * head_dim) as u32, Dtype::Bf16),
    )
    .expect("the norm fires");
    gpu.sync();
    let f32_bits = |at: u64, n: usize| -> Vec<u16> {
        let v: Vec<f32> = gpu.down(at, n);
        v.iter()
            .flat_map(|f| {
                let b = f.to_bits();
                [b as u16, (b >> 16) as u16]
            })
            .collect()
    };
    vec![
        gpu.down(folded, planes * M * h),
        gpu.down(gated, planes * h),
        f32_bits(post_out, planes * M),
        f32_bits(comb_out, planes * M * M),
        gpu.down(normed, planes * heads * head_dim),
    ]
}

fn compare(window: Option<(u32, u32, u32)>) {
    let before = rows_pass(false, window);
    let after = rows_pass(true, window);
    for (k, name) in ["fold", "gates", "post", "combine"].iter().enumerate() {
        assert_eq!(
            before[k], after[k],
            "the {name} lands bit for bit ({window:?})"
        );
    }
    for (i, (a, b)) in before[4].iter().zip(&after[4]).enumerate() {
        let (a, b) = (from_bf16(*a), from_bf16(*b));
        assert!(
            (a - b).abs() <= a.abs() * (1.0 / 128.0) + 1e-6,
            "norm element {i}: {b} where the block form lands {a} ({window:?})"
        );
    }
}

#[test]
fn the_vector_forms_land_what_the_forms_before_land() {
    compare(None);
    compare(Some((24, 17, 5)));
}

/// The projection, gates, fold and q norm at DeepSeek-V4-Flash's shapes, the
/// forms before against the tiled ones: microseconds a call.
#[test]
#[ignore = "a timing probe, run by hand"]
fn the_hyper_connection_forms_time() {
    let h = 4096usize;
    let fan = M * h;
    let mix_hc = 2 * M + M * M;
    for rows in [8usize, 64, 16384] {
        let mut lcg = Lcg::seeded(1);
        let mut gpu = Gpu::open();
        let normed = gpu.up(&(0..rows * fan).map(|_| lcg.unit()).collect::<Vec<f32>>());
        let w = gpu.up(&(0..mix_hc * fan)
            .map(|_| lcg.unit() * 0.01)
            .collect::<Vec<f32>>());
        let streams = gpu.up(&(0..rows * fan)
            .map(|_| to_bf16(lcg.unit()))
            .collect::<Vec<u16>>());
        let x = gpu.up(&(0..rows * h)
            .map(|_| to_bf16(lcg.unit()))
            .collect::<Vec<u16>>());
        let mixes = gpu.up(&(0..rows * mix_hc).map(|_| lcg.unit()).collect::<Vec<f32>>());
        let post = gpu.up(&(0..rows * M).map(|_| lcg.unit()).collect::<Vec<f32>>());
        let comb = gpu.up(&(0..rows * M * M).map(|_| lcg.unit()).collect::<Vec<f32>>());
        let scale = gpu.up(&[0.7f32, 1.3, 0.9]);
        let bias = gpu.up(&vec![0.1f32; mix_hc]);
        let q = gpu.up(&(0..rows * 64 * 512)
            .map(|_| to_bf16(lcg.unit()))
            .collect::<Vec<u16>>());
        let out_m = gpu.zeros(rows * mix_hc * 4);
        let out_s = gpu.zeros(rows * fan * 2);
        let out_x = gpu.zeros(rows * h * 2);
        let out_p = gpu.zeros(rows * M * 4);
        let out_c = gpu.zeros(rows * M * M * 4);
        let out_q = gpu.zeros(rows * 64 * 512 * 2);
        let r = rows as u32;
        for tile in [false, true] {
            let ctx = gpu.ctx().with_tuning(tuning(tile));
            let time = |what: &str, f: &dyn Fn()| {
                for _ in 0..3 {
                    f();
                }
                gpu.sync();
                let iters = if rows > 1000 { 20 } else { 200 };
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    f();
                }
                gpu.sync();
                let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
                eprintln!("rows {rows:5} tile {tile:5} {what:8} {us:9.1} us");
            };
            time("project", &|| {
                hc::project(
                    &ctx,
                    Tensor::new(normed, r, fan as u32, Dtype::F32),
                    Tensor::new(w, mix_hc as u32, fan as u32, Dtype::F32),
                    M as u32,
                    &mut Tensor::new(out_m, r, mix_hc as u32, Dtype::F32),
                )
                .unwrap();
            });
            time("gates", &|| {
                hc::gates(
                    &ctx,
                    Tensor::new(mixes, r, mix_hc as u32, Dtype::F32),
                    Tensor::new(streams, r, fan as u32, Dtype::Bf16),
                    Tensor::new(scale, 1, 3, Dtype::F32),
                    Tensor::new(bias, 1, mix_hc as u32, Dtype::F32),
                    M as u32,
                    1e-6,
                    2.0,
                    20,
                    true,
                    &mut Tensor::new(out_x, r, h as u32, Dtype::Bf16),
                    &mut Tensor::new(out_p, r, M as u32, Dtype::F32),
                    &mut Tensor::new(out_c, r, (M * M) as u32, Dtype::F32),
                )
                .unwrap();
            });
            time("fold", &|| {
                hc::fold(
                    &ctx,
                    Tensor::new(x, r, h as u32, Dtype::Bf16),
                    Tensor::new(streams, r, fan as u32, Dtype::Bf16),
                    Tensor::new(post, r, M as u32, Dtype::F32),
                    Tensor::new(comb, r, (M * M) as u32, Dtype::F32),
                    &mut Tensor::new(out_s, r, fan as u32, Dtype::Bf16),
                )
                .unwrap();
            });
            time("qnorm", &|| {
                norm::rmsnorm_no_scale(
                    &ctx,
                    Tensor::new(q, r, 64 * 512, Dtype::Bf16),
                    512,
                    1e-6,
                    &mut Tensor::new(out_q, r, 64 * 512, Dtype::Bf16),
                )
                .unwrap();
            });
        }
    }
}
