#![cfg(feature = "cuda")]

//! `kernel.elemwise.hc_tile` swaps the hyper-connection projection for its
//! shared-memory tiles, the fold and the gates' pre-mix for their 16-byte
//! forms and the scale-free per-head norm for a warp a head: the fold and
//! the gates land bit for bit what the forms before land, the projection
//! the host dot; on the staged window's rows and nowhere else.

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
    rows_pass_on(tuning(tile), window)
}

fn rows_pass_on(tuning: Tuning, window: Option<(u32, u32, u32)>) -> Vec<Vec<u16>> {
    let (rows, live, base) = window.unwrap_or((37, 37, 0));
    let planes = (base + rows) as usize;
    let h = 4096usize;
    let head_dim = 512usize;
    let heads = 64usize;
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
    let ctx = gpu.ctx().with_tuning(tuning);
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
    for (k, name) in ["fold", "gates", "post", "combine", "norm"]
        .iter()
        .enumerate()
    {
        assert_eq!(
            before[k], after[k],
            "the {name} lands bit for bit ({window:?})"
        );
    }
}

#[test]
fn the_vector_forms_land_what_the_forms_before_land() {
    compare(None);
    compare(Some((24, 17, 5)));
}

/// A prefill's rows under `kernel.elemwise.hc_mma` (the fold a block a row)
/// land what the vector forms land, bit for bit, whole and windowed.
#[test]
fn a_prefills_rows_fold_alike_a_block_a_row() {
    for window in [(200, 200, 0), (300, 170, 9)] {
        let before = rows_pass(true, Some(window));
        let after = rows_pass_on(
            Tuning {
                hc_mma: true,
                ..tuning(true)
            },
            Some(window),
        );
        assert_eq!(
            before[0], after[0],
            "the fold lands bit for bit ({window:?})"
        );
    }
}
/// The norm and the projection in one pass over the bf16 streams against a
/// host norm and dot, at decode and prefill row counts and on a window.
fn check_prenorm(rows: usize, mix_hc: usize, window: Option<(u32, u32)>) {
    check_prenorm_on(rows, mix_hc, window, false);
    // The tensor-core form (`kernel.elemwise.hc_mma`), a prefill's rows.
    if rows >= 128 {
        check_prenorm_on(rows, mix_hc, window, true);
    }
}

fn check_prenorm_on(rows: usize, mix_hc: usize, window: Option<(u32, u32)>, mma: bool) {
    let fan = M * 4096;
    let eps = 1e-6f32;
    let planes = rows + window.map_or(0, |(_, base)| base as usize);
    let mut lcg = Lcg::seeded(0x9e7 + rows as u64 + mix_hc as u64);
    let x: Vec<u16> = (0..planes * fan)
        .map(|_| to_bf16(lcg.unit() * 3.0))
        .collect();
    let w: Vec<f32> = (0..mix_hc * fan).map(|_| lcg.unit() * 0.01).collect();
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let w_at = gpu.up(&w);
    let m_at = gpu.zeros(planes * mix_hc * 4);
    let ctx = gpu.ctx().with_tuning(Tuning {
        hc_mma: mma,
        ..tuning(true)
    });
    if let Some((live, base)) = window {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    let landed = hc::prenorm_project(
        &ctx,
        Tensor::new(x_at, rows as u32, fan as u32, Dtype::Bf16),
        eps,
        Tensor::new(w_at, mix_hc as u32, fan as u32, Dtype::F32),
        M as u32,
        &mut Tensor::new(m_at, rows as u32, mix_hc as u32, Dtype::F32),
    )
    .expect("the fused projection fires");
    assert!(landed, "a 16384-wide row of {mix_hc} columns is the tile's");
    gpu.sync();
    let got: Vec<f32> = gpu.down(m_at, planes * mix_hc);
    let (live, base) = window.unwrap_or((rows as u32, 0));
    for r in 0..planes {
        let inside = r >= base as usize && r < (base + live) as usize;
        let row: Vec<f64> = (0..fan)
            .map(|d| f64::from(from_bf16(x[r * fan + d])))
            .collect();
        let s = 1.0 / (row.iter().map(|v| v * v).sum::<f64>() / fan as f64 + f64::from(eps)).sqrt();
        for o in 0..mix_hc {
            let g = f64::from(got[r * mix_hc + o]);
            if !inside {
                assert_eq!(g, 0.0, "row {r} lies outside the window and is left alone");
                continue;
            }
            let want: f64 = (0..fan)
                .map(|d| row[d] * s * f64::from(w[o * fan + d]))
                .sum();
            let mag: f64 = (0..fan)
                .map(|d| (row[d] * s * f64::from(w[o * fan + d])).abs())
                .sum();
            assert!(
                (g - want).abs() <= 1e-5 * mag.max(1e-3),
                "{rows} rows, {mix_hc} columns, mma {mma}: row {r} column {o} is {g}, the normed dot {want}"
            );
        }
    }
}

#[test]
fn the_fused_norm_and_projection_answer_the_host() {
    check_prenorm(1, 24, None);
    check_prenorm(8, 24, None);
    check_prenorm(70, 24, Some((50, 3)));
    check_prenorm(300, 24, None);
    check_prenorm(130, 8, None);
    check_prenorm(129, 4, None);
    check_prenorm(200, 24, Some((150, 6)));
}

/// The gates and the RMS norm over their pre-mixed row in one launch land
/// what the pair lands, bit for bit: post, combine and the normed row, on
/// the staged window's rows and nowhere else.
fn gates_norm_pass(fused: bool, window: Option<(u32, u32, u32)>) -> Vec<Vec<u16>> {
    let (rows, live, base) = window.unwrap_or((19, 19, 0));
    let planes = (base + rows) as usize;
    let h = 4096usize;
    let mix_hc = 2 * M + M * M;
    let mut lcg = Lcg::seeded(0x9a7e5);
    let mixes: Vec<f32> = (0..planes * mix_hc).map(|_| lcg.unit() * 3.0).collect();
    let streams: Vec<u16> = (0..planes * M * h)
        .map(|_| to_bf16(lcg.unit() * 2.0))
        .collect();
    let weight: Vec<u16> = (0..h).map(|_| to_bf16(1.0 + lcg.unit() * 0.5)).collect();
    let mut gpu = Gpu::open();
    let mixes_at = gpu.up(&mixes);
    let streams_at = gpu.up(&streams);
    let scale_at = gpu.up(&[0.7f32, 1.3, 0.9]);
    let base_v: Vec<f32> = (0..mix_hc).map(|_| lcg.unit()).collect();
    let base_at = gpu.up(&base_v);
    let w_at = gpu.up(&weight);
    let x_at = gpu.zeros(planes * h * 2);
    let y_at = gpu.zeros(planes * h * 2);
    let post_at = gpu.zeros(planes * M * 4);
    let comb_at = gpu.zeros(planes * M * M * 4);
    let ctx = gpu.ctx().with_tuning(tuning(true));
    if window.is_some() {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    let r = rows;
    let mixes_t = Tensor::new(mixes_at, r, mix_hc as u32, Dtype::F32);
    let streams_t = Tensor::new(streams_at, r, (M * h) as u32, Dtype::Bf16);
    let scale_t = Tensor::new(scale_at, 1, 3, Dtype::F32);
    let base_t = Tensor::new(base_at, 1, mix_hc as u32, Dtype::F32);
    let w_t = Tensor::new(w_at, h as u32, 1, Dtype::Bf16);
    let mut post = Tensor::new(post_at, r, M as u32, Dtype::F32);
    let mut comb = Tensor::new(comb_at, r, (M * M) as u32, Dtype::F32);
    let mut y = Tensor::new(y_at, r, h as u32, Dtype::Bf16);
    if fused {
        let gate = hc::Gate {
            scale: scale_t,
            base: base_t,
            gate_eps: 1e-6,
            alpha: 2.0,
            sinkhorn: 20,
        };
        let landed = hc::gates_norm(
            &ctx, mixes_t, streams_t, M as u32, gate, w_t, 1e-6, &mut post, &mut comb, &mut y,
        )
        .expect("the fused gates fire");
        assert!(landed, "a 4096-wide row of four streams is the kernel's");
    } else {
        let mut x = Tensor::new(x_at, r, h as u32, Dtype::Bf16);
        hc::gates(
            &ctx, mixes_t, streams_t, scale_t, base_t, M as u32, 1e-6, 2.0, 20, true, &mut x,
            &mut post, &mut comb,
        )
        .expect("the gates fire");
        norm::rmsnorm(&ctx, x, w_t, 1e-6, &mut y).expect("the norm fires");
    }
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
        f32_bits(post_at, planes * M),
        f32_bits(comb_at, planes * M * M),
        gpu.down(y_at, planes * h),
    ]
}

#[test]
fn the_fused_gates_and_norm_land_what_the_pair_lands() {
    for window in [None, Some((24, 17, 5))] {
        let pair = gates_norm_pass(false, window);
        let fused = gates_norm_pass(true, window);
        for (k, name) in ["post", "combine", "normed row"].iter().enumerate() {
            assert_eq!(
                pair[k], fused[k],
                "the {name} lands bit for bit ({window:?})"
            );
        }
    }
}

/// The whole pre-mix in two launches lands what the fused projection, the
/// gates and the RMS norm land one after another, bit for bit: at a
/// decode's rows (the gates summing the projection's slices), at a
/// prefill's, and on a window.
fn pre_pass(whole: bool, rows: u32, window: Option<(u32, u32)>) -> Vec<Vec<u16>> {
    let (live, base) = window.unwrap_or((rows, 0));
    let planes = (base + rows) as usize;
    let h = 4096usize;
    let fan = M * h;
    let mix_hc = 2 * M + M * M;
    let mut lcg = Lcg::seeded(0x5e7 + u64::from(rows));
    let streams: Vec<u16> = (0..planes * fan)
        .map(|_| to_bf16(lcg.unit() * 2.0))
        .collect();
    let plane: Vec<f32> = (0..mix_hc * fan).map(|_| lcg.unit() * 0.02).collect();
    let base_v: Vec<f32> = (0..mix_hc).map(|_| lcg.unit()).collect();
    let weight: Vec<u16> = (0..h).map(|_| to_bf16(1.0 + lcg.unit() * 0.5)).collect();
    let mut gpu = Gpu::open();
    let streams_at = gpu.up(&streams);
    let plane_at = gpu.up(&plane);
    let scale_at = gpu.up(&[0.7f32, 1.3, 0.9]);
    let base_at = gpu.up(&base_v);
    let w_at = gpu.up(&weight);
    let mixes_at = gpu.zeros(planes * mix_hc * 4);
    let x_at = gpu.zeros(planes * h * 2);
    let y_at = gpu.zeros(planes * h * 2);
    let post_at = gpu.zeros(planes * M * 4);
    let comb_at = gpu.zeros(planes * M * M * 4);
    let ctx = gpu.ctx().with_tuning(tuning(true));
    if window.is_some() {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    let streams_t = Tensor::new(streams_at, rows, fan as u32, Dtype::Bf16);
    let plane_t = Tensor::new(plane_at, mix_hc as u32, fan as u32, Dtype::F32);
    let mut mixes = Tensor::new(mixes_at, rows, mix_hc as u32, Dtype::F32);
    let gate = hc::Gate {
        scale: Tensor::new(scale_at, 1, 3, Dtype::F32),
        base: Tensor::new(base_at, 1, mix_hc as u32, Dtype::F32),
        gate_eps: 1e-6,
        alpha: 2.0,
        sinkhorn: 20,
    };
    let w_t = Tensor::new(w_at, h as u32, 1, Dtype::Bf16);
    let mut post = Tensor::new(post_at, rows, M as u32, Dtype::F32);
    let mut comb = Tensor::new(comb_at, rows, (M * M) as u32, Dtype::F32);
    let mut y = Tensor::new(y_at, rows, h as u32, Dtype::Bf16);
    if whole {
        let landed = hc::pre(
            &ctx, streams_t, 1e-6, plane_t, M as u32, &mut mixes, gate, w_t, 1e-6, None, &mut post,
            &mut comb, &mut y,
        )
        .expect("the pre-mix fires");
        assert!(landed, "a 4096-wide row of four streams is the kernels'");
    } else {
        assert!(
            hc::prenorm_project(&ctx, streams_t, 1e-6, plane_t, M as u32, &mut mixes)
                .expect("the projection fires")
        );
        let mut x = Tensor::new(x_at, rows, h as u32, Dtype::Bf16);
        hc::gates(
            &ctx, mixes, streams_t, gate.scale, gate.base, M as u32, 1e-6, 2.0, 20, true, &mut x,
            &mut post, &mut comb,
        )
        .expect("the gates fire");
        norm::rmsnorm(&ctx, x, w_t, 1e-6, &mut y).expect("the norm fires");
    }
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
        f32_bits(post_at, planes * M),
        f32_bits(comb_at, planes * M * M),
        gpu.down(y_at, planes * h),
    ]
}

#[test]
fn the_whole_pre_mix_lands_what_its_parts_land() {
    for (rows, window) in [
        (1, None),
        (8, None),
        (64, None),
        (300, None),
        (40, Some((29, 6))),
        (200, Some((150, 6))),
    ] {
        let parts = pre_pass(false, rows, window);
        let whole = pre_pass(true, rows, window);
        for (k, name) in ["post", "combine", "normed row"].iter().enumerate() {
            assert_eq!(
                parts[k], whole[k],
                "the {name} lands bit for bit ({rows} rows, {window:?})"
            );
        }
    }
}

/// The pre-mix's e4m3 operand planes hold what the GEMM's quantize lands of
/// its normed row, byte for byte: codes and ue8m0 atoms, at a decode's rows
/// and on a window.
#[test]
fn the_pre_mix_lands_the_operand_its_gemm_would_quantize() {
    use kernels_cuda::linear::act_quant::{Format, Source, held_planes, quantize};
    use kernels_cuda::linear::tc5::atom_bytes;
    for (rows, window) in [
        (1u32, None),
        (8, None),
        (64, None),
        (40, Some((29u32, 6u32))),
    ] {
        let (live, base) = window.unwrap_or((rows, 0));
        let planes = (base + rows) as usize;
        let h = 4096usize;
        let fan = M * h;
        let mix_hc = 2 * M + M * M;
        let mut lcg = Lcg::seeded(0xe4 + u64::from(rows));
        let streams: Vec<u16> = (0..planes * fan)
            .map(|_| to_bf16(lcg.unit() * 2.0))
            .collect();
        let plane: Vec<f32> = (0..mix_hc * fan).map(|_| lcg.unit() * 0.02).collect();
        let base_v: Vec<f32> = (0..mix_hc).map(|_| lcg.unit()).collect();
        let weight: Vec<u16> = (0..h).map(|_| to_bf16(1.0 + lcg.unit() * 0.5)).collect();
        let mut gpu = Gpu::open();
        let streams_at = gpu.up(&streams);
        let plane_at = gpu.up(&plane);
        let scale_at = gpu.up(&[0.7f32, 1.3, 0.9]);
        let base_at = gpu.up(&base_v);
        let w_at = gpu.up(&weight);
        let mixes_at = gpu.zeros(planes * mix_hc * 4);
        let y_at = gpu.zeros(planes * h * 2);
        let post_at = gpu.zeros(planes * M * 4);
        let comb_at = gpu.zeros(planes * M * M * 4);
        let ctx = gpu.ctx().with_tuning(tuning(true));
        let total = base + rows;
        let held = held_planes(&ctx, "test.held", total, h as u32).expect("the held planes");
        if window.is_some() {
            let win_at = gpu.up(&[live, base, 0u32, 0u32]);
            ctx.arm_stage(win_at);
        }
        let gate = hc::Gate {
            scale: Tensor::new(scale_at, 1, 3, Dtype::F32),
            base: Tensor::new(base_at, 1, mix_hc as u32, Dtype::F32),
            gate_eps: 1e-6,
            alpha: 2.0,
            sinkhorn: 20,
        };
        let landed = hc::pre(
            &ctx,
            Tensor::new(streams_at, rows, fan as u32, Dtype::Bf16),
            1e-6,
            Tensor::new(plane_at, mix_hc as u32, fan as u32, Dtype::F32),
            M as u32,
            &mut Tensor::new(mixes_at, rows, mix_hc as u32, Dtype::F32),
            gate,
            Tensor::new(w_at, h as u32, 1, Dtype::Bf16),
            1e-6,
            Some(held),
            &mut Tensor::new(post_at, rows, M as u32, Dtype::F32),
            &mut Tensor::new(comb_at, rows, (M * M) as u32, Dtype::F32),
            &mut Tensor::new(y_at, rows, h as u32, Dtype::Bf16),
        )
        .expect("the pre-mix fires");
        assert!(landed);
        ctx.disarm_stage();
        // The quantizer over every plane row of the landed y.
        let fmt = Format::E4m3 { gran: 128 };
        let codes_at = gpu.zeros(planes * h);
        let atoms_n = atom_bytes(planes, h, 128);
        let atoms_at = gpu.zeros(atoms_n);
        quantize(
            &ctx,
            "test.quantize",
            fmt,
            Source::rows(Tensor::new(y_at, total, h as u32, Dtype::Bf16), None),
            codes_at,
            atoms_at,
            atoms_n,
        )
        .expect("the quantize fires");
        gpu.sync();
        let want_codes: Vec<u8> = gpu.down(codes_at, planes * h);
        let got_codes: Vec<u8> = gpu.down(held[0], planes * h);
        let want_atoms: Vec<u8> = gpu.down(atoms_at, atoms_n);
        let got_atoms: Vec<u8> = gpu.down(held[1], atoms_n);
        let live_rows = base as usize..(base + live) as usize;
        for r in live_rows.clone() {
            assert_eq!(
                got_codes[r * h..(r + 1) * h],
                want_codes[r * h..(r + 1) * h],
                "row {r}'s codes ({rows} rows, {window:?})"
            );
        }
        // Atoms of the live rows: compare where every plane row is live.
        if window.is_none() {
            assert_eq!(got_atoms, want_atoms, "the atoms ({rows} rows)");
        }
    }
}
/// A prefill's pre-mix norm landing its e4m3 1x128 operand: the bf16 row the
/// plain norm lands (to its reduction order's last bit), and the codes and
/// atoms act_quant_e4m3 makes of that row.
#[test]
fn a_prefills_norm_lands_its_gemms_fp8_operand() {
    use kernels_cuda::linear::tc5::{atom_bytes, atom_offset};
    let (rows, h) = (300usize, 4096usize);
    let mut lcg = Lcg::seeded(0x0e17);
    let x: Vec<u16> = (0..rows * h).map(|_| to_bf16(lcg.unit() * 3.0)).collect();
    let w: Vec<u16> = (0..h).map(|_| to_bf16(0.5 + lcg.unit())).collect();
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let w_at = gpu.up(&w);
    let y_at = gpu.zeros(rows * h * 2);
    let r_at = gpu.zeros(rows * h * 2);
    let c_at = gpu.zeros(rows * h);
    let a_at = gpu.zeros(atom_bytes(rows, h, 128));
    let ctx = gpu.ctx();
    let xt = Tensor::new(x_at, rows as u32, h as u32, Dtype::Bf16);
    let wt = Tensor::new(w_at, h as u32, 1, Dtype::Bf16);
    assert!(
        hc::rmsnorm_emit(
            &ctx,
            xt,
            wt,
            1e-6,
            &mut Tensor::new(y_at, rows as u32, h as u32, Dtype::Bf16),
            [c_at, a_at]
        )
        .expect("the emitting norm fires")
    );
    norm::rmsnorm(
        &ctx,
        xt,
        wt,
        1e-6,
        &mut Tensor::new(r_at, rows as u32, h as u32, Dtype::Bf16),
    )
    .expect("the plain norm fires");
    gpu.sync();
    let y: Vec<u16> = gpu.down(y_at, rows * h);
    let r: Vec<u16> = gpu.down(r_at, rows * h);
    let codes: Vec<u8> = gpu.down(c_at, rows * h);
    let atoms: Vec<u8> = gpu.down(a_at, atom_bytes(rows, h, 128));
    for (i, (&a, &b)) in y.iter().zip(&r).enumerate() {
        assert!(
            a.abs_diff(b) <= 1,
            "element {i}: {a:#x} against the plain norm's {b:#x}"
        );
    }
    let e4m3 = |c: u8| -> f32 {
        let sign = if c & 0x80 != 0 { -1.0 } else { 1.0 };
        let exp = i32::from((c >> 3) & 0xf);
        let man = f32::from(c & 7);
        sign * if exp == 0 {
            man / 8.0 * 2f32.powi(-6)
        } else {
            (1.0 + man / 8.0) * 2f32.powi(exp - 7)
        }
    };
    for row in 0..rows {
        for b in 0..h / 128 {
            let vals: Vec<f32> = (0..128)
                .map(|i| from_bf16(y[row * h + b * 128 + i]))
                .collect();
            let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-4);
            let e = (amax / 448.0).log2().ceil() as i32;
            assert_eq!(
                atoms[atom_offset(row, b, h / 128)],
                (e + 127) as u8,
                "row {row} block {b}"
            );
            let s = 2f32.powi(e);
            for (i, v) in vals.iter().enumerate() {
                let c = codes[row * h + b * 128 + i];
                let got = e4m3(c);
                // The nearest code: no code lies nearer v / s.
                let err = (got - v / s).abs();
                let best = (0..=255u8)
                    .filter(|c| c & 0x7f != 0x7f)
                    .map(|c| (e4m3(c) - v / s.clamp(f32::MIN, f32::MAX)).abs())
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    err <= best || (v / s).abs() > 448.0,
                    "row {row} block {b} element {i}: code {c:#x}"
                );
            }
        }
    }
}
