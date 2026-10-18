//! The routed swiglu MLP kept by expert over hipBLASLt's grouped GEMM
//! kernels (`hip::lt_grouped`) against its steps. Run these on a ROCm
//! bump: they are what holds the argument contract.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::hip::lt_grouped;
use kernels_rocm::linear::moe;
use kernels_rocm::tuning::Tuning;

/// Distinct top-k routes over `experts`, every `dead`-th one -1.
fn routes(tokens: usize, top_k: usize, experts: usize, seed: u64, dead: usize) -> Vec<i32> {
    let keys = values(tokens * experts, seed, 1.0);
    let mut out = Vec::with_capacity(tokens * top_k);
    let mut ids: Vec<usize> = (0..experts).collect();
    for t in 0..tokens {
        let row = &keys[t * experts..(t + 1) * experts];
        ids.sort_by(|&a, &b| row[a].total_cmp(&row[b]));
        out.extend(ids[..top_k].iter().map(|&e| e as i32));
    }
    if dead > 0 {
        for i in (5..out.len()).step_by(dead) {
            out[i] = -1;
        }
    }
    // Under test, experts 3 and 9 take nothing: empty groups among the
    // full ones.
    if dead > 0 {
        for e in &mut out {
            if *e == 3 || *e == 9 {
                *e += 1;
            }
        }
    }
    out
}

#[test]
fn the_library_opens_at_a_known_version() {
    let Some(_gpu) = Gpu::open() else { return };
    let Ok(version) = lt_grouped::version() else {
        eprintln!("no hipBLASLt; skipping");
        return;
    };
    eprintln!("hipBLASLt version {version}");
    for dtype in [Dtype::Bf16, Dtype::F16] {
        match lt_grouped::library(dtype) {
            Ok(lib) => {
                assert!(!lib.solutions.is_empty());
                eprintln!("{dtype:?}: {} grouped solutions", lib.solutions.len());
            }
            Err(why) => eprintln!("{dtype:?}: not offered: {why}"),
        }
    }
}

/// One routed swiglu MLP's planes: the bank pair, the routes and weights,
/// and the steps' intermediate planes.
struct Mlp {
    p: moe::RoutedMlp,
    y: Tensor,
}

#[allow(clippy::too_many_arguments)]
fn mlp(
    gpu: &Gpu,
    dt: Dtype,
    tokens: usize,
    top_k: usize,
    experts: usize,
    hidden: usize,
    inter: usize,
    dead: usize,
) -> Mlp {
    let r = routes(tokens, top_k, experts, 11, dead);
    let count = r.len();
    let plane = |n: usize, seed: u64, scale: f32, rows: usize| {
        let v: Vec<f32> = values(n, seed, scale)
            .into_iter()
            .map(|v| if v == 0.0 { 1e-3 } else { v })
            .collect();
        gpu.tensor(&v, rows as u32, (n / rows) as u32, dt)
    };
    let x = plane(tokens * hidden, 1, 1.0, tokens);
    let gate_up = plane(experts * 2 * inter * hidden, 2, 0.05, experts);
    let down = plane(experts * hidden * inter, 3, 0.05, experts);
    let rt = gpu.upload(&r, tokens as u32, top_k as u32, Dtype::I32);
    let w: Vec<f32> = values(count, 4, 0.5)
        .iter()
        .map(|v| v.abs() + 0.1)
        .collect();
    let weights = gpu.tensor(&w, tokens as u32, top_k as u32, Dtype::F32);
    let packed = gpu.zeros(count as u32, (2 * inter) as u32, dt);
    let h = gpu.zeros(count as u32, inter as u32, dt);
    let routed = gpu.zeros(count as u32, hidden as u32, dt);
    let y = gpu.zeros(tokens as u32, hidden as u32, dt);
    Mlp {
        p: moe::RoutedMlp {
            x,
            routes: rt,
            weights,
            gate_up,
            gate_up_experts: moe::ExpertTable::RESIDENT,
            down,
            down_experts: moe::ExpertTable::RESIDENT,
            intermediate: inter as u32,
            limit: None,
            fan: top_k as u32,
            packed,
            h,
            routed,
        },
        y,
    }
}

/// The tuning a path's profile would name: the hipBLASLt sorted MLP from
/// `sorted` routes an expert, the native expert-ordered MLP from `ordered`,
/// at the tiles given.
fn tuning(sorted: u32, ordered: u32, tiles: Option<(u32, u32)>) -> Tuning {
    let mut t = Tuning::DEFAULT;
    t.moe_lt_grouped = sorted > 0;
    t.moe_lt_mlp_from = sorted;
    t.moe_ordered_from = ordered;
    if let Some((deep, shallow)) = tiles {
        t.moe_lt_tile_deep = deep;
        t.moe_lt_tile_shallow = shallow;
    }
    t
}

#[test]
fn the_sorted_routed_mlp_answers_the_steps() {
    let Some(gpu) = Gpu::open() else { return };
    if lt_grouped::library(Dtype::Bf16).is_err() {
        return;
    }
    let dt = Dtype::Bf16;
    let (tokens, hidden) = (600usize, 320usize);
    let m = mlp(&gpu, dt, tokens, 4, 8, hidden, 192, 9);
    let mut want = m.y;
    let mut have = gpu.zeros(tokens as u32, hidden as u32, dt);
    let steps = gpu.ctx().with_tuning(tuning(0, 0, None));
    moe::mlp_swiglu(&steps, &m.p, &mut want).unwrap();
    // The leg takes contractions from `lt_deep_k`; these are short of the
    // profile's.
    let sorted = gpu.ctx().with_tuning(Tuning {
        moe_lt_deep_k: 64,
        ..tuning(1, 0, None)
    });
    let fired = moe::lt_grouped_fires();
    moe::mlp_swiglu(&sorted, &m.p, &mut have).unwrap();
    assert_eq!(
        moe::lt_grouped_fires(),
        fired + 2,
        "the sorted MLP did not run"
    );
    let (want, have) = (gpu.read(&want), gpu.read(&have));
    for (i, (&h, &w)) in have.iter().zip(&want).enumerate() {
        assert!(
            (h - w).abs() <= w.abs() / 32.0 + 0.02,
            "y[{}][{}] = {h}, steps {w}",
            i / hidden,
            i % hidden
        );
    }
}
