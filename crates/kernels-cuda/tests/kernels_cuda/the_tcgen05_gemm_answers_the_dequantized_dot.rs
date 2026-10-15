#![cfg(feature = "cuda")]

use crate::common::tc5::{Side, side};
use crate::common::{Gpu, Lcg, from_bf16};
use kernels_cuda::linear::tc5::{Elem, Operand, Plan, gemm_with, supported};

#[derive(Clone, Copy, Debug)]
struct Case {
    act: (Elem, Option<usize>),
    w: (Elem, Option<usize>),
    m: usize,
    n: usize,
    k: usize,
    plan: Option<Plan>,
    alpha: f32,
}

fn check(case: Case) {
    let Case { m, n, k, .. } = case;
    let seed = 0x7c5 ^ ((m * 131 + n) * 7919 + k) as u64;
    let mut lcg = Lcg::seeded(seed);
    let mut bits = seed;
    let act = side(&mut lcg, &mut bits, case.act.0, case.act.1, m, k);
    let w = side(&mut lcg, &mut bits, case.w.0, case.w.1, n, k);

    let mut gpu = Gpu::open();
    let operand = |gpu: &mut Gpu, s: &Side, elem: Elem| -> Operand {
        let ptr = gpu.up(&s.bytes);
        match &s.scales {
            Some((atoms, _, gran)) => Operand::scaled(ptr, elem, gpu.up(atoms), *gran as u32),
            None => Operand::plain(ptr, elem),
        }
    };
    let a_op = operand(&mut gpu, &act, case.act.0);
    let w_op = operand(&mut gpu, &w, case.w.0);
    let tall = m + 1;
    let y = gpu.up(&vec![0x7fc0u16; tall * n]);
    let plan = case
        .plan
        .unwrap_or_else(|| Plan::choose(m as u32, n as u32, k as u32, 148));
    let ctx = gpu.ctx();
    gemm_with(
        &ctx,
        a_op,
        w_op,
        y,
        [m as u32, n as u32, k as u32],
        case.alpha,
        plan,
    )
    .unwrap_or_else(|e| panic!("{case:?}: {e}"));
    gpu.sync();
    // Split along K, a second launch lands the same bits: each split sums
    // into its own slab, added in split order.
    if SPLITS.load(std::sync::atomic::Ordering::Relaxed) {
        let first: Vec<u16> = gpu.down(y, tall * n);
        for _ in 0..3 {
            gemm_with(
                &ctx,
                a_op,
                w_op,
                y,
                [m as u32, n as u32, k as u32],
                case.alpha,
                plan,
            )
            .unwrap_or_else(|e| panic!("{case:?}: {e}"));
            gpu.sync();
            assert_eq!(
                gpu.down::<u16>(y, tall * n),
                first,
                "{case:?}: a split launch lands the same bits again"
            );
        }
    }
    let got: Vec<u16> = gpu.down(y, tall * n);

    let mut worst = 0f32;
    for r in 0..m {
        for c in 0..n {
            let (mut dot, mut mass) = (0f64, 0f64);
            for kk in 0..k {
                let t = f64::from(act.value(r, k, kk)) * f64::from(w.value(c, k, kk));
                dot += t;
                mass += t.abs();
            }
            let want = (dot * f64::from(case.alpha)) as f32;
            let slack = want.abs() / 128.0 + (mass * f64::from(case.alpha)) as f32 * 1e-5 + 1e-6;
            let g = from_bf16(got[r * n + c]);
            worst = worst.max((g - want).abs() / slack);
            assert!(
                (g - want).abs() <= slack,
                "{case:?} plan {plan:?}: y[{r}][{c}] = {g}, want {want} (slack {slack})"
            );
        }
    }
    assert!(
        got[m * n..].iter().all(|&v| v == 0x7fc0),
        "{case:?}: the row past m moved"
    );
    eprintln!("{case:?}: worst error {worst:.3} of the slack");
}

fn bf16(m: usize, n: usize, k: usize, plan: Option<Plan>) -> Case {
    Case {
        act: (Elem::Bf16, None),
        w: (Elem::Bf16, None),
        m,
        n,
        k,
        plan,
        alpha: 1.0,
    }
}

fn plan(swap: bool, bn: u32) -> Option<Plan> {
    Some(Plan {
        swap,
        pair: false,
        bn,
        stages: 0,
        splits: 0,
    })
}

#[test]
fn the_tcgen05_gemm_answers_the_dequantized_dot_every_case() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    the_bf16_kind_lands_the_dot_at_whole_and_ragged_tiles();
    the_bf16_kind_lands_what_cublas_lands_over_many_waves();
    the_wide_tiles_shared_columns_land_what_the_narrow_tile_lands();
    the_fp8_kind_lands_the_scaled_dot();
    the_deepseek_blocks_land_through_ue8m0();
    nvfp4_lands_its_sixteen_wide_scales_and_global_alpha();
    mxfp4_weights_meet_fp8_activations();
    a_split_launch_lands_the_same_bits_each_time();
}

/// The decode-width cases, split along K, run four times: each launch's
/// bits the first's.
static SPLITS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn a_split_launch_lands_the_same_bits_each_time() {
    SPLITS.store(true, std::sync::atomic::Ordering::Relaxed);
    let f8 = (Elem::E4m3, None);
    let ds = (Elem::E4m3, Some(128));
    check(scaled(f8, f8, [3, 256, 1024], None, 1.0));
    check(scaled(ds, ds, [1, 384, 1024], None, 1.0));
    check(scaled(ds, ds, [8, 1024, 4096], None, 1.0));
    check(scaled(ds, ds, [40, 256, 2048], None, 1.0));
    check(scaled(ds, ds, [64, 512, 8192], None, 1.0));
    check(bf16(16, 512, 4096, None));
    // Unswapped, split by hand.
    check(scaled(
        ds,
        ds,
        [128, 256, 4096],
        Some(Plan {
            splits: 4,
            ..plan(false, 128).unwrap()
        }),
        1.0,
    ));
    SPLITS.store(false, std::sync::atomic::Ordering::Relaxed);
}

fn scaled(
    act: (Elem, Option<usize>),
    w: (Elem, Option<usize>),
    [m, n, k]: [usize; 3],
    plan: Option<Plan>,
    alpha: f32,
) -> Case {
    Case {
        act,
        w,
        m,
        n,
        k,
        plan,
        alpha,
    }
}

/// Every scaled kind at more tiles than SMs on a 256-wide Q tile, where the
/// two accumulator stages share columns, lands what the 128-wide tile (whose
/// stages do not, and which the host reference holds above) lands.
fn the_wide_tiles_shared_columns_land_what_the_narrow_tile_lands() {
    let kinds = [
        ((Elem::E4m3, Some(128)), (Elem::E4m3, Some(128))),
        ((Elem::E2m1, Some(16)), (Elem::E2m1, Some(16))),
        ((Elem::E4m3, Some(128)), (Elem::E2m1, Some(32))),
    ];
    let (m, n, k) = (1536, 4096, 768);
    for (a, w) in kinds {
        let mut lcg = Lcg::seeded(0x3a);
        let mut bits = 0x3a;
        let act = side(&mut lcg, &mut bits, a.0, a.1, m, k);
        let wt = side(&mut lcg, &mut bits, w.0, w.1, n, k);
        let mut gpu = Gpu::open();
        let up = |gpu: &mut Gpu, s: &Side, elem: Elem| {
            let ptr = gpu.up(&s.bytes);
            let (atoms, _, gran) = s.scales.as_ref().expect("scaled");
            Operand::scaled(ptr, elem, gpu.up(atoms), *gran as u32)
        };
        let (a_op, w_op) = (up(&mut gpu, &act, a.0), up(&mut gpu, &wt, w.0));
        let plans = [plan(false, 128), plan(false, 256), pair(128), pair(256)];
        let outs: Vec<Vec<u16>> = plans
            .into_iter()
            .map(|plan| {
                let y = gpu.zeros(m * n * 2);
                let shape = [m as u32, n as u32, k as u32];
                gemm_with(&gpu.ctx(), a_op, w_op, y, shape, 1.0, plan.unwrap())
                    .unwrap_or_else(|e| panic!("{a:?} x {w:?} {plan:?}: {e}"));
                gpu.sync();
                gpu.down(y, m * n)
            })
            .collect();
        for (i, out) in outs.iter().enumerate().skip(1) {
            let differ = outs[0].iter().zip(out).filter(|(x, y)| x != y).count();
            assert_eq!(
                differ,
                0,
                "{a:?} x {w:?} under {:?}: {differ} of {} differ",
                plans[i],
                m * n
            );
        }
    }
}

fn pair(bn: u32) -> Option<Plan> {
    Some(Plan {
        swap: false,
        pair: true,
        bn,
        stages: 0,
        splits: 0,
    })
}

/// More tiles than SMs, so each CTA walks several and both accumulator
/// stages turn over; cuBLAS is the reference.
fn the_bf16_kind_lands_what_cublas_lands_over_many_waves() {
    use kernels_cuda::Tensor;
    use kernels_cuda::linear::gemm::matmul;
    use kernels_cuda::linear::tc5::gemm_bf16;

    for (m, n, k) in [(2048, 2304, 1024), (1000, 4096, 4104), (8, 7168, 2048)] {
        let mut lcg = Lcg::seeded(0x5eed ^ (m * n) as u64);
        let (a_raw, _) = lcg.row(m * k);
        let (w_raw, _) = lcg.row(n * k);
        let mut gpu = Gpu::open();
        let a = Tensor::new(gpu.up(&a_raw), m as u32, k as u32, dtype::Dtype::Bf16);
        let w = Tensor::new(gpu.up(&w_raw), n as u32, k as u32, dtype::Dtype::Bf16);
        let mut want = Tensor::new(gpu.zeros(m * n * 2), m as u32, n as u32, dtype::Dtype::Bf16);
        let got = gpu.zeros(m * n * 2);
        let ctx = gpu.ctx();
        matmul(&ctx, a, w, &mut want).expect("cuBLAS fires");
        gemm_bf16(&ctx, a.ptr, w.ptr, got, m as u32, n as u32, k as u32).expect("tc5 fires");
        gpu.sync();
        let want: Vec<u16> = gpu.down(want.ptr, m * n);
        let got: Vec<u16> = gpu.down(got, m * n);
        let mut off = 0usize;
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let (g, w) = (from_bf16(g), from_bf16(w));
            let slack = w.abs() / 64.0 + 2e-2;
            assert!(
                (g - w).abs() <= slack,
                "m={m} n={n} k={k}: y[{}][{}] = {g}, cuBLAS {w}",
                i / n,
                i % n
            );
            off += usize::from(g != w);
        }
        eprintln!(
            "m={m} n={n} k={k}: {off} of {} differ from cuBLAS by a rounding",
            m * n
        );
    }
}

fn the_fp8_kind_lands_the_scaled_dot() {
    let f8 = (Elem::E4m3, None);
    check(scaled(f8, f8, [128, 128, 128], plan(false, 128), 0.25));
    check(scaled(f8, f8, [200, 160, 400], plan(false, 32), 0.5));
    check(scaled(f8, f8, [3, 256, 1024], None, 1.0));
}

fn the_deepseek_blocks_land_through_ue8m0() {
    let f8 = (Elem::E4m3, Some(128));
    check(scaled(f8, f8, [128, 128, 512], plan(false, 128), 1.0));
    check(scaled(f8, f8, [300, 256, 1152], plan(false, 128), 1.0));
    check(scaled(f8, f8, [129, 192, 640], plan(false, 64), 1.0));
    check(scaled(f8, f8, [64, 512, 896], plan(false, 256), 1.0));
    check(scaled(f8, f8, [1, 384, 1024], None, 1.0));
    check(scaled(f8, f8, [40, 256, 2048], None, 1.0));
    check(scaled(f8, f8, [300, 256, 1152], pair(256), 1.0));
    check(scaled(f8, f8, [129, 192, 640], pair(64), 1.0));
    check(scaled(f8, f8, [256, 512, 8192], pair(256), 1.0));
    let mx = (Elem::E4m3, Some(32));
    check(scaled(mx, mx, [130, 128, 320], plan(false, 128), 1.0));
}

fn nvfp4_lands_its_sixteen_wide_scales_and_global_alpha() {
    let nv = (Elem::E2m1, Some(16));
    check(scaled(nv, nv, [128, 128, 256], plan(false, 128), 1.0));
    check(scaled(nv, nv, [140, 256, 1024], plan(false, 128), 0.03));
    check(scaled(nv, nv, [128, 128, 288], plan(false, 64), 1.0));
    check(scaled(nv, nv, [130, 512, 512], plan(false, 256), 1.0));
    check(scaled(nv, nv, [5, 384, 768], None, 2.0));
    check(scaled(nv, nv, [9, 256, 4096], None, 0.5));
    check(scaled(nv, nv, [140, 256, 1024], pair(256), 1.0));
    check(scaled(nv, nv, [300, 128, 288], pair(128), 1.0));
}

fn mxfp4_weights_meet_fp8_activations() {
    let w4 = (Elem::E2m1, Some(32));
    check(scaled(w4, w4, [128, 128, 128], plan(false, 128), 1.0));
    for act in [(Elem::E4m3, Some(128)), (Elem::E4m3, Some(32))] {
        check(scaled(act, w4, [128, 128, 256], plan(false, 128), 1.0));
        check(scaled(act, w4, [150, 192, 640], plan(false, 64), 1.0));
        check(scaled(act, w4, [7, 256, 1024], None, 1.0));
        check(scaled(act, w4, [150, 192, 640], pair(64), 1.0));
        check(scaled(act, w4, [130, 256, 2048], None, 1.0));
    }
}

fn the_bf16_kind_lands_the_dot_at_whole_and_ragged_tiles() {
    check(bf16(128, 128, 64, plan(false, 128)));
    check(bf16(128, 128, 256, plan(false, 128)));
    check(bf16(300, 200, 520, plan(false, 128)));
    check(bf16(256, 512, 1024, plan(false, 256)));
    check(bf16(130, 96, 136, plan(false, 64)));
    check(bf16(1, 384, 512, None));
    check(bf16(37, 1000, 264, None));
    check(bf16(16, 256, 4104, None));
    check(bf16(300, 200, 520, pair(128)));
    check(bf16(520, 512, 1024, pair(256)));
    check(bf16(257, 96, 136, pair(64)));
    check(bf16(200, 384, 2056, None));
}
