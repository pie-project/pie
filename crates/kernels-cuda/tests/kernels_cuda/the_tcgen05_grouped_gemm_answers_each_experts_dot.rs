#![cfg(feature = "cuda")]

use crate::common::tc5::{Side, random, side};
use crate::common::{Gpu, Lcg, from_bf16};
use kernels_cuda::linear::tc5::{
    Elem, GROUP_ALIGN, Grouped, Groups, Operand, Plan, gemm_with, grouped_gemm, grouped_gemm_with,
    masked_atom_bytes, supported,
};

type Spec = (Elem, Option<usize>);

const NAN: u16 = 0x7fc0;

/// The rows each expert is routed: none, a few, a block and a half, ...
fn counts(experts: usize, cap: usize, seed: u64) -> Vec<usize> {
    let mut bits = seed;
    (0..experts)
        .map(|e| match e % 4 {
            0 => 0,
            1 => cap.min(1 + random(&mut bits) as usize % 7),
            2 => cap.min(130 + random(&mut bits) as usize % 60),
            _ => random(&mut bits) as usize % (cap + 1),
        })
        .collect()
}

struct Bank {
    w: Side,
    alpha_e: Vec<f32>,
}

fn bank(lcg: &mut Lcg, bits: &mut u64, spec: Spec, experts: usize, n: usize, k: usize) -> Bank {
    let w = side(lcg, bits, spec.0, spec.1, experts * n, k);
    let alpha_e = (0..experts).map(|e| 0.5 + 0.25 * (e % 5) as f32).collect();
    Bank { w, alpha_e }
}

/// Uploads an operand whose scale atoms sit at `atoms`.
fn operand(gpu: &mut Gpu, s: &Side, elem: Elem, atoms: Option<Vec<u8>>) -> Operand {
    let ptr = gpu.up(&s.bytes);
    match (&s.scales, atoms) {
        (Some((_, _, gran)), Some(atoms)) => {
            Operand::scaled(ptr, elem, gpu.up(&atoms), *gran as u32)
        }
        _ => Operand::plain(ptr, elem),
    }
}

/// Checks row `r` of `y` against act row `a_row` times expert `e`'s weight.
#[allow(clippy::too_many_arguments)]
fn check_row(
    what: &str,
    got: &[u16],
    r: usize,
    act: &Side,
    a_row: usize,
    b: &Bank,
    e: usize,
    [n, k]: [usize; 2],
    alpha: f32,
    worst: &mut f32,
) {
    let alpha = alpha * b.alpha_e[e];
    for c in 0..n {
        let (mut dot, mut mass) = (0f64, 0f64);
        for kk in 0..k {
            let t = f64::from(act.value(a_row, k, kk)) * f64::from(b.w.value(e * n + c, k, kk));
            dot += t;
            mass += t.abs();
        }
        let want = (dot * f64::from(alpha)) as f32;
        let slack = want.abs() / 128.0 + (mass * f64::from(alpha)) as f32 * 1e-5 + 1e-6;
        let g = from_bf16(got[r * n + c]);
        *worst = worst.max((g - want).abs() / slack);
        assert!(
            (g - want).abs() <= slack,
            "{what}: y[{r}][{c}] (expert {e}) = {g}, want {want} (slack {slack})"
        );
    }
}

fn contiguous(a: Spec, w: Spec, experts: usize, [n, k]: [usize; 2], plan: Option<Plan>) {
    let what = format!("contiguous {a:?} x {w:?} e={experts} n={n} k={k} {plan:?}");
    let mut lcg = Lcg::seeded(0xc0 ^ (n * k + experts) as u64);
    let mut bits = 0xc0 ^ k as u64;
    let b = bank(&mut lcg, &mut bits, w, experts, n, k);
    let counts = counts(experts, 300, n as u64);
    let align = GROUP_ALIGN as usize;
    let mut offsets = vec![0usize];
    for &c in &counts {
        offsets.push(offsets.last().unwrap() + c.next_multiple_of(align));
    }
    let rows = offsets[experts] + align;
    let act = side(&mut lcg, &mut bits, a.0, a.1, rows, k);

    let mut gpu = Gpu::open();
    let a_op = operand(
        &mut gpu,
        &act,
        a.0,
        a.1.map(|_| act.atoms_at(k, rows, |r| r)),
    );
    let w_op = operand(
        &mut gpu,
        &b.w,
        w.0,
        w.1.map(|_| b.w.atoms_at(k, experts * n, |r| r)),
    );
    let offs: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
    let g = Grouped {
        experts: experts as u32,
        groups: Groups::Contiguous {
            offsets: gpu.up(&offs),
            rows: rows as u32,
        },
        alpha_e: gpu.up(&b.alpha_e),
    };
    let y = gpu.up(&vec![NAN; rows * n]);
    let routed = counts.iter().sum::<usize>() as u32;
    let shape = [n as u32, k as u32];
    let ctx = gpu.ctx().with_tuning(kernels_cuda::tuning::Tuning {
        gemm_nvfp4_k128: K128.load(std::sync::atomic::Ordering::Relaxed),
        ..kernels_cuda::tuning::Tuning::DEFAULT
    });
    match plan {
        Some(plan) => grouped_gemm_with(&ctx, a_op, w_op, y, g, shape, 0.75, plan),
        None => grouped_gemm(&ctx, a_op, w_op, y, g, shape, 0.75, routed),
    }
    .unwrap_or_else(|e| panic!("{what}: {e}"));
    gpu.sync();
    let got: Vec<u16> = gpu.down(y, rows * n);
    let mut worst = 0f32;
    for e in 0..experts {
        for i in 0..counts[e] {
            let r = offsets[e] + i;
            check_row(&what, &got, r, &act, r, &b, e, [n, k], 0.75, &mut worst);
        }
    }
    assert!(
        got[offsets[experts] * n..].iter().all(|&v| v == NAN),
        "{what}: a row past the routed ones moved"
    );
    eprintln!("{what}: worst error {worst:.3} of the slack");
}

fn masked(a: Spec, w: Spec, experts: usize, m_max: usize, [n, k]: [usize; 2], splits: u32) {
    let what =
        format!("masked {a:?} x {w:?} e={experts} m_max={m_max} n={n} k={k} splits={splits}");
    let mut lcg = Lcg::seeded(0x3a5 ^ (n * k + m_max) as u64);
    let mut bits = 0x3a5 ^ k as u64;
    let b = bank(&mut lcg, &mut bits, w, experts, n, k);
    let counts = counts(experts, m_max, k as u64);
    let rows = experts * m_max;
    let act = side(&mut lcg, &mut bits, a.0, a.1, rows, k);

    let mut gpu = Gpu::open();
    let pad = m_max.next_multiple_of(128);
    let atoms = a.1.map(|gran| {
        let atoms = act.atoms_at(k, experts * pad, |r| r / m_max * pad + r % m_max);
        assert_eq!(atoms.len(), masked_atom_bytes(experts, m_max, k, gran));
        atoms
    });
    let a_op = operand(&mut gpu, &act, a.0, atoms);
    let w_op = operand(
        &mut gpu,
        &b.w,
        w.0,
        w.1.map(|_| b.w.atoms_at(k, experts * n, |r| r)),
    );
    let cnt: Vec<i32> = counts.iter().map(|&c| c as i32).collect();
    let g = Grouped {
        experts: experts as u32,
        groups: Groups::Masked {
            counts: gpu.up(&cnt),
            m_max: m_max as u32,
        },
        alpha_e: gpu.up(&b.alpha_e),
    };
    let y = gpu.up(&vec![NAN; rows * n]);
    let plan = Plan {
        splits,
        ..Plan::grouped(
            &g,
            n as u32,
            k as u32,
            counts.iter().sum::<usize>() as u32,
            148,
        )
    };
    grouped_gemm_with(
        &gpu.ctx(),
        a_op,
        w_op,
        y,
        g,
        [n as u32, k as u32],
        1.25,
        plan,
    )
    .unwrap_or_else(|e| panic!("{what}: {e}"));
    gpu.sync();
    let got: Vec<u16> = gpu.down(y, rows * n);
    let mut worst = 0f32;
    for e in 0..experts {
        for i in 0..m_max {
            let r = e * m_max + i;
            if i < counts[e] {
                check_row(&what, &got, r, &act, r, &b, e, [n, k], 1.25, &mut worst);
            } else {
                assert!(
                    got[r * n..(r + 1) * n].iter().all(|&v| v == NAN),
                    "{what}: expert {e}'s row {i} past its count {} moved",
                    counts[e]
                );
            }
        }
    }
    eprintln!("{what}: worst error {worst:.3} of the slack");
}

/// A contiguous grouped GEMM lands what a loop of dense GEMMs over each
/// expert's rows and weight lands.
fn the_contiguous_group_lands_what_a_loop_of_dense_gemms_lands() {
    let (experts, n, k) = (6usize, 512usize, 1024usize);
    let spec = (Elem::E4m3, Some(128));
    let mut lcg = Lcg::seeded(0x100);
    let mut bits = 0x100;
    let b = bank(&mut lcg, &mut bits, spec, experts, n, k);
    let counts = [130usize, 0, 7, 256, 64, 300];
    let mut offsets = vec![0usize];
    for &c in &counts {
        offsets.push(offsets.last().unwrap() + c.next_multiple_of(128));
    }
    let rows = offsets[experts];
    let act = side(&mut lcg, &mut bits, spec.0, spec.1, rows, k);
    let mut gpu = Gpu::open();
    let a_op = operand(&mut gpu, &act, spec.0, Some(act.atoms_at(k, rows, |r| r)));
    let w_op = operand(
        &mut gpu,
        &b.w,
        spec.0,
        Some(b.w.atoms_at(k, experts * n, |r| r)),
    );
    let offs: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
    let alpha_e = gpu.up(&b.alpha_e);
    let g = Grouped {
        experts: experts as u32,
        groups: Groups::Contiguous {
            offsets: gpu.up(&offs),
            rows: rows as u32,
        },
        alpha_e,
    };
    let grouped = gpu.up(&vec![NAN; rows * n]);
    let looped = gpu.up(&vec![NAN; rows * n]);
    let ctx = gpu.ctx();
    grouped_gemm(&ctx, a_op, w_op, grouped, g, [n as u32, k as u32], 1.0, 957).expect("grouped");
    let atoms_along = k.div_ceil(128).div_ceil(4) * 512;
    for e in 0..experts {
        let m = counts[e];
        if m == 0 {
            continue;
        }
        let at = |o: Operand, row: usize, row_bytes: usize| {
            let s = o.scales.expect("scaled");
            Operand::scaled(
                o.ptr + (row * row_bytes) as u64,
                o.elem,
                s.ptr + (row / 128 * atoms_along) as u64,
                s.gran,
            )
        };
        let plan = Plan::choose(m as u32, n as u32, k as u32, 148);
        gemm_with(
            &ctx,
            at(a_op, offsets[e], k),
            at(w_op, e * n, k),
            looped + (offsets[e] * n * 2) as u64,
            [m as u32, n as u32, k as u32],
            b.alpha_e[e],
            plan,
        )
        .expect("dense");
    }
    gpu.sync();
    let grouped: Vec<u16> = gpu.down(grouped, rows * n);
    let looped: Vec<u16> = gpu.down(looped, rows * n);
    let (mut equal, mut all) = (0, 0);
    for e in 0..experts {
        for r in offsets[e]..offsets[e] + counts[e] {
            for c in 0..n {
                let (g, l) = (from_bf16(grouped[r * n + c]), from_bf16(looped[r * n + c]));
                assert!(
                    (g - l).abs() <= l.abs() / 64.0 + 1e-3,
                    "expert {e}: y[{r}][{c}] grouped {g}, looped {l}"
                );
                equal += usize::from(g == l);
                all += 1;
            }
        }
    }
    eprintln!("contiguous vs dense loop: {equal} of {all} bit-equal");
}

/// The contiguous cases' `kernel.gemm.nvfp4_k128`: NVFP4 at 128 K a stage.
static K128: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[test]
fn the_tcgen05_grouped_gemm_answers_each_experts_dot_every_case() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let bf = (Elem::Bf16, None);
    let f8 = (Elem::E4m3, None);
    let ds = (Elem::E4m3, Some(128));
    let nv = (Elem::E2m1, Some(16));
    let w4 = (Elem::E2m1, Some(32));
    for (a, w) in [(bf, bf), (f8, f8), (ds, ds), (nv, nv), (ds, w4)] {
        contiguous(a, w, 8, [256, 512], None);
        masked(a, w, 8, 16, [256, 512], 0);
    }
    // NVFP4 at 128 K a stage (64-byte rows), a ragged K tail and a K split.
    K128.store(true, std::sync::atomic::Ordering::Relaxed);
    contiguous(nv, nv, 8, [256, 512], None);
    contiguous(nv, nv, 3, [640, 640], None);
    let lone = |bn| {
        Some(Plan {
            swap: false,
            pair: false,
            bn,
            stages: 0,
            splits: 0,
        })
    };
    contiguous(bf, bf, 5, [384, 520], lone(128));
    contiguous(ds, ds, 4, [512, 1152], lone(256));
    contiguous(
        nv,
        nv,
        3,
        [640, 768],
        Some(Plan {
            splits: 3,
            ..lone(128).unwrap()
        }),
    );
    K128.store(false, std::sync::atomic::Ordering::Relaxed);
    // CTA pairs: 256-row tiles, each CTA its expert's next 128 rows.
    let pair = Some(Plan {
        pair: true,
        ..lone(256).unwrap()
    });
    for (a, w) in [(bf, bf), (ds, ds), (nv, nv), (ds, w4)] {
        contiguous(a, w, 6, [512, 1024], pair);
    }
    masked(bf, bf, 6, 1, [384, 4104], 1);
    masked(bf, bf, 6, 1, [384, 4104], 0);
    masked(nv, nv, 12, 40, [256, 2048], 4);
    masked(ds, w4, 5, 130, [128, 1024], 2);
    masked(ds, ds, 7, 64, [640, 512], 1);
    the_contiguous_group_lands_what_a_loop_of_dense_gemms_lands();
}
