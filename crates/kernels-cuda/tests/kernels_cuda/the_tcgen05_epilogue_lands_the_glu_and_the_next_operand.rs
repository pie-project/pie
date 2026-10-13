#![cfg(feature = "cuda")]

use crate::common::tc5::{Side, e2m1, e4m3, side};
use crate::common::{Gpu, Lcg, from_bf16};
use kernels_cuda::linear::tc5::{
    Elem, Epilogue, Glu, Grouped, Groups, Operand, Out, Plan, atom_bytes, atom_offset,
    gemm_epi_with, grouped_gemm_epi_with, supported,
};

type Spec = (Elem, Option<usize>);

const NAN: u16 = 0x7fc0;

fn glu(f: Glu, g: f64, u: f64) -> f64 {
    let silu = |g: f64, a: f64| g / (1.0 + (-a * g).exp());
    match f {
        Glu::Swiglu => silu(g, 1.0) * u,
        Glu::SwigluClamp { limit } => {
            let l = f64::from(limit);
            silu(g.min(l), 1.0) * u.clamp(-l, l)
        }
        Glu::SwigluClampAlpha { limit, alpha } => {
            let l = f64::from(limit);
            let g = g.min(l);
            (u.clamp(-l, l) + 1.0) * silu(g, f64::from(alpha))
        }
    }
}

fn operand(gpu: &mut Gpu, s: &Side, elem: Elem, atoms: Option<Vec<u8>>) -> Operand {
    let ptr = gpu.up(&s.bytes);
    match (&s.scales, atoms) {
        (Some((_, _, gran)), Some(atoms)) => {
            Operand::scaled(ptr, elem, gpu.up(&atoms), *gran as u32)
        }
        _ => Operand::plain(ptr, elem),
    }
}

/// One output row's want: `alpha * act[a_row] . w[w0 + c]`, through the GLU
/// (gate at `w0 + c`, up `inter` further) and the routed weight.
#[allow(clippy::too_many_arguments)]
fn want_row(
    act: &Side,
    a_row: usize,
    w: &Side,
    w0: usize,
    [n, k]: [usize; 2],
    alpha: f64,
    f: Option<Glu>,
    rs: f64,
) -> Vec<f64> {
    let dot = |wr: usize| -> f64 {
        (0..k)
            .map(|kk| f64::from(act.value(a_row, k, kk)) * f64::from(w.value(wr, k, kk)))
            .sum::<f64>()
            * alpha
    };
    (0..n)
        .map(|c| {
            let v = match f {
                Some(f) => glu(f, dot(w0 + c), dot(w0 + n + c)),
                None => dot(w0 + c),
            };
            v * rs
        })
        .collect()
}

/// Where an output lands, read back: bf16 rows, or the quantized operand
/// and its scales.
enum Landed {
    Bf16(Vec<u16>),
    Fp8 {
        data: Vec<u8>,
        scales: Vec<u8>,
    },
    Nvfp4 {
        data: Vec<u8>,
        scales: Vec<u8>,
        gscale: f32,
    },
}

impl Landed {
    /// Row `r`'s values (scale row `sr`), dequantized; `None` where a bf16
    /// row is untouched.
    fn row(&self, r: usize, sr: usize, n: usize) -> Vec<f64> {
        match self {
            Landed::Bf16(y) => (0..n).map(|c| f64::from(from_bf16(y[r * n + c]))).collect(),
            Landed::Fp8 { data, scales } => (0..n)
                .map(|c| {
                    let s = scales[atom_offset(sr, c / 128, n / 128)];
                    f64::from(e4m3(data[r * n + c])) * 2f64.powi(i32::from(s) - 127)
                })
                .collect(),
            Landed::Nvfp4 {
                data,
                scales,
                gscale,
            } => (0..n)
                .map(|c| {
                    let byte = data[r * (n / 2) + c / 2];
                    let code = if c % 2 == 0 { byte & 0xf } else { byte >> 4 };
                    let s = f64::from(e4m3(scales[atom_offset(sr, c / 16, n / 16)]));
                    f64::from(e2m1(code)) * s / f64::from(*gscale)
                })
                .collect(),
        }
    }

    /// The error a value may carry: bf16 rounding, or the block's quantum.
    fn slack(&self, want: &[f64], c: usize) -> f64 {
        let block = |w: usize| {
            let b = c / w * w;
            want[b..(b + w).min(want.len())]
                .iter()
                .fold(0f64, |a, v| a.max(v.abs()))
        };
        match self {
            Landed::Bf16(_) => want[c].abs() / 128.0 + 1e-5,
            Landed::Fp8 { .. } => want[c].abs() / 14.0 + block(128) / 448.0 + 1e-6,
            Landed::Nvfp4 { .. } => block(16) / 6.0 * 1.2 + want[c].abs() / 8.0 + 1e-6,
        }
    }
}

struct Target {
    out: Out,
    y: u64,
    rows: usize,
    scale_rows: usize,
}

fn target(gpu: &mut Gpu, kind: u32, rows: usize, scale_rows: usize, n: usize, amax: f64) -> Target {
    match kind {
        0 => Target {
            out: Out::Bf16,
            y: gpu.up(&vec![NAN; rows * n]),
            rows,
            scale_rows,
        },
        1 => Target {
            out: Out::Fp8Block {
                data: gpu.zeros(rows * n),
                scales: gpu.zeros(atom_bytes(scale_rows, n, 128)),
            },
            y: 0,
            rows,
            scale_rows,
        },
        _ => Target {
            out: Out::Nvfp4 {
                data: gpu.zeros(rows * n / 2),
                scales: gpu.zeros(atom_bytes(scale_rows, n, 16)),
                gscale: (448.0 * 6.0 / amax.max(1e-6)) as f32,
            },
            y: 0,
            rows,
            scale_rows,
        },
    }
}

fn land(gpu: &Gpu, t: &Target, n: usize) -> Landed {
    match t.out {
        Out::Bf16 => Landed::Bf16(gpu.down(t.y, t.rows * n)),
        Out::Fp8Block { data, scales } => Landed::Fp8 {
            data: gpu.down(data, t.rows * n),
            scales: gpu.down(scales, atom_bytes(t.scale_rows, n, 128)),
        },
        Out::Nvfp4 {
            data,
            scales,
            gscale,
        } => Landed::Nvfp4 {
            data: gpu.down(data, t.rows * n / 2),
            scales: gpu.down(scales, atom_bytes(t.scale_rows, n, 16)),
            gscale,
        },
    }
}

fn compare(what: &str, landed: &Landed, r: usize, sr: usize, want: &[f64]) {
    let got = landed.row(r, sr, want.len());
    for (c, (&g, &w)) in got.iter().zip(want).enumerate() {
        let slack = landed.slack(want, c);
        assert!(
            (g - w).abs() <= slack,
            "{what}: y[{r}][{c}] = {g}, want {w} (slack {slack})"
        );
    }
}

fn rows_scale(gpu: &mut Gpu, rows: usize, on: bool) -> (u64, Vec<f64>) {
    let rs: Vec<f32> = (0..rows).map(|r| 0.25 + (r % 7) as f32 * 0.125).collect();
    if on {
        (gpu.up(&rs), rs.iter().map(|&v| f64::from(v)).collect())
    } else {
        (0, vec![1.0; rows])
    }
}

#[allow(clippy::too_many_arguments)]
fn dense(
    a: Spec,
    w: Spec,
    f: Option<Glu>,
    out: u32,
    m: usize,
    [n, k]: [usize; 2],
    swap: bool,
    routed: bool,
) {
    let what = format!(
        "dense {a:?} x {w:?} {f:?} out={out} m={m} n={n} k={k} swap={swap} routed={routed}"
    );
    let mut lcg = Lcg::seeded(0xe9 ^ (m * n + k) as u64);
    let mut bits = 0xe9 ^ n as u64;
    let act = side(&mut lcg, &mut bits, a.0, a.1, m, k);
    let w_rows = if f.is_some() { 2 * n } else { n };
    let wt = side(&mut lcg, &mut bits, w.0, w.1, w_rows, k);
    let mut gpu = Gpu::open();
    let a_op = operand(&mut gpu, &act, a.0, a.1.map(|_| act.atoms_at(k, m, |r| r)));
    let w_op = operand(
        &mut gpu,
        &wt,
        w.0,
        w.1.map(|_| wt.atoms_at(k, w_rows, |r| r)),
    );
    let (row_scale, rs) = rows_scale(&mut gpu, m, routed);
    let alpha = 0.5;
    let wants: Vec<Vec<f64>> = (0..m)
        .map(|r| want_row(&act, r, &wt, 0, [n, k], f64::from(alpha), f, rs[r]))
        .collect();
    let amax = wants.iter().flatten().fold(0f64, |a, v| a.max(v.abs()));
    let t = target(&mut gpu, out, m, m, n, amax);
    let epi = Epilogue {
        glu: f,
        out: t.out,
        row_scale,
    };
    let plan = Plan {
        swap,
        pair: false,
        bn: if swap {
            m.next_multiple_of(32) as u32
        } else {
            256
        },
        stages: 0,
        splits: 0,
    };
    gemm_epi_with(
        &gpu.ctx(),
        a_op,
        w_op,
        t.y,
        [m as u32, n as u32, k as u32],
        alpha,
        epi,
        plan,
    )
    .unwrap_or_else(|e| panic!("{what}: {e}"));
    gpu.sync();
    let landed = land(&gpu, &t, n);
    for (r, want) in wants.iter().enumerate() {
        compare(&what, &landed, r, r, want);
    }
    eprintln!("{what}: lands");
}

#[allow(clippy::too_many_arguments)]
fn grouped(
    a: Spec,
    w: Spec,
    f: Glu,
    out: u32,
    masked: bool,
    experts: usize,
    [n, k]: [usize; 2],
    routed: bool,
) {
    let what =
        format!("grouped {a:?} x {w:?} {f:?} out={out} masked={masked} e={experts} n={n} k={k}");
    let mut lcg = Lcg::seeded(0x9e ^ (n * k) as u64);
    let mut bits = 0x9e ^ k as u64;
    let wt = side(&mut lcg, &mut bits, w.0, w.1, experts * 2 * n, k);
    let counts: Vec<usize> = (0..experts).map(|e| [0, 3, 40, 17][e % 4]).collect();
    let alpha_e: Vec<f32> = (0..experts).map(|e| 0.5 + 0.25 * (e % 3) as f32).collect();
    let mut gpu = Gpu::open();
    let w_op = operand(
        &mut gpu,
        &wt,
        w.0,
        w.1.map(|_| wt.atoms_at(k, experts * 2 * n, |r| r)),
    );
    let alpha_dev = gpu.up(&alpha_e);
    // (y row, its scale row, its act row, its expert) for every routed row.
    let mut rows_of = Vec::new();
    let (groups, act, act_rows, scale_rows, atoms) = if masked {
        let m_max = 40usize;
        let pad = 128;
        for e in 0..experts {
            for i in 0..counts[e] {
                rows_of.push((e * m_max + i, e * pad + i, e * m_max + i, e));
            }
        }
        let rows = experts * m_max;
        let act = side(&mut lcg, &mut bits, a.0, a.1, rows, k);
        let atoms =
            a.1.map(|_| act.atoms_at(k, experts * pad, |r| r / m_max * pad + r % m_max));
        let cnt: Vec<i32> = counts.iter().map(|&c| c as i32).collect();
        let g = Groups::Masked {
            counts: gpu.up(&cnt),
            m_max: m_max as u32,
        };
        (g, act, rows, experts * pad, atoms)
    } else {
        let mut offsets = vec![0usize];
        for (e, &c) in counts.iter().enumerate() {
            let o = *offsets.last().unwrap();
            for i in 0..c {
                rows_of.push((o + i, o + i, o + i, e));
            }
            offsets.push(o + c.next_multiple_of(128));
        }
        let rows = offsets[experts] + 128;
        let act = side(&mut lcg, &mut bits, a.0, a.1, rows, k);
        let atoms = a.1.map(|_| act.atoms_at(k, rows, |r| r));
        let offs: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
        let g = Groups::Contiguous {
            offsets: gpu.up(&offs),
            rows: rows as u32,
        };
        (g, act, rows, rows, atoms)
    };
    let a_op = operand(&mut gpu, &act, a.0, atoms);
    let (row_scale, rs) = rows_scale(&mut gpu, act_rows, routed);
    let wants: Vec<Vec<f64>> = rows_of
        .iter()
        .map(|&(r, _, ar, e)| {
            want_row(
                &act,
                ar,
                &wt,
                e * 2 * n,
                [n, k],
                f64::from(alpha_e[e]),
                Some(f),
                rs[r],
            )
        })
        .collect();
    let amax = wants.iter().flatten().fold(0f64, |a, v| a.max(v.abs()));
    let t = target(&mut gpu, out, act_rows, scale_rows, n, amax);
    let g = Grouped {
        experts: experts as u32,
        groups,
        alpha_e: alpha_dev,
    };
    let epi = Epilogue {
        glu: Some(f),
        out: t.out,
        row_scale,
    };
    let plan = Plan {
        swap: masked,
        pair: false,
        bn: if masked { 64 } else { 256 },
        stages: 0,
        splits: 0,
    };
    grouped_gemm_epi_with(
        &gpu.ctx(),
        a_op,
        w_op,
        t.y,
        g,
        [n as u32, k as u32],
        1.0,
        epi,
        plan,
    )
    .unwrap_or_else(|e| panic!("{what}: {e}"));
    gpu.sync();
    let landed = land(&gpu, &t, n);
    for (&(r, sr, _, _), want) in rows_of.iter().zip(&wants) {
        compare(&what, &landed, r, sr, want);
    }
    eprintln!("{what}: lands");
}

#[test]
fn the_tcgen05_epilogue_lands_the_glu_and_the_next_operand_every_case() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let bf = (Elem::Bf16, None);
    let ds = (Elem::E4m3, Some(128));
    let nv = (Elem::E2m1, Some(16));
    let w4 = (Elem::E2m1, Some(32));
    let sw = Some(Glu::Swiglu);
    let clamp = Some(Glu::SwigluClamp { limit: 7.0 });
    let alpha = Some(Glu::SwigluClampAlpha {
        limit: 7.0,
        alpha: 1.702,
    });
    for (a, w) in [(bf, bf), (ds, ds), (nv, nv), (ds, w4)] {
        dense(a, w, sw, 0, 200, [256, 512], false, false);
        dense(a, w, sw, 0, 5, [256, 512], true, false);
    }
    dense(bf, bf, clamp, 0, 130, [128, 256], false, false);
    dense(bf, bf, alpha, 0, 33, [256, 256], true, false);
    dense(ds, ds, alpha, 0, 129, [384, 640], false, true);
    dense(bf, bf, None, 0, 150, [256, 512], false, true);
    dense(nv, nv, None, 0, 7, [256, 512], true, true);
    for out in [1, 2] {
        let (a, w) = if out == 2 { (bf, bf) } else { (ds, ds) };
        dense(a, w, sw, out, 140, [256, 512], false, false);
        dense(nv, nv, clamp, out, 9, [256, 512], true, true);
    }
    for masked in [false, true] {
        for out in [0, 1, 2] {
            // NVFP4 spans 448 x 6 of dynamic range a tensor: the heavy-tailed
            // fp8 products would flush whole blocks to zero, so bf16 feeds it.
            let (a, w) = if out == 2 { (bf, bf) } else { (ds, ds) };
            grouped(a, w, Glu::Swiglu, out, masked, 6, [256, 512], out == 0);
        }
        grouped(
            nv,
            nv,
            Glu::SwigluClamp { limit: 7.0 },
            2,
            masked,
            5,
            [128, 512],
            true,
        );
    }
}
