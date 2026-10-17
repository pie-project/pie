#![cfg(feature = "cuda")]

//! W4A16 on the tcgen05 GEMM: bf16 activations against 4-bit weights under
//! row-major factors -- mxfp4 (e2m1, ue8m0 a 32), nvfp4 (e2m1, e4m3 a 16,
//! and an f32 tensor scale a row in the epilogue) and pie's affine u4 (a
//! bf16 scale and zero a group of 32, 64 or 128) -- which the kernel's
//! transform warps dequantize to bf16. An e2m1 weight is exact in bf16; an
//! affine one rounds as pie's affine kernels round it, which the reference
//! rounds alike. The product is then the f64 dot up to f32 accumulation.
//! Dense (swapped and not), grouped (contiguous and masked), with a bias a
//! weight row and a fused SwiGLU.

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use kernels_cuda::linear::tc5::{
    Elem, Epilogue, GROUP_ALIGN, Glu, Grouped, Groups, Operand, Out, Plan, gemm_epi_with,
    grouped_gemm_epi_with, supported,
};

const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// A W4A16 weight's format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fmt {
    Mx,
    Nv,
    Affine(usize),
}

impl Fmt {
    const fn gran(self) -> usize {
        match self {
            Self::Mx => 32,
            Self::Nv => 16,
            Self::Affine(group) => group,
        }
    }
}

/// A random 4-bit weight `[rows][k]` and its row-major factors.
struct Weight {
    fmt: Fmt,
    codes: Vec<u8>,
    scales: Vec<u8>,
    /// Affine: bf16 scales and zeros a group.
    factors: Vec<u16>,
    zeros: Vec<u16>,
    /// NVFP4: an f32 tensor scale a row.
    tensor: Vec<f32>,
    k: usize,
}

impl Weight {
    fn random(lcg: &mut Lcg, rows: usize, k: usize, fmt: Fmt) -> Self {
        let gran = fmt.gran();
        let mut units: Vec<f32> = (0..rows * k).map(|_| lcg.unit()).collect();
        let mut at = 0;
        let mut byte = || {
            at += 1;
            ((units[at % units.len()] + 1.0) * 127.99) as u8
        };
        let codes: Vec<u8> = (0..rows * k / 2).map(|_| byte()).collect();
        let scales = (0..rows * k / gran)
            .map(|_| {
                let b = byte();
                if gran == 32 {
                    // ue8m0: 2^-4 .. 2^3
                    123 + b % 8
                } else {
                    // e4m3, positive normals 2^-3 .. 2^2-ish
                    0x28 + b % 0x28
                }
            })
            .collect();
        let factor = |b: u8, lo: f32, span: f32| to_bf16(lo + span * f32::from(b) / 255.0);
        let factors = (0..rows * k / gran)
            .map(|_| factor(byte(), 0.01, 0.2))
            .collect();
        let zeros = (0..rows * k / gran)
            .map(|_| factor(byte(), -1.5, 1.0))
            .collect();
        let tensor = (0..rows)
            .map(|_| 0.5 + f32::from(byte() % 8) * 0.25)
            .collect();
        units.clear();
        Self {
            fmt,
            codes,
            scales,
            factors,
            zeros,
            tensor,
            k,
        }
    }

    fn scale(&self, row: usize, col: usize) -> f64 {
        let gran = self.fmt.gran();
        let b = self.scales[row * (self.k / gran) + col / gran];
        if gran == 32 {
            2f64.powi(i32::from(b) - 127)
        } else {
            let (e, m) = (i32::from((b >> 3) & 15), f64::from(b & 7));
            if e == 0 {
                m * 2f64.powi(-9)
            } else {
                (1.0 + m / 8.0) * 2f64.powi(e - 7)
            }
        }
    }

    fn value(&self, row: usize, col: usize) -> f64 {
        let byte = self.codes[(row * self.k + col) / 2];
        let nib = if col.is_multiple_of(2) {
            byte & 15
        } else {
            byte >> 4
        };
        if let Fmt::Affine(group) = self.fmt {
            let at = row * (self.k / group) + col / group;
            let (s, z) = (from_bf16(self.factors[at]), from_bf16(self.zeros[at]));
            return f64::from(from_bf16(to_bf16(f32::from(nib).mul_add(s, z))));
        }
        let mag = E2M1[usize::from(nib & 7)];
        let v = if nib & 8 != 0 { -mag } else { mag };
        let t = if self.fmt == Fmt::Nv {
            f64::from(self.tensor[row])
        } else {
            1.0
        };
        v * self.scale(row, col) * t
    }

    /// The operand and the epilogue's weight scale, uploaded.
    fn up(&self, gpu: &mut Gpu) -> (Operand, u64) {
        let codes = gpu.up(&self.codes);
        match self.fmt {
            Fmt::Mx | Fmt::Nv => (
                Operand::scaled(
                    codes,
                    Elem::E2m1,
                    gpu.up(&self.scales),
                    self.fmt.gran() as u32,
                ),
                if self.fmt == Fmt::Nv {
                    gpu.up(&self.tensor)
                } else {
                    0
                },
            ),
            Fmt::Affine(group) => (
                Operand::affine(
                    codes,
                    gpu.up(&self.factors),
                    gpu.up(&self.zeros),
                    group as u32,
                ),
                0,
            ),
        }
    }
}

fn bf16s(lcg: &mut Lcg, count: usize) -> (Vec<u16>, Vec<f64>) {
    let (raw, exact) = lcg.row(count);
    (raw, exact.into_iter().map(f64::from).collect())
}

fn silu(g: f64) -> f64 {
    g / (1.0 + (-g).exp())
}

/// `y[r][c]` for weight rows `w_row(c)` (and `w_row(c) + inter` for a GLU's
/// up), with the bias each weight row.
#[allow(clippy::too_many_arguments)]
fn want(
    x: &[f64],
    xr: usize,
    w: &Weight,
    bias: &[f64],
    row: usize,
    glu: Option<usize>,
    k: usize,
) -> f64 {
    let dot = |wr: usize| -> f64 {
        (0..k).map(|i| x[xr * k + i] * w.value(wr, i)).sum::<f64>() + bias[wr]
    };
    match glu {
        None => dot(row),
        Some(inter) => silu(dot(row)) * dot(row + inter),
    }
}

fn check(what: &str, got: &[u16], wants: &[(usize, f64)]) {
    let scale = wants.iter().fold(0f64, |a, (_, v)| a.max(v.abs()));
    let mut bad = Vec::new();
    for &(at, w) in wants {
        let g = f64::from(from_bf16(got[at]));
        let miss = (g - w).abs() / (w.abs() / 128.0 + scale * 1e-3 + 1e-6);
        if miss > 1.0 {
            bad.push((at, g, w));
        }
    }
    assert!(
        bad.is_empty(),
        "{what}: {} of {} miss, first {:?}",
        bad.len(),
        wants.len(),
        &bad[..bad.len().min(6)]
    );
}

fn dense(fmt: Fmt, [m, n, k]: [usize; 3], swap: bool, glu: bool) {
    let what = format!("dense {fmt:?} {m}x{n}x{k} swap={swap} glu={glu}");
    let mut lcg = Lcg::seeded((m * 31 + n * 7 + k + fmt.gran()) as u64);
    let wrows = if glu { 2 * n } else { n };
    let w = Weight::random(&mut lcg, wrows, k, fmt);
    let (x_raw, x) = bf16s(&mut lcg, m * k);
    let (b_raw, bias) = bf16s(&mut lcg, wrows);
    let mut gpu = Gpu::open();
    let act = Operand::plain(gpu.up(&x_raw), Elem::Bf16);
    let (wt, weight_scale) = w.up(&mut gpu);
    let y = gpu.zeros(m * n * 2);
    let epi = Epilogue {
        glu: glu.then_some(Glu::Swiglu),
        out: Out::Bf16,
        row_scale: 0,
        bias: gpu.up(&b_raw),
        weight_scale,
        live_rows: 0,
        input_scales: 0,
    };
    let plan = Plan {
        swap,
        pair: false,
        bn: if swap {
            m.next_multiple_of(32).min(256) as u32
        } else {
            128
        },
        stages: 0,
        splits: 0,
    };
    gemm_epi_with(
        &gpu.ctx(),
        act,
        wt,
        y,
        [m as u32, n as u32, k as u32],
        1.0,
        epi,
        plan,
    )
    .unwrap_or_else(|e| panic!("{what}: {e}"));
    gpu.sync();
    let got: Vec<u16> = gpu.down(y, m * n);
    let wants: Vec<(usize, f64)> = (0..m)
        .flat_map(|r| (0..n).map(move |c| (r, c)))
        .filter(|(r, c)| (r * 7 + c * 3) % 5 == 0)
        .map(|(r, c)| (r * n + c, want(&x, r, &w, &bias, c, glu.then_some(n), k)))
        .collect();
    check(&what, &got, &wants);
}

fn grouped(fmt: Fmt, experts: usize, [n, k]: [usize; 2], masked: bool, glu: bool) {
    let what = format!("grouped {fmt:?} e={experts} n={n} k={k} masked={masked} glu={glu}");
    let mut lcg = Lcg::seeded((experts * 131 + n + k + fmt.gran()) as u64);
    let n_per = if glu { 2 * n } else { n };
    let w = Weight::random(&mut lcg, experts * n_per, k, fmt);
    let (b_raw, bias) = bf16s(&mut lcg, experts * n_per);
    let counts: Vec<usize> = (0..experts).map(|e| (e * 37 + 5) % 70).collect();
    let mut gpu = Gpu::open();
    // (activation row, expert) of every routed row.
    let mut rows_of = Vec::new();
    let (groups, rows) = if masked {
        let m_max = 72usize;
        for (e, &c) in counts.iter().enumerate() {
            for i in 0..c {
                rows_of.push((e * m_max + i, e));
            }
        }
        let cnt: Vec<i32> = counts.iter().map(|&c| c as i32).collect();
        (
            Groups::Masked {
                counts: gpu.up(&cnt),
                m_max: m_max as u32,
            },
            experts * m_max,
        )
    } else {
        let mut offsets = vec![0usize];
        for (e, &c) in counts.iter().enumerate() {
            let o = *offsets.last().unwrap();
            for i in 0..c {
                rows_of.push((o + i, e));
            }
            offsets.push(o + c.next_multiple_of(GROUP_ALIGN as usize));
        }
        let rows = offsets[experts] + 128;
        let offs: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
        (
            Groups::Contiguous {
                offsets: gpu.up(&offs),
                rows: rows as u32,
            },
            rows,
        )
    };
    let (x_raw, x) = bf16s(&mut lcg, rows * k);
    let act = Operand::plain(gpu.up(&x_raw), Elem::Bf16);
    let (wt, weight_scale) = w.up(&mut gpu);
    let y = gpu.up(&vec![to_bf16(0.0); rows * n]);
    let epi = Epilogue {
        glu: glu.then_some(Glu::Swiglu),
        out: Out::Bf16,
        row_scale: 0,
        bias: gpu.up(&b_raw),
        weight_scale,
        live_rows: 0,
        input_scales: 0,
    };
    let plan = Plan {
        swap: masked,
        pair: false,
        bn: if masked { 96 } else { 256 },
        stages: 0,
        splits: 0,
    };
    let g = Grouped {
        experts: experts as u32,
        groups,
        alpha_e: 0,
    };
    grouped_gemm_epi_with(
        &gpu.ctx(),
        act,
        wt,
        y,
        g,
        [n as u32, k as u32],
        1.0,
        epi,
        plan,
    )
    .unwrap_or_else(|e| panic!("{what}: {e}"));
    gpu.sync();
    let got: Vec<u16> = gpu.down(y, rows * n);
    let wants: Vec<(usize, f64)> = rows_of
        .iter()
        .enumerate()
        .flat_map(|(i, &(r, e))| (0..n).map(move |c| (i, r, e, c)))
        .filter(|(i, _, _, c)| (i * 5 + c * 3) % 7 == 0)
        .map(|(_, r, e, c)| {
            (
                r * n + c,
                want(&x, r, &w, &bias, e * n_per + c, glu.then_some(n), k),
            )
        })
        .collect();
    check(&what, &got, &wants);
}

#[test]
fn the_tcgen05_w4a16_gemm_answers_the_dequantized_dot_every_case() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    // Dense swapped and not, grouped contiguous and masked, each with and
    // without the SwiGLU; K off the stage and on it.
    for fmt in [Fmt::Mx, Fmt::Nv, Fmt::Affine(64)] {
        dense(fmt, [200, 256, 576], false, false);
        dense(fmt, [24, 384, 2880], true, false);
        dense(fmt, [130, 256, 512], false, true);
        grouped(fmt, 6, [256, 576], false, false);
        grouped(fmt, 5, [128, 512], true, true);
    }
    // Affine groups of 32 (two a stage) and 128 (a stage's half).
    for fmt in [Fmt::Affine(32), Fmt::Affine(128)] {
        dense(fmt, [200, 256, 512], false, false);
        grouped(fmt, 5, [128, 512], true, true);
    }
}
