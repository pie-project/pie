//! Host references the GEMM, select, quantizer and router tests share: one
//! dot over decoded operands, one GLU, one quantizer, one routed MLP.
#![allow(dead_code)]

use super::host::{to_e2m1, to_e4m3};
use super::tc5::{Side, e4m3, side, ue8m0};
use super::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::act_quant::{fp8_tile_requant, scales_to_atoms};
use kernels_cuda::linear::tc5::{Elem, Glu, atom_bytes};
use kernels_cuda::tensor::Tensor;

/// The tolerance a bf16 product answers in, against its value plus its
/// terms' scale.
pub const TOL: f64 = 1.0 / 64.0;

/// A `[rows][width]` tensor at `at`.
pub fn t(at: u64, rows: usize, width: usize, dtype: Dtype) -> Tensor {
    Tensor::new(at, rows as u32, width as u32, dtype)
}

/// `(0..n).map(f)`, collected.
pub fn tab<T>(n: usize, f: impl FnMut(usize) -> T) -> Vec<T> {
    (0..n).map(f).collect()
}

/// `(0..n).map(f)`, summed.
pub fn sum(n: usize, f: impl FnMut(usize) -> f64) -> f64 {
    (0..n).map(f).sum()
}

/// `a` where `c` holds, else `b`.
pub fn pick<T>(c: bool, a: T, b: T) -> T {
    if c { a } else { b }
}

/// A GPU and a seeded draw of a test's operands.
pub struct Bench {
    pub gpu: Gpu,
    pub lcg: Lcg,
    pub bits: u64,
}

impl core::ops::Deref for Bench {
    type Target = Gpu;
    fn deref(&self) -> &Gpu {
        &self.gpu
    }
}

impl core::ops::DerefMut for Bench {
    fn deref_mut(&mut self) -> &mut Gpu {
        &mut self.gpu
    }
}

impl Bench {
    pub fn new(seed: u64) -> Self {
        let (gpu, lcg, bits) = (Gpu::open(), Lcg::seeded(seed), seed ^ 0x5eed);
        Self { gpu, lcg, bits }
    }

    /// `count` random bf16 values uploaded, and their values.
    pub fn bf16(&mut self, count: usize) -> (u64, Vec<f64>) {
        let (raw, exact) = self.lcg.row(count);
        (self.gpu.up(&raw), tab(count, |i| f64::from(exact[i])))
    }

    /// A zeroed bf16 `[rows][width]` tensor.
    pub fn out(&mut self, rows: usize, width: usize) -> Tensor {
        t(self.gpu.zeros(rows * width * 2), rows, width, Dtype::Bf16)
    }

    /// A random `[rows][k]` operand of `elem`, scaled a `gran` where stated.
    pub fn side(&mut self, elem: Elem, gran: Option<usize>, rows: usize, k: usize) -> Side {
        side(&mut self.lcg, &mut self.bits, elem, gran, rows, k)
    }

    /// The bf16 values at `at`.
    pub fn read(&self, at: u64, count: usize) -> Vec<f64> {
        self.gpu.sync();
        let raw: Vec<u16> = self.gpu.down(at, count);
        tab(count, |i| f64::from(from_bf16(raw[i])))
    }

    /// The bf16 at `at` against `want`, in `tol`.
    pub fn check(&self, at: u64, want: &[Want], tol: f64, what: &str) {
        check(&self.read(at, want.len()), want, tol, what);
    }
}

/// A side's values `[rows][k]`, scaled.
pub fn values(s: &Side, k: usize) -> Vec<f32> {
    tab(s.values.len(), |i| s.value(i / k, k, i % k))
}

/// An output's want: its value, and the scale its error is read against
/// (its terms' magnitudes over the root of their count).
pub type Want = (f64, f64);

/// `y[r][c] = sum_i a(r, i) * w(r, c, i)`, `m x n` over `k`.
pub fn dot(
    [m, n, k]: [usize; 3],
    a: impl Fn(usize, usize) -> f64,
    w: impl Fn(usize, usize, usize) -> f64,
) -> Vec<Want> {
    let mut out = Vec::with_capacity(m * n);
    for r in 0..m {
        let ar = tab(k, |i| a(r, i));
        for c in 0..n {
            let (mut sum, mut mass) = (0f64, 0f64);
            for (i, x) in ar.iter().enumerate() {
                let t = x * w(r, c, i);
                sum += t;
                mass += t.abs();
            }
            out.push((sum, mass / (k as f64).sqrt()));
        }
    }
    out
}

/// Every `got` within `tol` of its want.
pub fn check(got: &[f64], want: &[Want], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let off = |(g, (w, s)): (&f64, &Want)| !((g - w).abs() <= tol * (w.abs() + s) + 1e-6);
    let bad: Vec<_> = got.iter().zip(want).enumerate().collect();
    let bad: Vec<_> = bad.into_iter().filter(|x| off(x.1)).collect();
    let (n, first) = (bad.len(), &bad[..bad.len().min(4)]);
    assert!(n == 0, "{what}: {n} off: {first:?}");
}

/// The mean distance of `got` from `want`, over want's mean magnitude.
pub fn distance(got: &[f64], want: &[f64]) -> f64 {
    let far: f64 = got.iter().zip(want).map(|(g, w)| (g - w).abs()).sum();
    far / want.iter().map(|w| w.abs()).sum::<f64>()
}

/// The host's GLU of a gate and an up value.
pub fn glu(f: Glu, g: f64, u: f64) -> f64 {
    let silu = |g: f64, a: f64| g / (1.0 + (-a * g).exp());
    let clamp = |l: f32| (g.min(l.into()), u.clamp((-l).into(), l.into()));
    match f {
        Glu::Swiglu => silu(g, 1.0) * u,
        Glu::SwigluClamp { limit } => silu(clamp(limit).0, 1.0) * clamp(limit).1,
        Glu::SwigluClampAlpha { limit, alpha } => {
            (clamp(limit).1 + 1.0) * silu(clamp(limit).0, alpha.into())
        }
        Glu::GegluTanh => {
            let inner = (2.0 / std::f64::consts::PI).sqrt() * (g + 0.044_715 * g * g * g);
            0.5 * g * (1.0 + inner.tanh()) * u
        }
        Glu::Situ { beta, cap } => {
            let (b, c) = (f64::from(beta), f64::from(cap));
            let u = if c > 0.0 { c * (u / c).tanh() } else { u };
            b * (g / b).tanh() / (1.0 + (-g).exp()) * u
        }
    }
}

/// The GLU of a gate's and an up's wants.
pub fn glu_want(f: Glu, (g, sg): Want, (u, su): Want) -> Want {
    (glu(f, g, u), sg * (u.abs() + 1.0) + su * (g.abs() + 1.0))
}

/// An affine `bits`-bit weight `[rows][k]` (a bf16 scale and zero a
/// `group`), its values rounded through bf16 as the kernels round them, and
/// its factor and zero planes.
pub fn affine(lcg: &mut Lcg, [rows, k, group, bits]: [usize; 4]) -> (Side, Vec<u16>, Vec<u16>) {
    let mut unit = || (lcg.unit() + 1.0) * 0.5;
    let levels = ((1usize << bits) - 1) as f32;
    let mut plane = |lo: f32, span: f32| tab(rows * k / group, |_| to_bf16(lo + span * unit()));
    let (factors, zeros) = (plane(0.01, 3.0 / levels), plane(-1.5, 1.0));
    let per = 8 / bits;
    let mut bytes = vec![0u8; rows * k / per];
    let values = tab(rows * k, |i| {
        let code = (unit() * (levels + 0.99)) as u8;
        bytes[i / per] |= code << (bits * (i % per));
        let (s, z) = (from_bf16(factors[i / group]), from_bf16(zeros[i / group]));
        from_bf16(to_bf16(f32::from(code).mul_add(s, z)))
    });
    #[rustfmt::skip]
    let side = Side { bytes, values, scales: None, codes: Vec::new() };
    (side, factors, zeros)
}

/// A 128x128-block FP8 weight `[rows][k]` as a release stores it (e4m3
/// codes under an f32 factor a block), requantized on the device to ue8m0
/// exponents (`fp8_tile_requant`) spread to atoms a row: its device codes
/// and atoms, and the values it now holds.
pub fn fp8_bank(gpu: &mut Gpu, rows: usize, k: usize, seed: u64) -> (u64, u64, Vec<f32>) {
    let mut bits = seed | 1;
    let codes = tab(rows * k, |_| {
        bits ^= bits << 13;
        bits ^= bits >> 7;
        bits ^= bits << 17;
        (bits & 0xbf) as u8
    });
    let (cb, blocks) = (k / 128, rows / 128 * k / 128);
    let factors = tab(blocks, |i| 0.002 * (1.0 + 0.034 * (i * 7 % 11) as f32));
    let (at, factors_at) = (gpu.up(&codes), gpu.up(&factors));
    let (plane, atoms) = (gpu.zeros(blocks), gpu.zeros(atom_bytes(rows, k, 128)));
    let (ctx, r, k32) = (gpu.ctx(), rows as u32, k as u32);
    fp8_tile_requant(&ctx, "test", at, factors_at, plane, r, k32).expect("requants");
    scales_to_atoms(&ctx, "test", plane, atoms, r, cb as u32, 128).expect("lands");
    gpu.sync();
    let (got, exps): (Vec<u8>, Vec<u8>) = (gpu.down(at, rows * k), gpu.down(plane, blocks));
    let w = tab(rows * k, |i| {
        let b = i / k / 128 * cb + i % k / 128;
        let (release, exp) = (e4m3(codes[i]) * factors[b], ue8m0(exps[b]));
        let held = e4m3(got[i]) * exp;
        // The requant keeps each value to e4m3's rounding.
        let near = (held - release).abs() <= release.abs() / 15.0 + exp / 512.0;
        assert!(near, "value {i}: {held} for {release}");
        held
    });
    (at, atoms, w)
}

/// DeepSeek's `act_quant` of one block: its e4m3 codes and ue8m0 scale.
pub fn quant_e4m3(block: &[f32]) -> (Vec<u8>, u8) {
    let amax = block.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-4);
    let bits = (amax * (1.0 / 448.0)).to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127 + i32::from(bits & 0x7f_ffff != 0);
    let q = |v: f32| to_e4m3((v / 2f32.powi(e)).clamp(-448.0, 448.0));
    (tab(block.len(), |i| q(block[i])), (e + 127) as u8)
}

/// The NVFP4 recipe of one 16-block under global `gs`: its e2m1 codes and
/// e4m3 scale (exact reciprocals where the device's are approximate).
pub fn quant_nvfp4(block: &[f32], gs: f32) -> (Vec<u8>, u8) {
    let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
    let sf = to_e4m3(gs * (amax * (1.0 / 6.0)));
    let out = if e4m3(sf) == 0.0 {
        0.0
    } else {
        1.0 / (e4m3(sf) / gs)
    };
    (tab(block.len(), |i| to_e2m1(block[i] * out)), sf)
}

/// A routed MLP's host product: for each token, the sum over its routes of
/// `weight * down(glu(up . x))`, the GLU's rows rounded through bf16 (with
/// `first`, times the weight before they round, then summed plainly), plus
/// `shared` (times the sigmoid of `gate` where there is one).
#[allow(clippy::too_many_arguments)]
pub fn mlp(
    x: &[f64],
    (routes, weights): (&[i32], &[f32]),
    [hidden, inter, top]: [usize; 3],
    up: impl Fn(usize, usize, usize) -> f64,
    down: impl Fn(usize, usize, usize) -> f64,
    f: Glu,
    first: bool,
    shared: Option<(&[f64], Option<&[f64]>)>,
) -> Vec<f64> {
    let mut y = vec![0f64; routes.len() / top * hidden];
    for (r, &e) in routes.iter().enumerate() {
        let (t, e, wt) = (r / top, e as usize, f64::from(weights[r]));
        let row = |o: usize| sum(hidden, |i| x[t * hidden + i] * up(e, o, i));
        let h = tab(inter, |c| {
            let v = glu(f, row(c), row(inter + c)) * if first { wt } else { 1.0 };
            f64::from(from_bf16(to_bf16(v as f32)))
        });
        for o in 0..hidden {
            let v = sum(inter, |c| h[c] * down(e, o, c));
            y[t * hidden + o] += if first { v } else { wt * v };
        }
    }
    for (i, v) in y.iter_mut().enumerate().filter(|_| shared.is_some()) {
        let (s, gate) = shared.expect("a shared row");
        *v += s[i] * gate.map_or(1.0, |g| 1.0 / (1.0 + (-g[i / hidden]).exp()));
    }
    y
}
