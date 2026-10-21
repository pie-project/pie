//! Host references and fixtures the attention, rotation and norm tests share.
#![allow(dead_code)]

use super::host::{to_e2m1, to_e4m3};
use super::tc5::{e2m1, e4m3};
use super::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::rope::{Yarn, ramp_bounds};
use kernels_cuda::tensor::{KvPool, Tensor};
use kernels_cuda::tuning::Tuning;

/// DeepSeek-V4's YaRN.
pub const YARN: Yarn = Yarn {
    factor: 16.0,
    beta_fast: 32.0,
    beta_slow: 1.0,
    original_max_position: 65536,
};

/// Softmax attention in f64. `q` is `[rows][qh][dqk]`, `k` `[cell][kvh][dqk]`,
/// `v` `[cell][kvh][dv]`; `keys(row, head)` names the cells a query head
/// reads, `sinks` adds a head's logit to the denominator. Answers the output
/// `[rows][qh][dv]` and the base-2 log-sum-exp `[rows][qh]` (sinks apart).
pub fn attend(
    [q, k, v]: [&[f32]; 3],
    [qh, kvh, dqk, dv]: [usize; 4],
    scale: f32,
    sinks: Option<&[f32]>,
    keys: impl Fn(usize, usize) -> Vec<usize>,
) -> (Vec<f32>, Vec<f32>) {
    let rows = q.len() / (qh * dqk);
    let (mut o, mut lse) = (vec![0f32; rows * qh * dv], vec![0f32; rows * qh]);
    for (r, h) in (0..rows).flat_map(|r| (0..qh).map(move |h| (r, h))) {
        let (kh, at) = (h / (qh / kvh), r * qh + h);
        let qv = &q[at * dqk..][..dqk];
        let cells = keys(r, h);
        let dot = |c: usize| -> f64 {
            let kv = &k[(c * kvh + kh) * dqk..][..dqk];
            qv.iter().zip(kv).map(|(a, b)| f64::from(a * b)).sum()
        };
        let s: Vec<f64> = cells.iter().map(|&c| dot(c) * f64::from(scale)).collect();
        let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
        lse[at] = ((m + z.ln()) * core::f64::consts::LOG2_E) as f32;
        let den = z + sinks.map_or(0.0, |s| (f64::from(s[h]) - m).exp());
        for (x, &c) in s.iter().zip(&cells) {
            let p = (x - m).exp() / den;
            let val = &v[(c * kvh + kh) * dv..][..dv];
            for (acc, val) in o[at * dv..][..dv].iter_mut().zip(val) {
                *acc += (p * f64::from(*val)) as f32;
            }
        }
    }
    (o, lse)
}

/// The last `dim` dims of `head` turned by `pos` (`rope_partial_last`):
/// interleaved pairs or split halves, backwards where `inverse`, the
/// frequencies ramped where `yarn`.
pub fn turn(head: &mut [f32], pos: i32, dim: usize, il: bool, inv: bool, yarn: Option<Yarn>) {
    let (half, off) = (dim / 2, head.len() - dim);
    let ramp = yarn.map(|y| {
        let max = y.original_max_position as i32;
        let (lo, hi) = ramp_bounds(dim as i32, 1e4, y.beta_fast, y.beta_slow, max);
        (lo, if hi == lo { 1e-3 } else { hi - lo }, y.factor)
    });
    for p in 0..half {
        let mut freq = 1e4f64.powf(-2.0 * p as f64 / dim as f64);
        if let Some((lo, den, factor)) = ramp {
            let r = f64::from(((p as f32 - lo) / den).clamp(0.0, 1.0));
            freq *= (1.0 - r) + r / f64::from(factor);
        }
        let (s, c) = ([1.0, -1.0][usize::from(inv)] * f64::from(pos) * freq).sin_cos();
        let (i, j) = [(off + p, off + p + half), (off + 2 * p, off + 2 * p + 1)][usize::from(il)];
        let (a, b) = (f64::from(head[i]), f64::from(head[j]));
        (head[i], head[j]) = ((a * c - b * s) as f32, (b * c + a * s) as f32);
    }
}

/// `x` RMS-normed (each `unit` apart), scaled by `w` where given.
pub fn rmsnorm(x: &[f32], unit: usize, w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    for c in y.chunks_mut(unit) {
        let ms = c.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / unit as f64;
        let inv = (1.0 / (ms + f64::from(eps)).sqrt()) as f32;
        for (i, v) in c.iter_mut().enumerate() {
            *v *= inv * w.map_or(1.0, |w| w[i]);
        }
    }
    y
}

/// bf16 bits as values.
pub fn wide(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| from_bf16(b)).collect()
}

/// Asserts `got` within `tol` of `want` relative to `want` (at least `floor`).
pub fn near(got: &[f32], want: &[f32], tol: f32, floor: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let err = |(g, w): (&f32, &f32)| (g - w).abs() / w.abs().max(floor);
    let worst = got.iter().zip(want).map(err).fold(0.0, f32::max);
    assert!(worst <= tol, "{what}: off by {worst} relative");
}

/// The bf16 values at `at` within `tol` of `want` (see [`near`]).
pub fn landed(gpu: &Gpu, at: u64, want: &[f32], tol: f32, floor: f32, what: &str) {
    gpu.sync();
    let got = wide(&gpu.down::<u16>(at, want.len()));
    near(&got, want, tol, floor, what);
}

/// `a . b`, summed in f64.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum::<f64>() as f32
}

/// The running offsets of `sizes`, from 0.
pub fn offsets(sizes: impl IntoIterator<Item = usize>) -> Vec<i32> {
    let mut out = vec![0i32];
    for n in sizes {
        out.push(out[out.len() - 1] + n as i32);
    }
    out
}

/// Each row's `(request, position)` over requests of `(rows, keys)`, the
/// rows the last of the keys.
pub fn owners(lens: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let each = |(r, &(n, kv)): (usize, &(usize, usize))| (kv - n..kv).map(move |p| (r, p));
    lens.iter().enumerate().flat_map(each).collect()
}

/// `p` reading `keys` and `values`.
pub fn keyed(p: &KvPool, keys: Tensor, values: Tensor) -> KvPool {
    KvPool {
        keys,
        values,
        seq_stride: i64::from(keys.width),
        ..*p
    }
}

/// [`Tuning::DEFAULT`] with `set` applied.
pub fn tuned(set: impl FnOnce(&mut Tuning)) -> Tuning {
    let mut t = Tuning::DEFAULT;
    set(&mut t);
    t
}

/// A `[rows][width]` tensor of `dtype` at `at`.
pub fn ten(at: u64, rows: usize, width: usize, dtype: Dtype) -> Tensor {
    Tensor::new(at, rows as u32, width as u32, dtype)
}

/// A bf16 `[rows][width]` plane of random values, its device tensor and values.
pub fn plane(gpu: &mut Gpu, lcg: &mut Lcg, rows: usize, width: usize) -> (Tensor, Vec<f32>) {
    let (raw, exact) = lcg.row(rows * width);
    (ten(gpu.up(&raw), rows, width, Dtype::Bf16), exact)
}

/// `values` on the device as an `[n][1]` tensor of `dtype`.
pub fn col<T: Copy>(gpu: &mut Gpu, values: &[T], dtype: Dtype) -> Tensor {
    ten(gpu.up(values), values.len(), 1, dtype)
}

/// The ids below `visible` of `sel`'s row `r` (`k` a row) before its first
/// -1; every visible id where `k` is 0.
pub fn picks(sel: &[i32], r: usize, k: usize, visible: usize) -> Vec<usize> {
    let row = sel[r * k..(r + 1) * k].iter().take_while(|&&c| c >= 0);
    let row: Vec<usize> = row.map(|&c| c as usize).collect();
    let ids = if k == 0 { (0..visible).collect() } else { row };
    ids.into_iter().filter(|&c| c < visible).collect()
}

/// A paged pool over `keys` (values the same plane), `page` cells a page.
pub fn pool(keys: Tensor, pages: Tensor, indptr: Tensor, page: usize) -> KvPool {
    KvPool {
        keys,
        values: keys,
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: pages,
        page_indptr: indptr,
        last_page_lens: Tensor::ABSENT,
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: page as i32,
        seq_stride: i64::from(keys.width),
        head_stride: i64::from(keys.width),
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: pages.rows as i32,
        pages_in_batch: pages.rows as i32,
        live: [0, 0],
    }
}

/// The smallest power of two at least `v` (a ue8m0 scale).
pub fn pow2_ceil(v: f32) -> f32 {
    let bits = v.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 + i32::from(bits & 0x7f_ffff != 0);
    2f32.powi(e - 127)
}

/// A block-scaled quantizer's read-back of `x`: DeepSeek-V4.1's MXFP8
/// (e4m3 under ue8m0, 32 a block), NVFP4 (e2m1 under e4m3, 16) and its
/// index keys' FP4 (e2m1 under ue8m0, 32).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quant {
    Mx8,
    Nv4,
    Fp4,
}

pub fn quantized(x: &[f32], fmt: Quant) -> Vec<f32> {
    let mut y = x.to_vec();
    for b in y.chunks_mut(if fmt == Quant::Nv4 { 16 } else { 32 }) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let (s, top) = match fmt {
            Quant::Mx8 => (pow2_ceil(amax.max(1e-4) / 448.0), 448.0),
            Quant::Nv4 => (e4m3(to_e4m3(amax.max(6.0 * 2f32.powi(-9)) / 6.0)), 6.0),
            Quant::Fp4 => (pow2_ceil(amax.max(6.0 * 2f32.powi(-126)) / 6.0), 6.0),
        };
        for v in b.iter_mut() {
            let c = (*v / s).clamp(-top, top);
            *v = s * if fmt == Quant::Mx8 {
                e4m3(to_e4m3(c))
            } else {
                e2m1(to_e2m1(c))
            };
        }
    }
    y
}
