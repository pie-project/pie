//! Mamba-2's step and prefill scan against the serial recurrence on the
//! host, the gated output group-normed.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, STATE, bf16s, close_abs, indptr_of, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::attn::ssm::{self, Mamba2};
use kernels_rocm::{RaggedTensor, RecurrentPool, Tensor};

const H: usize = 4;
const G: usize = 2;
const P: usize = 64;
const N: usize = 128;
const INNER: usize = H * P;
const WIDTH: usize = INNER + 2 * G * N;
const HEAD: usize = P * N;
const SLOTS: usize = 4;
const EPS: f32 = 1e-5;

struct Case {
    xbc: Vec<f32>,
    dt: Vec<f32>,
    z: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    d: Vec<f32>,
    norm: Vec<f32>,
    state: Vec<f32>,
}

impl Case {
    fn new(rows: usize) -> Self {
        Self {
            xbc: bf16s(rows * WIDTH, 1, 1.0),
            dt: bf16s(rows * H, 2, 2.0),
            z: bf16s(rows * INNER, 3, 2.0),
            a_log: values(H, 4, 1.0),
            dt_bias: values(H, 5, 1.0),
            d: values(H, 6, 1.0),
            norm: bf16s(INNER, 7, 1.0),
            state: values(SLOTS * H * HEAD, 8, 0.5),
        }
    }

    /// The device's operands: the weights, the inputs and the pool.
    fn upload(&self, gpu: &Gpu, rows: u32, slots: &[i32]) -> (Mamba2, [Tensor; 3], RecurrentPool) {
        let m = Mamba2 {
            a_log: gpu.tensor(&self.a_log, 1, H as u32, Dtype::F32),
            dt_bias: gpu.tensor(&self.dt_bias, 1, H as u32, Dtype::F32),
            d: gpu.tensor(&self.d, 1, H as u32, Dtype::F32),
            norm: gpu.tensor(&self.norm, 1, INNER as u32, Dtype::Bf16),
            eps: EPS,
            heads: H as u32,
            groups: G as u32,
        };
        let inputs = [
            gpu.tensor(&self.xbc, rows, WIDTH as u32, Dtype::Bf16),
            gpu.tensor(&self.dt, rows, H as u32, Dtype::Bf16),
            gpu.tensor(&self.z, rows, INNER as u32, Dtype::Bf16),
        ];
        let pool = RecurrentPool {
            slab: gpu.tensor(&self.state, SLOTS as u32, (H * HEAD) as u32, Dtype::F32),
            slot_ids: gpu.column(slots),
            slot_stride_elems: (H * HEAD) as i64,
            ..STATE
        };
        (m, inputs, pool)
    }

    /// Row `row` through `slot`'s state on the host: the normed output row.
    fn token(&self, state: &mut [f32], slot: usize, row: usize) -> Vec<f32> {
        let at = &self.xbc[row * WIDTH..(row + 1) * WIDTH];
        let mut y = vec![0f32; INNER];
        for h in 0..H {
            let g = h / (H / G);
            let raw = self.dt[row * H + h] + self.dt_bias[h];
            let step = if raw > 20.0 { raw } else { raw.exp().ln_1p() };
            let decay = (-self.a_log[h].exp() * step).exp();
            let s = &mut state[(slot * H + h) * HEAD..(slot * H + h + 1) * HEAD];
            for p in 0..P {
                let x = at[h * P + p];
                let mut acc = 0f32;
                for n in 0..N {
                    let v = s[p * N + n] * decay + step * x * at[INNER + g * N + n];
                    s[p * N + n] = v;
                    acc += v * at[INNER + (G + g) * N + n];
                }
                let z = self.z[row * INNER + h * P + p];
                y[h * P + p] = round_bf16((acc + self.d[h] * x) * z / (1.0 + (-z).exp()));
            }
        }
        for (i, c) in y.chunks_mut(INNER / G).enumerate() {
            let inv = 1.0 / (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32 + EPS).sqrt();
            for (j, v) in c.iter_mut().enumerate() {
                *v = round_bf16(*v * inv * self.norm[i * (INNER / G) + j]);
            }
        }
        y
    }
}

#[test]
fn attn_mamba2_step() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let slots = [2i32, 0, 3];
    let rows = slots.len();
    let case = Case::new(rows);
    let (m, [xbc, dt, z], pool) = case.upload(&gpu, rows as u32, &slots);
    let mut y = gpu.zeros(rows as u32, INNER as u32, Dtype::Bf16);
    ssm::mamba2_step(&ctx, xbc, dt, z, m, &pool, N as u32, &mut y).unwrap();
    let got = gpu.read(&y);
    let mut state = case.state.clone();
    for (r, &slot) in slots.iter().enumerate() {
        let want = case.token(&mut state, slot as usize, r);
        close_abs(&format!("step row {r}"), &got[r * INNER..(r + 1) * INNER], &want, 3e-2);
    }
    close_abs("step state", &gpu.read(&pool.slab), &state, 1e-4);
}

#[test]
fn attn_mamba2_scan() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let lens = [5u32, 1, 40];
    let slots = [1i32, 3, 0];
    let indptr = indptr_of(&lens);
    let rows = *indptr.last().unwrap() as usize;
    let case = Case::new(rows);
    let (m, [xbc, dt, z], pool) = case.upload(&gpu, rows as u32, &slots);
    let xbc = RaggedTensor {
        data: xbc,
        indptr: gpu.column(&indptr),
    };
    let mut y = gpu.zeros(rows as u32, INNER as u32, Dtype::Bf16);
    ssm::mamba2_chunked(&ctx, xbc, dt, z, m, &pool, N as u32, &mut y).unwrap();
    let got = gpu.read(&y);
    let mut state = case.state.clone();
    for (r, &slot) in slots.iter().enumerate() {
        for row in indptr[r] as usize..indptr[r + 1] as usize {
            let want = case.token(&mut state, slot as usize, row);
            close_abs(&format!("scan row {row}"), &got[row * INNER..(row + 1) * INNER], &want, 3e-2);
        }
    }
    close_abs("scan state", &gpu.read(&pool.slab), &state, 1e-3);
}
