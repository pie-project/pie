//! Gated DeltaNet against a host reference that rounds where the device
//! does: the conv (decode and prefill, with its window state), the gates, the
//! one-token step and the per-request scan.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_abs, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::attn::ssm;
use kernels_rocm::{RaggedTensor, RecurrentPool, Tensor};

struct Geo {
    k_heads: usize,
    v_heads: usize,
    k_dim: usize,
    v_dim: usize,
    conv: usize,
}

impl Geo {
    fn conv_dim(&self) -> usize {
        2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim
    }
    fn state_elems(&self) -> usize {
        self.v_heads * self.k_dim * self.v_dim
    }
}

fn pool(slab: Tensor, conv_slab: Tensor, slots: Tensor, g: &Geo) -> RecurrentPool {
    RecurrentPool {
        slab,
        slot_ids: slots,
        slot_stride_elems: g.state_elems() as i64,
        conv_slab,
        conv_stride: ((g.conv - 1 + 1) * g.conv_dim()) as i64,
        write_state: true,
        write_state_mask: Tensor::ABSENT,
        commit_len: Tensor::ABSENT,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
    }
}

fn silu(z: f32) -> f32 {
    z / (1.0 + (-z).exp())
}

/// One token's conv over its window `win` (oldest first, `conv` rows ...
/// conv - 1 of history plus the new input), per channel.
fn conv_row(x: &[f32], hist: &[f32], w: &[f32], k: usize, c: usize) -> Vec<f32> {
    // hist holds the k - 1 previous inputs, oldest first, [k-1][c].
    (0..c)
        .map(|ch| {
            let mut acc = 0.0;
            for i in 0..k - 1 {
                acc += w[ch * k + i] * hist[i * c + ch];
            }
            acc += w[ch * k + k - 1] * x[ch];
            round_bf16(silu(acc))
        })
        .collect()
}

/// The recurrence over `n` tokens of one value head, rounding the state as
/// the device does. `s` is [k_dim][v_dim]; returns the outputs [n][v_dim].
#[allow(clippy::too_many_arguments)]
fn delta(
    s: &mut [f32],
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    g_log: &[f32],
    beta: &[f32],
    kd: usize,
    vd: usize,
) -> Vec<Vec<f32>> {
    let mut outs = Vec::new();
    for t in 0..q.len() {
        let g = g_log[t].exp();
        let mut o = vec![0f32; vd];
        for j in 0..vd {
            let kv: f32 = (0..kd).map(|i| s[i * vd + j] * g * k[t][i]).sum();
            let d = (v[t][j] - kv) * beta[t];
            let mut acc = 0.0;
            for i in 0..kd {
                let next = round_bf16(s[i * vd + j] * g) + k[t][i] * d;
                s[i * vd + j] = round_bf16(next);
                acc += next * q[t][i];
            }
            o[j] = acc;
        }
        outs.push(o);
    }
    outs
}

fn l2(x: &[f32], scale: f32) -> Vec<f32> {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt() * scale;
    x.iter().map(|v| v * inv).collect()
}

#[test]
fn attn_ssm() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let g = Geo {
        k_heads: 4,
        v_heads: 8,
        k_dim: 128,
        v_dim: 128,
        conv: 4,
    };
    let c = g.conv_dim();
    let k = g.conv;
    let slots_n = 4usize;

    // --- conv, decode: 3 rows on slots 2, 0, 3.
    let w: Vec<f32> = values(c * k, 1, 0.5).into_iter().map(round_bf16).collect();
    let wt = gpu.tensor(&w, c as u32, k as u32, Dtype::Bf16);
    let window = k; // (K-1)·dil + 1 rows
    let conv_state: Vec<f32> = values(slots_n * window * c, 2, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let conv_slab = gpu.tensor(
        &conv_state,
        (slots_n * window) as u32,
        c as u32,
        Dtype::Bf16,
    );
    let slab_v: Vec<f32> = values(slots_n * g.state_elems(), 3, 0.2)
        .into_iter()
        .map(round_bf16)
        .collect();
    let slab = gpu.tensor(&slab_v, slots_n as u32, g.state_elems() as u32, Dtype::Bf16);
    let slot_ids = [2i32, 0, 3];
    let slots = gpu.column(&slot_ids);
    let state = pool(slab, conv_slab, slots, &g);
    let xs: Vec<f32> = values(3 * c, 4, 2.0).into_iter().map(round_bf16).collect();
    let x = gpu.tensor(&xs, 3, c as u32, Dtype::Bf16);
    let mut y = gpu.zeros(3, c as u32, Dtype::Bf16);
    ssm::causal_conv1d(&ctx, x, wt, &state, k as u32, 1, &mut y).unwrap();
    let got = gpu.read(&y);
    let got_state = gpu.read(&conv_slab);
    for (r, &slot) in slot_ids.iter().enumerate() {
        let base = slot as usize * window * c;
        let hist = &conv_state[base + c..base + window * c];
        let want = conv_row(&xs[r * c..(r + 1) * c], hist, &w, k, c);
        close_abs(
            &format!("conv decode row {r}"),
            &got[r * c..(r + 1) * c],
            &want,
            1e-2,
        );
        let mut shifted = conv_state[base + c..base + window * c].to_vec();
        shifted.extend_from_slice(&xs[r * c..(r + 1) * c]);
        close_abs(
            &format!("conv decode state {r}"),
            &got_state[base..base + window * c],
            &shifted,
            0.0,
        );
    }

    // --- conv, prefill: requests of 5, 1 and 40 rows on slots 1, 3, 0.
    let lens = [5usize, 1, 40];
    let total: usize = lens.iter().sum();
    let conv_slab = gpu.tensor(
        &conv_state,
        (slots_n * window) as u32,
        c as u32,
        Dtype::Bf16,
    );
    let slot_ids = [1i32, 3, 0];
    let state = pool(slab, conv_slab, gpu.column(&slot_ids), &g);
    let xs: Vec<f32> = values(total * c, 5, 2.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let mut indptr = vec![0i32];
    for l in lens {
        indptr.push(indptr.last().unwrap() + l as i32);
    }
    let x = RaggedTensor {
        data: gpu.tensor(&xs, total as u32, c as u32, Dtype::Bf16),
        indptr: gpu.column(&indptr),
    };
    let mut y = gpu.zeros(total as u32, c as u32, Dtype::Bf16);
    ssm::causal_conv1d_chunked(&ctx, x, wt, &state, k as u32, 1, &mut y).unwrap();
    let got = gpu.read(&y);
    let got_state = gpu.read(&conv_slab);
    for (r, &slot) in slot_ids.iter().enumerate() {
        let base = slot as usize * window * c;
        let mut hist: Vec<f32> = conv_state[base..base + window * c].to_vec();
        for t in 0..lens[r] {
            let row = indptr[r] as usize + t;
            let xr = &xs[row * c..(row + 1) * c];
            let want = conv_row(xr, &hist[c..], &w, k, c);
            close_abs(
                &format!("conv prefill req {r} row {t}"),
                &got[row * c..(row + 1) * c],
                &want,
                1e-2,
            );
            hist.drain(..c);
            hist.extend_from_slice(xr);
        }
        close_abs(
            &format!("conv prefill state {r}"),
            &got_state[base..base + window * c],
            &hist,
            0.0,
        );
    }

    // --- gates: [b | a] → [g_log | β].
    let rows = 6usize;
    let ba: Vec<f32> = values(rows * 2 * g.v_heads, 6, 3.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let dt: Vec<f32> = values(g.v_heads, 7, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let a_log = values(g.v_heads, 8, 1.0);
    let mut gates = gpu.zeros(rows as u32, 2 * g.v_heads as u32, Dtype::F32);
    ssm::gdn_prep(
        &ctx,
        gpu.tensor(&ba, rows as u32, 2 * g.v_heads as u32, Dtype::Bf16),
        gpu.tensor(&dt, 1, g.v_heads as u32, Dtype::Bf16),
        gpu.tensor(&a_log, 1, g.v_heads as u32, Dtype::F32),
        &mut gates,
    )
    .unwrap();
    let got = gpu.read(&gates);
    let vh = g.v_heads;
    for r in 0..rows {
        for h in 0..vh {
            let z = ba[r * 2 * vh + vh + h] + dt[h];
            let sp = if z > 20.0 { z } else { z.exp().ln_1p() };
            let want_g = -a_log[h].exp() * sp;
            let want_b = 1.0 / (1.0 + (-ba[r * 2 * vh + h]).exp());
            close_abs(
                "g_log",
                &[got[r * 2 * vh + h]],
                &[want_g],
                1e-4 * want_g.abs().max(1.0),
            );
            close_abs("beta", &[got[r * 2 * vh + vh + h]], &[want_b], 1e-5);
        }
    }

    // The recurrence's inputs: post-conv qkv rows, and gates kept in a sane
    // range (decay in (0, 1), β in (0, 1)).
    let make = |n: usize, seed: u64| -> (Vec<f32>, Vec<f32>) {
        let qkv: Vec<f32> = values(n * c, seed, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let mut gates = Vec::new();
        for (i, v) in values(n * 2 * vh, seed + 1, 1.0).into_iter().enumerate() {
            gates.push(if i % (2 * vh) < vh {
                -0.05 - 0.5 * v.abs()
            } else {
                0.5 + 0.45 * v
            });
        }
        (qkv, gates)
    };
    let heads_of = |qkv: &[f32], gates: &[f32], row: usize, h: usize| {
        let kh = h / (g.v_heads / g.k_heads);
        let base = row * c;
        let q = l2(
            &qkv[base + kh * g.k_dim..base + (kh + 1) * g.k_dim],
            1.0 / (g.k_dim as f32).sqrt(),
        );
        let kk = l2(
            &qkv[base + g.k_heads * g.k_dim + kh * g.k_dim
                ..base + g.k_heads * g.k_dim + (kh + 1) * g.k_dim],
            1.0,
        );
        let v = qkv[base + 2 * g.k_heads * g.k_dim + h * g.v_dim
            ..base + 2 * g.k_heads * g.k_dim + (h + 1) * g.v_dim]
            .to_vec();
        (
            q,
            kk,
            v,
            gates[row * 2 * vh + h],
            gates[row * 2 * vh + vh + h],
        )
    };
    let head_state = |slab: &[f32], slot: usize, h: usize| -> Vec<f32> {
        let at = slot * g.state_elems() + h * g.k_dim * g.v_dim;
        slab[at..at + g.k_dim * g.v_dim].to_vec()
    };

    // --- the step: 3 rows on slots 2, 0, 3.
    let (qkv, gt) = make(3, 9);
    let slab = gpu.tensor(&slab_v, slots_n as u32, g.state_elems() as u32, Dtype::Bf16);
    let slot_ids = [2i32, 0, 3];
    let state = pool(slab, Tensor::ABSENT, gpu.column(&slot_ids), &g);
    let mut y = gpu.zeros(3, (vh * g.v_dim) as u32, Dtype::F32);
    ssm::gated_delta(
        &ctx,
        gpu.tensor(&qkv, 3, c as u32, Dtype::Bf16),
        Tensor::ABSENT,
        gpu.tensor(&gt, 3, 2 * vh as u32, Dtype::F32),
        &state,
        g.k_heads as u32,
        vh as u32,
        g.k_dim as u32,
        g.v_dim as u32,
        &mut y,
    )
    .unwrap();
    let got = gpu.read(&y);
    let got_slab = gpu.read(&slab);
    for (r, &slot) in slot_ids.iter().enumerate() {
        for h in 0..vh {
            let (q, kk, v, gl, b) = heads_of(&qkv, &gt, r, h);
            let mut s = head_state(&slab_v, slot as usize, h);
            let o = delta(&mut s, &[q], &[kk], &[v], &[gl], &[b], g.k_dim, g.v_dim);
            let at = (r * vh + h) * g.v_dim;
            close_abs(
                &format!("step row {r} head {h}"),
                &got[at..at + g.v_dim],
                &o[0],
                2e-3,
            );
            close_abs(
                &format!("step state row {r} head {h}"),
                &head_state(&got_slab, slot as usize, h),
                &s,
                8e-3,
            );
        }
    }

    // --- the scan: requests of 1, 9 and 70 rows on slots 3, 1, 2; the last
    // one starts 4 rows in and commits 50 of the rest.
    let lens = [1usize, 9, 70];
    let total: usize = lens.iter().sum();
    let (qkv, gt) = make(total, 11);
    let slab = gpu.tensor(&slab_v, slots_n as u32, g.state_elems() as u32, Dtype::Bf16);
    let slot_ids = [3i32, 1, 2];
    let mut state = pool(slab, Tensor::ABSENT, gpu.column(&slot_ids), &g);
    let begin = [0i32, 0, 4];
    let commit = [1i32, 9, 50];
    state.begin_at = gpu.column(&begin);
    state.commit_len = gpu.column(&commit);
    let mut indptr = vec![0i32];
    for l in lens {
        indptr.push(indptr.last().unwrap() + l as i32);
    }
    let qkv_t = RaggedTensor {
        data: gpu.tensor(&qkv, total as u32, c as u32, Dtype::Bf16),
        indptr: gpu.column(&indptr),
    };
    let mut y = gpu.zeros(total as u32, (vh * g.v_dim) as u32, Dtype::F32);
    ssm::gated_delta_chunked(
        &ctx,
        qkv_t,
        Tensor::ABSENT,
        gpu.tensor(&gt, total as u32, 2 * vh as u32, Dtype::F32),
        &state,
        g.k_heads as u32,
        vh as u32,
        g.k_dim as u32,
        g.v_dim as u32,
        &mut y,
    )
    .unwrap();
    let got = gpu.read(&y);
    let got_slab = gpu.read(&slab);
    for (r, &slot) in slot_ids.iter().enumerate() {
        let first = indptr[r] as usize + begin[r] as usize;
        let n = (lens[r] - begin[r] as usize).min(commit[r] as usize);
        for h in 0..vh {
            let (mut qs, mut ks, mut vs, mut gs, mut bs) = (vec![], vec![], vec![], vec![], vec![]);
            for t in first..first + n {
                let (q, kk, v, gl, b) = heads_of(&qkv, &gt, t, h);
                qs.push(q);
                ks.push(kk);
                vs.push(v);
                gs.push(gl);
                bs.push(b);
            }
            let mut s = head_state(&slab_v, slot as usize, h);
            let outs = delta(&mut s, &qs, &ks, &vs, &gs, &bs, g.k_dim, g.v_dim);
            for (t, o) in outs.iter().enumerate() {
                let at = ((first + t) * vh + h) * g.v_dim;
                close_abs(
                    &format!("scan req {r} head {h} row {t}"),
                    &got[at..at + g.v_dim],
                    o,
                    5e-3,
                );
            }
            close_abs(
                &format!("scan state req {r} head {h}"),
                &head_state(&got_slab, slot as usize, h),
                &s,
                1.6e-2,
            );
        }
    }
}
