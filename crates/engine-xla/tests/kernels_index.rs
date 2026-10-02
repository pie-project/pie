//! The sparse-attention indexer against host references.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::attn::index;
use kernels_xla::{KvPool, Tensor};

const PS: usize = 4;
/// Request 0 holds pages 3, 1, 4; request 1 pages 0, 2.
const PAGES: [&[u32]; 2] = [&[3, 1, 4], &[0, 2]];
const CELLS: usize = 5 * PS;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn bf16s(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| round_bf16(self.next() * scale)).collect()
    }
}

fn cell(req: usize, pos: usize) -> usize {
    PAGES[req][pos / PS] as usize * PS + pos % PS
}

fn kv_pool(b: &mut Bench, keys: Tensor) -> KvPool {
    let indices: Vec<u32> = PAGES.iter().flat_map(|p| p.iter().copied()).collect();
    let pi = b.u32(indices.len() as u32, 1, &indices);
    let pp = b.u32(3, 1, &[0, 3, 5]);
    KvPool {
        keys,
        values: keys,
        page_indices: pi,
        page_indptr: pp,
        page_size: PS as i32,
        max_pages: 3,
        seq_stride: u64::from(keys.width),
        head_stride: u64::from(keys.width),
    }
}

/// Interleaved-pair rope of the first `rope_dim` lanes of `x`.
fn rope_ref(x: &mut [f32], pos: i32, rope_dim: usize, theta: f32) {
    for i in 0..rope_dim / 2 {
        let freq = theta.powf(-2.0 * i as f32 / rope_dim as f32);
        let ang = pos as f32 * freq;
        let (c, s) = (ang.cos(), ang.sin());
        let (a, b) = (x[2 * i], x[2 * i + 1]);
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = b * c + a * s;
    }
}

#[test]
fn the_index_key_and_query_are_normed_and_roped_in_place() {
    let (rows, hd, rope_dim, theta, eps) = (5usize, 16usize, 8usize, 10_000.0f32, 1e-6f32);
    let mut rng = Rng(1);
    let k = rng.bf16s(rows * hd, 2.0);
    let w = rng.bf16s(hd, 1.0);
    let bias = rng.bf16s(hd, 0.5);
    let positions = [0i32, 1, 17, 300, 4095];
    let mut b = Bench::new();
    let kt = b.bf16(rows as u32, hd as u32, &k);
    let pt = b.i32(rows as u32, 1, &positions);
    let wt = b.bf16(1, hd as u32, &w);
    let bt = b.bf16(1, hd as u32, &bias);
    if !b
        .run(|ctx| index::layernorm_rope(ctx, kt, pt, wt, bt, eps, rope_dim as u32, theta))
        .unwrap()
    {
        return;
    }
    let mut want = Vec::new();
    for r in 0..rows {
        let x = &k[r * hd..(r + 1) * hd];
        let mean = x.iter().sum::<f32>() / hd as f32;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hd as f32;
        let inv = 1.0 / (var + eps).sqrt();
        let mut y: Vec<f32> = (0..hd)
            .map(|d| round_bf16((x[d] - mean) * inv * w[d] + bias[d]))
            .collect();
        rope_ref(&mut y, positions[r], rope_dim, theta);
        want.extend(y);
    }
    assert_close(&b.read_f32(kt), &want, 3e-2, 2e-2);

    // The query: three heads of 8, the first 4 lanes of each roped.
    let (heads, qd, qrope) = (3usize, 8usize, 4usize);
    let q = rng.bf16s(rows * heads * qd, 1.0);
    let mut b = Bench::new();
    let qt = b.bf16(rows as u32, (heads * qd) as u32, &q);
    let pt = b.i32(rows as u32, 1, &positions);
    b.run(|ctx| index::rope(ctx, qt, pt, heads as u32, qd as u32, qrope as u32, theta))
        .unwrap();
    let mut want = q.clone();
    for (r, &pos) in positions.iter().enumerate().take(rows) {
        for h in 0..heads {
            let at = (r * heads + h) * qd;
            rope_ref(&mut want[at..at + qd], pos, qrope, theta);
        }
    }
    assert_close(&b.read_f32(qt), &want, 3e-2, 2e-2);
}

#[test]
fn index_keys_are_filed_and_block_averaged() {
    let hd = 8usize;
    let mut rng = Rng(2);
    let keys0 = rng.bf16s(CELLS * hd, 1.0);
    let k = rng.bf16s(4 * hd, 1.0);
    let wpage = [2u32, u32::MAX, 4, 0];
    let woff = [3u32, 0, 1, 7];
    let mut b = Bench::new();
    let kt = b.bf16(4, hd as u32, &k);
    let keys = b.bf16(CELLS as u32, hd as u32, &keys0);
    let wp = b.u32(4, 1, &wpage);
    let wo = b.u32(4, 1, &woff);
    let pool = kv_pool(&mut b, keys);
    if !b
        .run(|ctx| index::kv_append(ctx, kt, &pool, wp, wo))
        .unwrap()
    {
        return;
    }
    let mut want = keys0.clone();
    for r in 0..4 {
        if (wpage[r] as i32) < 0 || woff[r] as usize >= PS {
            continue;
        }
        let at = wpage[r] as usize * PS + woff[r] as usize;
        want[at * hd..(at + 1) * hd].copy_from_slice(&k[r * hd..(r + 1) * hd]);
    }
    assert_close(&b.read_f32(keys), &want, 0.0, 0.0);

    let ratio = 4usize;
    let bpos = [3i32, -1, 11, 1, 7];
    let breq = [0i32, 0, 0, 1, 1];
    let mut b = Bench::new();
    let bp = b.i32(5, 1, &bpos);
    let br = b.i32(5, 1, &breq);
    let keys = b.bf16(CELLS as u32, hd as u32, &keys0);
    let out = b.zeros(Dtype::Bf16, 5, hd as u32);
    let pool = kv_pool(&mut b, keys);
    b.run(|ctx| index::block_mean(ctx, bp, br, &pool, hd as u32, ratio as u32, out))
        .unwrap();
    let mut want = vec![0.0; 5 * hd];
    for r in 0..5 {
        if bpos[r] < 0 {
            continue;
        }
        for i in 0..ratio {
            let pos = bpos[r] + i as i32 - (ratio as i32 - 1);
            if pos < 0 {
                continue;
            }
            let c = cell(breq[r] as usize, pos as usize);
            for d in 0..hd {
                want[r * hd + d] += keys0[c * hd + d];
            }
        }
        for d in 0..hd {
            want[r * hd + d] /= ratio as f32;
        }
    }
    assert_close(&b.read_f32(out), &want, 1e-2, 1e-2);
}

/// The GPU's bisected pick (kernels-wgpu `attn::index::bisect_select`).
fn bisect_select(scores: &[f32], topk: usize) -> Vec<i32> {
    let nkeys = scores.len();
    if nkeys <= topk {
        return (0..topk)
            .map(|n| if n < nkeys { n as i32 } else { -1 })
            .collect();
    }
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for s in scores {
        lo = lo.min(*s);
        hi = hi.max(*s);
    }
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let cnt = scores.iter().filter(|s| **s >= mid).count();
        if cnt > topk {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mut out: Vec<i32> = (0..nkeys)
        .filter(|&j| scores[j] >= hi)
        .map(|j| j as i32)
        .take(topk)
        .collect();
    out.resize(topk, -1);
    out
}

#[test]
fn the_index_ranks_cached_keys_and_keeps_the_top_k() {
    let (heads, d, top_k) = (3usize, 8usize, 3usize);
    let mut rng = Rng(5);
    let keys0 = rng.bf16s(CELLS * d, 1.0);
    let rows = 5usize;
    let positions = [1i32, 11, 6, 7, 0];
    let reqs = [0i32, 0, 1, 1, 1];
    let q = rng.bf16s(rows * heads * d, 1.0);
    let w = rng.bf16s(rows * heads, 1.0);
    for (ratio, weighted) in [(1usize, true), (2, false)] {
        let mut b = Bench::new();
        let qt = b.bf16(rows as u32, (heads * d) as u32, &q);
        let wt = b.bf16(rows as u32, heads as u32, &w);
        let keys = b.bf16(CELLS as u32, d as u32, &keys0);
        let pt = b.i32(rows as u32, 1, &positions);
        let rt = b.i32(rows as u32, 1, &reqs);
        let sel = b.zeros(Dtype::I32, rows as u32, top_k as u32);
        let pool = kv_pool(&mut b, keys);
        if !b
            .run(|ctx| {
                index::topk(
                    ctx,
                    qt,
                    weighted.then_some(wt),
                    &pool,
                    pt,
                    rt,
                    heads as u32,
                    d as u32,
                    top_k as u32,
                    ratio as u32,
                    sel,
                )
            })
            .unwrap()
        {
            return;
        }
        let got = b.read_i32(sel);
        for r in 0..rows {
            let nkeys = ((positions[r] + 1) as usize / ratio).min(3 * PS / ratio);
            let scores: Vec<f32> = (0..nkeys)
                .map(|j| {
                    let c = cell(reqs[r] as usize, (j + 1) * ratio - 1);
                    (0..heads)
                        .map(|h| {
                            let qh = &q[(r * heads + h) * d..(r * heads + h + 1) * d];
                            let dot: f32 = qh
                                .iter()
                                .zip(&keys0[c * d..(c + 1) * d])
                                .map(|(a, b)| a * b)
                                .sum();
                            dot.max(0.0) * if weighted { w[r * heads + h] } else { 1.0 }
                        })
                        .sum()
                })
                .collect();
            assert_eq!(
                &got[r * top_k..(r + 1) * top_k],
                &bisect_select(&scores, top_k)[..],
                "ratio {ratio} row {r} scores {scores:?}"
            );
        }
    }
}
