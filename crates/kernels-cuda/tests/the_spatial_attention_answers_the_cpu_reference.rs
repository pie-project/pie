#![cfg(feature = "cuda")]

mod common;

use common::spatial::{Box3, table};
use common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::spatial::{Segment, attention};
use kernels_cuda::tensor::Tensor;

fn attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    boxes: &[Box3],
    c: usize,
    scale: f32,
) -> Vec<f32> {
    let rows = q.len() / c;
    let mut y = vec![0f32; rows * c];
    let mut off = 0usize;
    for b in boxes {
        let n = b.voxels();
        for i in off..off + n {
            let mut s = vec![0f64; n];
            let mut m = f64::NEG_INFINITY;
            for (jj, j) in (off..off + n).enumerate() {
                let dot: f64 = (0..c)
                    .map(|e| f64::from(q[i * c + e]) * f64::from(k[j * c + e]))
                    .sum();
                s[jj] = dot * f64::from(scale);
                m = m.max(s[jj]);
            }
            let p: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let l: f64 = p.iter().sum();
            for e in 0..c {
                let acc: f64 = (0..n)
                    .map(|jj| p[jj] * f64::from(v[(off + jj) * c + e]))
                    .sum();
                y[i * c + e] = (acc / l) as f32;
            }
        }
        off += n;
    }
    y
}

fn check(c: usize) {
    let boxes = [Box3::new(1, 5, 7), Box3::new(2, 3, 4)];
    let (grid, live) = table(&boxes);
    let rows = live + 5;
    let scale = (c as f32).sqrt().recip();
    let mut lcg = Lcg::seeded(0xa77e ^ c as u64);
    let (q_raw, q) = lcg.row(rows * c);
    let (k_raw, k) = lcg.row(rows * c);
    let (v_raw, v) = lcg.row(rows * c);
    let want = attention_ref(&q, &k, &v, &boxes, c, scale * 4.0);

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let v_at = gpu.up(&v_raw);
    let grid_at = gpu.up(&grid);
    let y_at = gpu.zeros(rows * c * 2);
    let t = |at: u64| Tensor::new(at, rows as u32, c as u32, Dtype::Bf16);
    let mut y = t(y_at);
    attention(
        &gpu.ctx(),
        t(q_at),
        t(k_at),
        t(v_at),
        Tensor::new(grid_at, boxes.len() as u32, 4, Dtype::I32),
        Segment::Lane,
        scale * 4.0,
        &mut y,
    )
    .unwrap_or_else(|why| panic!("C {c}: {why}"));
    gpu.sync();
    let got: Vec<f32> = gpu
        .down::<u16>(y_at, rows * c)
        .into_iter()
        .map(from_bf16)
        .collect();

    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let err = (g - w).abs();
        worst = worst.max(err);
        assert!(
            err <= 1e-2 + 1e-2 * w.abs(),
            "C {c}: row {} channel {}: got {g}, want {w}",
            i / c,
            i % c
        );
    }
    assert!(
        got[live * c..].iter().all(|x| *x == 0.0),
        "C {c}: the padded rows past the last lane land zeros"
    );
    eprintln!("C {c}: {rows} rows, max |err| {worst:.5}");
}

#[test]
fn the_spatial_attention_answers_the_cpu_reference_every_case() {
    the_attention_answers_the_reference_at_256_512_and_1024_channels();
    the_attention_answers_the_reference_at_the_scalar_640_width();
}

fn the_attention_answers_the_reference_at_256_512_and_1024_channels() {
    check(256);
    check(512);
    check(1024);
}

fn the_attention_answers_the_reference_at_the_scalar_640_width() {
    check(640);
}
