#![cfg(feature = "cuda")]

use crate::common::spatial::{Box3, table};
use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::spatial::{Segment, attention};
use kernels_cuda::tensor::Tensor;

fn attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    boxes: &[Box3],
    c: usize,
    seg: usize,
    scale: f32,
) -> Vec<f32> {
    let rows = q.len() / c;
    let mut y = vec![0f32; rows * c];
    let mut off = 0usize;
    for b in boxes {
        let plane = b.plane();
        let per = if seg == 0 { b.t } else { seg };
        let mut frame = 0usize;
        while frame < b.t {
            let last = (frame + per).min(b.t);
            let begin = off + frame * plane;
            let end = off + last * plane;
            for i in begin..end {
                let mut s = vec![0f64; end - begin];
                let mut m = f64::NEG_INFINITY;
                for (jj, j) in (begin..end).enumerate() {
                    let dot: f64 = (0..c)
                        .map(|e| f64::from(q[i * c + e]) * f64::from(k[j * c + e]))
                        .sum();
                    s[jj] = dot * f64::from(scale);
                    m = m.max(s[jj]);
                }
                let p: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let l: f64 = p.iter().sum();
                for e in 0..c {
                    let acc: f64 = (0..end - begin)
                        .map(|jj| p[jj] * f64::from(v[(begin + jj) * c + e]))
                        .sum();
                    y[i * c + e] = (acc / l) as f32;
                }
            }
            frame = last;
        }
        off += b.voxels();
    }
    y
}

const C: usize = 1024;

const BOXES: [Box3; 2] = [Box3::new(5, 2, 3), Box3::new(3, 2, 3)];

fn run(gpu: &mut Gpu, at: (u64, u64, u64, u64, u64), rows: usize, segment: Segment, scale: f32) {
    let (q, k, v, grid, y) = at;
    let t = |ptr: u64| Tensor::new(ptr, rows as u32, C as u32, Dtype::Bf16);
    let mut o = t(y);
    attention(
        &gpu.ctx(),
        t(q),
        t(k),
        t(v),
        Tensor::new(grid, BOXES.len() as u32, 4, Dtype::I32),
        segment,
        scale,
        &mut o,
    )
    .unwrap_or_else(|why| panic!("{segment:?}: {why}"));
    gpu.sync();
}

#[test]
fn the_attention_blocks_per_frame_run_rather_than_per_clip() {
    let (grid, live) = table(&BOXES);
    let rows = live + 3;
    let scale = (C as f32).sqrt().recip() * 4.0;

    let mut lcg = Lcg::seeded(0x5e6);
    let (q_raw, q) = lcg.row(rows * C);
    let (k_raw, k) = lcg.row(rows * C);
    let (v_raw, v) = lcg.row(rows * C);

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let v_at = gpu.up(&v_raw);
    let grid_at = gpu.up(&grid);
    let y_at = gpu.zeros(rows * C * 2);
    let at = (q_at, k_at, v_at, grid_at, y_at);

    let mut answers = Vec::new();
    for (segment, seg) in [
        (Segment::Frames(1), 1usize),
        (Segment::Frames(2), 2),
        (Segment::Lane, 0),
    ] {
        let want = attention_ref(&q, &k, &v, &BOXES, C, seg, scale);
        run(&mut gpu, at, rows, segment, scale);
        let got: Vec<f32> = gpu
            .down::<u16>(y_at, rows * C)
            .into_iter()
            .map(from_bf16)
            .collect();

        let mut worst = 0f32;
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let err = (g - w).abs();
            worst = worst.max(err);
            assert!(
                err <= 1e-2 + 1e-2 * w.abs(),
                "{segment:?}: row {} channel {}: got {g}, want {w}",
                i / C,
                i % C
            );
        }
        assert!(
            got[live * C..].iter().all(|x| *x == 0.0),
            "{segment:?}: the padded rows past the last clip land zeros"
        );
        eprintln!("{segment:?}: {rows} rows of {C}, max |err| {worst:.5}");
        answers.push(got);
    }

    for (a, b) in [(0usize, 1usize), (1, 2), (0, 2)] {
        let moved = answers[a]
            .iter()
            .zip(&answers[b])
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            moved > 0.1,
            "answers {a} and {b} differ by only {moved}: the kernel is not \
             reading the segment it was handed"
        );
    }
}
