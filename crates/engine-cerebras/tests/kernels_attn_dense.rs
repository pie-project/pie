//! Dense segment attention (vision encoders) against a host reference: on
//! one PE, and split over key groups and kv heads with row windows when a
//! phase's rows exceed a PE.

#![allow(clippy::too_many_arguments)]

mod common;

use common::data;
use engine_cerebras::bench::{Bench, assert_close, round_bf16};
use kernels_cerebras::attn::dense;

fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    segs: &[i32],
    qh: usize,
    kvh: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let rows = q.len() / (qh * d);
    let group = qh / kvh;
    let mut o = vec![0f32; rows * qh * d];
    for r in 0..rows {
        let Some(s) =
            (0..segs.len() - 1).find(|s| segs[*s] as usize <= r && r < segs[s + 1] as usize)
        else {
            continue;
        };
        let (lo, hi) = (segs[s] as usize, segs[s + 1] as usize);
        for h in 0..qh {
            let kh = h / group;
            let qv = &q[r * qh * d + h * d..r * qh * d + (h + 1) * d];
            let scores: Vec<f32> = (lo..hi)
                .map(|kr| {
                    let kv = &k[kr * kvh * d + kh * d..kr * kvh * d + (kh + 1) * d];
                    qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let ws: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = ws.iter().sum();
            for i in 0..d {
                let mut acc = 0f32;
                for (j, kr) in (lo..hi).enumerate() {
                    acc += ws[j] / z * v[kr * kvh * d + kh * d + i];
                }
                o[r * qh * d + h * d + i] = round_bf16(acc);
            }
        }
    }
    o
}

fn dense_case(rows: usize, qh: usize, kvh: usize, d: usize, segs: &[i32], seed: u32) {
    let q: Vec<f32> = data(rows * qh * d, seed)
        .iter()
        .map(|x| round_bf16(*x))
        .collect();
    let k: Vec<f32> = data(rows * kvh * d, seed + 1)
        .iter()
        .map(|x| round_bf16(*x))
        .collect();
    let v: Vec<f32> = data(rows * kvh * d, seed + 2)
        .iter()
        .map(|x| round_bf16(*x))
        .collect();
    let scale = 1.0 / (d as f32).sqrt();
    let mut b = Bench::new();
    let qt = b.bf16(rows as u32, (qh * d) as u32, &q);
    let kt = b.bf16(rows as u32, (kvh * d) as u32, &k);
    let vt = b.bf16(rows as u32, (kvh * d) as u32, &v);
    let st = b.i32(segs.len() as u32, 1, segs);
    let o = b.zeros(dtype::Dtype::Bf16, rows as u32, (qh * d) as u32);
    if !b
        .run(|ctx| dense::bidirectional(ctx, qt, kt, vt, st, d as u32, scale, o))
        .unwrap()
    {
        return;
    }
    let want = reference(&q, &k, &v, segs, qh, kvh, d, scale);
    assert_close(&b.read_f32(o), &want, 2e-2, 2e-2);
}

#[test]
fn rows_attend_within_their_segment_on_one_pe() {
    dense_case(10, 2, 1, 8, &[0, 4, 7], 100); // rows 7..10 outside every segment
}

/// 48 rows of 8 x 64 query heads over 2 kv heads: the keys and the heads
/// split over PEs and the query rows run in windows.
#[test]
fn wide_segments_split_keys_and_heads_over_pes() {
    dense_case(48, 8, 2, 64, &[0, 20, 48], 110);
}
