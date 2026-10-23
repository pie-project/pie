//! Attention over a voxel grid against the host's f64 softmax: whole lanes
//! at every width the kernel is stamped at, and segmented by frame (one,
//! two, the whole lane), each segment landing its own answer.

#![cfg(feature = "rocm")]

mod common;
#[path = "common/spatial.rs"]
mod spatial;

use common::{Gpu, bf16s};
use dtype::Dtype;
use kernels_rocm::attn::dense::voxel::{Segment, attention};
use spatial::{Box3, spatial_attention_ref, table};

/// Rows past the last lane are part of the launch and must land zero.
const SLACK: usize = 5;

fn run(gpu: &Gpu, boxes: &[Box3], c: usize, segment: Segment, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let (grid, live) = table(boxes);
    let rows = live + SLACK;
    let scale = (c as f32).sqrt().recip() * 4.0;
    let q = bf16s(rows * c, seed, 1.0);
    let k = bf16s(rows * c, seed + 1, 1.0);
    let v = bf16s(rows * c, seed + 2, 1.0);
    let seg = match segment {
        Segment::Lane => 0,
        Segment::Frames(n) => n as usize,
    };
    let want = spatial_attention_ref([&q, &k, &v], boxes, c, seg, scale);
    let t = |vals: &[f32]| gpu.tensor(vals, rows as u32, c as u32, Dtype::Bf16);
    let mut y = t(&vec![f32::NAN; rows * c]);
    attention(
        &gpu.ctx(),
        t(&q),
        t(&k),
        t(&v),
        gpu.ints_shaped(&grid, boxes.len() as u32, 4),
        segment,
        scale,
        &mut y,
    )
    .unwrap_or_else(|why| panic!("C {c} {segment:?}: {why}"));
    let got = gpu.read(&y);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-2 + 1e-2 * w.abs(),
            "C {c} {segment:?}: row {} channel {}: got {g}, want {w}",
            i / c,
            i % c
        );
    }
    assert!(
        got[live * c..].iter().all(|x| *x == 0.0),
        "C {c} {segment:?}: the padded rows past the last lane land zeros"
    );
    (got, want)
}

#[test]
fn the_spatial_attention_answers_the_cpu_reference_every_case() {
    let Some(gpu) = Gpu::open() else { return };
    for c in [256, 512, 640, 1024] {
        run(
            &gpu,
            &[Box3::new(1, 5, 7), Box3::new(2, 3, 4)],
            c,
            Segment::Lane,
            0xa77e ^ c as u64,
        );
    }
}

#[test]
fn the_attention_blocks_per_frame_run_rather_than_per_clip() {
    let Some(gpu) = Gpu::open() else { return };
    let boxes = [Box3::new(5, 2, 3), Box3::new(3, 2, 3)];
    let answers: Vec<Vec<f32>> = [Segment::Frames(1), Segment::Frames(2), Segment::Lane]
        .into_iter()
        .map(|segment| run(&gpu, &boxes, 1024, segment, 0x5e6).0)
        .collect();
    for (a, b) in [(0usize, 1usize), (1, 2), (0, 2)] {
        let moved = answers[a]
            .iter()
            .zip(&answers[b])
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            moved > 0.1,
            "answers {a} and {b} differ by only {moved}: the kernel is not reading the \
             segment it was handed"
        );
    }
}
