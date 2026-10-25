//! Voxel attention on the first wgpu adapter against an f64 host reference
//! (engine-wgpu had no kernel of its own): two clips, each row attending its
//! whole clip or its run of frames, heads 64 and 300 wide (two of the
//! instantiations), and rows past every clip landing zero.

use kernels_slang::Tensor;
use poem_ir::Dtype;

use crate::gpu::landed;
use crate::gpu::{Gpu, noise};

/// Two clips: (t, h, w).
const CLIPS: [[usize; 3]; 2] = [[3, 2, 3], [1, 2, 2]];
const EXTRA: usize = 1;

#[test]
fn the_voxel_attention_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |buf: u32, r: usize, w: usize, d: Dtype| Tensor::new(buf, r as u32, w as u32, d);
    let voxels: usize = CLIPS.iter().map(|c| c[0] * c[1] * c[2]).sum();
    let rows = voxels + EXTRA;
    let mut table = Vec::new();
    let mut off = 0;
    for c in CLIPS {
        table.extend([c[0] as i32, c[1] as i32, c[2] as i32, off as i32]);
        off += c[0] * c[1] * c[2];
    }
    let gb = gpu.up_i32(&table);
    let grid = t(gb, CLIPS.len(), 4, Dtype::I32);
    for (i, c) in [64usize, 300].into_iter().enumerate() {
        let seed = 10 * i as u64;
        let (qb, qr) = gpu.up(&noise(seed + 1, rows * c));
        let (kb, kr) = gpu.up(&noise(seed + 2, rows * c));
        let (vb, vr) = gpu.up(&noise(seed + 3, rows * c));
        let scale = 1.0 / (c as f32).sqrt();
        for frames in [0usize, 1] {
            let mut want = vec![0.0f64; rows * c];
            let mut off = 0;
            for clip in CLIPS {
                let plane = clip[1] * clip[2];
                let n = clip[0] * plane;
                for r in off..off + n {
                    let (lo, hi) = match ((r - off) / plane).checked_div(frames) {
                        None => (off, off + n),
                        Some(segment) => {
                            let f = segment * frames;
                            (off + f * plane, off + (f + frames).min(clip[0]) * plane)
                        }
                    };
                    let scores: Vec<f64> = (lo..hi)
                        .map(|j| {
                            (0..c)
                                .map(|d| f64::from(qr[r * c + d]) * f64::from(kr[j * c + d]))
                                .sum::<f64>()
                                * f64::from(scale)
                        })
                        .collect();
                    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let w: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                    let total: f64 = w.iter().sum();
                    for d in 0..c {
                        want[r * c + d] = (lo..hi)
                            .zip(&w)
                            .map(|(j, w)| w * f64::from(vr[j * c + d]))
                            .sum::<f64>()
                            / total;
                    }
                }
                off += n;
            }
            let (q, k, v) = (
                t(qb, rows, c, Dtype::Bf16),
                t(kb, rows, c, Dtype::Bf16),
                t(vb, rows, c, Dtype::Bf16),
            );
            let n = rows * c;
            landed(
                &mut gpu,
                &format!("voxel attention c {c}, frames {frames}"),
                &|g| vec![(g.up(&vec![7.0; n]).0, n, false)],
                &[want],
                &|g, o| {
                    use kernels_slang::attn::voxel;
                    let segment = if frames == 0 {
                        voxel::Segment::Clip
                    } else {
                        voxel::Segment::Frames(frames as u32)
                    };
                    g.run(|x| {
                        voxel::attention(
                            x,
                            q,
                            k,
                            v,
                            grid,
                            segment,
                            scale,
                            t(o[0].0, rows, c, Dtype::Bf16),
                            0,
                        )
                    })
                },
            );
        }
    }
}
