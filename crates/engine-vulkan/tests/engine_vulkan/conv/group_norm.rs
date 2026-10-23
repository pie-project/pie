//! Group norm on the first Vulkan device, against an f64 host reference: two
//! clips normalized apart, channel groups of three (a lane's channel pair
//! straddles two groups), with and without the SiLU, and rows past every
//! clip landing zero.

use kernels_slang::Tensor;
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

const C: usize = 6;
const GROUPS: usize = 2;
const EPS: f32 = 1e-5;
/// Two clips of 40 and 9 voxels.
const CLIPS: [[usize; 3]; 2] = [[2, 4, 5], [1, 3, 3]];

#[test]
fn the_group_norm_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |buf: u32, r: usize, w: usize, d: Dtype| Tensor::new(buf, r as u32, w as u32, d);
    let extra = 2;
    let voxels: Vec<usize> = CLIPS.iter().map(|c| c[0] * c[1] * c[2]).collect();
    let rows = voxels.iter().sum::<usize>() + extra;
    let mut grid = Vec::new();
    let mut off = 0;
    for c in CLIPS {
        grid.extend([c[0] as i32, c[1] as i32, c[2] as i32, off as i32]);
        off += c[0] * c[1] * c[2];
    }
    let (xb, xr) = gpu.up(&noise(1, rows * C)
        .iter()
        .map(|v| v * 3.0 + 1.0)
        .collect::<Vec<_>>());
    let (wv, bv) = (noise(2, C), noise(3, C));
    let (wb, bb) = (gpu.up_f32(&wv), gpu.up_f32(&bv));
    let gb = gpu.up_i32(&grid);
    let scratch =
        kernels_slang::conv::group_norm::scratch_floats(CLIPS.len() as u32, GROUPS as u32) as usize;
    let parts = CLIPS.len() * kernels_slang::conv::group_norm::SPLITS as usize * GROUPS * 4;
    for silu in [false, true] {
        let mut want = vec![0.0f64; rows * C];
        let cg = C / GROUPS;
        let mut row0 = 0;
        for n in &voxels {
            for g in 0..GROUPS {
                let vals: Vec<f64> = (row0..row0 + n)
                    .flat_map(|r| (g * cg..(g + 1) * cg).map(move |c| (r, c)))
                    .map(|(r, c)| f64::from(xr[r * C + c]))
                    .collect();
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let var =
                    vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
                let inv = 1.0 / (var + f64::from(EPS)).sqrt();
                for r in row0..row0 + n {
                    for c in g * cg..(g + 1) * cg {
                        let mut v = (f64::from(xr[r * C + c]) - mean) * inv * f64::from(wv[c])
                            + f64::from(bv[c]);
                        if silu {
                            v /= 1.0 + (-v).exp();
                        }
                        want[r * C + c] = v;
                    }
                }
            }
            row0 += n;
        }
        let x = t(xb, rows, C, Dtype::Bf16);
        let grid = t(gb, CLIPS.len(), 4, Dtype::I32);
        let (w, b) = (t(wb, 1, C, Dtype::F32), t(bb, 1, C, Dtype::F32));
        let plane = |buf: u32, n: usize| t(buf, 1, n, Dtype::F32);
        both(
            &mut gpu,
            &format!("group norm, silu {silu}"),
            &|g| {
                vec![
                    (g.up(&vec![7.0; rows * C]).0, rows * C, false),
                    (g.up_f32(&vec![0.0; parts]), 0, true),
                    (g.up_f32(&vec![0.0; scratch - parts]), 0, true),
                ]
            },
            &[want],
            &|g, o| {
                let (p, s) = (plane(o[1].0, parts), plane(o[2].0, scratch - parts));
                g.run(|c| {
                    kernels_slang::conv::group_norm::group_norm(
                        c,
                        x,
                        grid,
                        GROUPS as u32,
                        w,
                        b,
                        EPS,
                        silu,
                        p,
                        s,
                        t(o[0].0, rows, C, Dtype::Bf16),
                    )
                })
            },
        );
    }
}
