//! Image input on the first Vulkan device, beside the hand-written kernels: patch bytes
//! to normalized bf16 rows (channel- and pixel-major, repeated in time), and
//! the position-table taps of each patch (bilinear and by axis).

use crate::gpu::{Gpu, both};
use kernels_vulkan::{Tensor, layout};
use poem_ir::Dtype;

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

#[test]
fn the_pixels_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let (rows, patch, temporal) = (3usize, 2usize, 2usize);
    let in_w = 3 * patch * patch;
    let width = temporal * in_w;
    let bytes: Vec<u8> = (0..rows * in_w).map(|i| (i * 37 % 256) as u8).collect();
    let words: Vec<u32> = bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let xb = gpu.up_u32(&words);
    let xt = Tensor::new(xb, rows as u32, in_w as u32, Dtype::U8);
    let (mean, sd) = ([0.48f32, 0.46, 0.41], [0.27f32, 0.26, 0.28]);
    for channel_major in [true, false] {
        let pixels = patch * patch;
        let want: Vec<f64> = (0..rows * width)
            .map(|i| {
                let (n, c) = (i / width, i % width);
                let (ch, at) = if channel_major {
                    let ch = c / (temporal * pixels);
                    (ch, (c % pixels) * 3 + ch)
                } else {
                    let at = c % in_w;
                    (at % 3, at)
                };
                let v = f64::from(bytes[n * in_w + at]) / 255.0;
                (v - f64::from(mean[ch])) / f64::from(sd[ch])
            })
            .collect();
        let yt = |buf: u32| Tensor::new(buf, rows as u32, width as u32, Dtype::Bf16);
        both(
            &mut gpu,
            &format!("pixels channel-major {channel_major}"),
            &|g| vec![(g.zeros(rows * width), rows * width, false)],
            &[want],
            &|g, o| {
                g.run(|c| {
                    layout::pixels(
                        c,
                        xt,
                        patch as u32,
                        mean,
                        sd,
                        channel_major,
                        temporal as u32,
                        yt(o[0].0),
                    )
                })
            },
            &|g, o| {
                g.run_slang(|c| {
                    kernels_slang::gather::pixels::pixels(
                        c,
                        st(xt),
                        patch as u32,
                        mean,
                        sd,
                        channel_major,
                        temporal as u32,
                        st(yt(o[0].0)),
                    )
                })
            },
        );
    }

    // Two images: a 3 x 4 grid (patches 0..12) and a 2 x 2 one (12..16).
    let side = 5i32;
    let mut pos = Vec::new();
    for (h, w) in [(3, 4), (2, 2)] {
        for r in 0..h {
            for c in 0..w {
                pos.extend([0, r, c]);
            }
        }
    }
    let n = pos.len() / 3;
    let grids = [0i32, 3, 4, 0, 2, 2];
    let segs = [0i32, 12, 16];
    let (pb, gb, sb) = (gpu.up_i32(&pos), gpu.up_i32(&grids), gpu.up_i32(&segs));
    let t = |buf: u32, r: usize, w: usize, d: Dtype| Tensor::new(buf, r as u32, w as u32, d);
    for bilinear in [true, false] {
        let taps = if bilinear { 4 } else { 2 };
        let mut want_ids = Vec::new();
        let mut want_w = Vec::new();
        for p in 0..n {
            let (row, col) = (pos[p * 3 + 1], pos[p * 3 + 2]);
            if !bilinear {
                want_ids.extend([col.min(side - 1), side + row.min(side - 1)]);
                want_w.extend([1.0, 1.0]);
                continue;
            }
            let img = if p < 12 { 0 } else { 1 };
            let axis = |i: i32, size: i32| {
                let src = f64::from(i) * f64::from(side - 1) / f64::from((size - 1).max(1));
                let fl = src.floor();
                [0, 1].map(|t| {
                    (
                        (fl as i32 + t).clamp(0, side - 1),
                        (1.0 - (src - fl - f64::from(t)).abs()).max(0.0),
                    )
                })
            };
            let hs = axis(row, grids[img * 3 + 1]);
            let ws = axis(col, grids[img * 3 + 2]);
            for (ha, hw) in hs {
                for (wb, ww) in ws {
                    want_ids.push(ha * side + wb);
                    want_w.push(hw * ww);
                }
            }
        }
        let fresh = |g: &mut Gpu| (g.up_i32(&vec![-7; n * taps]), g.zeros(2 * n * taps));
        let check = |g: &Gpu, who: &str, ids: u32, ws: u32| {
            let got: Vec<i32> = g
                .down_f32(ids, n * taps)
                .iter()
                .map(|v| v.to_bits() as i32)
                .collect();
            assert_eq!(got, want_ids, "{who}grid taps bilinear {bilinear}: ids");
            crate::gpu::close(
                &g.down_f32(ws, n * taps),
                &want_w,
                &format!("{who}grid taps bilinear {bilinear}"),
            );
        };
        let (ids, ws) = fresh(&mut gpu);
        gpu.run(|c| {
            layout::grid_taps(
                c,
                t(pb, n, 3, Dtype::I32),
                t(gb, 2, 3, Dtype::I32),
                t(sb, 3, 1, Dtype::I32),
                bilinear,
                side as u32,
                t(ids, n, taps, Dtype::I32),
                t(ws, n, taps, Dtype::F32),
            )
        });
        check(&gpu, "", ids, ws);
        let (ids, ws) = fresh(&mut gpu);
        gpu.run_slang(|c| {
            kernels_slang::gather::pixels::grid_taps(
                c,
                st(t(pb, n, 3, Dtype::I32)),
                st(t(gb, 2, 3, Dtype::I32)),
                st(t(sb, 3, 1, Dtype::I32)),
                bilinear,
                side as u32,
                st(t(ids, n, taps, Dtype::I32)),
                st(t(ws, n, taps, Dtype::F32)),
            )
        });
        check(&gpu, "slang ", ids, ws);
    }
}
