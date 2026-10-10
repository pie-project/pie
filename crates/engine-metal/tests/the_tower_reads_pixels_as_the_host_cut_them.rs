#![cfg(target_vendor = "apple")]

//! The host now sends a still as raw patch bytes in block order; the tower
//! normalizes, lays out and taps the position table on the device. Both
//! kernels must land what the host's own arithmetic once did: Qwen's
//! channel-major rows with the frame doubled and four bilinear taps, and
//! Gemma's pixel-major rows with two axis taps.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::{Tensor, layout};
use media::front::{Image, Rgb8};
use poem_ir::Dtype;

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn bound(device: &Context, handles: &Handles, bytes: &[u8], keep: &mut Vec<Buffer>) -> u32 {
    let mut buffer = Buffer::zeroed(device, (bytes.len() as u64).max(16)).expect("a buffer");
    buffer.write(0, bytes).expect("written");
    let handle = handles.bind(&buffer, 0, buffer.bytes()).expect("a handle");
    keep.push(buffer);
    handle
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|n| n.to_le_bytes()).collect()
}

/// The host's old patchify for Qwen: `(ch, t, r, c)` columns, the frame
/// repeated `temporal` times, `(v / 255 - mean) / std`.
fn qwen_reference(rgb: &[u8], h: usize, w: usize, p: usize, m: usize, temporal: usize) -> Vec<f32> {
    let (gh, gw) = (h / p, w / p);
    let pd = 3 * temporal * p * p;
    let mut out = Vec::new();
    for bh in 0..gh / m {
        for bw in 0..gw / m {
            for ih in 0..m {
                for iw in 0..m {
                    let (pr, pc) = (bh * m + ih, bw * m + iw);
                    let mut row = vec![0.0f32; pd];
                    for ch in 0..3 {
                        for t in 0..temporal {
                            for r in 0..p {
                                for col in 0..p {
                                    let off = ((ch * temporal + t) * p + r) * p + col;
                                    let src = ((pr * p + r) * w + (pc * p + col)) * 3 + ch;
                                    row[off] = (f32::from(rgb[src]) / 255.0 - 0.5) / 0.5;
                                }
                            }
                        }
                    }
                    out.extend(row);
                }
            }
        }
    }
    out
}

/// The host's old bilinear taps for Qwen over a `side`-wide table.
fn qwen_taps(gh: usize, gw: usize, m: usize, side: usize) -> (Vec<i32>, Vec<f32>) {
    let axis = |index: usize, size: usize| -> ([usize; 2], [f32; 2]) {
        let src = index as f32 * (side as f32 - 1.0) / (size.saturating_sub(1).max(1)) as f32;
        let floor = src.floor();
        let mut taps = [0usize; 2];
        let mut weights = [0f32; 2];
        for (t, offset) in [0f32, 1f32].into_iter().enumerate() {
            taps[t] = (floor as i64 + offset as i64).clamp(0, side as i64 - 1) as usize;
            weights[t] = (1.0 - (src - floor - offset).abs()).max(0.0);
        }
        (taps, weights)
    };
    let mut ids = Vec::new();
    let mut weights = Vec::new();
    for bh in 0..gh / m {
        for bw in 0..gw / m {
            for ih in 0..m {
                for iw in 0..m {
                    let (ht, hw) = axis(bh * m + ih, gh);
                    let (wt, ww) = axis(bw * m + iw, gw);
                    for a in 0..2 {
                        for b in 0..2 {
                            ids.push((ht[a] * side + wt[b]) as i32);
                            weights.push(hw[a] * ww[b]);
                        }
                    }
                }
            }
        }
    }
    (ids, weights)
}

struct Run {
    pixels: Vec<f32>,
    ids: Vec<i32>,
    weights: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    span: &media::front::EncodedSpan,
    patch: u32,
    channel_major: bool,
    temporal: u32,
    bilinear: bool,
    side: u32,
) -> Run {
    let mut keep = Vec::new();
    let rows = span.rows;
    let in_width = 3 * patch * patch;
    let out_width = temporal * in_width;
    let x = Tensor::new(
        bound(device, handles, &span.payload, &mut keep),
        rows,
        in_width,
        Dtype::U8,
    );
    let y_bytes = u64::from(rows) * u64::from(out_width) * 2;
    let y_buffer = Buffer::zeroed(device, y_bytes.next_multiple_of(16384)).expect("y");
    let y = Tensor::new(
        handles.bind(&y_buffer, 0, y_bytes).expect("y"),
        rows,
        out_width,
        Dtype::Bf16,
    );
    let positions: Vec<i32> = span
        .positions
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|rc| [0, rc[0] as i32, rc[1] as i32])
        .collect();
    let positions_t = Tensor::new(
        bound(device, handles, &i32_bytes(&positions), &mut keep),
        rows,
        3,
        Dtype::I32,
    );
    let g = span.patch_grid;
    let grids = Tensor::new(
        bound(
            device,
            handles,
            &i32_bytes(&[g.t as i32, g.h as i32, g.w as i32]),
            &mut keep,
        ),
        1,
        3,
        Dtype::I32,
    );
    let segments = Tensor::new(
        bound(device, handles, &i32_bytes(&[0, rows as i32]), &mut keep),
        2,
        1,
        Dtype::I32,
    );
    let taps = if bilinear { 4 } else { 2 };
    let ids_bytes = u64::from(rows) * u64::from(taps) * 4;
    let ids_buffer = Buffer::zeroed(device, ids_bytes.next_multiple_of(16384)).expect("ids");
    let ids = Tensor::new(
        handles.bind(&ids_buffer, 0, ids_bytes).expect("ids"),
        rows,
        taps,
        Dtype::I32,
    );
    let w_buffer = Buffer::zeroed(device, ids_bytes.next_multiple_of(16384)).expect("weights");
    let weights = Tensor::new(
        handles.bind(&w_buffer, 0, ids_bytes).expect("weights"),
        rows,
        taps,
        Dtype::F32,
    );
    let frame = device.frame().expect("a frame");
    let sink = Sink::new(device, &frame, pipelines, handles);
    layout::pixels(
        &sink,
        x,
        patch,
        [0.5; 3],
        [0.5; 3],
        channel_major,
        temporal,
        y,
    )
    .expect("pixels launches");
    layout::grid_taps(
        &sink,
        positions_t,
        grids,
        segments,
        bilinear,
        side,
        ids,
        weights,
    )
    .expect("grid_taps launches");
    frame.commit().expect("the frame lands");
    let pixels = handles
        .read(y.buf, y_bytes)
        .expect("y read back")
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16_to_f32(u16::from_le_bytes(*b)))
        .collect();
    let ids = handles
        .read(ids.buf, ids_bytes)
        .expect("ids read back")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    let weights = handles
        .read(weights.buf, ids_bytes)
        .expect("weights read back")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Run {
        pixels,
        ids,
        weights,
    }
}

fn still(h: u32, w: u32) -> Rgb8 {
    let data = (0..h * w)
        .flat_map(|i| {
            let (y, x) = (i / w, i % w);
            [
                (y * 7 % 256) as u8,
                (x * 13 % 256) as u8,
                ((x + y) * 3 % 256) as u8,
            ]
        })
        .collect();
    Rgb8::new(h, w, data).expect("a frame")
}

#[test]
fn the_tower_reads_pixels_as_the_host_cut_them() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();

    // Qwen: 16-pixel patches in 2 x 2 blocks, the frame doubled, four
    // bilinear taps into a 48 x 48 table.
    let (h, w) = (96u32, 160u32);
    let src = still(h, w);
    let span = Image {
        patch: 16,
        block: 2,
        mrope: true,
    }
    .encode(&src, (h, w), |s, _, _| s.clone())
    .expect("cut");
    let got = run(&device, &handles, &pipelines, &span, 16, true, 2, true, 48);
    let want = qwen_reference(&src.data, h as usize, w as usize, 16, 2, 2);
    assert_eq!(got.pixels.len(), want.len());
    let worst = got
        .pixels
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(worst < 1.0 / 64.0, "qwen pixels differ by {worst}");
    let (ids, weights) = qwen_taps((h / 16) as usize, (w / 16) as usize, 2, 48);
    assert_eq!(got.ids, ids, "qwen taps");
    let worst = got
        .weights
        .iter()
        .zip(&weights)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(worst < 1e-5, "qwen tap weights differ by {worst}");

    // Gemma: 16-pixel patches in 3 x 3 blocks, pixel-major, two axis taps
    // into a stacked table of 10240 + 10240 rows.
    let (h, w) = (96u32, 144u32);
    let src = still(h, w);
    let span = Image {
        patch: 16,
        block: 3,
        mrope: false,
    }
    .encode(&src, (h, w), |s, _, _| s.clone())
    .expect("cut");
    let got = run(
        &device, &handles, &pipelines, &span, 16, false, 1, false, 10240,
    );
    let want: Vec<f32> = span
        .payload
        .iter()
        .map(|&v| 2.0 * (f32::from(v) / 255.0 - 0.5))
        .collect();
    let worst = got
        .pixels
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(worst < 1.0 / 64.0, "gemma pixels differ by {worst}");
    let ids: Vec<i32> = span
        .positions
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|rc| [rc[1].min(10239) as i32, 10240 + rc[0].min(10239) as i32])
        .collect();
    assert_eq!(got.ids, ids, "gemma taps");
    assert!(got.weights.iter().all(|&w| w == 1.0), "gemma tap weights");
}
