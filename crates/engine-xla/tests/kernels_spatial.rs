//! The spatial family against host references transcribed from the CUDA
//! kernels (`kernels-cuda/kernels/spatial/*.cuh`). Every fire holds two clips
//! of different boxes and trailing rows no lane covers.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::spatial::{self, Conv3d, GridRule, Segment, TimePad};

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed.wrapping_mul(40503));
            round_bf16(((h >> 8) % 2000) as f32 / 1000.0 - 1.0)
        })
        .collect()
}

type Clip = [i32; 4];

fn flat(clips: &[Clip]) -> Vec<i32> {
    clips.iter().flatten().copied().collect()
}

fn lane_of(clips: &[Clip], row: i32) -> Option<(usize, Clip)> {
    clips
        .iter()
        .enumerate()
        .find(|(_, c)| row >= c[3] && row < c[3] + c[0] * c[1] * c[2])
        .map(|(l, c)| (l, *c))
}

fn unravel(c: Clip, local: i32) -> (i32, i32, i32) {
    let w = local % c[2];
    let rest = local / c[2];
    (rest / c[1], rest % c[1], w)
}

fn ravel(c: Clip, t: i32, h: i32, w: i32) -> i32 {
    c[3] + (t * c[1] + h) * c[2] + w
}

/// The output table a conv rule gives (host `GridRule::apply`).
fn conv_boxes(conv: &Conv3d, clips: &[Clip]) -> Vec<Clip> {
    let mut off = 0;
    clips
        .iter()
        .map(|c| {
            let e = conv.out_extent([c[0] as u32, c[1] as u32, c[2] as u32]).unwrap_or([0; 3]);
            let b = [e[0] as i32, e[1] as i32, e[2] as i32, off];
            off += b[0] * b[1] * b[2];
            b
        })
        .collect()
}

fn conv_ref(
    x: &[f32],
    clips: &[Clip],
    w: &[f32],
    bias: &[f32],
    conv: &Conv3d,
    cache: Option<&[f32]>,
    c_in: usize,
    c_out: usize,
    y_clips: &[Clip],
    y_rows: usize,
) -> Vec<f32> {
    let [kt, kh, kw] = conv.k.map(|v| v as i32);
    let [st, sh, sw] = conv.stride.map(|v| v as i32);
    let [pt, ph, pw] = conv.pad.map(|v| v as i32);
    let replicate = conv.time_pad == TimePad::Replicate;
    let taps = (kt * kh * kw) as usize;
    let mut y = vec![0.0f32; y_rows * c_out];
    for m in 0..y_rows as i32 {
        let Some((l, og)) = lane_of(y_clips, m) else { continue };
        let ig = clips[l];
        let (ot, oh, ow) = unravel(og, m - og[3]);
        let base: i32 = clips[..l].iter().map(|c| c[1] * c[2]).sum::<i32>() * pt;
        for n in 0..c_out {
            let mut acc = 0.0f64;
            for tap in 0..taps as i32 {
                let (it, ih, iw) = (tap / (kh * kw), (tap / kw) % kh, tap % kw);
                let hi = oh * sh - ph + ih;
                let wi = ow * sw - pw + iw;
                if hi < 0 || hi >= ig[1] || wi < 0 || wi >= ig[2] {
                    continue;
                }
                let mut ti = ot * st - pt + it;
                let mut row: Option<&[f32]> = None;
                if ti < 0 {
                    if conv.causal_t && cache.is_some() {
                        let r = (base + ((ti + pt) * ig[1] + hi) * ig[2] + wi) as usize;
                        row = Some(&cache.unwrap()[r * c_in..(r + 1) * c_in]);
                    } else if !replicate {
                        continue;
                    } else {
                        ti = 0;
                    }
                }
                if row.is_none() && ti >= ig[0] {
                    if conv.causal_t || !replicate {
                        continue;
                    }
                    ti = ig[0] - 1;
                }
                let row = row.unwrap_or_else(|| {
                    let r = ravel(ig, ti, hi, wi) as usize;
                    &x[r * c_in..(r + 1) * c_in]
                });
                for c in 0..c_in {
                    acc += f64::from(row[c]) * f64::from(w[n * taps * c_in + tap as usize * c_in + c]);
                }
            }
            y[m as usize * c_out + n] = round_bf16(acc as f32 + bias[n]);
        }
    }
    y
}

#[test]
fn conv3d_answers_by_gather_and_by_convolution() {
    let clips: Vec<Clip> = vec![[3, 4, 5, 0], [2, 3, 2, 60]];
    let rows = 76u32;
    let (c_in, c_out) = (8usize, 12usize);
    let cases = [
        Conv3d {
            k: [3, 3, 3],
            stride: [1, 2, 2],
            pad: [1, 1, 1],
            pad_back: [1, 1, 1],
            causal_t: false,
            time_pad: TimePad::Zero,
        },
        Conv3d {
            k: [3, 3, 3],
            stride: [1, 1, 1],
            pad: [2, 1, 1],
            pad_back: [2, 1, 1],
            causal_t: true,
            time_pad: TimePad::Replicate,
        },
        Conv3d {
            k: [3, 1, 3],
            stride: [2, 1, 1],
            pad: [1, 0, 1],
            pad_back: [1, 0, 1],
            causal_t: false,
            time_pad: TimePad::Replicate,
        },
    ];
    let x = data(rows as usize * c_in, 1);
    let bias = data(c_out, 3);
    let cache_rows = 2 * (20 + 6);
    let cache_host = data(cache_rows * c_in, 4);
    let mut b = Bench::new();
    let xt = b.bf16(rows, c_in as u32, &x);
    let gt = b.i32(2, 4, &flat(&clips));
    let bt = b.f32(1, c_out as u32, &bias);
    let ct = b.bf16(cache_rows as u32, c_in as u32, &cache_host);
    let mut runs = Vec::new();
    for (i, conv) in cases.iter().enumerate() {
        let taps = conv.taps() as usize;
        let w = data(c_out * taps * c_in, 10 + i as u32)
            .iter()
            .map(|v| round_bf16(v * 0.2))
            .collect::<Vec<_>>();
        let y_clips = conv_boxes(conv, &clips);
        let last = y_clips.last().unwrap();
        let y_rows = (last[3] + last[0] * last[1] * last[2] + 3) as usize;
        let wt = b.bf16(c_out as u32, (taps * c_in) as u32, &w);
        let yg = b.i32(2, 4, &flat(&y_clips));
        let y = b.bf16(y_rows as u32, c_out as u32, &vec![7.0; y_rows * c_out]);
        let yb = b.bf16(y_rows as u32, c_out as u32, &vec![7.0; y_rows * c_out]);
        let cache = conv.causal_t.then_some(ct);
        let want = conv_ref(
            &x,
            &clips,
            &w,
            &bias,
            conv,
            conv.causal_t.then_some(&cache_host[..]),
            c_in,
            c_out,
            &y_clips,
            y_rows,
        );
        runs.push((*conv, wt, yg, y, yb, cache, y_clips, want));
    }
    let host = |v: &[Clip]| -> Vec<[u32; 4]> { v.iter().map(|c| c.map(|n| n as u32)).collect() };
    if !b
        .run(|ctx| {
            for (conv, wt, yg, y, yb, cache, y_clips, _) in &runs {
                spatial::conv3d(ctx, xt, gt, *wt, Some(bt), *conv, *cache, *y, *yg)?;
                spatial::conv3d_boxed(ctx, xt, &host(&clips), *wt, Some(bt), *conv, *cache, *yb, &host(y_clips))?;
            }
            Ok(())
        })
        .map_err(|e| format!("{e}"))
        .unwrap()
    {
        return;
    }
    for (i, (_, _, _, y, yb, _, _, want)) in runs.iter().enumerate() {
        eprintln!("case {i}");
        assert_close(&b.read_f32(*y), want, 2e-2, 1e-2);
        assert_close(&b.read_f32(*yb), want, 2e-2, 1e-2);
    }
}

#[test]
fn grid_rules_map_each_box() {
    let clips: Vec<Clip> = vec![[3, 7, 8, 0], [1, 2, 2, 168], [5, 8, 12, 172]];
    let rules = [
        GridRule::Conv {
            k: [3, 3, 3],
            stride: [1, 2, 2],
            pad: [1, 1, 1],
            pad_back: [1, 1, 1],
            causal_t: false,
        },
        GridRule::Upsample {
            factor: [2, 2, 2],
            keep_first_frame: true,
        },
        GridRule::Shuffle {
            r: [2, 2, 2],
            trim_t: 1,
        },
        GridRule::Unshuffle { r: [1, 2, 2] },
        GridRule::AvgDown { factor: [2, 2, 2] },
    ];
    let mut b = Bench::new();
    let g = b.i32(3, 4, &flat(&clips));
    let outs: Vec<_> = rules.iter().map(|_| b.zeros(Dtype::I32, 3, 4)).collect();
    if !b
        .run(|ctx| {
            for (rule, y) in rules.iter().zip(&outs) {
                spatial::derive_grid(ctx, g, *rule, *y)?;
            }
            Ok(())
        })
        .unwrap()
    {
        return;
    }
    for (rule, y) in rules.iter().zip(&outs) {
        let mut off = 0;
        let mut want = Vec::new();
        for c in &clips {
            let [t, h, w] = [c[0], c[1], c[2]];
            let e: Option<[i32; 3]> = match *rule {
                GridRule::Conv {
                    k,
                    stride,
                    pad,
                    pad_back,
                    causal_t,
                } => {
                    let back_t = if causal_t { 0 } else { pad_back[0] };
                    let axis = |n: i32, i: usize, back: u32| {
                        let span = n + pad[i] as i32 + back as i32 - k[i] as i32;
                        (span >= 0).then(|| span / stride[i] as i32 + 1)
                    };
                    (|| Some([axis(t, 0, back_t)?, axis(h, 1, pad_back[1])?, axis(w, 2, pad_back[2])?]))()
                }
                GridRule::Upsample { factor, keep_first_frame } => Some([
                    if keep_first_frame && t > 0 { 1 + (t - 1) * factor[0] as i32 } else { t * factor[0] as i32 },
                    h * factor[1] as i32,
                    w * factor[2] as i32,
                ]),
                GridRule::Shuffle { r, trim_t } => {
                    let ot = t * r[0] as i32 - trim_t as i32;
                    (ot > 0).then_some([ot, h * r[1] as i32, w * r[2] as i32])
                }
                GridRule::Unshuffle { r } => {
                    let r = r.map(|v| v as i32);
                    (t % r[0] == 0 && h % r[1] == 0 && w % r[2] == 0).then_some([t / r[0], h / r[1], w / r[2]])
                }
                GridRule::AvgDown { factor } => {
                    let r = factor.map(|v| v as i32);
                    (h % r[1] == 0 && w % r[2] == 0).then_some([(t + r[0] - 1) / r[0], h / r[1], w / r[2]])
                }
            };
            let e = e.unwrap_or([0; 3]);
            want.extend_from_slice(&[e[0], e[1], e[2], off]);
            off += e[0] * e[1] * e[2];
        }
        assert_eq!(b.read_i32(*y), want, "{rule:?}");
    }
}

#[test]
fn group_norm_and_attention_stay_inside_their_clip() {
    let clips: Vec<Clip> = vec![[2, 3, 4, 0], [1, 2, 5, 24]];
    let rows = 37usize;
    let c = 32usize;
    let groups = 4usize;
    let x = data(rows * c, 1);
    let wt = data(c, 2);
    let bs = data(c, 3);
    let q = data(rows * c, 4);
    let kk = data(rows * c, 5);
    let v = data(rows * c, 6);
    let mut b = Bench::new();
    let g = b.i32(2, 4, &flat(&clips));
    let xt = b.bf16(rows as u32, c as u32, &x);
    let w = b.f32(1, c as u32, &wt);
    let bias = b.f32(1, c as u32, &bs);
    let y = b.zeros(Dtype::Bf16, rows as u32, c as u32);
    let qt = b.bf16(rows as u32, c as u32, &q);
    let kt = b.bf16(rows as u32, c as u32, &kk);
    let vt = b.bf16(rows as u32, c as u32, &v);
    let a = b.zeros(Dtype::Bf16, rows as u32, c as u32);
    let af = b.zeros(Dtype::Bf16, rows as u32, c as u32);
    let scale = 0.3f32;
    if !b
        .run(|ctx| {
            spatial::group_norm(ctx, xt, g, groups as u32, w, bias, 1e-5, true, y)?;
            spatial::attention(ctx, qt, kt, vt, g, Segment::Lane, scale, a)?;
            spatial::attention(ctx, qt, kt, vt, g, Segment::Frames(1), scale, af)
        })
        .unwrap()
    {
        return;
    }
    let cg = c / groups;
    let mut want = vec![0.0f32; rows * c];
    for (l, cl) in clips.iter().enumerate() {
        let _ = l;
        let (off, n) = (cl[3] as usize, (cl[0] * cl[1] * cl[2]) as usize);
        for gi in 0..groups {
            let vals: Vec<f64> = (off..off + n)
                .flat_map(|r| (gi * cg..(gi + 1) * cg).map(move |ch| (r, ch)))
                .map(|(r, ch)| f64::from(x[r * c + ch]))
                .collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            let rstd = 1.0 / (var + 1e-5).sqrt();
            for r in off..off + n {
                for ch in gi * cg..(gi + 1) * cg {
                    let v = (f64::from(x[r * c + ch]) - mean) * rstd * f64::from(wt[ch]) + f64::from(bs[ch]);
                    let v = v / (1.0 + (-v).exp());
                    want[r * c + ch] = v as f32;
                }
            }
        }
    }
    assert_close(&b.read_f32(y), &want, 2e-2, 1e-2);

    let attn = |frames: Option<i32>| -> Vec<f32> {
        let mut out = vec![0.0f32; rows * c];
        for i in 0..rows as i32 {
            let Some((_, cl)) = lane_of(&clips, i) else { continue };
            let plane = cl[1] * cl[2];
            let (lo, hi) = match frames {
                None => (cl[3], cl[3] + cl[0] * plane),
                Some(n) => {
                    let f = (i - cl[3]) / plane / n * n;
                    (cl[3] + f * plane, cl[3] + (f + n).min(cl[0]) * plane)
                }
            };
            let i = i as usize;
            let s: Vec<f64> = (lo..hi)
                .map(|j| {
                    let j = j as usize;
                    (0..c).map(|e| f64::from(q[i * c + e]) * f64::from(kk[j * c + e])).sum::<f64>()
                        * f64::from(scale)
                })
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let p: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
            let l: f64 = p.iter().sum();
            for e in 0..c {
                let acc: f64 = (lo..hi).zip(&p).map(|(j, p)| p * f64::from(v[j as usize * c + e])).sum();
                out[i * c + e] = (acc / l) as f32;
            }
        }
        out
    };
    assert_close(&b.read_f32(a), &attn(None), 1e-2, 1e-2);
    assert_close(&b.read_f32(af), &attn(Some(1)), 1e-2, 1e-2);
}

#[test]
fn resamples_move_voxels_between_clip_layouts() {
    let clips: Vec<Clip> = vec![[3, 4, 2, 0], [2, 2, 4, 24]];
    let rows = 42usize;
    let c = 3usize;
    let x = data(rows * c, 1);
    let up_boxes = |f: [i32; 3], keep: bool| -> Vec<Clip> {
        let mut off = 0;
        clips
            .iter()
            .map(|cl| {
                let t = if keep && cl[0] > 0 { 1 + (cl[0] - 1) * f[0] } else { cl[0] * f[0] };
                let b = [t, cl[1] * f[1], cl[2] * f[2], off];
                off += b[0] * b[1] * b[2];
                b
            })
            .collect()
    };
    let span = |bx: &[Clip]| {
        let l = bx.last().unwrap();
        (l[3] + l[0] * l[1] * l[2]) as usize + 2
    };
    // Upsample x2 in time (keeping frame 0), x2 in space.
    let ub = up_boxes([2, 2, 2], true);
    // Unshuffle / patchify by [1, 2, 2]: every box divides.
    let r = [1u32, 2, 2];
    let dn_boxes: Vec<Clip> = {
        let mut off = 0;
        clips
            .iter()
            .map(|cl| {
                let b = [cl[0], cl[1] / 2, cl[2] / 2, off];
                off += b[0] * b[1] * b[2];
                b
            })
            .collect()
    };
    // Shuffle back by [2, 2, 2] with trim 1 from the unshuffled boxes.
    let rs = [2u32, 2, 2];
    let sh_boxes: Vec<Clip> = {
        let mut off = 0;
        dn_boxes
            .iter()
            .map(|cl| {
                let b = [cl[0] * 2 - 1, cl[1] * 2, cl[2] * 2, off];
                off += b[0] * b[1] * b[2];
                b
            })
            .collect()
    };
    // Avg-down by [2, 2, 2] in groups of 4.
    let ra = [2u32, 2, 2];
    let group = 4u32;
    let avg_boxes: Vec<Clip> = {
        let mut off = 0;
        clips
            .iter()
            .map(|cl| {
                let b = [(cl[0] + 1) / 2, cl[1] / 2, cl[2] / 2, off];
                off += b[0] * b[1] * b[2];
                b
            })
            .collect()
    };
    let unshuf_rows = span(&dn_boxes);
    let packed = data(unshuf_rows * c * 8, 2);

    let mut b = Bench::new();
    let g = b.i32(2, 4, &flat(&clips));
    let xt = b.bf16(rows as u32, c as u32, &x);
    let ug = b.i32(2, 4, &flat(&ub));
    let uy = b.bf16(span(&ub) as u32, c as u32, &vec![5.0; span(&ub) * c]);
    let dg = b.i32(2, 4, &flat(&dn_boxes));
    let dy = b.zeros(Dtype::Bf16, unshuf_rows as u32, (c * 4) as u32);
    let py = b.zeros(Dtype::Bf16, unshuf_rows as u32, (c * 4) as u32);
    let pk = b.bf16(unshuf_rows as u32, (c * 8) as u32, &packed);
    let sg = b.i32(2, 4, &flat(&sh_boxes));
    let sy = b.zeros(Dtype::Bf16, span(&sh_boxes) as u32, c as u32);
    let pk4 = b.bf16(unshuf_rows as u32, (c * 4) as u32, &packed[..unshuf_rows * c * 4]);
    let ey = b.zeros(Dtype::Bf16, rows as u32, c as u32);
    let ag = b.i32(2, 4, &flat(&avg_boxes));
    let ay = b.zeros(Dtype::Bf16, span(&avg_boxes) as u32, (c * 8 / group as usize) as u32);
    if !b
        .run(|ctx| {
            spatial::upsample_nearest(ctx, xt, g, [2, 2, 2], true, uy, ug)?;
            spatial::pixel_unshuffle(ctx, xt, g, r, dy, dg)?;
            spatial::patchify(ctx, xt, g, r, py, dg)?;
            spatial::pixel_shuffle(ctx, pk, dg, rs, 1, sy, sg)?;
            spatial::unpatchify(ctx, pk4, dg, r, ey, g)?;
            spatial::avg_down(ctx, xt, g, ra, group, ay, ag)
        })
        .unwrap()
    {
        return;
    }

    // Upsample.
    let mut want = vec![0.0f32; span(&ub) * c];
    for m in 0..span(&ub) as i32 {
        let Some((l, og)) = lane_of(&ub, m) else { continue };
        let (t, h, w) = unravel(og, m - og[3]);
        let ti = if t == 0 { 0 } else { (t - 1) / 2 + 1 };
        let src = ravel(clips[l], ti, h / 2, w / 2) as usize;
        want[m as usize * c..(m as usize + 1) * c].copy_from_slice(&x[src * c..(src + 1) * c]);
    }
    assert_close(&b.read_f32(uy), &want, 0.0, 0.0);

    // Unshuffle and patchify.
    let unshuffle = |x: &[f32], cin_w: usize, r: [i32; 3], out: &[Clip], inp: &[Clip], rows_out: usize| {
        let vol = (r[0] * r[1] * r[2]) as usize;
        let cw = cin_w * vol;
        let mut want = vec![0.0f32; rows_out * cw];
        for m in 0..rows_out as i32 {
            let Some((l, og)) = lane_of(out, m) else { continue };
            let (t, h, w) = unravel(og, m - og[3]);
            for col in 0..cw {
                let (cin, blk) = (col / vol, (col % vol) as i32);
                let (i1, i2, i3) = (blk / (r[1] * r[2]), (blk / r[2]) % r[1], blk % r[2]);
                let src = ravel(inp[l], t * r[0] + i1, h * r[1] + i2, w * r[2] + i3) as usize;
                want[m as usize * cw + col] = x[src * cin_w + cin];
            }
        }
        want
    };
    let want = unshuffle(&x, c, [1, 2, 2], &dn_boxes, &clips, unshuf_rows);
    assert_close(&b.read_f32(dy), &want, 0.0, 0.0);
    assert_close(&b.read_f32(py), &want, 0.0, 0.0);

    // Shuffle (trimmed) and unpatchify.
    let shuffle = |x: &[f32], c: usize, r: [i32; 3], trim: i32, out: &[Clip], inp: &[Clip], rows_out: usize| {
        let vol = (r[0] * r[1] * r[2]) as usize;
        let mut want = vec![0.0f32; rows_out * c];
        for m in 0..rows_out as i32 {
            let Some((l, og)) = lane_of(out, m) else { continue };
            let (t, h, w) = unravel(og, m - og[3]);
            let t = t + trim;
            let src = ravel(inp[l], t / r[0], h / r[1], w / r[2]) as usize;
            let blk = (((t % r[0]) * r[1] + h % r[1]) * r[2] + w % r[2]) as usize;
            for col in 0..c {
                want[m as usize * c + col] = x[src * c * vol + col * vol + blk];
            }
        }
        want
    };
    let want = shuffle(&packed, c, [2, 2, 2], 1, &sh_boxes, &dn_boxes, span(&sh_boxes));
    assert_close(&b.read_f32(sy), &want, 0.0, 0.0);
    let want = shuffle(&packed[..unshuf_rows * c * 4], c, [1, 2, 2], 0, &clips, &dn_boxes, rows);
    assert_close(&b.read_f32(ey), &want, 0.0, 0.0);

    // Avg-down.
    let (r1, r2, r3) = (2i32, 2i32, 2i32);
    let vol = (r1 * r2 * r3) as usize;
    let c_out = c * vol / group as usize;
    let mut want = vec![0.0f32; span(&avg_boxes) * c_out];
    for m in 0..span(&avg_boxes) as i32 {
        let Some((l, og)) = lane_of(&avg_boxes, m) else { continue };
        let ig = clips[l];
        let (t, h, w) = unravel(og, m - og[3]);
        let pad_t = (r1 - ig[0] % r1) % r1;
        for n in 0..c_out {
            let mut acc = 0.0f32;
            for j in 0..group as usize {
                let q = n * group as usize + j;
                let (cin, blk) = (q / vol, (q % vol) as i32);
                let (i1, i2, i3) = (blk / (r2 * r3), (blk / r3) % r2, blk % r3);
                let ti = t * r1 + i1 - pad_t;
                if ti < 0 {
                    continue;
                }
                acc += x[ravel(ig, ti, h * r2 + i2, w * r3 + i3) as usize * c + cin];
            }
            want[m as usize * c_out + n] = round_bf16(acc / group as f32);
        }
    }
    assert_close(&b.read_f32(ay), &want, 1e-6, 1e-2);
}

#[test]
fn a_frame_cache_gathers_and_stores_each_clip_slot() {
    let clips: Vec<Clip> = vec![[3, 2, 2, 0], [1, 1, 3, 12]];
    let frames = 2u32;
    let c = 4usize;
    let rows = 16usize;
    let x = data(rows * c, 1);
    let slots = 3usize;
    let stride = 2 * 4 * c; // the widest clip's block
    let slab = data(slots * stride, 2);
    let slot_ids = [2, 0];
    let cache_rows = 2 * 4 + 2 * 3 + 2;
    let mut b = Bench::new();
    let g = b.i32(2, 4, &flat(&clips));
    let xt = b.bf16(rows as u32, c as u32, &x);
    let st = b.bf16(slots as u32, stride as u32, &slab);
    let sl = b.i32(2, 1, &slot_ids);
    let cache = b.zeros(Dtype::Bf16, cache_rows as u32, c as u32);
    if !b
        .run(|ctx| {
            spatial::cache_gather(ctx, st, sl, g, frames, cache)?;
            spatial::cache_store(ctx, xt, cache, sl, g, frames, st)
        })
        .unwrap()
    {
        return;
    }
    // Gather: rows packed per lane, frames * plane each.
    let mut want_cache = vec![0.0f32; cache_rows * c];
    let mut base = 0usize;
    let mut bases = Vec::new();
    for (l, cl) in clips.iter().enumerate() {
        let n = frames as usize * (cl[1] * cl[2]) as usize;
        bases.push(base);
        for local in 0..n * c {
            if local < stride {
                want_cache[base * c + local] = slab[slot_ids[l] as usize * stride + local];
            }
        }
        base += n;
    }
    assert_close(&b.read_f32(cache), &want_cache, 0.0, 0.0);
    // Store: the last `frames` frames, from the gathered cache where the clip
    // is shorter.
    let mut want_slab = slab.clone();
    for (l, cl) in clips.iter().enumerate() {
        let plane = (cl[1] * cl[2]) as usize;
        for lr in 0..frames as usize * plane {
            let (f, hw) = (lr / plane, lr % plane);
            let src_t = cl[0] - frames as i32 + f as i32;
            for col in 0..c {
                let local = lr * c + col;
                if local >= stride {
                    continue;
                }
                let v = if src_t >= 0 {
                    x[(cl[3] as usize + src_t as usize * plane + hw) * c + col]
                } else {
                    want_cache[(bases[l] + (f + cl[0] as usize) * plane + hw) * c + col]
                };
                want_slab[slot_ids[l] as usize * stride + local] = v;
            }
        }
    }
    assert_close(&b.read_f32(st), &want_slab, 0.0, 0.0);
}
