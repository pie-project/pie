//! The rest of the layout plane against host references: the fused-row
//! splits and slices, the embedding gathers (scaled, banded, hashed,
//! affine-quantised, interpolated), row permutations and folds, the vision
//! tower's pixel and position-tap preparation, the key envelopes, and the
//! per-row argmax / top-k the sampler reads every decode step. Each staged
//! kernel also runs under a live window: only the window's rows land.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, bf16, bf16s, close_bf16, e4m3_value, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::gather::embed_concat_mxfp8;
use kernels_rocm::gather::embed_weighted;
use kernels_rocm::gather::scatter_live_rows;
use kernels_rocm::gather::{embed_concat, embed_concat_mlxu4, embed_mlx_affine};
use kernels_rocm::gather::{grid_taps, pixels};
use kernels_rocm::gather::{merge_rows, pool_rows};
use kernels_rocm::linear::moe::GroupSeat;
use kernels_rocm::{ArgValue, Ctx, Fire, Launch, Tensor, gather, kernel, sampling};

/// bf16 values as the bytes of a u8 plane [rows, 2·width].
fn bf16_bytes(gpu: &Gpu, v: &[f32], rows: u32, width: u32) -> Tensor {
    let b: Vec<u8> = v.iter().flat_map(|&x| bf16(x).to_le_bytes()).collect();
    gpu.upload(&b, rows, 2 * width, Dtype::U8)
}

/// Runs `f` with the live window {rows, row_offset, lanes, 0} staged.
fn windowed(gpu: &Gpu, ctx: &Ctx, rows: u32, offset: u32, lanes: u32, f: impl FnOnce()) {
    let stage = gpu.ints(&[rows as i32, offset as i32, lanes as i32, 0]);
    ctx.arm_stage(stage.ptr);
    f();
    ctx.disarm_stage();
}

/// `want` with every row outside [offset, offset + live) zeroed: what a
/// windowed launch over a zeroed destination leaves.
fn only_rows(want: &[f32], width: usize, offset: usize, live: usize) -> Vec<f32> {
    want.iter()
        .enumerate()
        .map(|(i, &v)| {
            if (offset..offset + live).contains(&(i / width)) {
                v
            } else {
                0.0
            }
        })
        .collect()
}

#[test]
fn split_qkv_and_select() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();

    // A Gemma-4-like cut (16-byte widths) and a ragged one.
    for (qd, kvd) in [(8192u32, 1024u32), (1000, 300)] {
        let rows = 4u32;
        let stride = (qd + 2 * kvd) as usize;
        let xv = bf16s(rows as usize * stride, 1, 1.0);
        let x = gpu.tensor(&xv, rows, qd + 2 * kvd, Dtype::Bf16);
        let pick = |lo: usize, w: usize| -> Vec<f32> {
            (0..rows as usize)
                .flat_map(|r| xv[r * stride + lo..r * stride + lo + w].to_vec())
                .collect()
        };
        let (wq, wk, wv) = (
            pick(0, qd as usize),
            pick(qd as usize, kvd as usize),
            pick((qd + kvd) as usize, kvd as usize),
        );
        let mut q = gpu.zeros(rows, qd, Dtype::Bf16);
        let mut k = gpu.zeros(rows, kvd, Dtype::Bf16);
        let mut v = gpu.zeros(rows, kvd, Dtype::Bf16);
        gather::split_qkv(&ctx, x, qd, kvd, &mut q, &mut k, &mut v).unwrap();
        assert_eq!(gpu.read(&q), wq, "q {qd}/{kvd}");
        assert_eq!(gpu.read(&k), wk, "k {qd}/{kvd}");
        assert_eq!(gpu.read(&v), wv, "v {qd}/{kvd}");

        // Rows 1 and 2 live.
        let mut q = gpu.zeros(rows, qd, Dtype::Bf16);
        let mut k = gpu.zeros(rows, kvd, Dtype::Bf16);
        let mut v = gpu.zeros(rows, kvd, Dtype::Bf16);
        windowed(&gpu, &ctx, 2, 1, 0, || {
            gather::split_qkv(&ctx, x, qd, kvd, &mut q, &mut k, &mut v).unwrap();
        });
        assert_eq!(
            gpu.read(&q),
            only_rows(&wq, qd as usize, 1, 2),
            "windowed q"
        );
        assert_eq!(
            gpu.read(&k),
            only_rows(&wk, kvd as usize, 1, 2),
            "windowed k"
        );
        assert_eq!(
            gpu.read(&v),
            only_rows(&wv, kvd as usize, 1, 2),
            "windowed v"
        );
    }

    // select: layer 2's 77-wide slice of a 4-layer stacked row.
    let (rows, width, layers) = (5u32, 77u32, 4u32);
    let tv = bf16s((rows * width * layers) as usize, 2, 1.0);
    let table = gpu.tensor(&tv, rows, width * layers, Dtype::Bf16);
    let want: Vec<f32> = (0..rows as usize)
        .flat_map(|r| {
            let at = r * (width * layers) as usize + 2 * width as usize;
            tv[at..at + width as usize].to_vec()
        })
        .collect();
    let mut y = gpu.zeros(rows, width, Dtype::Bf16);
    gather::select(&ctx, table, 2, width, &mut y).unwrap();
    assert_eq!(gpu.read(&y), want, "select");
    let mut y = gpu.zeros(rows, width, Dtype::Bf16);
    windowed(&gpu, &ctx, 3, 2, 0, || {
        gather::select(&ctx, table, 2, width, &mut y).unwrap();
    });
    assert_eq!(
        gpu.read(&y),
        only_rows(&want, width as usize, 2, 3),
        "windowed select"
    );
}

#[test]
fn embed_scale_add_and_shards() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (vocab, hidden) = (40u32, 300u32);
    let h = hidden as usize;
    let tv = bf16s((vocab * hidden) as usize, 3, 1.0);
    let table = gpu.tensor(&tv, vocab, hidden, Dtype::Bf16);
    let ids = [5i32, 39, -1, 40, 0, 17];
    let rows = ids.len() as u32;
    let yv = bf16s(rows as usize * h, 4, 2.0);
    let stacked_v = bf16s(rows as usize * h * 3, 5, 2.0);
    let (a, b) = (1.7f32, 0.3f32);

    // e, ê, y, ŷ for each row: `y0` the residual each row starts from.
    let reference = |y0: &dyn Fn(usize, usize) -> f32| {
        let mut out = [(); 4].map(|()| vec![0f32; rows as usize * h]);
        for (n, &id) in ids.iter().enumerate() {
            let src = if (0..vocab as i32).contains(&id) {
                id as usize
            } else {
                0
            };
            for c in 0..h {
                let e = tv[src * h + c];
                let es = round_bf16(e * round_bf16(a));
                let yv = round_bf16(y0(n, c) + es);
                out[0][n * h + c] = e;
                out[1][n * h + c] = es;
                out[2][n * h + c] = yv;
                out[3][n * h + c] = round_bf16(yv * round_bf16(b));
            }
        }
        out
    };

    let want = reference(&|n, c| yv[n * h + c]);
    let id_t = gpu.column(&ids);
    let mut e = gpu.zeros(rows, hidden, Dtype::Bf16);
    let mut es = gpu.zeros(rows, hidden, Dtype::Bf16);
    let mut y = gpu.tensor(&yv, rows, hidden, Dtype::Bf16);
    let mut ys = gpu.zeros(rows, hidden, Dtype::Bf16);
    gather::embed_scale_add(
        &ctx, id_t, table, vocab, &mut e, a, &mut es, &mut y, b, &mut ys,
    )
    .unwrap();
    for (what, t, w) in [
        ("e", &e, &want[0]),
        ("ê", &es, &want[1]),
        ("y", &y, &want[2]),
    ] {
        assert_eq!(&gpu.read(t), w, "embed_scale_add {what}");
    }
    assert_eq!(gpu.read(&ys), want[3], "embed_scale_add ŷ");

    // Under a window: rows 2..5 only, the residual of the others untouched.
    let mut e = gpu.zeros(rows, hidden, Dtype::Bf16);
    let mut es = gpu.zeros(rows, hidden, Dtype::Bf16);
    let mut y = gpu.tensor(&yv, rows, hidden, Dtype::Bf16);
    let mut ys = gpu.zeros(rows, hidden, Dtype::Bf16);
    windowed(&gpu, &ctx, 3, 2, 0, || {
        gather::embed_scale_add(
            &ctx, id_t, table, vocab, &mut e, a, &mut es, &mut y, b, &mut ys,
        )
        .unwrap();
    });
    assert_eq!(gpu.read(&e), only_rows(&want[0], h, 2, 3), "windowed e");
    let y_want: Vec<f32> = (0..yv.len())
        .map(|i| {
            if (2..5).contains(&(i / h)) {
                want[2][i]
            } else {
                yv[i]
            }
        })
        .collect();
    assert_eq!(gpu.read(&y), y_want, "windowed y");
    assert_eq!(gpu.read(&ys), only_rows(&want[3], h, 2, 3), "windowed ŷ");

    // embed_scale_add_select: the residual is layer 1's slice of the stack.
    let want = reference(&|n, c| stacked_v[n * 3 * h + h + c]);
    let stacked = gpu.tensor(&stacked_v, rows, 3 * hidden, Dtype::Bf16);
    let mut e = gpu.zeros(rows, hidden, Dtype::Bf16);
    let mut es = gpu.zeros(rows, hidden, Dtype::Bf16);
    let mut y = gpu.zeros(rows, hidden, Dtype::Bf16);
    let mut ys = gpu.zeros(rows, hidden, Dtype::Bf16);
    gather::embed_scale_add_select(
        &ctx, id_t, table, vocab, &mut e, a, &mut es, stacked, 1, hidden, &mut y, b, &mut ys,
    )
    .unwrap();
    for (what, t, w) in [
        ("e", &e, &want[0]),
        ("ê", &es, &want[1]),
        ("y", &y, &want[2]),
        ("ŷ", &ys, &want[3]),
    ] {
        assert_eq!(&gpu.read(t), w, "embed_scale_add_select {what}");
    }

    // embed_vocab_shard (fired directly: its launcher asks a communicator
    // for the rank): rank 1's band of 20 rows, ids 20..40 land.
    let band = gpu.tensor(&tv[20 * h..], 20, hidden, Dtype::Bf16);
    let y = gpu.zeros(rows, hidden, Dtype::Bf16);
    ctx.fire(
        "test.embed_vocab_shard",
        Fire::at(kernel(
            "gather/rows.hip",
            "::pie::layout::embed_vocab_shard",
        ))
        .apply(Launch::grid([rows, 1, 1], [256, 1, 1])),
        &[
            id_t.arg(),
            band.arg(),
            y.arg(),
            ArgValue::I32(hidden as i32),
            ArgValue::I32(20),
            ArgValue::I32(20),
            ArgValue::Ptr(0),
        ],
    )
    .unwrap();
    let want: Vec<f32> = ids
        .iter()
        .flat_map(|&id| {
            if (20..40).contains(&id) {
                tv[id as usize * h..(id as usize + 1) * h].to_vec()
            } else {
                vec![0.0; h]
            }
        })
        .collect();
    assert_eq!(gpu.read(&y), want, "embed_vocab_shard");

    // gather_width_concat: three rank planes [rows, 50] side by side.
    let (world, shard) = (3usize, 50usize);
    let src_v = bf16s(world * rows as usize * shard, 6, 1.0);
    let src = gpu.tensor(&src_v, world as u32 * rows, shard as u32, Dtype::Bf16);
    let dst = gpu.zeros(rows, (world * shard) as u32, Dtype::Bf16);
    let total = rows * (world * shard) as u32;
    ctx.fire(
        "test.gather_width_concat",
        Fire::at(kernel(
            "gather/rows.hip",
            "::pie::layout::gather_width_concat",
        ))
        .apply(Launch::flat(total, 256)),
        &[
            src.arg(),
            dst.arg(),
            ArgValue::I32(rows as i32),
            ArgValue::I32(shard as i32),
            ArgValue::I32(world as i32),
        ],
    )
    .unwrap();
    let want: Vec<f32> = (0..rows as usize)
        .flat_map(|r| {
            (0..world).flat_map(move |k| {
                let at = (k * rows as usize + r) * shard;
                (at..at + shard).collect::<Vec<_>>()
            })
        })
        .map(|i| src_v[i])
        .collect();
    assert_eq!(gpu.read(&dst), want, "gather_width_concat");
}

#[test]
fn permutations_folds_and_scatters() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();

    // pack / unpack over the three copy units: 16-byte, 4-byte, 1-byte.
    for (width, dtype) in [(512u32, Dtype::Bf16), (13, Dtype::F32), (7, Dtype::Bf16)] {
        let rows = 6u32;
        let w = width as usize;
        let xv = bf16s(rows as usize * w, 7, 1.0);
        let x = gpu.tensor(&xv, rows, width, dtype);
        let perm = [3i32, 5, 0, 1, 4, 2];
        let perm_t = gpu.column(&perm);
        let mut packed = gpu.zeros(rows, width, dtype);
        gather::pack_rows(&ctx, x, perm_t, &mut packed).unwrap();
        let want: Vec<f32> = perm
            .iter()
            .flat_map(|&p| xv[p as usize * w..(p as usize + 1) * w].to_vec())
            .collect();
        assert_eq!(gpu.read(&packed), want, "pack_rows {width}");
        let mut back = gpu.zeros(rows, width, dtype);
        gather::unpack_rows(&ctx, packed, perm_t, &mut back).unwrap();
        assert_eq!(gpu.read(&back), xv, "unpack_rows {width}");

        // Rows 1..4 of the pack only.
        let mut packed = gpu.zeros(rows, width, dtype);
        windowed(&gpu, &ctx, 3, 1, 0, || {
            gather::pack_rows(&ctx, x, perm_t, &mut packed).unwrap();
        });
        assert_eq!(
            gpu.read(&packed),
            only_rows(&want, w, 1, 3),
            "windowed pack_rows {width}"
        );
    }

    // pool / merge over 2x2 squares, 9 source rows (one left over).
    let (side, width) = (2u32, 300u32);
    let w = width as usize;
    let xv = bf16s(9 * w, 8, 1.0);
    let x = gpu.tensor(&xv, 9, width, Dtype::Bf16);
    let mut pooled = gpu.zeros(2, width, Dtype::Bf16);
    pool_rows(&ctx, x, side, &mut pooled).unwrap();
    let want: Vec<f32> = (0..2)
        .flat_map(|o| {
            let xv = &xv;
            (0..w).map(move |c| {
                let mut acc = 0f32;
                for r in 0..4 {
                    acc += xv[(o * 4 + r) * w + c];
                }
                acc / 4.0
            })
        })
        .collect();
    close_bf16("pool_rows", &gpu.read(&pooled), &want, 0.5);
    let mut merged = gpu.zeros(2, 4 * width, Dtype::Bf16);
    merge_rows(&ctx, x, side, &mut merged).unwrap();
    assert_eq!(gpu.read(&merged), xv[..8 * w].to_vec(), "merge_rows");

    // scatter_live_rows: a negative destination is skipped.
    let width = 100u32;
    let w = width as usize;
    let tv = bf16s(4 * w, 9, 1.0);
    let tight = gpu.tensor(&tv, 4, width, Dtype::Bf16);
    let routes = gpu.column(&[6, -1, 0, 3]);
    let mut wide = gpu.zeros(8, width, Dtype::Bf16);
    scatter_live_rows(&ctx, tight, routes, &mut wide).unwrap();
    let mut want = vec![0f32; 8 * w];
    for (n, &at) in [6i32, -1, 0, 3].iter().enumerate() {
        if at >= 0 {
            want[at as usize * w..(at as usize + 1) * w].copy_from_slice(&tv[n * w..(n + 1) * w]);
        }
    }
    assert_eq!(gpu.read(&wide), want, "scatter_live_rows");
    let mut wide = gpu.zeros(8, width, Dtype::Bf16);
    windowed(&gpu, &ctx, 2, 2, 0, || {
        scatter_live_rows(&ctx, tight, routes, &mut wide).unwrap();
    });
    let mut want = vec![0f32; 8 * w];
    want[0..w].copy_from_slice(&tv[2 * w..3 * w]);
    want[3 * w..4 * w].copy_from_slice(&tv[3 * w..4 * w]);
    assert_eq!(gpu.read(&wide), want, "windowed scatter_live_rows");
}

#[test]
fn hashed_affine_and_weighted_embeds() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();

    // embed_concat: 3 ids a row, an id past the table lands zeros.
    let (vocab, width, heads) = (30u32, 64u32, 3u32);
    let w = width as usize;
    let tv = bf16s((vocab * width) as usize, 10, 1.0);
    let table = gpu.tensor(&tv, vocab, width, Dtype::Bf16);
    let idv = [1, 29, 30, -2, 0, 7, 12, 12, 3, 5, 6, 8];
    let rows = idv.len() as u32 / heads;
    let ids = gpu.ints_shaped(&idv, rows, heads);
    let mut y = gpu.zeros(rows, heads * width, Dtype::Bf16);
    embed_concat(&ctx, ids, table, vocab, &mut y).unwrap();
    let want: Vec<f32> = idv
        .iter()
        .flat_map(|&id| {
            if (0..vocab as i32).contains(&id) {
                tv[id as usize * w..(id as usize + 1) * w].to_vec()
            } else {
                vec![0.0; w]
            }
        })
        .collect();
    assert_eq!(gpu.read(&y), want, "embed_concat");
    let mut y = gpu.zeros(rows, heads * width, Dtype::Bf16);
    windowed(&gpu, &ctx, 2, 1, 0, || {
        embed_concat(&ctx, ids, table, vocab, &mut y).unwrap();
    });
    assert_eq!(
        gpu.read(&y),
        only_rows(&want, (heads * width) as usize, 1, 2),
        "windowed embed_concat"
    );

    // MLX affine tables, 4- and 8-bit codes, groups of 32.
    let group = 32usize;
    let per_row = w / group;
    let scales: Vec<f32> = bf16s(vocab as usize * per_row, 11, 0.1);
    let biases: Vec<f32> = bf16s(vocab as usize * per_row, 12, 0.5);
    let codes: Vec<u8> = values(vocab as usize * w, 13, 1.0)
        .iter()
        .map(|v| ((v + 1.0) * 127.9) as u8)
        .collect();
    let decode = |id: usize, c: usize, bits: u32| -> f32 {
        let code = codes[id * w + c] & if bits == 4 { 0xF } else { 0xFF };
        let f = id * per_row + c / group;
        round_bf16(f32::from(code) * scales[f] + biases[f])
    };
    let codes4: Vec<u8> = (0..vocab as usize * w / 2)
        .map(|i| (codes[2 * i] & 0xF) | ((codes[2 * i + 1] & 0xF) << 4))
        .collect();
    let s_t = bf16_bytes(&gpu, &scales, vocab, per_row as u32);
    let b_t = bf16_bytes(&gpu, &biases, vocab, per_row as u32);
    for bits in [4u32, 8] {
        let c_t = if bits == 4 {
            gpu.upload(&codes4, vocab, width / 2, Dtype::U8)
        } else {
            gpu.upload(&codes, vocab, width, Dtype::U8)
        };
        let tok = [4i32, 0, 29, 31, 17];
        let tok_t = gpu.column(&tok);
        let mut y = gpu.zeros(tok.len() as u32, width, Dtype::Bf16);
        embed_mlx_affine(
            &ctx,
            tok_t,
            c_t,
            s_t,
            Some(b_t),
            vocab,
            GroupSeat::RESIDENT,
            &mut y,
        )
        .unwrap();
        let want: Vec<f32> = tok
            .iter()
            .flat_map(|&id| {
                (0..w).map(move |c| {
                    if (0..vocab as i32).contains(&id) {
                        decode(id as usize, c, bits)
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        assert_eq!(gpu.read(&y), want, "embed_mlx_affine u{bits}");

        if bits == 4 {
            let mut y = gpu.zeros(rows, heads * width, Dtype::Bf16);
            embed_concat_mlxu4(
                &ctx,
                ids,
                c_t,
                s_t,
                Some(b_t),
                vocab,
                GroupSeat::RESIDENT,
                &mut y,
            )
            .unwrap();
            let want: Vec<f32> = idv
                .iter()
                .flat_map(|&id| {
                    (0..w).map(move |c| {
                        if (0..vocab as i32).contains(&id) {
                            decode(id as usize, c, 4)
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            assert_eq!(gpu.read(&y), want, "embed_concat_mlxu4");
        }
    }

    // embed_weighted: four bilinear taps a row over a 48-row table.
    let (vocab, hidden, taps) = (48u32, 200u32, 4u32);
    let h = hidden as usize;
    let tv = bf16s((vocab * hidden) as usize, 14, 1.0);
    let table = gpu.tensor(&tv, vocab, hidden, Dtype::Bf16);
    let rows = 5u32;
    let idv: Vec<i32> = (0..rows * taps).map(|i| (i as i32 * 7) % 50 - 1).collect();
    let wv: Vec<f32> = values((rows * taps) as usize, 15, 1.0);
    let ids = gpu.ints_shaped(&idv, rows, taps);
    let weights = gpu.tensor(&wv, rows, taps, Dtype::F32);
    let mut y = gpu.zeros(rows, hidden, Dtype::Bf16);
    embed_weighted(&ctx, ids, weights, table, vocab, &mut y).unwrap();
    let want: Vec<f32> = (0..rows as usize)
        .flat_map(|n| {
            let (idv, wv, tv) = (&idv, &wv, &tv);
            (0..h).map(move |c| {
                let mut acc = 0f32;
                for t in 0..taps as usize {
                    let id = idv[n * taps as usize + t];
                    let at = if (0..vocab as i32).contains(&id) {
                        id as usize
                    } else {
                        0
                    };
                    acc += wv[n * taps as usize + t] * tv[at * h + c];
                }
                acc
            })
        })
        .collect();
    close_bf16("embed_weighted", &gpu.read(&y), &want, 0.5);
    let mut y = gpu.zeros(rows, hidden, Dtype::Bf16);
    windowed(&gpu, &ctx, 2, 3, 0, || {
        embed_weighted(&ctx, ids, weights, table, vocab, &mut y).unwrap();
    });
    close_bf16(
        "windowed embed_weighted",
        &gpu.read(&y),
        &only_rows(&want, h, 3, 2),
        0.5,
    );
}

#[test]
fn pixels_and_grid_taps() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (patch, rows, temporal) = (4u32, 3u32, 2u32);
    let in_width = 3 * patch * patch;
    let px: Vec<u8> = (0..rows * in_width).map(|i| (i * 37 % 256) as u8).collect();
    let x = gpu.upload(&px, rows, in_width, Dtype::U8);
    let (mean, std) = ([0.48f32, 0.45, 0.40], [0.26f32, 0.26, 0.27]);
    for channel_major in [true, false] {
        let mut y = gpu.zeros(rows, temporal * in_width, Dtype::Bf16);
        pixels(&ctx, x, patch, mean, std, channel_major, temporal, &mut y).unwrap();
        let (area, w) = ((patch * patch) as usize, (temporal * in_width) as usize);
        let want: Vec<f32> = (0..rows as usize * w)
            .map(|i| {
                let (n, c) = (i / w, i % w);
                let (ch, at) = if channel_major {
                    let ch = c / (temporal as usize * area);
                    (ch, (c % area) * 3 + ch)
                } else {
                    let k = c % in_width as usize;
                    (k % 3, k)
                };
                let v = f32::from(px[n * in_width as usize + at]) / 255.0;
                round_bf16((v - mean[ch]) / std[ch])
            })
            .collect();
        close_bf16(
            &format!("pixels channel_major={channel_major}"),
            &gpu.read(&y),
            &want,
            1.0,
        );
    }

    // Two images: 2x3 and 3x2 patch grids onto a 5x5 table.
    let side = 5u32;
    let mut pos = Vec::new();
    for (gh, gw) in [(2, 3), (3, 2)] {
        for r in 0..gh {
            for c in 0..gw {
                pos.extend([0, r, c]);
            }
        }
    }
    let patches = pos.len() as u32 / 3;
    let positions = gpu.ints_shaped(&pos, patches, 3);
    let grids = gpu.ints_shaped(&[1, 2, 3, 1, 3, 2], 2, 3);
    let segments = gpu.column(&[0, 6, 12]);
    let axis = |index: i32, size: i32| -> ([i32; 2], [f32; 2]) {
        let src = index as f32 * (side - 1) as f32 / (size - 1).max(1) as f32;
        let fl = src.floor();
        let mut taps = [0; 2];
        let mut w = [0f32; 2];
        for t in 0..2 {
            taps[t] = (fl as i32 + t as i32).clamp(0, side as i32 - 1);
            w[t] = (1.0 - (src - fl - t as f32).abs()).max(0.0);
        }
        (taps, w)
    };
    for bilinear in [true, false] {
        let taps = if bilinear { 4 } else { 2 };
        let mut ids = gpu.zeros(patches, taps, Dtype::I32);
        let mut weights = gpu.zeros(patches, taps, Dtype::F32);
        grid_taps(
            &ctx,
            positions,
            grids,
            segments,
            bilinear,
            side,
            &mut ids,
            &mut weights,
        )
        .unwrap();
        let (mut want_i, mut want_w) = (Vec::new(), Vec::new());
        for n in 0..patches as usize {
            let (row, col) = (pos[n * 3 + 1], pos[n * 3 + 2]);
            if !bilinear {
                want_i.extend([
                    col.min(side as i32 - 1),
                    side as i32 + row.min(side as i32 - 1),
                ]);
                want_w.extend([1.0, 1.0]);
                continue;
            }
            let (gh, gw) = if n < 6 { (2, 3) } else { (3, 2) };
            let ((ht, hw), (wt, ww)) = (axis(row, gh), axis(col, gw));
            for a in 0..2 {
                for b in 0..2 {
                    want_i.push(ht[a] * side as i32 + wt[b]);
                    want_w.push(hw[a] * ww[b]);
                }
            }
        }
        assert_eq!(
            gpu.read_as::<i32>(&ids),
            want_i,
            "grid_taps ids bilinear={bilinear}"
        );
        common::close_abs(
            &format!("grid_taps weights bilinear={bilinear}"),
            &gpu.read(&weights),
            &want_w,
            1e-6,
        );
    }
}

/// The K best (value, column) of a row: larger first, a tie to the lower
/// column, NaNs never; unfilled places (0, 0).
fn host_topk(row: &[f32], k: usize) -> (Vec<f32>, Vec<i32>) {
    let mut order: Vec<usize> = (0..row.len()).filter(|&c| !row[c].is_nan()).collect();
    order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
    order.truncate(k);
    let mut v: Vec<f32> = order.iter().map(|&c| row[c]).collect();
    let mut i: Vec<i32> = order.iter().map(|&c| c as i32).collect();
    v.resize(k, 0.0);
    i.resize(k, 0);
    (v, i)
}

/// Logit rows that exercise the orders: random (bf16 rounding makes many
/// ties), a planted tie for the best, NaNs throughout, all -inf, all NaN,
/// and the best concentrated in one thread.
fn logit_rows(width: usize, seed: u64) -> Vec<f32> {
    let mut x = bf16s(6 * width, seed, 8.0);
    let row = |r: usize| r * width..(r + 1) * width;
    x[row(1)][width - 1] = 100.0;
    x[row(1)][width / 3] = 100.0;
    for c in row(2).step_by(3) {
        x[c] = f32::NAN;
    }
    x[row(3)].fill(f32::NEG_INFINITY);
    x[row(4)].fill(f32::NAN);
    // The best columns all in one thread's chunks (every 8192nd block of
    // eight), rising: the most columns the pool can be handed.
    for (i, c) in row(5).enumerate().filter(|(i, _)| i / 8 % 1024 == 0) {
        x[c] = round_bf16(50.0 + i as f32 / 64.0);
    }
    x
}

#[test]
fn argmax_and_topk() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();

    // A Gemma vocabulary (16-byte lanes), a Qwen one, and a ragged one
    // (single elements), in bf16 and f32.
    for (width, dtype) in [
        (262_144u32, Dtype::Bf16),
        (248_320, Dtype::F32),
        (1001, Dtype::Bf16),
        (5, Dtype::F32),
    ] {
        let w = width as usize;
        let xv = logit_rows(w, u64::from(width));
        let rows = 6u32;
        let x = gpu.tensor(&xv, rows, width, dtype);
        let want: Vec<(Vec<f32>, Vec<i32>)> = (0..rows as usize)
            .map(|r| host_topk(&xv[r * w..(r + 1) * w], 16))
            .collect();

        // argmax lands in column 1 of a 2-wide plane.
        let mut y = gpu.zeros(rows, 2, Dtype::I32);
        sampling::argmax(&ctx, x, 1, &mut y).unwrap();
        let got = gpu.read_as::<i32>(&y);
        for r in 0..rows as usize {
            assert_eq!(got[2 * r], 0, "argmax {width} row {r} column 0 untouched");
            assert_eq!(got[2 * r + 1], want[r].1[0], "argmax {width} row {r}");
        }

        for k in [8u32, 16] {
            let mut v = gpu.zeros(rows, k, Dtype::F32);
            let mut i = gpu.zeros(rows, k, Dtype::I32);
            sampling::top_k(&ctx, x, k, &mut v, &mut i).unwrap();
            let (gv, gi) = (gpu.read(&v), gpu.read_as::<i32>(&i));
            for r in 0..rows as usize {
                let k = k as usize;
                let (wv, wi) = (&want[r].0[..k], &want[r].1[..k]);
                assert_eq!(
                    &gi[r * k..(r + 1) * k],
                    wi,
                    "topk{k} {width} row {r} indices"
                );
                assert_eq!(
                    &gv[r * k..(r + 1) * k],
                    wv,
                    "topk{k} {width} row {r} values"
                );
            }
        }

        // Under a window of 2 rows, rows 2.. are not written.
        let mut y = gpu.zeros(rows, 1, Dtype::I32);
        windowed(&gpu, &ctx, 2, 0, 0, || {
            sampling::argmax(&ctx, x, 0, &mut y).unwrap()
        });
        let got = gpu.read_as::<i32>(&y);
        assert_eq!(
            &got[..2],
            &[want[0].1[0], want[1].1[0]],
            "windowed argmax {width}"
        );
        assert_eq!(
            &got[2..],
            &[0, 0, 0, 0],
            "windowed argmax {width} dead rows"
        );
    }
}

#[test]
fn envelope_update_appended() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (page_size, kv_heads, head_dim, pages) = (16usize, 2usize, 128usize, 5usize);
    let token = kv_heads * head_dim;
    let keys_v = bf16s(pages * page_size * token, 30, 2.0);
    let keys = gpu.tensor(
        &keys_v,
        (pages * page_size) as u32,
        token as u32,
        Dtype::Bf16,
    );

    // Three requests: r0 appends 9 tokens across its pages [3, 0] (5 live on
    // the last); r1 fills page [2]; r2 appends 1 token onto page 4 of [1, 4].
    let qo_indptr = gpu.ints(&[0, 9, 25, 26]);
    let page_indices = gpu.ints(&[3, 0, 2, 1, 4]);
    let page_indptr = gpu.ints(&[0, 2, 3, 5]);
    let last_lens = gpu.ints(&[5, 16, 1]);
    // (physical page, live tokens) of every touched page.
    let touched = [(3usize, 16usize), (0, 5), (2, 16), (4, 1)];
    let sentinel = 7.0f32;

    let envelope = |only: &[(usize, usize)]| -> (Vec<f32>, Vec<f32>) {
        let mut lo = vec![sentinel; pages * token];
        let mut hi = vec![sentinel; pages * token];
        for &(page, live) in only {
            for hd in 0..token {
                let col = (0..live).map(|t| keys_v[(page * page_size + t) * token + hd]);
                lo[page * token + hd] = col.clone().fold(f32::INFINITY, f32::min);
                hi[page * token + hd] = col.fold(f32::NEG_INFINITY, f32::max);
            }
        }
        (lo, hi)
    };

    for (lanes, live) in [(0u32, &touched[..]), (2, &touched[..3])] {
        let env_min = gpu.tensor(
            &vec![sentinel; pages * token],
            pages as u32,
            token as u32,
            Dtype::Bf16,
        );
        let env_max = gpu.tensor(
            &vec![sentinel; pages * token],
            pages as u32,
            token as u32,
            Dtype::Bf16,
        );
        let fire = || {
            ctx.fire(
                "test.envelope",
                Fire::at(kernel(
                    "attn/kv_envelope.hip",
                    "::pie::layout::update_appended",
                ))
                // Two slots past the four touched pages do nothing.
                .apply(Launch::grid(
                    [6, kv_heads as u32, 1],
                    [head_dim as u32, 1, 1],
                )),
                &[
                    keys.arg(),
                    qo_indptr.arg(),
                    page_indices.arg(),
                    page_indptr.arg(),
                    last_lens.arg(),
                    env_min.arg(),
                    env_max.arg(),
                    ArgValue::I32(3),
                    ArgValue::I32(page_size as i32),
                    ArgValue::I32(kv_heads as i32),
                    ArgValue::I32(head_dim as i32),
                    ctx.stage(),
                ],
            )
            .unwrap();
        };
        if lanes == 0 {
            fire();
        } else {
            windowed(&gpu, &ctx, 0, 0, lanes, fire);
        }
        let (lo, hi) = envelope(live);
        assert_eq!(gpu.read(&env_min), lo, "envelope min, {lanes} lanes");
        assert_eq!(gpu.read(&env_max), hi, "envelope max, {lanes} lanes");
    }
}

#[test]
fn an_mxfp8_table_gathers_its_rows_exactly() {
    // DeepSeek-V4.1's Engram table: e4m3 codes under one e8m0 exponent a
    // 1x32 block; an id past the table lands zeros.
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (vocab, width, heads) = (40u32, 64u32, 3u32);
    let w = width as usize;
    let codes: Vec<u8> = (0..vocab as usize * w)
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .map(|b| if b & 0x7F == 0x7F { 0 } else { b })
        .collect();
    let exps: Vec<u8> = (0..vocab as usize * w / 32)
        .map(|i| 118 + (i % 12) as u8)
        .collect();
    let table = gpu.upload(&codes, vocab, width, Dtype::E4m3);
    let scales = gpu.upload(&exps, vocab, width / 32, Dtype::E8m0);
    let idv = [1, 39, 40, -2, 0, 7, 12, 12, 3, 5, 6, 8];
    let rows = idv.len() as u32 / heads;
    let ids = gpu.ints_shaped(&idv, rows, heads);
    let mut y = gpu.zeros(rows, heads * width, Dtype::Bf16);
    embed_concat_mxfp8(&ctx, ids, table, scales, vocab, &mut y).unwrap();
    let mut want = Vec::new();
    for &id in &idv {
        for c in 0..w {
            want.push(if (0..vocab as i32).contains(&id) {
                let r = id as usize;
                round_bf16(
                    e4m3_value(codes[r * w + c])
                        * 2f32.powi(i32::from(exps[r * (w / 32) + c / 32]) - 127),
                )
            } else {
                0.0
            });
        }
    }
    assert_eq!(gpu.read(&y), want, "embed_concat_mxfp8");
}
