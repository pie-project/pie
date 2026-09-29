//! The layout family against host references.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::Bank;
use kernels_xla::layout;

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed.wrapping_mul(40503));
            round_bf16(((h >> 8) % 2000) as f32 / 1000.0 - 1.0)
        })
        .collect()
}

fn rows_of(x: &[f32], width: usize, ids: &[i32]) -> Vec<f32> {
    ids.iter()
        .flat_map(|&i| x[i as usize * width..(i as usize + 1) * width].to_vec())
        .collect()
}

#[test]
fn embeds_and_select_answer_the_host() {
    let (vocab, width) = (11u32, 24u32);
    let table = data((vocab * width) as usize, 1);
    let ids = [3, 0, 10, -1, 11, 7];
    let n = ids.len() as u32;
    // Concat: 3 rows x 2 ids.
    let cids = [1, 2, 99, 4, 5, -3];
    // Weighted: 3 rows x 2 taps.
    let wids = [1, 2, 20, 4, 5, 6];
    let ws = [0.25f32, 0.75, 0.5, 0.5, -1.0, 2.0];
    // Select: layer 1 of a 3 x 8 relay.
    let relay = data((4 * 24) as usize, 9);

    let mut b = Bench::new();
    let t = b.bf16(vocab, width, &table);
    let i = b.i32(n, 1, &ids);
    let y = b.zeros(Dtype::Bf16, n, width);
    let ci = b.i32(3, 2, &cids);
    let cy = b.zeros(Dtype::Bf16, 3, 2 * width);
    let wi = b.i32(3, 2, &wids);
    let ww = b.f32(3, 2, &ws);
    let wy = b.zeros(Dtype::Bf16, 3, width);
    let rt = b.bf16(4, 24, &relay);
    let sy = b.zeros(Dtype::Bf16, 3, 8);
    // Shard 1 of a two-rank split of the same table: rows 6..12 (5 real).
    let shard = b.bf16(6, width, &[&table[(6 * width) as usize..], &vec![0.0; width as usize][..]].concat());
    let vy = b.zeros(Dtype::Bf16, n, width);
    if !b
        .run(|ctx| {
            layout::embed(ctx, i, t, vocab, y)?;
            layout::embed_concat(ctx, ci, t, vocab, cy)?;
            layout::embed_weighted(ctx, wi, ww, t, vocab, wy)?;
            layout::select(ctx, rt, 1, 8, sy)?;
            layout::embed_vocab_shard(ctx, i, shard, 1, vy)
        })
        .unwrap()
    {
        return;
    }
    let w = width as usize;
    let guard = |i: i32| if (0..vocab as i32).contains(&i) { i } else { 0 };
    let want: Vec<i32> = ids.iter().map(|&i| guard(i)).collect();
    assert_close(&b.read_f32(y), &rows_of(&table, w, &want), 0.0, 0.0);

    let mut want = Vec::new();
    for &id in &cids {
        if (0..vocab as i32).contains(&id) {
            want.extend_from_slice(&table[id as usize * w..(id as usize + 1) * w]);
        } else {
            want.extend(std::iter::repeat_n(0.0, w));
        }
    }
    assert_close(&b.read_f32(cy), &want, 0.0, 0.0);

    let mut want = vec![0.0f32; 3 * w];
    for r in 0..3 {
        for c in 0..w {
            let mut acc = 0.0f32;
            for tap in 0..2 {
                let id = guard(wids[r * 2 + tap]) as usize;
                acc += ws[r * 2 + tap] * table[id * w + c];
            }
            want[r * w + c] = round_bf16(acc);
        }
    }
    assert_close(&b.read_f32(wy), &want, 1e-6, 1e-2);

    let want: Vec<f32> = (0..3).flat_map(|r| relay[r * 24 + 8..r * 24 + 16].to_vec()).collect();
    assert_close(&b.read_f32(sy), &want, 0.0, 0.0);

    let mut want = Vec::new();
    for &id in &ids {
        // Id 11 lands on the band's zero pad row.
        if (6..11).contains(&id) {
            want.extend_from_slice(&table[id as usize * w..(id as usize + 1) * w]);
        } else {
            want.extend(std::iter::repeat_n(0.0, w));
        }
    }
    assert_close(&b.read_f32(vy), &want, 0.0, 0.0);
}

/// Packs `codes` (each < 2^bits) LSB-first into little-endian bytes.
fn pack(codes: &[u32], bits: u32) -> Vec<u8> {
    let per = 8 / bits;
    codes
        .chunks(per as usize)
        .map(|c| c.iter().enumerate().fold(0u8, |acc, (i, &v)| acc | ((v as u8) << (i as u32 * bits))))
        .collect()
}

#[test]
fn affine_embeds_decode_their_rows() {
    let (vocab, width, group) = (7u32, 128u32, 32u32);
    let groups = width / group;
    let mut b = Bench::new();
    let make = |b: &mut Bench, bits: u32, words: bool, seed: u32| -> (Bank, Vec<f32>) {
        let codes: Vec<u32> = (0..vocab * width)
            .map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 7) % (1 << bits))
            .collect();
        let scales = data((vocab * groups) as usize, seed + 1)
            .iter()
            .map(|s| round_bf16(s * 0.1))
            .collect::<Vec<_>>();
        let biases = data((vocab * groups) as usize, seed + 2);
        let bytes = pack(&codes, bits);
        let plane = if words {
            let w: Vec<u32> = bytes
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            b.u32(vocab, width * bits / 32, &w)
        } else {
            b.u8(vocab, width * bits / 8, &bytes)
        };
        let bank = Bank {
            codes: plane,
            scales: b.bf16(vocab, groups, &scales),
            biases: Some(b.bf16(vocab, groups, &biases)),
            group,
            bits,
        };
        let full: Vec<f32> = (0..(vocab * width) as usize)
            .map(|i| {
                let g = i / group as usize;
                round_bf16(scales[g] * codes[i] as f32 + biases[g])
            })
            .collect();
        (bank, full)
    };
    let (b4, full4) = make(&mut b, 4, true, 3);
    let (b2, full2) = make(&mut b, 2, false, 5);
    let ids = [6, 0, 9, 2, -1];
    let i = b.i32(5, 1, &ids);
    let y = b.zeros(Dtype::Bf16, 5, width);
    let cids = [1, 3, 5, 7];
    let ci = b.i32(2, 2, &cids);
    let cy = b.zeros(Dtype::Bf16, 2, 2 * width);
    if !b
        .run(|ctx| {
            layout::embed_gather_mb_4bit(ctx, i, b4, vocab, y)?;
            layout::embed_concat_mb_4bit(ctx, ci, b2, vocab, cy)
        })
        .unwrap()
    {
        return;
    }
    let w = width as usize;
    let rows = |full: &[f32], ids: &[i32]| -> Vec<f32> {
        ids.iter()
            .flat_map(|&id| {
                if (0..vocab as i32).contains(&id) {
                    full[id as usize * w..(id as usize + 1) * w].to_vec()
                } else {
                    vec![0.0; w]
                }
            })
            .collect()
    };
    assert_close(&b.read_f32(y), &rows(&full4, &ids), 1e-6, 1e-2);
    assert_close(&b.read_f32(cy), &rows(&full2, &cids), 1e-6, 1e-2);
}

/// A table landed folded (`[V/f, f·row]` bytes, scales and biases each by
/// their own `f`, as engine-xla lands a gather-only table) reads the rows
/// it unfolds to.
#[test]
fn a_folded_affine_table_reads_its_rows() {
    let (rows, fold, width, group, bits) = (12u32, 4u32, 128u32, 64u32, 4u32);
    let groups = width / group;
    let vocab = 11; // the last stored row is padding past the vocabulary
    let codes: Vec<u32> = (0..rows * width)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 9) % (1 << bits))
        .collect();
    let scales: Vec<f32> = data((rows * groups) as usize, 7)
        .iter()
        .map(|s| round_bf16(s * 0.1))
        .collect();
    let biases = data((rows * groups) as usize, 8);
    let mut b = Bench::new();
    let bank = Bank {
        codes: b.u8(rows / fold, fold * width * bits / 8, &pack(&codes, bits)),
        // Each plane folds by its own factor.
        scales: b.bf16(rows / 2, 2 * groups, &scales),
        biases: Some(b.bf16(rows / fold, fold * groups, &biases)),
        group,
        bits,
    };
    let ids = [10, 0, 5, 11, 3, -2, 7];
    let i = b.i32(ids.len() as u32, 1, &ids);
    let y = b.zeros(Dtype::Bf16, ids.len() as u32, width);
    if !b
        .run(|ctx| layout::embed_gather_mb_4bit(ctx, i, bank, vocab, y))
        .unwrap()
    {
        return;
    }
    let w = width as usize;
    let want: Vec<f32> = ids
        .iter()
        .flat_map(|&id| {
            if (0..vocab as i32).contains(&id) {
                (0..w)
                    .map(|c| {
                        let at = id as usize * w + c;
                        let g = at / group as usize;
                        round_bf16(scales[g] * codes[at] as f32 + biases[g])
                    })
                    .collect()
            } else {
                vec![0.0; w]
            }
        })
        .collect();
    assert_close(&b.read_f32(y), &want, 1e-6, 1e-2);
}

#[test]
fn splits_cut_where_they_state() {
    let rows = 3u32;
    let (qw, kw) = (12u32, 6u32);
    let packed = data((rows * (qw + 2 * kw)) as usize, 1);
    let (hd, heads) = (6u32, 3u32);
    let qg = data((rows * 2 * hd * heads) as usize, 2);
    let lr = data((rows * 10) as usize, 3);
    let mut b = Bench::new();
    let p = b.bf16(rows, qw + 2 * kw, &packed);
    let q = b.zeros(Dtype::Bf16, rows, qw);
    let k = b.zeros(Dtype::Bf16, rows, kw);
    let v = b.zeros(Dtype::Bf16, rows, kw);
    let g_in = b.bf16(rows, 2 * hd * heads, &qg);
    let gq = b.zeros(Dtype::Bf16, rows, hd * heads);
    let gg = b.zeros(Dtype::Bf16, rows, hd * heads);
    let x = b.f32(rows, 10, &lr);
    let l = b.zeros(Dtype::F32, rows, 4);
    let r = b.zeros(Dtype::F32, rows, 6);
    if !b
        .run(|ctx| {
            layout::split_qkv(ctx, p, qw, kw, q, k, v)?;
            layout::split_q_gate(ctx, g_in, hd, gq, gg)?;
            layout::split_rows(ctx, x, 4, l, r)
        })
        .unwrap()
    {
        return;
    }
    let cut = |x: &[f32], width: usize, lo: usize, hi: usize| -> Vec<f32> {
        x.chunks(width).flat_map(|row| row[lo..hi].to_vec()).collect()
    };
    let pw = (qw + 2 * kw) as usize;
    let (qw, kw) = (qw as usize, kw as usize);
    assert_close(&b.read_f32(q), &cut(&packed, pw, 0, qw), 0.0, 0.0);
    assert_close(&b.read_f32(k), &cut(&packed, pw, qw, qw + kw), 0.0, 0.0);
    assert_close(&b.read_f32(v), &cut(&packed, pw, qw + kw, pw), 0.0, 0.0);
    let hd = hd as usize;
    assert_close(&b.read_f32(gq), &cut(&qg, 2 * hd, 0, hd), 0.0, 0.0);
    assert_close(&b.read_f32(gg), &cut(&qg, 2 * hd, hd, 2 * hd), 0.0, 0.0);
    assert_close(&b.read_f32(l), &cut(&lr, 10, 0, 4), 0.0, 0.0);
    assert_close(&b.read_f32(r), &cut(&lr, 10, 4, 10), 0.0, 0.0);
}

#[test]
fn row_moves_gather_scatter_and_permute() {
    let width = 5usize;
    let wide = data(7 * width, 1);
    let tight = data(3 * width, 2);
    let mut b = Bench::new();
    let w = b.bf16(7, width as u32, &wide);
    let gi = b.i32(3, 1, &[6, 0, 3]);
    let g = b.zeros(Dtype::Bf16, 3, width as u32);
    let t = b.bf16(3, width as u32, &tight);
    let si = b.i32(3, 1, &[4, 99, 1]);
    let s = b.bf16(7, width as u32, &wide);
    let li = b.i32(3, 1, &[-1, 2, 5]);
    let live = b.bf16(7, width as u32, &wide);
    let pi = b.i32(4, 1, &[2, 5, 0, 1]);
    let pk = b.zeros(Dtype::Bf16, 3, width as u32);
    let un = b.bf16(7, width as u32, &wide);
    if !b
        .run(|ctx| {
            layout::gather_rows(ctx, w, gi, g)?;
            layout::scatter_rows(ctx, t, si, s)?;
            layout::scatter_live_rows(ctx, t, li, live)?;
            layout::pack_rows(ctx, w, pi, pk)?;
            layout::unpack_rows(ctx, t, pi, un)
        })
        .unwrap()
    {
        return;
    }
    assert_close(&b.read_f32(g), &rows_of(&wide, width, &[6, 0, 3]), 0.0, 0.0);
    let scattered = |routes: &[i32]| {
        let mut out = wide.clone();
        for (i, &r) in routes.iter().enumerate() {
            if (0..7).contains(&r) {
                out[r as usize * width..(r as usize + 1) * width]
                    .copy_from_slice(&tight[i * width..(i + 1) * width]);
            }
        }
        out
    };
    assert_close(&b.read_f32(s), &scattered(&[4, 99, 1]), 0.0, 0.0);
    assert_close(&b.read_f32(live), &scattered(&[-1, 2, 5]), 0.0, 0.0);
    assert_close(&b.read_f32(pk), &rows_of(&wide, width, &[2, 5, 0]), 0.0, 0.0);
    assert_close(&b.read_f32(un), &scattered(&[2, 5, 0]), 0.0, 0.0);
}

#[test]
fn folds_pool_and_merge_blocks_of_rows() {
    let (side, width) = (2u32, 6usize);
    // 9 rows: two whole 4-row blocks and one row left over.
    let x = data(9 * width, 4);
    let mut b = Bench::new();
    let xs = b.bf16(9, width as u32, &x);
    let tail = data(3 * width, 5);
    let pool = b.bf16(3, width as u32, &tail);
    let merge = b.zeros(Dtype::Bf16, 2, 4 * width as u32);
    if !b
        .run(|ctx| {
            layout::pool_rows(ctx, xs, side, pool)?;
            layout::merge_rows(ctx, xs, side, merge)
        })
        .unwrap()
    {
        return;
    }
    let mut want = tail.clone();
    for o in 0..2 {
        for c in 0..width {
            let s: f32 = (0..4).map(|r| x[(o * 4 + r) * width + c]).sum();
            want[o * width + c] = round_bf16(s * 0.25);
        }
    }
    assert_close(&b.read_f32(pool), &want, 0.0, 0.0);
    assert_close(&b.read_f32(merge), &x[..8 * width], 0.0, 0.0);
}

fn top(row: &[f32], k: usize) -> Vec<(f32, i32)> {
    let mut idx: Vec<usize> = (0..row.len()).filter(|&i| !row[i].is_nan()).collect();
    idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
    let mut out: Vec<(f32, i32)> = idx.iter().take(k).map(|&i| (row[i], i as i32)).collect();
    out.resize(k, (0.0, 0));
    out
}

#[test]
fn argmax_and_topk_rank_rows_lowest_index_first() {
    let width = 37usize;
    let mut x = data(4 * width, 6);
    // Row 0: a tie at the max; row 1: NaN everywhere but two; row 2: all NaN;
    // row 3: -inf alone after NaN.
    x[5] = 2.0;
    x[30] = 2.0;
    for c in 0..width {
        x[width + c] = f32::NAN;
        x[2 * width + c] = f32::NAN;
        x[3 * width + c] = if c == 0 { f32::NAN } else { f32::NEG_INFINITY };
    }
    x[width + 9] = -0.5;
    x[width + 20] = 0.5;
    let x2 = data(4 * width, 7);
    let mut b = Bench::new();
    let xs = b.f32(4, width as u32, &x);
    let xb = b.bf16(4, width as u32, &x2);
    let y = b.i32(4, 2, &[-7; 8]);
    let k = 3u32;
    let tv = b.zeros(Dtype::F32, 4, k);
    let ti = b.zeros(Dtype::I32, 4, k);
    let bv = b.zeros(Dtype::F32, 4, k);
    let bi = b.zeros(Dtype::I32, 4, k);
    if !b
        .run(|ctx| {
            layout::argmax(ctx, xs, 0, y)?;
            layout::argmax(ctx, xb, 1, y)?;
            layout::topk(ctx, xs, k, tv, ti)?;
            layout::topk(ctx, xb, k, bv, bi)
        })
        .unwrap()
    {
        return;
    }
    let mut want_y = Vec::new();
    let (mut wv, mut wi, mut wbv, mut wbi) = (vec![], vec![], vec![], vec![]);
    for r in 0..4 {
        let row = &x[r * width..(r + 1) * width];
        let rb = &x2[r * width..(r + 1) * width];
        want_y.push(top(row, 1)[0].1);
        want_y.push(top(rb, 1)[0].1);
        for (v, i) in top(row, k as usize) {
            wv.push(v);
            wi.push(i);
        }
        for (v, i) in top(rb, k as usize) {
            wbv.push(v);
            wbi.push(i);
        }
    }
    assert_eq!(b.read_i32(y), want_y);
    assert_eq!(b.read_i32(ti), wi);
    assert_eq!(b.read_i32(bi), wbi);
    assert_eq!(b.read_f32(tv), wv);
    assert_eq!(b.read_f32(bv), wbv);
}
