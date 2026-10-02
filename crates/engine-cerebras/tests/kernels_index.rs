//! The sparse-attention indexer's host phases: block boundaries, the index
//! key cache, block means, the pooled entries and the bisected top-k.

#![allow(clippy::too_many_arguments)]

mod common;

use common::data;
use engine_cerebras::bench::{Bench, assert_close, round_bf16};
use kernels_cerebras::attn::index;
use kernels_cerebras::{KvPool, RaggedTensor};

/// A keys-only paged pool of `pages` pages of `ps` rows, `w` wide; lane
/// `l` holds the pages `lanes[l]`.
fn pool(b: &mut Bench, keys: kernels_cerebras::Tensor, ps: usize, lanes: &[Vec<i32>]) -> KvPool {
    let mut indptr = vec![0i32];
    let mut indices = Vec::new();
    for l in lanes {
        indices.extend_from_slice(l);
        indptr.push(indices.len() as i32);
    }
    let page_indices = b.i32(indices.len() as u32, 1, &indices);
    let page_indptr = b.i32(indptr.len() as u32, 1, &indptr);
    let max_pages = lanes.iter().map(Vec::len).max().unwrap_or(1) as u32;
    KvPool {
        keys,
        values: keys,
        page_indices,
        page_indptr,
        page_size: ps as i32,
        max_pages,
        seq_stride: u64::from(keys.width),
        head_stride: u64::from(keys.width),
    }
}

/// Rows closing a block of 4 are marked with their position and the
/// block's first position; invalid rows and open positions are not.
#[test]
fn boundaries_mark_the_rows_closing_a_block() {
    let positions = [3i32, 4, 7, 11, -1, 0];
    let req = [0i32, 0, 1, 1, 2, 2];
    let valid = [1i32, 1, 1, 0, 1, 1];
    let mut b = Bench::new();
    let p = b.i32(6, 1, &positions);
    let r = b.i32(6, 1, &req);
    let v = b.i32(6, 1, &valid);
    let (bp, br, bo) = (
        b.zeros(dtype::Dtype::I32, 6, 1),
        b.zeros(dtype::Dtype::I32, 6, 1),
        b.zeros(dtype::Dtype::I32, 6, 1),
    );
    let (pp, pr, po) = (
        b.zeros(dtype::Dtype::I32, 6, 1),
        b.zeros(dtype::Dtype::I32, 6, 1),
        b.zeros(dtype::Dtype::I32, 6, 1),
    );
    let ip = b.i32(2, 1, &[0, 6]);
    let ran = b
        .run(|ctx| {
            index::boundary_decode(ctx, p, r, v, 4, bp, br, bo)?;
            index::boundary_prefill(
                ctx,
                RaggedTensor {
                    data: p,
                    indptr: ip,
                },
                r,
                v,
                4,
                pp,
                pr,
                po,
            )
        })
        .unwrap();
    if !ran {
        return;
    }
    // Row 4 (position -1): -1 + 1 = 0 closes a block arithmetically, and it
    // is valid, so it marks position -1 as a boundary like the GPU kernels.
    let want_pos = vec![3, -1, 7, -1, -1, -1];
    let want_rope = vec![0, 0, 4, 0, 0, 0];
    for (pos, rq, rope) in [(bp, br, bo), (pp, pr, po)] {
        assert_eq!(b.read_i32(pos), want_pos);
        assert_eq!(b.read_i32(rq), req.to_vec());
        assert_eq!(b.read_i32(rope), want_rope);
    }
}

/// Index keys land at their write cells; the block mean averages the 4
/// keys ending at each boundary (missing positions add nothing); the pooled
/// entry lands at the boundary's cell.
#[test]
fn index_keys_append_then_average_into_blocks_that_file_at_their_boundary() {
    let (ps, hd) = (4usize, 6usize);
    let pages = 4usize;
    let table_data = data(pages * ps * hd, 151);
    let mut b = Bench::new();
    let keys = b.bf16((pages * ps) as u32, hd as u32, &table_data);
    let lanes = vec![vec![2i32, 0], vec![3]];
    let pool = pool(&mut b, keys, ps, &lanes);
    // Three new keys: lane 0 position 5 (page 0, offset 1), lane 1
    // position 3 (page 3, offset 3), and a dropped one (offset past the page).
    let new_keys = data(3 * hd, 152);
    let k = b.bf16(3, hd as u32, &new_keys);
    let wp = b.i32(3, 1, &[0, 3, 1]);
    let wo = b.i32(3, 1, &[1, 3, 4]);
    // Boundary rows: lane 0 at position 7 (block 4..7), lane 1 at position
    // 3 (block 0..3), lane 0 at position 1 (keys -2..1: two live), none.
    let bpos = b.i32(4, 1, &[7, 3, 1, -1]);
    let breq = b.i32(4, 1, &[0, 1, 0, 0]);
    let entries = b.zeros(dtype::Dtype::Bf16, 4, hd as u32);
    let ran = b
        .run(|ctx| {
            index::kv_append(ctx, k, &pool, wp, wo)?;
            index::block_mean(ctx, bpos, breq, &pool, hd as u32, 4, entries)?;
            index::pool_kv_append(ctx, entries, bpos, breq, &pool, wp, wo)
        })
        .unwrap();
    if !ran {
        return;
    }
    // The table after the appends.
    let mut table = table_data.clone();
    table[hd..2 * hd].copy_from_slice(&new_keys[..hd]);
    table[(3 * ps + 3) * hd..(3 * ps + 4) * hd].copy_from_slice(&new_keys[hd..2 * hd]);
    let cell = |lane: usize, pos: usize| lanes[lane][pos / ps] as usize * ps + pos % ps;
    let mean = |lane: usize, bpos: i64| -> Vec<f32> {
        let mut sum = vec![0.0f32; hd];
        for i in 0..4i64 {
            let pos = bpos + i - 3;
            if pos < 0 {
                continue;
            }
            let c = cell(lane, pos as usize);
            for (s, v) in sum.iter_mut().zip(&table[c * hd..(c + 1) * hd]) {
                *s += v;
            }
        }
        sum.iter().map(|s| round_bf16(s / 4.0)).collect()
    };
    let mut want_entries = Vec::new();
    want_entries.extend(mean(0, 7));
    want_entries.extend(mean(1, 3));
    want_entries.extend(mean(0, 1));
    want_entries.extend(std::iter::repeat_n(0.0, hd));
    assert_close(&b.read_f32(entries), &want_entries, 1e-2, 1e-2);
    // Then the entries filed at the boundaries' cells.
    let got = b.read_f32(keys);
    for (lane, pos, e) in [(0usize, 7usize, 0usize), (1, 3, 1), (0, 1, 2)] {
        let c = cell(lane, pos);
        assert_close(
            &got[c * hd..(c + 1) * hd],
            &want_entries[e * hd..(e + 1) * hd],
            1e-2,
            1e-2,
        );
    }
}

/// Each row ranks the closed blocks' keys by a ReLU-gated multi-head score:
/// a row with few blocks takes them all in order, a row with many keeps
/// the `top_k` at or above the bisected threshold, in key order.
#[test]
fn the_index_topk_picks_each_rows_best_blocks_in_key_order() {
    let (ps, hd, heads, ratio, top_k) = (4usize, 6usize, 2usize, 2usize, 3usize);
    let pages = 6usize;
    let table_data = data(pages * ps * hd, 153);
    let mut b = Bench::new();
    let keys = b.bf16((pages * ps) as u32, hd as u32, &table_data);
    let lanes = vec![vec![1i32, 4, 2, 5], vec![0, 3]];
    let pool = pool(&mut b, keys, ps, &lanes);
    // Rows: lane 0 at position 15 (8 closed blocks), lane 1 at position 5
    // (3 blocks: all taken), lane 0 at position 1 (one block), a padded row.
    let positions = [15i32, 5, 1, -1];
    let req = [0i32, 1, 0, 0];
    let qs = data(4 * heads * hd, 154);
    let ws: Vec<f32> = vec![1.0, 0.5, 1.0, 0.5, 2.0, 0.0, 1.0, 1.0];
    let q = b.bf16(4, (heads * hd) as u32, &qs);
    let w = b.f32(4, heads as u32, &ws);
    let p = b.i32(4, 1, &positions);
    let r = b.i32(4, 1, &req);
    let sel = b.zeros(dtype::Dtype::I32, 4, top_k as u32);
    let ran = b
        .run(|ctx| {
            index::topk(
                ctx,
                q,
                Some(w),
                &pool,
                p,
                r,
                heads as u32,
                hd as u32,
                top_k as u32,
                ratio as u32,
                sel,
            )
        })
        .unwrap();
    if !ran {
        return;
    }
    let cell = |lane: usize, pos: usize| lanes[lane][pos / ps] as usize * ps + pos % ps;
    let mut want = vec![-1i32; 4 * top_k];
    for row in 0..4 {
        let pos = positions[row];
        if pos < 0 {
            continue;
        }
        let lane = req[row] as usize;
        let nkeys = ((pos + 1) as usize / ratio).min(lanes[lane].len() * ps / ratio);
        let scores: Vec<f32> = (0..nkeys)
            .map(|j| {
                let c = cell(lane, (j + 1) * ratio - 1);
                let key = &table_data[c * hd..(c + 1) * hd];
                (0..heads)
                    .map(|h| {
                        let dot: f32 = qs[(row * heads + h) * hd..(row * heads + h + 1) * hd]
                            .iter()
                            .zip(key)
                            .map(|(a, b)| a * b)
                            .sum();
                        dot.max(0.0) * ws[row * heads + h]
                    })
                    .sum()
            })
            .collect();
        let taken: Vec<usize> = if nkeys <= top_k {
            (0..nkeys).collect()
        } else {
            let mut sorted = scores.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let thr = sorted[top_k - 1];
            (0..nkeys)
                .filter(|j| scores[*j] >= thr)
                .take(top_k)
                .collect()
        };
        for (slot, j) in taken.iter().enumerate() {
            want[row * top_k + slot] = *j as i32;
        }
    }
    assert_eq!(b.read_i32(sel), want);
}
