mod common;

use common::data;
use engine_cerebras::bench::{Bench, assert_close};
use kernels_cerebras::layout;

#[test]
fn an_embed_gathers_rows_and_guards_the_vocabulary() {
    let (vocab_rows, width) = (9u32, 12u32);
    let table = data((vocab_rows * width) as usize, 8);
    let ids = [3i32, 0, 8, 7, 42, -1, 5];
    let mut b = Bench::new();
    let id_t = b.i32(ids.len() as u32, 1, &ids);
    let tab = b.bf16(vocab_rows, width, &table);
    let y = b.zeros(dtype::Dtype::Bf16, ids.len() as u32, width);
    if !b.run(|ctx| layout::embed(ctx, id_t, tab, 8, y)).unwrap() {
        return;
    }
    let want: Vec<f32> = ids
        .iter()
        .flat_map(|id| {
            let r = if *id < 0 || *id >= 8 { 0 } else { *id as usize };
            table[r * width as usize..(r + 1) * width as usize].to_vec()
        })
        .collect();
    assert_close(&b.read_f32(y), &want, 0.0, 0.0);
}

fn cut(x: &[f32], width: usize, lo: usize, hi: usize) -> Vec<f32> {
    x.chunks(width)
        .flat_map(|row| row[lo..hi].to_vec())
        .collect()
}

#[test]
fn splits_cut_rows_where_they_are_told() {
    let (rows, hd, heads) = (3u32, 4u32, 3u32);
    let qg = data((rows * 2 * hd * heads) as usize, 101);
    let lr = data((rows * 10) as usize, 102);
    let mut b = Bench::new();
    let packed = b.bf16(rows, 2 * hd * heads, &qg);
    let q = b.zeros(dtype::Dtype::Bf16, rows, hd * heads);
    let g = b.zeros(dtype::Dtype::Bf16, rows, hd * heads);
    let x = b.bf16(rows, 10, &lr);
    let l = b.zeros(dtype::Dtype::Bf16, rows, 4);
    let r = b.zeros(dtype::Dtype::Bf16, rows, 6);
    let ran = b
        .run(|ctx| {
            layout::split_q_gate(ctx, packed, hd, q, g)?;
            layout::split_rows(ctx, x, 4, l, r)
        })
        .unwrap();
    if !ran {
        return;
    }
    let per = (2 * hd) as usize;
    let want_q: Vec<f32> = qg
        .chunks(per)
        .flat_map(|h| h[..hd as usize].to_vec())
        .collect();
    let want_g: Vec<f32> = qg
        .chunks(per)
        .flat_map(|h| h[hd as usize..].to_vec())
        .collect();
    assert_close(&b.read_f32(q), &want_q, 0.0, 0.0);
    assert_close(&b.read_f32(g), &want_g, 0.0, 0.0);
    assert_close(&b.read_f32(l), &cut(&lr, 10, 0, 4), 0.0, 0.0);
    assert_close(&b.read_f32(r), &cut(&lr, 10, 4, 10), 0.0, 0.0);
}

#[test]
fn row_moves_gather_scatter_and_merge() {
    let width = 5usize;
    let wide = data(7 * width, 111);
    let tight = data(3 * width, 112);
    let mut b = Bench::new();
    let wt = b.bf16(7, width as u32, &wide);
    let ids = b.i32(3, 1, &[6, 0, 3]);
    let gathered = b.zeros(dtype::Dtype::Bf16, 3, width as u32);
    let tt = b.bf16(3, width as u32, &tight);
    let routes = b.i32(3, 1, &[-1, 2, 5]);
    let scattered = b.bf16(7, width as u32, &wide);
    let nine = data(9 * width, 113);
    let xt = b.bf16(9, width as u32, &nine);
    // Three rows tall for two merged rows: the third keeps its zeros.
    let merged = b.zeros(dtype::Dtype::Bf16, 3, 4 * width as u32);
    let ran = b
        .run(|ctx| {
            layout::gather_rows(ctx, wt, ids, gathered)?;
            layout::scatter_live_rows(ctx, tt, routes, scattered)?;
            layout::merge_rows(ctx, xt, 2, merged)
        })
        .unwrap();
    if !ran {
        return;
    }
    let want_g: Vec<f32> = [6usize, 0, 3]
        .iter()
        .flat_map(|r| wide[r * width..(r + 1) * width].to_vec())
        .collect();
    assert_close(&b.read_f32(gathered), &want_g, 0.0, 0.0);
    let mut want_s = wide.clone();
    for (i, r) in [-1i32, 2, 5].iter().enumerate() {
        if (0..7).contains(r) {
            let r = *r as usize;
            want_s[r * width..(r + 1) * width].copy_from_slice(&tight[i * width..(i + 1) * width]);
        }
    }
    assert_close(&b.read_f32(scattered), &want_s, 0.0, 0.0);
    let mut want_m = nine[..8 * width].to_vec();
    want_m.extend(std::iter::repeat_n(0.0, 4 * width));
    assert_close(&b.read_f32(merged), &want_m, 0.0, 0.0);
}

/// Each row's largest value ranks into a column of the i32 plane: ties to
/// the lowest index, NaN never picked, an all-NaN row 0.
#[test]
fn an_argmax_ranks_rows_on_one_pe() {
    let rows = 4u32;
    let xs: Vec<f32> = vec![
        0.5,
        2.0,
        2.0,
        -1.0, // tie: index 1
        f32::NAN,
        1.0,
        3.0,
        0.0, // NaN skipped: index 2
        f32::NAN,
        f32::NAN,
        f32::NAN,
        f32::NAN, // all NaN: 0
        -3.0,
        -2.0,
        -9.0,
        -2.5, // index 1
    ];
    let mut b = Bench::new();
    let x = b.f32(rows, 4, &xs);
    let y = b.i32(rows, 2, &[7; 8]);
    if !b.run(|ctx| layout::argmax(ctx, x, 1, y)).unwrap() {
        return;
    }
    assert_eq!(b.read_i32(y), vec![7, 1, 7, 2, 7, 0, 7, 1]);
}

/// Rows too wide for a PE split into column blocks (2 x 16384: four blocks
/// over two row groups); each block's PE finds its greatest lane and the
/// row group's first PE picks among the blocks over the fabric: the
/// winner may sit in any block, ties go to the lowest index across blocks.
#[test]
fn an_argmax_over_wide_rows_merges_its_blocks_on_the_fabric() {
    let (rows, width) = (2usize, 16384usize);
    let mut xs: Vec<f32> = data(rows * width, 77).iter().map(|v| v * 0.5).collect();
    xs[10_000] = 9.0; // row 0: block 2
    xs[width + 3] = 9.0; // row 1: block 0 ...
    xs[width + 9_000] = 9.0; // ... ties with block 2: index 3 wins
    let mut b = Bench::new();
    let x = b.f32(rows as u32, width as u32, &xs);
    let y = b.i32(rows as u32, 2, &[5; 4]);
    if !b.run(|ctx| layout::argmax(ctx, x, 0, y)).unwrap() {
        return;
    }
    let phase = b.phases.first().expect("the argmax phase");
    assert_eq!(phase.manifest.rect, (4, 2), "four blocks a row group");
    assert!(phase.pe.contains("mpi_x.gather"), "the blocks gather on the fabric");
    assert_eq!(b.read_i32(y), vec![10_000, 5, 3, 5]);
}

/// The host cuts `[q | k | v]` rows and gathers weighted embeddings.
#[test]
fn qkv_splits_and_weighted_embeddings_gather_on_the_host() {
    let rows = 3u32;
    let (qw, kw) = (4u32, 2u32);
    let xs = data((rows * (qw + 2 * kw)) as usize, 68);
    let mut b = Bench::new();
    let packed = b.f32(rows, qw + 2 * kw, &xs);
    let q = b.zeros(dtype::Dtype::F32, rows, qw);
    let k = b.zeros(dtype::Dtype::F32, rows, kw);
    let v = b.zeros(dtype::Dtype::F32, rows, kw);
    let table = data(5 * 3, 69);
    let ids = [1i32, 4, 0, 2, 9, 3];
    let weights = [0.5f32, 0.25, 1.0, -1.0, 2.0, 0.0];
    let t = b.f32(5, 3, &table);
    let it = b.i32(rows, 2, &ids);
    let wt = b.f32(rows, 2, &weights);
    let y = b.zeros(dtype::Dtype::F32, rows, 3);
    if !b
        .run(|ctx| {
            layout::split_qkv(ctx, packed, qw, kw, q, k, v)?;
            layout::embed_weighted(ctx, it, wt, t, 5, y)
        })
        .unwrap()
    {
        return;
    }
    let width = (qw + 2 * kw) as usize;
    let cut = |lo: usize, hi: usize| -> Vec<f32> {
        xs.chunks(width).flat_map(|r| r[lo..hi].to_vec()).collect()
    };
    assert_close(&b.read_f32(q), &cut(0, qw as usize), 0.0, 0.0);
    assert_close(
        &b.read_f32(k),
        &cut(qw as usize, (qw + kw) as usize),
        0.0,
        0.0,
    );
    assert_close(&b.read_f32(v), &cut((qw + kw) as usize, width), 0.0, 0.0);
    let mut want = vec![0f32; 9];
    for r in 0..3 {
        for tap in 0..2 {
            let id = ids[r * 2 + tap];
            let id = if (0..5).contains(&id) { id as usize } else { 0 };
            for c in 0..3 {
                want[r * 3 + c] += weights[r * 2 + tap] * table[id * 3 + c];
            }
        }
    }
    assert_close(&b.read_f32(y), &want, 1e-6, 1e-6);
}

/// Each row lands its ids' table rows side by side; an id outside the
/// vocabulary lands zeros.
#[test]
fn an_embed_concat_lands_each_ids_row_side_by_side() {
    let (vocab, width, heads) = (7u32, 5u32, 3u32);
    let table_data = data((vocab * width) as usize, 131);
    let mut b = Bench::new();
    let table = b.bf16(vocab, width, &table_data);
    let ids = b.i32(2, heads, &[6, 0, 3, 1, 7, -1]);
    let y = b.bf16(2, heads * width, &vec![9.0; (2 * heads * width) as usize]);
    if !b
        .run(|ctx| layout::embed_concat(ctx, ids, table, vocab, y))
        .unwrap()
    {
        return;
    }
    let mut want = Vec::new();
    for id in [6i32, 0, 3, 1, 7, -1] {
        if (0..vocab as i32).contains(&id) {
            let id = id as usize;
            want.extend_from_slice(&table_data[id * width as usize..(id + 1) * width as usize]);
        } else {
            want.extend(std::iter::repeat_n(0.0, width as usize));
        }
    }
    assert_close(&b.read_f32(y), &want, 0.0, 0.0);
}

/// The k largest of each row, largest first: ties to the lower column, NaN
/// never picked, an unfilled slot 0 at column 0.
#[test]
fn a_topk_ranks_each_rows_largest_values() {
    let rows = [
        [0.5f32, -1.0, 2.0, 2.0, 0.25],
        [f32::NAN, 1.0, f32::NAN, f32::NAN, f32::NAN],
    ];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let mut b = Bench::new();
    let x = b.f32(2, 5, &flat);
    let values = b.zeros(dtype::Dtype::F32, 2, 3);
    let indices = b.zeros(dtype::Dtype::I32, 2, 3);
    if !b
        .run(|ctx| layout::topk(ctx, x, 3, values, indices))
        .unwrap()
    {
        return;
    }
    assert_close(
        &b.read_f32(values),
        &[2.0, 2.0, 0.5, 1.0, 0.0, 0.0],
        0.0,
        0.0,
    );
    assert_eq!(b.read_i32(indices), vec![2, 3, 0, 1, 0, 0]);
}
