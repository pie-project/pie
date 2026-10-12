//! Embedding gathers, row splits and row moves: exact copies.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::layout;

#[test]
fn layout_rows() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();

    // embed: an aligned width (16-byte lanes) and an odd one; id 99 is out
    // of the vocabulary and reads row 0.
    for hidden in [1024u32, 1001] {
        let vocab = 50u32;
        let table_v: Vec<f32> = values((vocab * hidden) as usize, 1, 1.0).into_iter().map(round_bf16).collect();
        let table = gpu.tensor(&table_v, vocab, hidden, Dtype::Bf16);
        let ids = [3i32, 0, 49, 99, 7];
        let id_t = gpu.column(&ids);
        let mut y = gpu.zeros(ids.len() as u32, hidden, Dtype::Bf16);
        layout::embed(&ctx, id_t, table, vocab, &mut y).unwrap();
        let got = gpu.read(&y);
        for (n, &id) in ids.iter().enumerate() {
            let src = if (0..vocab as i32).contains(&id) { id as usize } else { 0 };
            let h = hidden as usize;
            assert_eq!(&got[n * h..(n + 1) * h], &table_v[src * h..(src + 1) * h], "embed row {n}");
        }
    }

    // split_q_gate: 3 rows x 4 heads of [q | g], head 256.
    let (rows, heads, hd) = (3u32, 4u32, 256u32);
    let packed_v: Vec<f32> = values((rows * heads * 2 * hd) as usize, 2, 1.0).into_iter().map(round_bf16).collect();
    let packed = gpu.tensor(&packed_v, rows, heads * 2 * hd, Dtype::Bf16);
    let mut q = gpu.zeros(rows, heads * hd, Dtype::Bf16);
    let mut g = gpu.zeros(rows, heads * hd, Dtype::Bf16);
    layout::split_q_gate(&ctx, packed, hd, &mut q, &mut g).unwrap();
    let (qv, gv) = (gpu.read(&q), gpu.read(&g));
    for head in 0..(rows * heads) as usize {
        let hd = hd as usize;
        assert_eq!(&qv[head * hd..(head + 1) * hd], &packed_v[head * 2 * hd..head * 2 * hd + hd]);
        assert_eq!(&gv[head * hd..(head + 1) * hd], &packed_v[head * 2 * hd + hd..(head + 1) * 2 * hd]);
    }

    // split_rows: the GDN in_proj cut (16-byte path) and an odd one.
    for (lw, rw, dtype) in [(4096u32, 2048u32, Dtype::Bf16), (13, 7, Dtype::Bf16), (40, 24, Dtype::F32)] {
        let rows = 5u32;
        let xv: Vec<f32> = values((rows * (lw + rw)) as usize, 3, 1.0).into_iter().map(round_bf16).collect();
        let x = gpu.tensor(&xv, rows, lw + rw, dtype);
        let mut l = gpu.zeros(rows, lw, dtype);
        let mut r = gpu.zeros(rows, rw, dtype);
        layout::split_rows(&ctx, x, lw, &mut l, &mut r).unwrap();
        let (lv, rv) = (gpu.read(&l), gpu.read(&r));
        for n in 0..rows as usize {
            let row = &xv[n * (lw + rw) as usize..(n + 1) * (lw + rw) as usize];
            assert_eq!(&lv[n * lw as usize..(n + 1) * lw as usize], &row[..lw as usize], "left {lw}+{rw} row {n}");
            assert_eq!(&rv[n * rw as usize..(n + 1) * rw as usize], &row[lw as usize..], "right {lw}+{rw} row {n}");
        }
    }

    // gather_rows: pick rows 4, 0, 2 of a 5-row plane.
    let width = 2048u32;
    let wide_v: Vec<f32> = values((5 * width) as usize, 4, 1.0).into_iter().map(round_bf16).collect();
    let wide = gpu.tensor(&wide_v, 5, width, Dtype::Bf16);
    let index = gpu.column(&[4, 0, 2]);
    let mut tight = gpu.zeros(3, width, Dtype::Bf16);
    layout::gather_rows(&ctx, wide, index, &mut tight).unwrap();
    let got = gpu.read(&tight);
    for (n, src) in [4usize, 0, 2].into_iter().enumerate() {
        let w = width as usize;
        assert_eq!(&got[n * w..(n + 1) * w], &wide_v[src * w..(src + 1) * w], "gather row {n}");
    }
}
