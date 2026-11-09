//! The token-id row gathers (PLE's, the n-gram tables) through a row table:
//! the contiguous table with no row table, with its own, and with its row
//! blocks scattered across a pool (the table garbled after) land the same
//! bits; a row table with two blocks swapped does not.

use crate::common::expert_pool::{Plane, degenerate, garble, scatter, swapped};
use crate::common::linear_ref::{t, tab};
use crate::common::{Gpu, Lcg};
use dtype::Dtype::{Bf16, E4m3, E4m3tensor, E8m0, I32, U8};
use kernels_cuda::gather::{self, RowTable};
use kernels_cuda::linear::moe::GroupSeat;

/// `fire` under each row table over `blocks` row blocks of `block_rows`
/// rows of `planes`, the result `y_bytes` at `y`.
fn same_bits(
    g: &mut Gpu,
    what: &str,
    (blocks, block_rows): (usize, u32),
    planes: &[Plane],
    (y, y_bytes): (u64, usize),
    mut fire: impl FnMut(RowTable),
) {
    let mut run = |g: &mut Gpu, table: RowTable| -> Vec<u8> {
        g.fill(y, 0x7f, y_bytes);
        fire(table);
        g.sync();
        g.down::<u8>(y, y_bytes)
    };
    let want = run(g, RowTable::RESIDENT);
    let own = degenerate(g, blocks, planes);
    assert!(
        run(g, RowTable::located(own, block_rows)) == want,
        "{what}: the contiguous table's own row table moved a bit"
    );
    let pool = scatter(g, blocks, planes, 0x0ef ^ blocks as u64);
    let wrong = swapped(g, &pool.entries, planes.len(), 0, 1);
    garble(g, blocks, planes);
    assert!(
        run(g, RowTable::located(pool.table, block_rows)) == want,
        "{what}: the scattered row table moved a bit"
    );
    assert!(
        run(g, RowTable::located(wrong, block_rows)) != want,
        "{what}: a row table with two blocks swapped landed the same bits"
    );
}

/// `rows x heads` ids over `vocab`, one in 13 outside it, blocks 0 and 1
/// among them.
fn ids(rows: usize, heads: usize, vocab: usize) -> Vec<i32> {
    tab(rows * heads, |i| {
        if i % 13 == 4 {
            -1
        } else {
            ((i * 89 + 7) % vocab) as i32
        }
    })
}

#[test]
fn a_bf16_row_gather_lands_the_same_bits_through_any_row_table() {
    let mut g = Gpu::open();
    let mut lcg = Lcg::seeded(0x40e);
    let (block_rows, blocks, width, rows, heads) = (8u32, 12usize, 96usize, 7usize, 4usize);
    let vocab = blocks * block_rows as usize;
    let (raw, _) = lcg.row(vocab * width);
    let table = g.up(&raw);
    let ids_t = t(g.up(&ids(rows, heads, vocab)), rows, heads, I32);
    let y = g.zeros(rows * heads * width * 2);
    let ctx = g.ctx();
    let stride = u64::from(block_rows) * width as u64 * 2;
    same_bits(
        &mut g,
        "bf16 embed_concat",
        (blocks, block_rows),
        &[Plane::of(table, stride)],
        (y, rows * heads * width * 2),
        |rows_at| {
            let mut yt = t(y, rows, heads * width, Bf16);
            let tt = t(table, vocab, width, Bf16);
            gather::embed_concat_at(&ctx, ids_t, tt, vocab as u32, rows_at, &mut yt)
                .expect("the gather fires");
        },
    );
}

#[test]
fn an_e4m3_row_gather_lands_the_same_bits_through_any_row_table() {
    let mut g = Gpu::open();
    // A code a thread (24), sixteen a thread (160), MXFP8 (256).
    for (width, mx) in [(24usize, false), (160, false), (256, true)] {
        let (block_rows, blocks, rows, heads) = (16u32, 9usize, 5usize, 16usize);
        let vocab = blocks * block_rows as usize;
        let br = block_rows as u64;
        let codes = g.up(&tab(vocab * width, |i| ((i * 37 + 11) % 247) as u8 & 0xf7));
        let exps = g.up(&tab(vocab * width / 32, |i| 112 + (i * 13 % 24) as u8));
        let scale = g.up(&[0.0123f32]);
        let ids_t = t(g.up(&ids(rows, heads, vocab)), rows, heads, I32);
        let y_bytes = rows * heads * width * 2;
        let y = g.zeros(y_bytes);
        let ctx = g.ctx();
        let mut planes = vec![Plane::of(codes, br * width as u64)];
        if mx {
            planes.push(Plane::of(exps, br * width as u64 / 32));
        }
        let v = vocab as u32;
        same_bits(
            &mut g,
            &format!("e4m3 embed_concat {width} mx {mx}"),
            (blocks, block_rows),
            &planes,
            (y, y_bytes),
            |rows_at| {
                let mut yt = t(y, rows, heads * width, Bf16);
                if mx {
                    let (c, e) = (
                        t(codes, vocab, width, E4m3),
                        t(exps, vocab, width / 32, E8m0),
                    );
                    gather::embed_concat_mxfp8_at(&ctx, ids_t, c, e, v, rows_at, &mut yt)
                } else {
                    let (c, s) = (t(codes, vocab, width, E4m3tensor), t(scale, 1, 4, U8));
                    gather::embed_concat_e4m3_at(&ctx, ids_t, c, s, v, rows_at, &mut yt)
                }
                .expect("the gather fires");
            },
        );
    }
}

#[test]
fn an_affine_row_gather_lands_the_same_bits_through_any_row_table() {
    let mut g = Gpu::open();
    let mut lcg = Lcg::seeded(0xaff);
    // mlx u4 (a code a nibble) and u8 rows, a scale and a bias a 32.
    for bits in [4usize, 8] {
        let (block_rows, blocks, width, group, rows) = (4u32, 10usize, 128usize, 32usize, 9usize);
        let vocab = blocks * block_rows as usize;
        let br = block_rows as u64;
        let code_row = width * bits / 8;
        let codes = g.up(&tab(vocab * code_row, |i| (i * 151 + 3) as u8));
        let (sraw, _) = lcg.row(vocab * width / group);
        let (braw, _) = lcg.row(vocab * width / group);
        let (scales, biases) = (g.up(&sraw), g.up(&braw));
        let ids_t = t(g.up(&ids(rows, 1, vocab)), rows, 1, I32);
        let y_bytes = rows * width * 2;
        let y = g.zeros(y_bytes);
        let ctx = g.ctx();
        let factor_row = (width / group * 2) as u64;
        let planes = [
            Plane::of(codes, br * code_row as u64),
            Plane::of(scales, br * factor_row),
            Plane::of(biases, br * factor_row),
        ];
        let v = vocab as u32;
        same_bits(
            &mut g,
            &format!("mlx u{bits} embed"),
            (blocks, block_rows),
            &planes,
            (y, y_bytes),
            |rows_at| {
                let mut yt = t(y, rows, width, Bf16);
                let c = t(codes, vocab, code_row, U8);
                let s = t(scales, vocab, width / group * 2, U8);
                let b = t(biases, vocab, width / group * 2, U8);
                let seat = GroupSeat::RESIDENT;
                gather::embed_mlx_affine_at(&ctx, ids_t, c, s, Some(b), v, seat, rows_at, &mut yt)
                    .expect("the gather fires");
            },
        );
    }
}
