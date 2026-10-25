#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, captured, forget, replay};
use dtype::Dtype;
use kernels_cuda::gather::{
    PagedPlane, gather_live_rows, page_gather_rows, page_scatter_rows, scatter_live_rows,
};
use kernels_cuda::tensor::Tensor;

/// A paged buffer laid as the recurrent store lays one: every layer's
/// planes of a page slot behind one another, each plane `PAGE_TOKENS` rows
/// of its width. The three widths take the 16-byte, 4-byte and 2-byte copy
/// units.
const PAGE_TOKENS: u32 = 4;
const SLOTS: u32 = 6;
const LAYERS: u64 = 2;
const WIDTHS: [u32; 3] = [64, 6, 5];
const ROWS: u32 = 8;

fn per_page() -> u64 {
    u64::from(PAGE_TOKENS) * WIDTHS.iter().map(|&w| u64::from(w)).sum::<u64>()
}

fn pages_elems() -> usize {
    (u64::from(SLOTS) * LAYERS * per_page()) as usize
}

fn plane(base: u64, layer: u64, which: usize) -> PagedPlane {
    let before: u64 = WIDTHS[..which].iter().map(|&w| u64::from(w)).sum();
    PagedPlane {
        base,
        per_slot: per_page() * LAYERS,
        at: layer * per_page() + u64::from(PAGE_TOKENS) * before,
        width: WIDTHS[which],
        page_tokens: PAGE_TOKENS,
        dtype: Dtype::Bf16,
    }
}

/// The element token `g` of `plane` starts at.
fn token_elem(plane: &PagedPlane, g: i32) -> usize {
    let g = g as u64;
    let (slot, in_page) = (g / u64::from(PAGE_TOKENS), g % u64::from(PAGE_TOKENS));
    (slot * plane.per_slot + plane.at + in_page * u64::from(plane.width)) as usize
}

/// A table of `count` page tokens, some rows naming none (-1), no token
/// twice.
fn tokens(lcg: &mut Lcg, count: u32) -> Vec<i32> {
    let mut free: Vec<i32> = (0..(SLOTS * PAGE_TOKENS) as i32).collect();
    (0..count)
        .map(|_| {
            let pick = ((lcg.unit() + 1.0) * 0.5 * free.len() as f32) as usize % free.len();
            if lcg.unit() < -0.6 {
                -1
            } else {
                free.swap_remove(pick)
            }
        })
        .collect()
}

/// What the page scatter must leave: `pages` with each live row whose token
/// is named copied to it.
fn scattered(
    pages: &[u16],
    plane: &PagedPlane,
    rows: &[u16],
    tokens: &[i32],
    live: std::ops::Range<usize>,
) -> Vec<u16> {
    let width = plane.width as usize;
    let mut out = pages.to_vec();
    for row in live {
        if tokens[row] >= 0 {
            let at = token_elem(plane, tokens[row]);
            out[at..at + width].copy_from_slice(&rows[row * width..(row + 1) * width]);
        }
    }
    out
}

/// What the page gather must leave: `rows` with each row whose token is
/// named read from it.
fn gathered(pages: &[u16], plane: &PagedPlane, rows: &[u16], tokens: &[i32]) -> Vec<u16> {
    let width = plane.width as usize;
    let mut out = rows.to_vec();
    for (row, &g) in tokens.iter().enumerate() {
        if g >= 0 {
            let at = token_elem(plane, g);
            out[row * width..(row + 1) * width].copy_from_slice(&pages[at..at + width]);
        }
    }
    out
}

/// A fire's rows land in their pages through one table-driven launch at
/// exactly the tokens the table names, for each plane width and layer:
/// eagerly over every row, and through a graph recorded once and replayed
/// over other fires' rows, tables and stage words (live rows and offset).
#[test]
fn the_page_scatter_lands_each_row_at_the_token_its_table_names() {
    for which in 0..WIDTHS.len() {
        for layer in 0..LAYERS {
            let mut lcg = Lcg::seeded(0x5ca7 + which as u64 * 7 + layer);
            let mut gpu = Gpu::open();
            let ctx = gpu.ctx();
            let (pages_raw, _) = lcg.row(pages_elems());
            let pages_at = gpu.up(&pages_raw);
            let plane = plane(pages_at, layer, which);
            let width = plane.width as usize;
            let (rows_raw, _) = lcg.row(ROWS as usize * width);
            let rows_at = gpu.up(&rows_raw);
            let table = tokens(&mut lcg, ROWS);
            let table_at = gpu.up(&table);
            let rows = Tensor::new(rows_at, ROWS, plane.width, Dtype::Bf16);
            let tokens_t = Tensor::new(table_at, ROWS, 1, Dtype::I32);

            page_scatter_rows(&ctx, rows, tokens_t, plane).expect("the eager scatter fires");
            gpu.sync();
            let want = scattered(&pages_raw, &plane, &rows_raw, &table, 0..ROWS as usize);
            assert_eq!(
                gpu.down::<u16>(pages_at, pages_elems()),
                want,
                "width {} layer {layer}: eager",
                plane.width
            );

            // Recorded at the ceiling with a stage word, replayed for two
            // fires of fewer live rows past other offsets.
            let win_at = gpu.up(&[ROWS, 0u32, 0, 0]);
            ctx.arm_stage(win_at);
            let exec = captured(&ctx, || {
                page_scatter_rows(&ctx, rows, tokens_t, plane).expect("the scatter captures");
            });
            ctx.disarm_stage();
            let mut have = want;
            for (live, offset) in [(3u32, 2u32), (5, 1)] {
                let (fresh_rows, _) = lcg.row(ROWS as usize * width);
                let fresh_table = tokens(&mut lcg, ROWS);
                gpu.put(rows_at, &fresh_rows);
                gpu.put(table_at, &fresh_table);
                gpu.put(win_at, &[live, offset, 0, 0]);
                replay(&ctx, exec);
                gpu.sync();
                have = scattered(
                    &have,
                    &plane,
                    &fresh_rows,
                    &fresh_table,
                    offset as usize..(offset + live) as usize,
                );
                assert_eq!(
                    gpu.down::<u16>(pages_at, pages_elems()),
                    have,
                    "width {} layer {layer}: replay at {live} live rows from {offset}",
                    plane.width
                );
            }
            forget(exec);
        }
    }
}

/// The replayed tokens come out of their pages into the rows the table
/// names, the rest left as they were, eagerly and replayed over another
/// fire's table.
#[test]
fn the_page_gather_reads_each_row_from_the_token_its_table_names() {
    for which in 0..WIDTHS.len() {
        let mut lcg = Lcg::seeded(0x9a7e + which as u64);
        let mut gpu = Gpu::open();
        let ctx = gpu.ctx();
        let (pages_raw, _) = lcg.row(pages_elems());
        let pages_at = gpu.up(&pages_raw);
        let plane = plane(pages_at, 1, which);
        let width = plane.width as usize;
        let (rows_raw, _) = lcg.row(ROWS as usize * width);
        let rows_at = gpu.up(&rows_raw);
        let table = tokens(&mut lcg, ROWS);
        let table_at = gpu.up(&table);
        let mut rows = Tensor::new(rows_at, ROWS, plane.width, Dtype::Bf16);
        let tokens_t = Tensor::new(table_at, ROWS, 1, Dtype::I32);

        page_gather_rows(&ctx, plane, tokens_t, &mut rows).expect("the eager gather fires");
        gpu.sync();
        let want = gathered(&pages_raw, &plane, &rows_raw, &table);
        assert_eq!(
            gpu.down::<u16>(rows_at, ROWS as usize * width),
            want,
            "width {}: eager",
            plane.width
        );

        let exec = captured(&ctx, || {
            page_gather_rows(&ctx, plane, tokens_t, &mut rows).expect("the gather captures");
        });
        let fresh = tokens(&mut lcg, ROWS);
        gpu.put(table_at, &fresh);
        replay(&ctx, exec);
        gpu.sync();
        assert_eq!(
            gpu.down::<u16>(rows_at, ROWS as usize * width),
            gathered(&pages_raw, &plane, &want, &fresh),
            "width {}: replayed over another table",
            plane.width
        );
        forget(exec);
    }
}

/// The live-row gather is the live-row scatter read the other way: rows
/// scattered out by a route table come back by the same table, for the
/// live rows a stage word names.
#[test]
fn the_live_gather_brings_back_what_the_live_scatter_placed() {
    let mut lcg = Lcg::seeded(0x1ce);
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    const WIDE: u32 = 12;
    for width in [64u32, 6, 5] {
        let (tight_raw, _) = lcg.row((ROWS * width) as usize);
        let tight_at = gpu.up(&tight_raw);
        let routes: Vec<i32> = vec![3, -1, 7, 0, 11, 5, -1, 9];
        let routes_at = gpu.up(&routes);
        let wide_at = gpu.up(&vec![0u16; (WIDE * width) as usize]);
        let back_at = gpu.up(&vec![0u16; (ROWS * width) as usize]);
        let tight = Tensor::new(tight_at, ROWS, width, Dtype::Bf16);
        let routes_t = Tensor::new(routes_at, ROWS, 1, Dtype::I32);
        let mut wide = Tensor::new(wide_at, WIDE, width, Dtype::Bf16);
        let mut back = Tensor::new(back_at, ROWS, width, Dtype::Bf16);
        let win_at = gpu.up(&[5u32, 2, 0, 0]);
        ctx.arm_stage(win_at);
        scatter_live_rows(&ctx, tight, routes_t, &mut wide).expect("the scatter fires");
        gather_live_rows(&ctx, wide, routes_t, &mut back).expect("the gather fires");
        ctx.disarm_stage();
        gpu.sync();
        let got: Vec<u16> = gpu.down(back_at, (ROWS * width) as usize);
        let w = width as usize;
        for row in 0..ROWS as usize {
            let live = (2..7).contains(&row) && routes[row] >= 0;
            let want: Vec<u16> = if live {
                tight_raw[row * w..(row + 1) * w].to_vec()
            } else {
                vec![0; w]
            };
            assert_eq!(
                got[row * w..(row + 1) * w],
                want[..],
                "width {width}, row {row}"
            );
        }
    }
}
