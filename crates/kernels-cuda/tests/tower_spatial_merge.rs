//! **THE SPATIAL MERGE, AGAINST `Qwen3_5VisionPatchMerger`'s OWN `view`, AND
//! THE SCATTER THAT LETS ITS TAIL SAY NOWHERE.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test tower_spatial_merge -- --nocapture
//! ```
//!
//! Two ops, one file, because the second exists for the first.
//! `layout_fold::merge_rows` is `.wiki/alto/multimodal.md` §8.1's owed
//! statement — the merger consumes `spatial_merge_size²` patch rows and
//! answers one, which is `x.view(-1, hidden_size · spatial_merge_size**2)`
//! and why `merger.linear_fc1.weight` is `[4·hidden, 4·hidden]`.
//! `layout_scatter_live::scatter_live_rows` is §8.6's answer to what that
//! leaves behind.
//!
//! ```text
//! (a) THE VIEW: the merged rectangle is `x.view(-1, side²·width)` exactly —
//!     element for element, bit for bit, over a rectangle whose rows are
//!     distinguishable so a transposed fold could not pass
//! (b) side = 1 IS THE IDENTITY, and side = 3 works too, so the op is not
//!     wired to qwen's 2
//! (c) THE TAIL: rows past the last whole fold are neither read nor written
//! (d) THE SENTINEL PLACES AND DROPS: `scatter_live_rows` writes every row
//!     whose route names one, leaves every row whose route is negative, and
//!     touches no destination row nobody named
//! (e) AND THE PAIR COMPOSES: a folded rectangle scattered through the
//!     sentinel lands its live rows and its garbage tail lands nowhere —
//!     which is the whole of §8.6, end to end
//! (f) the refusals fire by name
//! ```

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, to_bf16};

use dtype::Dtype;
use kernels_cuda::{layout_fold, layout_scatter_live};
use kernels_cuda::tensor::Tensor;

const WIDTH: u32 = 64;

fn fire_merge(x: &[u16], rows: u32, width: u32, side: u32) -> Vec<u16> {
    let block = side * side;
    let out_rows = rows / block;
    let mut gpu = Gpu::open();
    let x_at = gpu.up(x);
    let y_at = gpu.zeros((out_rows * block * width) as usize * 2);
    let mut y = Tensor::new(y_at, out_rows, block * width, Dtype::Bf16);
    layout_fold::merge_rows(
        &gpu.ctx(),
        Tensor::new(x_at, rows, width, Dtype::Bf16),
        side,
        &mut y,
    )
    .expect("the spatial merge enqueues");
    gpu.sync();
    gpu.down(y_at, (out_rows * block * width) as usize)
}

/// (a) THE VIEW. The reference is `torch`'s: `[rows, width]` read as
/// `[rows/side², side²·width]`, row-major, which puts source row `r`'s
/// element `i` at merged row `r/side²`, column `(r % side²)·width + i`.
/// Written out that way rather than as a memcpy so a fold that transposed the
/// block — merged the rows in the wrong ORDER inside a block — would fail it.
#[test]
fn the_merged_rectangle_is_the_view_the_merger_takes() {
    const SIDE: u32 = 2;
    const BLOCK: u32 = SIDE * SIDE;

    let rows = 12u32;
    let (raw, _) = Lcg::seeded(9).row((rows * WIDTH) as usize);
    let landed = fire_merge(&raw, rows, WIDTH, SIDE);

    for r in 0..rows as usize {
        let out_row = r / BLOCK as usize;
        let within = r % BLOCK as usize;
        for i in 0..WIDTH as usize {
            let at = out_row * (BLOCK * WIDTH) as usize + within * WIDTH as usize + i;
            assert_eq!(
                landed[at],
                raw[r * WIDTH as usize + i],
                "source row {r} element {i} did not land at merged row {out_row} column {}",
                within * WIDTH as usize + i
            );
        }
    }
}

/// (b) `side == 1` is the identity, and `side == 3` is not qwen's 2.
#[test]
fn the_fold_is_not_wired_to_one_merge_size() {
    let (raw, _) = Lcg::seeded(21).row((18 * WIDTH) as usize);

    let identity = fire_merge(&raw, 18, WIDTH, 1);
    assert_eq!(identity, raw, "a 1x1 merge moved a word");

    // 18 rows at side 3 is two folds of nine.
    let wide = fire_merge(&raw, 18, WIDTH, 3);
    assert_eq!(
        wide, raw,
        "a merge is a re-reading of the same bytes in the same order, whatever the side"
    );
}

/// (c) THE TAIL: rows past the last whole fold are untouched.
#[test]
fn the_rows_past_the_last_whole_fold_are_neither_read_nor_written() {
    const SIDE: u32 = 2;
    const BLOCK: u32 = SIDE * SIDE;

    // 14 rows: three whole folds and two left over.
    let rows = 14u32;
    let whole = rows / BLOCK;
    let (raw, _) = Lcg::seeded(33).row((rows * WIDTH) as usize);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&raw);
    // A destination that could hold every row, so an overrun shows up as a
    // written word rather than as a fault.
    let y_at = gpu.up(&vec![0x3f80u16; (rows * WIDTH) as usize]);
    let before: Vec<u16> = gpu.down(y_at, (rows * WIDTH) as usize);
    let mut y = Tensor::new(y_at, rows / BLOCK, BLOCK * WIDTH, Dtype::Bf16);
    layout_fold::merge_rows(
        &gpu.ctx(),
        Tensor::new(x_at, rows, WIDTH, Dtype::Bf16),
        SIDE,
        &mut y,
    )
    .expect("a rung-padded rectangle merges its whole folds");
    gpu.sync();
    let after: Vec<u16> = gpu.down(y_at, (rows * WIDTH) as usize);

    let written = (whole * BLOCK * WIDTH) as usize;
    for at in written..after.len() {
        assert_eq!(
            after[at], before[at],
            "word {at} is past the {whole} whole folds and was written"
        );
    }
    assert!(
        after[..written] != before[..written],
        "no whole fold was written"
    );
}

/// (d) THE SENTINEL PLACES AND DROPS.
#[test]
fn a_negative_route_places_nothing_and_a_named_one_places_a_row() {
    let src_rows = 6u32;
    let dst_rows = 8u32;
    let (src, _) = Lcg::seeded(45).row((src_rows * WIDTH) as usize);

    // Rows 0, 2 and 4 land at token rows 5, 1 and 7; the rest say nowhere.
    let routes: Vec<i32> = vec![5, -1, 1, -1, 7, -1];
    let sentinel = to_bf16(2.5);

    let mut gpu = Gpu::open();
    let src_at = gpu.up(&src);
    let routes_at = gpu.up(&routes);
    let dst_at = gpu.up(&vec![sentinel; (dst_rows * WIDTH) as usize]);
    let mut dst = Tensor::new(dst_at, dst_rows, WIDTH, Dtype::Bf16);
    layout_scatter_live::scatter_live_rows(
        &gpu.ctx(),
        Tensor::new(src_at, src_rows, WIDTH, Dtype::Bf16),
        Tensor::new(routes_at, src_rows, 1, Dtype::I32),
        &mut dst,
    )
    .expect("the dropping scatter enqueues");
    gpu.sync();
    let landed: Vec<u16> = gpu.down(dst_at, (dst_rows * WIDTH) as usize);

    for row in 0..dst_rows as usize {
        let at = row * WIDTH as usize;
        match routes.iter().position(|&r| r == row as i32) {
            Some(from) => {
                let want = &src[from * WIDTH as usize..(from + 1) * WIDTH as usize];
                assert_eq!(
                    &landed[at..at + WIDTH as usize],
                    want,
                    "token row {row} was named by source row {from} and did not get it"
                );
            }
            None => {
                assert!(
                    landed[at..at + WIDTH as usize].iter().all(|&w| w == sentinel),
                    "token row {row} was named by nobody and was written anyway"
                );
            }
        }
    }
}

/// (e) AND THE PAIR COMPOSES — §8.6 end to end.
#[test]
fn a_folded_rectangles_tail_lands_nowhere() {
    const SIDE: u32 = 2;
    const BLOCK: u32 = SIDE * SIDE;

    // Eight patch rows fold into two merged rows; the routes vector is
    // `[Dim::Patches]`, so it has EIGHT entries and six of them are the
    // sentinel — which is exactly the shape §8.6 describes.
    let patches = 8u32;
    let merged_rows = patches / BLOCK;
    let merged_width = BLOCK * WIDTH;
    let (raw, _) = Lcg::seeded(57).row((patches * WIDTH) as usize);
    let merged = fire_merge(&raw, patches, WIDTH, SIDE);

    let dst_rows = 5u32;
    let routes: Vec<i32> = vec![3, 0, -1, -1, -1, -1, -1, -1];
    let sentinel = to_bf16(-0.75);

    let mut gpu = Gpu::open();
    // The FULL patch rectangle as the arena would hold it: the merged rows
    // followed by garbage, which is what makes the gate about the tail.
    let mut slab = merged.clone();
    slab.extend(std::iter::repeat_n(
        to_bf16(9.0),
        ((patches - merged_rows) * merged_width) as usize,
    ));
    let src_at = gpu.up(&slab);
    let routes_at = gpu.up(&routes);
    let dst_at = gpu.up(&vec![sentinel; (dst_rows * merged_width) as usize]);
    let mut dst = Tensor::new(dst_at, dst_rows, merged_width, Dtype::Bf16);
    layout_scatter_live::scatter_live_rows(
        &gpu.ctx(),
        Tensor::new(src_at, patches, merged_width, Dtype::Bf16),
        Tensor::new(routes_at, patches, 1, Dtype::I32),
        &mut dst,
    )
    .expect("the folded rectangle scatters");
    gpu.sync();
    let landed: Vec<u16> = gpu.down(dst_at, (dst_rows * merged_width) as usize);

    for (row, want_from) in [(3usize, 0usize), (0usize, 1usize)] {
        let at = row * merged_width as usize;
        assert_eq!(
            &landed[at..at + merged_width as usize],
            &merged[want_from * merged_width as usize..(want_from + 1) * merged_width as usize],
            "token row {row} should hold merged row {want_from}"
        );
    }
    for row in [1usize, 2, 4] {
        let at = row * merged_width as usize;
        assert!(
            landed[at..at + merged_width as usize]
                .iter()
                .all(|&w| w == sentinel),
            "token row {row} was named by nobody and the fold's garbage tail reached it"
        );
    }
}

/// (f): the refusals, by name.
#[test]
fn the_refusals_fire_by_name() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let x_at = gpu.zeros((16 * WIDTH) as usize * 2);
    let y_at = gpu.zeros((16 * WIDTH) as usize * 4);
    let routes_at = gpu.up(&vec![0i32; 16]);

    let source = Tensor::new(x_at, 16, WIDTH, Dtype::Bf16);

    // The merge: a destination whose rows are not `side²` times as wide.
    let mut same_width = Tensor::new(y_at, 4, WIDTH, Dtype::Bf16);
    let narrow = layout_fold::merge_rows(&ctx, source, 2, &mut same_width);
    assert!(
        format!("{:?}", narrow.expect_err("an unwidened destination is refused"))
            .contains("concatenate into"),
        "a merge into rows of the source's width is refused by name"
    );

    let mut roomy = Tensor::new(y_at, 4, 4 * WIDTH, Dtype::Bf16);
    let short = layout_fold::merge_rows(
        &ctx,
        Tensor::new(x_at, 2, WIDTH, Dtype::Bf16),
        3,
        &mut roomy,
    );
    assert!(
        format!("{:?}", short.expect_err("fewer rows than one fold is refused"))
            .contains("3x3 fold"),
        "a rectangle that does not fill one fold is refused by name"
    );

    // The dropping scatter: a route vector of the wrong length, and one of
    // the wrong element.
    let mut dst = Tensor::new(y_at, 16, WIDTH, Dtype::Bf16);
    let miscounted = layout_scatter_live::scatter_live_rows(
        &ctx,
        source,
        Tensor::new(routes_at, 4, 1, Dtype::I32),
        &mut dst,
    );
    assert!(
        format!(
            "{:?}",
            miscounted.expect_err("a short route vector is refused")
        )
        .contains("destinations named"),
        "a route vector that does not name every row is refused by name"
    );

    let wrong_element = layout_scatter_live::scatter_live_rows(
        &ctx,
        source,
        Tensor::new(routes_at, 16, 1, Dtype::Bf16),
        &mut dst,
    );
    assert!(
        format!(
            "{:?}",
            wrong_element.expect_err("a non-i32 route vector is refused")
        )
        .contains("i32 row map"),
        "a route vector that is not i32 is refused by name"
    );
}
