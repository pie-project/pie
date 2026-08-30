//! **THE LEARNED POSITION EMBEDDING, AS A GATHER — AGAINST TEXTBOOK BILINEAR
//! AND AGAINST THE PLAIN GATHER IT DEGENERATES TO.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test tower_pos_embed -- --nocapture
//! ```
//!
//! `.wiki/alto/multimodal.md` §9.2 killed §6.4's one-hot bake — the one-hot is
//! `num_position_embeddings` wide, so an image of 2304 patches would ship
//! 10.6 MiB of bf16 zeros to address a 3.4 MiB table, and an import cannot
//! compute a resample anyway. The working spelling is a gather:
//! `layout.embed` on the native grid, `layout.embed_weighted` off it.
//!
//! **THE RESAMPLE ARITHMETIC IS THE HOST'S, so this file owns a transcription
//! of it** — `_interpolation_axis_taps_weights` + the 2-D outer product from
//! `transformers/vision_utils.py` (v5.15.1), which is what a submission runs.
//! qwen3_5's tower states `num_position_embeddings: 2304` (`num_grid_per_side
//! = 48`), `interpolation_mode = "bilinear"` and `interpolation_align_corners
//! = True`, all three read off `Qwen3_5VisionModel.__init__`.
//!
//! ```text
//! (a) THE RESAMPLE IS THE RESAMPLE: taps and weights from the transcription,
//!     gathered by the kernel, equal a TEXTBOOK align_corners bilinear resample
//!     of the same table written without reference to the helper — so the gate
//!     pins the host formula and the kernel at once
//! (b) THE NATIVE GRID DEGENERATES: at target grid == stored grid the helper
//!     answers weight 1 on the patch's own row and 0 elsewhere, so the cheap
//!     path is a fact about the arithmetic and not a special case
//! (c) ONE TAP AT WEIGHT ONE IS `layout::embed`, bit for bit — which is what
//!     lets a native-grid text write the plain op and get this op's answer
//! (d) the weights are honoured: perturbing one moves the row by exactly the
//!     tap it weights
//! (e) an id past the table clamps to row zero, `layout::embed`'s own rule
//! (f) the refusals fire by name
//! ```

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};

use dtype::Dtype;
use kernels_cuda::tensor::Tensor;
use kernels_cuda::{layout, layout_embed_weighted};

/// The stored grid's side. qwen3_5 states 48 (`2304 = 48²`); this is the same
/// arithmetic at a size a golden can print.
const SIDE: usize = 8;

const HIDDEN: u32 = 96;

const TAPS: u32 = 4;

/// **`_interpolation_axis_taps_weights`, TRANSCRIBED** — bilinear, two taps.
///
/// `index` is the target position on an axis of length `size`; `side` is the
/// stored table's. `align_corners` picks the mapping, and qwen states `True`.
fn axis_taps(index: usize, size: usize, side: usize, align_corners: bool) -> ([usize; 2], [f32; 2]) {
    #[allow(clippy::cast_precision_loss)]
    let src = if align_corners {
        // Closed form of `linspace(0, side-1, size)[index]`; `clamp(min=1)`
        // is the size == 1 guard, where index is 0 and src is 0 too.
        index as f32 * (side as f32 - 1.0) / (size.saturating_sub(1).max(1)) as f32
    } else {
        (index as f32 + 0.5) * side as f32 / size as f32 - 0.5
    };
    let floor = src.floor();
    let mut taps = [0usize; 2];
    let mut weights = [0f32; 2];
    for (t, offset) in [0f32, 1f32].into_iter().enumerate() {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let tap = (floor as i64 + offset as i64).clamp(0, side as i64 - 1) as usize;
        taps[t] = tap;
        // The linear hat kernel.
        weights[t] = (1.0 - (src - floor - offset).abs()).max(0.0);
    }
    (taps, weights)
}

/// The 2-D case: the separable outer product of the two axes' taps and
/// weights, flattened to `n·n` per patch — `indices = h_taps·side + w_taps`.
fn interp(
    row: usize,
    col: usize,
    grid_h: usize,
    grid_w: usize,
    side: usize,
    align_corners: bool,
) -> ([i32; 4], [f32; 4]) {
    let (h_taps, h_w) = axis_taps(row, grid_h, side, align_corners);
    let (w_taps, w_w) = axis_taps(col, grid_w, side, align_corners);
    let mut ids = [0i32; 4];
    let mut weights = [0f32; 4];
    for a in 0..2 {
        for b in 0..2 {
            ids[a * 2 + b] = (h_taps[a] * side + w_taps[b]) as i32;
            weights[a * 2 + b] = h_w[a] * w_w[b];
        }
    }
    (ids, weights)
}

/// **TEXTBOOK `align_corners=True` BILINEAR**, written without reference to
/// the helper above so the two can disagree: map the target cell to a source
/// coordinate, take the four surrounding table rows, blend by the fractional
/// parts.
fn textbook_bilinear(
    table: &[f32],
    side: usize,
    grid_h: usize,
    grid_w: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; grid_h * grid_w * hidden];
    for i in 0..grid_h {
        for j in 0..grid_w {
            #[allow(clippy::cast_precision_loss)]
            let y = i as f32 * (side as f32 - 1.0) / (grid_h.saturating_sub(1).max(1)) as f32;
            #[allow(clippy::cast_precision_loss)]
            let x = j as f32 * (side as f32 - 1.0) / (grid_w.saturating_sub(1).max(1)) as f32;
            let (y0, x0) = (y.floor(), x.floor());
            let (dy, dx) = (y - y0, x - x0);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let (y0, x0) = (y0 as usize, x0 as usize);
            let y1 = (y0 + 1).min(side - 1);
            let x1 = (x0 + 1).min(side - 1);
            let at = (i * grid_w + j) * hidden;
            for h in 0..hidden {
                let corner = |r: usize, c: usize| table[(r * side + c) * hidden + h];
                out[at + h] = (1.0 - dy) * (1.0 - dx) * corner(y0, x0)
                    + (1.0 - dy) * dx * corner(y0, x1)
                    + dy * (1.0 - dx) * corner(y1, x0)
                    + dy * dx * corner(y1, x1);
            }
        }
    }
    out
}

fn fire_weighted(
    table: &[u16],
    vocab: u32,
    ids: &[i32],
    weights: &[f32],
    rows: u32,
    taps: u32,
) -> Vec<u16> {
    let mut gpu = Gpu::open();
    let table_at = gpu.up(table);
    let ids_at = gpu.up(ids);
    let w_at = gpu.up(weights);
    let y_at = gpu.zeros((rows * HIDDEN) as usize * 2);
    let mut y = Tensor::new(y_at, rows, HIDDEN, Dtype::Bf16);
    layout_embed_weighted::embed_weighted(
        &gpu.ctx(),
        Tensor::new(ids_at, rows, taps, Dtype::I32),
        Tensor::new(w_at, rows, taps, Dtype::F32),
        Tensor::new(table_at, vocab, HIDDEN, Dtype::Bf16),
        vocab,
        &mut y,
    )
    .expect("the interpolating gather enqueues");
    gpu.sync();
    gpu.down(y_at, (rows * HIDDEN) as usize)
}

fn fire_plain(table: &[u16], vocab: u32, ids: &[i32], rows: u32) -> Vec<u16> {
    let mut gpu = Gpu::open();
    let table_at = gpu.up(table);
    let ids_at = gpu.up(ids);
    let y_at = gpu.zeros((rows * HIDDEN) as usize * 2);
    let mut y = Tensor::new(y_at, rows, HIDDEN, Dtype::Bf16);
    layout::embed(
        &gpu.ctx(),
        Tensor::new(ids_at, rows, 1, Dtype::I32),
        Tensor::new(table_at, vocab, HIDDEN, Dtype::Bf16),
        vocab,
        &mut y,
    )
    .expect("the plain gather enqueues");
    gpu.sync();
    gpu.down(y_at, (rows * HIDDEN) as usize)
}

/// The stored table, and the f32 the device will read it back as.
fn table(seed: u64) -> (Vec<u16>, Vec<f32>) {
    Lcg::seeded(seed).row(SIDE * SIDE * HIDDEN as usize)
}

/// Every patch of a `grid_h × grid_w` image, in RASTER order, with its taps
/// and weights.
fn streams(grid_h: usize, grid_w: usize, align_corners: bool) -> (Vec<i32>, Vec<f32>) {
    let mut ids = Vec::with_capacity(grid_h * grid_w * 4);
    let mut weights = Vec::with_capacity(grid_h * grid_w * 4);
    for i in 0..grid_h {
        for j in 0..grid_w {
            let (tap, weight) = interp(i, j, grid_h, grid_w, SIDE, align_corners);
            ids.extend_from_slice(&tap);
            weights.extend_from_slice(&weight);
        }
    }
    (ids, weights)
}

/// (a) THE RESAMPLE IS THE RESAMPLE.
#[test]
fn the_gathered_taps_are_a_bilinear_resample_of_the_table() {
    let (raw, exact) = table(11);
    let vocab = (SIDE * SIDE) as u32;

    // Two grids and both align_corners settings: a grid finer than the table
    // and one coarser, so the gate is not about one direction of resampling.
    for (grid_h, grid_w) in [(6usize, 5usize), (12usize, 10usize)] {
        let rows = (grid_h * grid_w) as u32;
        let (ids, weights) = streams(grid_h, grid_w, true);
        let landed = fire_weighted(&raw, vocab, &ids, &weights, rows, TAPS);
        let want = textbook_bilinear(&exact, SIDE, grid_h, grid_w, HIDDEN as usize);

        for (at, expected) in want.iter().enumerate() {
            let got = from_bf16(landed[at]);
            assert!(
                close(got, *expected),
                "{grid_h}x{grid_w}: element {at} landed {got}, textbook bilinear says {expected}"
            );
        }
    }
}

/// (b) THE NATIVE GRID DEGENERATES — the arithmetic says so, not a branch.
#[test]
fn the_native_grid_puts_all_the_weight_on_its_own_row() {
    for i in 0..SIDE {
        for j in 0..SIDE {
            let (ids, weights) = interp(i, j, SIDE, SIDE, SIDE, true);
            let own = (i * SIDE + j) as i32;
            let mut mass = 0.0f32;
            for (tap, weight) in ids.iter().zip(&weights) {
                if *weight > 0.0 {
                    assert_eq!(
                        *tap, own,
                        "patch ({i}, {j}) of a native grid put weight {weight} on table row \
                         {tap}, and its own row is {own}"
                    );
                }
                mass += weight;
            }
            assert!(
                (mass - 1.0).abs() < 1.0e-6,
                "patch ({i}, {j})'s taps carry mass {mass}"
            );
        }
    }
}

/// (c) ONE TAP AT WEIGHT ONE IS `layout::embed`, bit for bit.
#[test]
fn a_single_tap_at_weight_one_is_the_plain_gather() {
    let (raw, _) = table(29);
    let vocab = (SIDE * SIDE) as u32;
    let rows = 24u32;

    let ids: Vec<i32> = (0..rows as i32).map(|r| (r * 7) % vocab as i32).collect();
    let weights = vec![1.0f32; rows as usize];

    let weighted = fire_weighted(&raw, vocab, &ids, &weights, rows, 1);
    let plain = fire_plain(&raw, vocab, &ids, rows);
    assert_eq!(
        weighted, plain,
        "one tap at weight one answered something other than the gather it is"
    );
}

/// (d) the weights are honoured, one tap at a time.
#[test]
fn moving_one_weight_moves_exactly_its_own_tap() {
    let (raw, exact) = table(43);
    let vocab = (SIDE * SIDE) as u32;
    let rows = 4u32;

    // Four distinct table rows per patch, so a tap that was ignored shows up.
    let ids: Vec<i32> = (0..rows as i32 * 4).map(|t| (t * 5) % vocab as i32).collect();
    let base = vec![0.25f32; (rows * TAPS) as usize];
    let landed = fire_weighted(&raw, vocab, &ids, &base, rows, TAPS);

    let mut bumped = base.clone();
    bumped[TAPS as usize] += 1.0; // row 1's first tap
    let after = fire_weighted(&raw, vocab, &ids, &bumped, rows, TAPS);

    let hidden = HIDDEN as usize;
    for r in 0..rows as usize {
        for h in 0..hidden {
            let at = r * hidden + h;
            let (before, now) = (from_bf16(landed[at]), from_bf16(after[at]));
            if r == 1 {
                let tap = exact[ids[TAPS as usize] as usize * hidden + h];
                assert!(
                    close(now, before + tap),
                    "row 1 element {h}: bumping the first weight by one should add that tap \
                     ({tap}), and {before} became {now}"
                );
            } else {
                assert_eq!(
                    landed[at], after[at],
                    "row {r} element {h} moved when only row 1's weight did"
                );
            }
        }
    }
}

/// (e) an id past the table clamps to row zero — `layout::embed`'s own rule,
/// so a checked-host-side vector that slips through reads a defined row.
#[test]
fn an_id_past_the_table_reads_row_zero() {
    let (raw, exact) = table(57);
    let vocab = (SIDE * SIDE) as u32;

    let ids = vec![vocab as i32 + 3, -9, 0, 1];
    let weights = vec![1.0f32, 0.0, 0.0, 0.0];
    let landed = fire_weighted(&raw, vocab, &ids, &weights, 1, TAPS);
    for h in 0..HIDDEN as usize {
        let got = from_bf16(landed[h]);
        assert!(
            close(got, exact[h]),
            "element {h} of a row whose only live tap is out of range landed {got}, and table \
             row zero holds {}",
            exact[h]
        );
    }
}

/// (f): the refusals, by name.
#[test]
fn the_refusals_fire_by_name() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let vocab = (SIDE * SIDE) as u32;
    let table_at = gpu.up(&vec![to_bf16(1.0); SIDE * SIDE * HIDDEN as usize]);
    let ids_at = gpu.up(&vec![0i32; 32]);
    let w_at = gpu.up(&vec![0.25f32; 32]);
    let y_at = gpu.zeros((8 * HIDDEN) as usize * 2);

    let bank = Tensor::new(table_at, vocab, HIDDEN, Dtype::Bf16);
    let ids = Tensor::new(ids_at, 8, TAPS, Dtype::I32);
    let weights = Tensor::new(w_at, 8, TAPS, Dtype::F32);
    let mut y = Tensor::new(y_at, 8, HIDDEN, Dtype::Bf16);

    let bf16_weights = layout_embed_weighted::embed_weighted(
        &ctx,
        ids,
        Tensor::new(w_at, 8, TAPS, Dtype::Bf16),
        bank,
        vocab,
        &mut y,
    );
    assert!(
        format!(
            "{:?}",
            bf16_weights.expect_err("bf16 interpolation weights are refused")
        )
        .contains("preprocessor's arithmetic"),
        "weights that are not f32 are refused by name"
    );

    let int_ids = layout_embed_weighted::embed_weighted(
        &ctx,
        Tensor::new(ids_at, 8, TAPS, Dtype::F32),
        weights,
        bank,
        vocab,
        &mut y,
    );
    assert!(
        format!("{:?}", int_ids.expect_err("non-i32 taps are refused")).contains("i32 rows"),
        "taps that are not i32 are refused by name"
    );

    let mismatched = layout_embed_weighted::embed_weighted(
        &ctx,
        ids,
        Tensor::new(w_at, 8, 2, Dtype::F32),
        bank,
        vocab,
        &mut y,
    );
    assert!(
        format!(
            "{:?}",
            mismatched.expect_err("a weight rectangle of another shape is refused")
        )
        .contains("every tap is weighted"),
        "taps and weights that disagree are refused by name"
    );

    let mut short = Tensor::new(y_at, 2, HIDDEN, Dtype::Bf16);
    let rows = layout_embed_weighted::embed_weighted(&ctx, ids, weights, bank, vocab, &mut short);
    assert!(
        format!("{:?}", rows.expect_err("a destination of another row count is refused"))
            .contains("one row per index row"),
        "a gather that would answer a different row count is refused by name"
    );
}
