//! **THE SPATIAL POOL, AGAINST `Gemma4VisionPooler` AND AGAINST THE ONE-HOT
//! MATMUL IT REFUSES TO BE.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test tower_spatial_pool -- --nocapture
//! ```
//!
//! `layout_fold::pool_rows` is `.wiki/alto/multimodal.md` §6.5's second op,
//! landed as the reduction §7.4 argued for rather than as the
//! `[soft_tokens, patches]` GEMM §6.5 sketched. The gates are about the two
//! things that separate the two spellings and the one thing the reduction
//! asks of the submission:
//!
//! ```text
//! (a) THE POSITION-DRIVEN REFERENCE: a CPU transcription of
//!     `Gemma4VisionPooler._avg_pool_by_positions` — one-hot by
//!     (x/k) + (W/k)*(y/k), divided by k^2, accumulated in f32 — answers
//!     exactly what the strided kernel does, on a 6x9 grid at k=3. That is
//!     the whole claim of §7.4: the two spellings compute one function once
//!     the patches are pool-block-major
//! (b) k = 1 IS THE IDENTITY, bit for bit. The upstream pooler skips itself
//!     when an image's patch count already equals its soft-token count, so
//!     this is a live case and not a degenerate one
//! (c) MULTI-IMAGE: two images of DIFFERENT grids, concatenated, pool to the
//!     concatenation of their separate pools — which is what says the op
//!     needs no image indptr, because no block straddles a boundary
//! (d) A NON-SQUARE, NON-DIVISIBLE-LOOKING GRID still pools cleanly: 4x6 at
//!     k=2 is eight blocks, and the edges are not a case because the
//!     preprocessor rounds them away (see below)
//! (e) THE RUNG TAIL: rows past the last whole block are neither read nor
//!     written -- the destination keeps what it had, which is what makes the
//!     floor safe on a patch ladder whose rungs are not multiples of nine
//! (f) the refusals fire by name: a zero side, a rectangle shorter than one
//!     block, a destination too short, a mismatched width
//! ```
//!
//! # The edge question, and why it has no edge
//!
//! Asked: does gemma4 pool a non-divisible grid by floor, by ceil, or by
//! padding? **None of the three — the case cannot arise, and the pooler
//! refuses it if it does.** Two facts from `transformers` v5.15.1 settle it:
//!
//! * `image_processing_gemma4.get_aspect_ratio_preserving_size` resizes so the
//!   target dimensions "have height and width divisible by
//!   `pooling_kernel_size * patch_size`" — it rounds DOWN to a multiple of
//!   `side_mult = pooling_kernel_size * patch_size` before patchifying. So an
//!   image's patch grid is exactly divisible by `k` on both axes, and its
//!   patch count by `k^2`;
//! * and `Gemma4VisionPooler._avg_pool_by_positions` raises rather than rounds:
//!   `if k_squared * length != input_seq_len: raise ValueError`.
//!
//! So the kernel's contract is the same one: whole blocks, and the only
//! partial block it tolerates is the RUNG TAIL, which is padding by
//! construction and gate (e) pins.
//!
//! And the divisor is `k^2` and NOT the count of live rows — `_avg_pool_by
//! _positions` builds `one_hot / k_squared` and the padding patches it zeroed
//! still sit in the denominator. Gate (a)'s reference divides the same way.

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16};

use dtype::Dtype;
use kernels_cuda::layout_fold;
use kernels_cuda::tensor::Tensor;

const WIDTH: u32 = 128;

fn fire_pool(x: &[u16], rows: u32, width: u32, side: u32, out_rows: u32) -> Vec<u16> {
    let mut gpu = Gpu::open();
    let x_at = gpu.up(x);
    let y_at = gpu.zeros((out_rows.max(1) * width) as usize * 2);
    let mut y = Tensor::new(y_at, out_rows, width, Dtype::Bf16);
    layout_fold::pool_rows(
        &gpu.ctx(),
        Tensor::new(x_at, rows, width, Dtype::Bf16),
        side,
        &mut y,
    )
    .expect("the spatial pool enqueues");
    gpu.sync();
    gpu.down(y_at, (out_rows * width) as usize)
}

/// **`Gemma4VisionPooler._avg_pool_by_positions`, WRITTEN OUT** — by
/// POSITIONS, the way transformers does it, so the strided kernel and the
/// one-hot spelling can disagree.
///
/// `positions[i]` is patch `i`'s `(x, y)` in its grid. The destination is
/// `(x/k) + (W/k) * (y/k)` with `W = max_x + 1`; the weight is `1/k²`; the
/// accumulation is f32. This deliberately scales each contribution BEFORE
/// summing, as the one-hot matmul does, where the kernel sums and then
/// divides — two roundings of one expression, and a golden that made both
/// choices the same way would be comparing the kernel to itself.
fn pool_by_positions(
    values: &[f32],
    positions: &[(u32, u32)],
    width: usize,
    side: u32,
    out_rows: usize,
) -> Vec<f32> {
    let k = side as usize;
    #[allow(clippy::cast_precision_loss)]
    let weight = 1.0f32 / (k * k) as f32;
    let grid_w = positions.iter().map(|&(x, _)| x).max().unwrap_or(0) as usize + 1;
    let blocks_w = grid_w / k;

    let mut out = vec![0.0f32; out_rows * width];
    for (row, &(x, y)) in positions.iter().enumerate() {
        let dest = (x as usize / k) + blocks_w * (y as usize / k);
        assert!(dest < out_rows, "patch {row} pools into row {dest} of {out_rows}");
        for i in 0..width {
            out[dest * width + i] += values[row * width + i] * weight;
        }
    }
    out
}

/// One image's patches in **POOL-BLOCK-MAJOR** order — the statute the op
/// asks of the submission, and §2's merge-block-major statute at `side`
/// instead of 2. Returns each row's `(x, y)`, in the order the rows are laid
/// out: block by block, and within a block row-major.
fn pool_block_major(grid_w: u32, grid_h: u32, side: u32) -> Vec<(u32, u32)> {
    assert_eq!(grid_w % side, 0, "the resize rule makes the grid divisible");
    assert_eq!(grid_h % side, 0, "the resize rule makes the grid divisible");
    let mut out = Vec::with_capacity((grid_w * grid_h) as usize);
    for by in 0..grid_h / side {
        for bx in 0..grid_w / side {
            for dy in 0..side {
                for dx in 0..side {
                    out.push((bx * side + dx, by * side + dy));
                }
            }
        }
    }
    out
}

/// (a) THE POSITION-DRIVEN REFERENCE, on gemma4's own `k = 3`.
#[test]
fn the_stride_and_the_one_hot_compute_one_function() {
    const SIDE: u32 = 3;

    let (grid_w, grid_h) = (9u32, 6u32);
    let positions = pool_block_major(grid_w, grid_h, SIDE);
    let rows = positions.len() as u32;
    let out_rows = rows / (SIDE * SIDE);

    let (raw, exact) = Lcg::seeded(7).row((rows * WIDTH) as usize);
    let landed = fire_pool(&raw, rows, WIDTH, SIDE, out_rows);
    let want = pool_by_positions(
        &exact,
        &positions,
        WIDTH as usize,
        SIDE,
        out_rows as usize,
    );

    for (at, expected) in want.iter().enumerate() {
        let got = from_bf16(landed[at]);
        assert!(
            close(got, *expected),
            "element {at} landed {got} and `_avg_pool_by_positions` says {expected}"
        );
    }
}

/// (b) `side == 1` IS THE IDENTITY, bit for bit.
#[test]
fn a_unit_side_is_the_identity() {
    let rows = 12u32;
    let (raw, _) = Lcg::seeded(64).row((rows * WIDTH) as usize);
    let landed = fire_pool(&raw, rows, WIDTH, 1, rows);
    assert_eq!(
        landed, raw,
        "a 1x1 pool moved a word, and the upstream pooler skips itself in exactly this case"
    );
}

/// (c) MULTI-IMAGE: two grids of different shapes, concatenated, and no block
/// straddles the boundary — which is why the op reads no indptr.
#[test]
fn two_images_pool_as_one_concatenation() {
    const SIDE: u32 = 3;

    // 9x6 = 54 patches = 6 blocks; 6x3 = 18 patches = 2 blocks.
    let first = pool_block_major(9, 6, SIDE);
    let second = pool_block_major(6, 3, SIDE);
    let (rows_a, rows_b) = (first.len() as u32, second.len() as u32);
    let block = SIDE * SIDE;

    let (raw_a, exact_a) = Lcg::seeded(101).row((rows_a * WIDTH) as usize);
    let (raw_b, exact_b) = Lcg::seeded(202).row((rows_b * WIDTH) as usize);

    let alone_a = fire_pool(&raw_a, rows_a, WIDTH, SIDE, rows_a / block);
    let alone_b = fire_pool(&raw_b, rows_b, WIDTH, SIDE, rows_b / block);

    let mut together_in = raw_a.clone();
    together_in.extend_from_slice(&raw_b);
    let together = fire_pool(
        &together_in,
        rows_a + rows_b,
        WIDTH,
        SIDE,
        (rows_a + rows_b) / block,
    );

    let mut want = alone_a;
    want.extend_from_slice(&alone_b);
    assert_eq!(
        together, want,
        "two images pooled together answered something other than the two pooled apart, so a \
         block crossed an image boundary"
    );

    // And the concatenation still matches the position reference, image by
    // image — the claim above with the arithmetic behind it rather than two
    // kernel runs agreeing with each other.
    for (values, positions, at) in [
        (&exact_a, &first, 0usize),
        (&exact_b, &second, (rows_a / block) as usize),
    ] {
        let out_rows = positions.len() / (block as usize);
        let reference = pool_by_positions(values, positions, WIDTH as usize, SIDE, out_rows);
        for (i, expected) in reference.iter().enumerate() {
            let got = from_bf16(together[at * WIDTH as usize + i]);
            assert!(
                close(got, *expected),
                "concatenated element {i} of the image at {at} landed {got}, reference \
                 {expected}"
            );
        }
    }
}

/// (d) a non-square grid at `k = 2`, so (a) is not a statement about one shape.
#[test]
fn a_non_square_grid_pools_by_its_own_blocks() {
    const SIDE: u32 = 2;

    let positions = pool_block_major(6, 4, SIDE);
    let rows = positions.len() as u32;
    let out_rows = rows / (SIDE * SIDE);
    let (raw, exact) = Lcg::seeded(303).row((rows * WIDTH) as usize);

    let landed = fire_pool(&raw, rows, WIDTH, SIDE, out_rows);
    let want = pool_by_positions(
        &exact,
        &positions,
        WIDTH as usize,
        SIDE,
        out_rows as usize,
    );
    for (at, expected) in want.iter().enumerate() {
        let got = from_bf16(landed[at]);
        assert!(
            close(got, *expected),
            "element {at} landed {got}, the reference says {expected}"
        );
    }
}

/// (e) THE RUNG TAIL: a rectangle whose row count is not a whole number of
/// blocks pools its whole blocks and touches nothing else.
#[test]
fn the_rows_past_the_last_whole_block_are_neither_read_nor_written() {
    const SIDE: u32 = 3;
    const BLOCK: u32 = SIDE * SIDE;

    // 64 is a real patch rung and 64 % 9 == 1: seven whole blocks and a tail.
    let rows = 64u32;
    let whole = rows / BLOCK;
    let (raw, _) = Lcg::seeded(404).row((rows * WIDTH) as usize);

    // A destination wide enough to hold the whole rectangle, so an overrun
    // would show up as a written row rather than as a fault.
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&raw);
    let y_at = gpu.up(&vec![0x3f80u16; (rows * WIDTH) as usize]);
    let before: Vec<u16> = gpu.down(y_at, (rows * WIDTH) as usize);
    let mut y = Tensor::new(y_at, rows, WIDTH, Dtype::Bf16);
    layout_fold::pool_rows(
        &gpu.ctx(),
        Tensor::new(x_at, rows, WIDTH, Dtype::Bf16),
        SIDE,
        &mut y,
    )
    .expect("a rung-padded rectangle pools its whole blocks");
    gpu.sync();
    let after: Vec<u16> = gpu.down(y_at, (rows * WIDTH) as usize);

    for at in (whole * WIDTH) as usize..after.len() {
        assert_eq!(
            after[at], before[at],
            "row {} of the destination is past the {whole} whole blocks and was written",
            at / WIDTH as usize
        );
    }
    // And the blocks that ARE whole moved, so the gate is not measuring a
    // kernel that did nothing.
    assert!(
        after[..(whole * WIDTH) as usize] != before[..(whole * WIDTH) as usize],
        "no whole block was written"
    );
}

/// (f): the refusals, by name.
#[test]
fn the_refusals_fire_by_name() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let x_at = gpu.zeros((36 * WIDTH) as usize * 2);
    let y_at = gpu.zeros((36 * WIDTH) as usize * 2);

    let source = Tensor::new(x_at, 36, WIDTH, Dtype::Bf16);
    let mut roomy = Tensor::new(y_at, 36, WIDTH, Dtype::Bf16);

    let zero_side = layout_fold::pool_rows(&ctx, source, 0, &mut roomy);
    assert!(
        format!("{:?}", zero_side.expect_err("a zero side is refused"))
            .contains("folding square"),
        "a folding square with no side is refused by name"
    );

    let mut narrow = Tensor::new(y_at, 36, WIDTH, Dtype::Bf16);
    let short = layout_fold::pool_rows(
        &ctx,
        Tensor::new(x_at, 4, WIDTH, Dtype::Bf16),
        3,
        &mut narrow,
    );
    assert!(
        format!("{:?}", short.expect_err("fewer rows than one block is refused"))
            .contains("3x3 fold"),
        "a rectangle that does not fill one block is refused by name"
    );

    let mut tiny = Tensor::new(y_at, 1, WIDTH, Dtype::Bf16);
    let cramped = layout_fold::pool_rows(&ctx, source, 3, &mut tiny);
    assert!(
        format!("{:?}", cramped.expect_err("a short destination is refused"))
            .contains("pooled rows"),
        "a destination too short for the blocks the source has is refused by name"
    );

    let mut wrong_width = Tensor::new(y_at, 36, WIDTH / 2, Dtype::Bf16);
    let widths = layout_fold::pool_rows(&ctx, source, 3, &mut wrong_width);
    assert!(
        format!("{:?}", widths.expect_err("a re-shaped row is refused"))
            .contains("folds rows"),
        "a pool that would change a row's width is refused by name"
    );

    let mut wrong = Tensor::new(y_at, 36, WIDTH, Dtype::F32);
    assert!(
        matches!(
            layout_fold::pool_rows(&ctx, Tensor::new(x_at, 36, WIDTH, Dtype::F32), 3, &mut wrong)
                .expect_err("f32 has no pool here"),
            kernels_cuda::Error::DtypeUnsupported { .. }
        ),
        "an element with no kernel is refused as a dtype"
    );
}
