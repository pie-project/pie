//! The three layout points, on the GPU, for the first time.
//!
//! `layout/embed.metal` and `layout/deinterleave.metal` were written last
//! week on a Linux box with no Metal compiler and no Apple device, and both
//! headers say so in as many words: "Never compiled, never run, no number
//! compared against anything." They compile now -- the whole entrypoint
//! census builds a pipeline -- and this file is the other half of that
//! sentence.
//!
//! # These three are EXACT, and that is the whole shape of the test
//!
//! Every other point this sweep measures does arithmetic, so it gets a
//! reference and a tolerance. These three do not: `embed` copies a bf16 row
//! out of a table, `split_rows` copies a packed row into two, `select_slice`
//! copies a column window. No element is ever widened, combined or rounded,
//! so the only correct answer is the INPUT BIT, and the comparison below is
//! `assert_eq!` over `u16`-exact values rather than a bound.
//!
//! That makes the reference a CPU model in the test, and it makes it a
//! trivially honest one: a gather written twice cannot disagree about
//! rounding, only about an address. Addresses are what these kernels are, so
//! that is the right thing for the two spellings to be able to disagree
//! about.
//!
//! # What each fixture is shaped to catch
//!
//! **The vocab clamp.** `embed`'s ids arrive from a wire payload, and the
//! shader's `(raw >= 0 && raw < vocab) ? raw : 0` is the difference between a
//! wrong answer and an out-of-bounds read into the largest tensor in the
//! model. So the id stream carries `-1` and `vocab` beside three real tokens,
//! and both must land on row zero.
//!
//! **The threadgroup tail.** `elementwise_rows` states the grid as the exact
//! rectangle -- `[width, rows, 1]` -- against a threadgroup of 256, and
//! `driver-metal` turns that into `dispatchThreads:`, which launches a
//! PARTIAL final threadgroup rather than rounding the grid up. Both shader
//! headers claim the opposite ("the grid is rounded up to whole
//! threadgroups, so the tail runs over the end of a row") and guard against
//! it. The widths here are 260 and 257 -- one and two past a threadgroup --
//! so if the driver ever did round up, the guard is what would be keeping
//! this file green, and if it does not, the row after the tail must be
//! untouched. Both are checked: the fixtures allocate a slack row past every
//! result and require it to still hold its poison.
//!
//! **The pitch.** `select_slice`'s source row is `layers * width` wide and
//! its result row is `width`, which is two different strides in one
//! expression; `split_rows`' source row is `left + right` while its two
//! results are `left` and `right`, which is three. A gather that used one of
//! them for another produces a plausible rectangle full of the wrong
//! elements, so the fixtures give every width a different value and no width
//! divides another.
//!
//! # The mutations
//!
//! Five, one per address these kernels compute, and each is a defect someone
//! could plausibly write rather than a value perturbed to make a number
//! move: the clamp landing on row one, the embed row stride dropped, the
//! packed row read at the left half's pitch, the layer offset dropped, and
//! the relay row read at the slice's pitch. Every one is in bounds -- a
//! mutation that faults proves nothing about the comparison -- and every one
//! must turn this file red.

#![cfg(target_vendor = "apple")]

mod plane;

use driver_metal::skip::skipped;
use plane::{Arg, Rig};

const FILE_EMBED: &str = "layout/embed.metal";
const FILE_CUTS: &str = "layout/deinterleave.metal";

/// Wider than one threadgroup by four lanes, so the last group is partial.
const HIDDEN: usize = 260;
const VOCAB: usize = 6;
const TOKENS: usize = 5;

/// The two halves of `split_rows`, neither a multiple of the other and their
/// sum one past a threadgroup.
const LEFT: usize = 100;
const RIGHT: usize = 157;

/// `select_slice`'s relay: four layers of forty columns.
const LAYERS: usize = 4;
const SLICE: usize = 40;
const PICKED: usize = 2;
const RELAY_ROWS: usize = 3;

/// What an unwritten output element holds, so that a column the kernel never
/// reaches reads as poison rather than as a plausible zero.
const POISON: f32 = -99.0;

/// The ids the gather is fired with: three in range, and the two ways a wire
/// payload is out of it.
const IDS: [i32; TOKENS] = [3, 0, -1, VOCAB as i32, 2];

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_embedding_gather_clamps_a_token_the_vocabulary_does_not_have() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `layout.embed` was not fired");
        return;
    };

    let table = plane::spread(VOCAB * HIDDEN, 1);
    let got = embed(&rig, plane::kernels_dir().as_path(), &table);
    let want = embed_reference(&table);

    assert_eq!(
        got[..TOKENS * HIDDEN],
        want[..],
        "`embed_bfloat16` gathers the row the clamp names"
    );
    slack_is_untouched(&got, TOKENS * HIDDEN, "embed");

    // The clamp, landing one row over. Out-of-vocabulary reads row 1 rather
    // than row 0, which is in bounds, finite, and the wrong token.
    let root = plane::mutant(FILE_EMBED, "? raw : 0", "? raw : 1");
    let bent = embed(&rig, root.path(), &table);
    assert_ne!(
        bent[..TOKENS * HIDDEN],
        want[..],
        "a clamp that lands on row 1 must not agree with one that lands on row 0"
    );

    // The row stride, dropped. Every token gathers row zero, which is what a
    // gather that forgot it is indexing a table looks like.
    let root = plane::mutant(
        FILE_EMBED,
        "table[size_t(row) * size_t(hidden) + size_t(c)]",
        "table[size_t(c)]",
    );
    let bent = embed(&rig, root.path(), &table);
    assert_ne!(
        bent[..TOKENS * HIDDEN],
        want[..],
        "a gather with no row stride must not agree with one that has it"
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_packed_row_splits_at_the_column_the_statement_states() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `layout.split_rows` was not fired");
        return;
    };

    let rows = RELAY_ROWS;
    let src = plane::spread(rows * (LEFT + RIGHT), 2);
    let (left, right) = split(&rig, plane::kernels_dir().as_path(), &src, rows);
    let (want_left, want_right) = split_reference(&src, rows);

    assert_eq!(
        left[..rows * LEFT],
        want_left[..],
        "`split_rows_bfloat16` lands the first {LEFT} columns of every row"
    );
    assert_eq!(
        right[..rows * RIGHT],
        want_right[..],
        "`split_rows_bfloat16` lands the remaining {RIGHT}"
    );
    slack_is_untouched(&left, rows * LEFT, "split_rows' left half");
    slack_is_untouched(&right, rows * RIGHT, "split_rows' right half");

    // The packed row read at the LEFT half's pitch, which is the shape of
    // every wrong answer this kernel can give: a rectangle of the right size
    // holding another row's elements.
    let root = plane::mutant(
        FILE_CUTS,
        "src[row * size_t(total) + size_t(c)]",
        "src[row * size_t(left_dim) + size_t(c)]",
    );
    let (left, right) = split(&rig, root.path(), &src, rows);
    assert!(
        left[..rows * LEFT] != want_left[..] || right[..rows * RIGHT] != want_right[..],
        "a cut that reads the packed row at the left half's pitch must not \
         agree with one that reads it at the packed row's"
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_relay_select_takes_the_layer_the_host_computed_the_offset_for() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `layout.select` was not fired");
        return;
    };

    let table = plane::spread(RELAY_ROWS * LAYERS * SLICE, 3);
    let got = select(&rig, plane::kernels_dir().as_path(), &table);
    let want = select_reference(&table);

    assert_eq!(
        got[..RELAY_ROWS * SLICE],
        want[..],
        "`select_slice_bfloat16` takes layer {PICKED}'s columns"
    );
    slack_is_untouched(&got, RELAY_ROWS * SLICE, "select");

    // The offset, dropped: every layer's statement answers layer zero. The
    // shader's header calls the offset "the whole arithmetic" of this point,
    // and this is what it looks like when it is not there.
    let root = plane::mutant(
        FILE_CUTS,
        "table[row * size_t(stride) + size_t(offset) + size_t(c)]",
        "table[row * size_t(stride) + size_t(c)]",
    );
    let bent = select(&rig, root.path(), &table);
    assert_ne!(
        bent[..RELAY_ROWS * SLICE],
        want[..],
        "a select with no layer offset must not agree with one that has it"
    );

    // The relay row read at the SLICE's pitch rather than the relay's, which
    // is the other of the two strides in that one expression.
    let root = plane::mutant(
        FILE_CUTS,
        "table[row * size_t(stride) + size_t(offset) + size_t(c)]",
        "table[row * size_t(width) + size_t(offset) + size_t(c)]",
    );
    let bent = select(&rig, root.path(), &table);
    assert_ne!(
        bent[..RELAY_ROWS * SLICE],
        want[..],
        "a select that strides by the slice must not agree with one that \
         strides by the relay"
    );
}

/// `y[n, :] = table[clamp(ids[n]), :]`, in Rust.
fn embed_reference(table: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0; TOKENS * HIDDEN];
    for (n, id) in IDS.iter().enumerate() {
        let row = if *id >= 0 && (*id as usize) < VOCAB {
            *id as usize
        } else {
            0
        };
        for c in 0..HIDDEN {
            y[n * HIDDEN + c] = plane::narrowed(table[row * HIDDEN + c]);
        }
    }
    y
}

fn split_reference(src: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>) {
    let total = LEFT + RIGHT;
    let mut left = vec![0.0; rows * LEFT];
    let mut right = vec![0.0; rows * RIGHT];
    for r in 0..rows {
        for c in 0..total {
            let v = plane::narrowed(src[r * total + c]);
            if c < LEFT {
                left[r * LEFT + c] = v;
            } else {
                right[r * RIGHT + (c - LEFT)] = v;
            }
        }
    }
    (left, right)
}

fn select_reference(table: &[f32]) -> Vec<f32> {
    let stride = LAYERS * SLICE;
    let mut y = vec![0.0; RELAY_ROWS * SLICE];
    for r in 0..RELAY_ROWS {
        for c in 0..SLICE {
            y[r * SLICE + c] = plane::narrowed(table[r * stride + PICKED * SLICE + c]);
        }
    }
    y
}

/// The grid `kernels_metal::layout` states for all three: the exact
/// rectangle, against a threadgroup of 256.
fn rows_grid(width: usize, rows: usize) -> ([u32; 3], [u32; 3]) {
    ([width as u32, rows as u32, 1], [256, 1, 1])
}

/// A result allocated one row long, so that a kernel writing past its
/// rectangle is visible rather than silent.
fn slack(n: usize, width: usize) -> Vec<f32> {
    vec![POISON; n + width]
}

fn slack_is_untouched(got: &[f32], n: usize, what: &str) {
    assert!(
        got[n..].iter().all(|v| *v == POISON),
        "{what} wrote past the rectangle its point states"
    );
}

fn embed(rig: &Rig, root: &std::path::Path, table: &[f32]) -> Vec<f32> {
    let ids = plane::alloc_i32(&rig.context, &IDS, "ids");
    let table = plane::alloc_bf16(&rig.context, table, "table");
    let y = plane::alloc_bf16(&rig.context, &slack(TOKENS * HIDDEN, HIDDEN), "y");
    let (grid, group) = rows_grid(HIDDEN, TOKENS);
    plane::fire(
        rig,
        root,
        FILE_EMBED,
        "embed_bfloat16",
        grid,
        group,
        &[
            Arg::Buf(&ids),
            Arg::Buf(&table),
            Arg::Buf(&y),
            Arg::I32(HIDDEN as i32),
            Arg::I32(VOCAB as i32),
        ],
    );
    plane::read_bf16(&y, TOKENS * HIDDEN + HIDDEN)
}

fn split(rig: &Rig, root: &std::path::Path, src: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>) {
    let src = plane::alloc_bf16(&rig.context, src, "packed");
    let left = plane::alloc_bf16(&rig.context, &slack(rows * LEFT, LEFT), "left");
    let right = plane::alloc_bf16(&rig.context, &slack(rows * RIGHT, RIGHT), "right");
    let (grid, group) = rows_grid(LEFT + RIGHT, rows);
    plane::fire(
        rig,
        root,
        FILE_CUTS,
        "split_rows_bfloat16",
        grid,
        group,
        &[
            Arg::Buf(&src),
            Arg::Buf(&left),
            Arg::Buf(&right),
            Arg::I32(LEFT as i32),
            Arg::I32(RIGHT as i32),
        ],
    );
    (
        plane::read_bf16(&left, rows * LEFT + LEFT),
        plane::read_bf16(&right, rows * RIGHT + RIGHT),
    )
}

fn select(rig: &Rig, root: &std::path::Path, table: &[f32]) -> Vec<f32> {
    let stride = LAYERS * SLICE;
    let table = plane::alloc_bf16(&rig.context, table, "relay");
    let y = plane::alloc_bf16(&rig.context, &slack(RELAY_ROWS * SLICE, SLICE), "y");
    let (grid, group) = rows_grid(SLICE, RELAY_ROWS);
    plane::fire(
        rig,
        root,
        FILE_CUTS,
        "select_slice_bfloat16",
        grid,
        group,
        &[
            Arg::Buf(&table),
            Arg::Buf(&y),
            Arg::I32(stride as i32),
            Arg::I32((PICKED * SLICE) as i32),
            Arg::I32(SLICE as i32),
        ],
    );
    plane::read_bf16(&y, RELAY_ROWS * SLICE + SLICE)
}
