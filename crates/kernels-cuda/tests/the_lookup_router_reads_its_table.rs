//! **THE LOOKUP ROUTER, ON A REAL DEVICE**: `linear.moe_hash_route` —
//! `kernels/linear/moe_route.cuh`'s `hash_route_gather` — held against a host
//! recomputation of the table lookup it is.
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test the_lookup_router_reads_its_table
//! ```
//!
//! # What there is to get wrong
//!
//! This router computes nothing. Every number it lands is either a row of the
//! `[vocab, top_k]` table, narrowed from i64 to i32, or the constant
//! `1/top_k` — so a golden against a reference implementation would be a
//! golden against itself, and what the two tests below check is instead the
//! three ways the addressing can be wrong:
//!
//! ```text
//! (a) the row the id names: a table read at the wrong row is the whole bug
//!     this op exists to avoid, because a softmax gate substituted for it
//!     answers different experts and computes something plausible anyway.
//!     Checked at the table's FIRST row, at its LAST (a boundary id must
//!     read row `vocab - 1`, not off the end), at an id past the vocabulary
//!     and at a negative one (both fall to row 0, `layout.embed`'s rule),
//!     and at a row that names one expert twice (copied as it stands — the
//!     hash may repeat).
//! (b) the weight: uniform `1/top_k` in every slot, exactly, and the point
//!     of asserting it to the bit is that a fold behind an almost-uniform
//!     router still produces a number.
//! (c) the staged-geometry seat: this entry is on `engine_cuda::SHIFTED`,
//!     and it is the first router there whose GRID counts (token row, slot)
//!     pairs rather than rows. Armed at `(count 3, start 2)` over a six-row
//!     plane handed at its own base, it must gather for plane rows 2, 3 and
//!     4 — the ids at those rows, the routes at those rows — and leave the
//!     other three rows as the arena left them. A guard-only reading passes
//!     (b) and fails this, which is why the fill outside the window is
//!     asserted as loudly as the contents inside it.
//! ```

#![cfg(feature = "_cuda")]

mod common;

use common::Gpu;
use dtype::Dtype;
use kernels_cuda::linear::moe_route::hash_route;
use kernels_cuda::tensor::Tensor;

const VOCAB: u32 = 12;
const TOP_K: u32 = 3;

/// Ids and weights the gather never lands, so a slot it did not write is
/// visible as one.
const ROUTE_FILL: i32 = -12_345;
const WEIGHT_FILL: f32 = -12_345.0;

/// The `[VOCAB, TOP_K]` table, and the one place its contents are stated: the
/// device reads this and the host expectation recomputes from it.
///
/// Row 2 names the same expert in slots 0 and 1 on purpose — a hash may
/// repeat, and a uniform fold weights the repeat like anything else.
fn table() -> Vec<i64> {
    let mut t = vec![0i64; (VOCAB * TOP_K) as usize];
    for v in 0..VOCAB as usize {
        for s in 0..TOP_K as usize {
            t[v * TOP_K as usize + s] = ((v * 7 + s * 5) % 9) as i64;
        }
    }
    t[2 * TOP_K as usize + 1] = t[2 * TOP_K as usize];
    t
}

/// The row of the table a token id selects: itself when the id is one the
/// table has, and row 0 otherwise — the rule `layout.embed`'s gather states.
fn row_of(id: i32) -> usize {
    if id >= 0 && id < VOCAB as i32 {
        id as usize
    } else {
        0
    }
}

/// The ids the first test hands over: the table's first row, an interior one,
/// the LAST one, one past the end, a negative one, and the row that repeats.
fn ids() -> Vec<i32> {
    vec![0, 5, VOCAB as i32 - 1, VOCAB as i32, -1, 2]
}

#[test]
fn the_gather_lands_the_table_row_the_id_names() {
    let ids = ids();
    let table = table();
    let rows = ids.len() as u32;

    let mut gpu = Gpu::open();
    let ids_at = gpu.up(&ids);
    let table_at = gpu.up(&table);
    let routes_at = gpu.up(&vec![ROUTE_FILL; (rows * TOP_K) as usize]);
    let weights_at = gpu.up(&vec![WEIGHT_FILL; (rows * TOP_K) as usize]);

    let mut routes = Tensor::new(routes_at, rows, TOP_K, Dtype::I32);
    let mut weights = Tensor::new(weights_at, rows, TOP_K, Dtype::F32);
    hash_route(
        &gpu.ctx(),
        Tensor::new(ids_at, rows, 1, Dtype::I32),
        Tensor::new(table_at, VOCAB, TOP_K, Dtype::I64),
        VOCAB,
        TOP_K,
        &mut routes,
        &mut weights,
    )
    .expect("the lookup router fires");
    gpu.sync();

    let got_routes: Vec<i32> = gpu.down(routes_at, (rows * TOP_K) as usize);
    let got_weights: Vec<f32> = gpu.down(weights_at, (rows * TOP_K) as usize);
    let uniform = 1.0f32 / TOP_K as f32;
    for (r, id) in ids.iter().enumerate() {
        for s in 0..TOP_K as usize {
            let at = r * TOP_K as usize + s;
            let want = table[row_of(*id) * TOP_K as usize + s] as i32;
            assert_eq!(
                got_routes[at], want,
                "row {r} (id {id}) slot {s}: the gather read another table row"
            );
            assert_eq!(
                got_weights[at].to_bits(),
                uniform.to_bits(),
                "row {r} slot {s} is not the uniform {uniform}"
            );
        }
    }
}

#[test]
fn the_armed_seat_gathers_the_window_the_pair_names() {
    const START: u32 = 2;
    const LIVE: u32 = 3;

    let ids = ids();
    let table = table();
    let rows = ids.len() as u32;
    assert!(
        START + LIVE < rows,
        "the window must leave rows on both sides"
    );

    let mut gpu = Gpu::open();
    let ids_at = gpu.up(&ids);
    let table_at = gpu.up(&table);
    let routes_at = gpu.up(&vec![ROUTE_FILL; (rows * TOP_K) as usize]);
    let weights_at = gpu.up(&vec![WEIGHT_FILL; (rows * TOP_K) as usize]);
    // The seat is four words — `[rows, row_offset, lanes, lane_offset]` — and
    // this row-gridded entry reads the first two. The lane pair stays zero
    // and is never consulted; the buffer is four wide anyway, because the
    // engine's seat is.
    let staged_at = gpu.up(&[LIVE, START, 0, 0]);

    let ctx = gpu.ctx();
    let mut routes = Tensor::new(routes_at, rows, TOP_K, Dtype::I32);
    let mut weights = Tensor::new(weights_at, rows, TOP_K, Dtype::F32);
    ctx.arm_stage(staged_at);
    hash_route(
        &ctx,
        Tensor::new(ids_at, rows, 1, Dtype::I32),
        Tensor::new(table_at, VOCAB, TOP_K, Dtype::I64),
        VOCAB,
        TOP_K,
        &mut routes,
        &mut weights,
    )
    .expect("the armed lookup router fires");
    ctx.disarm_stage();
    gpu.sync();

    let got_routes: Vec<i32> = gpu.down(routes_at, (rows * TOP_K) as usize);
    let got_weights: Vec<f32> = gpu.down(weights_at, (rows * TOP_K) as usize);
    let uniform = 1.0f32 / TOP_K as f32;
    for r in 0..rows as usize {
        let inside = r >= START as usize && r < (START + LIVE) as usize;
        for s in 0..TOP_K as usize {
            let at = r * TOP_K as usize + s;
            if inside {
                // The ordinal contract: the block that owns launch row
                // `r - START` gathered by `ids[r]` and wrote `routes[r]`.
                let want = table[row_of(ids[r]) * TOP_K as usize + s] as i32;
                assert_eq!(
                    got_routes[at], want,
                    "row {r} slot {s}: the armed gather read the id of another row"
                );
                assert_eq!(
                    got_weights[at].to_bits(),
                    uniform.to_bits(),
                    "row {r} slot {s} is inside the window and is not the uniform weight"
                );
            } else {
                assert_eq!(
                    got_routes[at], ROUTE_FILL,
                    "row {r} slot {s} is outside the window and was written"
                );
                assert_eq!(
                    got_weights[at].to_bits(),
                    WEIGHT_FILL.to_bits(),
                    "row {r} slot {s} is outside the window and was written"
                );
            }
        }
    }
}
