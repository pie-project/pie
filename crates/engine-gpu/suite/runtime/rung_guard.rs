//! A fire wider than the input store's regions is refused by name before a
//! byte of it is staged. The store lays the token ids, positions,
//! rotations, readout rows, windows, slot tables and page tables end to end
//! at the rows and lanes the budget states: a fire past them, staged, would
//! run its rows on into the next region (a kv write table into its offsets,
//! read as page ids), as Metal's M5 did at a fire of 297 rows over a rung of
//! 256.

use engine_gpu::error::Fault;
use engine_gpu::inputs::{Fire, Inputs};
use engine_gpu::store::kv::{Geometry, Paging};
use poem_compiler::Budget;

use crate::backend::{ALLOWANCE, Api, Context, DeviceBoot, Handles};

const ROWS: u32 = 8;
const LANES: u32 = 2;

/// Stages a fire of `rows` token rows over `lanes` lanes into a store laid
/// for `ROWS` and `LANES`, its every byte first set to a sentinel; what the
/// write answered, and whether the store still holds the sentinel whole.
fn stage(rows: usize, lanes: usize) -> Option<(Result<(), Fault>, bool)> {
    let device = Context::bind(&DeviceBoot::default(), ALLOWANCE).ok()?;
    let paging = Paging::of(16, 64, LANES, 8).expect("paging");
    let mut inputs = Inputs::<Api>::reserve(
        &device,
        &Budget::new(LANES, ROWS),
        paging,
        1,
        paging.pages_per_slot,
        1,
        0,
        None,
        false,
        &[],
        0,
        None,
        1,
        false,
    )
    .expect("the input store lays");
    let bytes = inputs.bytes();
    let sentinel = vec![0xa5u8; bytes as usize];
    inputs
        .write_input(0, &sentinel)
        .expect("the sentinel lands");

    let rows_of = vec![0i32; rows];
    let lanes_of = vec![0i32; lanes];
    let indptr: Vec<i32> = (0..=lanes as i32).collect();
    let geometry = Geometry {
        indptr: indptr.clone(),
        indices: lanes_of.clone(),
        last_page_len: vec![1; lanes],
        kv_len: vec![1; lanes],
        write_page: rows_of.clone(),
        write_offset: rows_of.clone(),
        ..Geometry::default()
    };
    let windows = [0i32, rows as i32];
    let fire = Fire {
        tokens: &rows_of,
        positions: &rows_of,
        rotations: &rows_of,
        readout_rows: &lanes_of[..lanes.min(rows)],
        windows: &windows,
        slot_ids: &lanes_of,
        slot_of_row: &rows_of,
        adapter_routes: None,
        request_of_token: &rows_of,
        page_tables: std::slice::from_ref(&geometry),
        mask: None,
        patches: None,
        mrope_positions: None,
        rs_replay: None,
        rs_commit: None,
        group_of_lane: &[],
        packings: &[],
        voxels: None,
        capture: &[],
    };
    let wrote = inputs.write(&Handles::new(), &fire).map(drop);
    let mut held = vec![0u8; bytes as usize];
    engine_gpu::backend::Buffer::read(inputs.store(), 0, &mut held).expect("the store reads back");
    Some((wrote, held == sentinel))
}

#[test]
fn a_fire_one_row_past_the_laid_rows_is_refused_and_stages_nothing() {
    let Some((wrote, untouched)) = stage(ROWS as usize + 1, LANES as usize) else {
        return;
    };
    match wrote {
        Err(Fault::Ceiling { what, need, have }) => {
            assert!(what.starts_with("token rows a fire stages"), "{what}");
            assert_eq!((need, have), (u64::from(ROWS) + 1, u64::from(ROWS)));
        }
        other => panic!("a fire past the laid rows was not refused by name: {other:?}"),
    }
    assert!(untouched, "the refused fire staged bytes into the store");
}

#[test]
fn a_fire_one_lane_past_the_laid_lanes_is_refused_and_stages_nothing() {
    let Some((wrote, untouched)) = stage(ROWS as usize, LANES as usize + 1) else {
        return;
    };
    match wrote {
        Err(Fault::Ceiling { what, need, have }) => {
            assert!(what.starts_with("lanes a fire stages"), "{what}");
            assert_eq!((need, have), (u64::from(LANES) + 1, u64::from(LANES)));
        }
        other => panic!("a fire past the laid lanes was not refused by name: {other:?}"),
    }
    assert!(untouched, "the refused fire staged bytes into the store");
}

#[test]
fn a_fire_at_the_laid_rows_and_lanes_is_staged() {
    let Some((wrote, untouched)) = stage(ROWS as usize, LANES as usize) else {
        return;
    };
    wrote.expect("a fire at the laid rows and lanes stages");
    assert!(!untouched, "the fire staged nothing");
}
