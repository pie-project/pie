//! `Fallback::Copy`, on the metal plane, without a device: exercises the
//! class table (`Windows::of`) that decides whether a copy happens and over
//! which rows. Does not check that a copy's bytes equal a split's (needs a
//! device).

use engine_metal::window::{Copies, Windows};
use model_compiler::{Budget, CompiledModel, DeviceProfile, Fallback, compile};
use model_exec::fire::{ClassWindow, WindowTable};
use model_exec::store::kv::Geometry;
use model_ir::{ClassSet, Platform, Trace};

const SERVED: &str = "qwen35-d0.8b";

/// Tokens per page; small, so a lane's page span exceeds one entry.
const PAGE: i32 = 4;

/// Must straddle the crossover: `model_compiler::layout` writes `Copy` below
/// it and `Split` above.
const LATTICE: [u32; 5] = [16, 64, 256, 1024, 4096];

fn budget() -> Budget {
    Budget {
        max_lanes: 64,
        max_tokens: 4096,
        buckets: LATTICE.to_vec(),
        max_adapters: 8,
    }
}

fn profile() -> DeviceProfile {
    DeviceProfile {
        side_streams: 0,
        ..DeviceProfile::default()
    }
}

/// The served text, baked for this plane.
fn baked() -> (Trace, CompiledModel) {
    let trace = models::skus()
        .find(|row| row.name.starts_with(SERVED))
        .unwrap_or_else(|| panic!("the catalog no longer names a {SERVED} text"))
        .trace;
    let trace = trace(Platform::Metal);
    let compiled = compile(&trace, &budget(), &profile()).expect("the served text bakes");
    (trace, compiled)
}

/// One row and one lane per class, in the artifact's shipped order.
fn every_class_once(compiled: &CompiledModel) -> WindowTable {
    let classes = compiled.classes.classes.len();
    let order = compiled
        .order
        .class_order(&ClassSet::of(0..classes));
    let mut table = vec![ClassWindow::default(); classes];
    for (at, &class) in order.iter().enumerate() {
        table[class as usize] = ClassWindow {
            row_offset: at as u32,
            rows: 1,
            lane_offset: at as u32,
            lanes: 1,
        };
    }
    WindowTable::new(table)
}

/// This fire's qo boundaries: one row per lane, so the prefix sum counts.
fn indptr(lanes: usize) -> Vec<i32> {
    (0..=lanes as i32).collect()
}

/// Lanes own different numbers of pages, so a gathered lane's span of the
/// page-id list differs from what a slice would give.
fn geometry(lanes: usize) -> Geometry {
    let mut indptr = vec![0i32];
    let mut indices = Vec::new();
    let mut last_page_len = Vec::new();
    let mut kv_len = Vec::new();
    let mut page = 0i32;
    for lane in 0..lanes {
        // lane l holds l % 3 + 1 pages (one to three).
        let pages = (lane % 3 + 1) as i32;
        for _ in 0..pages {
            indices.push(page);
            page += 1;
        }
        indptr.push(indices.len() as i32);
        last_page_len.push(1 + (lane as i32 % PAGE));
        kv_len.push((pages - 1) * PAGE + 1 + (lane as i32 % PAGE));
    }
    Geometry {
        indptr,
        indices,
        last_page_len,
        kv_len,
        write_page: vec![0; lanes],
        write_offset: vec![0; lanes],
    }
}

/// `positions` is `100 + row` so a permuted vector can't be confused with an
/// identity one.
fn ambient(rows: usize) -> (Vec<i32>, Vec<i32>) {
    (
        (0..rows as i32).map(|row| 100 + row).collect(),
        (0..rows as i32).collect(),
    )
}

/// The windows this fire cuts, at one arm of the switch.
fn windows(trace: &Trace, compiled: &CompiledModel, enabled: bool) -> Windows {
    let classes = every_class_once(compiled);
    // no patch row on this artifact: every Window::patch is (0, 0).
    let no_patches = WindowTable::new(vec![
        ClassWindow::default();
        compiled.classes.classes.len()
    ]);
    let lanes = compiled.classes.classes.len();
    let (positions, request_of_token) = ambient(lanes);
    let spaces = [geometry(lanes)];
    Windows::of(
        trace,
        compiled,
        &classes,
        &no_patches,
        &indptr(lanes),
        Copies {
            // bucket 0 is 16 rows, below the crossover.
            bucket: 0,
            enabled,
            spaces: &spaces,
            positions: &positions,
            request_of_token: &request_of_token,
        },
        &[],
    &[],
    )
    .expect("a fire over every class is a fire the artifact promised")
}

/// Every region index with a `Fallback::Copy` row, and its mask.
fn withdrawn(compiled: &CompiledModel) -> Vec<(u32, ClassSet)> {
    let mut out: Vec<(u32, ClassSet)> = Vec::new();
    for (at, region) in compiled.template().iter().enumerate() {
        let owed = compiled.fallback.rows.iter().any(|row| {
            region.nodes.contains(&row.node) && row.fallback == Fallback::Copy
        });
        if owed {
            out.push((at as u32, region.mask.clone()));
        }
    }
    out
}

#[test]
fn the_bake_writes_a_copy_row_below_the_crossover() {
    let (_, compiled) = baked();
    let owed = withdrawn(&compiled);
    assert!(
        !owed.is_empty(),
        "the {SERVED} text no longer owes a `Fallback::Copy` row at any bucket \
         of {LATTICE:?}; every claim in this file is about a table that is now \
         empty"
    );
    // the other half is still written, above the crossover.
    assert!(
        compiled
            .fallback
            .rows
            .iter()
            .any(|row| matches!(row.fallback, Fallback::Split { .. })),
        "the lattice no longer straddles the crossover"
    );
}

#[test]
fn a_withdrawn_window_splits_when_the_shell_does_not_copy() {
    let (trace, compiled) = baked();
    let split = windows(&trace, &compiled, false);
    assert_eq!(split.copied(), 0, "copies are off and something gathered");
    let owed = withdrawn(&compiled);
    assert!(
        owed.iter().any(|(at, _)| split.runs(*at) > 1),
        "a fire carrying every class found no withdrawn window in pieces; the \
         shipped class order no longer breaks the mask P4 withdrew"
    );
}

#[test]
fn the_same_window_is_one_encode_when_it_does() {
    let (trace, compiled) = baked();
    let split = windows(&trace, &compiled, false);
    let copy = windows(&trace, &compiled, true);

    assert!(
        copy.copied() > 0,
        "the shell was told to copy, P4 wrote a `Copy` row at this bucket, and \
         nothing gathered — `window::copyable` declined every region over the \
         withdrawn mask"
    );
    assert!(
        copy.launches() < split.launches(),
        "a copy that costs as many encodes as the split it replaced ({} vs {}) \
         is not a copy",
        copy.launches(),
        split.launches()
    );

    // every region the copy gathered ran r > 1 times under the split, once now.
    let mut gathered = 0;
    for at in 0..compiled.template().len() as u32 {
        if copy.runs(at) == split.runs(at) {
            continue;
        }
        gathered += 1;
        assert_eq!(copy.runs(at), 1, "a gathered region costs one encode");
        assert!(split.runs(at) > 1, "and it cost more than one before");
    }
    assert_eq!(
        gathered,
        copy.copied(),
        "`Windows::copied` counts the regions whose window carries a row map, \
         and this counts the regions whose encode count moved; they are the \
         same regions"
    );
}

#[test]
fn the_gathered_window_names_the_rows_the_split_ran_over() {
    let (trace, compiled) = baked();
    let split = windows(&trace, &compiled, false);
    let copy = windows(&trace, &compiled, true);

    let at = (0..compiled.template().len() as u32)
        .find(|&at| copy.runs(at) == 1 && split.runs(at) > 1)
        .expect("some region is copied");

    // the rows the split's r encodes covered, in encode order.
    let mut expected: Vec<i32> = Vec::new();
    for run in 0..split.runs(at) {
        let span = split.at(at, run).span;
        expected.extend((span.row_offset..span.row_offset + span.rows).map(|row| row as i32));
    }

    let window = copy.at(at, 0);
    let gathered = window
        .gathered
        .as_ref()
        .expect("the copied region's window carries a row map");
    assert_eq!(
        gathered.rows_host, expected,
        "the gathered rectangle is the split's rows, in the split's order"
    );
    assert_eq!(
        window.span.row_offset, 0,
        "a gathered span is the compacted rectangle and starts at its own zero"
    );
    assert_eq!(window.span.rows, expected.len() as u32);
    assert_eq!(
        gathered.runs.len() as u32,
        split.runs(at),
        "and it remembers how many intervals it compacted"
    );
}

#[test]
fn the_boundaries_are_rebased_over_the_union_and_not_over_one_run() {
    let (trace, compiled) = baked();
    let copy = windows(&trace, &compiled, true);
    let split = windows(&trace, &compiled, false);
    let at = (0..compiled.template().len() as u32)
        .find(|&at| copy.runs(at) == 1 && split.runs(at) > 1)
        .expect("some region is copied");

    let window = copy.at(at, 0);
    assert_eq!(
        window.indptr_host.first().copied(),
        Some(0),
        "a window's boundaries start at its own zero"
    );
    assert_eq!(
        window.indptr_host.len() as u32,
        window.span.lanes + 1,
        "one boundary per gathered lane, plus the terminator"
    );
    assert_eq!(
        window.indptr_host.last().copied(),
        Some(window.span.rows as i32),
        "and they sum to the rows the one encode stands over — this fire gives \
         every lane one row, so the union's last boundary IS its row count"
    );
    assert!(
        window.indptr_host.windows(2).all(|pair| pair[1] > pair[0]),
        "every gathered lane carries rows, so the prefix sum is strictly \
         increasing; a vector that repeated a bound would be one run's \
         boundaries pasted onto another's"
    );
}

#[test]
fn the_ambient_row_tables_are_permuted_and_not_sliced() {
    let (trace, compiled) = baked();
    let copy = windows(&trace, &compiled, true);
    let split = windows(&trace, &compiled, false);
    let at = (0..compiled.template().len() as u32)
        .find(|&at| copy.runs(at) == 1 && split.runs(at) > 1)
        .expect("some region is copied");
    let lanes = compiled.classes.classes.len();
    let (positions, request_of_token) = ambient(lanes);

    let gathered = copy.at(at, 0).gathered.as_ref().expect("a row map");
    let want_positions: Vec<i32> = gathered
        .rows_host
        .iter()
        .map(|&row| positions[row as usize])
        .collect();
    let want_requests: Vec<i32> = gathered
        .rows_host
        .iter()
        .map(|&row| request_of_token[row as usize])
        .collect();

    assert_eq!(
        gathered.positions_host, want_positions,
        "the sdpa shaders read `position_ids[row]` by the LAUNCH's row, so a \
         gathered launch needs the positions of the rows it gathered"
    );
    assert_eq!(
        gathered.request_of_token_host, want_requests,
        "and the lane ids beside them, in the same order"
    );
    assert_ne!(
        gathered.positions_host,
        positions[..gathered.rows_host.len()].to_vec(),
        "a SLICE of the fire's table would be the first n rows in fire order — \
         plausible numbers, the wrong rows, and no fault anywhere"
    );
    assert!(
        gathered
            .request_of_token_host
            .iter()
            .all(|&lane| lane >= 0 && (lane as usize) < lanes),
        "the entries stay ABSOLUTE lane ids — a permutation moves rows and does \
         not renumber them, which is what keeps `Run::pool` fire-wide"
    );
}

#[test]
fn the_page_tables_are_re_cut_lane_by_lane_and_not_sliced() {
    let (trace, compiled) = baked();
    let copy = windows(&trace, &compiled, true);
    let split = windows(&trace, &compiled, false);
    let at = (0..compiled.template().len() as u32)
        .find(|&at| copy.runs(at) == 1 && split.runs(at) > 1)
        .expect("some region is copied");
    let lanes = compiled.classes.classes.len();
    let fire = geometry(lanes);

    let window = copy.at(at, 0);
    let gathered = window.gathered.as_ref().expect("a row map");
    assert_eq!(gathered.spaces.len(), 1, "one kv space was handed in");
    let space = &gathered.spaces[0];

    // this fire gives every class one lane at the same index as its row, so
    // the lane list is the row map.
    let lanes_of: Vec<usize> = gathered.rows_host.iter().map(|&row| row as usize).collect();

    assert_eq!(
        space.page_indptr_host.len(),
        lanes_of.len() + 1,
        "one bound per gathered lane, plus the terminator"
    );
    assert_eq!(space.page_indptr_host[0], 0, "a fresh prefix sum starts at 0");

    let mut expected: Vec<i32> = Vec::new();
    for &lane in &lanes_of {
        let start = fire.indptr[lane] as usize;
        let end = fire.indptr[lane + 1] as usize;
        expected.extend_from_slice(&fire.indices[start..end]);
    }
    assert_eq!(
        space.page_indices_host, expected,
        "the page-id LIST is compacted with the lanes: gathered lanes own spans \
         of it with other lanes' pages standing between them, and no bounds \
         vector over the whole list can name two such spans as requests 0 and 1"
    );
    assert_eq!(
        space.page_indptr_host.last().copied(),
        Some(space.page_indices_host.len() as i32),
        "and the bounds cut the compacted list, not the fire's"
    );

    assert_eq!(
        space.last_page_lens_host,
        lanes_of
            .iter()
            .map(|&lane| fire.last_page_len[lane])
            .collect::<Vec<_>>()
    );
    assert_eq!(
        space.kv_len_host,
        lanes_of
            .iter()
            .map(|&lane| fire.kv_len[lane])
            .collect::<Vec<_>>()
    );
}

#[test]
fn the_masked_window_is_refused_by_name_however_the_switch_is_set() {
    let (trace, compiled) = baked();
    let copy = windows(&trace, &compiled, true);
    let split = windows(&trace, &compiled, false);

    // regions owed a Copy row that still split with the switch on
    // (window::copyable declines RuntimeInput::Mask).
    let refused: Vec<u32> = withdrawn(&compiled)
        .into_iter()
        .map(|(at, _)| at)
        .filter(|&at| split.runs(at) > 1 && copy.runs(at) > 1)
        .collect();
    assert!(
        !refused.is_empty(),
        "every withdrawn window on this catalog was gathered; the masked \
         window names `RuntimeInput::Mask` and `window::copyable` must decline \
         it — a copy that permuted the activations and sliced the mask plane \
         would compute the wrong bytes with nothing to fault on"
    );
    for at in refused {
        assert_eq!(
            copy.runs(at),
            split.runs(at),
            "a region the copy path declined takes the split it always took"
        );
        assert!(
            copy.at(at, 0).gathered.is_none(),
            "and its window carries no row map, which is what `Serve::copies` \
             reads"
        );
    }
}

#[test]
fn only_a_withdrawn_mask_is_ever_in_pieces() {
    let (trace, compiled) = baked();
    let split = windows(&trace, &compiled, false);
    let copy = windows(&trace, &compiled, true);
    let masks: Vec<ClassSet> = withdrawn(&compiled).into_iter().map(|(_, m)| m).collect();
    for (at, region) in compiled.template().iter().enumerate() {
        let at = at as u32;
        if split.runs(at) == 1 {
            // a copy is for windows the order could not seat.
            assert_eq!(copy.runs(at), 1, "region {at} was seated and gathered anyway");
            assert!(
                copy.at(at, 0).gathered.is_none(),
                "region {at} was seated and carries a row map"
            );
            continue;
        }
        assert!(
            masks.contains(&region.mask),
            "region {at} came back in {} pieces and its mask is not one P4 \
             withdrew; `Fault::Fragmented` exists for exactly that and this \
             fire did not raise it",
            split.runs(at)
        );
    }
}

#[test]
fn the_builder_inherits_its_readers_answer() {
    let (trace, compiled) = baked();
    let split = windows(&trace, &compiled, false);
    let copy = windows(&trace, &compiled, true);

    // fallback::copies is keyed on the mask, not the region.
    let owed: Vec<u32> = withdrawn(&compiled).into_iter().map(|(at, _)| at).collect();
    let inherited: Vec<u32> = (0..compiled.template().len() as u32)
        .filter(|at| !owed.contains(at) && split.runs(*at) > 1)
        .collect();
    assert!(
        !inherited.is_empty(),
        "no region without a `Copy` row of its own came back in pieces; the \
         inheritance this test is about has nothing to inherit"
    );
    for at in inherited {
        let mask = &compiled.template()[at as usize].mask;
        let readers_gather = (0..compiled.template().len() as u32).any(|other| {
            &compiled.template()[other as usize].mask == mask && copy.runs(other) == 1
        });
        assert_eq!(
            copy.runs(at) == 1,
            readers_gather,
            "region {at} answers one thing and the regions over its mask \
             answer another; a copy is resolved per MASK and one region \
             disagreeing is the split-builder / gathered-reader bug"
        );
    }
}

#[test]
fn the_packed_blob_and_the_bind_walk_it_in_one_order() {
    let (trace, compiled) = baked();
    let copy = windows(&trace, &compiled, true);
    // what Inputs::write stages and Windows::bind cuts apart: one blob,
    // walked twice; lengths must agree.
    let mut want = 0usize;
    let mut seen: Vec<usize> = Vec::new();
    for at in 0..compiled.template().len() as u32 {
        for run in 0..copy.runs(at) {
            let window = copy.at(at, run);
            let here = window.indptr_host.as_ptr() as usize;
            if seen.contains(&here) {
                continue;
            }
            seen.push(here);
            want += window.indptr_host.len();
            let Some(gathered) = &window.gathered else {
                continue;
            };
            want += gathered.rows_host.len()
                + gathered.positions_host.len()
                + gathered.request_of_token_host.len();
            for space in &gathered.spaces {
                want += space.page_indptr_host.len()
                    + space.page_indices_host.len()
                    + space.last_page_lens_host.len()
                    + space.kv_len_host.len();
            }
        }
    }
    assert_eq!(
        copy.packed().len(),
        want,
        "the packed blob is every distinct window's vectors end to end, and a \
         gathered window contributes seven more than a plain one"
    );
}
