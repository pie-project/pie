//! **`Fallback::Copy`, on the metal plane, without a device.**
//!
//! Every claim below is arithmetic over a real baked artifact and a fire's
//! class table, which is exactly what `engine_metal::window::Windows::of` is —
//! so the whole file runs on any target and needs no Metal, no checkpoint and
//! no weights. The device half (a copy's bytes equal a split's bytes) is a
//! separate gate and is named in the verify queue; what is settled here is the
//! half that decides WHETHER a copy happens and over WHICH rows, and that half
//! has no bytes in it.
//!
//! # The fire these tests build
//!
//! A fire whose lanes cover EVERY class the artifact declares, one lane and
//! one row each, laid down in the order the artifact ships
//! (`ClassOrder::class_order`). That is not a contrived table: it is precisely
//! what `model_exec::fire::compose` produces for a submission that happens to
//! carry one lane of each behavior, and it is the fire in which P4's
//! withdrawal is visible — a mask whose classes the shipped order could not
//! keep adjacent comes back in several runs.
//!
//! ```text
//! shipped order:   [ 4 | 8 | 0 | 2 | 6 | 10 | 7 | 11 | 3 | 1 | 5 | 9 ]
//! mask {4,5,6,7}:    ^           ^        ^                     ^
//!                  run          run      run                   run
//! ```
//!
//! # VERIFY QUEUE — what this file does NOT settle
//!
//! **The numeric identity: a copy's bytes are a split's bytes.** That is the
//! claim `Fallback::Copy` rests on and it needs a device, a checkpoint and a
//! fire that actually reaches a gathered `attention.prefill_lse` window — one
//! shell, one set of addresses, `Shell::set_copies` flipped between two fires,
//! diffed row for row. The CUDA plane banks it as
//! `engine-cuda/tests/a_copied_window_and_a_split_one_are_the_same_bytes.rs`;
//! this plane cannot reach the window cheaply from any fixture in the tree,
//! for two reasons that are both about the BAKE and neither about the copy:
//!
//!   * `engine_metal::api`'s `bake_budgets` passes the deployment's lattice
//!     through and the deployment states none, so a smoke load has ONE
//!     implicit bucket at `max_tokens` and P4 writes `Split` at it. A device
//!     gate has to state rungs below the crossover before there is a `Copy`
//!     row to serve.
//!   * the withdrawn mask is `captures_scores`, so the fire has to carry a
//!     lane whose word sets that bit AND a lane of some class the shipped
//!     order puts between two of the mask's — which is a submission shape no
//!     existing metal fixture builds.
//!
//! Until that gate is written, this shell ships with `Shell::set_copies`
//! defaulting to OFF and the split — always correct, and the oracle — is what
//! every load runs.

use engine_metal::window::{Copies, Windows};
use model_compiler::{Budget, CompiledModel, DeviceProfile, Fallback, compile};
use model_exec::fire::{ClassWindow, WindowTable};
use model_exec::store::kv::Geometry;
use model_ir::{ClassSet, Platform, Trace};

/// The SKU every other gate in this crate is written over, and the one whose
/// texts P4 actually withdraws a window from.
const SERVED: &str = "qwen35-d0.8b";

/// Tokens per page in the synthetic geometry — small, so that a lane's page
/// span is more than one entry and a re-cut page-id list has something to be
/// wrong about.
const PAGE: i32 = 4;

/// The bucket lattice these tests bake against.
///
/// **IT HAS TO STRADDLE THE CROSSOVER, AND THAT IS THE WHOLE POINT OF STATING
/// ONE.** `model_compiler::layout` writes `Fallback::Copy` below
/// `CROSSOVER_ROWS` (scaled by the profile's SM count) and `Fallback::Split`
/// above it, so an artifact baked with NO lattice has one implicit bucket at
/// `max_tokens` and the menu writes `Split` at the only point there is. That
/// is the shape `engine_metal::api`'s `bake_budgets` produces for a deployment
/// that states nothing — which is why `Shell::set_copies(true)` on such a load
/// changes nothing, and why a deployment that wants this path states rungs
/// below the crossover.
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
    let (_, _, trace, _) = models::catalog()
        .into_iter()
        .find(|(sku, ..)| sku.starts_with(SERVED))
        .unwrap_or_else(|| panic!("the catalog no longer names a {SERVED} text"));
    let trace = trace(Platform::Metal);
    let compiled = compile(&trace, &budget(), &profile()).expect("the served text bakes");
    (trace, compiled)
}

/// One row and one lane per class, in the order the artifact ships.
///
/// The table is indexed by CLASS and the fire's order is where each class's
/// interval STARTS, so this is the shipped order turned into offsets — the
/// same thing `compose` does once it has counted a submission's rows.
fn every_class_once(compiled: &CompiledModel) -> WindowTable {
    let classes = compiled.classes.classes.len();
    let order = compiled
        .order
        .class_order(&ClassSet::of(0..classes), None);
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

/// A synthetic kv geometry whose lanes own DIFFERENT numbers of pages, so
/// that a gathered lane's span of the page-id list is not the span a slice
/// would have handed it.
fn geometry(lanes: usize) -> Geometry {
    let mut indptr = vec![0i32];
    let mut indices = Vec::new();
    let mut last_page_len = Vec::new();
    let mut kv_len = Vec::new();
    let mut page = 0i32;
    for lane in 0..lanes {
        // Lane `l` holds `l % 3 + 1` pages — one, two or three.
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

/// This fire's ambient row tables. `positions` is stated as `100 + row` so a
/// permuted vector cannot be confused with an identity one; `request_of_token`
/// is the lane that owns the row, which for this fire is the row itself.
fn ambient(rows: usize) -> (Vec<i32>, Vec<i32>) {
    (
        (0..rows as i32).map(|row| 100 + row).collect(),
        (0..rows as i32).collect(),
    )
}

/// The windows this fire cuts, at one arm of the switch.
fn windows(trace: &Trace, compiled: &CompiledModel, enabled: bool) -> Windows {
    let classes = every_class_once(compiled);
    // **THE SECOND SERIATION'S TABLE, AND IT IS THE EMPTY ONE.** This
    // artifact states no patch row, so its patch table is one all-zero window
    // per class — which is what a text-only fire's composition answers and
    // what makes every `Window::patch` here `(0, 0)`.
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
            // Bucket 0 is 16 rows, which is below the crossover on every
            // profile this file states — see [`LATTICE`].
            bucket: 0,
            enabled,
            spaces: &spaces,
            positions: &positions,
            request_of_token: &request_of_token,
        },
    )
    .expect("a fire over every class is a fire the artifact promised")
}

/// Every region index whose mask P4 wrote a `Fallback::Copy` row for, and the
/// mask itself.
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
    // The other half of the menu is still written, above the crossover — a
    // lattice that got `Copy` at every point would mean the crossover moved
    // off the end of it and these tests would be measuring one arm.
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

    // Every region the copy gathered ran `r > 1` times under the split and
    // runs ONCE now. That is the whole claim `model_exec::fire::walk` reads
    // off this table.
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

    // The rows the split's `r` encodes covered, in encode order.
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

    // The gathered LANES, in gathered order — this fire gives every class one
    // lane at the same index as its row, so the lane list is the row map.
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

    // Every region P4 owes a `Copy` row for, that this fire found in pieces,
    // and that STILL splits with the switch on. There is at least one on this
    // catalog — the masked window — and `window::copyable` is why: its nodes
    // name `RuntimeInput::Mask`, a plane of (query, key) BYTES at a stated
    // stride, and permuting it is a second gather this plane has not written.
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
            // P4 promised this window one interval and a fire over every
            // class is where that promise is hardest. Neither arm may touch
            // it: a copy is for windows the order could not seat.
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

    // **THE PREPARE REGION OWES NO ROW AND IS COPIED ANYWAY.**
    // `model_compiler::layout` offers only capture regions to its C1P
    // instance, so the region that carries an attention plan is never
    // withdrawn by name — yet its window fragments in exactly the fires its
    // readers' does, and a builder that split while its reader gathered would
    // carry ONE set of tables where `r` were read.
    // `model_exec::fire::fallback::copies` is keyed on the MASK for that
    // reason, and this is the reason cashed.
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
    // What `Inputs::write` stages, and what `Windows::bind` cuts apart: one
    // blob, walked twice. The two functions cannot be checked against each
    // other without a handle table, but their LENGTHS can, and a length that
    // did not account for a gathered window's tables is exactly the drift
    // that would point one window's view at another window's vector.
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
