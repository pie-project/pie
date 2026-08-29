//! The shell's window table, over a fire P4 could not seat whole.
//!
//! **WHAT THIS PINS.** `Windows::of` used to answer `Fault::Fragmented` for
//! any region whose class set this fire's order left in pieces, on the
//! (false) premise that the catalog bakes an empty `FallbackTable`. It bakes
//! 12 rows for qwen35-d0.8b at an adapter capacity the text can actually seat,
//! and 132 of that SKU's 255 compositions fragment something. So the table now
//! cuts one window per interval — `Fallback::Split { r }`, design §3 — and
//! `Fault::Fragmented` is kept for the case it always named: a window P4
//! PROMISED consecutive that a fire found broken.
//!
//! Three claims, and each is a way the split can be silently wrong:
//!
//! - **every run gets a window, and they are different windows.** A table that
//!   handed run 1 run 0's span would compute the first interval twice;
//! - **every run's qo boundaries are rebased to its own zero.** They are
//!   offsets into the rectangle the run cuts, and a sub-rectangle starts at 0.
//!   Sharing run 0's vector is the failure that does not fault — the schedule
//!   walks work items against a boundary list describing other lanes;
//! - **the runs partition the mask's rows and lanes exactly**, checked against
//!   the composition's own lane placement.
//!
//! NO DEVICE. `Windows::of` is arithmetic over a class table and a boundary
//! vector; nothing here opens a context, allocates, or launches.

use engine::fire::{Lane, MaskSpan, WindowTable, compose, fallback};
use engine_cuda::window::Windows;
use model_compiler::{CompiledModel, Budget, DeviceProfile, compile};
use model_dsl::Platform;
use model_ir::Trace;

/// The SKU whose `captures_scores` window P4 withdraws, and the one the
/// reproduction numbers in this file's doc were measured on.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// A deployment's ceilings at an adapter capacity the catalog can seat. At the
/// 32 the other catalog sweeps ask for, `compile` refuses every SKU on adapter
/// capacity and this file would assert nothing.
fn budget() -> Budget {
    Budget {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: 8,
    }
}

fn sku() -> (Trace, CompiledModel) {
    let (_, _, trace, _) = model::catalog()
        .into_iter()
        .find(|(sku, ..)| *sku == SKU)
        .unwrap_or_else(|| panic!("`{SKU}` is in the catalog"));
    let trace = trace(Platform::Cuda);
    let compiled = compile(&trace, &budget(), &DeviceProfile::default())
        .unwrap_or_else(|refusal| panic!("`{SKU}` bakes: {refusal:?}"));
    assert!(
        !compiled.fallback.rows.is_empty(),
        "`{SKU}` owes fallback rows — that is the premise of this file",
    );
    (trace, compiled)
}

/// The fire's qo boundaries: one entry per lane plus the closing bound, which
/// is what the shell stages and what `Windows::of` rebases per window.
fn indptr(rows: &[u32]) -> Vec<i32> {
    let mut out = vec![0i32];
    for &n in rows {
        out.push(out[out.len() - 1] + n as i32);
    }
    out
}

#[test]
fn every_run_of_a_split_window_gets_its_own_span_and_its_own_boundaries() {
    let (plan, compiled) = sku();

    // Classes 0, 4 and 5 at once — the smallest composition that leaves the
    // score-capture window in pieces, with class 0's rows standing between
    // class 4's and class 5's.
    let lanes: Vec<Lane> = [(0usize, 3u32), (4, 5), (5, 2)]
        .iter()
        .map(|&(class, rows)| Lane::new(compiled.classes.classes[class].word(), rows))
        .collect();
    let fire = compose(&compiled, &budget(), &lanes).expect("three lanes compose");
    assert_eq!(fire.present(), [4, 0, 5]);

    // The boundaries, in FIRE order: class 4's lane, then class 0's, then
    // class 5's.
    let boundaries = indptr(&[5, 3, 2]);
    // Copies OFF: this file is the SPLIT's gate, and the copy the same
    // table asks for below the crossover is gated in
    // `a_copied_window_is_one_launch_over_the_same_rows.rs`.
    let windows = Windows::of(
        &plan,
        &compiled,
        fire.classes(),
        &boundaries,
        engine_cuda::window::Copies::off(),
    )
    .expect("a fragmented window is a slow path, not a fault");

    let mut split = 0usize;
    for (at, region) in compiled.template().iter().enumerate() {
        let at = at as u32;
        let spans = fire.classes().spans(&region.mask);
        assert_eq!(
            windows.runs(at),
            spans.len().max(1) as u32,
            "region {at} was cut into a different number of runs than the walk will loop over",
        );
        if spans.len() < 2 {
            continue;
        }
        split += 1;

        // Different windows, in ascending row order, and each with its own
        // boundaries rebased to its own zero.
        let mut seen: Vec<MaskSpan> = Vec::new();
        for run in 0..windows.runs(at) {
            let window = windows.at(at, run);
            assert_eq!(window.span, spans[run as usize], "region {at} run {run}");
            assert!(
                !seen.contains(&window.span),
                "region {at} handed run {run} a window it already handed another",
            );
            seen.push(window.span);

            let first = window.span.lane_offset as usize;
            let last = first + window.span.lanes as usize;
            let want: Vec<i32> = boundaries[first..=last]
                .iter()
                .map(|bound| bound - boundaries[first])
                .collect();
            assert_eq!(
                window.indptr_host, want,
                "region {at} run {run} does not rebase to its own zero",
            );
            assert_eq!(window.indptr_host[0], 0);
            assert_eq!(
                *window.indptr_host.last().expect("a closing bound"),
                window.span.rows as i32,
                "a run's boundaries close at its own row count",
            );
        }

        // And the runs partition the mask, rows and lanes both.
        assert_eq!(
            seen.iter().map(|span| span.rows).sum::<u32>(),
            fire.classes().rows_of(&region.mask),
        );
        assert_eq!(
            seen.iter().map(|span| span.lanes).sum::<u32>(),
            fire.classes().lanes_of(&region.mask),
        );
    }
    assert!(
        split > 0,
        "classes 0, 4 and 5 at once leave the score-capture window in pieces",
    );
}

/// A window P4 promised whole and a fire found broken is still refused by
/// name — the reading `Fault::Fragmented` keeps, and the reason the variant
/// did not go away with the hard fault it used to raise.
#[test]
fn a_window_p4_promised_whole_is_still_a_bake_integrity_refusal() {
    let (plan, compiled) = sku();

    // A class table the shell was NOT handed by `compose`: every class one
    // row and one lane, in ASCENDING class order rather than the baked one.
    // That is what a `CompiledModel` and a `WindowTable` built from different things
    // look like, and it is exactly what the variant is for.
    let count = compiled.classes.classes.len();
    let ascending = WindowTable::new(
        (0..count)
            .map(|at| engine::fire::ClassWindow {
                row_offset: at as u32,
                rows: 1,
                lane_offset: at as u32,
                lanes: 1,
            })
            .collect(),
    );

    // A capture region P4 seated — it owes no row, so its mask is an interval
    // of the shipped order and of every sub-order of it — that this table
    // nevertheless breaks. `{1,3,5,7}` is one: the qo_one classes, contiguous
    // in the frontier P4 found and every other class apart in `0..8`.
    let seated = compiled
        .template()
        .iter()
        .find(|region| {
            fallback::promised(&compiled, region) && ascending.span(&region.mask).is_err()
        })
        .expect("some seated window is not an interval of the ascending order");
    assert_eq!(fallback::bound(&compiled, &seated.mask), 1);

    let refusal = Windows::of(
        &plan,
        &compiled,
        &ascending,
        &indptr(&vec![1; count]),
        engine_cuda::window::Copies::off(),
    )
    .expect_err("a promise P4 made and this table broke");
    let said = refusal.to_string();
    assert!(said.contains("no fallback row"), "{said}");
}
