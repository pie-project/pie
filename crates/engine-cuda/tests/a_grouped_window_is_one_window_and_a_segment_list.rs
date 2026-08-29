//! The shell's window table, over a fire P4 answered `Fallback::Grouped` for.
//!
//! **WHAT THIS PINS.** `Fallback::Split { r }` is `r` windows and `r`
//! launches, each over its own rectangle; `Fallback::Grouped` is ONE window
//! over the UNION of those rectangles plus the list of which of its rows are
//! actually the consumer's. The union contains foreign rows — the classes P4
//! could not keep out of the gaps — so every part of the arithmetic has to be
//! right or the launch corrects somebody else's tokens:
//!
//! - **one window, not `r`.** `Windows::runs` must answer `1`, because
//!   `engine::fire::walk` turns its launch loop once for this region and a
//!   table that cut `r` would leave runs 1..r unreachable and their rows
//!   uncorrected;
//! - **the union is exactly the union.** First row of the first interval to
//!   last row of the last, rows and lanes both;
//! - **the segments are the split arm's own spans, rebased.** Compared against
//!   the OTHER bake's window table rather than against the arithmetic that
//!   produced them, which is what makes this a cross-check and not a
//!   restatement;
//! - **the segments partition the mask's rows**, exactly as the split's
//!   windows do — no row twice, no row missing, and nothing from the gaps;
//! - **they are staged beside the boundaries, in the one copy.**
//!   `Windows::packed` and `Windows::bind` are two statements of one layout
//!   and a disagreement between them is a launch reading a qo vector as a
//!   segment list.
//!
//! NO DEVICE. `Windows::of` is arithmetic over a class table and a boundary
//! vector; nothing here opens a context, allocates, or launches.

use engine::fire::{Lane, MaskSpan, compose, fallback};
use engine_cuda::window::{Copies, Windows};
use model_compiler::{CompiledModel, Budget, DeviceProfile, Fallback, FamilyCosts, PqTree, compile};
use model_dsl::Platform;
use model_ir::{ClassSet, Operands, Trace};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The op both profile lists are keyed on.
const CORRECTION: &str = "linear.lora_correct";

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

/// The two arms: the same withdrawal, two answers. `grouped` is the PoC
/// scaffold that makes the correction the withdrawn consumer at all (see
/// `DeviceProfile::grouped` — it is not the cost model); `grouped` is
/// the shell's real statement about its own kernel table, and it is the only
/// thing that differs between these two.
/// The two arms: the SAME withdrawal, two answers.
///
/// The withdrawal is chosen by cost (`model_compiler::layout::choose`), and
/// naming an op groupable is itself a discount on withdrawing it — so the
/// grouped arm withdraws the correction because it is groupable, and the split
/// arm is told the correction is cheap instead. Same mask, same row order, and
/// the only thing left differing is the ANSWER, which is what this file prices.
fn arms() -> (DeviceProfile, DeviceProfile) {
    let split = DeviceProfile {
        family_us: FamilyCosts {
            linear: 1.0,
            ..DeviceProfile::default().family_us
        },
        ..DeviceProfile::default()
    };
    let grouped = DeviceProfile {
        grouped: vec![CORRECTION.to_string()],
        ..DeviceProfile::default()
    };
    (split, grouped)
}

fn sku() -> (Trace, CompiledModel, CompiledModel) {
    let (_, _, trace, _) = model::catalog()
        .into_iter()
        .find(|(sku, ..)| *sku == SKU)
        .unwrap_or_else(|| panic!("`{SKU}` is in the catalog"));
    let trace = trace(Platform::Cuda);
    let (split, grouped) = arms();
    let split = compile(&trace, &budget(), &split).expect("the split arm bakes");
    let grouped = compile(&trace, &budget(), &grouped).expect("the grouped arm bakes");
    (trace, split, grouped)
}

/// Which regions hold a correction, and the mask they state.
fn correction_regions(trace: &Trace, compiled: &CompiledModel) -> Vec<u32> {
    compiled
        .template()
        .iter()
        .enumerate()
        .filter(|(_, region)| {
            region.nodes.clone().any(|node| {
                trace.nodes
                    .get(node as usize)
                    .is_some_and(|node| node.op.name() == CORRECTION)
            })
        })
        .map(|(at, _)| at as u32)
        .collect()
}

/// The fire's qo boundaries: one entry per lane plus the closing bound.
fn indptr(rows: &[u32]) -> Vec<i32> {
    let mut out = vec![0i32];
    for &n in rows {
        out.push(out[out.len() - 1] + n as i32);
    }
    out
}

/// One lane per class, prefill lanes three rows and decode lanes one — the
/// composition that presents every behaviour at once, and therefore the one
/// that fragments the adapter window the most.
fn one_lane_per_class(compiled: &CompiledModel) -> Vec<Lane> {
    compiled
        .classes
        .classes
        .iter()
        .map(|class| Lane::new(class.word(), if class.word() & 1 == 1 { 1 } else { 3 }))
        .collect()
}

#[test]
fn the_grouped_region_gets_one_window_whose_segments_are_the_splits_own_spans() {
    let (plan, split, grouped) = sku();

    // The premise, checked rather than assumed: the two bakes are one row
    // order, and they differ only in the answer the correction gets.
    let mut every = ClassSet::default();
    for class in 0..split.classes.classes.len() {
        every.insert(class);
    }
    assert_eq!(
        split.order.class_order(&every, None),
        grouped.order.class_order(&every, None),
        "the arms must be the same artifact with a different answer on it",
    );
    let regions = correction_regions(&plan, &split);
    assert!(
        !regions.is_empty(),
        "`{SKU}` states no correction region, and then this file is vacuous",
    );
    assert_eq!(regions, correction_regions(&plan, &grouped));

    let lanes = one_lane_per_class(&split);
    let fire = compose(&split, &budget(), &lanes).expect("eight lanes compose");
    let rows: Vec<u32> = fire.lanes().iter().map(|lane| lane.rows).collect();
    let boundaries = indptr(&rows);

    let split_windows = Windows::of(&plan, &split, fire.classes(), &boundaries, Copies::off())
        .expect("a fragmented window is a slow path, not a fault");
    let grouped_windows = Windows::of(&plan, &grouped, fire.classes(), &boundaries, Copies::off())
        .expect("a grouped window is one window");

    let mut checked = 0usize;
    for &at in &regions {
        let mask = &split.template()[at as usize].mask;
        let spans = fire.classes().spans(mask);
        assert!(
            spans.len() > 1,
            "region {at}'s window is one interval in this fire, so there is nothing \
             for `Grouped` to do and nothing here to check",
        );
        checked += 1;

        // ONE WINDOW, NOT `r` — the number `engine::fire::walk` loops on.
        assert_eq!(split_windows.runs(at), spans.len() as u32);
        assert_eq!(grouped_windows.runs(at), 1, "region {at}");

        // The union, rows and lanes both.
        let held = grouped_windows.at(at, 0);
        let first = spans[0];
        let last = *spans.last().expect("more than one span");
        assert_eq!(
            held.span,
            MaskSpan {
                row_offset: first.row_offset,
                rows: last.row_offset + last.rows - first.row_offset,
                lane_offset: first.lane_offset,
                lanes: last.lane_offset + last.lanes - first.lane_offset,
            },
            "region {at}'s grouped window is not the union of its intervals",
        );

        // The segments ARE the split arm's windows, rebased to the union —
        // read off the other bake's table, not recomputed here.
        assert_eq!(held.segs() as usize, spans.len(), "region {at}");
        let mut want: Vec<i32> = Vec::new();
        for run in 0..split_windows.runs(at) {
            let cut = split_windows.at(at, run).span;
            want.push((cut.row_offset - held.span.row_offset) as i32);
            want.push(cut.rows as i32);
        }
        assert_eq!(held.segments_host, want, "region {at}");

        // And they partition the mask's rows: every row the mask admits, once,
        // and nothing from the gaps.
        let mut covered: Vec<u32> = held
            .segments_host
            .chunks_exact(2)
            .flat_map(|pair| {
                let base = held.span.row_offset + pair[0] as u32;
                base..base + pair[1] as u32
            })
            .collect();
        covered.sort_unstable();
        let mut owed: Vec<u32> = fire
            .lanes()
            .iter()
            .filter(|lane| mask.contains(lane.class as usize))
            .flat_map(|lane| lane.row_offset..lane.row_offset + lane.rows)
            .collect();
        owed.sort_unstable();
        assert_eq!(covered, owed, "region {at}'s segments are not its rows");

        // The grid's segment axis is the ARTIFACT's bound and not this fire's
        // count, so a capture recorded at one composition is not sized at
        // another's (decision #15).
        assert_eq!(held.segment_cap, fallback::max_runs(&grouped));
        assert!(held.segment_cap >= held.segs());
        assert_eq!(
            held.segment_rows(),
            spans.iter().map(|span| span.rows).max().unwrap_or(0),
        );
    }
    assert!(checked > 0, "no correction region fragments in this fire");

    // The whole point of the answer: fewer launches for the same rows.
    let split_launches: u32 = regions.iter().map(|&at| split_windows.runs(at)).sum();
    let grouped_launches: u32 = regions.iter().map(|&at| grouped_windows.runs(at)).sum();
    assert!(grouped_launches < split_launches);
    eprintln!(
        "{SKU}, eight classes: {checked} correction regions, {split_launches} \
         windows split -> {grouped_launches} grouped",
    );
}

/// **THE SEGMENTS RIDE THE BOUNDARIES' STAGING PATH**, and the two statements
/// of that layout — `packed` writes it, `bind` reads it — agree.
///
/// A disagreement here is not a fault: it is a launch handed a pointer into
/// the middle of somebody's qo vector and told it is a segment list, which
/// corrects whichever rows those integers happen to name.
#[test]
fn the_segment_lists_are_staged_beside_the_boundaries_in_the_one_copy() {
    let (plan, _, grouped) = sku();
    let lanes = one_lane_per_class(&grouped);
    let fire = compose(&grouped, &budget(), &lanes).expect("eight lanes compose");
    let rows: Vec<u32> = fire.lanes().iter().map(|lane| lane.rows).collect();
    let mut windows = Windows::of(&plan, &grouped, fire.classes(), &indptr(&rows), Copies::off()).expect("the windows");

    let packed = windows.packed();
    // A base that is not zero, so an implementation that forgot to add it
    // shows up as a wrong pointer rather than as a coincidence.
    const BASE: u64 = 0x1000;
    windows.bind(BASE);

    let mut segmented = 0usize;
    for at in 0..grouped.template().len() as u32 {
        for run in 0..windows.runs(at) {
            let held = windows.at(at, run);

            // The boundaries land where `packed` put them.
            let start = ((held.indptr.ptr - BASE) / 4) as usize;
            assert_eq!(
                &packed[start..start + held.indptr_host.len()],
                held.indptr_host.as_slice(),
                "region {at} run {run}: the staged boundaries are not this window's",
            );
            assert_eq!(held.indptr.rows as usize, held.indptr_host.len());

            if held.segs() == 0 {
                assert_eq!(held.segments.rows, 0);
                continue;
            }
            segmented += 1;

            // The segments land immediately after them, in the same buffer.
            let after = start + held.indptr_host.len();
            assert_eq!(
                held.segments.ptr,
                BASE + (after as u64) * 4,
                "region {at}: the segments do not follow this window's boundaries",
            );
            assert_eq!(
                &packed[after..after + held.segments_host.len()],
                held.segments_host.as_slice(),
            );
            assert_eq!(held.segments.rows, held.segs());
            assert_eq!(held.segments.width, 2, "an `[segs, 2]` rectangle");
        }
    }
    assert!(
        segmented > 0,
        "no window in this fire carries a segment list, and then the staging \
         layout under test is never exercised",
    );
}

/// **AND NAMING THE SHELL'S OWN GROUPED OPS MOVES THE WITHDRAWAL ONTO THEM.**
///
/// This test used to assert that naming a groupable op "changes no bake while
/// nothing groupable is withdrawn" — true while an arbitrary tie-break picked
/// the loser, and false now that cost does. `Grouped` is one launch where a
/// split is `r`, so a groupable consumer is nearly free to lose and
/// `layout::choose` picks it: the score window keeps its interval, the
/// correction takes a segment list, and the artifact goes from twelve rows
/// that cost launches to twenty-four that cost none. That is the composition
/// the two features were built for, and it is why
/// `engine_cuda::serve`'s `PIE_CUDA_GROUPED` is opt-in — it is an improvement,
/// but it is a DIFFERENT artifact from the one that ships by default.
#[test]
fn naming_the_shells_grouped_ops_moves_the_withdrawal_onto_them() {
    let (_, _, trace, _) = model::catalog()
        .into_iter()
        .find(|(sku, ..)| *sku == SKU)
        .expect("the catalog ships the SKU");
    let trace = trace(Platform::Cuda);
    // The shell's real profile with its grouped-capable ops named.
    let profile = DeviceProfile {
        grouped: engine_cuda::GROUPED.iter().map(|op| (*op).to_string()).collect(),
        ..DeviceProfile::default()
    };
    let shipped = compile(&trace, &budget(), &profile).expect("the shipped bake");
    let plain = compile(&trace, &budget(), &DeviceProfile::default()).expect("the plain bake");
    // The default bake pays launches: the score window is withdrawn and owed a
    // `Copy` below the crossover and a `Split` above it, two rows per node.
    assert!(
        !plain.fallback.rows.is_empty()
            && plain
                .fallback
                .rows
                .iter()
                .all(|row| matches!(row.fallback, Fallback::Copy | Fallback::Split { .. })),
        "the default bake owes {:?}",
        plain.fallback.rows,
    );
    // Naming the ops moves the withdrawal onto them, and every row it owes is
    // one launch over a segment list — at every bucket, since one launch beats
    // `r` at every scale and there is no crossover to consult.
    assert!(
        !shipped.fallback.rows.is_empty()
            && shipped
                .fallback
                .rows
                .iter()
                .all(|row| row.fallback == Fallback::Grouped),
        "naming the grouped ops left somebody paying launches: {:?}",
        shipped.fallback.rows,
    );
    assert_ne!(
        shipped.order.tree().map(PqTree::frontier),
        plain.order.tree().map(PqTree::frontier),
        "the withdrawal did not move, so the discount is not reaching the search",
    );

    let lanes = one_lane_per_class(&shipped);
    let fire = compose(&shipped, &budget(), &lanes).expect("eight lanes compose");
    let rows: Vec<u32> = fire.lanes().iter().map(|lane| lane.rows).collect();
    let windows =
        Windows::of(&trace, &shipped, fire.classes(), &indptr(&rows), Copies::off())
            .expect("the windows");
    // A segment list where the artifact answered `Grouped` and NOWHERE ELSE:
    // one window, `r` segments; and every region the table said nothing about
    // still gets the ordinary window it always got.
    let (mut grouped_regions, mut split_regions) = (0usize, 0usize);
    for at in 0..shipped.template().len() as u32 {
        let owed = fallback::grouped(&shipped, shipped.template()[at as usize].nodes.clone());
        for run in 0..windows.runs(at) {
            let segs = windows.at(at, run).segs();
            if owed {
                assert_eq!(windows.runs(at), 1, "region {at} was answered `Grouped`");
                assert!(segs > 1, "region {at} carries {segs} segments");
            } else {
                assert_eq!(segs, 0, "region {at} run {run} carries a segment list");
            }
        }
        if owed && windows.runs(at) == 1 && windows.at(at, 0).segs() > 1 {
            grouped_regions += 1;
        }
        if windows.runs(at) > 1 {
            split_regions += 1;
        }
    }
    assert!(
        grouped_regions > 0,
        "this fire groups nothing under the shipped bake, so the assertions \
         above are about a presence nobody could have produced",
    );
    assert_eq!(
        split_regions, 0,
        "naming the grouped ops was supposed to leave nobody splitting",
    );
}
