//! Does the Metal text state a polymorphic program? Measured, not assumed.
//!
//! tart's claim is that concurrent requests running **structurally different
//! programs** merge into one supergraph, so operators they share execute
//! exactly once (`.wiki/tart/README.md`). The mechanism is in the lowering and
//! is backend-neutral: `lower` takes `&[Row]`, a `Row` is a request's feature
//! point (`depth_k`, `lora`, `multi_token`, `custom_mask`, `hooked`,
//! `wants_scores`, `samples`), and a `Launch` covers a **row range** — so rows
//! sharing an operator share one rectangle and rows that differ get their own.
//!
//! Having the mechanism is not the same as using it: a text has to state
//! guards on those axes. This file used to record that the Metal text stated
//! **none** — 367 launches, every one covering the whole fire, on every row set
//! tried, including sets the CUDA lowering refuses outright.
//!
//! **That changed when the text declared its depth axis.** `m.depth_window()`
//! makes every layer-tagged statement implicitly `rows(depth > layer)`, and
//! this file now asserts what that buys and what it costs:
//!
//! * rows truncating at different layers produce **narrowing rectangles**
//!   rather than one rectangle per op — the shared prefix executes once, which
//!   is the supergraph claim;
//! * and the text now imposes the **seriation contract**, refusing a row order
//!   whose depth runs are not contiguous with `Discontiguous { axis: "depth" }`
//!   — the same refusal the CUDA text makes, and the reason the frame bridge
//!   has to hand rows over in seriated order.
//!
//! The axes still unstated are `lora`, `custom_mask` and `hooked`; the last two
//! need `RowPred`'s partitions, which need seams this backend does not have
//! yet. Each is asserted below as unstated, so landing one fails here.

use std::collections::BTreeSet;

use model::families::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::families::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Fire, Lowered, Row, Uncovered, lower};
use model_compiler::trace::FireClass;

fn plan(class: FireClass) -> model_compiler::trace::ForwardPlan {
    llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        class,
    )
}

fn fire(class: FireClass, rows: &[Row]) -> Result<Lowered, Uncovered> {
    lower(&plan(class), rows, Fire {
        captures_across_splits: false,
    })
}

/// `n` rows, all the same point — the monomorphic fire.
fn uniform(n: usize) -> Vec<Row> {
    vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ]
}

/// `n` rows **seriated for depth**: the full-depth rows first, the truncated
/// ones last, so every layer's alive set is a prefix.
///
/// That order is not a convenience — it is the contract a depth split imposes,
/// and producing it is the scheduler's job (the region table the frame bridge
/// reads is its output).
fn seriated_by_depth(n: usize, truncated: usize, k: u32) -> Vec<Row> {
    (0..n)
        .map(|i| Row {
            samples: true,
            depth_k: (i >= n - truncated).then_some(k),
            ..Row::default()
        })
        .collect()
}

/// The distinct row ranges a fire's rectangles cover.
fn ranges(low: &Lowered) -> BTreeSet<(u32, u32)> {
    low.launches
        .iter()
        .map(|l| (l.rows.start, l.rows.end))
        .collect()
}

#[test]
fn the_metal_text_splits_on_depth_so_a_shared_prefix_executes_once() {
    // Eight rows, half of them stopping at layer 4. Layers 0..4 serve all
    // eight; layers 4.. serve only the four that survive. If the text stated
    // no depth axis, every rectangle would cover all eight rows and the four
    // truncated ones would compute 20 layers nobody reads.
    let low = fire(FireClass::Prefill, &seriated_by_depth(8, 4, 4)).expect("a seriated fire lowers");
    let ranges = ranges(&low);
    assert!(
        ranges.len() > 1,
        "the depth axis produced ONE row range, so the text has stopped \
         splitting on depth: {ranges:?}"
    );
    assert!(
        ranges.contains(&(0, 8)),
        "the shared prefix should cover the whole fire: {ranges:?}"
    );
    assert!(
        ranges.contains(&(0, 4)),
        "the surviving rows should run the tail alone: {ranges:?}"
    );

    // And the narrowing is real work avoided, not just a different spelling.
    let narrowed = low
        .launches
        .iter()
        .filter(|l| l.rows.end - l.rows.start < 8)
        .count();
    assert!(
        narrowed > 0,
        "no rectangle narrowed, so nothing was saved by the split"
    );
}

#[test]
fn a_uniform_fire_still_lowers_to_one_range_because_nothing_differs() {
    // The axis costs nothing when no row uses it: rows that agree share one
    // rectangle, which is the other half of the supergraph claim.
    for (class, n) in [(FireClass::Decode, 4usize), (FireClass::Prefill, 8)] {
        let low = fire(class, &uniform(n)).expect("a uniform fire lowers");
        assert_eq!(
            ranges(&low).into_iter().collect::<Vec<_>>(),
            vec![(0, n as u32)],
            "{class:?}: rows that agree must share one rectangle"
        );
    }
}

#[test]
fn an_unseriated_depth_order_is_refused_rather_than_lowered_wrong() {
    // The contract a split imposes. The truncated rows come FIRST here, so at
    // layer 4 the alive set is a suffix rather than a prefix — and a rectangle
    // is a row RANGE, so there is no honest way to state that.
    //
    // This is the refusal the CUDA text has always made and the Metal one
    // could not, because it stated no axis to be discontiguous on. Producing
    // rows in seriated order is the frame bridge's job.
    let unseriated: Vec<Row> = (0..8)
        .map(|i| Row {
            samples: true,
            depth_k: (i < 4).then_some(4),
            ..Row::default()
        })
        .collect();
    assert!(
        matches!(
            fire(FireClass::Prefill, &unseriated),
            Err(Uncovered::Discontiguous { axis: "depth", .. })
        ),
        "an unseriated depth order lowered anyway, which means the rows a \
         rectangle covers are not the rows the text meant"
    );
}

#[test]
fn the_axes_the_text_does_not_yet_state_are_named_here() {
    // Each of these fails when its guard lands, and that failure is the signal
    // to move it into the split assertions above.
    //
    // `lora` needs a span-grouped correction this backend has no kernel for;
    // `custom_mask` and `hooked` need `RowPred`'s partitions, which need the
    // seams the Metal text still lacks (its own header records their absence).
    let axes: [(&str, Vec<Row>); 3] = [
        (
            "lora",
            (0..8)
                .map(|i| Row {
                    samples: true,
                    lora: i >= 4,
                    ..Row::default()
                })
                .collect(),
        ),
        (
            "custom_mask",
            (0..8)
                .map(|i| Row {
                    samples: true,
                    custom_mask: i >= 4,
                    ..Row::default()
                })
                .collect(),
        ),
        (
            "hooked",
            (0..8)
                .map(|i| Row {
                    samples: true,
                    hooked: i >= 4,
                    ..Row::default()
                })
                .collect(),
        ),
    ];
    for (axis, rows) in axes {
        let low = fire(FireClass::Prefill, &rows).expect("lowers");
        assert_eq!(
            ranges(&low).len(),
            1,
            "the text has gained a `{axis}` guard — move it into the split \
             assertions and say what it buys"
        );
    }
}
