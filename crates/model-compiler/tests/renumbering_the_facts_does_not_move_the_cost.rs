//! A fact's bit number is a POSITION AND NOTHING ELSE (`model_dsl::facts`
//! says so out loud), and this is the test that keeps it true where it was
//! quietly false.
//!
//! WHAT WAS WRONG. P4 seats as many windowed consumers as a PQ-tree can and
//! withdraws the rest. Which one loses used to be decided by offering the
//! masks in descending class count and breaking the tie — on qwen3.5 all four
//! contending masks are size four — on the CLASS INDICES, which are the fact
//! bit numbers the model text happened to choose. `captures_scores` paid
//! because it is `fact(3)` while `has_adapter` is `fact(1)`; swapping those
//! two declarations would have moved a 2.65x cost with nothing anywhere
//! saying so.
//!
//! WHY THE BIT NUMBER MUST NOT BE A PRIORITY. It is one half of a pair: the
//! other half is the shift in `Classify::word`, and that word is `Lane::word`,
//! the runtime's submission field (decision #18). If the bit number also chose
//! the layout, then retuning the layout would mean renumbering the bits, which
//! means changing what the runtime sends — a performance knob welded to a wire
//! format, and therefore a knob nobody can A/B.
//!
//! WHAT THIS ASSERTS. Permute the fact bits of a traced plan — every
//! `Guard::Fact` on every node and every merge arm — and bake it again. The
//! plan is the same forward pass said in a different numbering, so the SET OF
//! NODES owed a fallback must not move. Node indices are untouched by the
//! permutation, which is what makes the comparison exact rather than
//! up-to-relabelling.

use std::collections::BTreeSet;

use model_compiler::{Budget, DeviceProfile, compile};
use model_dsl::Platform;
use model_ir::{Guard, Def, Trace, fact_width};

/// A budget the catalog can actually seat.
///
/// NOT `max_adapters: 32`, which is what four other catalog test files ask
/// for and why they walk their loops asserting nothing: no text seats more
/// than eight, so `compile` refuses every SKU and every body is skipped. The
/// non-vacuity assert at the end of each test here is the other half of not
/// repeating that.
fn budget() -> Budget {
    Budget {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: 0,
    }
}

/// The same guard, said with the bits renamed.
///
/// The smart constructors fold — `Guard::and(Always, x)` is `x` — which is
/// semantics-preserving and is what a trace would have produced had the text
/// numbered its facts this way in the first place.
fn renumber(cond: &Guard, perm: &[u8]) -> Guard {
    match cond {
        Guard::Always => Guard::Always,
        Guard::Fact(bit) => Guard::Fact(perm[*bit as usize]),
        Guard::Not(a) => Guard::not(renumber(a, perm)),
        Guard::And(a, b) => Guard::and(renumber(a, perm), renumber(b, perm)),
        Guard::Or(a, b) => Guard::or(renumber(a, perm), renumber(b, perm)),
    }
}

/// Every `Guard` in the plan lives in one of two places: a node's guard and a
/// merge arm's. Miss either and the permuted plan is a different forward pass.
fn renumbered(trace: &Trace, perm: &[u8]) -> Trace {
    let mut out = trace.clone();
    for node in &mut out.nodes {
        node.guard = renumber(&node.guard, perm);
    }
    for value in &mut out.values {
        if let Def::Merge(arms) = &mut value.def {
            for (_, cond) in arms.iter_mut() {
                *cond = renumber(cond, perm);
            }
        }
    }
    out
}

/// Every permutation of `n` items, `n!` of them. `n` is `fact_width`, which
/// is four on the widest text in the catalog.
fn permutations(n: usize) -> Vec<Vec<u8>> {
    let mut out = vec![Vec::new()];
    for _ in 0..n {
        let mut next = Vec::new();
        for partial in &out {
            for bit in 0..n as u8 {
                if !partial.contains(&bit) {
                    let mut grown = partial.clone();
                    grown.push(bit);
                    next.push(grown);
                }
            }
        }
        out = next;
    }
    out
}

fn owes(trace: &Trace) -> Option<BTreeSet<u32>> {
    let compiled = compile(trace, &budget(), &DeviceProfile::default()).ok()?;
    Some(compiled.fallback.rows.iter().map(|row| row.node).collect())
}

#[test]
fn no_catalog_text_changes_what_it_owes_when_its_facts_are_renumbered() {
    let mut moved: Vec<String> = Vec::new();
    let (mut compiled, mut with_rows) = (0usize, 0usize);

    for (sku, _, trace, _) in model::catalog() {
        let trace = trace(Platform::Cuda);
        let Some(before) = owes(&trace) else { continue };
        compiled += 1;
        if !before.is_empty() {
            with_rows += 1;
        }

        for perm in permutations(fact_width(&trace)) {
            let Some(after) = owes(&renumbered(&trace, &perm)) else {
                moved.push(format!("`{sku}` under {perm:?}: refused after renumbering"));
                continue;
            };
            if after != before {
                moved.push(format!(
                    "`{sku}` under {perm:?}: owed {before:?}, now owes {after:?}",
                ));
            }
        }
    }

    // NOT VACUOUS, TWICE OVER: the catalog bakes, and some of it actually has
    // a withdrawal to get wrong. A green run over an empty fallback table
    // would prove only that zero equals zero.
    assert!(compiled >= 16, "only {compiled} SKUs compiled");
    assert!(
        with_rows >= 4,
        "only {with_rows} SKUs owe a fallback at all, so nothing was at stake",
    );
    assert!(moved.is_empty(), "\n{}\n", moved.join("\n"));
}
