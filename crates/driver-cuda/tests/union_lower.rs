//! The union lowering: guards KEPT rather than answered.
//!
//! `lower()` reads the fire's rows, decides every `GuardPred`, and emits
//! only the arm that won — which is what makes the eager executor simple
//! and what makes a union supergraph impossible, because the flat list is
//! already specialised to one fire's variant bits.
//!
//! `lower_with(.., GuardMode::Union)` decides nothing. What this file
//! proves, GPU-free, is the three properties `executor::run_captured`
//! walks on:
//!
//! * the tree is well-formed — every parent is in range, and the two arms
//!   of a chain node are recognisable as siblings;
//! * the launch order is WELL-NESTED — a region, once left, is never
//!   returned to, which is what lets the walk be a stack diff instead of a
//!   sort;
//! * the union is a SUPERSET — it emits at least what the resolved
//!   lowering does, because it emits every arm rather than one.

#![cfg(feature = "_cuda")]

use std::collections::BTreeSet;

use model::families::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
use model::families::llama_like::forward::llama_like_cuda;
use model_compiler::lower::{CondRegion, Fire, GuardMode, Launch, Lowered, Row, lower_with};
use driver_cuda::gpu::fire::recordings::predicate_of;
use model_compiler::trace::FireClass;

fn lowered(class: FireClass, rows: usize, guards: GuardMode) -> Lowered {
    let plan = llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        class,
    );
    // `lora` on, so the family's `HasLora` guard has something to be
    // undecided ABOUT — under `Resolve` this is what picks the arm, and
    // under `Union` it must stop mattering.
    let rows: Vec<Row> =
        vec![Row { samples: true, lora: true, ..Row::default() }; rows];
    lower_with(&plan, &rows, Fire { captures_across_splits: false }, guards)
        .expect("the live form lowers")
}

/// The path from the root to `cond`, as tree-node indices — the same walk
/// the driver does.
fn path(conds: &[CondRegion], cond: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut at = cond;
    while at != Launch::NO_COND {
        out.push(at);
        let Some(node) = conds.get(at as usize) else { break };
        at = node.parent;
    }
    out.reverse();
    out
}

#[test]
fn resolve_mode_leaves_no_conditions() {
    let l = lowered(FireClass::Decode, 4, GuardMode::Resolve);
    assert!(l.conds.is_empty(), "a resolved lowering has no guard tree");
    assert!(
        l.launches.iter().all(|x| x.cond == Launch::NO_COND),
        "a resolved lowering puts no rectangle under a condition"
    );
}

#[test]
fn the_union_keeps_a_well_formed_tree() {
    let l = lowered(FireClass::Decode, 4, GuardMode::Union);
    assert!(
        !l.conds.is_empty(),
        "llama_like states a HasLora guard; the union must keep it"
    );
    for (i, r) in l.conds.iter().enumerate() {
        assert!(
            r.parent == Launch::NO_COND || (r.parent as usize) < i,
            "node {i} names a parent that is not an EARLIER node: {}",
            r.parent
        );
        assert!(
            (r.slot as usize) < driver_cuda::gpu::device::PRED_SLOTS,
            "node {i} names predicate slot {}, outside the device word",
            r.slot
        );
    }
    // Every arm is PAIRED with its sibling, and the pairing is stated
    // rather than derived: a family states the same guard once per layer,
    // so `(parent, slot, param)` identifies one conditional per layer and
    // a driver deriving the pair would match the wrong node.
    for (i, r) in l.conds.iter().enumerate() {
        let sib = l.conds.get(r.sibling as usize).unwrap_or_else(|| {
            panic!("node {i} names sibling {}, which is not in the tree", r.sibling)
        });
        assert_eq!(sib.sibling as usize, i, "node {i}'s sibling does not name it back");
        assert_ne!(sib.on_true, r.on_true, "node {i} and its sibling are the same arm");
        assert_eq!(sib.parent, r.parent, "node {i} and its sibling sit under different parents");
        assert_eq!(sib.slot, r.slot, "node {i} and its sibling read different predicates");
    }
}

#[test]
fn every_rectangle_names_a_region_that_exists() {
    let l = lowered(FireClass::Decode, 4, GuardMode::Union);
    for (i, x) in l.launches.iter().enumerate() {
        assert!(
            x.cond == Launch::NO_COND || (x.cond as usize) < l.conds.len(),
            "launch {i} names region {}, which is not in the tree",
            x.cond
        );
    }
}

#[test]
fn the_launch_order_is_well_nested() {
    let l = lowered(FireClass::Decode, 4, GuardMode::Union);

    // The property the stack diff rests on: once the walk LEAVES a
    // region it never comes back. If it did, `run_captured` would have to
    // reopen a closed conditional — which CUDA has no call for, and which
    // would silently produce a second node on the same predicate.
    let mut closed: BTreeSet<u32> = BTreeSet::new();
    let mut open: Vec<u32> = Vec::new();

    for (i, x) in l.launches.iter().enumerate() {
        let target = path(&l.conds, x.cond);
        let mut keep = 0;
        while keep < open.len() && keep < target.len() && open[keep] == target[keep] {
            keep += 1;
        }
        for &left in &open[keep..] {
            closed.insert(left);
        }
        for &entered in &target[keep..] {
            assert!(
                !closed.contains(&entered),
                "launch {i} re-enters region {entered}, which the walk already left"
            );
        }
        open = target;
    }
}

#[test]
fn the_union_is_a_superset_of_the_resolved_form() {
    let resolved = lowered(FireClass::Decode, 4, GuardMode::Resolve);
    let union = lowered(FireClass::Decode, 4, GuardMode::Union);
    assert!(
        union.launches.len() >= resolved.launches.len(),
        "the union emits every arm, so it cannot be shorter: {} < {}",
        union.launches.len(),
        resolved.launches.len()
    );
    // And the arena has to hold all of them at once, which is the union's
    // stated cost.
    assert!(union.arena_bytes >= resolved.arena_bytes);
}

#[test]
fn the_union_filtered_by_its_predicates_is_the_resolved_form() {
    // THE test of the whole design, and it needs no GPU.
    //
    // The eager leg answers a guard at lowering time (`select`). The
    // captured leg answers it by reading a byte the host computed
    // (`predicate_of`) out of device memory. If those two ever disagree,
    // the replay runs a different program from the one the A/Bs proved —
    // silently, because both are valid launch lists.
    //
    // So: take the union, decide every region with the predicate word,
    // keep the launches whose whole path was taken, and require the
    // result to be exactly what `Resolve` emitted.
    for lora in [false, true] {
        for hooked in [false, true] {
            let plan = llama_like_cuda(
                &LlamaLikeFacts::qwen3_0_6b(),
                &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
                FireClass::Decode,
            );
            let fire = Fire { captures_across_splits: false };
            let rows: Vec<Row> =
                vec![Row { samples: true, lora, hooked, ..Row::default() }; 4];

            let resolved = lower_with(&plan, &rows, fire, GuardMode::Resolve).expect("lowers");
            let union = lower_with(&plan, &rows, fire, GuardMode::Union).expect("lowers");

            let live = |cond: u32| -> bool {
                path(&union.conds, cond).into_iter().all(|n| {
                    let r = union.conds[n as usize];
                    predicate_of(r.slot, r.param, &rows).is_none_or(|v| v == r.on_true)
                })
            };

            let taken: Vec<_> = union
                .launches
                .iter()
                .filter(|x| live(x.cond))
                .map(|x| {
                    (union.kernels[x.kernel as usize].as_str(), x.rows.clone(), x.layers.clone())
                })
                .collect();
            let want: Vec<_> = resolved
                .launches
                .iter()
                .map(|x| {
                    (
                        resolved.kernels[x.kernel as usize].as_str(),
                        x.rows.clone(),
                        x.layers.clone(),
                    )
                })
                .collect();

            assert_eq!(
                taken, want,
                "lora={lora} hooked={hooked}: the union decided by its own \
                 predicates is not the program the eager leg runs"
            );
        }
    }
}

#[test]
fn a_lora_free_fire_unions_to_the_same_program() {
    // The point of the union: the SAME capture serves a fire that carries
    // adapters and one that does not. Under `Resolve` these two are
    // different launch lists; under `Union` they must not be, because the
    // predicate moved to device memory.
    let plan = llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        FireClass::Decode,
    );
    let fire = Fire { captures_across_splits: false };
    let with: Vec<Row> = vec![Row { samples: true, lora: true, ..Row::default() }; 4];
    let without: Vec<Row> = vec![Row { samples: true, ..Row::default() }; 4];

    let a = lower_with(&plan, &with, fire, GuardMode::Union).expect("lowers");
    let b = lower_with(&plan, &without, fire, GuardMode::Union).expect("lowers");
    assert_eq!(
        a.launches.len(),
        b.launches.len(),
        "the union must not vary with a variant bit"
    );
    assert_eq!(a.conds, b.conds, "nor may the guard tree");

    // While the resolved forms DO differ — which is the whole reason the
    // union has to exist.
    let ra = lower_with(&plan, &with, fire, GuardMode::Resolve).expect("lowers");
    let rb = lower_with(&plan, &without, fire, GuardMode::Resolve).expect("lowers");
    assert_ne!(
        ra.launches.len(),
        rb.launches.len(),
        "if these matched, llama_like's lora guard would not be a variant axis at all"
    );
}
