//! The catalog, baked. Six model texts, four planes, one `compile` each.
//!
//! WHY THE REAL CATALOG AND NOT A FIXTURE. Every unit test in `src/` builds
//! its plan by hand out of `Def` and `Cond`, which means every one of them
//! checks this crate against this crate's own idea of what a plan looks like.
//! What the catalog adds is the only thing a fixture cannot: plans written by
//! somebody else, in the authoring surface, at the size and the shape the
//! deployment actually ships — sixty layers of residual chains, MoE fan-outs,
//! nested split/merge over four facts, per-layer seams. A carve that is right
//! on a five-node chain and wrong on those is a carve that computes.
//!
//! WHAT IT ASSERTS, and each is a bug that does not fault:
//!
//! - **it bakes at all** — a refusal here is a plan the deployment cannot load;
//! - **the arena is clash-free** — two values sharing bytes while both live is
//!   the failure mode the whole `Span` argument exists to prevent, and it
//!   presents as wrong numbers rather than as a crash;
//! - **the refined rule and the v1 oracle agree** — the per-class tightening
//!   is a strict weakening of a predicate that was already correct, so on a
//!   plan where it shares no column the two must answer identically, and
//!   `ArenaMap::clashes_blind` is what asks;
//! - **the arena is finite and non-empty** — a forward pass that needs no
//!   scratch computed nothing, and one whose bytes overflowed would place
//!   every rectangle at zero;
//! - **the reuse is real** — the busiest instant, not the sum. The claim the
//!   carve exists for, measured against the floor it cannot beat;
//! - **the regions tile the node list** — P2 covers every node exactly once,
//!   in program order, so the record script and the plan cannot disagree about
//!   what runs;
//! - **collectives stay always-launch** — decision #5, checked here so that
//!   the day P3 starts choosing, this test is already standing.
//!
//! SILENT ON PURPOSE. Nothing here prints: the numbers ride in the assert
//! messages, where a failing run shows them and a passing run costs nothing.

use model_compiler::{
    Budgets, DeviceProfile, Lowering, Slot, collectives_are_never_elided, compile,
};
use model_dsl::Plane;
use model_ir::ValueId;

/// Every plane a plan can be traced at. A model text may emit a different op
/// per plane, so the split-and-merge structure is not the same graph on each,
/// and one plane passing says nothing about the others.
const PLANES: [Plane; 4] = [Plane::Cuda, Plane::Metal, Plane::Wgpu, Plane::Vulkan];

/// A deployment's ceilings: 256 concurrent requests, 8192 token rows, the
/// bucket lattice a decode-heavy serve rounds up to.
fn budgets() -> Budgets {
    Budgets {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: 32,
    }
}

#[test]
fn every_sku_bakes_on_every_plane() {
    let mut refused: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for plane in PLANES {
            let plan = trace(plane);
            if let Err(refusal) = compile(&plan, &budgets(), &DeviceProfile::default()) {
                refused.push(format!("`{sku}` as {plane:?}: {}", refusal.say(&plan)));
            }
        }
    }

    assert!(refused.is_empty(), "\n{}\n", refused.join("\n"));
}

#[test]
fn no_sku_carves_an_arena_that_shares_live_bytes() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for plane in PLANES {
            let plan = trace(plane);
            let Ok(baked) = compile(&plan, &budgets(), &DeviceProfile::default()) else {
                continue; // the test above is the one that says so.
            };

            let clashes = baked.arena.clashes(&baked.concurrency);
            if !clashes.is_empty() {
                let named: Vec<String> = clashes
                    .iter()
                    .take(8)
                    .map(|(a, b)| format!("v{}/v{}", a.0, b.0))
                    .collect();
                wrong.push(format!(
                    "`{sku}` as {plane:?}: {} pairs share bytes while both are \
                     live — {}",
                    clashes.len(),
                    named.join(", "),
                ));
            }

            let bytes = baked.arena.bytes;
            if bytes == 0 {
                wrong.push(format!(
                    "`{sku}` as {plane:?}: the arena is empty, so the forward \
                     pass computed nothing",
                ));
            }
            let floor = baked.arena.live_bound();
            if bytes < floor {
                wrong.push(format!(
                    "`{sku}` as {plane:?}: the arena is {bytes} bytes and the \
                     busiest instant needs {floor} — some rectangle was placed \
                     outside it",
                ));
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

/// The per-class tightening, pinned against the predicate it refines.
///
/// **WHAT THE CATALOG SAYS TODAY, PLAINLY: nothing shares a column.** Every
/// SKU has class-exclusive rectangles — up to 350 of them on gemma's three
/// classes — but no two of them are ever live at the same node, because a
/// model text writes its decode branch and its prefill branch one AFTER the
/// other and each branch's scratch dies at the merge that closes it. The v1
/// walk was already giving those pairs the same bytes for the older reason,
/// and the tightening has nothing left to take.
///
/// So the assertion is not a number. It is the pair of invariants that hold
/// whether or not a plan ever gives the pass something to do:
///
/// - the refined guard is clean, which is the arena's own invariant; and
/// - where no column is shared, the refined guard and the v1 oracle answer the
///   SAME list — so a green run of this file is not the refinement quietly
///   excusing a clash the older predicate would have caught.
///
/// The day an axis lands whose guarded regions interleave — MTP, a masked
/// second pass, an adapter bank — this test is what will be standing when the
/// second clause stops being vacuous.
#[test]
fn the_refined_clash_rule_agrees_with_the_v1_oracle_wherever_no_column_is_shared() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for plane in PLANES {
            let plan = trace(plane);
            let Ok(baked) = compile(&plan, &budgets(), &DeviceProfile::default()) else {
                continue;
            };
            let arena = &baked.arena;

            // Every rectangle got a class mask, and it is the same table the
            // spans are indexed by.
            if arena.live_in.len() != arena.slots.len() {
                wrong.push(format!(
                    "`{sku}` as {plane:?}: {} class masks for {} slots",
                    arena.live_in.len(),
                    arena.slots.len(),
                ));
            }

            let shared: Vec<(ValueId, ValueId)> = co_tenants(arena);
            let blind = arena.clashes_blind(&baked.concurrency);
            if shared.is_empty() && !blind.is_empty() {
                wrong.push(format!(
                    "`{sku}` as {plane:?}: the carve shares no column and the v1 \
                     oracle still reports {} clashing pairs — the refinement \
                     is excusing something it cannot name",
                    blind.len(),
                ));
            }
            // And in every case the refined answer is a SUBSET of the older
            // one: a weakening, never a different question.
            let refined = arena.clashes(&baked.concurrency);
            if refined.iter().any(|pair| !blind.contains(pair)) {
                wrong.push(format!(
                    "`{sku}` as {plane:?}: the refined guard reports a pair the \
                     v1 oracle does not",
                ));
            }

            // The floor is the busiest instant over COLUMNS, so a carve that
            // shares none of them must sit on exactly the number the v1 walk
            // computed — which is the "never larger" claim, on the only side
            // of it a bake can check from out here.
            if shared.is_empty() && arena.bytes != arena.live_bound() {
                wrong.push(format!(
                    "`{sku}` as {plane:?}: {} bytes against a {}-byte floor with \
                     no column shared",
                    arena.bytes,
                    arena.live_bound(),
                ));
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

/// Every pair of values this map put in one column. Quadratic, and the plans
/// are a few hundred rectangles, so it is cheaper than the compile that made
/// them.
fn co_tenants(arena: &model_compiler::ArenaMap) -> Vec<(ValueId, ValueId)> {
    let rectangles: Vec<ValueId> = (0..arena.slots.len() as u32)
        .map(ValueId)
        .filter(|v| matches!(arena.slots[v.0 as usize], Slot::Arena { .. }))
        .collect();
    let mut found = Vec::new();
    for (i, a) in rectangles.iter().enumerate() {
        for b in &rectangles[i + 1..] {
            if arena.co_tenants(*a, *b) {
                found.push((*a, *b));
            }
        }
    }
    found
}

/// The reuse is the whole point, so it is measured rather than assumed: the
/// arena must be far smaller than the sum of every rectangle the plan mints,
/// and it must sit on — not merely above — the floor the liveness bound
/// states.
///
/// A FACTOR OF TWO IS A DELIBERATELY LOOSE FLOOR FOR THE CLAIM. The rewrite
/// measured 21.8 MiB to 1 MiB on gemma and 2.45 MiB to 487 KiB on qwen, which
/// is 20x and 5x; asserting the measured ratio would make this a regression
/// test on a number that legitimately moves when a model text changes. What
/// cannot legitimately move is that a transformer's scratch is reused at all.
#[test]
fn the_carve_lands_on_the_busiest_instant_and_not_the_sum() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for plane in PLANES {
            let plan = trace(plane);
            let Ok(baked) = compile(&plan, &budgets(), &DeviceProfile::default()) else {
                continue;
            };
            let total: u64 = baked
                .arena
                .slots
                .iter()
                .map(model_compiler::Slot::bytes)
                .sum();
            let bytes = baked.arena.bytes;
            if bytes * 2 > total {
                wrong.push(format!(
                    "`{sku}` as {plane:?}: the arena is {bytes} bytes against a \
                     {total}-byte sum — the liveness walk shared almost nothing",
                ));
            }
            let floor = baked.arena.live_bound();
            if bytes > floor {
                wrong.push(format!(
                    "`{sku}` as {plane:?}: the arena is {bytes} bytes and the \
                     floor is {floor} — the greedy placement left \
                     {} bytes in holes",
                    bytes - floor,
                ));
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

#[test]
fn the_regions_tile_every_plan_in_program_order() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for plane in PLANES {
            let plan = trace(plane);
            let Ok(baked) = compile(&plan, &budgets(), &DeviceProfile::default()) else {
                continue;
            };
            let mut covered = 0u32;
            for region in &baked.regions {
                if region.nodes.start != covered {
                    wrong.push(format!(
                        "`{sku}` as {plane:?}: a region starts at {} with {covered} \
                         nodes covered",
                        region.nodes.start,
                    ));
                    break;
                }
                covered = region.nodes.end;
            }
            if covered as usize != plan.nodes.len() {
                wrong.push(format!(
                    "`{sku}` as {plane:?}: the regions cover {covered} of {} nodes",
                    plan.nodes.len(),
                ));
            }
            // Coalescing has to actually coalesce: a plan whose every node is
            // its own region is a pass that ran and did nothing.
            if !plan.nodes.is_empty() && baked.regions.len() == plan.nodes.len() {
                wrong.push(format!(
                    "`{sku}` as {plane:?}: {} regions for {} nodes — nothing \
                     coalesced",
                    baked.regions.len(),
                    plan.nodes.len(),
                ));
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

#[test]
fn v1_lowers_every_region_the_one_way_that_is_correctness() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for plane in PLANES {
            let plan = trace(plane);
            let Ok(baked) = compile(&plan, &budgets(), &DeviceProfile::default()) else {
                continue;
            };
            if !collectives_are_never_elided(&baked) {
                wrong.push(format!(
                    "`{sku}` as {plane:?}: a region carrying a collective is not \
                     always-launch",
                ));
            }
            if baked
                .regions
                .iter()
                .any(|r| r.lowering != Lowering::AlwaysLaunch)
            {
                wrong.push(format!(
                    "`{sku}` as {plane:?}: a region lowers as something P3 has \
                     not been written to choose yet",
                ));
            }
            if baked.regions.iter().any(|r| r.stream != 0) {
                wrong.push(format!("`{sku}` as {plane:?}: a region is off stream 0"));
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}
