//! The catalog, baked. Six model texts, four platforms, one `compile` each.
//!
//! WHY THE REAL CATALOG AND NOT A FIXTURE. Every unit test in `src/` builds
//! its plan by hand out of `Def` and `Guard`, which means every one of them
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
    Budget, DeviceProfile, Lowering, Phase, Placement, collectives_are_never_elided, compile,
};
use model_dsl::Platform;
use model_ir::ValueId;

/// Every platform a plan can be traced at. A model text may emit a different op
/// per platform, so the split-and-merge structure is not the same graph on each,
/// and one platform passing says nothing about the others.
const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

/// A deployment's ceilings: 256 concurrent requests, 8192 token rows, the
/// bucket lattice a decode-heavy serve rounds up to, and as many adapters as
/// THIS plan's banks seat.
///
/// **`max_adapters` STOPPED BEING A NUMBER NOBODY READ** (palo C2). It sat
/// here at a flat 32 while the IR had no bank seat, so it named an intention
/// and checked nothing; `compile` now refuses a load whose ask is bigger than
/// the model text's own capacity (design §8: the budget IS the shape), and a
/// flat 32 across the catalog would refuse five families for declaring no
/// bank and qwen for declaring eight. So the fixture asks each plan for what
/// it seats — which is what a worker does, and which keeps the bank-declaring
/// SKUs baking AT their ceiling rather than under it.
fn budgets_for(trace: &model_ir::Trace) -> Budget {
    let seats = trace
        .params
        .iter()
        .filter(|param| param.source == model_ir::ParamSource::Registered)
        .map(|param| param.shape.first().copied().unwrap_or(0))
        .min()
        .unwrap_or(0);
    Budget {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: u32::try_from(seats).unwrap_or(u32::MAX),
    }
}

#[test]
fn every_sku_bakes_on_every_platform() {
    let mut refused: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            if let Err(refusal) = compile(&trace, &budgets_for(&trace), &DeviceProfile::default()) {
                refused.push(format!("`{sku}` as {platform:?}: {}", refusal.say(&trace)));
            }
        }
    }

    assert!(refused.is_empty(), "\n{}\n", refused.join("\n"));
}

#[test]
fn no_sku_carves_an_arena_that_shares_live_bytes() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = compile(&trace, &budgets_for(&trace), &DeviceProfile::default()) else {
                continue; // the test above is the one that says so.
            };

            let clashes = compiled.arena.clashes(&compiled.concurrency);
            if !clashes.is_empty() {
                let named: Vec<String> = clashes
                    .iter()
                    .take(8)
                    .map(|(a, b)| format!("v{}/v{}", a.0, b.0))
                    .collect();
                wrong.push(format!(
                    "`{sku}` as {platform:?}: {} pairs share bytes while both are \
                     live — {}",
                    clashes.len(),
                    named.join(", "),
                ));
            }

            let bytes = compiled.arena.bytes;
            if bytes == 0 {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: the arena is empty, so the forward \
                     pass computed nothing",
                ));
            }
            let floor = compiled.arena.live_bound();
            if bytes < floor {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: the arena is {bytes} bytes and the \
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
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = compile(&trace, &budgets_for(&trace), &DeviceProfile::default()) else {
                continue;
            };
            let arena = &compiled.arena;

            // Every rectangle got a class mask, and it is the same table the
            // spans are indexed by.
            if arena.live_in.len() != arena.placements.len() {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: {} class masks for {} placements",
                    arena.live_in.len(),
                    arena.placements.len(),
                ));
            }

            let shared: Vec<(ValueId, ValueId)> = co_tenants(arena);
            let blind = arena.clashes_blind(&compiled.concurrency);
            if shared.is_empty() && !blind.is_empty() {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: the carve shares no column and the v1 \
                     oracle still reports {} clashing pairs — the refinement \
                     is excusing something it cannot name",
                    blind.len(),
                ));
            }
            // And in every case the refined answer is a SUBSET of the older
            // one: a weakening, never a different question.
            let refined = arena.clashes(&compiled.concurrency);
            if refined.iter().any(|pair| !blind.contains(pair)) {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: the refined guard reports a pair the \
                     v1 oracle does not",
                ));
            }

            // The floor is the busiest instant over COLUMNS, so a carve that
            // shares none of them must sit on exactly the number the v1 walk
            // computed — which is the "never larger" claim, on the only side
            // of it a bake can check from out here.
            if shared.is_empty() && arena.bytes != arena.live_bound() {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: {} bytes against a {}-byte floor with \
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
    let rectangles: Vec<ValueId> = (0..arena.placements.len() as u32)
        .map(ValueId)
        .filter(|v| matches!(arena.placements[v.0 as usize], Placement::Arena { .. }))
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
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = compile(&trace, &budgets_for(&trace), &DeviceProfile::default()) else {
                continue;
            };
            let total: u64 = compiled
                .arena
                .placements
                .iter()
                .map(model_compiler::Placement::bytes)
                .sum();
            let bytes = compiled.arena.bytes;
            if bytes * 2 > total {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: the arena is {bytes} bytes against a \
                     {total}-byte sum — the liveness walk shared almost nothing",
                ));
            }
            let floor = compiled.arena.live_bound();
            if bytes > floor {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: the arena is {bytes} bytes and the \
                     floor is {floor} — the greedy placement left \
                     {} bytes in holes",
                    bytes - floor,
                ));
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

/// **THE TILING IS THE CLAIM; PROGRAM ORDER IS NOW THE CLAIM PER PHASE**
/// (design §5). Every node still lands in exactly one region — that is what
/// makes a region the unit the descriptor carries a row count for — but the
/// region TABLE is no longer one ascending run over the nodes, because P5
/// hoists the prepare half in front of the capture half
/// (`model_compiler::region::hoist`). It has to: prepare ops are host work
/// that writes descriptor slots the graph then reads, and a model text may
/// state one anywhere it likes. qwen3.6 states its multi-token-prediction plan
/// build after the trunk, and before the hoist that made every composition of
/// that SKU unfireable.
///
/// So the claim is asked in three parts, and together they are stricter than
/// the single ascending run they replace: the regions COVER every node exactly
/// once; each PHASE's regions ascend, which is the dataflow inside the graph
/// and a topological order among the plan builds; and every prepare region
/// stands before every capture one, which is the property
/// `driver::fire::walk` refuses a template for lacking.
#[test]
fn the_regions_tile_every_plan_prepare_first_and_in_order_within_each_phase() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = compile(&trace, &budgets_for(&trace), &DeviceProfile::default()) else {
                continue;
            };
            // Exactly once: a node counted twice is a rectangle written twice,
            // and a node counted never is a launch nothing emits.
            let mut times = vec![0u32; trace.nodes.len()];
            for region in &compiled.regions {
                for node in region.nodes.clone() {
                    if let Some(slot) = times.get_mut(node as usize) {
                        *slot += 1;
                    }
                }
            }
            let covered = times.iter().filter(|&&n| n == 1).count();
            if covered != trace.nodes.len() {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: the regions cover {covered} of {} nodes \
                     exactly once — {} twice or more, {} not at all",
                    trace.nodes.len(),
                    times.iter().filter(|&&n| n > 1).count(),
                    times.iter().filter(|&&n| n == 0).count(),
                ));
            }
            // Prepare first, whole, and each half ascending inside itself.
            let mut captured = false;
            let mut at = [0u32; 2];
            for (r, region) in compiled.regions.iter().enumerate() {
                let half = usize::from(region.phase == Phase::Capture);
                if region.phase == Phase::Capture {
                    captured = true;
                } else if captured {
                    wrong.push(format!(
                        "`{sku}` as {platform:?}: prepare region {r} stands after the \
                         graph body that reads its slots",
                    ));
                }
                if region.nodes.start < at[half] {
                    wrong.push(format!(
                        "`{sku}` as {platform:?}: region {r} starts at {} behind {} in \
                         its own phase",
                        region.nodes.start, at[half],
                    ));
                }
                at[half] = region.nodes.end;
            }
            // Coalescing has to actually coalesce: a plan whose every node is
            // its own region is a pass that ran and did nothing.
            if !trace.nodes.is_empty() && compiled.regions.len() == trace.nodes.len() {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: {} regions for {} nodes — nothing \
                     coalesced",
                    compiled.regions.len(),
                    trace.nodes.len(),
                ));
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

/// The one lowering rule that is not an optimization, and the one region in
/// the catalog that is one.
///
/// **P3 HAS LANDED, SO "EVERY REGION IS ALWAYS-LAUNCH" IS NO LONGER THE
/// CLAIM.** What survives it is decision #5 — a collective is never elided —
/// and the fact that a conditional is a rare, deliberate, structural thing:
/// exactly one region of one catalog SKU at the default profile.
/// `tests/which_skus_get_a_conditional.rs` is where the gates that decided it
/// are asked; this is the arena file's own restatement, so that a change to
/// the lowering has to break something here too.
#[test]
fn a_collective_is_never_elided_and_a_conditional_is_a_structural_arm() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = compile(&trace, &budgets_for(&trace), &DeviceProfile::default()) else {
                continue;
            };
            if !collectives_are_never_elided(&compiled) {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: a region carrying a collective is not \
                     always-launch",
                ));
            }
            let conditional = compiled
                .regions
                .iter()
                .filter(|r| r.lowering != Lowering::AlwaysLaunch)
                .count();
            let expected = usize::from(sku.starts_with("qwen36-27b"));
            if conditional != expected {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: {conditional} conditional regions, and \
                     the catalog's answer at this profile is {expected} — the MTP \
                     head and nothing else",
                ));
            }
            // **P6 LANDED, SO "EVERY REGION IS ON STREAM 0" IS NO LONGER THE
            // CLAIM** — what survives it is the part that was never about the
            // lowering: host work and collectives stay where the walk puts
            // them. Which regions fork, and that their writes are disjoint, is
            // `tests/no_concurrent_pair_shares_a_write.rs`.
            for (at, region) in compiled.regions.iter().enumerate() {
                if region.stream == 0 {
                    continue;
                }
                if region.phase != model_compiler::Phase::Capture {
                    wrong.push(format!(
                        "`{sku}` as {platform:?}: region {at} is host work and was put                          on a side stream",
                    ));
                }
                if region.collective {
                    wrong.push(format!(
                        "`{sku}` as {platform:?}: region {at} carries a collective and                          left the main stream — NCCL matches by call order",
                    ));
                }
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

/// **THE CORRECTION'S WINDOW IS ONE RUN, AND IT IS ONE RUN BECAUSE P4 WAS
/// ASKED** (palo C2, decision #9's seat).
///
/// Design decision #9 says correction and weight-varied ops are excluded from
/// layout constraints, "gather absorbs them". Read as written it would take
/// `linear.lora_correct`'s region out of the C1P matrix — and that is the one
/// reading this test exists to refute, because in this system a region's rows
/// are a SLICE, not a gather: `Windows::of` refuses a region whose classes are
/// not consecutive in the fire's order, whatever the layout thought of it.
///
/// So: every region carrying a correction must have a mask that P4's chosen
/// order makes into one run. `layout::gather_absorbs`' own note carries the
/// measurement that says withdrawing it does not.
#[test]
fn every_correction_region_gets_a_window_of_one_run() {
    use model_ir::{Linear, Operation};

    let mut checked = 0usize;
    let mut broken: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = compile(&trace, &budgets_for(&trace), &DeviceProfile::default()) else {
                continue; // `every_sku_bakes_on_every_platform` is what says so.
            };
            // Where each class stands in the order P4 chose, with every class
            // present — which is the worst case a fire can present, and the
            // only one that has to hold for all `2^K` of the others (a
            // sub-order of an ordering that makes a set consecutive still
            // makes it consecutive).
            let mut every = model_ir::ClassSet::default();
            for class in 0..compiled.classes.classes.len() {
                every.insert(class);
            }
            let order = compiled.order.class_order(&every, None);
            let mut at = vec![0usize; order.len()];
            for (position, &class) in order.iter().enumerate() {
                at[class as usize] = position;
            }
            for (index, region) in compiled.template().iter().enumerate() {
                let corrects = region.nodes.clone().any(|node| {
                    matches!(
                        trace.nodes.get(node as usize).map(|n| &n.op),
                        Some(Operation::Linear(Linear::LoraCorrect { .. }))
                    )
                });
                if !corrects {
                    continue;
                }
                checked += 1;
                let mut positions: Vec<usize> =
                    region.mask.iter().map(|class| at[class]).collect();
                positions.sort_unstable();
                let runs = 1 + positions
                    .windows(2)
                    .filter(|pair| pair[1] != pair[0] + 1)
                    .count();
                if runs != 1 {
                    broken.push(format!(
                        "`{sku}` as {platform:?}: correction region {index} covers classes \
                         at positions {positions:?} of the order, which is {runs} runs; \
                         `Windows::of` refuses that fire by name"
                    ));
                }
            }
        }
    }

    assert!(broken.is_empty(), "\n{}\n", broken.join("\n"));
    assert!(
        checked > 0,
        "no SKU in the catalog states a correction op, and then this test is vacuous"
    );
}
