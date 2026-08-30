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
    Budget, Lowering, PATCH_LATTICE_FLOOR, Phase, Placement, collectives_are_never_elided,
};
use model_ir::ValueId;

mod common;
use common::{PLATFORMS, bake, patch_ladder_for, states_patches};

/// **THE RESTATEMENT IS CHECKED AGAINST THE RULE IT RESTATES.**
///
/// `patch_ladder_for` above is a second statement of
/// `engine_cuda::api::patch_ladder`, and model-compiler cannot call the
/// authority to diff against it — the dependency runs the other way, which is
/// the whole reason there are two. What CAN be asserted here is that this copy
/// obeys the RULE in prose rather than some list somebody typed: read the
/// ladder back and check each clause separately from the loop that built it.
///
/// A cross-crate diff belongs on engine-cuda's side, where `api.rs`'s own unit
/// tests can depend on model-compiler and ask both. This is the half that can
/// be asked from here, and it is what turns "somebody changed the rule and not
/// this file" from a silence into a red line.
///
/// NOT `#[ignore]`d: it bakes nothing and reads no catalog.
#[test]
fn the_ladder_this_file_derives_is_the_one_the_rule_describes() {
    for max_tokens in [8192u32, 4096, 2048, 1024, 96, 8] {
        let budget = Budget::new(256, max_tokens);
        let ladder = patch_ladder_for(&budget);

        // The ceiling: the token rectangle's, capped at two whole images, and
        // never below one whole image.
        let want = max_tokens.min(4096).max(PATCH_LATTICE_FLOOR);
        assert_eq!(ladder.max_patches, want, "the ceiling at {max_tokens} tokens");

        // The rungs: they start at the floor, they double, and the last one is
        // the ceiling. Asked of the vector rather than of the loop.
        assert_eq!(
            ladder.buckets.first().copied(),
            Some(PATCH_LATTICE_FLOOR),
            "the ladder starts at the smallest whole image: {:?}",
            ladder.buckets,
        );
        assert_eq!(
            ladder.buckets.last().copied(),
            Some(ladder.max_patches),
            "the ladder ends at its ceiling: {:?}",
            ladder.buckets,
        );
        for pair in ladder.buckets.windows(2) {
            let (low, high) = (pair[0], pair[1]);
            assert!(
                high == low * 2 || high == ladder.max_patches,
                "rung {high} follows {low} and is neither its double nor the \
                 ceiling: {:?}",
                ladder.buckets,
            );
        }

        // `max_images` is the ceiling AT the floor, and never zero — a
        // deployment that admits patch rows admits at least one image.
        assert_eq!(
            ladder.max_images,
            (ladder.max_patches / PATCH_LATTICE_FLOOR).max(1),
            "as many images as the ceiling holds at the floor",
        );
        assert!(ladder.max_images >= 1, "a ladder admits at least one image");
    }

    // AND IT IS THE LADDER THE SIBLING FILE STATES BY HAND. At the 8192-token
    // deployment every sweep in this tree uses,
    // `the_second_row_axis_costs_the_first_nothing`'s `also_admitting_patches`
    // writes the rungs out as a literal; a derivation that disagreed with the
    // one hand-written ladder in the crate would be one of the two being wrong.
    let ladder = patch_ladder_for(&Budget::new(256, 8192));
    assert_eq!(ladder.max_patches, 4096);
    assert_eq!(ladder.buckets, vec![64, 128, 256, 512, 1024, 2048, 4096]);
}

#[test]
#[ignore = "catalog sweep: bakes every SKU on every platform (and, for the renumbering gate, every permutation of its fact bits); minutes, not seconds. Run it with `-- --ignored`, which CI's workspace-verify job does"]
fn every_sku_bakes_on_every_platform() {
    let mut refused: Vec<String> = Vec::new();
    let (mut pairs, mut towers) = (0usize, 0usize);

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            pairs += 1;
            towers += usize::from(states_patches(&trace));
            if let Err(refusal) = bake(&trace) {
                refused.push(format!("`{sku}` as {platform:?}: {}", refusal.say(&trace)));
            }
        }
    }

    assert!(refused.is_empty(), "\n{}\n", refused.join("\n"));
    // NOT VACUOUS ON EITHER AXIS. This sweep baked the whole catalog through
    // token-only ceilings until the towers arrived, and then refused the two
    // rows that state `Dim::Patches` — correctly, and by name: token ceilings
    // size no patch rectangle. What fixed it was the fixture deriving a ladder,
    // so a run that exercised no patch-stating row would be green for the
    // reason it was green before, and a run that exercised ONLY them would have
    // stopped covering the catalog.
    assert!(
        towers > 0,
        "no row of the catalog states `Dim::Patches`, so the ladder this file \
         derives is never asked for and the second axis is untested here",
    );
    assert!(
        towers < pairs,
        "every row states `Dim::Patches`, so the token-only path is never taken",
    );
}

#[test]
#[ignore = "catalog sweep: bakes every SKU on every platform (and, for the renumbering gate, every permutation of its fact bits); minutes, not seconds. Run it with `-- --ignored`, which CI's workspace-verify job does"]
fn no_sku_carves_an_arena_that_shares_live_bytes() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = bake(&trace) else {
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
#[ignore = "catalog sweep: bakes every SKU on every platform (and, for the renumbering gate, every permutation of its fact bits); minutes, not seconds. Run it with `-- --ignored`, which CI's workspace-verify job does"]
fn the_refined_clash_rule_agrees_with_the_v1_oracle_wherever_no_column_is_shared() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = bake(&trace) else {
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
#[ignore = "catalog sweep: bakes every SKU on every platform (and, for the renumbering gate, every permutation of its fact bits); minutes, not seconds. Run it with `-- --ignored`, which CI's workspace-verify job does"]
fn the_carve_lands_on_the_busiest_instant_and_not_the_sum() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = bake(&trace) else {
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
/// `engine::fire::walk` refuses a template for lacking.
#[test]
#[ignore = "catalog sweep: bakes every SKU on every platform (and, for the renumbering gate, every permutation of its fact bits); minutes, not seconds. Run it with `-- --ignored`, which CI's workspace-verify job does"]
fn the_regions_tile_every_plan_prepare_first_and_in_order_within_each_phase() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = bake(&trace) else {
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
#[ignore = "catalog sweep: bakes every SKU on every platform (and, for the renumbering gate, every permutation of its fact bits); minutes, not seconds. Run it with `-- --ignored`, which CI's workspace-verify job does"]
fn a_collective_is_never_elided_and_a_conditional_is_a_structural_arm() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = bake(&trace) else {
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
            // **ASKED OF THE TEXT, NOT OF THE SKU NAME**, and there are now
            // TWO things a text can declare that earn one.
            //
            // The first is the MTP head: `trace.seams`, the same predicate
            // `engine-cuda`'s `export_axes` asks. A prefix match said
            // `qwen36-27b*` and meant "the drafting SKU", and the two stopped
            // being the same set when the quant wave shipped
            // `qwen36-27b-mlxu4-kv-bf16` off `Model::d27b_undrafted`: mlx_lm
            // implements no multi-token-prediction arm for this family, so
            // the 4-bit artifacts carry none of the fifteen `mtp.*` planes and
            // a text that demanded them would refuse every one of them.
            //
            // The second is the VISION TOWER, and it is the newer half. A
            // tower is guarded by `Facts::media` — alto M-1 put the guard on
            // the merge — so a fire whose lanes carry no image skips the whole
            // tower rather than launching it over an empty window. That is a
            // conditional in exactly the sense this gate means: rare,
            // deliberate and structural, and the entire reason the media fact
            // exists. It was invisible here until this file learned to derive
            // a patch ladder, because a tower row refused the bake and every
            // sweep in this tree skipped past it.
            let expected = usize::from(trace.seams.iter().any(|seam| seam.seam == "mtp"))
                + usize::from(states_patches(&trace));
            if conditional != expected {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: {conditional} conditional regions, and \
                     the catalog's answer at this profile is {expected} — one per \
                     `mtp` seam, one per media-guarded tower, and nothing else",
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

/// **THE CORRECTION'S WINDOW IS ONE RUN, OR THE MENU SAYS HOW IT IS SERVED**
/// (palo C2, decision #9's seat).
///
/// Design decision #9 says correction and weight-varied ops are excluded from
/// layout constraints, "gather absorbs them". Read as written it would take
/// `linear.lora_correct`'s region out of the C1P matrix — and that is the one
/// reading this test exists to refute, because in this system a region's rows
/// are a SLICE, not a gather: a region whose classes are not consecutive in
/// the fire's order cannot be one launch over one rectangle, whatever the
/// layout thought of it.
///
/// **WHAT THE REFUSAL ACTUALLY IS.** `Windows::of` refuses a fragmented region
/// that was owed NOTHING — a region P4 seated and then did not. It does not
/// refuse one the fallback menu wrote a row for: that region is SERVED, as
/// `r` launches over its own intervals, or as one over a copy of their union,
/// or as one over a segment list. So the invariant is not "every correction
/// region is an interval" — it is that a correction region is an interval OR
/// it carries an answer. A region that is neither is the bug: a fire the shell
/// meets at run time with no way to launch it.
///
/// Both arms occur in the shipped catalog and both are asserted, because a
/// gate that only ever saw one of them would be half a gate. Since alto A-6
/// put the correction in five more texts, gemma's three attention arms win
/// C1P and its correction lands at order positions `[1, 3, 5]` — three runs,
/// and answered `Copy` below the crossover and `Split { r: 3 }` above it.
/// `layout::gather_absorbs`' own note carries the measurement that says
/// withdrawing it does not come free.
#[test]
#[ignore = "catalog sweep: bakes every SKU on every platform (and, for the renumbering gate, every permutation of its fact bits); minutes, not seconds. Run it with `-- --ignored`, which CI's workspace-verify job does"]
fn every_correction_region_gets_a_window_of_one_run() {
    use model_ir::{Linear, Operation};

    let mut checked = 0usize;
    let (mut intervals, mut answered) = (0usize, 0usize);
    let mut broken: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = bake(&trace) else {
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
                // The answer the menu wrote for this region, if it wrote one:
                // the rows are keyed by node, and any node of the region
                // carrying one is the region being served.
                let served = compiled
                    .fallback
                    .rows
                    .iter()
                    .any(|row| region.nodes.contains(&row.node));
                if runs == 1 {
                    intervals += 1;
                } else if served {
                    answered += 1;
                } else {
                    broken.push(format!(
                        "`{sku}` as {platform:?}: correction region {index} covers classes \
                         at positions {positions:?} of the order, which is {runs} runs, \
                         and the fallback menu wrote it no row; `Windows::of` refuses \
                         that fire by name"
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
    // BOTH ARMS, OR IT IS HALF A GATE. A run where every correction region is
    // an interval never exercises the menu; one where none of them is never
    // exercises the seat, and would pass on a P4 that had stopped trying.
    assert!(
        intervals > 0,
        "no correction region in the catalog is an interval of its order, so P4 \
         seats none of them and this gate is only reading the fallback menu",
    );
    assert!(
        answered > 0,
        "every correction region in the catalog is an interval, so the served \
         arm is never taken — true before alto A-6 and worth re-reading rather \
         than deleting if it is true again",
    );
}
