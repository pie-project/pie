//! P6's safety argument, over the whole catalog: **two regions the compiler
//! put on different streams write disjoint values and disjoint arena bytes.**
//!
//! WHY THIS FILE EXISTS AND NOT A SENTENCE IN A DOC. Every other property of
//! a fork group is checked by something that faults — a template with an
//! unjoined side stream ends its capture on an error, a region on a stream the
//! shell never opened is a refusal by name. A RACE IS NOT LIKE THAT. Two
//! concurrent kernels writing one rectangle both succeed, the graph replays,
//! the fire returns, and what changed is a logit. So the argument that they
//! cannot is made statically, here, against the artifact `compile` actually
//! produced for every SKU on every platform.
//!
//! The three clauses, and where each one is checked:
//!
//! 1. **Disjoint values** — asserted directly below: the written-value sets of
//!    a concurrent pair do not meet. This is the dependency DAG's own
//!    guarantee restated on its output (a shared write is a WAW edge, and a
//!    pair with an edge is not a candidate), which is exactly the kind of
//!    claim worth checking on the far side of the pass that makes it.
//! 2. **Disjoint bytes** — two ways. Per pair, no rectangle written by one
//!    overlaps a rectangle written by the other, EXCEPT where they are two
//!    arms of one `Def::Merge`, which the carve folded onto one column on
//!    purpose and whose disjoint class masks are what make their rows
//!    disjoint. And globally, `ArenaMap::clashes` under the concurrency
//!    relation P6 built, which is the arena's own invariant asked with the
//!    wider notion of "at one instant" that this pass introduced.
//! 3. **Everything else is read-only during the capture phase** — weights,
//!    the arena base, the pools, the fire inputs and every attention schedule.
//!    That one is structural rather than checkable from here: it is a
//!    statement about when a shell writes those, and `engine-cuda`'s
//!    `inputs.rs`, `weights.rs` and the prepare/capture phase split are what
//!    make it true. `model_compiler::stream`'s module doc argues it.
//!
//! And the pin: WHICH SKUs fork at all, and how many streams they ask for.
//! A pass that quietly stopped finding anything would still be green on every
//! assertion above, because nothing shares anything when nothing is
//! concurrent.

use model_compiler::{ArenaMap, Budget, DeviceProfile, Phase, Region, Placement, compile};
use model_dsl::Platform;
use model_ir::{Operands, Trace, ValueId};

/// Every platform a plan can be traced at — a model text may emit a different op
/// per platform, so the DAG is not the same DAG on each.
const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

fn budgets_for(trace: &Trace) -> Budget {
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
fn no_concurrent_pair_shares_a_written_value() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = compile(&trace, &budgets_for(&trace), &DeviceProfile::default()) else {
                continue; // `every_sku_carves_an_arena` is what says so.
            };
            for &(a, b) in &compiled.streams.pairs {
                let (ra, rb) = (&compiled.regions[a as usize], &compiled.regions[b as usize]);
                let shared: Vec<ValueId> = writes(&trace, ra)
                    .into_iter()
                    .filter(|value| writes(&trace, rb).contains(value))
                    .collect();
                if !shared.is_empty() {
                    wrong.push(format!(
                        "`{sku}` as {platform:?}: regions {a} and {b} run beside each \
                         other and both write {shared:?}",
                    ));
                }
                // The other half of the candidacy rule, restated on the
                // output: disjoint class masks are what make disjoint rows.
                if ra.mask.iter().any(|class| rb.mask.contains(class)) {
                    wrong.push(format!(
                        "`{sku}` as {platform:?}: regions {a} and {b} run beside each \
                         other and share class {:?}",
                        ra.mask.iter().find(|c| rb.mask.contains(*c)),
                    ));
                }
                // And the two rules that are not about sharing at all.
                for (at, region) in [(a, ra), (b, rb)] {
                    if region.phase != Phase::Capture {
                        wrong.push(format!(
                            "`{sku}` as {platform:?}: region {at} is host work and was \
                             put on a stream",
                        ));
                    }
                    if region.collective {
                        wrong.push(format!(
                            "`{sku}` as {platform:?}: region {at} carries a collective \
                             and left the main stream — NCCL matches by call order",
                        ));
                    }
                }
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

#[test]
fn no_concurrent_pair_shares_an_arena_byte_it_writes() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = compile(&trace, &budgets_for(&trace), &DeviceProfile::default()) else {
                continue;
            };

            // The global statement first: the arena's own invariant, asked
            // with P6's wider notion of "live at one instant".
            let clashes = compiled.arena.clashes(&compiled.concurrency);
            if !clashes.is_empty() {
                let named: Vec<String> = clashes
                    .iter()
                    .take(8)
                    .map(|(a, b)| format!("v{}/v{}", a.0, b.0))
                    .collect();
                wrong.push(format!(
                    "`{sku}` as {platform:?}: {} pairs share bytes while both are live \
                     under the concurrency relation — {}",
                    clashes.len(),
                    named.join(", "),
                ));
            }

            // Then the per-pair one, which is the sharper question: not "may
            // these two values be live together" but "do these two REGIONS
            // write into one another's rectangles".
            for &(a, b) in &compiled.streams.pairs {
                let (ra, rb) = (&compiled.regions[a as usize], &compiled.regions[b as usize]);
                for x in writes(&trace, ra) {
                    for y in writes(&trace, rb) {
                        if !overlaps(&compiled.arena, x, y) {
                            continue;
                        }
                        // THE ONE LEGAL SHARING, AND IT IS THE POINT OF THE
                        // WHOLE DESIGN: a merge's arms are ONE column, written
                        // at disjoint row windows — design §0's
                        // zero-instruction phi. Three things have to hold and
                        // all three are checked: the two values fold to the
                        // same root (so it is one rectangle by construction,
                        // not two that happened to collide), that root is
                        // ROW-shaped (`cut_per_class` — a `Dim::Const` column
                        // is handed to a windowed kernel whole, so two classes
                        // may never share one), and the two regions' masks are
                        // disjoint (so `Run::cut` sends each write to rows the
                        // other never touches).
                        if root(&compiled.arena, x) == root(&compiled.arena, y)
                            && cut_per_class(&compiled.arena, x)
                            && !ra.mask.iter().any(|class| rb.mask.contains(class))
                        {
                            continue;
                        }
                        wrong.push(format!(
                            "`{sku}` as {platform:?}: regions {a} and {b} run beside \
                             each other and write overlapping bytes — v{} at {:?} \
                             against v{} at {:?}",
                            x.0,
                            compiled.arena.placements[x.0 as usize],
                            y.0,
                            compiled.arena.placements[y.0 as usize],
                        ));
                    }
                }
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

/// **WHO GETS SIDE STREAMS AT ALL, AND WHO DOES NOT.** A pass that stopped
/// finding candidates would pass every assertion above by being vacuous —
/// nothing shares anything when nothing is concurrent — so the answer itself
/// is asserted.
///
/// **WHAT IS PINNED IS THE PREDICATE, NOT THE COUNT**, and that is deliberate.
/// How many regions a SKU forks is a function of how many layers its model
/// text declares and how its arms are written; those move when somebody edits
/// a family, and a test that failed on every such edit would be a test people
/// learn to update without reading. What cannot move without something being
/// wrong is WHICH families have a fork to find: a plan whose merge arms are
/// windowed over disjoint classes has one, and a plan with no such split does
/// not. The full table is printed in the failure message, so a reader who
/// wants the counts gets them from the run that noticed.
#[test]
fn the_catalog_forks_where_it_declares_disjoint_windows() {
    /// Families whose model text splits attention into arms over disjoint
    /// classes. Every one of them must find a fork group.
    const FORKS: [&str; 5] = ["gemma4-", "glm5-", "gptoss-", "kimik3-", "qwen3"];

    let mut table: Vec<String> = Vec::new();
    let mut wrong: Vec<String> = Vec::new();
    for (sku, _, trace, _) in model::catalog() {
        let trace = trace(Platform::Cuda);
        let compiled =
            compile(&trace, &budgets_for(&trace), &DeviceProfile::default()).expect("bakes");
        let forked = compiled.regions.iter().filter(|r| r.stream != 0).count();
        table.push(format!(
            "  {sku}: {} streams, {} events, {forked} forked of {} regions",
            compiled.streams.streams,
            compiled.streams.events,
            compiled.regions.len(),
        ));

        // A fork and a join are the same fact counted twice: an artifact that
        // opened a stream and minted no event would be a side stream nothing
        // ever waited for, and one that minted an event and opened no stream
        // would be a synchronization with itself.
        if (compiled.streams.streams > 1) != (compiled.streams.events > 0)
            || (compiled.streams.streams > 1) != (forked > 0)
            || (compiled.streams.streams > 1) != !compiled.streams.pairs.is_empty()
        {
            wrong.push(format!(
                "`{sku}`: {} streams, {} events, {forked} forked, {} pairs — a fork \
                 without a join, or the reverse",
                compiled.streams.streams,
                compiled.streams.events,
                compiled.streams.pairs.len(),
            ));
        }

        // **GEMMA REACHES THREE STREAMS**, because its model text declares
        // three attention arms over one window — decode, prefill and masked,
        // the C1b golden. That is the shape this wave was built for and the
        // one count worth stating exactly. It is a FLOOR, not an equality: a
        // family that later declares a fourth arm should reach four, and
        // another family reaching three is not gemma's business.
        if sku.starts_with("gemma4-") && compiled.streams.streams < 3 {
            wrong.push(format!(
                "`{sku}` asked for {} streams: gemma declares three attention arms \
                 over disjoint classes and they should reach three",
                compiled.streams.streams,
            ));
        }
        if FORKS.iter().any(|family| sku.starts_with(family)) && compiled.streams.streams == 1 {
            wrong.push(format!(
                "`{sku}` found no fork group, and its model text splits attention \
                 into arms over disjoint classes",
            ));
        }
    }

    assert!(
        wrong.is_empty(),
        "\n{}\n\nthe whole table:\n{}\n",
        wrong.join("\n"),
        table.join("\n"),
    );
}

/// The off arm is the artifact P6 never touched, on every SKU: stream 0
/// everywhere, no event, no hint, no pair — and an arena no bigger than the
/// one the streams-on bake carves, because the relation only ever WIDENS what
/// counts as one instant.
#[test]
fn the_off_arm_bakes_what_the_pass_never_ran_and_costs_no_bytes() {
    let off = DeviceProfile {
        side_streams: 0,
        ..DeviceProfile::default()
    };
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let budget = budgets_for(&trace);
            let (Ok(on), Ok(quiet)) = (
                compile(&trace, &budget, &DeviceProfile::default()),
                compile(&trace, &budget, &off),
            ) else {
                continue;
            };
            if quiet.regions.iter().any(|r| {
                r.stream != 0 || !r.wait.is_empty() || r.open.is_some() || r.close.is_some()
            }) {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: the off arm forked anyway"
                ));
            }
            if !quiet.concurrency.pairs().is_empty() || quiet.streams.events != 0 {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: the off arm carries a concurrency relation",
                ));
            }
            // The two arms differ in the region table and in nothing else the
            // fire path reads: same classes, same layout, same fallbacks.
            if on.classes != quiet.classes || on.order != quiet.order {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: P6 moved a pass that runs before it",
                ));
            }
            if on.arena.bytes < quiet.arena.bytes {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: streams-on carved {} bytes against \
                     streams-off's {} — a wider relation cannot make the arena \
                     smaller",
                    on.arena.bytes, quiet.arena.bytes,
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

/// Every value a region's nodes write.
fn writes(trace: &Trace, region: &Region) -> Vec<ValueId> {
    let mut all = Vec::new();
    let mut scratch = Vec::new();
    for node in region.nodes.clone() {
        let Some(node) = trace.nodes.get(node as usize) else {
            continue;
        };
        scratch.clear();
        node.op.outputs(&mut scratch);
        all.extend_from_slice(&scratch);
    }
    all.sort_unstable();
    all.dedup();
    all
}

/// The rectangle a value ends up in, following `Placement::Alias` to its root.
fn root(arena: &ArenaMap, value: ValueId) -> ValueId {
    let mut at = value;
    for _ in 0..arena.placements.len() {
        match arena.placements.get(at.0 as usize) {
            Some(Placement::Alias(to)) => at = *to,
            _ => return at,
        }
    }
    at
}

/// Do these two values' rectangles share a byte?
fn overlaps(arena: &ArenaMap, a: ValueId, b: ValueId) -> bool {
    let (Some(x), Some(y)) = (rect(arena, a), rect(arena, b)) else {
        return false;
    };
    x.0 < y.0 + y.1 && y.0 < x.0 + x.1
}

fn rect(arena: &ArenaMap, value: ValueId) -> Option<(u64, u64)> {
    match arena.placements.get(root(arena, value).0 as usize) {
        Some(Placement::Arena { offset, bytes, .. }) => Some((*offset, *bytes)),
        _ => None,
    }
}

/// Is this value's rectangle one a window CUTS, rather than one a windowed
/// kernel is handed whole?
///
/// The same question `RowExpr::cut_per_class` answers for the carve, asked
/// here because it is the precondition of the merge-arm exemption above: two
/// classes may share a column only if each of them can be given its own rows
/// of it.
fn cut_per_class(arena: &ArenaMap, value: ValueId) -> bool {
    matches!(
        arena.placements.get(root(arena, value).0 as usize),
        Some(Placement::Arena { rows, .. }) if rows.cut_per_class(),
    )
}
