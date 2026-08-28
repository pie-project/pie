//! The two axes palo C3 and C4 declare, held against the plans that carry them
//! and against the plans that do not.
//!
//! Design §8's rule is that a mid-fire axis is a MODEL-DECLARED supergraph
//! axis: a fact bit, a window, and either a correction over the window or an
//! arm of a merge. Build log 22 recorded C3 (MTP) and C4 (score capture) as
//! "vacuous today, by doctrine — no catalog model text declares them". This
//! file is what stops them being vacuous, and it asks the three questions a
//! declared axis has to survive:
//!
//! - **does the bit reach the arm?** A lane's word is `Facts::word`, packed in
//!   `qwen_3/forward.rs`; a class is `resolve_classes`, in the IR. Nothing
//!   makes the two agree except that they read the same bit positions, and the
//!   cross-check goes THROUGH THE PLAN — the class a drafting word belongs to
//!   must run the draft head's ops, and the class a non-drafting word belongs
//!   to must not. A test that asserted "bit 2 is `drafts`" would be the
//!   classifier restated and could not catch it being wrong.
//! - **does the export survive the fire?** A draft column is read after the
//!   graph has run, by the same sampler that reads the trunk's logits
//!   (`driver::program`'s `MtpLogits`/`MtpDrafts` index the readout at
//!   `mtp_draft_row`). `model_compiler::arena` gives that delivery tail to the
//!   `"out"` seam by name and to no other, so the model text buys it by
//!   ORDERING — and this is what notices if the order moves.
//! - **do the new axes cost the old words anything?** The claim the whole
//!   window mechanism exists for: a fire whose lanes set neither new fact must
//!   run the nodes it ran before, in the classes it had before, out of the
//!   arena it had before. Pinned against numbers measured at the commit before
//!   these axes landed.

use std::collections::BTreeSet;

use model_compiler::{Budgets, DeviceProfile, compile};
use model_dsl::{
    Attention, Operands, Operation, Param, ParamSource, Plan, Platform, ValueId, resolve_classes,
};

/// Every platform a plan can be traced at: a model text may emit a different op
/// per platform, so an axis that survives on one says nothing about the others.
const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

/// The one shipping SKU whose checkpoint publishes a draft head.
const DRAFTING: &str = "qwen36-27b-bf16-kv-bf16";

/// The workhorse the capture arm is asked about — small, dense, and the SKU
/// palo C2 ran the adapter goldens on.
const CAPTURING: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// `drafts` and `captures_scores`, as bits of a lane's word.
const NEW_BITS: u64 = (1 << 2) | (1 << 3);

fn budgets_for(plan: &Plan) -> Budgets {
    let seats = plan
        .params
        .iter()
        .filter(|p: &&Param| p.source == ParamSource::Registered)
        .map(|p| p.shape.first().copied().unwrap_or(0))
        .min()
        .unwrap_or(0);
    Budgets {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: u32::try_from(seats).unwrap_or(u32::MAX),
    }
}

fn plan_of(sku: &str, platform: Platform) -> Plan {
    let trace = model::trace_of(sku).unwrap_or_else(|| panic!("this build ships `{sku}`"));
    trace(platform)
}

/// The one value a named seam exports. A seam over several values is legal —
/// `attn.qv` names two, and `attn.scores` names one PER LAYER — but the two
/// whole-fire export seams name one each, and a second would mean the text
/// changed under the test.
fn exported(plan: &Plan, seam: &str) -> ValueId {
    let mut found: Vec<ValueId> = plan
        .seams
        .iter()
        .filter(|s| s.seam == seam)
        .flat_map(|s| s.values.iter().copied())
        .collect();
    found.dedup();
    match found.as_slice() {
        [one] => *one,
        many => panic!("the `{seam}` seam names {} values, not one", many.len()),
    }
}

/// **C3: THE DRAFT WINDOW IS THE WINDOW THE DRAFT HEAD RUNS IN.**
///
/// The cross-check is through the plan and not through the bit: the class a
/// drafting lane composes as must run the head's attention, and the class a
/// non-drafting lane composes as must not. That holds only if the bit
/// `Facts::word` sets is the bit `Facts::drafts()`'s guard reads.
///
/// The head is told apart from the trunk by its KV ROW. Both are
/// `Attention::Prefill` nodes over the one page-id space, and what separates
/// them is which cache each walks — `kv.mtp` is the head's and nobody else's.
#[test]
fn a_drafting_lane_lands_in_a_class_that_runs_the_draft_head() {
    let classify = model::classify_of(DRAFTING).expect("and its classifier");
    for platform in PLATFORMS {
        let plan = plan_of(DRAFTING, platform);
        let classes = resolve_classes(&plan).expect("the qwen36 plan resolves every merge");

        // Which cache index is the head's own row.
        let head_cache = plan
            .caches
            .iter()
            .position(|row| matches!(row, model_dsl::CacheRow::Kv { name, .. } if name == "kv.mtp"))
            .unwrap_or_else(|| panic!("{platform:?}: the draft head declares no kv row"))
            as u32;
        let head_pages = plan
            .values
            .iter()
            .position(|v| v.def == model_dsl::Def::Cache(head_cache))
            .unwrap_or_else(|| panic!("{platform:?}: nothing reads the draft head's kv row"))
            as u32;

        let runs_the_head = |class: usize| {
            plan.nodes
                .iter()
                .zip(&classes.node_mask)
                .any(|(node, mask)| {
                    mask.contains(class)
                        && matches!(
                            &node.op,
                            Operation::Attention(Attention::Prefill { cache, .. })
                                if cache.0 == head_pages
                        )
                })
        };

        let drafting = classify(&model::Request::new(1, false).drafting(true)) & classes.mask;
        let plain = classify(&model::Request::new(1, false)) & classes.mask;
        assert_ne!(
            drafting, plain,
            "{platform:?}: a drafting lane and a plain one are the same word, so \
             `drafts` is not a fact this classifier computes"
        );

        let drafting = classes
            .class_of(drafting)
            .unwrap_or_else(|| panic!("{platform:?}: the drafting word names no class"));
        let plain = classes
            .class_of(plain)
            .unwrap_or_else(|| panic!("{platform:?}: the plain word names no class"));

        assert!(
            runs_the_head(drafting),
            "{platform:?}: the class a drafting lane composes as does not run the \
             draft head's attention, so the axis is declared and unreachable"
        );
        assert!(
            !runs_the_head(plain),
            "{platform:?}: the class a NON-drafting lane composes as runs the \
             draft head — the window is not a window and every fire pays for a \
             second transformer block and a second vocabulary GEMM"
        );
    }
}

/// **C4: THE CAPTURE WINDOW IS THE WINDOW THE LSE ATTENTION RUNS IN.**
///
/// Same shape of argument, through the op vocabulary this time: the capture
/// arm is `Attention::PrefillLse`, which hands back the per-query normalizing
/// mass beside `o`, and no other arm of this text's merge is that variant.
#[test]
fn a_capturing_lane_lands_in_a_class_that_runs_the_lse_attention() {
    let classify = model::classify_of(CAPTURING).expect("this build ships the qwen row");

    for platform in PLATFORMS {
        let plan = plan_of(CAPTURING, platform);
        let classes = resolve_classes(&plan).expect("the qwen plan resolves every merge");

        let kernels = |class: usize| -> (bool, bool, bool) {
            let (mut lse, mut decode, mut prefill) = (false, false, false);
            for (node, mask) in plan.nodes.iter().zip(&classes.node_mask) {
                if !mask.contains(class) {
                    continue;
                }
                match node.op {
                    Operation::Attention(Attention::PrefillLse { .. }) => lse = true,
                    Operation::Attention(Attention::Decode { .. }) => decode = true,
                    Operation::Attention(Attention::Prefill { .. }) => prefill = true,
                    _ => {}
                }
            }
            (lse, decode, prefill)
        };

        // A capturing lane takes the capture arm WHATEVER its row count —
        // that is what putting `captures_scores` first in the three-way split
        // means, and a capturing decode lane is the case that proves it.
        for rows in [1, 8] {
            let word =
                classify(&model::Request::new(rows, false).capturing_scores(true)) & classes.mask;
            let class = classes
                .class_of(word)
                .unwrap_or_else(|| panic!("{platform:?}: the capturing word names no class"));
            assert_eq!(
                kernels(class),
                (true, false, false),
                "{platform:?}: a {rows}-row capturing lane's class must run the \
                 lse attention and only it — anything else is a lane that \
                 asked for an observation and silently did not get one"
            );
        }

        for (rows, want) in [(1, (false, true, false)), (8, (false, false, true))] {
            let word = classify(&model::Request::new(rows, false)) & classes.mask;
            let class = classes
                .class_of(word)
                .unwrap_or_else(|| panic!("{platform:?}: the plain word names no class"));
            assert_eq!(
                kernels(class),
                want,
                "{platform:?}: a {rows}-row lane that captured nothing must run \
                 the plain arm and pay for no second output"
            );
        }
    }
}

/// **NO EXPORT IS CARVED OVER, AND IT IS THE COMPILER THAT BUYS THAT NOW.**
///
/// The name is the one this test was born with (palo C3) and the question has
/// widened under it (palo C3b). Then: `model_compiler::arena`'s delivery tail
/// — liveness to the end of the node list — belonged to the `"out"` seam BY
/// NAME, every other seam's span ended where its producer did, and what kept
/// the draft column safe was that the model text states it LAST. That is a
/// true statement about one statement order and not a property of the
/// artifact, which is why it was pinned and why the pin came with a number:
/// with the draft stated BEFORE the trunk's `lm_head`, qwen36-27b carved
/// 4,236,247,040 bytes — the two `[rows, vocab]` columns sharing an address,
/// 151 MiB over the trunk-only carve, a real latent clobber.
///
/// Now: `model_compiler::arena::EXPORTS` is a SET, every member takes the
/// tail, and the model text's order is still right but no longer load-bearing.
/// So the question this asks is the general one — **every declared export is
/// nobody's co-tenant, in both directions, over every platform** — and the draft
/// pair it was written for is one case of it. The score columns are the case
/// that could not have been asked before: there are sixteen of them in this
/// SKU, they are `[rows, heads]` F32 where the readouts are `[rows, vocab]`
/// bf16, and a carve that gave two of THEM one address would be just as wrong
/// and far easier to miss.
#[test]
fn the_draft_readout_outlives_the_trunk_readout() {
    for sku in [DRAFTING, CAPTURING] {
        for platform in PLATFORMS {
            let plan = plan_of(sku, platform);
            // Every value on every export seam, in the order the plan states
            // them — `EXPORT_SEAMS` is the compiler's own list, so a name that
            // stopped taking the tail stops being asked about here too, which
            // is the coupling being one table rather than two.
            let exports: Vec<(&str, ValueId)> = model_compiler::EXPORT_SEAMS
                .iter()
                .flat_map(|name| {
                    plan.seams
                        .iter()
                        .filter(move |s| s.seam == *name)
                        .flat_map(move |s| s.values.iter().map(move |v| (*name, *v)))
                })
                .collect();
            assert!(
                exports.iter().any(|(name, _)| *name == "out"),
                "`{sku}` as {platform:?}: a plan with no `out` export computes \
                 nothing a reader can take"
            );
            // The two WHOLE-FIRE exports name one value each — one trunk
            // readout, one draft readout — and `exported` panics if either
            // ever names two.
            let out = exported(&plan, "out");
            assert!(exports.contains(&("out", out)));
            if sku == DRAFTING {
                let draft = exported(&plan, "mtp");
                assert!(
                    exports.contains(&("mtp", draft)),
                    "`{sku}` is the SKU whose checkpoint publishes a draft head"
                );
            }

            let baked = compile(&plan, &budgets_for(&plan), &DeviceProfile::default())
                .unwrap_or_else(|why| panic!("`{sku}` as {platform:?}: {}", why.say(&plan)));

            for (i, (a_name, a)) in exports.iter().enumerate() {
                for (b_name, b) in &exports[i + 1..] {
                    assert_ne!(
                        a, b,
                        "`{sku}` as {platform:?}: one value is both the `{a_name}` \
                         export and the `{b_name}` one"
                    );
                    assert!(
                        !baked.arena.co_tenants(*a, *b),
                        "`{sku}` as {platform:?}: the `{a_name}` export (value {}) \
                         and the `{b_name}` export (value {}) share bytes, so \
                         whichever the graph writes second clobbers the other \
                         between the launch and the reader's read",
                        a.0,
                        b.0,
                    );
                }
            }
        }
    }
}

/// **THE ORDER IS NO LONGER LOAD-BEARING, AND HERE IS THE PROOF.**
///
/// The C3 argument for the draft column's safety was "nothing runs after it".
/// If that were still the only thing holding it up, an export stated in the
/// MIDDLE of a plan would be carved over — and the compiler's own delivery
/// tail is what this asks about, without needing a model text that states one
/// badly: the capture columns are stated in every attention layer, sixty
/// nodes deep and a thousand nodes from the end, and they are exports all the
/// same. If the tail were still one name, `attn.scores` would end at its
/// producing node and the next layer's rectangles would be free to take its
/// address.
#[test]
fn a_mid_plan_export_lives_to_the_end_of_the_fire() {
    for platform in PLATFORMS {
        let plan = plan_of(CAPTURING, platform);
        let end = plan.nodes.len() as u32;
        let baked = compile(&plan, &budgets_for(&plan), &DeviceProfile::default())
            .unwrap_or_else(|why| panic!("{platform:?}: {}", why.say(&plan)));

        let mut seen = 0usize;
        for seam in plan.seams.iter().filter(|s| s.seam == "attn.scores") {
            for value in &seam.values {
                seen += 1;
                let root = baked.arena.root(*value);
                let span = baked.arena.spans[root.0 as usize]
                    .unwrap_or_else(|| panic!("{platform:?}: export {} has no span", value.0));
                assert!(
                    span.first < end,
                    "{platform:?}: a capture column stated in a layer is not a \
                     mid-plan value at all — this test would be vacuous"
                );
                assert_eq!(
                    span.last, end,
                    "{platform:?}: the `attn.scores` export at value {} dies at \
                     node {} of {end}, so every rectangle after it may be \
                     carved on its address and the reader gets whatever ran \
                     last",
                    value.0, span.last,
                );
            }
        }
        assert!(
            seen > 1,
            "{platform:?}: `{CAPTURING}` exports {seen} capture column(s); the \
             point of asking here is that there are MANY and they are deep"
        );
    }
}

/// **THE ZERO-COST CLAIM, PINNED.**
///
/// A fire whose lanes set neither new fact must trace and compile to what it
/// traced and compiled before the axes landed. Three numbers say it, and all
/// three are measured at the commit before this wave:
///
/// | SKU | classes | nodes run | arena bytes | + the capture columns |
/// |---|---|---|---|---|
/// | qwen35-a3b | 4 | 1125 | 4102029312 | 4107272192 |
/// | qwen35-d3b | 4 | 485 | 2522873856 | 2526019584 |
/// | qwen35-d0.8b | 4 | 485 | 4085252096 | 4086824960 |
/// | qwen35-a3b-tp2 | 4 | 1205 | 4102029312 | 4104650752 |
///
/// **THE ARENA GREW WHEN THE EXPORT BECAME REACHABLE, AND THE GROWTH IS ONE
/// MULTIPLICATION** (palo C4b). The fourth column is what C3/C4 measured, and
/// it was honest for the artifact C3/C4 produced: nothing read the capture
/// column, so its life ended at the node that wrote it and the busiest-instant
/// carve reused those bytes like every other per-layer value. Making it
/// readable is what changed — `model_compiler::arena`'s delivery tail is now
/// the whole EXPORT SET rather than the `"out"` seam alone, so every `lse`
/// column is held open past the last node, exactly as the trunk's logits are
/// and for exactly the same reason: the reader comes after the graph, and a
/// value carved on top of it would be clobbered between the launch and the
/// read.
///
/// **AND IT IS `layers × max_tokens × heads × 4`, TO THE BYTE.**
///
/// | SKU | attention layers | q heads | Δ bytes | the product |
/// |---|---|---|---|---|
/// | qwen35-a3b | 10 | 16 | 5 242 880 | 10 × 8192 × 16 × 4 |
/// | qwen35-d3b | 6 | 16 | 3 145 728 | 6 × 8192 × 16 × 4 |
/// | qwen35-d0.8b | 6 | 8 | 1 572 864 | 6 × 8192 × 8 × 4 |
/// | qwen35-a3b-tp2 | 10 | 8 | 2 621 440 | 10 × 8192 × 8 × 4 |
///
/// The tp2 row is the check on the reading: it carved IDENTICALLY to the
/// one-rank a3b before, and it now carves 2.5 MiB less, because a rank holds
/// half the heads and the capture column is per head. Nothing else moved — no
/// column shifted, no rectangle grew — which is what says the delta is the new
/// pins and not a placement the tail perturbed. The price is paid at the
/// budget's ceiling by every load of a SKU whose text declares a capture arm,
/// and it is 0.04–0.13% of these carves.
///
/// **NODES RUN, NOT NODES DECLARED.** The capture schedule is a `Cond::Always`
/// prepare node, so it is LIVE in every class; demand narrows it to the
/// classes that read its struct, and the old classes do not (build log 7).
/// That is the difference between 486 and 485, and asking about `node_mask`
/// rather than about `Class::live` is what makes the question the right one.
#[test]
fn the_new_axes_cost_the_old_words_nothing() {
    // (sku, classes at HEAD, nodes run at HEAD, arena bytes at HEAD)
    const BEFORE: [(&str, usize, usize, u64); 4] = [
        ("qwen35-a3b-bf16-kv-bf16", 4, 1125, 4_107_272_192),
        ("qwen35-d3b-bf16-kv-bf16", 4, 485, 2_526_019_584),
        ("qwen35-d0.8b-bf16-kv-bf16", 4, 485, 4_086_824_960),
        ("qwen35-a3b-bf16-kv-bf16-tp2", 4, 1205, 4_104_650_752),
    ];

    for (sku, was_classes, was_nodes, was_arena) in BEFORE {
        for platform in PLATFORMS {
            let plan = plan_of(sku, platform);
            let classes = resolve_classes(&plan).expect("the plan resolves every merge");

            // The classes a lane that set neither new fact can compose as.
            let old: Vec<usize> = (0..classes.classes.len())
                .filter(|c| {
                    classes.classes[*c]
                        .words
                        .iter()
                        .any(|word| word & NEW_BITS == 0)
                })
                .collect();
            assert_eq!(
                old.len(),
                was_classes,
                "`{sku}` as {platform:?}: a word with neither new fact composes as \
                 one of {} classes and used to have {was_classes}; the axes \
                 split behavior that was one behavior",
                old.len(),
            );

            let run: BTreeSet<usize> = plan
                .nodes
                .iter()
                .enumerate()
                .filter(|(i, _)| old.iter().any(|c| classes.node_mask[*i].contains(*c)))
                .map(|(i, _)| i)
                .collect();
            assert_eq!(
                run.len(),
                was_nodes,
                "`{sku}` as {platform:?}: the classes a lane with neither new fact \
                 composes as run {} nodes and used to run {was_nodes}",
                run.len(),
            );

            let baked = compile(&plan, &budgets_for(&plan), &DeviceProfile::default())
                .unwrap_or_else(|why| panic!("`{sku}` as {platform:?}: {}", why.say(&plan)));
            assert_eq!(
                baked.arena.bytes, was_arena,
                "`{sku}` as {platform:?}: the carve is {} bytes and used to be \
                 {was_arena}; the capture columns' delivery tail is a known \
                 `layers × max_tokens × heads × 4` and nothing else may move",
                baked.arena.bytes,
            );
        }
    }
}

/// **NOTHING IS DEAD, AND THE EXPORTS ARE WHY.**
///
/// The draft head hands its logits to nobody: no node reads them, and a
/// backward demand walk with only cache writes and the `"out"` seam as roots
/// would conclude that the whole head — a transformer block and a vocabulary
/// GEMM — computes something nothing reads, and drop it. What roots it is that
/// `model_ir::check::classes` roots EVERY seam value, which is the sentence
/// its own doc writes for this case: "a model that exports a second value gets
/// the same treatment without this file learning a new name". Same for the
/// per-layer score columns.
#[test]
fn every_export_is_demanded_and_nothing_else_is_dead() {
    for sku in [DRAFTING, CAPTURING] {
        for platform in PLATFORMS {
            let plan = plan_of(sku, platform);
            let classes = resolve_classes(&plan).expect("the plan resolves every merge");
            assert!(
                classes.dead.is_empty(),
                "`{sku}` as {platform:?}: {} node(s) are demanded in no class — \
                 first is `{}`",
                classes.dead.len(),
                plan.nodes[classes.dead[0] as usize].op.name(),
            );
        }
    }
}
