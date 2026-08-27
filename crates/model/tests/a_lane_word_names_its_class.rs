//! The catalog's fourth column, held against the plan the third one traces.
//!
//! A lane's word is the model's own `Classify::of(&Request)`, packed; a class
//! is what the driver composes a fire out of, and a class is a set of words.
//! The two are written in different places — `Facts::word` in a family's
//! `forward.rs`, `resolve_classes` in the IR — and NOTHING MAKES THEM AGREE
//! except that they both read the same bit positions. This is the test that
//! says they do.
//!
//! Uncaught, a disagreement here is not a compile error and not a refusal: it
//! is a decode lane composed as a prefill one (`palo B-word`, which is exactly
//! what the engine shipped while the column did not exist), which runs, and
//! returns a plausible token computed by the wrong kernel over the wrong rows.

use model_dsl::{Attention, Operation, Platform, resolve_classes};

/// Every platform a plan can be traced at — a model text may emit a different
/// op per platform, so a class table is per platform too.
const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

/// The row the assertion below is written against: one qwen SKU, whose
/// `Facts` is the one bit `qo_one` and whose attention splits on it.
const QWEN: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// A one-token request and a many-token one land in DIFFERENT classes, and
/// each class is the one that runs the attention kernel for its shape.
///
/// The cross-check is deliberately not "bit 0 is `qo_one`" — that would be the
/// classifier restated, and a test that restates its subject cannot catch it
/// being wrong. It goes through the plan instead: the class the one-token word
/// belongs to must run `Attention::Decode` and NOT `Attention::Prefill`, and
/// the many-token word's class the other way round. That holds only if the bit
/// `Facts::word` sets is the bit `Facts::qo_one()`'s guard reads.
#[test]
fn a_one_token_lane_lands_in_the_decode_class() {
    let trace = model::trace_of(QWEN).expect("this build ships the qwen row");
    let classify = model::classify_of(QWEN).expect("and its classifier");

    for platform in PLATFORMS {
        let plan = trace(platform);
        let classes = resolve_classes(&plan).expect("the qwen plan resolves every merge");

        // The sweep enumerates only the bits some GUARD reads; the word packs
        // every fact the model computes. Masking is what keeps a fact the plan
        // does not split on from reading as an unknown class.
        let decode_word = classify(&model::Request::new(1, false)) & classes.mask;
        let prefill_word = classify(&model::Request::new(8, false)) & classes.mask;
        assert_ne!(
            decode_word, prefill_word,
            "{platform:?}: a one-token lane and an eight-token lane are the same \
             word, so `qo_one` is not a fact this classifier computes"
        );

        let decode = classes
            .class_of(decode_word)
            .unwrap_or_else(|| panic!("{platform:?}: word {decode_word} names no class"));
        let prefill = classes
            .class_of(prefill_word)
            .unwrap_or_else(|| panic!("{platform:?}: word {prefill_word} names no class"));
        assert_ne!(
            decode, prefill,
            "{platform:?}: both shapes compose as one class, so every fire runs \
             one attention kernel over both"
        );

        // Which attention kernels a class runs: `(decode, prefill)`. A
        // closure rather than a function because naming the sweep's own types
        // would widen `model_dsl`'s re-export door for one assertion.
        let kernels = |class: usize| -> (bool, bool) {
            let (mut decode, mut prefill) = (false, false);
            for (node, mask) in plan.nodes.iter().zip(&classes.node_mask) {
                if !mask.contains(class) {
                    continue;
                }
                match node.op {
                    Operation::Attention(Attention::Decode { .. }) => decode = true,
                    Operation::Attention(Attention::Prefill { .. }) => prefill = true,
                    _ => {}
                }
            }
            (decode, prefill)
        };

        assert_eq!(
            kernels(decode),
            (true, false),
            "{platform:?}: the class a one-token lane composes as must run the \
             decode attention and only it"
        );
        assert_eq!(
            kernels(prefill),
            (false, true),
            "{platform:?}: the class a multi-token lane composes as must run the \
             prefill attention and only it"
        );
    }
}

/// Every word every SKU's classifier can produce names a class of that SKU's
/// own plan.
///
/// A word with no class is the refusal `driver_api::fire::Lane` describes —
/// "the engine and the shell disagree about what is loaded" — and it is a fact
/// about a build, not about a deployment, so it is settled here rather than at
/// the first fire.
#[test]
fn every_sku_classifies_into_its_own_plan() {
    let mut faults = Vec::new();

    for (sku, _, trace, classify) in model::catalog() {
        for platform in PLATFORMS {
            let plan = trace(platform);
            let Ok(classes) = resolve_classes(&plan) else {
                continue; // `every_class_resolves_every_merge` is what says so.
            };
            // The two shapes every fire is made of, and every axis fact
            // beside them. **THE SWEEP IS THE PRODUCT AND NOT A SAMPLE**: a
            // lane may set any combination — a masked drafting lane routed to
            // an adapter is a legal submission — and the class table has to
            // name every one of them. Thirty-two requests cover every bit any
            // shipped `Facts` reads, which is what keeps a new axis from
            // arriving with a corner nobody asked about.
            for rows in [1, 8] {
                for masked in [false, true] {
                    for adapted in [false, true] {
                        for drafts in [false, true] {
                            for scores in [false, true] {
                                let r = model::Request::new(rows, masked)
                                    .adapted(adapted)
                                    .drafting(drafts)
                                    .capturing_scores(scores);
                                let word = classify(&r) & classes.mask;
                                if classes.class_of(word).is_none() {
                                    faults.push(format!(
                                        "`{sku}` as {platform:?}: a {rows}-row lane \
                                         with masked={masked} adapted={adapted} \
                                         drafts={drafts} scores={scores} \
                                         classifies to word {word}, which names \
                                         no class of its own plan"
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
