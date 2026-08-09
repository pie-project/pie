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
//! Having the mechanism is not the same as using it. **A text has to state
//! guards on those axes**, and this measures whether the Metal one does.
//!
//! Today it does not, and that is what the test pins. Every launch covers the
//! whole fire whatever the rows say — including row sets the CUDA lowering
//! outright refuses as `Discontiguous`, which is the seriation contract a text
//! that DOES split on an axis imposes. So `llama_like`'s Metal text is
//! monomorphic: correct for one program, and not yet the thing the north star
//! names.
//!
//! When the text gains its guards this test fails, and that failure is the
//! signal to rewrite it against the new expectation.

use std::collections::BTreeSet;

use model::families::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::families::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Fire, Row, lower};
use model_compiler::trace::FireClass;

/// Rows that are structurally different programs: half truncated at layer 4,
/// a quarter carrying a LoRA.
fn mixed(n: usize) -> Vec<Row> {
    (0..n)
        .map(|i| Row {
            samples: true,
            depth_k: if i < n / 2 { Some(4) } else { None },
            lora: i < n / 4,
            ..Row::default()
        })
        .collect()
}

#[test]
fn the_metal_text_is_monomorphic_today_and_this_is_the_measurement() {
    for (class, n) in [(FireClass::Decode, 4usize), (FireClass::Prefill, 8usize)] {
        let plan = llama_like_metal(
            &LlamaLikeFacts::qwen3_0_6b(),
            &LlamaLikeMetalFacts::synthetic(),
            class,
        );
        let low = lower(&plan, &mixed(n), Fire {
            captures_across_splits: false,
        })
        .expect("the metal text lowers whatever the rows say — which is the finding");

        let ranges: BTreeSet<(u32, u32)> = low
            .launches
            .iter()
            .map(|l| (l.rows.start, l.rows.end))
            .collect();

        assert_eq!(
            ranges.len(),
            1,
            "{class:?}: the text produced {} distinct row ranges. If that is \
             deliberate, the text has gained polymorphism and this test should \
             be rewritten to assert WHICH axes split.",
            ranges.len()
        );
        assert_eq!(
            ranges.iter().next().copied(),
            Some((0, n as u32)),
            "{class:?}: the single range covers the whole fire"
        );
        assert!(
            low.launches.iter().all(|l| l.peel.is_none()),
            "{class:?}: a peel region appeared, which is a row split"
        );
        assert_eq!(
            low.rectangles,
            low.launches.len(),
            "{class:?}: rectangles and launches agree while nothing splits"
        );
    }
}
