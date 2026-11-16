//! `attn.scores` states how each layer's row factors: a run of `kv_max` keys
//! per query head ([`poem_ir::Output::planes`]), so an eviction program
//! folds over the heads the model has. A guest that guessed the count
//! before misread a GQA model: snapkv, h2o and tova took Qwen3-0.6B's
//! 16x2048 row as 8x4096, dropping half the heads below 2048 keys and
//! scoring keys 2048..4095 with another head's first keys.

use poem::{Attn, Operation, Platform};

/// Qwen3-0.6B: 16 query heads over 8 kv heads; Qwen3.5-0.8B's full
/// attention layers.
const ROWS: [&str; 2] = ["qwen3-0.6b-bf16-kv-bf16", "qwen35-d0.8b-bf16-kv-bf16"];

#[test]
fn a_score_output_states_its_heads_and_keys() {
    let mut gqa = false;
    for name in ROWS {
        let row = poem_compiler::catalog::deployment(name)
            .unwrap_or_else(|| panic!("the catalog lists {name}"));
        let trace = row.trace(Platform::Metal);
        let outputs: Vec<_> = trace
            .outputs
            .iter()
            .filter(|output| output.name == "attn.scores")
            .collect();
        assert!(!outputs.is_empty(), "{name} reads out no attention scores");
        for output in outputs {
            let (kv_heads, kv_max) = trace
                .nodes
                .iter()
                .find_map(|node| match &node.op {
                    Operation::Attn(Attn::ScoreCapture {
                        scores,
                        kv_heads,
                        kv_max,
                        ..
                    }) if *scores == output.value => Some((*kv_heads, *kv_max)),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("{name}: attn.scores is no score capture's"));
            let width = match &trace.values[output.value.0 as usize].ty {
                poem::Ty::Tensor { shape, .. } => match shape.last() {
                    Some(poem::Dim::Const(width)) => u32::try_from(*width).expect("a width"),
                    other => panic!("{name}: the score row is {other:?} wide"),
                },
                other => panic!("{name}: the score output is {other:?}"),
            };
            let q_heads = width / kv_max;
            assert_eq!(
                output.planes,
                vec![q_heads, kv_max],
                "{name}: attn.scores states its {width}-wide row as {:?}, not [{q_heads} query \
                 heads, {kv_max} keys]",
                output.planes
            );
            gqa |= q_heads != kv_heads;
        }
    }
    assert!(
        gqa,
        "no row has more query heads than kv heads, so a guest folding over kv heads would pass"
    );
}
