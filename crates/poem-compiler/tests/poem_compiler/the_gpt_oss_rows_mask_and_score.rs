use poem::{Attn, Operation, Platform};

const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

fn count(trace: &poem::Trace, wanted: impl Fn(&Operation) -> bool) -> usize {
    trace.nodes.iter().filter(|node| wanted(&node.op)).count()
}

// A gpt-oss attention layer folds its sink into every arm, so each arm
// writes the log-sum-exp the sink reads. A lane's mask is data its arm
// reads, never a kind of lane (`Attn::Decode`'s and `Attn::Prefill`'s
// `mask`): every layer has two sunk arms, decode and prefill, and both read
// the fire's masks, a lane stating none reading every key in the same body.
// The full-attention layers (the odd ones) capture their prefill scores for
// the eviction programs, each adding into the row the layer before filled,
// and the trace reads that row out once (`attn.scores`, `attn.scores.pages`).
#[test]
fn every_gpt_oss_row_declares_a_masked_arm_and_exports_its_scores() {
    let rows: Vec<_> = poem_compiler::catalog::deployments()
        .filter(|row| row.name.starts_with("gptoss-"))
        .collect();
    assert!(!rows.is_empty(), "this build ships no gpt-oss row");
    for row in rows {
        for platform in PLATFORMS {
            let trace = row.trace(platform);
            let sinks = count(&trace, |op| {
                matches!(op, Operation::Attn(Attn::Sink { .. }))
            });
            let decode = count(&trace, |op| {
                matches!(
                    op,
                    Operation::Attn(Attn::Decode {
                        mask: Some(_),
                        selection: None,
                        bias: None,
                        lse: Some(_),
                        ..
                    })
                )
            });
            let prefill = count(&trace, |op| {
                matches!(
                    op,
                    Operation::Attn(Attn::Prefill {
                        mask: Some(_),
                        selection: None,
                        bias: None,
                        lse: Some(_),
                        ..
                    })
                )
            });
            assert!(
                prefill > 0 && decode == prefill && sinks == 2 * prefill,
                "{} on {platform:?}: {decode} decode and {prefill} prefill arm(s) reading \
                 the masks with their log-sum-exp, against {sinks} sink(s); every layer \
                 has two sunk arms (decode, prefill), both masked",
                row.name
            );
            let captures = count(&trace, |op| {
                matches!(op, Operation::Attn(Attn::ScoreCapture { .. }))
            });
            assert_eq!(
                captures,
                prefill / 2,
                "{} on {platform:?}: {captures} score capture(s) for {prefill} layer(s); \
                 every full-attention layer (the odd ones) adds its scores to the row",
                row.name
            );
            for name in ["attn.scores", "attn.scores.pages"] {
                let read = trace.outputs.iter().filter(|s| s.name == name).count();
                assert_eq!(
                    read, 1,
                    "{} on {platform:?}: the summed row is read out once as `{name}`",
                    row.name
                );
            }
        }
    }
}
