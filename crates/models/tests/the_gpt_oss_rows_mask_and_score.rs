use model_dsl::{Attention, Operation, Platform, seam};

const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

fn count(trace: &model_dsl::Trace, wanted: impl Fn(&Operation) -> bool) -> usize {
    trace.nodes.iter().filter(|node| wanted(&node.op)).count()
}

// A gpt-oss attention layer folds its sink into every arm, so the masked arm
// needs the log-sum-exp the plain `attention.masked` never returns; this is
// why the family went without one. Each layer now reads `attention.masked_lse`
// once, and the full-attention layers export their prefill scores for the
// eviction programs (`attn_score`).
#[test]
fn every_gpt_oss_row_declares_a_masked_arm_and_exports_its_scores() {
    let rows: Vec<_> = models::skus()
        .filter(|row| row.name.starts_with("gptoss-"))
        .collect();
    assert!(!rows.is_empty(), "this build ships no gpt-oss row");
    for row in rows {
        for platform in PLATFORMS {
            let trace = (row.trace)(platform);
            let sinks = count(&trace, |op| {
                matches!(op, Operation::Attention(Attention::Sink { .. }))
            });
            let masked = count(&trace, |op| {
                matches!(
                    op,
                    Operation::Attention(Attention::MaskedLse { causal: true, .. })
                )
            });
            assert!(
                masked > 0 && sinks == 4 * masked,
                "{} on {platform:?}: {masked} causal `attention.masked_lse` arm(s) against \
                 {sinks} sink(s); every layer has four sunk arms (masked, scored, decode, \
                 prefill)",
                row.name
            );
            let scores = trace
                .seams
                .iter()
                .filter(|s| s.seam == seam::SCORES.name)
                .count();
            assert_eq!(
                scores,
                masked / 2,
                "{} on {platform:?}: {scores} score seam(s) for {masked} layer(s); every \
                 full-attention layer (the odd ones) exports one",
                row.name
            );
        }
    }
}
