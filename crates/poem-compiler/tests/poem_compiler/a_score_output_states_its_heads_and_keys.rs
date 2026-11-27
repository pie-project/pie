//! `attn.scores` is one row a capturing lane: the attention mass a key of
//! the load's context, summed over the layers and averaged over every query
//! head (`[Dim::Capturing, Dim::ContextTimes(1)]`); `attn.scores.pages` is
//! that row summed a kv page (`[Dim::Capturing, Dim::ContextPages]`). No
//! arena holds either, so a fire pays for the lanes that capture, and the
//! guest folds nothing: the eviction examples summed every plane of 28
//! per-layer, per-head rows, and guessed the head count to do it (misreading
//! Qwen3-0.6B's 16x2048 row as 8x4096).

use poem::{Attn, Dim, Operation, Platform, Ty};

const ROWS: [&str; 2] = ["qwen3-0.6b-bf16-kv-bf16", "qwen35-d0.8b-bf16-kv-bf16"];

fn shape_of(trace: &poem::Trace, name: &str) -> Vec<Dim> {
    let outputs: Vec<_> = trace
        .outputs
        .iter()
        .filter(|output| output.name == name)
        .collect();
    assert_eq!(outputs.len(), 1, "one `{name}` output, not one a layer");
    assert_eq!(
        outputs[0].layer, None,
        "`{name}` is read out once, after the layers"
    );
    match &trace.values[outputs[0].value.0 as usize].ty {
        Ty::Tensor { shape, .. } => shape.clone(),
        other => panic!("`{name}` is {other:?}"),
    }
}

#[test]
fn a_score_output_states_its_heads_and_keys() {
    for name in ROWS {
        let row = poem_compiler::catalog::deployment(name)
            .unwrap_or_else(|| panic!("the catalog lists {name}"));
        let trace = row.trace(Platform::Metal);
        assert_eq!(
            shape_of(&trace, "attn.scores"),
            [Dim::Capturing, Dim::ContextTimes(1)],
            "{name}: a mass a key of the context, one row a capturing lane"
        );
        assert_eq!(
            shape_of(&trace, "attn.scores.pages"),
            [Dim::Capturing, Dim::ContextPages],
            "{name}: a mass a page of the context, one row a capturing lane"
        );
        // Each attention layer adds into the row the one before it filled.
        let captures: Vec<_> = trace
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Attn(Attn::ScoreCapture { into, scores, .. }) => Some((*into, *scores)),
                _ => None,
            })
            .collect();
        assert!(!captures.is_empty(), "{name} captures no scores");
        assert_eq!(
            captures[0].0, None,
            "{name}: the first layer's capture zeroes the row"
        );
        for pair in captures.windows(2) {
            assert_eq!(
                pair[1].0,
                Some(pair[0].1),
                "{name}: a layer's capture adds into the layer's before it"
            );
        }
    }
}
