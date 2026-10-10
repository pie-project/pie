use poem::{Attention, BlockDrafter, Dtype, Layout, Operation, Platform, Trace};

const PLATFORMS: [Platform; 2] = [Platform::Metal, Platform::Cuda];

fn row(prefix: &str) -> &'static poem_compiler::catalog::Deployment {
    poem_compiler::catalog::deployments()
        .find(|row| row.name.starts_with(prefix))
        .unwrap_or_else(|| panic!("this build ships no `{prefix}*` row"))
}

fn count(trace: &Trace, op: impl Fn(&Operation) -> bool) -> usize {
    trace.nodes.iter().filter(|n| op(&n.op)).count()
}

fn non_causal(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Attention(Attention::Masked { causal: false, .. })
    )
}

fn block_dyn_conv(op: &Operation) -> bool {
    matches!(op, Operation::Attention(Attention::BlockDynConv { .. }))
}

fn top_k(op: &Operation) -> bool {
    matches!(op, Operation::Layout(Layout::TopK { .. }))
}

#[test]
fn the_block_drafters_state_their_facts_every_case() {
    every_drafter_text_states_its_block();
    a_plain_row_states_no_drafter();
    the_dflash2_plan_convolves_and_reads_its_block_out_once();
    the_dspark_plan_walks_a_bigram_from_the_anchor();
    a_drafter_head_reads_its_block_both_ways();
}

fn every_drafter_text_states_its_block() {
    let drafter = |rows, mask_token, bidirectional, proposals_from| BlockDrafter {
        rows,
        mask_token,
        bidirectional,
        proposals_from,
    };
    let table = [
        ("qwen36-27b-dflash-", drafter(16, 248_070, true, 1)),
        ("qwen36-35b-a3b-dflash-", drafter(16, 248_077, true, 1)),
        ("qwen38-27b-dflash2-", drafter(8, 248_070, false, 1)),
        ("qwen38-27b-dspark-", drafter(15, 248_200, true, 0)),
        ("gemma4-26b-a4b-dflash-", drafter(16, 4, true, 1)),
        ("gptoss-20b-dflash-", drafter(8, 200_000, true, 1)),
    ];
    for (prefix, want) in table {
        for platform in PLATFORMS {
            let trace = row(prefix).trace(platform);
            assert_eq!(trace.drafter, Some(want), "{prefix}* on {platform:?}");
            let seams: Vec<&str> = trace.seams.iter().map(|s| s.seam.as_str()).collect();
            assert!(
                seams.iter().any(|s| s.contains("mtp")),
                "{prefix}* on {platform:?} plants no draft seam; seams are {seams:?}"
            );
        }
    }
}

fn a_plain_row_states_no_drafter() {
    for (id, weights) in [("qwen38-27b", Some(Dtype::U4g64)), ("gemma4-26b-a4b", None)] {
        let row = poem_compiler::catalog::deployments()
            .find(|row| {
                row.model.id == id
                    && row.deploy.drafter.is_none()
                    && row.deploy.parts.is_empty()
                    && weights.is_none_or(|w| row.deploy.weights.contains(&w))
            })
            .unwrap_or_else(|| panic!("this build ships no plain `{id}` row"));
        assert_eq!(row.trace(Platform::Metal).drafter, None, "{}", row.name);
    }
}

fn the_dflash2_plan_convolves_and_reads_its_block_out_once() {
    for platform in PLATFORMS {
        let trace = row("qwen38-27b-dflash2-").trace(platform);
        assert_eq!(
            count(&trace, block_dyn_conv),
            20,
            "{platform:?}: five blocks x two sublayers x two sides"
        );
        let walks = count(&trace, |op| {
            matches!(op, Operation::Attention(Attention::SelectorWalk { .. }))
        });
        assert_eq!(
            (count(&trace, top_k), walks),
            (1, 1),
            "{platform:?}: the selector reads the block out once"
        );
    }
}

fn the_dspark_plan_walks_a_bigram_from_the_anchor() {
    let trace = row("qwen38-27b-dspark-").trace(Platform::Metal);
    assert_eq!(count(&trace, block_dyn_conv), 0);
    assert_eq!(count(&trace, top_k), 1);
    let walks: Vec<_> = trace
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            Operation::Attention(Attention::SelectorWalk { hp, first, .. }) => Some((*hp, *first)),
            _ => None,
        })
        .collect();
    assert_eq!(
        walks,
        [(None, 0)],
        "a bigram lattice walked from the anchor row"
    );
}

fn a_drafter_head_reads_its_block_both_ways() {
    for (prefix, full, why) in [
        (
            "gemma4-26b-a4b-dflash-",
            1,
            "the head's full layer is the one non-causal read",
        ),
        (
            "gptoss-20b-dflash-",
            8,
            "every layer of this head is full attention over the block",
        ),
    ] {
        let trace = row(prefix).trace(Platform::Metal);
        assert_eq!(count(&trace, non_causal), full, "{prefix}*: {why}");
    }
}
