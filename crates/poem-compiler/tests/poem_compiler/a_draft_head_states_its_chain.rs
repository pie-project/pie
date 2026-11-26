use poem::Platform;

fn mtp_row(id: &str) -> &'static poem_compiler::catalog::Deployment {
    poem_compiler::catalog::deployments()
        .find(|row| {
            row.model.id == id
                && row.deploy.drafter.as_deref() == Some("mtp")
                && row.deploy.parts.is_empty()
        })
        .unwrap_or_else(|| panic!("this build ships no `{id}` mtp row"))
}

#[test]
fn a_draft_head_states_its_chain_every_case() {
    one_unchained_layer_drafts_one();
    a_chained_head_drafts_its_chain();
    every_head_output_states_a_depth();
}

/// Qwen3.5's head is one MTP layer, not chained: its `mtp.drafts` holds four
/// copies of the one proposal (`models/qwen_3.poem`), and the depth a guest
/// reads is 1.
fn one_unchained_layer_drafts_one() {
    for platform in [Platform::Cuda, Platform::Metal] {
        let trace = mtp_row("qwen35-d0.8b").trace(platform);
        assert_eq!(trace.facts.draft_depth, Some(1), "{platform:?}");
    }
}

/// Qwen 3.8's head feeds each draft back in: its chain is its depth.
fn a_chained_head_drafts_its_chain() {
    let trace = mtp_row("qwen38-flash-next").trace(Platform::Cuda);
    assert_eq!(trace.facts.draft_depth, Some(2));
}

/// A text that reads `mtp.drafts` out from a head states the head's depth;
/// a block drafter's block is its own fact, and nothing states a depth for
/// an output it does not have.
fn every_head_output_states_a_depth() {
    let mut heads = 0;
    for row in poem_compiler::catalog::deployments() {
        let Ok(trace) = row.try_trace(Platform::Cuda) else {
            continue;
        };
        let drafts = trace.outputs.iter().find(|o| o.name == "mtp.drafts");
        match (drafts, trace.drafter, trace.facts.draft_depth) {
            (Some(_), None, Some(depth)) => {
                assert!(depth >= 1, "{}", row.name);
                heads += 1;
            }
            (Some(_), None, None) => panic!("`{}` reads drafts out and states no depth", row.name),
            (None, _, Some(depth)) => {
                panic!("`{}` states depth {depth} and drafts nothing", row.name)
            }
            _ => {}
        }
    }
    assert!(heads > 0, "the catalog ships draft heads");
}
