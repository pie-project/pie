use model_dsl::{Collective, Def, Linear, Operation, Platform, Shard, Trace, ValueId};

fn banded(trace: &Trace, w: ValueId) -> bool {
    match trace.values[w.0 as usize].def {
        Def::Weight(at) => matches!(trace.params[at as usize].shard, Shard::Cut { axis: 0, .. }),
        _ => false,
    }
}

#[test]
fn a_banded_head_gathers_the_logits_it_holds_a_band_of_every_case() {
    a_banded_head_gathers_the_logits_it_holds_a_band_of();
    a_single_rank_bands_nothing_and_gathers_nothing();
}

fn a_banded_head_gathers_the_logits_it_holds_a_band_of() {
    let mut faults = Vec::new();

    for row in models::skus() {
        let trace = (row.trace)(Platform::Cuda);

        let gathered: Vec<ValueId> = trace
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Collective(Collective::AllGather { x, .. }) => Some(*x),
                _ => None,
            })
            .collect();

        for node in &trace.nodes {
            let Operation::Linear(Linear::LmHead { w, y, .. }) = &node.op else {
                continue;
            };
            if banded(&trace, *w) && !gathered.contains(y) {
                faults.push(format!(
                    "`{}` reads out through a vocab-banded head and nothing gathers \
                     the result: this rank lands only its columns of the logits, so \
                     the readout owes an `all_gather` before anything reads them",
                    row.name,
                ));
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

fn a_single_rank_bands_nothing_and_gathers_nothing() {
    let mut faults = Vec::new();

    for row in models::skus() {
        if row.recipe.tp > 1 {
            continue;
        }
        let trace = (row.trace)(Platform::Cuda);

        if let Some(cut) = trace
            .params
            .iter()
            .find(|p| matches!(p.shard, Shard::Cut { axis: 0, .. }) && p.name.contains("head"))
        {
            faults.push(format!(
                "`{}` ships one rank and cuts `{}` on the vocabulary axis; there is \
                 no second rank to hold the other band",
                row.name, cut.name,
            ));
        }

        let gathers = trace
            .nodes
            .iter()
            .filter(|node| {
                matches!(
                    &node.op,
                    Operation::Collective(Collective::AllGather { .. })
                )
            })
            .count();
        if gathers != 0 {
            faults.push(format!(
                "`{}` ships one rank and gathers {gathers} time(s); a gather over a \
                 world of one is a copy nobody asked for",
                row.name,
            ));
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
