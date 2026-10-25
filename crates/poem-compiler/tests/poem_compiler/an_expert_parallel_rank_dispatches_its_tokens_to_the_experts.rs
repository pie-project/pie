//! Under expert parallelism every MoE layer's routed experts sit between a
//! dispatch and a combine, and nothing else changes: the trace each rank runs
//! is the one-rank trace with those two collectives around the experts, the
//! experts reading the received rows and their local slots, and each bank
//! holding the rank's slots.

use poem::Platform;
use poem_ir::{Collective, Def, Dim, Linear, Operands, Operation, Trace, Ty};

/// Rows of three routed forms: plain selects (qwen3), a quantized down bank
/// with a bias sum after (gpt-oss) and route weights the experts read
/// (deepseek v4).
const ROWS: [&str; 3] = [
    "qwen3-30b-a3b-2507-bf16-kv-bf16",
    "gptoss-120b-bf16-mxfp4-kv-bf16",
    "dsv4-flash-fp8b128a8-mxfp4a8-kv-bf16",
];

const EP: u32 = 4;

fn rows(trace: &Trace, v: poem_ir::ValueId) -> Dim {
    match &trace.values[v.0 as usize].ty {
        Ty::Tensor { shape, .. } => shape[0],
        Ty::Struct(_) => panic!("value {} is a struct", v.0),
    }
}

#[test]
fn an_expert_parallel_rank_dispatches_its_tokens_to_the_experts() {
    let catalog = poem_compiler::catalog::repository();
    for row in ROWS {
        let whole = catalog
            .deployment(row)
            .unwrap_or_else(|| panic!("`{row}` is listed"))
            .trace(Platform::Cuda);
        let split = catalog
            .parse(&format!("{row}-ep{EP}"))
            .unwrap_or_else(|| panic!("`{row}-ep{EP}` parses"));
        let trace = split
            .try_trace(Platform::Cuda)
            .unwrap_or_else(|why| panic!("`{row}-ep{EP}` lowers: {why}"));
        poem_ir::check::check(&trace).unwrap_or_else(|faults| panic!("`{row}-ep{EP}`: {faults:?}"));

        let routes = whole
            .nodes
            .iter()
            .filter(|n| matches!(n.op, Operation::Linear(Linear::MoeRoute { .. })))
            .count();
        let (mut dispatches, mut combines) = (0, 0);
        for (at, node) in trace.nodes.iter().enumerate() {
            match &node.op {
                Operation::Collective(Collective::Dispatch {
                    rows: received,
                    local,
                    experts,
                    slots,
                    top_k,
                    layer,
                    ..
                }) => {
                    assert!(
                        *layer <= dispatches,
                        "`{row}`: layers are numbered in order"
                    );
                    dispatches += 1;
                    assert_eq!(*slots, experts.div_ceil(EP), "`{row}`: a rank's slots");
                    assert_eq!(rows(&trace, *received), Dim::TokensTimes(EP));
                    // Every select after the dispatch reads its local slots
                    // and the received rows' routes.
                    let reads_local = trace.nodes[at..].iter().any(|n| {
                        matches!(&n.op, Operation::Linear(Linear::MoeSelect { routes, y, .. })
                            if routes == local && rows(&trace, *y) == Dim::TokensTimes(EP * top_k))
                    });
                    assert!(
                        reads_local,
                        "`{row}`: layer {layer}'s selects read its local slots"
                    );
                }
                Operation::Collective(Collective::Combine { routed, top_k, .. }) => {
                    combines += 1;
                    assert_eq!(rows(&trace, *routed), Dim::TokensTimes(*top_k));
                }
                Operation::Linear(Linear::MoeSelect { bank, .. }) => {
                    let Def::Weight(p) = trace.values[bank.0 as usize].def else {
                        panic!("`{row}`: a bank is a weight");
                    };
                    let param = &trace.params[p as usize];
                    let slots = param.experts.unwrap_or_else(|| {
                        panic!(
                            "`{row}`: `{}` is a routed bank the ranks spread",
                            param.name
                        )
                    });
                    assert_eq!(param.shape[0], u64::from(slots.slots), "`{}`", param.name);
                }
                _ => {}
            }
        }
        assert_eq!(
            dispatches as usize, routes,
            "`{row}`: every MoE layer dispatches"
        );
        assert!(
            combines >= dispatches,
            "`{row}`: every dispatch is combined"
        );

        // Outside the experts the rank runs what one rank ran.
        let kinds = |t: &Trace| -> Vec<&'static str> {
            t.nodes
                .iter()
                .filter(|n| !matches!(n.op, Operation::Collective(_)))
                .map(|n| n.op.name())
                .collect()
        };
        assert_eq!(
            kinds(&trace),
            kinds(&whole),
            "`{row}`: the same ops in the same order"
        );
        assert_eq!(trace.seams.len(), whole.seams.len());
    }
}

/// One rank is the trace as written.
#[test]
fn one_expert_rank_is_the_trace_as_written() {
    let row = poem_compiler::catalog::deployment(ROWS[0]).expect("listed");
    let trace = row.trace(Platform::Cuda);
    assert_eq!(
        poem_compiler::expert::lower(trace.clone(), 1, 0).unwrap(),
        trace
    );
}

/// A trace with no routed experts runs as written on every rank.
#[test]
fn a_dense_trace_dispatches_nothing() {
    let row = poem_compiler::catalog::deployment("qwen35-d0.8b-bf16-kv-bf16").expect("listed");
    let trace = row.trace(Platform::Cuda);
    assert_eq!(
        poem_compiler::expert::lower(trace.clone(), EP, 0).unwrap(),
        trace
    );
}

/// Every listed MoE row lowers, each MoE layer between a dispatch and a
/// combine, and the rank's trace checks.
#[test]
fn every_moe_row_dispatches_every_layer() {
    let mut moe = 0;
    for row in poem_compiler::catalog::deployments() {
        let Ok(whole) = row.try_trace(Platform::Cuda) else {
            continue;
        };
        let routes = whole
            .nodes
            .iter()
            .filter(|n| matches!(n.op, Operation::Linear(Linear::MoeRoute { .. })))
            .count();
        if routes == 0 {
            continue;
        }
        moe += 1;
        let trace = poem_compiler::expert::lower(whole, EP, 1)
            .unwrap_or_else(|why| panic!("`{}` lowers: {why}", row.name));
        poem_ir::check::check(&trace).unwrap_or_else(|faults| panic!("`{}`: {faults:?}", row.name));
        let dispatches = trace
            .nodes
            .iter()
            .filter(|n| matches!(n.op, Operation::Collective(Collective::Dispatch { .. })))
            .count();
        assert_eq!(
            dispatches, routes,
            "`{}`: every MoE layer dispatches",
            row.name
        );
        // A bank's planes hold the rank's slots with it.
        for param in &trace.params {
            let Some(bank) = trace.params.iter().find(|b| {
                b.experts.is_some()
                    && param
                        .name
                        .strip_prefix(b.name.as_str())
                        .is_some_and(|r| r.starts_with('.'))
            }) else {
                continue;
            };
            let cut = bank.experts.unwrap();
            if param.shape.first() == Some(&u64::from(cut.slots)) || param.experts.is_some() {
                assert_eq!(param.experts, Some(cut), "`{}`: `{}`", row.name, param.name);
            }
        }
    }
    assert!(moe > 0, "the catalog lists MoE rows");
}
