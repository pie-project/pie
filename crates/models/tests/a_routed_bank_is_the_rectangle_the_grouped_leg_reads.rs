use model_dsl::{Def, Dim, Linear, Operation, Platform, Trace, Ty, ValueId};

fn width(trace: &Trace, v: ValueId) -> Option<u64> {
    match &trace.values[v.0 as usize].ty {
        Ty::Tensor { shape, .. } => match shape.last() {
            Some(Dim::Const(w)) => Some(*w),
            _ => None,
        },
        _ => None,
    }
}

fn declared(trace: &Trace, v: ValueId) -> Option<Vec<u64>> {
    match trace.values[v.0 as usize].def {
        Def::Weight(at) => Some(trace.params[at as usize].shape.clone()),
        _ => None,
    }
}

#[test]
fn a_routed_bank_is_the_rectangle_the_grouped_leg_reads() {
    let mut faults = Vec::new();
    let mut checked = 0usize;

    for row in models::skus() {
        let trace = (row.trace)(Platform::Cuda);
        for node in &trace.nodes {
            let Operation::Linear(Linear::MoeMatmulSelect { x, bank, y, .. }) = &node.op else {
                continue;
            };
            let (Some(bank_shape), Some(k), Some(n)) = (
                declared(&trace, *bank),
                width(&trace, *x),
                width(&trace, *y),
            ) else {
                faults.push(format!(
                    "`{}`: a routed select whose bank is not a declared weight, or whose \
                     activation and result are not constant-width rectangles — the grouped \
                     leg reads its expert count off exactly those three",
                    row.name,
                ));
                continue;
            };
            checked += 1;
            match bank_shape.as_slice() {
                [experts, bank_n, bank_k] => {
                    if *bank_n != n || *bank_k != k {
                        faults.push(format!(
                            "`{}`: a routed bank of {bank_shape:?} against a {k}-wide \
                             activation and a {n}-wide result. The grouped leg wants \
                             `[experts, {n}, {k}]`, which reaches it as rows of \
                             {}; this one falls back to the GEMV and the family \
                             keeps the per-route read",
                            row.name,
                            n * k,
                        ));
                    } else if *experts == 0 {
                        faults.push(format!("`{}`: a routed bank of no experts", row.name));
                    }
                }
                other => faults.push(format!(
                    "`{}`: a routed bank declared {other:?}, not the three-axis \
                     `[experts, N, K]` the grouped leg divides",
                    row.name,
                )),
            }
        }
    }

    assert!(
        checked > 0,
        "no catalog SKU traces a routed select, so this gate proved nothing — either the \
         MoE families left the catalog or the op was renamed"
    );
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
