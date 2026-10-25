//! A hyper-connection sublayer's fold and the next sublayer's whole pre-mix
//! over the streams it folds (`fused.hc_fold_pre`) land as one op between
//! DeepSeek-V4's sublayers, and nowhere else: not over the first sublayer's
//! expanded streams, not at the last fold the head collapses, not where an
//! Engram writes into the folded streams, and not under V4.1's Single-Pass
//! mHC, whose normed row is the previous sublayer's mixes' pre-mix.

use poem::Value;
use poem::ops::elemwise;
use poem::pattern::{Pattern, template};
use poem_compiler::fuse::rules;
use poem_ir::{Operands, Trace};

/// The hyper-connection rules CUDA forms.
const HC: [&str; 4] = [
    "fused.hc_fold_pre",
    "fused.hc_pre",
    "fused.hc_prenorm_project",
    "fused.hc_gates_norm",
];

fn fuse(trace: Trace) -> Trace {
    poem_compiler::fuse::fuse(trace, &rules::forming(&HC))
}

fn ops(trace: &Trace) -> Vec<&'static str> {
    trace.nodes.iter().map(|node| node.op.name()).collect()
}

fn count(trace: &Trace, name: &str) -> usize {
    trace
        .nodes
        .iter()
        .filter(|node| node.op.name() == name)
        .count()
}

/// A sublayer's output folded into `streams` by the previous sublayer's
/// post and combine mixes.
fn fold(p: &Pattern, streams: &Value) -> Value {
    elemwise::hc_fold(
        &p.rows("o", 64),
        streams,
        &p.rows("post_in", 4),
        &p.rows("comb_in", 16),
    )
}

/// The mixes of `streams`' norm through the projection `plane`.
fn mixes(p: &Pattern, streams: &Value, plane: &'static str) -> Value {
    let normed = elemwise::hc_rmsnorm_f32(streams, 1e-6);
    elemwise::hc_project(&normed, &p.weight(plane, [24, 256]), 4)
}

/// `mixes`' gates over `streams` through the mix `[scale, base]`: the
/// pre-mixed row and the post and combine mixes.
fn gates(
    p: &Pattern,
    mixes: &Value,
    streams: &Value,
    [scale, base]: [&'static str; 2],
) -> (Value, Value, Value) {
    elemwise::hc_gates(
        mixes,
        streams,
        &p.weight(scale, [3]),
        &p.weight(base, [24]),
        4,
        1e-6,
        2.0,
        20,
    )
}

/// The next sublayer's input over `streams` (its pre-mixed row normed), and
/// its output folded back in.
fn sublayer(p: &Pattern, streams: &Value) -> Value {
    let (x, post, comb) = gates(p, &mixes(p, streams, "plane"), streams, ["scale", "base"]);
    let y = elemwise::rmsnorm(&x, &p.weight("norm", [64]), 1e-6);
    elemwise::silu(&y);
    elemwise::hc_fold(&y, streams, &post, &comb)
}

#[test]
fn a_fold_and_the_next_pre_mix_land_as_one_op() {
    let mid = template(|p| {
        let streams = fold(p, &p.rows("streams", 256));
        sublayer(p, &streams);
    })
    .trace;
    assert_eq!(
        ops(&fuse(mid)),
        ["fused.hc_fold_pre", "elemwise.act", "elemwise.hc_fold"]
    );

    // The first sublayer's streams are the embedding's, expanded: no fold
    // before its pre-mix.
    let first = template(|p| {
        let streams = elemwise::hc_expand(&p.rows("embedded", 64), 4);
        sublayer(p, &streams);
    })
    .trace;
    assert_eq!(
        ops(&fuse(first)),
        [
            "elemwise.hc_expand",
            "fused.hc_pre",
            "elemwise.act",
            "elemwise.hc_fold"
        ]
    );

    // The last fold's streams collapse into one row: projected, not gated.
    let last = template(|p| {
        let streams = fold(p, &p.rows("streams", 256));
        elemwise::hc_collapse(
            &mixes(p, &streams, "head"),
            &streams,
            &p.weight("head_scale", [1]),
            &p.weight("head_base", [4]),
            4,
            1e-6,
        );
    })
    .trace;
    assert_eq!(
        ops(&fuse(last)),
        [
            "elemwise.hc_fold",
            "fused.hc_prenorm_project",
            "elemwise.hc_collapse"
        ]
    );

    // An Engram layer writes its memories into the folded streams before
    // the next pre-mix norms them.
    let engram = template(|p| {
        let folded = fold(p, &p.rows("streams", 256));
        let streams = elemwise::residual_add(&p.rows("gated", 256), &folded);
        sublayer(p, &streams);
    })
    .trace;
    assert_eq!(
        ops(&fuse(engram)),
        [
            "elemwise.hc_fold",
            "elemwise.residual_add",
            "fused.hc_pre",
            "elemwise.act",
            "elemwise.hc_fold"
        ]
    );

    // Single-Pass mHC: the folded streams' own mixes gate only the post and
    // combine mixes, and the row the next sublayer reads is the previous
    // sublayer's mixes' pre-mix of them.
    let shifted = template(|p| {
        let streams = fold(p, &p.rows("streams", 256));
        let own = mixes(p, &streams, "plane");
        let (_, post, comb) = gates(p, &own, &streams, ["scale", "base"]);
        let prev = p.value(
            "prev",
            poem_ir::Ty::Tensor {
                shape: vec![poem_ir::Dim::Tokens, poem_ir::Dim::Const(24)],
                dtype: poem_ir::Dtype::F32,
            },
        );
        let (x, _, _) = gates(p, &prev, &streams, ["prev_scale", "prev_base"]);
        let y = elemwise::rmsnorm(&x, &p.weight("norm", [64]), 1e-6);
        elemwise::silu(&y);
        elemwise::hc_fold(&y, &streams, &post, &comb);
    })
    .trace;
    let kept = ops(&fuse(shifted));
    assert!(!kept.contains(&"fused.hc_fold_pre"), "{kept:?}");
    assert_eq!(count_of(&kept, "elemwise.hc_fold"), 2, "{kept:?}");
}

fn count_of(ops: &[&str], name: &str) -> usize {
    ops.iter().filter(|op| **op == name).count()
}

/// Every fold of DeepSeek-V4-Flash but the last lands with the pre-mix after
/// it, the first sublayer's pre-mix alone; V4.1-Flash keeps every fold.
#[test]
fn every_fold_of_a_deepseek_v4_trunk_but_the_last_lands_with_the_next_pre_mix() {
    let traced = |name: &str| {
        poem_compiler::catalog::deployment(name)
            .expect("the row is in the catalog")
            .trace(poem::Platform::Cuda)
    };
    let v4 = traced("dsv4-flash-mini-bf16-kv-bf16");
    let folds = count(&v4, "elemwise.hc_fold");
    assert!(folds > 2, "{folds} folds");
    let fused = fuse(v4);
    assert_eq!(count(&fused, "fused.hc_fold_pre"), folds - 1);
    assert_eq!(count(&fused, "elemwise.hc_fold"), 1);
    assert_eq!(count(&fused, "fused.hc_pre"), 1);

    let v41 = traced("dsv41-flash-mini-bf16-mxfp4-kv-bf16");
    let folds = count(&v41, "elemwise.hc_fold");
    let fused = fuse(v41);
    assert_eq!(count(&fused, "fused.hc_fold_pre"), 0);
    assert_eq!(count(&fused, "elemwise.hc_fold"), folds);
}
