//! `branch::split` lays each join's independent chains side by side, and
//! `branch::blocks` finds them again: on DeepSeek-V4-Flash's trace (as CUDA
//! fuses its attention) every MoE layer's shared and routed experts, each
//! attention's query chain beside its key/value or index chain. The split
//! trace is a valid plan, every op reads only what an earlier op wrote, and
//! it bakes with the branches forked onto side streams.

use std::collections::BTreeSet;

use poem_compiler::DeviceProfile;
use poem_ir::{Def, Operands};

use crate::common::bake_with;

#[test]
fn a_joins_branches_lie_side_by_side() {
    let row = poem_compiler::catalog::deployment("dsv4-flash-fp8b128a8-mxfp4a8-kv-bf16")
        .expect("the row is in the catalog");
    let fused = poem_compiler::fuse::fuse(
        row.trace(poem::Platform::Cuda),
        &["attention.pool_attention", "attention.lse_sink"],
    );
    let split = poem_compiler::branch::split(fused.clone());
    assert_eq!(
        split.nodes.len(),
        fused.nodes.len(),
        "a split moves ops, it drops none"
    );
    // As valid a plan as the fused one it came from.
    let faults = |t: &poem_ir::Trace| poem_ir::check(t).err().map_or(0, |f| f.len());
    assert_eq!(
        faults(&split),
        faults(&fused),
        "the split trace is as valid as the fused one"
    );

    // Every op reads values an earlier op wrote (or inputs, weights, caches).
    let mut ins = Vec::new();
    for (j, node) in split.nodes.iter().enumerate() {
        ins.clear();
        node.op.inputs(&mut ins);
        for v in &ins {
            if let Def::Op(i) = split.values[v.0 as usize].def {
                assert!(
                    (i as usize) < j,
                    "op {j} ({}) reads a value op {i} writes later",
                    node.op.name()
                );
            }
        }
    }

    let blocks = poem_compiler::branch::blocks(&split);
    let joins: BTreeSet<&str> = blocks
        .iter()
        .map(|b| split.nodes[b.join as usize].op.name())
        .collect();
    for wanted in [
        "linear.moe_sum_add",
        "attention.pool_attention",
        "attention.lse_sink",
    ] {
        assert!(
            joins.contains(wanted),
            "no block joins at {wanted}: {joins:?}"
        );
    }
    let sums = blocks
        .iter()
        .filter(|b| split.nodes[b.join as usize].op.name() == "linear.moe_sum_add")
        .count();
    assert_eq!(
        sums, 43,
        "each of the 43 MoE layers forks its shared expert"
    );

    let profile = DeviceProfile {
        branches: true,
        ..DeviceProfile::default()
    };
    let compiled = bake_with(&split, &profile).expect("the split trace bakes");
    assert!(
        compiled.streams.streams > 1,
        "the branches run on side streams"
    );
    for region in &compiled.regions {
        if region.stream > 0 {
            assert!(region.nodes.end > region.nodes.start);
        }
    }
}
