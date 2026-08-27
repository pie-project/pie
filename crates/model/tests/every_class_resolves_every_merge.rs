//! The authoring-time half of palo design §1: trace the whole catalog and
//! sweep every class of every plan, so a merge that leaves a hole — or two
//! arms that write the same rows — is a sentence here rather than a garbled
//! token under one particular batch mix months from now.

use model_dsl::{Operands, Plane, resolve_classes};

/// Every plane a plan can be traced at. A model text may emit a different op
/// per plane (`Input::cuda()` picks the fused qkv write on CUDA), so the split
/// and merge structure is not the same graph on each, and one plane passing
/// says nothing about the others.
const PLANES: [Plane; 4] = [Plane::Cuda, Plane::Metal, Plane::Wgpu, Plane::Vulkan];

#[test]
fn every_class_resolves_every_merge() {
    let mut faults = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for plane in PLANES {
            let plan = trace(plane);
            if let Err(unresolved) = resolve_classes(&plan) {
                for fault in &unresolved {
                    faults.push(format!("`{sku}` as {plane:?}: {}", fault.say(&plan)));
                }
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

/// A node no class demands is not a fault — the sweep reports it, and the
/// compiler is free to drop it — but in a shipped model it is a forgotten
/// consumer, since every op a forward pass writes is written to be read.
#[test]
fn no_shipped_plan_computes_something_nothing_reads() {
    let mut faults = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for plane in PLANES {
            let plan = trace(plane);
            let Ok(classes) = resolve_classes(&plan) else {
                continue; // the test above is the one that says so.
            };
            for &node in &classes.dead {
                let op = &plan.nodes[node as usize];
                faults.push(format!(
                    "`{sku}` as {plane:?}: node {node} ({}) is demanded in no \
                     class — nothing reads what it computes",
                    op.op.name(),
                ));
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
