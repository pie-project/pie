//! The authoring-time half of palo design §1: trace the whole catalog and
//! sweep every class of every plan, so a merge that leaves a hole — or two
//! arms that write the same rows — is a sentence here rather than a garbled
//! token under one particular batch mix months from now.

use model_dsl::{Operands, Platform, resolve_classes};

/// Every platform a plan can be traced at. A model text may emit a different op
/// per platform (`model_dsl::platform()` picks the fused qkv write on CUDA), so
/// the split and merge structure is not the same graph on each, and one
/// platform passing says nothing about the others.
const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

#[test]
fn every_class_resolves_every_merge() {
    let mut faults = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            if let Err(unresolved) = resolve_classes(&trace) {
                for fault in &unresolved {
                    faults.push(format!("`{sku}` as {platform:?}: {}", fault.say(&trace)));
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
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(classes) = resolve_classes(&trace) else {
                continue; // the test above is the one that says so.
            };
            for &node in &classes.dead {
                let op = &trace.nodes[node as usize];
                faults.push(format!(
                    "`{sku}` as {platform:?}: node {node} ({}) is demanded in no \
                     class — nothing reads what it computes",
                    op.op.name(),
                ));
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
