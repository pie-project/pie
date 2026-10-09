use std::collections::{BTreeMap, BTreeSet};

use poem_ir::Operands;
use poem_ir::Platform;

const PLATFORM: Platform = Platform::Metal;

const SHELL: &str = "engine-metal";

struct Refusal {
    op: &'static str,
    why: &'static str,
    file: &'static str,
    needle: &'static str,
}

const REFUSED: &[Refusal] = &[
    Refusal {
        op: "collective.all_reduce",
        why: "one device; kernels-vulkan refuses it too, kernels-cuda answers it through NCCL",
        file: "../kernels-metal/src/collective.rs",
        needle: "op: \"collective.all_reduce\"",
    },
    Refusal {
        op: "collective.all_gather",
        why: "one device; kernels-vulkan refuses it too, kernels-cuda answers it through NCCL",
        file: "../kernels-metal/src/collective.rs",
        needle: "op: \"collective.all_gather\"",
    },
    Refusal {
        op: "collective.reduce_scatter",
        why: "one device; kernels-vulkan refuses it too, kernels-cuda answers it through NCCL",
        file: "../kernels-metal/src/collective.rs",
        needle: "op: \"collective.reduce_scatter\"",
    },
    Refusal {
        op: "spatial.patchify",
        why: "voxels to patch tokens has no arm here; no catalog row reaches it",
        file: "src/dispatch/custom.rs",
        needle: "Spatial::Patchify { .. } | Spatial::Unpatchify { .. }",
    },
];

fn ops_of(deployment: &str) -> BTreeSet<String> {
    let row = models::deployment(deployment).expect("the row is in the catalog");
    row.trace(PLATFORM)
        .nodes
        .iter()
        .map(|node| node.op.name().to_string())
        .collect()
}

fn refused() -> BTreeMap<&'static str, &'static Refusal> {
    REFUSED.iter().map(|r| (r.op, r)).collect()
}

fn stopped() -> BTreeMap<String, BTreeSet<String>> {
    let refused = refused();
    let mut stopped = BTreeMap::new();
    for row in models::deployments() {
        let blocked: BTreeSet<String> = ops_of(&row.name)
            .into_iter()
            .filter(|op| refused.contains_key(op.as_str()))
            .collect();
        if !blocked.is_empty() {
            stopped.insert(row.name.clone(), blocked);
        }
    }
    stopped
}

#[test]
fn every_catalog_deployment_dispatches_every_case() {
    every_catalog_deployment_dispatches();
    every_refusal_is_still_carried();
    every_catalog_deployment_traces();
}

fn every_catalog_deployment_dispatches() {
    let refused = refused();

    let stopped: Vec<String> = stopped()
        .into_iter()
        .map(|(deployment, ops)| {
            let ops: Vec<&str> = ops.iter().map(String::as_str).collect();
            format!(
                "{deployment} names {}, which {SHELL} refuses ({})",
                ops.join(" and "),
                ops.iter()
                    .map(|op| refused[op].why)
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        })
        .collect();

    assert!(
        stopped.is_empty(),
        "{} catalog row(s) name an op {SHELL} refuses; cover the op:\n  {}",
        stopped.len(),
        stopped.join("\n  ")
    );
}

fn every_refusal_is_still_carried() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut gone = Vec::new();
    for refusal in REFUSED {
        let path = root.join(refusal.file);
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{} reads: {error}", refusal.file));
        if !source.contains(refusal.needle) {
            gone.push(format!(
                "`{}` is listed as refused, but `{}` no longer carries `{}` — either it is \
                 covered now (drop it from REFUSED) or the site \
                 moved (repoint the entry)",
                refusal.op, refusal.file, refusal.needle
            ));
        }
    }
    assert!(gone.is_empty(), "{}", gone.join("\n  "));
}

fn every_catalog_deployment_traces() {
    let mut empty = Vec::new();
    for row in models::deployments() {
        let trace = row.trace(PLATFORM);
        if trace.nodes.is_empty() {
            empty.push(row.name.clone());
        }
        assert_eq!(
            trace.platform, PLATFORM,
            "{} traced for {:?}, not {PLATFORM:?}",
            row.name, trace.platform
        );
    }
    assert!(empty.is_empty(), "rows that trace to nothing: {empty:?}");
}

#[test]
#[ignore = "a report, not a claim"]
fn report() {
    let refused = refused();
    let mut named: BTreeMap<String, usize> = BTreeMap::new();
    for row in models::deployments() {
        for op in ops_of(&row.name) {
            *named.entry(op).or_default() += 1;
        }
    }

    println!(
        "{SHELL} on {PLATFORM:?}: {} rows",
        models::deployments().count()
    );
    println!("\n== ops named by the catalog ({}) ==", named.len());
    for (op, rows) in &named {
        let mark = if refused.contains_key(op.as_str()) {
            "REFUSED"
        } else {
            "ok"
        };
        println!("{mark:>8}  {op}  ({rows} row(s))");
    }

    let stopped = stopped();
    println!("\n== per row ==");
    for row in models::deployments() {
        let ops = ops_of(&row.name);
        let verdict = match stopped.get(&row.name) {
            None => "serves".to_string(),
            Some(blocked) => format!(
                "REFUSES ({})",
                blocked
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        println!("{:<56} {:>3} ops  {verdict}", row.name, ops.len());
    }
    println!(
        "\n{} of {} rows serve",
        models::deployments().count() - stopped.len(),
        models::deployments().count()
    );
}
