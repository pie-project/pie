use std::collections::{BTreeMap, BTreeSet};

use poem_ir::Operands;
use poem_ir::Platform;

const PLATFORM: Platform = Platform::Cuda;

const SHELL: &str = "engine-cuda";

struct Refusal {
    op: &'static str,
    why: &'static str,
    file: &'static str,
    needle: &'static str,
}

const REFUSED: &[Refusal] = &[Refusal {
    op: "attention.pool_lse_selected",
    why: "no selected reader in pool.cuh; metal, vulkan and wgpu all cover it",
    file: "src/dispatch/attn.rs",
    needle: "op: \"attention.pool_lse_selected\"",
}];

fn refuses_split_mrope(op: &poem_ir::ops::Operation) -> bool {
    matches!(
        op,
        poem_ir::ops::Operation::Elementwise(poem_ir::ops::Elementwise::RopeMrope {
            form: poem_ir::ops::MropeForm::Split,
            ..
        })
    )
}

const CANNOT_SERVE: &[(&str, &[&str])] = &[
    ("dsv4-flash-bf16-kv-bf16", &["attention.pool_lse_selected"]),
    (
        "dsv4-flash-mini-bf16-kv-bf16",
        &["attention.pool_lse_selected"],
    ),
    (
        "dsv4-flash-mini-bf16-mxfp4-kv-bf16",
        &["attention.pool_lse_selected"],
    ),
    (
        "dsv41-flash-u4g64-kv-bf16",
        &["attention.pool_lse_selected"],
    ),
    (
        "dsv41-flash-bf16-mxfp4-kv-bf16",
        &["attention.pool_lse_selected"],
    ),
    (
        "dsv41-flash-mini-bf16-mxfp4-kv-bf16",
        &["attention.pool_lse_selected"],
    ),
    (
        "dsv4-flash-mini-u4g64-u2g64-kv-bf16",
        &["attention.pool_lse_selected"],
    ),
    (
        "dsv4-flash-u4g64-u2g64-kv-bf16",
        &["attention.pool_lse_selected"],
    ),
    (
        "dsv4-flash-mini-mtp-u4g64-u2g64-mxfp4-kv-bf16",
        &["attention.pool_lse_selected"],
    ),
    (
        "dsv4-flash-mtp-u4g64-u2g64-mxfp4-kv-bf16",
        &["attention.pool_lse_selected"],
    ),
    ("gemma4-26b-a4b-vision-u4g64-kv-bf16", &[SPLIT_MROPE]),
    ("gemma4-31b-vision-u4g64-kv-bf16", &[SPLIT_MROPE]),
    ("gemma4-e4b-vision-bf16-kv-bf16", &[SPLIT_MROPE]),
];

fn ops_of(deployment: &str) -> BTreeSet<String> {
    let row = poem_compiler::catalog::deployment(deployment).expect("the row is in the catalog");
    row.trace(PLATFORM)
        .nodes
        .iter()
        .map(|node| {
            if refuses_split_mrope(&node.op) {
                SPLIT_MROPE.to_string()
            } else {
                node.op.name().to_string()
            }
        })
        .collect()
}

const SPLIT_MROPE: &str = "elementwise.rope_mrope(form=Split)";

fn refused() -> BTreeMap<&'static str, &'static Refusal> {
    let mut refused: BTreeMap<&'static str, &'static Refusal> =
        REFUSED.iter().map(|r| (r.op, r)).collect();
    refused.insert(SPLIT_MROPE, &SPLIT_MROPE_REFUSAL);
    refused
}

static SPLIT_MROPE_REFUSAL: Refusal = Refusal {
    op: SPLIT_MROPE,
    why: "gemma's per-block rotate_half has no CUDA kernel; metal, vulkan and wgpu cover it",
    file: "src/dispatch/attn.rs",
    needle: "",
};

fn stopped() -> BTreeMap<String, BTreeSet<String>> {
    let refused = refused();
    let mut stopped = BTreeMap::new();
    for row in poem_compiler::catalog::deployments().chain(poem_compiler::catalog::splits()) {
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
    no_exemption_outlives_its_reason();
    every_refusal_is_still_carried();
    every_catalog_deployment_traces();
}

fn every_catalog_deployment_dispatches() {
    let refused = refused();
    let exempt: BTreeMap<&str, &[&str]> = CANNOT_SERVE.iter().copied().collect();

    let unlisted: Vec<String> = stopped()
        .into_iter()
        .filter(|(deployment, _)| !exempt.contains_key(one_rank(deployment)))
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
        unlisted.is_empty(),
        "{} catalog row(s) name a refused op and are not in CANNOT_SERVE. Either cover the op, \
         or list the row WITH the op that stops it:\n  {}",
        unlisted.len(),
        unlisted.join("\n  ")
    );
}

/// The one-rank deployment `deployment` splits: a split is stopped by what stops the
/// deployment it splits, so it is exempted under that one's name.
fn one_rank(deployment: &str) -> &str {
    deployment
        .rsplit_once("-tp")
        .filter(|(_, ranks)| ranks.parse::<u32>().is_ok())
        .map_or(deployment, |(whole, _)| whole)
}

fn no_exemption_outlives_its_reason() {
    let stopped = stopped();
    let mut stale = Vec::new();

    for (deployment, stoppers) in CANNOT_SERVE {
        if poem_compiler::catalog::deployment(deployment).is_none() {
            stale.push(format!("{deployment} is exempted but is not a catalog row"));
            continue;
        }
        let listed: BTreeSet<String> = stoppers.iter().map(|op| (*op).to_string()).collect();
        match stopped.get(*deployment) {
            Some(blocked) if *blocked == listed => {}
            Some(blocked) => stale.push(format!(
                "{deployment} is exempted over {listed:?}, but what stops it is {blocked:?}"
            )),
            None => stale.push(format!(
                "{deployment} is exempted over {listed:?}, but nothing refused stops it any more — \
                 drop the exemption"
            )),
        }
    }

    assert!(
        stale.is_empty(),
        "{} stale exemption(s):\n  {}",
        stale.len(),
        stale.join("\n  ")
    );
}

fn every_refusal_is_still_carried() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut gone = Vec::new();
    for refusal in REFUSED.iter().chain(std::iter::once(&SPLIT_MROPE_REFUSAL)) {
        if refusal.needle.is_empty() {
            let elemwise = root.join("src/dispatch/elemwise.rs");
            let source = std::fs::read_to_string(&elemwise).expect("elemwise.rs reads");
            assert!(
                source.contains("MropeForm::Split"),
                "the split M-RoPE refusal has gone from dispatch/elemwise.rs"
            );
            continue;
        }
        let path = root.join(refusal.file);
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{} reads: {error}", refusal.file));
        if !source.contains(refusal.needle) {
            gone.push(format!(
                "`{}` is listed as refused, but `{}` no longer carries `{}` — either it is \
                 covered now (drop it from REFUSED, and drop the rows it exempted) or the site \
                 moved (repoint the entry)",
                refusal.op, refusal.file, refusal.needle
            ));
        }
    }
    assert!(gone.is_empty(), "{}", gone.join("\n  "));
}

fn every_catalog_deployment_traces() {
    let mut empty = Vec::new();
    for row in poem_compiler::catalog::deployments().chain(poem_compiler::catalog::splits()) {
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
    for row in poem_compiler::catalog::deployments().chain(poem_compiler::catalog::splits()) {
        for op in ops_of(&row.name) {
            *named.entry(op).or_default() += 1;
        }
    }

    println!(
        "{SHELL} on {PLATFORM:?}: {} rows",
        poem_compiler::catalog::deployments()
            .chain(poem_compiler::catalog::splits())
            .count()
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
    for row in poem_compiler::catalog::deployments().chain(poem_compiler::catalog::splits()) {
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
        poem_compiler::catalog::deployments()
            .chain(poem_compiler::catalog::splits())
            .count()
            - stopped.len(),
        poem_compiler::catalog::deployments()
            .chain(poem_compiler::catalog::splits())
            .count()
    );
}
