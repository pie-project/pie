use std::collections::{BTreeMap, BTreeSet};

use model_ir::Operands;
use model_ir::Platform;

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
        op: "custom_cuda.qkv_fused_qknorm_rope_vnorm_write",
        why: "a CUDA-only fusion; engine-vulkan refuses it too",
        file: "src/dispatch/custom.rs",
        needle: "Err(KernelError::Unsupported { op: op.name() })",
    },
    Refusal {
        op: "spatial.patchify",
        why: "voxels to patch tokens has no arm here; no catalog row reaches it",
        file: "src/dispatch/custom.rs",
        needle: "Spatial::Patchify { .. } | Spatial::Unpatchify { .. }",
    },
];

const CANNOT_SERVE: &[(&str, &[&str])] = &[
    ("dsv4-base-bf16-kv-bf16-tp2", &["collective.all_reduce"]),
    (
        "gemma4-e4b-bf16-kv-bf16-tp2",
        &["collective.all_gather", "collective.all_reduce"],
    ),
    (
        "gemma4-31b-bf16-kv-bf16-tp2",
        &["collective.all_gather", "collective.all_reduce"],
    ),
    (
        "glm5-a12b-bf16-kv-bf16-tp2",
        &["collective.all_gather", "collective.all_reduce"],
    ),
    (
        "gptoss-120b-bf16-mxfp4-kv-bf16-tp2",
        &["collective.all_gather", "collective.all_reduce"],
    ),
    (
        "qwen35-a3b-bf16-kv-bf16-tp2",
        &["collective.all_gather", "collective.all_reduce"],
    ),
    (
        "muse-glimmer-30b-bf16-kv-bf16-tp2",
        &["collective.all_gather", "collective.all_reduce"],
    ),
    (
        "kimik3-bf16-mxfp4-kv-bf16-tp2",
        &["collective.all_gather", "collective.all_reduce"],
    ),
    (
        "hunyuanimage3-80b-a13b-bf16-u8g64-kv-bf16-tp4",
        &["collective.all_reduce"],
    ),
    (
        "hunyuanimage3-80b-a13b-bf16-u4g64-kv-bf16-tp4",
        &["collective.all_reduce"],
    ),
    ("mini-dit-bf16-kv-bf16-tp2", &["collective.all_reduce"]),
    ("mini-dit-bf16-kv-bf16-tp4", &["collective.all_reduce"]),
];

fn ops_of(sku: &str) -> BTreeSet<String> {
    let row = models::sku(sku).expect("the row is in the catalog");
    (row.trace)(PLATFORM)
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
    for row in models::skus() {
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
fn every_catalog_sku_dispatches_every_case() {
    every_catalog_sku_dispatches();
    no_exemption_outlives_its_reason();
    every_refusal_is_still_carried();
    every_catalog_sku_traces();
}

fn every_catalog_sku_dispatches() {
    let refused = refused();
    let exempt: BTreeMap<&str, &[&str]> = CANNOT_SERVE.iter().copied().collect();

    let unlisted: Vec<String> = stopped()
        .into_iter()
        .filter(|(sku, _)| !exempt.contains_key(sku.as_str()))
        .map(|(sku, ops)| {
            let ops: Vec<&str> = ops.iter().map(String::as_str).collect();
            format!(
                "{sku} names {}, which {SHELL} refuses ({})",
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

fn no_exemption_outlives_its_reason() {
    let stopped = stopped();
    let mut stale = Vec::new();

    for (sku, stoppers) in CANNOT_SERVE {
        if models::sku(sku).is_none() {
            stale.push(format!("{sku} is exempted but is not a catalog row"));
            continue;
        }
        let listed: BTreeSet<String> = stoppers.iter().map(|op| (*op).to_string()).collect();
        match stopped.get(*sku) {
            Some(blocked) if *blocked == listed => {}
            Some(blocked) => stale.push(format!(
                "{sku} is exempted over {listed:?}, but what stops it is {blocked:?}"
            )),
            None => stale.push(format!(
                "{sku} is exempted over {listed:?}, but nothing refused stops it any more — \
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
    for refusal in REFUSED {
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

fn every_catalog_sku_traces() {
    let mut empty = Vec::new();
    for row in models::skus() {
        let trace = (row.trace)(PLATFORM);
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
    for row in models::skus() {
        for op in ops_of(&row.name) {
            *named.entry(op).or_default() += 1;
        }
    }

    println!("{SHELL} on {PLATFORM:?}: {} rows", models::skus().count());
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
    for row in models::skus() {
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
        models::skus().count() - stopped.len(),
        models::skus().count()
    );
}
