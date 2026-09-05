//! **WHICH CATALOG ROWS THIS SHELL CAN SERVE, ASSERTED RATHER THAN ASSUMED.**
//!
//! `model_ir::ops` states the rule this test enforces from the outside: *"one
//! variant per family, so 'does this backend cover this op' is a missing match
//! arm in its `Dispatch` impl, caught at compile time."* Compile time settles
//! the arms; it cannot settle an arm that exists and refuses, and it cannot
//! settle an arm that dispatches into a kernel entry which itself refuses.
//! Those two are the coverage holes, they are few, and each is named in
//! [`REFUSED`] beside the reason it stands and the source that carries it.
//!
//! Every row of `models::skus()` is traced on this shell's platform, every
//! node's `Operation::name()` is read off the trace, and a row that names a
//! refused op fails — unless it sits in [`CANNOT_SERVE`] with the op that
//! stops it. So the test fails in four directions:
//!
//! * a shell or kernels crate that grows a refusal an unlisted row reaches;
//! * a catalog row that grows an op this backend refuses;
//! * an exemption that stops being true — [`no_exemption_outlives_its_reason`];
//! * a refusal that stops being real — [`every_refusal_is_still_carried`],
//!   which reads each listed site back and fails if the refusal has gone.
//!   Without it a covered op left in the list would silently exempt every row
//!   that names it.
//!
//! It runs anywhere: a trace is arithmetic over the SKU's recipe, and no device
//! is opened. It does NOT check that a covered op *computes* the right thing —
//! `kernels_*.rs` and the artifact tests carry that.
//!
//! # Why op names and not the dispatch arms themselves
//!
//! Driving the arms would be stronger and is not available without a device:
//! `Run::new` wants a `&Scratch`, and `Scratch::reserve` allocates. The
//! refusals are declared here instead and pinned to their source.

use std::collections::{BTreeMap, BTreeSet};

use model_ir::Operands;
use model_ir::Platform;

/// The platform this shell serves.
const PLATFORM: Platform = Platform::Cuda;

/// Which backend the refusal list below was read from.
const SHELL: &str = "engine-cuda";

/// One refused op: what it is called, why the refusal stands, the file that
/// carries it (relative to this crate, `../` for a sibling), and a fragment of
/// that file which must still be there.
struct Refusal {
    op: &'static str,
    why: &'static str,
    file: &'static str,
    needle: &'static str,
}

/// **EVERY OP THIS BACKEND REFUSES**, whether the refusal is in the shell's
/// dispatch arm or in the kernel entry the arm dispatches into. An op absent
/// from this list is covered.
const REFUSED: &[Refusal] = &[
    // `dispatch/attn.rs`. `pool.cuh` ships only the dense `pool_lse_paged`
    // reader; falling back to it would answer a different attention under this
    // op's name. `kernels-metal`, `kernels-vulkan` and `kernels-wgpu` all
    // carry the selected reader.
    Refusal {
        op: "attention.pool_lse_selected",
        why: "no selected reader in pool.cuh; metal, vulkan and wgpu all cover it",
        file: "src/dispatch/attn.rs",
        needle: "op: \"attention.pool_lse_selected\"",
    },
];

/// The one refusal a name cannot express: `elementwise.rope_mrope` is covered
/// for two of its three section layouts and refused for the third, so the arm
/// is read off the op's own `form` field. Metal, vulkan and wgpu serve all
/// three.
fn refuses_split_mrope(op: &model_ir::ops::Operation) -> bool {
    matches!(
        op,
        model_ir::ops::Operation::Elementwise(model_ir::ops::Elementwise::RopeMrope {
            form: model_ir::ops::MropeForm::Split,
            ..
        })
    )
}

/// Rows this backend cannot serve, each with EVERY op that stops it — the set
/// is compared for equality, so a row exempted over one refusal cannot quietly
/// acquire another. Every entry is a debt, not a decision.
const CANNOT_SERVE: &[(&str, &[&str])] = &[
    // deepseek-v4-flash's indexer chooses compressed rows; `pool.cuh` can only
    // read every row. Metal, vulkan and wgpu all ship the selected reader.
    ("dsv4-flash-bf16-kv-bf16", &["attention.pool_lse_selected"]),
    (
        "dsv4-flash-u4g64-u2g64-kv-bf16",
        &["attention.pool_lse_selected"],
    ),
    (
        "dsv4-flash-full-u4g64-u2g64-kv-bf16",
        &["attention.pool_lse_selected"],
    ),
    (
        "dsv4-flash-mtp-u4g64-u2g64-mxfp4-kv-bf16",
        &["attention.pool_lse_selected"],
    ),
    (
        "dsv4-flash-full-mtp-u4g64-u2g64-mxfp4-kv-bf16",
        &["attention.pool_lse_selected"],
    ),
    // gemma-4's vision tower turns M-RoPE by the split (per-block
    // rotate_half) layout, which has no CUDA kernel.
    ("gemma4-26b-a4b-vision-u4g64-kv-bf16", &[SPLIT_MROPE]),
    ("gemma4-31b-vision-u4g64-kv-bf16", &[SPLIT_MROPE]),
    ("gemma4-e4b-vision-bf16-kv-bf16", &[SPLIT_MROPE]),
];

/// The ops one row names. A split M-RoPE is reported under its own synthetic
/// name, because the op name alone does not say which layout the row asks for.
fn ops_of(sku: &str) -> BTreeSet<String> {
    let row = models::sku(sku).expect("the row is in the catalog");
    (row.trace)(PLATFORM)
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

/// What [`ops_of`] calls a split-layout M-RoPE.
const SPLIT_MROPE: &str = "elementwise.rope_mrope(form=Split)";

fn refused() -> BTreeMap<&'static str, &'static Refusal> {
    let mut refused: BTreeMap<&'static str, &'static Refusal> =
        REFUSED.iter().map(|r| (r.op, r)).collect();
    refused.insert(SPLIT_MROPE, &SPLIT_MROPE_REFUSAL);
    refused
}

/// The field-conditional refusal, as a [`Refusal`] so it reads like the rest.
static SPLIT_MROPE_REFUSAL: Refusal = Refusal {
    op: SPLIT_MROPE,
    why: "gemma's per-block rotate_half has no CUDA kernel; metal, vulkan and wgpu cover it",
    file: "src/dispatch/attn.rs",
    needle: "",
};

/// Which refused ops stop each row, by row.
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

/// **EVERY CATALOG ROW NAMES ONLY OPS THIS BACKEND COVERS.**
#[test]
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

/// **NO EXEMPTION OUTLIVES ITS REASON.** A row in [`CANNOT_SERVE`] must still
/// be stopped, and stopped by the op the list names.
#[test]
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

/// **EVERY REFUSAL NAMED HERE IS STILL CARRIED.** Reads each listed site back
/// and fails if its refusal has gone — the guard that keeps the list from
/// becoming fiction.
#[test]
fn every_refusal_is_still_carried() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut gone = Vec::new();
    for refusal in REFUSED.iter().chain(std::iter::once(&SPLIT_MROPE_REFUSAL)) {
        if refusal.needle.is_empty() {
            // The field-conditional refusal lives in `dispatch/elemwise.rs`;
            // pin it there rather than by the empty needle above.
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

/// **EVERY ROW TRACES**, and traces for this platform. A row that traces to
/// nothing is a hole no refusal list would show.
#[test]
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

/// The op-by-op and row-by-row reading, printed rather than asserted. Run with
/// `cargo test -p engine-vulkan --test every_catalog_sku_dispatches -- --ignored --nocapture`.
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
