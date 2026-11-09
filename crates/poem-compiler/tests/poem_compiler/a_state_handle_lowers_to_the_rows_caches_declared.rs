//! A package that declares its state as `state(shape, dtype)` handles traces
//! the cache rows its `caches()` once declared: the same rows in the same
//! order, planes, dtypes, spaces, windows and cuts, only named by where the
//! handle sits in the layout. Every listed deployment of each migrated
//! package, at every rank count it splits across, is held to the digest its
//! rows had before the move (`state_handles.caches`), each row renamed back
//! by the package's renaming below.
//!
//! Scaffolding for the move off `caches()`: it goes once every package
//! declares handles. `PIE_RECORD_CACHES=1` rewrites the digests.

use std::sync::Arc;

use poem::star::Deploy;
use poem_compiler::catalog::{Deployment, RANKS, repository};
use poem_ir::{CacheRow, Platform, Trace};

const DIGESTS: &str = "tests/poem_compiler/state_handles.caches";

/// The packages moved to handles, and each one's renaming: a row's name by
/// its layout path, back to the name its `caches()` gave it.
const MOVED: &[(&str, fn(&str) -> String)] =
    &[("qwen3_dense", qwen3_dense), ("nemotron_h", nemotron_h)];

fn qwen3_dense(path: &str) -> String {
    match path.split('.').collect::<Vec<_>>()[..] {
        ["layers", l, "kv"] => format!("kv.{l}"),
        ["mtp", "kv"] => "kv.mtp".to_string(),
        ["draft_kv"] => "kv.draft".to_string(),
        _ => path.to_string(),
    }
}

fn nemotron_h(path: &str) -> String {
    match path.split('.').collect::<Vec<_>>()[..] {
        ["layers", l, "attn", "kv"] => format!("kv.{l}"),
        ["layers", l, "mamba", "conv_state"] => format!("conv.{l}"),
        ["layers", l, "mamba", "ssm_state"] => format!("ssm.{l}"),
        _ => path.to_string(),
    }
}

/// Every listed deployment of `package`, at one rank and at each count of
/// [`RANKS`] it traces at.
fn deployments(package: &str) -> Vec<Deployment> {
    let listed: Vec<&Deployment> = repository()
        .deployments()
        .filter(|d| d.package.name() == package)
        .collect();
    assert!(!listed.is_empty(), "`{package}` lists no deployment");
    let mut all = Vec::new();
    for whole in listed {
        all.push(whole.clone());
        for tp in RANKS {
            let deploy = Deploy {
                tp,
                ..whole.deploy.clone()
            };
            let split = Deployment::of(Arc::clone(&whole.package), &whole.model, deploy);
            if split.check(Platform::Cuda).is_ok() {
                all.push(split);
            }
        }
    }
    all
}

fn renamed(trace: &Trace, rename: fn(&str) -> String) -> Vec<CacheRow> {
    let mut rows = trace.caches.clone();
    for row in &mut rows {
        let (CacheRow::Kv { name, .. } | CacheRow::State { name, .. }) = row;
        *name = rename(name);
    }
    rows
}

/// FNV-1a over the rows as `Debug` spells them.
fn digest(rows: &[CacheRow]) -> u64 {
    format!("{rows:?}")
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
}

#[test]
fn a_state_handle_lowers_to_the_rows_caches_declared() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(DIGESTS);
    let mut lines = Vec::new();
    for (package, rename) in MOVED {
        for d in deployments(package) {
            let trace = d.trace(Platform::Cuda);
            lines.push(format!(
                "{} {:016x}",
                d.name,
                digest(&renamed(&trace, *rename))
            ));
        }
    }
    let traced = lines.join("\n") + "\n";
    if std::env::var_os("PIE_RECORD_CACHES").is_some() {
        std::fs::write(&path, &traced).unwrap();
        return;
    }
    let recorded = std::fs::read_to_string(&path).unwrap();
    let unlike: Vec<String> = recorded
        .lines()
        .zip(traced.lines())
        .filter(|(was, is)| was != is)
        .map(|(was, is)| format!("  was {was}\n  is  {is}"))
        .collect();
    assert!(
        unlike.is_empty() && recorded.lines().count() == traced.lines().count(),
        "{} of {} deployments trace other cache rows than before the move \
         ({} recorded):\n{}",
        unlike.len(),
        traced.lines().count(),
        recorded.lines().count(),
        unlike.join("\n")
    );
}
