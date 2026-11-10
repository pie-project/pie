//! A package that declares its state as `state(shape, dtype)` handles traces
//! the cache rows its `caches()` once declared: the same rows in the same
//! order, planes, dtypes, spaces, windows, heads, formats and cuts, only
//! named by where the handle sits in the layout. Every listed deployment of
//! each migrated package, at every rank count it splits across, is held to
//! the digest its rows had before the move
//! (`state_handles/<package>.caches`), each row renamed back by the
//! package's renaming below.
//!
//! Scaffolding for the move off `caches()`: it goes once every package
//! declares handles. `PIE_RECORD_CACHES=1` rewrites the digests of the
//! packages it runs.

use std::sync::Arc;

use poem::star::Deploy;
use poem_compiler::catalog::{Deployment, RANKS, repository};
use poem_ir::{CacheRow, Platform, Trace};

/// A row's name by its layout path (and what the row is), back to the name
/// `caches()` gave it.
type Renaming = fn(&str, &CacheRow) -> String;

/// The path `path` split at its dots.
fn parts(path: &str) -> Vec<&str> {
    path.split('.').collect()
}

fn qwen3_dense(path: &str, _row: &CacheRow) -> String {
    match parts(path)[..] {
        ["layers", l, "kv"] => format!("kv.{l}"),
        ["mtp", "kv"] => "kv.mtp".to_string(),
        ["draft_kv"] => "kv.draft".to_string(),
        _ => path.to_string(),
    }
}

fn nemotron_h(path: &str, _row: &CacheRow) -> String {
    match parts(path)[..] {
        ["layers", l, "attn", "kv"] => format!("kv.{l}"),
        ["layers", l, "mamba", "conv_state"] => format!("conv.{l}"),
        ["layers", l, "mamba", "ssm_state"] => format!("ssm.{l}"),
        _ => path.to_string(),
    }
}

fn dots3_note(path: &str, row: &CacheRow) -> String {
    let windowed = matches!(
        row,
        CacheRow::Kv {
            window: Some(_),
            ..
        }
    );
    match parts(path)[..] {
        ["layers", l, "attn", "kv"] if windowed => format!("swa.{l}"),
        ["layers", l, "attn", "kv"] => format!("kv.{l}"),
        ["layers", l, "attn", "indexer", "keys"] => format!("index.{l}"),
        _ => path.to_string(),
    }
}

fn inkling(path: &str, _row: &CacheRow) -> String {
    match parts(path)[..] {
        ["layers", l, "kv"] => format!("kv.{l}"),
        ["layers", l, "k_state"] => format!("conv.{l}.k"),
        ["layers", l, "v_state"] => format!("conv.{l}.v"),
        ["layers", l, "attn_state"] => format!("conv.{l}.attn"),
        ["layers", l, "mlp_state"] => format!("conv.{l}.mlp"),
        _ => path.to_string(),
    }
}

fn deepseek_v4(path: &str, _row: &CacheRow) -> String {
    match parts(path)[..] {
        ["layers", l, "attn", "kv"] => format!("kv.{l}"),
        ["layers", l, "attn", "pool", "entries"] => format!("pool.{l}"),
        ["layers", l, "attn", "indexer", "keys"] => format!("index.{l}"),
        ["layers", l, "engram", "ids_state"] => format!("engram.{l}"),
        ["mtp", "block", "attn", "kv"] => "kv.mtp".to_string(),
        ["dspark", "stages", s, "attn", "kv"] => format!("dspark.{s}"),
        _ => path.to_string(),
    }
}

fn gemma_4(path: &str, _row: &CacheRow) -> String {
    match parts(path)[..] {
        ["layers", l, "attn", "kv"] => format!("kv.{l}"),
        ["draft", "attn", "kv"] => "kv.mtp".to_string(),
        ["dflash", "blocks", b, "attn", "kv"] => format!("kv.dflash.{b}"),
        _ => path.to_string(),
    }
}

fn qwen_3(path: &str, _row: &CacheRow) -> String {
    match parts(path)[..] {
        ["layers", l, "attn", "kv"] => format!("kv.{l}"),
        ["layers", l, "gdn", "conv_state"] => format!("conv.{l}"),
        ["layers", l, "gdn", "delta_state"] => format!("delta.{l}"),
        ["mtp", "attn", "kv"] => "kv.mtp".to_string(),
        ["draft_kv"] => "kv.draft".to_string(),
        ["dflash", "blocks", b, "attn", "kv"] => format!("kv.dflash.{b}"),
        _ => path.to_string(),
    }
}

fn glm_5(path: &str, _row: &CacheRow) -> String {
    match parts(path)[..] {
        ["layers", l, "attn", "kv"] => format!("kv.{l}"),
        ["layers", l, "attn", "indexer", "keys"] => format!("index.{l}"),
        ["mtp", "attn", "kv"] => "kv.mtp".to_string(),
        ["mtp", "attn", "indexer", "keys"] => "index.mtp".to_string(),
        _ => path.to_string(),
    }
}

fn kimi_k3(path: &str, _row: &CacheRow) -> String {
    match parts(path)[..] {
        ["layers", l, "mixer", "kv"] => format!("kv.{l}"),
        ["layers", l, "mixer", "conv_state"] => format!("conv.{l}"),
        ["layers", l, "mixer", "delta_state"] => format!("delta.{l}"),
        _ => path.to_string(),
    }
}

fn glm_5_next(path: &str, row: &CacheRow) -> String {
    match parts(path)[..] {
        ["layers", l, "mixer", "kv"] => format!("kv.{l}"),
        ["layers", l, "mixer", "indexer", "keys"] => format!("index.{l}"),
        ["layers", l, "mixer", "conv_state"] => format!("conv.{l}"),
        ["layers", l, "mixer", "delta_state"] => format!("delta.{l}"),
        _ => glm_5(path, row),
    }
}

fn hy_4(path: &str, row: &CacheRow) -> String {
    glm_5(path, row)
}

fn minimax_m3(path: &str, _row: &CacheRow) -> String {
    match parts(path)[..] {
        ["layers", l, "attn", "kv"] => format!("kv.{l}"),
        ["layers", l, "indexer", "keys"] => format!("index.{l}"),
        _ => path.to_string(),
    }
}

fn qwen_4(path: &str, _row: &CacheRow) -> String {
    match parts(path)[..] {
        ["layers", l, "attn", "kv"] => format!("kv.{l}"),
        ["layers", l, "indexer", "keys"] => format!("index.{l}"),
        ["layers", l, "gdn", "conv_state"] => format!("conv.{l}"),
        ["layers", l, "gdn", "delta_state"] => format!("delta.{l}"),
        ["mtp", "block", "attn", "kv"] => "kv.mtp".to_string(),
        ["ple", "ids_state"] => "ple.ids".to_string(),
        ["ple", "conv_state"] => "ple.conv".to_string(),
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

/// Whether `caches()` kept the row `name` of `package` whole though only
/// attentions reading it through a window read it: the handle's row is kept
/// through that window, a space of its own, where `caches()` kept it in the
/// package's whole-sequence space, space 0. The rows are otherwise alike.
fn kept_whole(package: &str, name: &str) -> bool {
    match (package, &parts(name)[..]) {
        // Inkling's local layers read their kv through the local window.
        ("inkling", ["kv", _]) => true,
        // A DFlash block read through a window.
        ("qwen_3" | "gemma_4", ["kv", "dflash", _]) => true,
        _ => false,
    }
}

/// Whether `caches()` declared `deployment` a space 0 no row is in: a
/// one-layer Gemma 4 of a sliding layer alone, whose row sat in space 1
/// behind the global layers' empty space. A space now is made only for the
/// rows in it.
fn left_space_zero_empty(deployment: &str) -> bool {
    deployment.starts_with("gemma4-e4b-mini-l1-")
}

/// Whether `caches()` gave each index-key row of `package` a space of its
/// own, made in its loop over the layers: the handles' rows are alike, all
/// in the one space of their dtype, numbered after every other.
fn index_space_a_layer(package: &str) -> bool {
    matches!(package, "minimax_m3" | "glm_5_next" | "qwen_4")
}

/// Whether `caches()` gave `package`'s latent rows a rope plane of no
/// width: GLM-5-next's latent attention rotates no dims, and a state holds
/// no axis of zero, so its handle's row is the latent's one plane.
fn keeps_an_empty_rope_plane(package: &str) -> bool {
    package == "glm_5_next"
}

fn renamed(package: &str, deployment: &str, trace: &Trace, rename: Renaming) -> Vec<CacheRow> {
    let mut rows = trace.caches.clone();
    if left_space_zero_empty(deployment) {
        for row in &mut rows {
            if let CacheRow::Kv { space, .. } = row {
                *space += 1;
            }
        }
    }
    for row in &mut rows {
        let was = row.clone();
        let (CacheRow::Kv { name, .. } | CacheRow::State { name, .. }) = row;
        *name = rename(name, &was);
        if let CacheRow::Kv {
            name,
            space,
            window,
            ..
        } = row
            && kept_whole(package, name)
        {
            *space = 0;
            *window = None;
        }
    }
    if keeps_an_empty_rope_plane(package) {
        for row in &mut rows {
            if let CacheRow::Kv {
                name,
                planes,
                rows: None,
                ..
            } = row
                && name.starts_with("kv.")
                && planes.len() == 1
            {
                planes.push(0);
            }
        }
    }
    if index_space_a_layer(package) {
        let mut next = None;
        for row in &mut rows {
            if let CacheRow::Kv { name, space, .. } = row
                && name.starts_with("index.")
            {
                let at = next.unwrap_or(*space);
                *space = at;
                next = Some(at + 1);
            }
        }
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

/// `package`'s deployments trace the rows recorded for it.
fn holds(package: &str, rename: Renaming) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/poem_compiler/state_handles")
        .join(format!("{package}.caches"));
    let lines: Vec<String> = deployments(package)
        .iter()
        .map(|d| {
            let trace = d.trace(Platform::Cuda);
            format!(
                "{} {:016x}",
                d.name,
                digest(&renamed(package, &d.name, &trace, rename))
            )
        })
        .collect();
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
        "{} of {} `{package}` deployments trace other cache rows than before the move \
         ({} recorded):\n{}",
        unlike.len(),
        traced.lines().count(),
        recorded.lines().count(),
        unlike.join("\n")
    );
}

macro_rules! moved {
    ($($package:ident),* $(,)?) => {
        $(
            #[test]
            fn $package() {
                super::holds(stringify!($package), super::$package);
            }
        )*
    };
}

/// The packages moved to handles, each held to its digests by its renaming.
mod moved {
    moved!(
        qwen3_dense,
        nemotron_h,
        dots3_note,
        inkling,
        deepseek_v4,
        gemma_4,
        qwen_3,
        glm_5,
        kimi_k3,
        glm_5_next,
        hy_4,
        minimax_m3,
        qwen_4,
    );
}
