//! Every name a body can compose is one the shader tree declares.
//!
//! # What this replaces, and what it restores
//!
//! Nineteen tables of literals, 291 rows, indexed by folding two to four axis
//! values into one integer: `QMM_T[qmm_point(group, bits, bm, bn)?]`. The name
//! IS those values, so the fold and the table were a round trip.
//!
//! A table could only hold a name somebody had typed, and `format!` will
//! compose anything — so this is where the guarantee comes back, and stronger:
//! a table was checked against nothing, and this is checked against the
//! `// pie:instantiate` lines the shaders carry.
//!
//! # Why it reads the tree rather than the build's census
//!
//! `build.rs` emits `CENSUS` from those same lines, but only under `native` —
//! without it the table is written EMPTY on purpose, so `model-ir` can read the
//! signature table without owning a shader toolchain. A test keyed on `CENSUS`
//! would pass by finding nothing on every box that has no `slangc`. Parsing the
//! directives here is the same parse against the same files, and it runs
//! everywhere.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Every entrypoint the shader tree declares, read off its directives.
fn declared() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels");
    walk(&root, &mut |text: &str| {
        for line in text.lines() {
            let Some(rest) = line.trim_start().strip_prefix("//") else {
                continue;
            };
            let rest = rest.trim_start();
            let Some(rest) = rest.strip_prefix("pie:instantiate") else {
                continue;
            };
            if let Some(name) = rest.split_whitespace().next() {
                out.insert(name.to_owned());
            }
        }
    });
    out
}

fn walk(dir: &Path, visit: &mut impl FnMut(&str)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, visit);
        } else if let Ok(text) = std::fs::read_to_string(&path) {
            visit(&text);
        }
    }
}

#[test]
fn every_composable_name_is_declared() {
    let declared = declared();
    let missing: Vec<&str> = kernels_vulkan::quant::composable()
        .into_iter()
        .filter(|name| !declared.contains(*name))
        .collect();

    assert!(
        missing.is_empty(),
        "{} composable name(s) name a point the shader tree does not declare. \
         A body reaching one asks for a module the build never emitted:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

/// Both vacuity guards, because either side going empty makes the walk above
/// pass by having nothing to compare.
#[test]
fn neither_side_of_the_comparison_is_empty() {
    let declared = declared();
    assert!(
        declared.len() > 400,
        "the shader tree declared {} entrypoints; it carries ~496, so the \
         directive parse has stopped matching",
        declared.len()
    );
    // 291 ON THE NOSE, which is what the nineteen tables held. A count that
    // moves means an axis moved, and an axis moving without the shader tree
    // moving with it is what the walk above then catches.
    assert_eq!(
        kernels_vulkan::quant::composable().len(),
        291,
        "the composers no longer produce what the tables they replaced held"
    );
}
