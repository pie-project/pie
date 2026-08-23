//! Every file this crate carries is reached by something.
//!
//! The fourth and last of the family, after the Vulkan, wgpu and CUDA gates.
//! Written on a tree that is already clean -- all 35 files are reached, 27 by
//! stamping at least one entrypoint and 8 by being included -- which is the
//! only good time to write one of these. The three that came before it were all
//! written *after* finding an orphan, and in every case the orphan's cost was
//! not the bytes: it was that `norm/rms_params.slang`, `moe/params.inc.wgsl` and
//! `attn/pack_dense_mask.cuh` each went on making confident present-tense
//! claims about a tree they had stopped being part of, with no reader left who
//! could notice. One of them was even *edited* for an unrelated citation
//! without anyone seeing it was unreachable.
//!
//! # What "reached" means here
//!
//! Cleaner than on the other three backends, because `build.rs` already
//! answers half of it. [`kernels_metal::STAMPED`] is what it reads out of the
//! `.metal` tree: one `(file, entrypoint)` row per point a device can compile.
//! A file that stamps a point is reached by definition. A file that stamps none
//! is a header, and a header is reached only by an `#include` -- resolved
//! relative to the INCLUDING file, which matters here and nowhere else:
//! `quant/qmm_t.metal` says `"mxfp4_codec.h"` for a sibling and
//! `"../third_party/mlx/steel_mma.metal"` for a cousin, and a probe that
//! resolves only from the tree root reports both as orphans.
//!
//! # What this does NOT ask
//!
//! Whether a stamped entrypoint is ever FIRED. That is a real and different
//! question -- the host-stamped `qmm_t` family composes its names at the fire
//! rather than listing them, so `census()` deliberately has two sources and the
//! file's own list is not the census. Asking it properly means asking which
//! names `quant::qmm_point` can construct, which is `lib.rs`'s subject, not
//! this file's. This asks only the question `build.rs`'s indiscriminate walk
//! makes possible to get wrong: a FILE carried into the binary that no compile
//! and no include can arrive at.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels")
}

/// Every file under `dir`, as a `/`-spelled path relative to `base`.
fn walk(dir: &Path, base: &Path, out: &mut Vec<String>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            walk(&path, base, out);
        } else {
            out.push(
                path.strip_prefix(base)
                    .expect("walked from base")
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
}

/// The `#include "…"` targets in one file, verbatim.
///
/// Angled includes are the toolchain's `<metal_stdlib>` and friends and are
/// never this tree's, so they are not this test's business.
fn quoted_includes(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix('#')?.trim_start();
            let rest = rest.strip_prefix("include")?.trim_start();
            let rest = rest.strip_prefix('"')?;
            let (target, _) = rest.split_once('"')?;
            Some(target.to_string())
        })
        .collect()
}

/// `a/b/../c` -> `a/c`, lexically -- `canonicalize` needs the file to exist,
/// and whether it exists is the question being asked.
fn normalise(spelling: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in spelling.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

/// Where one include spelling resolves: beside the includer.
fn resolve(from: &str, spelling: &str) -> String {
    let dir = from.rsplit_once('/').map_or("", |(d, _)| d);
    normalise(&if dir.is_empty() { spelling.to_string() } else { format!("{dir}/{spelling}") })
}

/// Every file another file includes, as tree-relative paths.
fn included(files: &[String]) -> BTreeSet<String> {
    let root = kernels_dir();
    let mut out = BTreeSet::new();
    for rel in files {
        let Ok(text) = std::fs::read_to_string(root.join(rel)) else {
            continue;
        };
        for spelling in quoted_includes(&text) {
            out.insert(resolve(rel, &spelling));
        }
    }
    out
}

#[test]
fn no_file_is_carried_that_stamps_nothing_and_nothing_includes() {
    let root = kernels_dir();
    let mut files = Vec::new();
    walk(&root, &root, &mut files);
    assert!(files.len() > 20, "the tree shrank to {} files -- is the walk right?", files.len());

    let stamps: BTreeSet<&str> = kernels_metal::STAMPED.iter().map(|(file, _)| *file).collect();
    let includes = included(&files);

    let orphans: Vec<&String> = files
        .iter()
        .filter(|rel| !stamps.contains(rel.as_str()) && !includes.contains(*rel))
        .collect();

    assert!(
        orphans.is_empty(),
        "carried but unreachable -- nothing stamps a point in them and nothing \
         includes them. Delete them, or include them from the shader whose ABI \
         they state: {orphans:?}"
    );
}

/// The other half of the rule: an `#include` naming a file the tree has not
/// got.
///
/// Metal's own compiler would say so, but only for the arms a device actually
/// asks to build, and only on a machine with one. Every backend in this
/// repository is developed on machines that mostly do not have the device the
/// backend targets, which is why all four of these gates are plain file walks.
#[test]
fn every_quoted_include_names_a_file_the_tree_carries() {
    let root = kernels_dir();
    let mut files = Vec::new();
    walk(&root, &root, &mut files);
    let carried: BTreeSet<&String> = files.iter().collect();

    let mut dangling = Vec::new();
    for rel in &files {
        let Ok(text) = std::fs::read_to_string(root.join(rel)) else {
            continue;
        };
        for spelling in quoted_includes(&text) {
            if !carried.contains(&resolve(rel, &spelling)) {
                dangling.push(format!("{rel} includes {spelling}"));
            }
        }
    }

    assert!(dangling.is_empty(), "includes that resolve to nothing carried: {dangling:?}");
}

/// Every `STAMPED` row names a file the tree still carries.
///
/// The third direction, and the one the other backends cannot ask: `STAMPED` is
/// regenerated from the tree by `build.rs`, so this can only fail if a stale
/// generated file survives a deletion -- which is precisely the failure that
/// makes a rebuild look green while the device compile of that point cannot
/// find its text.
#[test]
fn every_stamped_row_names_a_carried_file() {
    let root = kernels_dir();
    let mut missing = Vec::new();
    for (file, point) in kernels_metal::STAMPED {
        if !root.join(file).is_file() {
            missing.push(format!("{point} claims to live in {file}"));
        }
    }
    assert!(missing.is_empty(), "stamped points with no source: {missing:?}");
}
