//! Every `.rs` file under `src/` is reachable from `lib.rs`.
//!
//! # Why this exists
//!
//! Commit `a86d6478d` deleted `src/metal/bind.rs` — **886 lines the compiler
//! had never once read**, left behind when the module list was trimmed. The
//! commit message is explicit that nothing in the tree would ever have
//! reported it, and `.wiki/driver/findings.md` §4 keeps it as one of the
//! family's two structural hazards:
//!
//! > A file with no `mod` declaration is invisible, not dead.
//!
//! That is worse than dead code, and in both directions. Nothing warns, so
//! the file rots — it keeps compiling in a reviewer's head and nowhere else,
//! and it is edited, reviewed and trusted while contributing nothing. And a
//! `mod` line lost in a rebase silently removes working code from the build,
//! which is the same failure the Apple-only re-export had.
//!
//! # What it does not check
//!
//! Whether a reachable module is *useful*. This answers the question
//! `dead_code` structurally cannot: a file it never saw.
//!
//! It used to say here that `dead_code` "already answers that for items".
//! It does not, for the items this crate is mostly made of. The lint is
//! silent on `pub` items in a library, because a library's public surface is
//! reachable from outside the build and rustc cannot know that nothing
//! outside the workspace links `driver-metal` — so `pub` switches the lint
//! off, and nine `pub fn`s here had no caller anywhere when anyone finally
//! counted. `every_public_function_has_a_reader.rs` is that count, kept.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The module files `root` reaches, transitively.
///
/// Declarations only: `mod foo;` with a semicolon names a FILE, while
/// `mod tests { … }` with a body is inline and names nothing on disk. The
/// distinction is the whole parse, which is why this is a scan rather than a
/// syntax tree — anything that resolves a path is doing the compiler's job
/// twice.
fn reachable(src: &Path) -> BTreeSet<PathBuf> {
    let mut seen = BTreeSet::new();
    let mut queue = vec![src.join("lib.rs")];
    while let Some(file) = queue.pop() {
        if !seen.insert(file.clone()) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        // A module's children live beside it: `foo.rs` looks in `foo/`, and
        // `foo/mod.rs` looks in `foo/`.
        let dir = if file
            .file_name()
            .is_some_and(|n| n == "lib.rs" || n == "mod.rs")
        {
            file.parent().map(Path::to_path_buf)
        } else {
            file.parent()
                .map(|p| p.join(file.file_stem().unwrap_or_default()))
        };
        let Some(dir) = dir else { continue };
        for line in text.lines() {
            let line = line.trim();
            // `#[cfg(test)] mod tests;` counts: a cfg'd-out module is still
            // declared, and this asks whether the file is NAMED.
            let Some(rest) = line
                .strip_prefix("mod ")
                .or_else(|| line.strip_prefix("pub mod "))
                .or_else(|| line.strip_prefix("pub(crate) mod "))
                .or_else(|| line.strip_prefix("pub(super) mod "))
            else {
                continue;
            };
            // A body, not a file.
            let Some(name) = rest.strip_suffix(';') else {
                continue;
            };
            let name = name.trim();
            if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                continue;
            }
            for candidate in [
                dir.join(format!("{name}.rs")),
                dir.join(name).join("mod.rs"),
            ] {
                if candidate.is_file() {
                    queue.push(candidate);
                }
            }
        }
    }
    seen
}

/// Every `.rs` file under `src/`.
fn on_disk(dir: &Path, out: &mut BTreeSet<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            on_disk(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.insert(path);
        }
    }
}

#[test]
fn every_source_file_is_named_by_a_mod_declaration() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let reached = reachable(&src);
    let mut present = BTreeSet::new();
    on_disk(&src, &mut present);

    let invisible: Vec<_> = present
        .difference(&reached)
        .filter_map(|p| p.strip_prefix(&src).ok())
        .collect();

    assert!(
        invisible.is_empty(),
        "these files are under src/ and no `mod` declaration names them, so \
         the compiler has never read a line of them: {invisible:?}\n\n\
         This is not dead code — `dead_code` cannot see a file it was never \
         given. It is 886 lines of `metal/bind.rs` waiting to happen again \
         (findings.md §4). Either declare the module or delete the file."
    );

    // And the audit itself has to be able to fail: a walker that reaches
    // nothing would pass this test forever.
    assert!(
        reached.len() > 10,
        "the module walk reached only {} files, which means it is not \
         walking — a broken audit passes silently",
        reached.len()
    );
}

/// Every `tests/device_*.rs` is declared in `Cargo.toml` with
/// `required-features = ["metal-4"]`.
///
/// # The same hazard, one directory across
///
/// `mod_audit`'s subject is a file the compiler never reads. This is its
/// mirror: a file cargo reads **when it should not**. `Cargo.toml` says, in
/// prose, that `required-features` *"keeps `cargo test -p driver-metal` with
/// no feature to the portable half"* and lists the portable tests by name as
/// *"the claim this file makes about what needs no GPU"*.
///
/// A `tests/*.rs` with no `[[test]]` entry is auto-discovered with no
/// required features at all, so it joins that claim silently. `device_add_bias.rs`
/// did: 31 files on disk, 30 declared, and on Linux it failed to compile
/// against `driver_metal::{bind, device, program}` — three items that only
/// exist behind `metal-4`. Every other `device_*` target was correctly
/// listed, which is exactly why nobody looked: the convention was visibly
/// held by thirty of thirty-one.
///
/// The manifest is parsed as TEXT rather than through a TOML crate. The
/// check is "does a `[[test]]` block name this and require the feature",
/// which is a two-line question, and adding a dependency to `driver-metal`
/// to ask it would put a build edge on a portable test.
#[test]
fn every_device_test_requires_the_metal_feature() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("Cargo.toml");

    // The declared set: a `name = "..."` that is followed, before the next
    // `[[`, by the feature requirement.
    let declared: BTreeSet<String> = manifest
        .split("[[test]]")
        .skip(1)
        .filter(|block| block.contains(r#"required-features = ["metal-4"]"#))
        .filter_map(|block| {
            let rest = block.split_once("name = \"")?.1;
            Some(rest.split_once('"')?.0.to_string())
        })
        .collect();

    let mut undeclared = Vec::new();
    let mut count = 0usize;
    for entry in std::fs::read_dir(root.join("tests"))
        .expect("tests/")
        .flatten()
    {
        let path = entry.path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if path.extension().is_none_or(|e| e != "rs") || !stem.starts_with("device_") {
            continue;
        }
        count += 1;
        if !declared.contains(stem) {
            undeclared.push(stem.to_string());
        }
    }

    assert!(
        undeclared.is_empty(),
        "{} device test(s) have no `[[test]]` entry requiring `metal-4`: {undeclared:?}. \
         Cargo auto-discovers them with NO required features, so `cargo test -p driver-metal` \
         on a host without the feature compiles them against items that are not there. \
         Add the entry or move the test off the `device_` prefix.",
        undeclared.len()
    );

    // A DENOMINATOR, because a `read_dir` that matched nothing passes the
    // loop above and proves nothing -- the failure this file exists to name,
    // applied to itself.
    assert!(
        count > 20,
        "only {count} `device_*` tests were scanned, so this is not reading the \
         test directory and a missing declaration would not be seen"
    );

    // AND THE OTHER DIRECTION: a declared target with no file is a manifest
    // that has outlived its test. Cargo errors on it, but only for the
    // feature combination that builds it -- so on a host with no `metal-4`
    // it is invisible, which is the same blind spot from the other side.
    let missing: Vec<&String> = declared
        .iter()
        .filter(|name| !root.join("tests").join(format!("{name}.rs")).exists())
        .collect();
    assert!(
        missing.is_empty(),
        "`Cargo.toml` declares {} test target(s) with no file: {missing:?}",
        missing.len()
    );
}
