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
//! Whether a reachable module is *useful*. `dead_code` already answers that
//! for items, and this answers the question `dead_code` structurally cannot:
//! a file it never saw.

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
