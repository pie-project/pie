//! The portable half never names the Apple-only half.
//!
//! # Why this exists
//!
//! `src/lib.rs` opens with the crate's central claim, and it is the reason
//! the crate is split the way it is rather than by subsystem:
//!
//! > The portable half is not a convenience. It is the half that can be
//! > tested without a GPU, and keeping it importable from a Linux
//! > `cargo test` is what stops it from drifting back into the untestable
//! > half.
//!
//! `.wiki/driver/north-star.md` rule 2 makes the same point as a property to
//! hold rather than a tree to draw: *the cfg gate should end at one module
//! boundary*. The payoff is that geometry, facts, planning and binding are
//! testable on a box with no card.
//!
//! **Nothing enforced it.** The `driver-metal` CI job runs on `macos-latest`
//! only (`.github/workflows/ci.yml`), so every job in the tree compiles this
//! crate with `target_vendor = "apple"` true, and the off-Apple half of the
//! `#[cfg]` has no reader. That is rule 4's failure mode — *if it can be
//! skipped, it will be* — applied to rule 2.
//!
//! And it had already drifted when this was written: `model::tables` was
//! declared ungated while its first line is `use crate::metal::{Context,
//! Handle, allocate}`, so a non-Apple build resolved a module that the gate
//! at `lib.rs` had removed. Every one of its callers was Apple-only already;
//! only the declaration was not.
//!
//! # Why the check is two assertions and not one scan
//!
//! `.wiki/driver/real-metal-north-star.md` §6 asks for **one** gated subtree
//! rather than a gate per subsystem, and states what the alternative cost:
//!
//! > Four gates inside one module is how `tables` and `resolve` came to sit
//! > ungated beside gated siblings and reach across. One gated subtree makes
//! > that unrepresentable rather than merely discouraged.
//!
//! So there are two properties, and the first is what makes the second
//! cheap:
//!
//! 1. **Exactly one module declaration in `src/` carries the gate**, and it
//!    is [`SHELL`]. An earlier draft of this file had to collect every gated
//!    module NAME and look for each one, because there were five gates and a
//!    reference could name any of them. With one, a crossing has exactly one
//!    spelling.
//! 2. **No ungated file names that spelling.** One string, not a set.
//!
//! Losing assertion 1 makes assertion 2 pass vacuously — a second gate
//! elsewhere is a boundary this test cannot see — which is why it is an
//! assertion rather than a comment.
//!
//! # Why a scan when the build already answers this
//!
//! It did not, when this file was written. The gate was
//! `cfg(target_vendor = "apple")` and a Linux job was the only thing that
//! could have read the other side of it — not free, because `driver-metal`
//! pulls `zstd-sys` through `model-loader`, a C build that needs a cross
//! toolchain, which is why that job never existed.
//!
//! The gate is `feature = "metal-4"` now, so `cargo test -p driver-metal`
//! with no features compiles the portable half **on a Mac**, and that build
//! is the real check. This stays for the two things a build cannot say:
//! that the gate sits on exactly ONE declaration, and WHICH file crossed —
//! a build reports an unresolved path, not a boundary.
//!
//! # What it does not check
//!
//! That the portable half is *correct* off-Apple — only that it does not
//! NAME the gated half. A portable file calling a portable function that
//! happens to be wrong on Linux is a different question, and one only a
//! Linux build can answer.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// A `mod NAME;` declaration, and whether the line above it gated the module
/// to Apple.
///
/// Declarations only, for `mod_audit.rs`'s reason: `mod foo;` with a
/// semicolon names a FILE, while `mod tests { … }` with a body names nothing
/// on disk.
fn declarations(text: &str) -> Vec<(String, bool)> {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(rest) = line
            .strip_prefix("mod ")
            .or_else(|| line.strip_prefix("pub mod "))
            .or_else(|| line.strip_prefix("pub(crate) mod "))
            .or_else(|| line.strip_prefix("pub(super) mod "))
        else {
            continue;
        };
        let Some(name) = rest.strip_suffix(';') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        // The gate sits on the line above the declaration, which is the only
        // spelling this crate uses. An attribute stack deeper than one line
        // would need a parse, and a scan that guessed would be worse than one
        // that is narrow and says so.
        let gated = i > 0 && lines[i - 1].contains(GATE);
        out.push((name.to_string(), gated));
    }
    out
}

/// Walk from `lib.rs`, collecting every module file and whether it is Apple
/// only — by its own gate or by any ancestor's.
fn walk(src: &Path) -> Vec<(PathBuf, bool)> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut queue = vec![(src.join("lib.rs"), false)];
    while let Some((file, apple)) = queue.pop() {
        if !seen.insert(file.clone()) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        out.push((file.clone(), apple));
        let dir = if file.file_name().is_some_and(|n| n == "lib.rs" || n == "mod.rs") {
            file.parent().map(Path::to_path_buf)
        } else {
            file.parent()
                .map(|p| p.join(file.file_stem().unwrap_or_default()))
        };
        let Some(dir) = dir else { continue };
        for (name, gated) in declarations(&text) {
            for candidate in [dir.join(format!("{name}.rs")), dir.join(&name).join("mod.rs")] {
                if candidate.is_file() {
                    // Gatedness is inherited: everything under `gpu/` is
                    // Apple-only because `gpu` itself is, and no file in it
                    // repeats the attribute.
                    queue.push((candidate, apple || gated));
                }
            }
        }
    }
    out
}

/// The gate itself, as it is spelled in the source.
///
/// A feature and not `cfg(target_vendor = "apple")`, and the difference is
/// the reason this file can be trusted at all:
///
/// > **A platform cfg cannot be tested. A feature can.**
///
/// On macOS `target_vendor = "apple"` is always true, so the portable half
/// is never compiled; on Linux it is always false, so the Apple half never
/// is. No machine builds both. With a feature, one macOS runner builds both
/// sides -- so this scan is the cheap check and the BUILD is the real one.
const GATE: &str = r#"cfg(feature = "metal-4")"#;

/// The one gated module. Everything Apple-only is under it, and a portable
/// file that reaches across names it.
const SHELL: &str = "gpu";

/// Does `line` name the module path `spelling`, as a whole path segment?
///
/// The boundary is the whole point: `super::kv_move` CONTAINS `super::kv`,
/// and a substring test reports the portable `store::kv_move` as a reference
/// to the Apple-only `model::kv`. A path segment ends where an identifier
/// character stops.
fn names_module(line: &str, spelling: &str) -> bool {
    line.match_indices(spelling).any(|(at, _)| {
        let after = line[at + spelling.len()..].chars().next();
        !after.is_some_and(|c| c.is_alphanumeric() || c == '_')
    })
}

#[test]
fn no_portable_module_names_the_apple_only_half() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let modules = walk(&src);

    // One gate. Not "the gates end at a boundary" as prose -- the count,
    // which is the form of the claim a build can refuse.
    let gates: Vec<String> = modules
        .iter()
        .filter_map(|(p, _)| {
            let text = std::fs::read_to_string(p).ok()?;
            let named: Vec<String> = declarations(&text)
                .into_iter()
                .filter(|(_, gated)| *gated)
                .map(|(name, _)| name)
                .collect();
            if named.is_empty() {
                None
            } else {
                Some(named)
            }
        })
        .flatten()
        .collect();

    assert_eq!(
        gates,
        vec![SHELL.to_string()],
        "the Apple gate must sit on exactly one module declaration, `{SHELL}`. \
         Found {} instead. `.wiki/driver/real-metal-north-star.md` §6: four \
         gates inside one module is how `tables` and `resolve` came to sit \
         ungated beside gated siblings and reach across. A second gate also \
         makes the reference check below pass vacuously, because a crossing \
         into it has a spelling this test does not look for.",
        gates.len()
    );

    // With one gated subtree, a crossing has exactly one spelling.
    let mut offences = Vec::new();
    for (file, apple) in &modules {
        if *apple {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        let shown = file.strip_prefix(&src).unwrap_or(file).display().to_string();
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        for (n, line) in lines.iter().enumerate() {
            // Doc and comment lines may name the gated half freely: prose
            // that says "`crate::gpu` adds the two impls" is the
            // documentation working, not a dependency.
            if line.starts_with("//") {
                continue;
            }
            // An item the file gated ITSELF is the boundary held, not
            // crossed -- `lib.rs` re-exports the shell exactly this way.
            if n > 0 && lines[n - 1].contains(GATE) {
                continue;
            }
            for spelling in [format!("crate::{SHELL}"), format!("super::{SHELL}")] {
                if names_module(line, &spelling) {
                    offences.push(format!("{shown}:{}: {line}", n + 1));
                }
            }
        }
    }
    offences.sort();
    offences.dedup();

    assert!(
        offences.is_empty(),
        "these files are declared WITHOUT `#[cfg(target_vendor = \"apple\")]` \
         and name `crate::{SHELL}`, which only exists WITH it, so the crate \
         does not compile off-Apple and no job in the tree would say so:\n\n  \
         {}\n\nThe portable half is the half that can be tested without a GPU \
         (`src/lib.rs`), and rule 2 of `.wiki/driver/north-star.md` is that \
         the cfg gate ends at one module boundary. Either gate the module \
         that made the reference, or move the arithmetic it wanted into \
         `layout/` -- which is what `kv::Shape` did.",
        offences.join("\n  ")
    );
}
