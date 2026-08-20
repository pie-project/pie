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
//! So there are three properties, and the first is what makes the rest
//! cheap:
//!
//! 1. **The gate sits on exactly the modules that need a device**, and the
//!    set is [`ROOMS`]. An earlier draft asserted the gate sat on exactly
//!    ONE declaration — the `gpu` module, whose only job was to carry it for
//!    seven children — because with one gate a crossing has one spelling.
//!    That module is gone, and the equality is now against a set. What made
//!    the earlier draft's warning true was not the *count* but staleness: a
//!    hardcoded set drifts. This asserts the gated set READ from the source
//!    equals [`ROOMS`], in both directions, so it cannot drift silently.
//! 2. **Every module is classified**: in [`ROOMS`] or in [`PORTABLE`], never
//!    neither. This is the only one of the three the portable BUILD cannot
//!    report, and the only one the old nesting did not cover either.
//! 3. **No ungated file names a gated room.** One spelling per room, and
//!    property 1 is what guarantees the list of spellings is complete.
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
        for (name, gated) in declarations(&text) {
            for candidate in [
                dir.join(format!("{name}.rs")),
                dir.join(&name).join("mod.rs"),
            ] {
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

/// The gated rooms. Everything Apple-only is one of these or under one, and
/// a portable file that reaches across names one of them.
///
/// This was a single `SHELL = "gpu"` while the seven sat under a module
/// whose only job was to carry one `#[cfg]` for all of them. That module is
/// gone -- it charged a path segment on every reference in a crate that is
/// already a Metal driver, where `gpu::device::allocator` says GPU twice and
/// means it once -- so the claim is now a set. This file's own header
/// described that as the earlier, worse draft. It was worse for a reason
/// that no longer applies: the set was *hardcoded* then, so it went stale
/// silently. The set of gated names is now READ from the source by
/// [`walk`], and this list is what it must equal -- a disagreement in either
/// direction is a failure, so the check cannot pass vacuously.
const ROOMS: &[&str] = &[
    "bind", "device", "fire", "pools", "program", "serve", "weights",
];

/// The modules that answer questions no GPU changes.
///
/// Named, rather than derived as "everything not in [`ROOMS`]", because the
/// interesting failure is a module in NEITHER list: a new device module
/// placed at `src/` beside `layout` and `lowering` -- which is where someone
/// unfamiliar with the split would put it -- that talks to `objc2` directly
/// and to no gated sibling. Measured: that compiles clean in the portable
/// build, because `objc2` is an unconditional dependency of this crate, and
/// it breaks on Linux. It is also the one case the old nested layout did not
/// cover either, since a subtree can only protect files placed inside it.
/// `facts` was here and is DELETED, which is the entry worth a sentence. It
/// was portable and it passed this audit every time, because "does this need
/// a device" and "does this belong in a driver at all" are different
/// questions and only the first was being asked here. It was a second
/// definition of every model in the workspace -- a `pie.model/1` descriptor
/// parsed into a private facts struct, plus an enum of family names -- and it
/// needed no GPU to be wrong.
const PORTABLE: &[&str] = &[
    "batch", "channel", "envelope", "error", "layout", "loader", "lowering",
    "model",
];

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
            if named.is_empty() { None } else { Some(named) }
        })
        .flatten()
        .collect();

    assert_eq!(
        gates,
        ROOMS.iter().map(|r| (*r).to_string()).collect::<Vec<_>>(),
        "the Apple gate must sit on exactly the rooms that need a device, \
         {ROOMS:?}. The source says {gates:?}.\n\nA room missing the gate is \
         compiled on a machine with no Metal; a name here that the source \
         does not gate is this list gone stale. `.wiki/driver/\
         real-metal-north-star.md` §6 asked for one gated subtree so that a \
         crossing had exactly one spelling -- the subtree is gone and the \
         spelling is a set, so this equality is what stops the reference \
         check below from passing vacuously."
    );

    // Every declared module is in one list or the other. A module in neither
    // is one nothing checks: it is not required to be gated, and it is not
    // required to stay portable. That is the one hole the nested layout had
    // too, and the only one the portable BUILD cannot see -- a self-contained
    // device module that names no gated sibling compiles clean on a Mac.
    let roots: Vec<String> = std::fs::read_to_string(src.join("lib.rs"))
        .expect("lib.rs is beside this test")
        .lines()
        .map(str::trim)
        .filter_map(|l| {
            l.strip_prefix("pub mod ")
                .or_else(|| l.strip_prefix("mod "))
                .and_then(|r| r.strip_suffix(';'))
                .map(str::to_string)
        })
        .collect();
    let unclassified: Vec<&String> = roots
        .iter()
        .filter(|n| !ROOMS.contains(&n.as_str()) && !PORTABLE.contains(&n.as_str()))
        .collect();
    assert!(
        unclassified.is_empty(),
        "`lib.rs` declares {unclassified:?}, which is in neither ROOMS nor \
         PORTABLE.\n\nSay which it is. The cut is *does answering this need a \
         device*, not *is this about the GPU*: `layout::tuning` is entirely \
         about the GPU and is above the line, because its inputs are two \
         integers. An unclassified module is the one failure the portable \
         build cannot report -- if it needs a device and names no gated \
         sibling, it compiles on this Mac and breaks on Linux."
    );
    assert!(
        roots.len() >= ROOMS.len() + PORTABLE.len(),
        "read {} root declarations from lib.rs and expected at least {} -- \
         the scan is not reading lib.rs, and a scan that reads nothing passes",
        roots.len(),
        ROOMS.len() + PORTABLE.len()
    );

    // With the gated set known, a crossing has one spelling per room.
    let mut offences = Vec::new();
    for (file, apple) in &modules {
        if *apple {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        let shown = file
            .strip_prefix(&src)
            .unwrap_or(file)
            .display()
            .to_string();
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
            for spelling in ROOMS
                .iter()
                .flat_map(|r| [format!("crate::{r}"), format!("super::{r}")])
            {
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
         and name one of {ROOMS:?}, which only exist WITH it, so the crate \
         does not compile off-Apple and no job in the tree would say so:\n\n  \
         {}\n\nThe portable half is the half that can be tested without a GPU \
         (`src/lib.rs`), and rule 2 of `.wiki/driver/north-star.md` is that \
         the cfg gate ends at one module boundary. Either gate the module \
         that made the reference, or move the arithmetic it wanted into \
         `layout/` -- which is what `kv::Shape` did.",
        offences.join("\n  ")
    );
}
