//! Every `pub fn` in this crate is called by something, or is listed below
//! with the reason it is not.
//!
//! # Why this exists
//!
//! `mod_audit.rs` says, in its own "what it does not check" section, that
//! `dead_code` "already answers that for items". For this crate that is not
//! true, and the exception is not a corner: **rustc's `dead_code` lint does
//! not fire for `pub` items in a library crate.** It cannot, because a
//! library's public surface is by definition reachable from outside the
//! build, and rustc has no way to know that nothing outside this workspace
//! links `driver-metal`. So `pub` is, accidentally, a way to switch the
//! lint off — and every accessor here is `pub`.
//!
//! The gap is measurable. When this test was written, nine `pub fn`s in
//! `src/` had no call site anywhere under `crates/` — not in the crate, not
//! in a sibling, not in a test. Eight of them arrived in a single commit on
//! 2026-08-09 and had never been called since; they are the API surface a
//! port brings with it rather than something that accreted. The ninth,
//! `lowering::hold::Hold::state_mut`, was worse than unused: its body was
//! byte-for-byte identical to `state` beside it while its doc comment read
//! "the same, written". Nothing distinguished the two, so the name promised
//! a write binding and returned a read one. It was deleted with this test,
//! because the danger was not the dead line — it was the first caller who
//! would one day pick it for the semantics it advertised.
//!
//! # What this is not
//!
//! It is not an argument that an uncalled function is a bug. Several of the
//! eight below describe work that is stated but not yet wired, and a
//! forward-looking accessor is a reasonable thing to have. The claim is
//! narrower: an uncalled `pub fn` should be uncalled **on purpose**, and the
//! purpose should be written down where it can be read and where it becomes
//! wrong out loud when the function finally gets a caller.
//!
//! # What it does not check
//!
//! Whether a call site is on a path that ever runs. A function called only
//! from another dead function is live to this test, exactly as it is to
//! `dead_code`. Reachability is the compiler's question; this one is only
//! about whether a name has ever been written down twice.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The `pub fn`s with no caller, and why each is allowed to have none.
///
/// A name leaves this list the moment something calls it — the assertion is
/// an equality, so a stale excuse fails as loudly as a new absence. That
/// direction matters more than it looks: the failure a list like this
/// usually suffers is not a missing entry but an entry nobody retired, and
/// an excuse for a function that is now called is a lie in the same file
/// that exists to prevent them.
const NO_READER: &[(&str, &str)] = &[
    (
        "archives",
        "The pipeline cache's own directory handle. The compiler is \
         constructed with one and reads it internally; this hands it back \
         out for a caller that would report or clear it, and nothing yet \
         reports or clears it.",
    ),
    (
        "experts_per_layer",
        "One of the four numbers describing a mixture slab. Its three \
         siblings -- `layers`, `slots`, and the per-expert tensor count -- \
         are read when the slab is sized; this one is derivable from them \
         and so nothing has needed to ask.",
    ),
    (
        "outstanding_bytes",
        "The byte half of `outstanding_buffers`, which IS read. Both are \
         differences of two stats the allocator already publishes, so this \
         is a convenience over `stats()` rather than a fact only it knows.",
    ),
    (
        "program_entries",
        "A count of compiled programs held, beside `stage_entries` and \
         `stats`. `stats` carries what the eviction policy needs, so the \
         two entry counts are for a diagnostic that has not been written.",
    ),
    (
        "reserved_pages",
        "The KV pool's RESERVED address space, as distinct from `pages`, \
         which is mapped. The doc is explicit that admission should be \
         against this rather than `pages`, and admission currently is not. \
         This is a stated intention with no caller yet, not a leftover.",
    ),
    (
        "set_capacity",
        "Re-budgeting the device allocator at runtime. The budget is set \
         once at construction today; nothing raises or lowers it while a \
         model is resident, so the only path that would call this is a \
         memory-pressure response the driver does not implement.",
    ),
    (
        "set_logits_row",
        "The late rebind of a fused command's logits row, ported with the \
         M2 path from the C++. The forward that decides a member's row \
         late is the M3 group builder, and the group builder does not yet \
         rebind -- it composes each member against a row it already knows.",
    ),
    (
        "tickets",
        "The tickets a single fire was composed against. Its doc names its \
         intended reader by name: the M3 group builder, which re-checks \
         readiness per candidate and writes these into the lane table. \
         That builder does not read them yet.",
    ),
];

/// Every `.rs` file under `dir`.
fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The `pub fn` names declared in `text`, with the line each is on.
///
/// A declaration is recognised at the start of a line so that `pub fn` in
/// prose, in a doc example, or inside a string cannot be mistaken for one.
/// The visibility forms this crate uses are bare `pub` and `pub(crate)`;
/// only the first is collected, because `pub(crate)` is precisely the case
/// `dead_code` still sees and this test does not need to duplicate.
fn declarations(text: &str) -> Vec<(String, usize)> {
    let mut found = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let Some(rest) = line.trim_start().strip_prefix("pub ") else {
            continue;
        };
        let mut rest = rest;
        for modifier in ["const ", "unsafe ", "async ", "extern \"C\" "] {
            rest = rest.strip_prefix(modifier).unwrap_or(rest);
        }
        let Some(rest) = rest.strip_prefix("fn ") else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            found.push((name, index + 1));
        }
    }
    found
}

/// How many times each identifier in `text` is APPLIED -- followed by `(`
/// or by a turbofish.
///
/// Identifiers immediately preceded by `fn` are skipped, so a declaration
/// does not read as a use of itself and an unrelated private `fn` of the
/// same name does not vouch for a public one. Everything else counts,
/// including `Self::name(`, `.name(` and a bare `name(`: this deliberately
/// ignores which type a method is on, because resolving that is the
/// compiler's job and doing it here twice is how a scan starts disagreeing
/// with the build. The cost of the shortcut is that a private `foo` called
/// somewhere makes a public `foo` look read; the check errs toward
/// silence, which is the right direction for a list that must be curated
/// by hand.
fn applications(text: &str, out: &mut BTreeMap<String, usize>) {
    // Bytes, not chars, and every test is an ASCII one. A Rust identifier
    // cannot contain a non-ASCII byte, so a multi-byte sequence can only be
    // a separator here -- and treating its continuation bytes as separators
    // is both correct and the only way to stay on a char boundary without
    // paying for a full decode of every comment in the workspace.
    let bytes = text.as_bytes();
    let mut index = 0;
    let mut previous = String::new();
    while index < bytes.len() {
        let byte = bytes[index];
        if byte.is_ascii_alphabetic() || byte == b'_' {
            let start = index;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            let word = &text[start..index];
            let mut after = index;
            while after < bytes.len() && bytes[after].is_ascii_whitespace() {
                after += 1;
            }
            let applied = bytes.get(after) == Some(&b'(') || bytes[after..].starts_with(b"::<");
            if applied && previous != "fn" {
                *out.entry(word.to_string()).or_default() += 1;
            }
            previous = word.to_string();
        } else {
            if !byte.is_ascii_whitespace() {
                previous.clear();
            }
            index += 1;
        }
    }
}

#[test]
fn every_public_function_has_a_reader_or_a_stated_reason() {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = crate_root
        .parent()
        .expect("crates/driver-metal has a parent");

    let mut own = Vec::new();
    sources(&crate_root.join("src"), &mut own);
    let mut declared: BTreeMap<String, (PathBuf, usize)> = BTreeMap::new();
    for file in &own {
        let text = std::fs::read_to_string(file).expect("a source file reads");
        for (name, line) in declarations(&text) {
            declared.entry(name).or_insert((file.clone(), line));
        }
    }

    // The scan is over text, so an empty one is indistinguishable from a
    // clean crate. Pin it. The number moves whenever the crate's public
    // surface does, which is the point: this is the denominator every
    // conclusion below is drawn against, and a conclusion drawn against a
    // denominator nobody looked at is not a measurement.
    assert!(
        declared.len() > 300,
        "scanned only {} `pub fn` declarations under {}, and this crate has \
         several hundred. Either the scan is looking at the wrong tree -- a \
         `git worktree` sharing CARGO_TARGET_DIR bakes the WRONG \
         CARGO_MANIFEST_DIR into an already-built test binary, and \
         `cargo clean -p driver-metal` is the recovery -- or `declarations` \
         no longer recognises the form this crate writes.",
        declared.len(),
        crate_root.display(),
    );

    let mut used: BTreeMap<String, usize> = BTreeMap::new();
    let mut all = Vec::new();
    sources(workspace, &mut all);
    for file in &all {
        let text = std::fs::read_to_string(file).unwrap_or_default();
        applications(&text, &mut used);
    }

    let unread: BTreeSet<&str> = declared
        .keys()
        .filter(|name| used.get(*name).copied().unwrap_or(0) == 0)
        .map(String::as_str)
        .collect();
    let excused: BTreeSet<&str> = NO_READER.iter().map(|(name, _)| *name).collect();

    let unexplained: Vec<&&str> = unread.difference(&excused).collect();
    assert!(
        unexplained.is_empty(),
        "these `pub fn`s have no caller anywhere under {} and no entry in \
         NO_READER: {:?}\n\nBefore adding one, check that the name is not \
         reached by a macro or a registry -- `kernels-macros` turns \
         `#[routine(no_join)]` into a call this scan cannot see, and a \
         `build.rs` can generate one too. If it really has no reader, \
         either delete it or add a line saying who its reader will be.",
        workspace.display(),
        unexplained,
    );

    let retired: Vec<&&str> = excused.difference(&unread).collect();
    assert!(
        retired.is_empty(),
        "NO_READER still excuses these, but something calls them now: {:?}. \
         Delete the entries -- an excuse for a function that has a reader is \
         a false statement in the file whose job is to prevent them.",
        retired,
    );
}
