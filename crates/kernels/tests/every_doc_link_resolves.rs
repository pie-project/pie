//! Every relative link in a markdown file has to land on a file this
//! repository carries.
//!
//! A sibling of `every_cited_source_exists.rs`, which makes the same argument
//! about kernel paths named in Rust prose. The argument is that a path written
//! down is a CLAIM: it says "this exists, and it is here". Prose that names a
//! file nobody moved is free. Prose that names a file somebody deleted is
//! worse than no prose at all, because a reader who follows it and finds
//! nothing cannot tell whether the file went away or they mistyped it, and the
//! sentence around the link keeps asserting whatever it asserted.
//!
//! A markdown link is the sharpest form of that claim, because a reader does
//! not have to type anything: they click, and GitHub answers 404. There is no
//! interpretation left to do.
//!
//! # What this was written for
//!
//! Twenty-two links across twenty-two inferlet READMEs pointed at
//! `../../../inference-time-algorithms/10-implementation-faithfulness-audit.md`
//! and `../../../inference-time-algorithms/11-ptir-limits.md`. From
//! `tests/inferlets/<name>/README.md`, three levels up is `tests/`'s
//! grandparent -- the repository root -- and the fourth component then leaves
//! it. Those two documents live in a working tree beside this one, not inside
//! it, so the link could never resolve for anyone but the person who has both
//! checked out side by side. Every one of them was a guaranteed 404.
//!
//! The citations themselves were not wrong, and they were not removed. The
//! sibling `src/lib.rs` files had been naming the same two documents as bare
//! code spans -- `inference-time-algorithms/10-implementation-faithfulness-
//! audit.md` -- all along, which states exactly as much and promises nothing
//! about where the reader will find it. The READMEs say it that way now too.
//!
//! # The rule this gate encodes
//!
//! A link to something OUTSIDE the repository is a URL, not a path. A path is
//! for what the repository carries. If a document lives elsewhere and there is
//! no public URL for it, name it in a code span and let the reader go looking
//! -- an honest dead end beats a link that pretends.

use std::path::{Path, PathBuf};

/// The repository root: this crate's manifest directory is `crates/kernels`.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels has two ancestors")
        .to_path_buf()
}

/// Directories that hold build output rather than authored text. `target/`
/// carries vendored crates' own READMEs by the hundred, and a dependency's
/// broken link is not this repository's claim to answer for.
///
/// `.wiki/` is here for a different reason, and it is the more interesting
/// one. That tree is a design-notes ARCHIVE -- `design-old.md`,
/// `northstar-old.md`, `progress-cuda.md`, `real-metal-north-star.md` -- whose
/// purpose is to record what was true when each note was written. Several of
/// its links name files that have since been deleted, including
/// `crates/driver-metal/src/lowering/consts.rs` and a whole
/// `crates/kernels-cuda-new` crate. Repointing those would not repair the
/// notes; it would falsify them. A note that says "the plan was to put this in
/// `kernels-cuda-new`" is a correct sentence about a plan, and the fact that
/// the crate never survived is the note's subject matter, not a defect in it.
///
/// So the rule this gate enforces applies to documentation that describes the
/// repository AS IT IS. An archive describes the repository as it was, and the
/// two cannot be held to the same standard by the same test.
const SKIPPED_DIRS: &[&str] = &["target", ".git", ".wiki", "node_modules", "__pycache__"];

/// Every `*.md` under `root`, in no particular order.
fn markdown(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if !SKIPPED_DIRS.contains(&name.as_ref()) {
                markdown(&path, out);
            }
        } else if name.ends_with(".md") {
            out.push(path);
        }
    }
}

/// The destinations of every inline link in `text`.
///
/// Deliberately narrow: `](` up to the closing paren, which is the form every
/// link in this repository uses. Reference-style definitions and autolinks are
/// not matched, and if one ever appears this gate says nothing about it rather
/// than guessing.
fn links(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i + 1 < bytes.len() {
        if bytes[i] == b']' && bytes[i + 1] == b'(' {
            let start = i + 2;
            let Some(end) = text[start..].find(')') else {
                break;
            };
            let target = &text[start..start + end];
            // A link whose target holds whitespace is either a titled link
            // (`](path "title")`) or not a link at all; neither is a bare path
            // claim, so it is left alone.
            if !target.is_empty() && !target.contains(char::is_whitespace) {
                out.push(target.to_string());
            }
            i = start + end + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// True for the targets that are not filesystem paths at all.
///
/// A same-document anchor (`#section`) and an absolute URL both name something
/// this gate has no way to check and no business checking.
fn is_not_a_path(target: &str) -> bool {
    target.starts_with('#')
        || target.starts_with("http://")
        || target.starts_with("https://")
        || target.starts_with("mailto:")
        || target.starts_with("//")
}

/// Every relative markdown link resolves to a file or directory that is here.
#[test]
fn no_markdown_link_points_outside_or_at_nothing() {
    let root = repo_root();
    let mut files = Vec::new();
    markdown(&root, &mut files);
    assert!(
        files.len() > 30,
        "found only {} markdown files under {}; the walk is broken, and a gate \
         that scans nothing passes for the wrong reason. There were 37 outside \
         `.wiki/` when this was written, 32 of them inferlet READMEs",
        files.len(),
        root.display()
    );

    let mut dangling = Vec::new();
    for file in &files {
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        let dir = file.parent().expect("a file has a parent");
        for target in links(&text) {
            if is_not_a_path(&target) {
                continue;
            }
            // The fragment names a heading inside the target, not a second
            // path component; headings drift and this gate does not chase them.
            let path = target.split('#').next().unwrap_or(&target);
            if path.is_empty() {
                continue;
            }
            let resolved = normalize(&dir.join(path));
            if !resolved.exists() {
                let shown = file.strip_prefix(&root).unwrap_or(file);
                dangling.push(format!("{} -> {target}", shown.display()));
            }
        }
    }
    dangling.sort();
    assert!(
        dangling.is_empty(),
        "these markdown links resolve to nothing:\n  {}\n\nEither the file \
         moved and the link should follow it, or the target lives outside this \
         repository -- in which case it is not a path. Name it in a code span, \
         the way the inferlet `src/lib.rs` headers name \
         `inference-time-algorithms/10-implementation-faithfulness-audit.md`, \
         and the claim stays true without promising a click that cannot work.",
        dangling.join("\n  ")
    );
}

/// `..` collapsed lexically.
///
/// `Path::canonicalize` cannot be used here: it fails on a path that does not
/// exist, which is exactly the case being detected, and it would also resolve
/// symlinks the reader never sees.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for part in path.components() {
        match part {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// The walk really does reach the inferlet READMEs the escaping links were in.
///
/// Without this, a `SKIPPED_DIRS` entry that grew too broad -- or a `tests/`
/// tree moved under one -- would empty the scan and leave the gate above
/// passing on nothing. The count assert there catches a total collapse; this
/// catches losing the one directory that motivated the gate.
#[test]
fn the_walk_reaches_the_inferlet_readmes() {
    let root = repo_root();
    let mut files = Vec::new();
    markdown(&root, &mut files);
    let inferlets = root.join("tests").join("inferlets");
    let found = files.iter().filter(|f| f.starts_with(&inferlets)).count();
    assert!(
        found >= 20,
        "the walk found {found} markdown files under tests/inferlets; there \
         were 32 READMEs when this gate was written, and twenty-two of them \
         held the links it exists to forbid"
    );
}
