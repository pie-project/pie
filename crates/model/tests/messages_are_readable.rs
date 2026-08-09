//! Every message an operator reads is one an operator can read.
//!
//! # The slip this catches
//!
//! A refusal or panic message long enough to wrap gets written as a
//! multi-line string. Rust has two ways to spell that, and only one is
//! right:
//!
//! ```text
//! "the row ships no v_proj \        <- the backslash eats the newline
//!      and the text projects one"      AND the indent. Correct.
//!
//! "the row ships no v_proj
//!      and the text projects one"   <- keeps BOTH. The operator reads
//!                                      five words, twenty spaces, and
//!                                      five more.
//! ```
//!
//! and a third way that is neither: an editor or a script joining the
//! two source lines, which drops the newline and leaves the indent
//! behind. That is what produced every site this test was written
//! against — nine of them, across six crates, each a message someone
//! would eventually read out of a log with a run of spaces jammed
//! through the middle of a sentence.
//!
//! # Why a test and not a lint
//!
//! Nothing else looks. `rustfmt` does not reformat inside a string
//! literal, on purpose — it cannot know whether the spaces are
//! meaningful. `clippy` has no lint for it. And the compiler is
//! perfectly happy: the string is well-formed, it is just wrong about
//! what it says. A garbled message is the one defect class that gets
//! LESS attention the worse it gets, because the reader assumes the
//! garbling is the log's fault.
//!
//! # What it does NOT flag
//!
//! Runs of spaces that are doing work: alignment inside a table, an
//! ASCII diagram, an indent in a `{:>width$}` template. The predicate
//! is narrow on purpose — a run of six or more spaces landing between
//! two pieces of ORDINARY PROSE (lowercase word, spaces, letter). A
//! table's cells are rarely that, and a diagram's are never.

use std::path::{Path, PathBuf};

/// The workspace's `crates/` directory.
///
/// This test scans every crate, not just `model`, because the slip is
/// not a `model` habit — it is a property of how long strings get
/// edited, and it landed in `driver`, `engine`, `model-compiler` and
/// `driver-metal` too. [`deployment_is_read`] already reads outside
/// this crate for the same reason: the thing being measured does not
/// live inside one crate's boundary.
///
/// [`deployment_is_read`]: ../deployment_is_read.rs
fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/model has a parent")
        .to_path_buf()
}

/// The workspace root.
///
/// `crates/` is not all of the workspace: the `pie` binary's own `src/`
/// and `tests/` sit beside it, and stopping at `crates/` left them
/// unscanned by an argument that never applied to them. The slip is a
/// property of editing long strings, and the root edits long strings.
///
/// It measured clean when this widened, so nothing was fixed to make it
/// pass — which is the point of doing it while it is free rather than
/// after the first offender lands.
fn workspace_root() -> PathBuf {
    crates_dir()
        .parent()
        .expect("crates/ has a parent")
        .to_path_buf()
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // `target/` holds generated sources nobody edits.
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Whether `run` — a stretch of spaces inside a string literal — sits
/// between two pieces of prose.
///
/// `before` ends at the run and `after` begins at it. Prose on the left
/// is at least three lowercase letters: a word, not a column heading, a
/// hex digit, or the tail of a `{}`. Prose on the right is a letter or
/// an opening quote/bracket that would start one.
fn splits_prose(before: &str, after: &str) -> bool {
    let tail: Vec<char> = before.chars().rev().take(3).collect();
    let lhs = tail.len() == 3 && tail.iter().all(|c| c.is_ascii_lowercase());
    let rhs = after
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '(' || c == '`' || c == '\'');
    lhs && rhs
}

/// The width at which a run of spaces stops being a gap and starts
/// being an indent.
///
/// Six. A sentence never has six spaces in it, and the shortest indent
/// that could be swallowed is a continuation line inside a nested
/// block, which is deeper than that in every case here — the nine
/// original sites carried runs of 14 to 22.
const INDENT_WIDTH: usize = 6;

fn collapsed_literals(src: &str) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    for (idx, line) in src.lines().enumerate() {
        // Only inside a string literal, and only one per line: the
        // slip joins two source lines, so the result is always a
        // single-line literal.
        let Some(open) = line.find('"') else { continue };
        let rest = &line[open + 1..];
        let Some(close) = rest.rfind('"') else {
            continue;
        };
        let body = &rest[..close];
        let mut at = 0;
        while let Some(pos) = body[at..].find(&" ".repeat(INDENT_WIDTH)) {
            let start = at + pos;
            let end = start
                + body[start..]
                    .find(|c: char| c != ' ')
                    .unwrap_or(body.len() - start);
            if splits_prose(&body[..start], &body[end..]) {
                let shown: String = body.chars().take(90).collect();
                found.push((idx + 1, shown));
                break;
            }
            at = end.max(start + 1);
        }
    }
    found
}

#[test]
fn no_message_carries_a_swallowed_indent() {
    let mut sources = Vec::new();
    rust_sources(&crates_dir(), &mut sources);
    rust_sources(&workspace_root().join("src"), &mut sources);
    rust_sources(&workspace_root().join("tests"), &mut sources);
    assert!(
        sources.len() > 200,
        "the scan found only {} sources; it is not reaching the workspace",
        sources.len()
    );

    let mut offenders = Vec::new();
    for path in &sources {
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        for (line, text) in collapsed_literals(&src) {
            let rel = path.strip_prefix(workspace_root()).unwrap_or(path);
            offenders.push(format!("  {}:{line}\n    {text}", rel.display()));
        }
    }

    assert!(
        offenders.is_empty(),
        "{} message(s) carry a run of {INDENT_WIDTH}+ spaces through the middle of a \
         sentence — a source line joined without dropping its indent. Write the \
         continuation with a trailing `\\` so the newline AND the indent are eaten:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

/// The predicate itself, on the shapes it must tell apart.
///
/// Without this the test above could pass by being blind, which is the
/// failure mode every source scan has: a scan that matches nothing
/// reports the same green as a codebase that is clean.
#[test]
fn the_predicate_tells_prose_from_alignment() {
    // Assembled rather than written: a literal carrying the defect
    // would be a real offender, and the scan above reads this file.
    let gap = " ".repeat(12);
    let fixture = format!("    \"the row ships no v_proj{gap}and the text projects one\",");
    let caught = collapsed_literals(&fixture);
    assert_eq!(caught.len(), 1, "a swallowed indent must be caught");

    for benign in [
        // A table: the run follows a heading or a number, not a word.
        "    \"NAME            WIDTH       PAGES\",",
        "    println!(\"{:>8}            {}\", a, b);",
        // A diagram.
        "    \"  +---+            +---+\",",
        // A short gap: two spaces after a period is not an indent.
        "    \"one sentence.  another sentence.\",",
    ] {
        assert!(
            collapsed_literals(benign).is_empty(),
            "flagged alignment as prose: {benign}"
        );
    }
}
