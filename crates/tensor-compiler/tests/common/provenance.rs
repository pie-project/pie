// Included by several test binaries via `#[path]`; each uses a different
// subset, so "never used" here means "unused by *this* binary".
#![allow(dead_code)]

//! The shape of a checked-in golden dump, and the guard on rewriting one.
//!
//! Every golden here is a `#` provenance header followed by a body of
//! `=== <id>` cases. The header records where the bytes came from and never
//! takes part in a comparison, so splitting the two is one rule and it lives
//! next to the guard that decides when the header may be overwritten.

use std::path::Path;

const STAMP: &str = "# REGENERATED: the body below is this compiler's output, not the recorded output of the source of truth above";
const REASON_PREFIX: &str = "#   reason: ";

/// Rewrite `path` as `header` + `body`, recording in the header that the body
/// is no longer what the header claims produced it.
///
/// `golden-{msl,cuda}/` and `golden-stage-identity.txt` are dumps of C++ that
/// is gone: `32c2a4a09` deleted the oracle harnesses, and the emitters and
/// `program_identity.hpp` they compiled went before that. Their `#` headers are
/// the only surviving statement of where the bytes came from, and the old
/// regeneration path copied those headers verbatim onto a body this compiler
/// produced. One `PTIR_REGEN=1` therefore turned an oracle dump into a
/// self-portrait that still advertised oracle provenance -- silently, in a file
/// whose body diff is too large to read, and with nothing left to re-derive the
/// original from.
///
/// These dumps stay a usable baseline for a rewrite; what they cannot do is
/// adjudicate. They record what the oracle *did*, which is not always what it
/// should have done: `INTENDED_ORACLE_DIVERGENCES` in `metal_msl_golden.rs`
/// lists two cases where it emitted a kernel that read the wrong buffer. So a
/// mismatch during a rewrite is a question, not an answer, and blessing it is a
/// claim that has to be written down.
///
/// A regeneration that produces the bytes already on disk is not a rewrite and
/// is left alone. The thing worth guarding is evidence being *discarded*, not
/// the command being run: stamping a file whose body still matches the oracle
/// would retract a true provenance claim for nothing. That is not hypothetical
/// either — `golden-stage-identity.txt` was stamped by a regeneration that
/// changed none of its 21 values.
fn regenerate_foreign(path: &Path, header: &str, body: &str) {
    if let Some(existing) = Golden::read(path)
        && existing.body == body
    {
        return;
    }
    let reason = std::env::var("PTIR_REGEN_REASON").unwrap_or_default();
    let reason = reason.trim();
    assert!(
        !reason.is_empty(),
        "{} records the output of a source of truth that no longer exists, so \
         overwriting it discards evidence that cannot be recovered. Set \
         PTIR_REGEN_REASON=\"why the new bytes are correct\" to regenerate it \
         anyway; the reason is written into the file.",
        path.display()
    );
    let mut lines: Vec<String> = header
        .lines()
        .filter(|line| *line != STAMP && !line.starts_with(REASON_PREFIX))
        .map(str::to_string)
        .collect();
    lines.push(STAMP.to_string());
    lines.push(format!("{REASON_PREFIX}{reason}"));
    let text = lines.join("\n") + "\n" + body;
    std::fs::write(path, text)
        .unwrap_or_else(|error| panic!("{} could not be rewritten ({error})", path.display()));
}

/// A checked-in golden: its `#` provenance header, and the body under test.
struct Golden {
    header: String,
    body: String,
}

impl Golden {
    fn read(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let header: String = text
            .lines()
            .take_while(|line| line.starts_with('#'))
            .map(|line| format!("{line}\n"))
            .collect();
        let body = text[header.len()..].to_string();
        Some(Self { header, body })
    }

    fn expect(path: &Path, advice: &str) -> Self {
        Self::read(path).unwrap_or_else(|| panic!("{} missing; {advice}", path.display()))
    }
}

/// What re-pinning a golden costs.
pub enum Regenerate<'a> {
    /// This compiler is the source of truth, so `PTIR_REGEN=1` means no more
    /// than "the new output is the output". `header` is written above it.
    Own { header: &'a str },
    /// The source of truth is gone, so the body is the only record of it.
    /// See [`regenerate_foreign`] for what that costs and what it demands.
    Foreign,
}

/// The recorded body to diff against, or `None` when there is nothing left to
/// compare — either the caller was asked to regenerate, or it already matches.
///
/// Every golden here is read through this, so the choice between the two
/// regeneration policies is made at the call site and visible there. It used
/// to be implicit in which comparator someone had copied, and the header split
/// that decides what counts as provenance was written out four times.
pub fn body_to_diff(path: &Path, body: &str, how: Regenerate) -> Option<String> {
    const MISSING: &str = "re-pin it with PTIR_REGEN=1";
    const MISSING_FOREIGN: &str =
        "it records a source of truth that is gone, so it cannot simply be re-pinned";
    if std::env::var("PTIR_REGEN").is_ok() {
        match how {
            Regenerate::Own { header } => {
                if let Some(dir) = path.parent() {
                    std::fs::create_dir_all(dir)
                        .unwrap_or_else(|e| panic!("{} could not be created ({e})", dir.display()));
                }
                std::fs::write(path, format!("{header}{body}"))
                    .unwrap_or_else(|e| panic!("{} could not be written ({e})", path.display()));
            }
            // The header comes off the file rather than from the caller: a
            // second regeneration must not drop the first one's recorded
            // reason.
            Regenerate::Foreign => {
                let golden = Golden::expect(path, MISSING_FOREIGN);
                regenerate_foreign(path, &golden.header, body);
            }
        }
        return None;
    }
    let golden = Golden::expect(
        path,
        match how {
            Regenerate::Own { .. } => MISSING,
            Regenerate::Foreign => MISSING_FOREIGN,
        },
    );
    (golden.body != body).then_some(golden.body)
}

/// Report the first line on which two case-structured dumps disagree, naming
/// the `=== ` case it falls in.
///
/// A line number on its own is no use: these dumps run to tens of thousands of
/// lines and the number moves whenever anything above it does.
pub fn assert_same_lines(mine: &str, theirs: &str, what: &str, hint: &str) {
    let mut case = String::from("<before the first case>");
    for (index, (mine, theirs)) in mine.lines().zip(theirs.lines()).enumerate() {
        if let Some(id) = theirs.strip_prefix("=== ") {
            case = id.to_string();
        }
        assert_eq!(
            mine,
            theirs,
            "{what} diverged at line {} (case `{case}`){hint}",
            index + 1
        );
    }
    assert_eq!(
        mine.lines().count(),
        theirs.lines().count(),
        "{what} produced a different number of lines"
    );
}
