//! Regeneration guard for goldens whose source of truth no longer exists.

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
pub fn regenerate_foreign(path: &Path, header: &str, body: &str) {
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
    std::fs::write(path, text).unwrap_or_else(|error| {
        panic!("{} could not be rewritten ({error})", path.display())
    });
}
