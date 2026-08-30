//! The edges the crate system used to hold, now that the crates are modules.
//!
//! `plan`, `eval` and `codegen` were three crates over `eta-ir`. Two facts
//! about that graph were load-bearing, and both were enforced by cargo simply
//! refusing to resolve a dependency that did not exist:
//!
//! 1. **`plan` ⊥ `eval`.** They were siblings with no edge between them.
//!    Planning decides HOW a trace executes; the interpreter says WHAT it
//!    means. The interpreter's entire value as a tier-0 oracle rests on its
//!    not having seen the planner's answer — a backend diffed against an
//!    interpreter that shares the planner's assumptions is diffed against
//!    itself. `eval` did name `plan` once, in DEV-dependencies, for the parity
//!    tests; that direction is preserved here by this file being a test.
//! 2. **`codegen` is the top.** It reads `plan`; nothing reads it but the
//!    battery. An emitter's output is checked, never consumed in-tree.
//!
//! Folded into one crate, `use crate::eval::…` from inside `plan` compiles
//! fine. So the rule that was a build error becomes this test, which is a
//! textual check and honest about being one: it reads the module sources and
//! fails on a path that crosses an edge the graph did not have.

use std::path::{Path, PathBuf};

fn src(module: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join(module)
}

/// Every `.rs` under `dir`, with its repo-relative-ish label for diagnostics.
fn sources(dir: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("module directory exists") {
            let path = entry.expect("readable entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let label = path
                    .strip_prefix(Path::new(env!("CARGO_MANIFEST_DIR")))
                    .unwrap_or(&path)
                    .display()
                    .to_string();
                out.push((
                    label,
                    std::fs::read_to_string(&path).expect("readable source"),
                ));
            }
        }
    }
    assert!(!out.is_empty(), "no sources under {}", dir.display());
    out
}

/// A mention that is PRODUCTION CODE.
///
/// Two things are not: prose (the module names are all over the doc comments
/// explaining this very layering), and test code. The second is not a
/// loophole — it is the old graph restated. `eta-compiler` DID depend on
/// `eta-compiler`, as a dev-dependency, so its tier-0 parity tests could check the
/// interpreter against planner-partitioned regions. What cargo forbade was the
/// production edge, and that is what this forbids.
///
/// Test code is recognized two ways, because it is written two ways: an
/// inline `#[cfg(test)] mod tests` (everything after the attribute counts),
/// and an out-of-line `mod tests;` whose body is a whole `tests.rs` file —
/// there the `#[cfg(test)]` sits in the PARENT, so the file is judged by its
/// name instead (see [`is_test_file`]).
///
/// The inline form over-approximates: a `#[cfg(test)] mod` followed by more
/// production code would be let through. The alternative is parsing Rust, and
/// these modules put their test module last, as the convention has them do.
fn code_mentions(text: &str, other: &str) -> Vec<String> {
    let needle = format!("crate::{other}::");
    let mut in_test_code = false;
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#[cfg(test)]") {
            in_test_code = true;
        }
        if in_test_code || trimmed.starts_with("//") {
            continue;
        }
        if line.contains(&needle) {
            out.push(line.trim().to_string());
        }
    }
    out
}

/// An out-of-line test module: `mod tests;` in the parent, body in `tests.rs`.
/// The `#[cfg(test)]` gating it is in the parent file, so the name is the only
/// signal this file has.
fn is_test_file(label: &str) -> bool {
    label.ends_with("tests.rs") || label.contains("/tests/")
}

fn assert_does_not_name(module: &str, other: &str, why: &str) {
    let mut offenders = Vec::new();
    for (label, text) in sources(&src(module)) {
        if is_test_file(&label) {
            continue;
        }
        for line in code_mentions(&text, other) {
            offenders.push(format!("  {label}: {line}"));
        }
    }
    assert!(
        offenders.is_empty(),
        "`{module}` names `{other}`, which the crate graph did not allow.\n{why}\n{}",
        offenders.join("\n")
    );
}

#[test]
fn eval_does_not_name_plan() {
    assert_does_not_name(
        "eval",
        "plan",
        "The interpreter is the oracle backends are diffed against. Once it can \
         see the planner's decisions it is no longer an independent answer, and \
         a backend that agrees with it proves nothing. Parity tests may go the \
         other way -- that is what `tests/` is for.",
    );
}

#[test]
fn plan_does_not_name_eval() {
    assert_does_not_name(
        "plan",
        "eval",
        "Planning is backend-neutral and infallible over a bound trace. Reaching \
         the interpreter would make an execution strategy depend on running the \
         program, which is the one thing a plan is supposed to precede.",
    );
}

#[test]
fn nothing_below_codegen_names_it() {
    for module in ["plan", "eval"] {
        assert_does_not_name(
            module,
            "codegen",
            "`codegen` is the top of the toolchain: it reads plans and emits \
             source. A module below it reaching up inverts the layering the \
             three crates had.",
        );
    }
}
