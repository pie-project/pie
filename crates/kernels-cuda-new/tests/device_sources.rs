//! Two claims about the device text this crate owns, rehomed from the archive.
//!
//! Both lived in `kernels-cuda/tests/sources.rs` until that crate was deleted,
//! and neither was ever really about it. The first walked THIS crate's `csrc/`
//! from inside that one; the second was quantified over a set this crate
//! produces. A guard whose subject and whose host are two different crates
//! survives only until one of them is deleted, and then it goes silently —
//! which is what made finding them the precondition for the deletion rather
//! than a step in it.
//!
//! # Why the counts here are derived and not pinned
//!
//! At the move the walk saw 121 `.cuh` holding 371 `__global__` definitions
//! under 371 distinct qualified names. Those three numbers are recorded here
//! as an observation with a date on it and are asserted NOWHERE: a number that
//! names a length is the defect this tree has caught most often, because it
//! agrees with the tree on the day it is written and afterwards only ever
//! agrees with the past. Every assertion below is structural — it compares two
//! things the same walk derived, so it stays true as the tree grows and fails
//! only when the property genuinely breaks.

use std::path::{Path, PathBuf};

/// This crate's device sources.
fn csrc_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/src")
}

/// Every `.cuh` under `csrc/src`, as absolute paths.
///
/// Panics rather than returning empty if the root is missing. The version of
/// this walk in the archive returned `Vec::new()` on a failed `read_dir` and
/// skipped unreadable files with `if let Ok(..)`, so a moved directory or a
/// permission error would have made every caller pass by finding nothing.
fn device_headers() -> Vec<PathBuf> {
    let root = csrc_src();
    assert!(
        root.is_dir(),
        "{root:?} does not read. Every assertion in this file is quantified \
         over the files under it, so a missing root makes all of them \
         vacuously true — the failure this walk is written to refuse."
    );
    let mut out = Vec::new();
    walk(&root, &mut out);
    out.sort();
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{dir:?} reads: {e}"));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("{dir:?} entry reads: {e}"))
            .path();
        if path.is_dir() {
            if path
                .file_name()
                .is_some_and(|n| n == "third_party" || n == "vendor")
            {
                continue;
            }
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "cuh") {
            out.push(path);
        }
    }
}

/// A `__global__` may be defined once.
///
/// Two definitions of one name is a half-finished migration: one copy gets
/// edited and the other drifts, each right for whichever half of the tree its
/// tests exercise. `norm/altup_aux` shipped exactly that for a release with
/// every gate green, which is why this is a test and not a review note.
///
/// Names are compared QUALIFIED. A bare leaf is not an identity — `k_matmul`
/// is a `kernels::ptir` template and an anonymous-namespace helper in `model`,
/// and those are two kernels that share a spelling, which is what a namespace
/// is for. Comparing leaves would report them and teach a reader to skip this.
#[test]
fn no_global_is_defined_twice() {
    let headers = device_headers();
    let mut seen: Vec<(String, PathBuf)> = Vec::new();
    let mut clashes: Vec<String> = Vec::new();

    for path in &headers {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?} reads: {e}"));
        let mut ns = String::new();
        for line in text.lines() {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed.strip_prefix("namespace ")
                && let Some(named) = rest.split(&[' ', '{'][..]).next()
                && !named.is_empty()
            {
                ns = named.to_string();
            }
            let Some(after) = trimmed.strip_prefix("__global__ void ") else {
                continue;
            };
            let leaf: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if leaf.is_empty() {
                continue;
            }
            let name = format!("{ns}::{leaf}");
            if let Some((_, first)) = seen.iter().find(|(n, _)| *n == name) {
                clashes.push(format!(
                    "{name}: {} and {}",
                    first.display(),
                    path.display()
                ));
            } else {
                seen.push((name, path.clone()));
            }
        }
    }

    // Anti-vacuity, both halves derived by the walk that just ran rather than
    // compared against a remembered number. The first fails if the tree stops
    // holding device headers; the second fails if it holds them and the
    // scanner stops recognising a definition — a changed `__global__` spelling
    // or a formatter putting the return type on its own line. Either one turns
    // the assertion below into a statement about nothing.
    assert!(
        !headers.is_empty(),
        "{:?} holds no `.cuh` at all, so this test passes by looking at \
         nothing.",
        csrc_src()
    );
    assert!(
        !seen.is_empty(),
        "the walk read {} device headers and found no `__global__ void` in \
         any of them. The files are there and the scanner is not seeing \
         them, which is the quietest way for this test to stop testing.",
        headers.len()
    );

    assert!(
        clashes.is_empty(),
        "a `__global__` is defined in two places, so a migration was left \
         half-done and the two copies can drift. {} definitions under {} \
         distinct names across {} headers:\n  {}",
        seen.len() + clashes.len(),
        seen.len(),
        headers.len(),
        clashes.join("\n  ")
    );
}

/// The ahead-of-time C shim emits NOTHING, and that is the terminal state.
///
/// # What this replaces, and why the replacement is not the same claim
///
/// `kernels-cuda`'s `every_emitted_shim_entry_has_a_launcher` took each entry
/// `emit_c_shim` produced and looked for a launcher declaration in a C++
/// corpus spanning three trees, one of them the archive's own `csrc/src`. It
/// opened with `assert!(wanted.len() > 40)` — an anti-vacuity guard against
/// the emitter silently producing nothing.
///
/// That guard cannot tell "the emitter broke" from "the tree finished", and
/// the tree has finished. `abi::stated()` keeps only rows with a non-empty
/// `operands`; `table::TABLES` is `ROW_TABLES` — `&[]` — concatenated with
/// `x::SIGS`; and every sig in `x::SIGS` is a `Contract::sig()`, which takes
/// `operands` from `SIG_BASE` where it is `&[]` and never overrides it. So the
/// emitted set is empty by construction, and the guard written to catch an
/// empty set now fires on the correct answer.
///
/// The honest replacement is not to widen the guard but to invert it: assert
/// the set IS empty. That claim needs no launcher corpus and no `.hpp`, which
/// is the whole point — it is what let the archive's seventeen host headers go
/// without taking a check with them. It is falsifiable in the direction that
/// matters: it goes red the moment anything re-introduces a shim entry.
///
/// # The population half
///
/// "Emits nothing" is also what a table with no rows in it emits, so the
/// emptiness is asserted together with the two facts that make it meaningful —
/// that there are sigs to emit from, and that the reason they emit nothing is
/// the empty `operands` rather than an emitter that stopped running.
#[test]
fn the_ahead_of_time_shim_emits_nothing() {
    let tables = kernels_cuda_new::table::TABLES.to_vec();

    let sigs: Vec<_> = tables.iter().flat_map(|t| t.iter()).collect();
    assert!(
        !sigs.is_empty(),
        "`table::TABLES` is empty, so `emit_c_shim` emitting nothing says \
         nothing. This test would then be a statement about an absent \
         population rather than about the shim."
    );

    let stated: Vec<&str> = sigs
        .iter()
        .filter(|k| !k.operands.is_empty())
        .map(|k| k.symbol)
        .collect();
    assert!(
        stated.is_empty(),
        "{} of {} rows now carry operands, so `abi::stated()` keeps them and \
         the ahead-of-time shim has entries again. Nothing in this workspace \
         compiles the generated `shim.cpp` — there is no nvcc, no CMake and \
         no `cc` step left — so an entry is a forwarder to a host launcher \
         that no longer exists in C++ anywhere. If an ahead-of-time launcher \
         is genuinely wanted again, that is a decision to argue in \
         `new-horizon.md`:\n  {}",
        stated.len(),
        sigs.len(),
        stated.join("\n  ")
    );

    let shim = kernels_cuda_new::abi::emit_c_shim(
        &tables,
        &[],
        &kernels_cuda_new::device::jit_dispatched(),
    )
    .expect("two rows may not claim one entry point");

    const MARK: &str = "&::pie_cuda_driver::kernels::";
    let entries: Vec<String> = shim
        .match_indices(MARK)
        .map(|(i, _)| {
            shim[i + MARK.len()..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == ':')
                .collect()
        })
        .collect();

    assert!(
        entries.is_empty(),
        "`emit_c_shim` produced {} forwarder(s) from a table whose rows all \
         have empty operands. The two derivations disagree, so one of them is \
         wrong — either `stated()` no longer filters on `operands`, or the \
         marker `{MARK}` has changed shape in `abi::cpp_path` and this scan \
         is reading something else:\n  {}",
        entries.len(),
        entries.join("\n  ")
    );
}
