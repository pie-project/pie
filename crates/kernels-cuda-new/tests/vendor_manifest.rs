//! `MODIFICATIONS` is a claim about the vendored tree. This checks it.
//!
//! # The claim, and why prose was not enough
//!
//! `csrc/vendor/MODIFICATIONS` opens by asserting a property rather than
//! describing one:
//!
//! > *"Generated from the tree, and kept honest by the property it asserts:
//! > strip every `// PIE:` marker, the `#ifndef __CUDACC_RTC__` under it and
//! > that directive's matching `#endif`, and each file below is byte-for-byte
//! > FlashInfer v0.6.15. That was checked per file before this list was
//! > written."*
//!
//! **"Was checked" is a past-tense fact about a person, not a property of the
//! tree.** Nothing re-checks it. A guard widened by one line, a stray edit
//! inside a `#ifndef __CUDACC_RTC__` block, or a row left stale after a file
//! changed all leave a `MODIFICATIONS` that reads correct and is not — and
//! the whole value of vendoring upstream *byte-identically* is that a reader
//! can diff against upstream and a bump is a re-fetch rather than a merge.
//!
//! # What this test can check without a network, and what it cannot
//!
//! Byte-identity against upstream needs upstream, and a unit test must not
//! fetch. So the property is split, and this file takes the half that is a
//! function of the tree alone — the three columns:
//!
//! * `guards` — the count of `// PIE:` markers in the file;
//! * `lines` — the file's line count;
//! * `added` — how many lines the strip removes, which is what the guards
//!   cost over upstream.
//!
//! Every one of those is recomputable here, and a drift that changes the
//! vendored bytes changes at least one of them unless it is *exactly*
//! line-count-neutral outside every guard. That residue is what the
//! network-side check covers, and `.wiki/driver/new-horizon.md` §23.8 records
//! it being run: **28 of 28 vendored FlashInfer files were byte-identical to
//! v0.6.15 after stripping**, with two negative controls proving the checker
//! can fail.
//!
//! So: this test catches the drift a maintainer actually causes (edit a
//! guard, forget the row), and the doc carries the one-time proof of the part
//! that needs the internet.
//!
//! # The fourth column, and the blind spot it closes
//!
//! Three columns were `guards`, `added` and `lines`, and the sentence above
//! — *"a drift that changes the vendored bytes changes at least one of them
//! unless it is exactly line-count-neutral outside every guard"* — was
//! stating a blind spot as if it were a corner case. It is not a corner
//! case. Changing a constant, reflowing an expression, swapping two
//! arguments: every one of those keeps the line count, and all three columns
//! sit still while the file stops being upstream's.
//!
//! `bytes` is the fourth column. It is not a checksum and does not close the
//! hole completely — a length-preserving edit still gets through — but it
//! catches every edit that changes a file's LENGTH, which is nearly all of
//! them, and it has a property a checksum does not: `wc -c` and
//! [`str::len`] cannot disagree about the algorithm, so the number in the
//! manifest and the number this test computes are produced by two
//! implementations that have nothing to get wrong. A digest would be
//! stronger and would need a hash function written here and a hash function
//! run in a shell to agree, which is the failure mode this session has been
//! collecting.
//!
//! # The second half of the transform, standing before it is needed
//!
//! `csrc/` is being re-cut by ROLE rather than by provenance: the shims that
//! impersonate NVIDIA headers into `csrc/shim/`, the device stdlib into
//! `csrc/device/`, the attention algorithms into `csrc/attn/`, so that a
//! `norm` unit stops carrying somebody else's attention library and our own
//! text stops pointing at somebody else's namespace to find a `mma.sync`
//! wrapper.
//!
//! A vendored file that moves has to have its `#include` lines rewritten,
//! and that is the exact moment vendoring turns into a fork. Upstream
//! reaches its own siblings by relative path — 58 quoted directives across
//! the closure, `"../cp_async.cuh"`, `"../utils.cuh"`, and **not one**
//! `<flashinfer/…>` — so no move is spelling-neutral. Once the bytes on
//! disk are not upstream's bytes, the `MODIFICATIONS` claim is prose again.
//!
//! So the claim gets a second step: strip the markers, then rewrite every
//! `#include <pie/…>` back to what upstream wrote, and only THEN is the
//! result byte-identical. [`PIE_INCLUDES`] is that map and
//! [`denormalise_includes`] applies it.
//!
//! **The map is empty today and the step is the identity, and that is the
//! point.** It is standing before the first file moves, because a transform
//! written after the move is a transform derived from the thing it is meant
//! to check. What makes it non-vacuous now is not the map — it is
//! [`the_manifest_check_can_fail_on_every_drift_it_claims_to_catch`], which
//! mutates the real vendored files and counts how many mutations the check
//! rejects. A negative control whose discriminating-input count is unstated
//! is a negative control nobody has checked.
//!
//! # Why the stripper lives in the test rather than the build
//!
//! Nothing in the crate needs it at run time. It exists to answer a question
//! about the repository, which is what a test is for.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn csrc() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc")
}

/// One row of `MODIFICATIONS`: the four numbers it asserts about a file.
#[derive(Debug, PartialEq, Eq)]
struct Row {
    guards: usize,
    added: usize,
    lines: usize,
    bytes: usize,
}

/// What a file measures, which is what its row has to say.
///
/// Extracted from the test that used to compute it inline, because the
/// mutation battery has to run the same measurement over text that is not on
/// disk. A check whose measurement exists only inside the passing path cannot
/// be shown to fail.
fn measure(text: &str) -> Row {
    Row {
        guards: text.lines().filter(|l| l.trim_start().starts_with("// PIE:")).count(),
        added: strip(text).1,
        lines: text.lines().count(),
        bytes: text.len(),
    }
}

/// Parse the table. The format is fixed-ish columns under a `---` rule, and
/// the parse is deliberately strict: a row this cannot read is a row the test
/// would otherwise skip silently, which is how a manifest check goes vacuous.
fn manifest() -> BTreeMap<String, Row> {
    let text = std::fs::read_to_string(csrc().join("vendor/MODIFICATIONS"))
        .expect("csrc/vendor/MODIFICATIONS");
    let mut out = BTreeMap::new();
    let mut in_table = false;
    for line in text.lines() {
        if line.starts_with("---") {
            in_table = true;
            continue;
        }
        if !in_table || line.trim().is_empty() {
            continue;
        }
        // The table ends with its own totals row and then prose. Both are
        // stopped on here; the totals get their own assertion below, because
        // a summary line that drifts from the rows above it is exactly the
        // kind of thing a reader trusts without checking.
        if line.starts_with(char::is_numeric) || !line.starts_with(|c: char| c.is_ascii_graphic())
        {
            in_table = false;
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(file), Some(g), Some(a), Some(l), Some(b)) =
            (it.next(), it.next(), it.next(), it.next(), it.next())
        else {
            continue;
        };
        let (Ok(guards), Ok(added), Ok(lines), Ok(bytes)) =
            (g.parse::<usize>(), a.parse::<usize>(), l.parse::<usize>(), b.parse::<usize>())
        else {
            panic!("MODIFICATIONS row is not four numbers: {line:?}");
        };
        out.insert(file.to_string(), Row { guards, added, lines, bytes });
    }
    assert!(
        out.len() > 20,
        "parsed only {} rows from MODIFICATIONS — the format moved and this \
         test would have passed vacuously",
        out.len()
    );
    out
}

/// The directive a `// PIE:` marker guards, in both spellings this tree uses.
///
/// One definition, called from [`strip`] and from the mutation battery. It
/// was two — a closure inside `strip` and a copy inside the test — and two
/// implementations of one rule that are never fed an input on which they
/// could disagree is the defect this file was extended to stop shipping.
fn is_rtc_guard(line: &str) -> bool {
    let line = line.trim_start();
    line.starts_with("#ifndef __CUDACC_RTC__") || line.starts_with("#if !defined(__CUDACC_RTC__)")
}

/// The strip `MODIFICATIONS` describes: drop a `// PIE:` marker with its
/// continuation comments, and if a `#ifndef __CUDACC_RTC__` follows, drop that
/// directive and its matching `#endif` while KEEPING the body.
///
/// Keeping the body is the point. The guards exist to hide host-only code
/// from NVRTC, so the body is upstream's and must survive the strip; only the
/// three lines this tree added come out.
fn strip(text: &str) -> (String, usize) {
    let src: Vec<&str> = text.lines().collect();
    let mut out: Vec<&str> = Vec::new();
    let mut removed = 0usize;
    let mut i = 0usize;
    while i < src.len() {
        let t = src[i].trim_start();
        if !t.starts_with("// PIE:") {
            out.push(src[i]);
            i += 1;
            continue;
        }
        // the marker and its continuation comment lines
        i += 1;
        removed += 1;
        while i < src.len() {
            let c = src[i].trim_start();
            if c.starts_with("//") && !c.starts_with("// PIE:") {
                i += 1;
                removed += 1;
            } else {
                break;
            }
        }
        // the guard under it, if there is one
        if i < src.len() && is_rtc_guard(src[i]) {
            i += 1;
            removed += 1;
            let mut depth = 1usize;
            while i < src.len() && depth > 0 {
                let d = src[i].trim_start();
                if d.starts_with("#if") {
                    depth += 1;
                } else if d.starts_with("#endif") {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        removed += 1;
                        break;
                    }
                }
                out.push(src[i]);
                i += 1;
            }
        }
    }
    (out.join("\n"), removed)
}

/// The path rewrite a role-cut move applies to a vendored file, written
/// backwards so it can be undone.
///
/// `(file, the directive as it stands in the tree, the directive upstream
/// wrote)`. Whole lines on both sides, not paths: upstream's spelling carries
/// its bracket style and its indentation, and restoring
/// `#include "../cp_async.cuh"` from `#include <pie/device/cp_async.cuh>`
/// needs both. Matching whole lines also makes the map exact where a path map
/// could not be — the same target is reached as `"../vec_dtypes.cuh"` from
/// `attention/decode.cuh` and as `"vec_dtypes.cuh"` from a file one directory
/// up, so the reverse of a canonical `<pie/device/vec.cuh>` is not a function
/// of the target alone.
///
/// # Empty, on purpose, and early on purpose
///
/// Nothing has moved yet. Every vendored file still sits under
/// `csrc/vendor/flashinfer/` reaching its siblings the way upstream wrote,
/// so there is no `<pie/…>` anywhere and this map has nothing to say.
///
/// It is here anyway because of the order the work has to happen in. The
/// first file to move out of this directory changes upstream's bytes — there
/// is no way to move it without changing an `#include` — and at that instant
/// the manifest's claim is either backed by a written-down transform or it is
/// a sentence someone remembers being true. A map built after the move is
/// derived from the move and agrees with it by construction, which is the
/// shape of every check this session has had to throw away.
///
/// A row is added by the same commit that moves the file. A row whose file no
/// longer holds that directive is a dead row and
/// [`every_normalisation_row_is_used_and_no_pie_include_survives`] fails on
/// it, in both directions.
const PIE_INCLUDES: &[(&str, &str, &str)] = &[];

/// Does this line reach for our namespace?
///
/// Spelled as a predicate over the directive rather than as a substring
/// search, because `<pie/` appears in prose inside these files and a comment
/// mentioning the new layout must not be mistaken for an include the map owes
/// a row.
fn is_pie_include(line: &str) -> bool {
    line.trim_start()
        .strip_prefix('#')
        .map(str::trim_start)
        .and_then(|rest| rest.strip_prefix("include"))
        .is_some_and(|rest| rest.trim_start().starts_with("<pie/"))
}

/// Step 2 of the transform: put upstream's `#include` lines back.
///
/// # Errors
///
/// A `#include <pie/…>` with no row in [`PIE_INCLUDES`]. **Refused, never
/// passed through.** Passing it through would produce text that is not
/// upstream's and claim it was — the manifest would go on describing a tree
/// that had quietly become a fork, which is the one failure this whole file
/// exists to make impossible. It is also the only way the map can be caught
/// being incomplete: a move that forgets its row fails here, and nowhere
/// else.
fn denormalise_includes(name: &str, text: &str) -> Result<String, String> {
    let mut out: Vec<&str> = Vec::new();
    for (at, line) in text.lines().enumerate() {
        if !is_pie_include(line) {
            out.push(line);
            continue;
        }
        let Some((_, _, upstream)) =
            PIE_INCLUDES.iter().find(|(file, pie, _)| *file == name && *pie == line)
        else {
            return Err(format!(
                "{name}:{}: `{}` has no row in PIE_INCLUDES, so this file cannot \
                 be turned back into upstream's. Add \
                 (\"{name}\", \"{}\", \"<the line upstream wrote>\") in the same \
                 change that moved the file.",
                at + 1,
                line.trim(),
                line
            ));
        };
        out.push(*upstream);
    }
    Ok(out.join("\n"))
}

/// The whole transform `MODIFICATIONS` claims: strip, then denormalise.
///
/// Returns the recovered text and the line count the strip removed, which is
/// the manifest's `added` column. The second step is line-count-neutral by
/// construction — [`denormalise_includes`] pushes exactly one string per
/// input line, so a rewrite cannot split a directive across two lines and
/// move `lines` for every row at once, which would read as a manifest that
/// had gone stale rather than as a transform that had.
///
/// # Errors
///
/// [`denormalise_includes`]'.
fn normalise(name: &str, text: &str) -> Result<(String, usize), String> {
    let (stripped, removed) = strip(text);
    Ok((denormalise_includes(name, &stripped)?, removed))
}

/// Every vendored FlashInfer file, as `MODIFICATIONS` names it.
fn vendored() -> Vec<(String, PathBuf)> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, PathBuf)>) {
        for e in std::fs::read_dir(dir).expect("read vendor dir").flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, out);
            } else if p.file_name().is_some_and(|n| n != "LICENSE") {
                let rel = p.strip_prefix(root).expect("under root");
                out.push((rel.to_string_lossy().replace('\\', "/"), p.clone()));
            }
        }
    }
    let root = csrc().join("vendor/flashinfer");
    let mut v = Vec::new();
    walk(&root, &root, &mut v);
    v.sort();
    v
}

/// The manifest describes the tree it is in.
///
/// Checks all three columns against the files, and both directions of the set
/// — a vendored file with no row, and a row with no file, are each a way for
/// the manifest to stop being a description.
#[test]
fn modifications_describes_the_vendored_tree() {
    let rows = manifest();
    let files = vendored();
    assert!(!files.is_empty(), "no vendored FlashInfer files found");

    let mut wrong = Vec::new();
    for (name, path) in &files {
        let text = std::fs::read_to_string(path).expect("read vendored file");

        // Step 2 of the claim, before the columns are compared: a file whose
        // `#include <pie/…>` has no row cannot be turned back into upstream's,
        // and three columns that agree about a file we can no longer recover
        // are three columns describing a fork.
        if let Err(why) = normalise(name, &text) {
            wrong.push(why);
            continue;
        }

        let Some(row) = rows.get(name) else {
            wrong.push(format!("{name}: vendored but absent from MODIFICATIONS"));
            continue;
        };
        let got = measure(&text);
        if got != *row {
            wrong.push(format!(
                "{name}: MODIFICATIONS says guards={} added={} lines={} bytes={}, file has \
                 guards={} added={} lines={} bytes={}",
                row.guards, row.added, row.lines, row.bytes, got.guards, got.added,
                got.lines, got.bytes
            ));
        }
    }
    for name in rows.keys() {
        if !files.iter().any(|(n, _)| n == name) {
            wrong.push(format!("{name}: in MODIFICATIONS but not vendored"));
        }
    }

    assert!(
        wrong.is_empty(),
        "MODIFICATIONS no longer describes csrc/vendor/flashinfer:\n  {}",
        wrong.join("\n  ")
    );
}

/// The table's own totals row agrees with the rows above it.
///
/// `28 files   33   206   18187   798875` is a summary, and a summary is the
/// line a reader takes on trust. It is also the line that survives a per-file
/// edit unchanged, so it is the one most likely to go stale.
#[test]
fn the_totals_row_sums_the_table() {
    let text = std::fs::read_to_string(csrc().join("vendor/MODIFICATIONS")).expect("manifest");
    let totals = text
        .lines()
        .find(|l| l.split_whitespace().next().is_some_and(|w| w.parse::<usize>().is_ok()))
        .expect("MODIFICATIONS has a totals row");
    let n: Vec<usize> = totals
        .split_whitespace()
        .filter_map(|w| w.parse().ok())
        .collect();
    assert_eq!(n.len(), 5, "totals row is not five numbers: {totals:?}");

    let rows = manifest();
    let files = n[0];
    let (guards, added, lines, bytes) = rows.values().fold((0, 0, 0, 0), |(g, a, l, b), r| {
        (g + r.guards, a + r.added, l + r.lines, b + r.bytes)
    });
    assert_eq!(
        (files, n[1], n[2], n[3], n[4]),
        (rows.len(), guards, added, lines, bytes),
        "the totals row disagrees with the rows it summarises"
    );
}

/// The prose's *"14 files are untouched"* is a count of the rows below it.
///
/// A sentence in the same file as the table it describes, and nothing tied
/// the two together: vendor one more file with a guard and the sentence is
/// wrong in a document whose entire purpose is being right about this tree.
#[test]
fn the_untouched_count_in_the_prose_is_the_table_s() {
    let text = std::fs::read_to_string(csrc().join("vendor/MODIFICATIONS")).expect("manifest");
    let words: Vec<&str> = text.split_whitespace().collect();
    // `N files are untouched`, and not the totals row's `28 files`, which
    // appears first and would answer with the wrong number.
    let quoted: usize = words
        .windows(3)
        .find(|w| w[1] == "files" && w[2] == "are")
        .and_then(|w| w[0].parse().ok())
        .expect("MODIFICATIONS quotes `N files are untouched`");

    let untouched = manifest().values().filter(|r| r.guards == 0).count();
    assert_eq!(
        quoted, untouched,
        "MODIFICATIONS says {quoted} files are untouched, {untouched} rows have no guard"
    );
}

/// The marker total the manifest's own prose quotes.
///
/// Separate from the per-row check because it is the number a reader greps
/// for — `grep -rn "// PIE:" csrc/vendor/flashinfer/` — and a prose number
/// nothing checks is the thing this file exists about.
#[test]
fn the_quoted_marker_total_is_the_real_one() {
    let text = std::fs::read_to_string(csrc().join("vendor/MODIFICATIONS")).expect("manifest");
    let quoted: usize = text
        .split_whitespace()
        .zip(text.split_whitespace().skip(1))
        .find_map(|(a, b)| (b == "markers.").then(|| a.parse().ok())?)
        .expect("MODIFICATIONS quotes a marker total");

    let actual: usize = vendored()
        .iter()
        .map(|(_, p)| {
            std::fs::read_to_string(p)
                .expect("read")
                .lines()
                .filter(|l| l.trim_start().starts_with("// PIE:"))
                .count()
        })
        .sum();

    assert_eq!(
        quoted, actual,
        "MODIFICATIONS says {quoted} markers, the tree has {actual}"
    );
}

/// The stripper can fail.
///
/// Without this the two tests above pass on a stripper that returned its
/// input. §20.11's rule, one level down: a check that cannot fail has not
/// checked anything — which this session learned by publishing two
/// conclusions from a probe that matched `SUCCESS` inside its own label.
#[test]
fn the_stripper_is_not_a_no_op() {
    let guarded = "\
before
// PIE: guarded for NVRTC -- host-only include
// continuation of the marker's explanation
#ifndef __CUDACC_RTC__
#include <iostream>
#endif
after";
    let (out, removed) = strip(guarded);
    assert_eq!(out, "before\n#include <iostream>\nafter", "body must survive");
    assert_eq!(removed, 4, "marker + continuation + ifndef + endif");

    let (same, none) = strip("nothing\nto strip\n");
    assert_eq!(same, "nothing\nto strip\n".trim_end());
    assert_eq!(none, 0, "a file with no markers loses no lines");

    // a nested #if inside the guard must not end it early
    let nested = "\
// PIE: guarded
#ifndef __CUDACC_RTC__
#if FOO
int x;
#endif
#endif
tail";
    let (out, removed) = strip(nested);
    assert_eq!(out, "#if FOO\nint x;\n#endif\ntail");
    assert_eq!(removed, 3);
}

/// The second step refuses what it cannot recover, and only what it cannot.
///
/// Two directions, because a detector is wrong in two ways. A
/// `#include <pie/…>` with no row must stop the check — passing it through
/// would hand back text that is not upstream's while the manifest went on
/// saying it was. And a *mention* of the new layout in a comment must not:
/// these files acquire prose as the restructure proceeds, and a checker that
/// demanded a map row for a sentence would be a checker somebody switched
/// off.
#[test]
fn the_second_step_refuses_an_unmapped_include_and_nothing_else() {
    let unmapped = "\
#include <pie/device/vec.cuh>
int x;";
    let why = denormalise_includes("attention/decode.cuh", unmapped)
        .expect_err("an unmapped `<pie/…>` must refuse");
    assert!(why.contains("PIE_INCLUDES"), "the refusal must name the map: {why}");
    assert!(why.contains("attention/decode.cuh"), "and the file: {why}");

    // Indented, and with space after the `#` — both forms appear inside `#if`
    // blocks in this tree, and a detector that only saw column zero would let
    // the interesting half through.
    assert!(is_pie_include("    #  include <pie/attn/fa2_decode.cuh>"));
    assert!(is_pie_include("#include\t<pie/device/mma.cuh>"));

    // ...and the four ways a line can look like one and not be.
    assert!(!is_pie_include("// moved to <pie/device/vec.cuh> by the role cut"));
    assert!(!is_pie_include(" * see <pie/device/math.cuh>"));
    assert!(!is_pie_include("#include <flashinfer/attention/decode.cuh>"));
    assert!(!is_pie_include("#define PIE_INCLUDE <pie/device/vec.cuh>"));

    // The identity case, which is every file in the tree today.
    let clean = "#include \"../cp_async.cuh\"\nint x;";
    assert_eq!(
        denormalise_includes("attention/decode.cuh", clean).expect("no pie includes"),
        clean,
        "a file with nothing to rewrite must come back unchanged, byte for byte"
    );
}

/// Every row of the map is used, and nothing in the tree needs a row it has
/// not got.
///
/// Both directions, for [`modifications_describes_the_vendored_tree`]'s
/// reason. A dead row is a file that moved back, or a directive that was
/// edited after its row was written, and it rots silently because the passing
/// path never reads it. An unmapped directive is a fork.
///
/// With an empty map this asserts the tree holds no `<pie/…>` at all, which
/// is the true statement about a tree where nothing has moved yet — and it is
/// the assertion that flips the day the first file does.
#[test]
fn every_normalisation_row_is_used_and_no_pie_include_survives() {
    let files = vendored();
    let mut seen: Vec<(String, String)> = Vec::new();
    for (name, path) in &files {
        for line in std::fs::read_to_string(path).expect("read").lines() {
            if is_pie_include(line) {
                seen.push((name.clone(), line.to_string()));
            }
        }
    }

    let mut wrong = Vec::new();
    for (file, pie, upstream) in PIE_INCLUDES {
        if !seen.iter().any(|(f, l)| f.as_str() == *file && l.as_str() == *pie) {
            wrong.push(format!(
                "PIE_INCLUDES has a row for {file} -> {upstream:?} whose line {pie:?} is not \
                 in that file any more"
            ));
        }
        if !files.iter().any(|(n, _)| n.as_str() == *file) {
            wrong.push(format!("PIE_INCLUDES names {file}, which is not vendored"));
        }
    }
    for (file, line) in &seen {
        if !PIE_INCLUDES
            .iter()
            .any(|(f, p, _)| *f == file.as_str() && *p == line.as_str())
        {
            wrong.push(format!("{file} spells {line:?} and PIE_INCLUDES has no row for it"));
        }
    }
    assert!(wrong.is_empty(), "the normalisation map and the tree disagree:\n  {}", wrong.join("\n  "));

    assert_eq!(
        PIE_INCLUDES.len(),
        seen.len(),
        "one row per `<pie/…>` directive in the tree, and there are {} of them",
        seen.len()
    );
}

/// **The negative control, and the count of inputs on which it could fail.**
///
/// The three tests above walk the tree and compare it to a table written from
/// the same tree. That is the shape this session has published three
/// conclusions from and had to withdraw all three: a walk agreeing with a
/// constant derived from the walk, two rule implementations agreeing across
/// fifty-six files with no input on which they could have disagreed, and a
/// file-count check blind to every in-place edit. Agreement is not evidence
/// unless disagreement was possible, and *how possible* is a number.
///
/// So this mutates the real vendored files — not synthetic snippets — and
/// requires the check to reject every mutant. **154 discriminating inputs**,
/// in two families:
///
/// * **126 measurement mutants**, each of which must make [`measure`]
///   disagree with the file's row: three per file over 28 files (append a
///   line, drop the last line, pad a line without changing the line count)
///   and three per guarded file over 14 (rename a marker, delete a marker
///   line, insert a line inside a guard).
/// * **28 recoverability mutants**, one per file: inject a
///   `#include <pie/…>` with no map row, which [`normalise`] must refuse.
///
/// The 28 `pad_a_line` mutants are why the fourth column exists and are
/// asserted as such: each is invisible to `guards`, `added` and `lines`
/// together, and `bytes` alone catches it. Before the column, those 28 inputs
/// were 28 ways for this file to be wrong and say nothing.
///
/// # What it still cannot catch
///
/// A length-preserving edit outside every guard — swap two arguments, change
/// `1` to `2`. Four columns computed from the tree cannot see it, and no
/// number of mutants makes them. That residue needs upstream's bytes and so
/// needs the network; §23.8 records the one-time run.
#[test]
fn the_manifest_check_can_fail_on_every_drift_it_claims_to_catch() {
    let rows = manifest();
    let files = vendored();

    let (mut measured, mut unrecoverable) = (0usize, 0usize);
    let (mut with_markers, mut with_guards) = (0usize, 0usize);
    let mut survived: Vec<String> = Vec::new();

    for (name, path) in &files {
        let text = std::fs::read_to_string(path).expect("read vendored file");
        let row = rows.get(name).expect("every vendored file has a row");
        let lines: Vec<&str> = text.lines().collect();

        let mut mutants: Vec<(&str, String)> = Vec::new();

        mutants.push(("append a line", format!("{text}// drift\n")));
        mutants.push(("drop the last line", lines[..lines.len() - 1].join("\n")));

        // Line-count-neutral and byte-count-changing: the class the first
        // three columns are blind to. A space on the end of the first line —
        // built from `text` rather than from a re-`join`, because a join of
        // `lines()` silently drops the trailing newline and would cancel the
        // byte this mutant is trying to add, leaving a mutant that proved the
        // opposite of what it was written to prove. The first line of every
        // vendored file is upstream's comment banner, so padding it cannot
        // turn into a marker or a guard by accident.
        mutants.push(("pad a line", text.replacen('\n', " \n", 1)));

        if text.contains("// PIE:") {
            with_markers += 1;
            mutants.push(("rename a marker", text.replacen("// PIE:", "// PIE_DRIFT:", 1)));
            let at = lines
                .iter()
                .position(|l| l.trim_start().starts_with("// PIE:"))
                .expect("contains() said there is one");
            let mut cut = lines.clone();
            cut.remove(at);
            mutants.push(("delete a marker line", cut.join("\n")));
        }
        if let Some(at) = lines.iter().copied().position(is_rtc_guard) {
            with_guards += 1;
            let mut grown = lines.clone();
            grown.insert(at + 1, "#include <drift.h>");
            mutants.push(("grow a guard body", grown.join("\n")));
        }

        for (how, mutant) in &mutants {
            let got = measure(mutant);
            if got == *row {
                survived.push(format!("{name}: `{how}` left every column unmoved"));
            }
            if *how == "pad a line" {
                // The column's whole justification, asserted rather than
                // asserted-about: this mutant is invisible to the other
                // three, so if `bytes` did not move nothing would have.
                assert_eq!(
                    (got.guards, got.added, got.lines),
                    (row.guards, row.added, row.lines),
                    "{name}: `pad a line` was supposed to be invisible to the first three columns"
                );
                assert_ne!(got.bytes, row.bytes, "{name}: and visible to `bytes`");
            }
            measured += 1;
        }

        // Recoverability: a directive into our namespace that the map does
        // not know. Indented, and at the top, so the detector is exercised
        // somewhere other than column zero of the last line.
        let mut injected = lines.clone();
        injected.insert(0, "    #include <pie/device/vec.cuh>");
        if normalise(name, &injected.join("\n")).is_ok() {
            survived.push(format!("{name}: an unmapped `<pie/…>` was accepted"));
        }
        unrecoverable += 1;
    }

    assert!(
        survived.is_empty(),
        "the manifest check passed on {} mutated tree(s) — it is not checking what it says:\n  {}",
        survived.len(),
        survived.join("\n  ")
    );

    assert_eq!(
        (measured, unrecoverable),
        (3 * files.len() + 2 * with_markers + with_guards, files.len()),
        "the battery lost inputs: {} files, {with_markers} with markers, {with_guards} with guards",
        files.len()
    );
    assert_eq!(
        measured + unrecoverable,
        154,
        "this control ran on {} inputs, and the number it is documented as running on is 154. \
         If a FlashInfer bump changed the file count, update both this number and the doc \
         comment above — a control whose input count nobody states is a control nobody has \
         checked.",
        measured + unrecoverable
    );
}

/// The counted claims `csrc/shim/cooperative_groups.h` makes about the closure
/// it replaces.
///
/// That header justifies its own existence with a census — *"`cg::this_thread_block()`
/// at seven sites (four in `decode.cuh`, three in `prefill.cuh`),
/// `block.sync()` at forty-nine, and `cg::this_grid()` at two"* — and the
/// census is the argument: a shim that answers a hand of call sites is
/// defensible where one answering an open-ended API is not. Nothing checked
/// it, and a FlashInfer bump moves every one of these numbers.
///
/// The `.sync()` figure is deliberately spelled as decode + prefill here,
/// because that is what forty-nine counts and the tree holds fifty. The
/// fiftieth is in `attention/mla.cuh`, which is vendored and which nothing
/// includes (`new-horizon.md` §23.7) — so the prose is right about the
/// reachable closure and short by one about the directory. Asserting both
/// numbers is what keeps that distinction from being re-derived.
#[test]
fn the_cooperative_groups_census_still_holds() {
    let dir = csrc().join("vendor/flashinfer");
    let read = |rel: &str| std::fs::read_to_string(dir.join(rel)).expect(rel);
    let count = |hay: &str, needle: &str| hay.matches(needle).count();

    let decode = read("attention/decode.cuh");
    let prefill = read("attention/prefill.cuh");
    let mla = read("attention/mla.cuh");

    assert_eq!(count(&decode, "this_thread_block()"), 4, "decode.cuh");
    assert_eq!(count(&prefill, "this_thread_block()"), 3, "prefill.cuh");
    assert_eq!(
        count(&decode, ".sync()") + count(&prefill, ".sync()"),
        49,
        "the forty-nine the header names are decode + prefill"
    );
    assert_eq!(count(&mla, ".sync()"), 1, "and mla.cuh, unreachable, holds the fiftieth");

    let all: usize = vendored()
        .iter()
        .map(|(_, p)| count(&std::fs::read_to_string(p).expect("read"), "this_grid()"))
        .sum();
    assert_eq!(all, 2, "cg::this_grid() sites across the vendored closure");

    // The three doors. If a fourth file learns to include it, the header's
    // "exactly four doors" argument needs rewriting before the shim does.
    let files = vendored();
    let doors: Vec<&str> = files
        .iter()
        .filter(|(_, p)| {
            std::fs::read_to_string(p)
                .expect("read")
                .contains("include <cooperative_groups.h>")
        })
        .map(|(n, _)| n.as_str())
        .collect();
    assert_eq!(
        doors,
        ["attention/decode.cuh", "attention/mla.cuh", "attention/prefill.cuh"],
        "the files that open the cooperative_groups door"
    );
}

/// Transitive includers of a vendored file, by the two forms FlashInfer uses.
fn transitive_includers(target: &str) -> Vec<String> {
    let files = vendored();
    let text: BTreeMap<String, String> = files
        .iter()
        .map(|(n, p)| (n.clone(), std::fs::read_to_string(p).expect("read")))
        .collect();
    let direct = |of: &str| -> Vec<String> {
        let src = &text[of];
        let dir = std::path::Path::new(of).parent().unwrap_or(std::path::Path::new(""));
        let mut out = Vec::new();
        for line in src.lines() {
            let t = line.trim_start();
            if let Some(rest) = t.strip_prefix("#include \"") {
                if let Some(rel) = rest.split('"').next() {
                    let joined = dir.join(rel).to_string_lossy().into_owned();
                    let mut norm: Vec<&str> = Vec::new();
                    for c in joined.split('/') {
                        match c {
                            "." | "" => {}
                            ".." => { norm.pop(); }
                            other => norm.push(other),
                        }
                    }
                    let key = norm.join("/");
                    if text.contains_key(&key) { out.push(key); }
                }
            } else if let Some(rest) = t.strip_prefix("#include <flashinfer/") {
                if let Some(rel) = rest.split('>').next() {
                    if text.contains_key(rel) { out.push(rel.to_string()); }
                }
            }
        }
        out
    };
    let mut reach: Vec<String> = Vec::new();
    loop {
        let before = reach.len();
        for (name, _) in &files {
            if reach.iter().any(|r| r == name) { continue; }
            let d = direct(name);
            if d.iter().any(|x| x == target || reach.iter().any(|r| r == x)) {
                reach.push(name.clone());
            }
        }
        if reach.len() == before { break; }
    }
    reach.sort();
    reach
}

/// `csrc/shim/cuda_fp16.h` justifies an eight-line alias block with a cost.
///
/// Its claim is a causal chain rather than a count: *"One missing alias, one
/// file that uses it, seven files that include that file. An alias costs
/// nothing and its absence cost a quarter of the closure."* Each link is
/// checkable, and the middle one is what a FlashInfer bump moves.
#[test]
fn the_fp16_alias_cost_still_holds() {
    let users: Vec<String> = vendored()
        .iter()
        .filter(|(_, p)| std::fs::read_to_string(p).expect("read").contains("nv_half"))
        .map(|(n, _)| n.clone())
        .collect();
    assert_eq!(users, ["page.cuh"], "the one file that uses `nv_half`");

    let reachers = transitive_includers("page.cuh");
    assert_eq!(
        reachers.len(),
        7,
        "seven files reach page.cuh, and the alias's absence cost exactly them: {reachers:?}"
    );
}

/// `csrc/shim/cuda_fp8.h` refuses the e8m0 family on a zero-use census.
///
/// The refusal is the interesting half — the header declines to implement
/// `__nv_fp8_e8m0` because nothing reaches it, and names the sites in the
/// wider FlashInfer tree that do. If the closure ever grows one, the refusal
/// needs revisiting before a compile finds out.
#[test]
fn the_fp8_e8m0_refusal_is_still_unreached() {
    let names = ["__nv_fp8_e8m0", "__nv_cvt_float_to_e8m0", "__nv_fp8x2_e8m0"];
    let used: Vec<String> = vendored()
        .iter()
        .filter(|(_, p)| {
            let t = std::fs::read_to_string(p).expect("read");
            names.iter().any(|n| t.contains(n))
        })
        .map(|(n, _)| n.clone())
        .collect();
    assert!(
        used.is_empty(),
        "cuda_fp8.h declines the e8m0 family because the closure never reaches \
         it; these files now do: {used:?}"
    );
}

/// `pie_mma.cuh` is the one header whose correctness no test can check here,
/// so this checks that the thing which CAN check it still exists and still
/// says it can fail.
///
/// The shim is a per-lane register map: wrong, it compiles and returns
/// plausible numbers (`examples/mma_probe.rs` opens with why). Its parity
/// against `nvcuda::wmma` needs `nvcc` and a device, which `tests/` in this
/// crate deliberately does not have — so the guarantee lives in an example,
/// and an example is only a guarantee while something can invoke it and read
/// a verdict.
///
/// §23.11 found that probe exiting 0 on every path, pass or fail, for as long
/// as it had existed. §23.12 then found it was the ONLY example that was both
/// unfailable and the sole custodian of its claim. This keeps the second half
/// from silently coming back: if the exit path goes, the shim's only check
/// becomes a report again.
#[test]
fn the_mma_shim_still_has_something_that_can_fail() {
    let probe = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/mma_probe.rs");
    let src = std::fs::read_to_string(&probe).expect("examples/mma_probe.rs");

    assert!(
        src.contains("std::process::exit(1)"),
        "mma_probe must exit non-zero on failure, or nothing can be gated on \
         the only check `pie_mma.cuh` has"
    );
    assert!(
        src.contains("pub fn run() -> bool"),
        "mma_probe's run() must report a verdict rather than print one"
    );
    assert!(
        src.contains("fn sensitivity(") && src.contains("-> bool"),
        "the transposed-store control must report whether it was CAUGHT; an \
         unmeasured control is not a pass"
    );

    // And the shim it checks is still the one the units compile against.
    let shim = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/src/pie_mma.cuh");
    assert!(shim.is_file(), "csrc/src/pie_mma.cuh");
    assert!(
        src.contains("pie_mma.cuh"),
        "mma_probe must name the header it is the check for"
    );
}

/// A tile kernel needs CUDA **13.3 or newer runtime headers**, and nothing in
/// the toolchain says so when they are older.
///
/// CUDA 13.3's tile frontend injects `-D__NV_TL_BUILTIN__=__tile_builtin__`,
/// and 13.3's `cuda_bf16.h` carries that marker on the struct:
///
/// ```text
///   13.0   struct                   __CUDA_ALIGN__(2) __nv_bfloat16 {...}
///   13.3   struct __NV_TL_BUILTIN__ __CUDA_ALIGN__(2) __nv_bfloat16 {...}
/// ```
///
/// Without it `__nv_bfloat16` is an ordinary two-byte aggregate, every tile
/// of it lowers as `tile<2xi8>`, and tile codegen dies a thousand lines deep
/// with `"Unexpected element type in tile!"` naming a type the user never
/// wrote. Adding the attribute by hand to a 13.0 header is the entire fix.
///
/// This is easy to hit because the toolchain arrives as independently
/// versioned pip wheels — `nvidia-cuda-nvcc` and `nvidia-cuda-nvrtc` can be
/// 13.3 while `nvidia-cuda-runtime`, which owns these headers, is 13.0 — and
/// no version check fires. It cost this tree a day and a retracted bug
/// report; `.wiki/driver/cutile-16bit-header-trap.cu` is the account.
///
/// So the detector is written down where a tile build can reach it. There is
/// no tile build in this crate yet, which is exactly why: the knowledge has
/// to outlive the session that bought it.
///
/// `cuda_tf32.h` is the cheap version of the same check — it ships only in
/// 13.3+, so its absence dates the headers in one `stat`.
#[test]
fn the_cutile_header_floor_is_written_down_where_a_build_can_find_it() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let kernel = root.join("csrc/src/moe/moe_grouped_gemm_tile.cuh");
    let src = std::fs::read_to_string(&kernel).expect("moe_grouped_gemm_tile.cuh");

    for needle in ["__NV_TL_BUILTIN__", "__tile_builtin__", "cuda_tf32.h", "CUDA_ROOT"] {
        assert!(
            src.contains(needle),
            "the tile kernel no longer records `{needle}`, so the 13.3 runtime \
             requirement is undocumented at the one place a tile build would \
             look. The four are load-bearing: the first three date the runtime \
             headers, and CUDA_ROOT is what tileiras silently needs -- without \
             it every input, including nvcc's own .tilebc, fails with a bare \
             `failed to compile Tile IR program`. See \
             .wiki/driver/cutile-16bit-header-trap.cu and new-horizon 23.18"
        );
    }

    assert!(
        src.contains("__nv_bfloat16"),
        "the tile kernel no longer names NVIDIA's bf16. If it went back to \
         carrying bf16 as `unsigned short`, note what that cost when it was \
         last done: 224 registers against 92 at 16x64x32, and 255 with spills \
         against 160 at kTileM=64, plus every performance conclusion drawn \
         while it was in place"
    );
}

/// The tile kernel must not include this tree's `cuda_bf16.h` adapter.
///
/// `cuda::tiles` constrains tile elements to the scalar types it knows, so
/// this tree's `device::bf16` is refused outright — `template constraint not
/// satisfied` — even carrying `__tile_builtin__`. A tile kernel must name
/// NVIDIA's `__nv_bfloat16`.
///
/// But `csrc/shim/cuda_bf16.h` aliases that same name to `device::bf16`, so
/// FlashInfer stays byte-identical to upstream. The two cannot share a
/// translation unit: `cuda_tile.h` forward-declares `struct __nv_bfloat16;`
/// and a struct declaration cannot share a name with a type alias. Whichever
/// include directory comes first decides, and when it is the tree's the
/// build dies on a redefinition rather than on anything informative.
///
/// The kernel includes nothing that wants the adapter, which is what keeps
/// this tractable. This test is here so it stays that way.
#[test]
fn the_tile_kernel_stays_out_of_the_adapter_headers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(root.join("csrc/src/moe/moe_grouped_gemm_tile.cuh"))
        .expect("moe_grouped_gemm_tile.cuh");

    let includes: Vec<&str> = src
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("#include"))
        .collect();

    for bad in ["pie_mma.cuh", "\"cuda_bf16.h\"", "\"cuda_fp16.h\""] {
        assert!(
            !includes.iter().any(|l| l.contains(bad)),
            "the tile kernel now includes {bad}, which redeclares \
             __nv_bfloat16 or __half against cuda_tile.h's own forward \
             declarations. Includes are: {includes:?}"
        );
    }

    assert!(
        includes.iter().any(|l| l.contains("<cuda_bf16.h>")),
        "the tile kernel no longer includes NVIDIA's <cuda_bf16.h>. It needs \
         the real type, and it needs NVIDIA's include directory to precede \
         csrc/shim so the angle-bracket include does not find the adapter. \
         Includes are: {includes:?}"
    );
}

/// The fused MoE tile kernel is a NEGATIVE result and must keep saying so.
///
/// `moe_fused_tile.cuh` writes fc1 + swiglu + fc2 as one `__tile_global__`
/// with the intermediate never stored — the thing that would close the
/// CUTLASS island's remaining advantage at the decode census. It is correct
/// and it is slower than not fusing: 1.778 ms against 0.933 for two unfused
/// tile GEMMs and 0.581 for the island.
///
/// The cause is shared memory. The tile compiler stages `partition_view`
/// loads through it, and the fused working set takes 92-99 KB of a 100 KB
/// budget — one block per SM, where the unfused grouped GEMM takes 16 KB.
///
/// A file like this is dangerous precisely because it looks like a kernel
/// someone should finish. The banner is the only thing stopping that, so the
/// banner is a gate.
#[test]
fn the_fused_tile_kernel_still_declares_itself_a_negative_result() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("csrc/src/moe/moe_fused_tile.cuh");
    let src = std::fs::read_to_string(&path).expect("csrc/src/moe/moe_fused_tile.cuh");

    assert!(
        src.contains("NEGATIVE result"),
        "moe_fused_tile.cuh no longer announces that it is a negative result. \
         It is slower than not fusing (1.778 ms against 0.933) and the next \
         reader needs to know that before they invest in it"
    );

    for needle in ["SHARED", "occupancy", "0.581"] {
        assert!(
            src.contains(needle),
            "moe_fused_tile.cuh no longer records `{needle}`, so the reason it \
             loses — shared-memory staging collapsing occupancy, against a \
             named island figure — is no longer on the file. See \
             .wiki/driver/new-horizon.md 23.17"
        );
    }
}

/// A `Toolchain` floor is necessary and NOT sufficient for the tile unit, and
/// `unit.rs` must keep saying so.
///
/// Every other unit fails safe: NVRTC rejects source it cannot compile,
/// loudly, in `tests/units.rs`. `moe/moe_grouped_gemm_tile` does not. Measured
/// with NVRTC 13.3.33 and a bf16 tile `mma`:
///
/// ```text
///   nvrtcCompileProgram   rc = 0
///   nvrtcGetCUBIN         .note.nv.tkinfo and NO .text
///   cuModuleLoadData      SUCCESS
///   cuModuleGetFunction   CUDA_ERROR_NOT_FOUND
/// ```
///
/// A tile kernel compiles to Tile IR, not SASS, and something downstream must
/// assemble it — a driver new enough to do it at load, or `tileiras` over
/// `nvrtcGetTileIR`'s output before the cubin is cached. So a floored tile
/// unit under a 13.3 NVRTC would compile clean, cache, load, and fail at the
/// FIRST LAUNCH.
///
/// That is the one shape this crate's gates cannot see, which is exactly why
/// it has to be written down where the person adding the demand will read it.
#[test]
fn the_tile_units_floor_is_still_marked_insufficient_on_its_own() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(root.join("src/unit.rs")).expect("src/unit.rs");

    for needle in ["tileiras", "CUDA_ROOT", "NOT_FOUND", "nvrtcGetTileIR"] {
        assert!(
            src.contains(needle),
            "src/unit.rs no longer records `{needle}`. A Toolchain floor alone \
             makes a tile unit compile clean and fail at the first launch, and \
             the DEMANDS table is where whoever adds that floor will look. See \
             .wiki/driver/new-horizon.md 23.18"
        );
    }
}

/// RMSNorm was rewritten in CuTile twice. The second one is FASTER, and both
/// files must keep saying so.
///
/// The first attempt measured 3.84 us against the tree's 2.93 and was written
/// off as "a code-size argument, not a speed one". It was written in a naive
/// dialect. In NVIDIA's own idiom -- `ct::iota` plus a `ct::load` gather
/// rather than a `partition_view` over a 1-D row, `latency=1` on each load,
/// the hidden size a template parameter, `assume_aligned<16>` -- it is
/// 1.51x faster at H=4096 and 1.59x at H=7168, exact at both, and ties the
/// hand-vectorised `rmsnorm_vec8` without needing its alignment check.
///
/// Both halves are gated. A negative result about an uncommitted thing leaves
/// no artifact; a REVERSED negative result leaves a worse one, because the
/// old conclusion is already quoted elsewhere.
#[test]
fn the_rmsnorm_cutile_result_is_still_recorded_and_still_says_faster() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let hand = std::fs::read_to_string(root.join("csrc/src/norm/rmsnorm.cuh"))
        .expect("csrc/src/norm/rmsnorm.cuh");
    let tile = std::fs::read_to_string(root.join("csrc/src/norm/rmsnorm_tile.cuh"))
        .expect("csrc/src/norm/rmsnorm_tile.cuh");

    assert!(
        hand.contains("FASTER"),
        "rmsnorm.cuh no longer records that the CuTile twin beats it. That was \
         a reversal of this file's own earlier claim, and a reversal that goes \
         missing leaves the withdrawn version standing"
    );
    for needle in ["iota", "latency=1", "assume_aligned"] {
        assert!(
            tile.contains(needle),
            "rmsnorm_tile.cuh no longer records `{needle}`. Those three are the \
             difference between the CuTile RMSNorm that lost and the one that \
             won -- see .wiki/driver/new-horizon.md 23.20"
        );
    }
    assert!(
        tile.contains("0.1103") || tile.contains("masked"),
        "rmsnorm_tile.cuh no longer records that the tail must be masked. \
         Unmasked it looks healthy at H=4096 and is wrong at H=7168, which is \
         the shape a careless bench does not pick"
    );
}

/// The elementwise result is a ROOFLINE result and must keep saying so.
///
/// `mlp/swiglu_tile.cuh` is bit-exact against `swiglu.cuh` and 1.53x faster
/// at 25 MB — and 4% SLOWER at 805 MB, where both sit at 77-80% of the
/// L40S's ~864 GB/s HBM peak. Quoting only the first number would make an
/// elementwise CuTile rewrite look like a free 1.5x, which it is not: no
/// programming model makes a kernel at the memory roofline faster.
///
/// This is the shape of claim that decays worst, because the favourable half
/// is the one that gets repeated.
#[test]
fn the_swiglu_tile_result_still_carries_both_halves() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(root.join("csrc/src/mlp/swiglu_tile.cuh"))
        .expect("csrc/src/mlp/swiglu_tile.cuh");

    assert!(
        src.contains("roofline"),
        "swiglu_tile.cuh no longer names the roofline, which is the half of \
         its result that bounds the claim: it is 1.53x faster cached and 4% \
         slower at 805 MB"
    );
    for needle in ["0.008", "1.218", "load_masked", "store_masked"] {
        assert!(
            src.contains(needle),
            "swiglu_tile.cuh no longer records `{needle}`. The two timings are \
             the two ends of the roofline finding; the masks are what make the \
             kernel general rather than correct-at-one-shape. See \
             .wiki/driver/new-horizon.md 23.21"
        );
    }
}

/// The router top-K result has three traps on it and all three must stay.
///
/// `moe/topk_softmax_tile.cuh` beats `topk_softmax_warp_x1` — a hand-tuned
/// warp-resident reduction, not a first draft — by 1.28x at decode, with
/// identical expert indices. The traps are what make that result reproducible
/// rather than lucky:
///
/// * the weights renormalise by the WINNERS' own sum, not all experts. Get it
///   wrong and the indices still match while the weights differ by 0.108;
/// * a local `int[TOPK]` array costs 6.7x — 20.38 us against 3.05 — because a
///   `__tile_global__` has no per-thread scratch;
/// * `ct::exp` on a scalar is FREE, which is the hypothesis that was almost
///   published as the cause of the 6.7x and was wrong.
///
/// A trap that produced a plausible wrong answer is worth more written down
/// than the result it guards.
#[test]
fn the_topk_tile_traps_are_still_on_the_file() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(root.join("csrc/src/moe/topk_softmax_tile.cuh"))
        .expect("csrc/src/moe/topk_softmax_tile.cuh");

    for (needle, why) in [
        ("0.108", "the weight-definition trap: dividing by the sum over ALL \
                   experts leaves the indices identical and the weights wrong"),
        ("20.38", "the local-array cost, which is 6.7x and survives \
                   #pragma unroll and a compile-time TOPK"),
        ("3.06", "the measured A/B showing scalar ct::exp is free -- the \
                  explanation that was almost published in place of the real one"),
        ("IDENTICAL", "the correctness bar: expert indices must match, because \
                       a different expert is a different model"),
    ] {
        assert!(
            src.contains(needle),
            "topk_softmax_tile.cuh no longer records `{needle}` -- {why}. See \
             .wiki/driver/new-horizon.md 23.22"
        );
    }
}

/// The tile kernels are ADDITIONS. Every one must carry a preference
/// predicate, and no incumbent may be described as replaced.
///
/// This was asked for explicitly and it is also the only defensible shape:
/// the alternatives need NVRTC 13.3, 13.3 runtime headers and `tileiras`,
/// and this crate loads NVRTC 13.0.88. An alternative that cannot be
/// selected on the machine in front of you is not an alternative, it is a
/// removal.
///
/// `csrc/src/tile_alternatives.cuh` pins each predicate to the rows of the
/// sweep that produced it with `static_assert`s, so a bound that gets
/// rounded fails a compile rather than quietly firing the slower kernel.
/// That is not hypothetical: `swiglu_tile_preferred` was first written as
/// `6 * n <= 100 MB`, which excluded the very point it was measured at.
#[test]
fn every_tile_kernel_is_an_alternative_with_a_predicate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/src");

    for (file, pred) in [
        ("moe/moe_grouped_gemm_tile.cuh", "moe_grouped_gemm_tile_preferred"),
        ("norm/rmsnorm_tile.cuh", "rmsnorm_tile_preferred"),
        ("mlp/swiglu_tile.cuh", "swiglu_tile_preferred"),
        ("moe/topk_softmax_tile.cuh", "topk_softmax_tile_preferred"),
        ("norm/rmsnorm_rasr_tile.cuh", "rmsnorm_rasr_tile_preferred"),
        ("moe/moe_fused_tile.cuh", "moe_fused_tile_preferred"),
        ("layout/gather_rows_tile.cuh", "gather_rows_tile_preferred"),
        ("quant/dequant_wna16_tile.cuh", "dequant_wna16_tile_preferred"),
        ("quant/wna16_gemv_tile.cuh", "wna16_gemv_tile_preferred"),
        ("rope/rope_tile.cuh", "rope_partial_tile_preferred"),
    ] {
        let src = std::fs::read_to_string(root.join(file)).expect(file);
        assert!(
            src.contains(&format!("constexpr bool {pred}")),
            "{file} no longer defines `{pred}`. A tile kernel without a \
             preference predicate is a replacement by default, and these are \
             additions"
        );
        assert!(
            src.contains("ALTERNATIVE"),
            "{file} no longer says it is an alternative. The incumbent is the \
             fallback for every toolchain that cannot compile a tile kernel, \
             which today is every toolchain this crate loads"
        );
    }

    let alts = std::fs::read_to_string(root.join("tile_alternatives.cuh"))
        .expect("csrc/src/tile_alternatives.cuh");
    let asserts = alts.matches("static_assert").count();
    assert!(
        asserts >= 14,
        "tile_alternatives.cuh has {asserts} static_asserts; it had 14, one per \
         measured endpoint. A predicate that is no longer pinned to its sweep \
         is a comment with a type"
    );
    for measured in ["1.94 vs 2.93", "0.038 vs 0.057", "3.06 vs 3.90", "7.22 vs 6.08",
                     "2.41 vs 4.33"] {
        assert!(
            alts.contains(measured),
            "tile_alternatives.cuh no longer cites `{measured}`. The point of \
             that file is that each bound names the measurement it came from"
        );
    }
}

/// No tile kernel may spell a `<cstdint>` type. NVRTC does not have them.
///
/// `nvcc` force-includes `cuda_runtime.h` and so has `<cstdint>`
/// transitively; NVRTC does not, and every tile kernel here is destined for
/// NVRTC. `moe_fused_tile.cuh` said `ct::extents<uint32_t, ...>` — copied
/// from NVIDIA's own `matmul.cuh` — compiled clean under `nvcc`, and failed
/// through the JIT path with `identifier "uint32_t" is undefined`.
///
/// An AOT build cannot see this class of defect at all, which is why it is a
/// gate on the text rather than a note in a header. The builtin spellings
/// (`unsigned`, `int`, `long long`) always work.
#[test]
fn no_tile_kernel_spells_a_cstdint_type() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/src");

    let tile_kernels = [
        "moe/moe_grouped_gemm_tile.cuh",
        "moe/moe_fused_tile.cuh",
        "layout/gather_rows_tile.cuh",
        "quant/dequant_wna16_tile.cuh",
        "quant/wna16_gemv_tile.cuh",
        "rope/rope_tile.cuh",
        "moe/topk_softmax_tile.cuh",
        "norm/rmsnorm_tile.cuh",
        "norm/rmsnorm_rasr_tile.cuh",
        "mlp/swiglu_tile.cuh",
    ];

    for file in tile_kernels {
        let src = std::fs::read_to_string(root.join(file)).expect(file);
        for (line_no, line) in src.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") || code.starts_with("///") {
                continue;
            }
            for ty in ["uint32_t", "uint64_t", "int32_t", "int64_t", "uint16_t", "size_t"] {
                assert!(
                    !line.contains(ty),
                    "{file}:{} spells `{ty}`, which NVRTC does not have -- nvcc \
                     only sees it because it force-includes cuda_runtime.h. Use \
                     a builtin. This exact defect was found by JIT-compiling \
                     these kernels rather than by any AOT build.",
                    line_no + 1
                );
            }
        }
    }
}

/// The census of all 455 kernels must stay on the file, and must keep its
/// wash and its exclusions.
///
/// `tile_alternatives.cuh` classifies every `__global__` in `csrc/src` into
/// seven buckets, six of which have a measured representative in this crate.
/// It is easy for a table like that to drift into listing only the
/// favourable rows — so the gate checks the two that are NOT favourable:
/// the COPY bucket is a measured wash, and a third of the tree cannot or
/// should not move at all.
#[test]
fn the_kernel_census_keeps_its_unfavourable_rows() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/src");
    let src = std::fs::read_to_string(root.join("tile_alternatives.cuh"))
        .expect("csrc/src/tile_alternatives.cuh");

    for (needle, why) in [
        ("263", "the win column after the inferred bucket was measured and moved"),
        ("455", "the census total. Without it the buckets are percentages of nothing"),
        ("WASH", "the COPY bucket's verdict -- 35 kernels that a survey would \
                  otherwise put in the win column"),
        ("cannot, or should not", "the third of the tree that is excluded"),
        ("No inferred rows", "the census closed its last hole by measuring the \
                             GEMV bucket, and that claim is the reason to \
                             trust the percentages"),
    ] {
        assert!(
            src.contains(needle),
            "tile_alternatives.cuh no longer records `{needle}` -- {why}"
        );
    }

    let wash = std::fs::read_to_string(root.join("layout/gather_rows_tile.cuh"))
        .expect("csrc/src/layout/gather_rows_tile.cuh");
    assert!(
        wash.contains("not faster"),
        "gather_rows_tile.cuh no longer says it is not faster. It exists to \
         make the COPY bucket's verdict a measurement rather than an \
         assumption, and a kernel that does not announce that will be read as \
         one that won"
    );
}

/// The roofline band is an OBSERVATION and the predicates are MEASUREMENTS,
/// and the file has to keep them apart.
///
/// Four bounds in this tree were derived independently and turned out to be
/// the same machine fact: the tile advantage vanishes at roughly 3x L2 of
/// touched bytes — last gap at 138 MB, first convergence at 134 MB, across
/// an elementwise activation, a three-pass reduction, a 4x expansion and a
/// weight-streaming GEMV.
///
/// The temptation is to rewrite the predicates in terms of it. They must not
/// be: each is bounded at its own largest MEASURED point, which is tighter
/// than the band, and widening a bound to a derived constant trades a
/// measurement for a model. This whole file exists because a bound that is a
/// model gets quoted as a measurement.
#[test]
fn the_roofline_band_stays_an_observation_not_a_bound() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/src");
    let alts = std::fs::read_to_string(root.join("tile_alternatives.cuh"))
        .expect("csrc/src/tile_alternatives.cuh");

    for (needle, why) in [
        ("138 MB", "the last measured point with a gap open"),
        ("134", "the first measured point converged"),
        ("do NOT use it", "the reason the predicates keep their own tighter \
                           bounds instead of the derived band"),
        ("do NOT unify", "the saturation crossovers are 1.5 and 7.2 blocks per \
                          SM and are two facts, not one line"),
    ] {
        assert!(
            alts.contains(needle),
            "tile_alternatives.cuh no longer records `{needle}` -- {why}. See \
             .wiki/driver/new-horizon.md 23.32"
        );
    }

    // The bounds themselves must stay at their measured points, not drift out
    // to the band.
    let swig = std::fs::read_to_string(root.join("mlp/swiglu_tile.cuh")).expect("swiglu_tile");
    assert!(
        swig.contains("(16LL << 20)"),
        "swiglu_tile_preferred no longer stops at 16 Mi elements -- 100.7 MB, \
         its largest measured point. If it now stops at the roofline band it \
         is a model wearing a measurement's clothes"
    );
}

/// The rope alternative is faster AND more accurate, and both halves must
/// stay on the file — the second is why it is declined.
///
/// It wins 1.2-1.4x on time and is three orders of magnitude closer to an
/// fp64 reference at large rope angles, because `rope.cuh` uses `__sincosf`
/// and this uses `ct::sin`/`ct::cos`. That is a behaviour change: rope error
/// feeds attention scores, so the incumbent's trade may be deliberate.
///
/// A file recording only the speed would read as an obvious merge.
#[test]
fn the_rope_alternative_declines_on_accuracy_not_speed() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/src");
    let src = std::fs::read_to_string(root.join("rope/rope_tile.cuh")).expect("rope_tile.cuh");

    for (needle, why) in [
        ("__sincosf", "the incumbent's fast intrinsic, which is the cause"),
        ("4.62e-04", "the incumbent's measured error against fp64"),
        ("1.52e-07", "this kernel's, at the same point"),
        ("behaviour change", "why a faster and more accurate kernel is still \
                              declined rather than merged"),
    ] {
        assert!(
            src.contains(needle),
            "rope_tile.cuh no longer records `{needle}` -- {why}. A file that \
             kept only the 1.4x would read as an obvious merge"
        );
    }
}

/// The rope `__sincosf` finding is about pie's own kernel, and it is a
/// measured trade rather than a filed bug. Both halves have to stay: the
/// error table (which says WHEN it matters) and the cost table (which says
/// what the fix costs). A finding with only the first half is an alarm; with
/// only the second it is an excuse.
#[test]
fn the_sincosf_trade_keeps_both_of_its_tables() {
    let src = include_str!("../csrc/src/rope/rope.cuh");

    // The error side, with the bf16 floor that turns it into a threshold.
    for probe in ["1.91e-03", "1.05e-02", "3.9e-03", "64K"] {
        assert!(
            src.contains(probe),
            "rope.cuh dropped `{probe}` from the __sincosf error table -- \
             without it the reader cannot tell at what context length the \
             fast intrinsic stops being free"
        );
    }

    // The cost side, both forms, because the hoisting is the whole reason
    // they differ and quoting one number for both would be wrong.
    for probe in ["+5.2%", "+12.5%", "+20.4%"] {
        assert!(
            src.contains(probe),
            "rope.cuh dropped `{probe}` from the cost table -- the accurate \
             form is not free and the price differs 2x between the hoisted \
             and unhoisted kernels"
        );
    }

    assert!(
        src.contains("trade, not a bug"),
        "rope.cuh should say plainly that this is a trade being stated for \
         its owner, not a defect being reported"
    );
}

/// The ARGMAX census row was backwards for as long as it was priced off a
/// plausible representative instead of a measured one. Both halves of the
/// correction have to stay: the census must say LOSES, and the kernel must
/// keep the piece-by-piece table that shows the loss is structural rather
/// than a coding error someone could "fix".
#[test]
fn the_argmax_bucket_stays_corrected() {
    let census = include_str!("../csrc/src/tile_alternatives.cuh");
    let kernel = include_str!("../csrc/src/sample/argmax_tile.cuh");

    assert!(
        census.contains("ARGMAX        26   argmax_tile            LOSES 4x"),
        "the ARGMAX census row must name argmax_tile and say it loses -- it \
         previously quoted topk_softmax_tile at 1.28-1.40x, which is a \
         128-wide reduction standing in for a 151,936-wide one"
    );
    assert!(
        !census.contains("ARGMAX        26   topk_softmax_tile"),
        "the withdrawn ARGMAX representative must not come back"
    );

    // The withdrawn mechanism must stay withdrawn, and visibly so.
    assert!(
        kernel.contains("WITHDRAWN"),
        "argmax_tile.cuh first blamed the reduction WIDTH, which a \
         fixed-grid sweep disproved (256 tiles costs nothing). The wrong \
         mechanism has to stay on the page as withdrawn -- deleting it \
         invites the next reader to derive it again"
    );

    // The floor is the whole argument: no index, no mask, still behind.
    assert!(
        kernel.contains("1.9x behind"),
        "argmax_tile.cuh must keep the floor measurement -- reduce_max alone \
         is already 1.9x behind, which is what makes the loss structural"
    );
    for probe in ["26.52", "48.16", "61.43", "0.24x"] {
        assert!(
            kernel.contains(probe),
            "argmax_tile.cuh dropped `{probe}` from the piece-by-piece \
             table; without all four rows the reader cannot tell which part \
             is hopeless"
        );
    }

    assert!(
        kernel.contains("return false;"),
        "argmax_tile_preferred must stay declined"
    );
}

/// A tile hint attached to a variable declaration is silently ignored --
/// nvcc says so with warning #20364-D. Every load in this tree's tile
/// kernels therefore declares the tile on one line and assigns on the next.
/// That shape looks like clumsy style and is not; this pins it.
#[test]
fn tile_loads_do_not_hint_a_declaration() {
    for (name, src) in [
        ("norm/rmsnorm_tile.cuh", include_str!("../csrc/src/norm/rmsnorm_tile.cuh")),
        ("sample/argmax_tile.cuh", include_str!("../csrc/src/sample/argmax_tile.cuh")),
    ] {
        let mut lines = src.lines().peekable();
        while let Some(line) = lines.next() {
            if !line.contains("hint(1000") {
                continue;
            }
            let Some(next) = lines.peek() else { continue };
            let t = next.trim_start();
            assert!(
                !(t.starts_with("auto ") || t.starts_with("const auto ")),
                "{name}: a cutile hint sits on a declaration (`{t}`) -- nvcc \
                 warning #20364-D says the hint is IGNORED there. Declare the \
                 tile first, assign on the next line."
            );
        }
    }
}

/// The reduction surface took three sweeps and two pushed, wrong one-line
/// laws. The gate protects the thing that cost the most to learn: that the
/// surface is two-dimensional and neither variable alone predicts a cell.
/// A later edit that "simplifies" this back to one sentence is repeating
/// the mistake, so both withdrawn laws stay on the page.
#[test]
fn the_reduction_surface_stays_two_dimensional() {
    let census = include_str!("../csrc/src/tile_alternatives.cuh");

    assert!(
        census.contains("WITHDRAWN #1") && census.contains("WITHDRAWN #2"),
        "both withdrawn laws must stay visible -- `width is what costs` and \
         `width is free, blocks are everything` were each fitted to one \
         slice, each read well, and each was wrong. Deleting them invites a \
         third"
    );
    assert!(
        census.contains("There is no
// one-line law here."),
        "the census must say plainly that there is no one-line law -- that \
         sentence is the whole finding"
    );

    // The unconfounded column is the evidence that width is not free. It is
    // the first thing a later edit would drop as redundant.
    assert!(
        census.contains("1.07    1.07    1.08    1.09    1.12    1.24    1.29"),
        "the 1-tile row of the surface must stay"
    );
    assert!(
        census.contains("0.66    0.67    0.75*   1.01*   1.12*   1.15*   1.18*"),
        "the 256-tile row must stay, with its L2 markers -- it is the row \
         that shows both variables acting at once"
    );
    assert!(
        census.contains("all L2-resident, so nothing but the width changes"),
        "the census must keep the note that the 8-block column is the clean \
         one; without it the width claim has the same defect as the two \
         withdrawn laws"
    );

    // And the non-monotone row, which is why no single rule fits.
    assert!(
        census.contains("is not monotone in either"),
        "the 16-tile row is not monotone in width OR grid, and saying so is \
         what stops the next reader from fitting a third law"
    );
}

/// The REDUCE row is the census's second-biggest bucket and it was priced
/// off one representative, exactly like ARGMAX was. The surface made that
/// checkable, so it was checked by counting every strided reduction in the
/// tree rather than sampling one. This gate holds the count AND the
/// residual, because an enumeration that quietly drops its unknowns is
/// worse than the inference it replaced.
#[test]
fn the_reduce_bucket_is_enumerated_not_sampled() {
    let census = include_str!("../csrc/src/tile_alternatives.cuh");

    assert!(
        census.contains("The counting rule is a test, not this comment"),
        "the census must point at the executable counting rule. The first \
         version of this audit used a shell pipeline that gave 43, 45 or 46 \
         for the same bucket depending on the regex, and a count that moves \
         with the pattern is not an enumeration"
    );
    assert!(
        census.contains("reduction_axis_counts"),
        "the census must name the function that defines the count, so the \
         next reader can re-run it instead of trusting the table"
    );

    // The honest denominator. Folding the unclassified sites into the safe
    // bucket would make the audit look complete and make it a sample again.
    assert!(
        census.contains("axis not in the list above"),
        "the census must keep the unclassified count visible -- two thirds \
         classified is the real coverage and hiding the third is how a \
         representative becomes a claim"
    );
    assert!(
        census.contains("`inter`, the FFN intermediate width"),
        "the census must name `inter` as the one unclassified axis that can \
         cross the 16-tile boundary; that is the whole value of having \
         looked at the unclassified set"
    );

    // The residual is the honest half and the first thing to go stale.
    assert!(
        census.contains("cannot be settled"),
        "the 11 data-dependent sites must stay marked as unresolved -- \
         ptir's `len` is structurally argmax and its width is a property of \
         the PTIR program, not of this tree"
    );
    assert!(
        census.contains("len <= 16 * 1024") && census.contains("len >= 64 * 1024"),
        "the census must hand the next person the boundary a ptir predicate \
         would need, in both directions, or they will do what the ARGMAX \
         row did"
    );
}

/// The reduction count in the census is derived from source, and the first
/// attempt at it gave 43, 45 or 46 for the same bucket depending on which
/// regex asked. A count that moves with the pattern is not an enumeration,
/// so the counting RULE lives here, in code that runs, and the census quotes
/// this test rather than a shell pipeline nobody can re-run.
///
/// The rule: a line that, after trimming, starts with `for (`, mentions
/// `threadIdx.x` or `= tid;`, and compares against the named axis.
fn reduction_axis_counts() -> std::collections::BTreeMap<String, usize> {
    use std::collections::BTreeMap;
    use std::path::Path;

    // Axis -> width bucket. Sizes are the ones this tree actually names.
    const SMALL: &[&str] = &[
        "head_dim", "K_d", "V_d", "kv_lora", "half", "D", "H", "cols", "dim",
        "num_experts", "num_routes", "nkeys", "heads_here", "nvec", "vecs",
        "width", "C", "E", "N", "d", "t", "ratio", "window",
    ];
    const MEDIUM: &[&str] = &["hidden", "hidden_size"];
    const DYNAMIC: &[&str] = &["len", "total", "num_tokens"];
    const LONG: &[&str] = &["vocab"];

    let mut out: BTreeMap<String, usize> = BTreeMap::new();
    let mut stack = vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/src")];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if p.extension().is_none_or(|x| x != "cuh") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else { continue };
            for line in text.lines() {
                let l = line.trim();
                if !l.starts_with("for (") {
                    continue;
                }
                if !(l.contains("threadIdx.x") || l.contains("= tid;")) {
                    continue;
                }
                *out.entry("all_strided".to_string()).or_default() += 1;
                let mut hit = false;
                for (bucket, axes) in [
                    ("small", SMALL), ("medium", MEDIUM),
                    ("dynamic", DYNAMIC), ("long", LONG),
                ] {
                    if axes.iter().any(|a| l.contains(&format!("< {a};"))
                        || l.contains(&format!("< {a} "))
                        || l.contains(&format!("< {a})")))
                    {
                        *out.entry(bucket.to_string()).or_default() += 1;
                        hit = true;
                        break;
                    }
                }
                if !hit {
                    *out.entry("unclassified".to_string()).or_default() += 1;
                }
            }
        }
    }
    out
}

/// The census's REDUCE audit must match what the tree actually contains.
/// When this fails the census needs re-counting -- that is the outcome it
/// exists to force, not an annoyance to silence.
#[test]
fn the_reduction_count_still_matches_the_tree() {
    let counts = reduction_axis_counts();
    let census = include_str!("../csrc/src/tile_alternatives.cuh");

    let long = *counts.get("long").unwrap_or(&0);
    assert_eq!(
        long, 2,
        "the census records exactly 2 reductions over a LONG axis (argmax \
         over `vocab`, both declined) and the tree now has {long}. A new one \
         is a new ARGMAX row waiting to happen: price it against the surface \
         in tile_alternatives.cuh before assuming it wins."
    );

    // Checked in context, not as a bare substring: `census.contains("25")`
    // passes on any four-digit number that happens to contain it, which is
    // how the first version of this test passed while the census said 244
    // and the tree said 161.
    for (bucket, label) in [
        ("small", "<= 1 tile"),
        ("medium", "3-7 tiles (hidden 2816-7168)"),
        ("dynamic", "data-dependent"),
        ("long", "> 16 tiles"),
        ("all_strided", "all strided block reductions"),
        ("unclassified", "axis not in the list above"),
    ] {
        let n = *counts.get(bucket).unwrap_or(&0);
        let row = census
            .lines()
            .find(|l| l.contains(label))
            .unwrap_or_else(|| panic!(
                "the census lost the `{label}` row of the REDUCE audit table"
            ));
        let found: Vec<usize> = row
            .split_whitespace()
            .filter_map(|w| w.parse::<usize>().ok())
            .collect();
        assert!(
            found.contains(&n),
            "the census row `{label}` does not carry {n}, the count this \
             test derives from the tree for `{bucket}` (row reads: {row}). \
             Re-derive the REDUCE audit table -- a census derived from \
             source is worth exactly what its last re-derivation was."
        );
    }

    let classified: usize = ["small", "medium", "dynamic", "long"]
        .iter()
        .map(|b| counts.get(*b).copied().unwrap_or(0))
        .sum();
    let row = census
        .lines()
        .find(|l| l.trim_start().starts_with("//     classified"))
        .expect("the census lost the `classified` subtotal");
    assert!(
        row.split_whitespace().any(|w| w == classified.to_string()),
        "the census says a different classified subtotal than the {classified} \
         this test counts"
    );
}

#[test]
#[ignore]
fn print_reduction_counts() {
    for (k, v) in reduction_axis_counts() {
        println!("REDUCTION_COUNT {k} = {v}");
    }
}

/// The Tile IR assembly step is the one piece that turns nine measured
/// kernels into kernels this crate could actually fire. It is deliberately
/// not wired into `compile_under` -- that is a policy call about subprocesses
/// in a compile path -- but the recipe and its traps must not be the next
/// person's problem to rediscover.
#[test]
fn the_assembly_recipe_keeps_its_traps() {
    let src = include_str!("../src/runtime/nvrtc.rs");

    assert!(
        src.contains("pub fn assemble_tile_ir"),
        "the assembly step must stay public and callable; it is the whole \
         point of separating it from the policy question"
    );

    // The trap, with the measurement that pins it. `CUDA_ROOT` is needed
    // only when the binary is outside a toolkit, which is precisely the pip
    // layout -- an earlier note said it was always needed and that is wrong.
    assert!(
        src.contains("no CUDA_ROOT   ->  ok")
            && src.contains("no CUDA_ROOT   ->  FAILS"),
        "the three-row experiment must stay: in-tree without CUDA_ROOT works, \
         out-of-tree without it fails, out-of-tree with it works. Two of \
         those rows are needed to state the rule and the third to bound it"
    );
    assert!(
        src.contains("error: failed to compile Tile IR program"),
        "the exact failure text must stay -- it is the only thing a searcher \
         will have, and it names nothing that caused it"
    );

    // The byte-identity claim is what makes this safe to adopt.
    assert!(
        src.contains("byte-identical")
            && src.contains("`--tilecubin`\n/// IS `--tilebc` plus this"),
        "the equivalence with nvcc must stay stated; without it, adopting \
         this step looks like inventing a build path rather than spelling \
         out the one nvcc already runs"
    );

    // And the binding gap, which is the reason this takes bytes.
    assert!(
        src.contains("nvrtcGetTileIR") && src.contains("does **not**
/// bind"),
        "the cudarc binding gap must stay recorded -- it is why the \
         byte-producing half is not here, and someone will look for it"
    );
}
