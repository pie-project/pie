//! The proofs these two crates cite by name — do they exist?
//!
//! This tree argues in comments. A module says why it does NOT have a `Drop`,
//! or why a port is checked against the shader instead of the original, and
//! then names the test that settles it. That naming is the load-bearing part:
//! the argument is only worth reading if the evidence can be found, and a
//! reader who greps the name and finds nothing has to redo the reasoning from
//! scratch — or, worse, believe it.
//!
//! Nothing checked those names, and three had rotted:
//!
//! * `shell.rs` cited `a_shell_can_be_dropped_with_its_buffers_still_alive` as
//!   the evidence that dropping a `Shell` with live buffers is safe. No such
//!   test exists anywhere in the workspace. The proof does —
//!   `a_buffer_outlives_the_device_handle_that_made_it` — under a name the
//!   comment never had.
//! * `rope.rs` said the port is checked against the shader, "see
//!   `the_ladder_this_driver_builds_is_the_one_the_shader_raises`". That test
//!   is `driver-vulkan`'s. This crate's is
//!   `the_ladder_is_what_the_base_form_raises`, and a reader grepping here
//!   would conclude the port has no proof.
//! * `kernels-wgpu/tests/gpu.rs` had an intra-doc link to
//!   `how_long_a_paged_decode_takes`, which was renamed to
//!   `how_long_a_decodes_kernels_take` when the benchmark was rewritten to
//!   stop timing the shader compiler. rustdoc does not lint links in a test
//!   binary, so nothing said.
//!
//! None was a missing proof. All three were a reader sent to the wrong place,
//! which is the same cost.
//!
//! # Why the filter is derived and not written
//!
//! A comment in these crates backticks a great many identifiers, and most are
//! not tests: kernel entrypoints (`affine_qmm_t_bias`), `wgpu` limits
//! (`max_storage_buffers_per_shader_stage`), WGSL helpers (`pie_f32_to_bf16`),
//! plan fields (`kv_write_lower_bounds`). A list of exclusions would need
//! maintaining and would go stale exactly the way the citations did.
//!
//! So a citation has to pass two tests, and each removes what the other lets
//! through.
//!
//! **It has to START like a test.** Test names here are sentences, and their
//! first word comes from a small vocabulary — `a`, `the`, `every`, `no`,
//! `one`, `how` — that no `wgpu` limit or WGSL helper uses. That vocabulary is
//! collected from the `#[test]` functions that actually exist, so it widens
//! when a test is added with a new opening word and can never narrow below the
//! words in use. It disposes of `max_…`, `min_…`, `pie_…`, `kv_…`, `sdpa_…`
//! and `affine_…` without naming any of them.
//!
//! **It has to READ like one.** The vocabulary alone is not enough, because
//! some tests here open with a domain word — there are tests beginning `attn`,
//! `moe` and `rope` — which then admits `attn_sinks`, `moe_tile_rows` and
//! `rope_neox_decode`, none of which is a test. So a citation must also
//! contain a GRAMMAR word: `is`, `the`, `with`, `its`, `still`, `which`. A
//! sentence has one and an identifier does not. That list is written out
//! rather than derived, and it is deliberately short of the prepositions that
//! appear in API names — no `to`, or `copy_buffer_to_buffer` would qualify.
//! It also has to be four words long, which is the shortest `#[test]` in
//! either crate that reads like a sentence (`a_partial_tile_refuses`); below
//! that the citations are `a_rows` and `no_mangle`.
//!
//! The three rotted citations pass both: `…_with_its_buffers_still_alive`,
//! `the_ladder_this_driver_builds_is_…`, `how_long_a_paged_decode_takes`.
//!
//! # The escape, and why it is a whole word
//!
//! Citing a SIBLING's test is legitimate and common — this backend is a port
//! and its arguments lean on the ports it came from. It is also exactly what
//! `rope.rs` did wrong, by citing one without saying so. So a citation is
//! allowed to name something absent from these crates if the sentence it sits
//! in names the crate it came from, within a two-line window because prose
//! wraps. That turns the loose reference into the useful one: not "see this
//! test" but "see `driver-vulkan`'s this test".

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The two crates whose comments are checked, and whose tests resolve them.
fn crates() -> Vec<PathBuf> {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    vec![here.clone(), here.join("../kernels-wgpu")]
}

/// A citation may name something absent if it says whose it is.
const SIBLINGS: &[&str] = &[
    "driver-vulkan",
    "driver-metal",
    "driver-cuda",
    "kernels-vulkan",
    "kernels-metal",
    "kernels-cuda",
    // The shared crate, and a real citation target: the cross-backend gate
    // lives in `kernels/tests/shader_backends_agree.rs`, so a comment in
    // either of these crates that points at a fleet-wide proof points there.
    // Not ambiguous with the four above — the rule matches the whole
    // possessive, and `kernels-vulkan`'s does not end in "kernels`'s".
    "kernels",
];

/// Words a sentence has and an identifier does not.
///
/// Short of the prepositions that turn up in API names — no `to`, or
/// `copy_buffer_to_buffer` would read as a test — and short of anything that
/// is also a tensor or kernel term.
const GRAMMAR: &str = "\
    a is are was were the an that this these those it its with without than
    then when where why what which who whose does did cannot still only
    every each both same other own but never ever nothing anything
    something because rather instead and or not no any all more less much
    many enough would will must should asks says said means here there now
    yet even just";

fn rust_files(dir: &Path, into: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if name == "target" {
            continue;
        }
        if path.is_dir() {
            rust_files(&path, into);
        } else if path.extension().is_some_and(|e| e == "rs") {
            into.push(path);
        }
    }
}

/// Every `fn` these crates define, and the first word of every `#[test]` one.
fn defined() -> (BTreeSet<String>, BTreeSet<String>) {
    let mut names = BTreeSet::new();
    let mut vocabulary = BTreeSet::new();
    for root in crates() {
        let mut files = Vec::new();
        rust_files(&root, &mut files);
        for file in files {
            let text = std::fs::read_to_string(&file).expect("a readable source file");
            let mut is_test = false;
            for line in text.lines() {
                let trimmed = line.trim_start();
                if trimmed.starts_with("#[test]") {
                    is_test = true;
                    continue;
                }
                let Some(name) = after(trimmed, "fn ") else {
                    if !trimmed.starts_with('#') && !trimmed.is_empty() {
                        is_test = false;
                    }
                    continue;
                };
                if is_test {
                    if let Some(word) = name.split('_').next() {
                        vocabulary.insert(word.to_string());
                    }
                    is_test = false;
                }
                names.insert(name);
            }
        }
    }
    (names, vocabulary)
}

/// A path from `crates/` down, because the absolute one is noise in a failure.
fn shorten(path: &Path) -> String {
    let full = path.display().to_string();
    let from = match full.rfind("crates/") {
        Some(at) => full[at..].to_string(),
        None => full,
    };
    // `crates()` reaches the sibling crate as `driver-wgpu/../kernels-wgpu`.
    match from.find("/../") {
        Some(at) => {
            let head = &from[..at];
            let keep = head.rfind('/').map_or("", |s| &head[..=s]);
            format!("{keep}{}", &from[at + 4..])
        }
        None => from,
    }
}

/// The identifier following `what`, if the line begins one there.
fn after(line: &str, what: &str) -> Option<String> {
    let rest = line.strip_prefix(what)?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty() && name.starts_with(|c: char| c.is_ascii_lowercase() || c == '_'))
        .then_some(name)
}

/// A file's comments as one stream, with the source line of each character.
///
/// Line-at-a-time will not do, for two reasons. The possessive that licenses a
/// sibling citation is routinely a line above the name it licenses:
///
/// ```text
/// /// It is the counterpart of `driver-vulkan`'s
/// /// `a_windowed_row_is_staged_as_that_window_and_padded_to_the_fires_pitch`,
/// ```
///
/// And the citations themselves wrap, mid-identifier, in both directions —
/// `` `a_decode_attention_module_is_half_the_ `` / `` head_it_serves` `` in
/// `geometry.rs`, and `` `every_source_a_stated_row_names_is_one `` /
/// `` _this_driver_can_work_out` `` in `binding.rs`. Those are the LONGEST
/// names in the tree, which is to say the ones most likely to be renamed and
/// least likely to be re-grepped; a scanner that reads one line at a time
/// misses exactly them. So the separator between lines is dropped while a
/// backtick span is open.
///
/// A run of code between two comment blocks pushes a NUL, so nothing can read
/// backwards from one block into another.
fn prose(text: &str) -> (String, Vec<usize>) {
    let mut stream = String::new();
    let mut at = Vec::new();
    let mut inside = false;
    for (n, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        let body = trimmed
            .strip_prefix("///")
            .or_else(|| trimmed.strip_prefix("//!"))
            .or_else(|| trimmed.strip_prefix("//"));
        let (join, body) = match body {
            Some(body) => {
                if inside {
                    // A wrapped name resumes after the comment marker and the
                    // one space that follows it, and neither is part of it.
                    (' ', body.strip_prefix(' ').unwrap_or(body))
                } else {
                    (' ', body)
                }
            }
            None => {
                inside = false;
                ('\0', "")
            }
        };
        for c in body.chars() {
            if c == '`' {
                inside = !inside;
            }
            stream.push(c);
            at.push(n + 1);
        }
        if !inside || join == '\0' {
            stream.push(join);
            at.push(n + 1);
        }
    }
    (stream, at)
}

/// Does a sibling crate's possessive stand immediately before this citation?
///
/// `` `driver-vulkan`'s `a_windowed_row_…` `` licenses the name; a
/// `driver-metal` mentioned two clauses earlier in the same paragraph does
/// not, which is the difference between "see that sibling's test" and "this
/// module was ported from somewhere". Sixty characters is the whole allowance,
/// and a NUL — a line of code — ends it.
/// Is this citation inside a block that RETIRES the test it names?
///
/// A retirement has to name what it retired — "It was `x`, and it asserted
/// ..." — and that name necessarily no longer resolves, because retiring it is
/// what deleted it. Without this, the rule would forbid the one sentence a
/// retirement exists to write, and the only way to satisfy it would be to
/// describe the lost test WITHOUT naming it, which is exactly the vagueness
/// this file is meant to prevent.
///
/// Scoped to the CURRENT comment block, and to the WHOLE of it. `prose`
/// separates blocks with `\0`, so a `RETIRED:` three comments earlier cannot
/// excuse a live citation -- but the marker may sit on either side of the name
/// within one block, because the two idioms put it on opposite sides: a
/// retirement writes "RETIRED: it was `x`" and an epitaph writes "`x` STOOD
/// HERE". Looking only at the text BEFORE the citation admits the first and
/// refuses the second.
///
/// # Two markers, because the tree writes epitaphs two ways
///
/// `RETIRED` was the only one here and `X STOOD HERE` is the commoner idiom by
/// a wide margin -- sixty-odd sites across ten crates, against a handful. The
/// four this missed were all the second form, in `kernels-wgpu`'s entrypoint
/// census, where two tests went tautological when the census started being
/// READ OFF the tree instead of written beside it.
///
/// Both are deliberate and both are loud. What neither admits is a citation
/// with no marker at all, which is still the case this exists to catch: a
/// comment that names a test as its evidence and is simply out of date.
fn retiring(stream: &str, at: usize) -> bool {
    let from = stream[..at].rfind('\0').map_or(0, |i| i + 1);
    let to = stream[at..].find('\0').map_or(stream.len(), |i| at + i);
    let block = &stream[from..to];
    block.contains("RETIRED") || block.contains("STOOD HERE")
}

fn whose(before: &str) -> bool {
    let from = before.len().saturating_sub(60);
    let tail = before[from..].trim_end();
    if tail.contains('\0') {
        return false;
    }
    let tail = tail.strip_suffix('`').unwrap_or(tail);
    SIBLINGS
        .iter()
        .any(|s| tail.ends_with(&format!("{s}`'s")) || tail.ends_with(&format!("{s}'s")))
}

/// No source file in either crate carries a merge conflict marker.
///
/// # Why this is a test and not a habit
///
/// It happened. `turns.rs` went to `origin/rewrite` with `<<<<<<< HEAD` in
/// its import block, and the branch did not compile for anyone who pulled it.
/// The mechanism is worth writing down because it is invisible: a rebase stops
/// on a conflict, you resolve the file you were shown, and `git add -A` stages
/// EVERY conflicted file — including one you were not looking at.
/// `git rebase --continue` then reports success, because from git's side the
/// conflict is resolved.
///
/// Nothing caught it for hours. `cargo test -p kernels-wgpu` and `-p kernels`
/// were green throughout, because the marker was in a crate those do not
/// build, and the `--lib` run that would have caught it was made before the
/// rebase that introduced it.
///
/// # What this actually buys, which is narrower than it first looks
///
/// `rustc` rejects a conflict marker in code it COMPILES, with a note saying
/// exactly that. So this test cannot fire for a marker in `driver-wgpu`
/// itself — the test binary would not build either. What it covers is the gap
/// that let the real one through: a marker in a crate the command you ran
/// does not build. This target compiles `driver-wgpu` and READS
/// `kernels-wgpu`, so `cargo test -p driver-wgpu` now fails on a marker in
/// either.
///
/// The rest of the answer is procedural and belongs in a commit message, not
/// a test: grep for markers before `git rebase --continue`, never after.
///
/// Falsified by pointing the detector at `//! `, which every module here
/// opens with: 84 files, all flagged, so the walk is reading them.
#[test]
fn no_source_file_carries_a_conflict_marker() {
    fn walk(dir: &std::path::Path, into: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                walk(&path, into);
            } else if path
                .extension()
                .is_some_and(|x| x == "rs" || x == "wgsl" || x == "toml")
            {
                into.push(path);
            }
        }
    }

    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .to_path_buf();
    let mut files = Vec::new();
    for crate_dir in ["driver-wgpu", "kernels-wgpu"] {
        walk(&root.join(crate_dir), &mut files);
    }
    assert!(
        files.len() > 60,
        "only {} files were scanned, which is too few for two crates -- the \
         walk is not finding them",
        files.len()
    );

    let mut wounded = Vec::new();
    for file in &files {
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        for (at, line) in text.lines().enumerate() {
            if line.starts_with("<<<<<<< ") || line.starts_with(">>>>>>> ") {
                wounded.push(format!("{}:{}", file.display(), at + 1));
            }
        }
    }
    assert!(
        wounded.is_empty(),
        "a merge conflict marker is in the source, so this crate does not \
         compile. `git add -A` during a rebase stages every conflicted file, \
         not the one you were shown. {wounded:#?}"
    );
}

/// Every test-shaped name a comment cites resolves, or says whose it is.
#[test]
fn every_proof_these_crates_cite_by_name_can_be_found() {
    let (names, vocabulary) = defined();
    // A FLOOR ON PURPOSE, and worth saying why when its sibling below is not.
    // This counts the distinct first words of `#[test]` names, so its subject
    // is prose that moves whenever anyone writes a test with a new verb --
    // pinning it would fail on a correct addition, every time, and a gate
    // that cries on correct work is one somebody deletes. It reads 27 as
    // this is written. Eight is not near that and does not need to be: the
    // only thing it can catch is the scan finding nothing at all, which is
    // the only failure a vocabulary this loose HAS.
    assert!(
        vocabulary.len() >= 8,
        "only {} first-words were collected from `#[test]` functions, which is \
         too few for the filter to mean anything — the scan is not finding the \
         tests. It read 27 when this floor was last measured. Vocabulary: \
         {vocabulary:?}",
        vocabulary.len(),
    );

    let mut dangling: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for root in crates() {
        let mut files = Vec::new();
        rust_files(&root, &mut files);
        for file in files {
            // This file's own header quotes the three rotted names as the
            // examples that motivate it, which is the one place they are
            // supposed to appear and not resolve. Only its PROSE is skipped:
            // its `fn`s must still be visible, or a citation of a test defined
            // here cannot resolve.
            if file.file_name().is_some_and(|n| n == "citations.rs") {
                continue;
            }
            let text = std::fs::read_to_string(&file).expect("a readable source file");
            let shown = shorten(&file);
            let (stream, at) = prose(&text);

            let mut i = 0;
            while let Some(open) = stream[i..].find('`').map(|o| i + o) {
                let Some(close) = stream[open + 1..].find('`').map(|c| open + 1 + c) else {
                    break;
                };
                i = close + 1;
                let cited: String = stream[open + 1..close]
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if cited.is_empty() {
                    continue;
                }
                // Four tokens, because the shortest `#[test]` in either crate
                // that reads like a sentence has four —
                // `a_partial_tile_refuses`, `an_unstated_rule_refuses`. Below
                // that a name is `a_rows` or `no_mangle`, which pass both
                // filters and are not tests.
                let words: Vec<&str> = cited.split('_').collect();
                let opens_like_a_test = vocabulary.contains(words[0]);
                let reads_like_a_test = words.len() >= 4
                    && words
                        .iter()
                        .any(|w| GRAMMAR.split_whitespace().any(|g| g == *w));
                if !opens_like_a_test || !reads_like_a_test || names.contains(&cited) {
                    continue;
                }
                if whose(&stream[..open]) || retiring(&stream, open) {
                    continue;
                }
                let line = at.get(open).copied().unwrap_or(0);
                dangling
                    .entry(cited)
                    .or_default()
                    .push(format!("{shown}:{line}"));
            }
        }
    }

    let report: Vec<String> = dangling
        .iter()
        .map(|(name, at)| format!("  `{name}` — cited at {}", at.join(", ")))
        .collect();
    assert!(
        report.is_empty(),
        "{} proof(s) cited by name that neither crate defines. A comment that \
         names its evidence is only worth reading if the name resolves: either \
         the test was renamed and the citation was not, or it belongs to a \
         sibling — in which case say so in the same sentence (`driver-vulkan`'s \
         `{}`) and this test will accept it.\n{}",
        dangling.len(),
        dangling.keys().next().map_or("...", String::as_str),
        report.join("\n"),
    );
}

/// Which refusals a test names, and which twenty-three it does not.
///
/// `device.rs` gives the principle, about `Failed` and its by-value
/// comparison: "a test that asserts WHICH refusal came back is the only way
/// an alignment failure stays distinguishable from a length one."
///
/// It is a principle this crate mostly keeps — **seventy-seven of one hundred**
/// refusal variants are named in a test — and nothing measured the rest. A
/// refusal nothing names is one whose condition could be inverted, or whose
/// message could describe a different fault, with every suite still green: it
/// is CONSTRUCTED, so the sweep in `frames.rs` passes it, and it is `pub`, so
/// `dead_code` says nothing.
///
/// So the gap is written down, and the list may only SHRINK. Naming one in a
/// test fails this, and the fix is to delete its line — that edit is what
/// records the progress. Adding a refusal without a test that names it fails
/// it the other way.
///
/// # The twenty-five are four different things, and only one is a real gap
///
/// * **wrappers, about half of them** — `Unstepped::{Failed, Unfired, Unread,
///   Unhoused, Unstageable, Uncovered}`, `Unlaunched::{Unserved, Unstepped}`,
///   `Unopened::{Absent, Device, Unservable}`, `Unforked::*`, `Unresized::*`.
///   Each carries an inner refusal, and the tests assert the INNER one, which
///   is the part that says what went wrong. They are on the list because the
///   census cannot see through a wrapper, and leaving them off would have
///   meant a hand-maintained exception rule doing the same job less visibly.
/// * **reachable and untested** — down to the `Undispatchable` planning
///   refusals: `Layout`, `Unresolved` and `Contiguous`, since `Scalars` has
///   left. Each needs a driver NUMBER the resolver cannot work out, and every
///   row naming one is a paged attention with eleven buffers to satisfy first
///   — a fixture, not a test. The `Unfired` family is down to `Impossible`,
///   which this crate calls unreachable on purpose, and the `Unread` family is
///   closed. Twelve have left this group. The `Unfired` family is now down
///   to `Impossible`, which is the one this crate calls unreachable on
///   purpose. What made the other three cheap was a single fixture --
///   `one_launch`, a plan of one rectangle over a 256-byte arena -- pointed at
///   three different things: a symbol the tree has not got (`NoModule`), a
///   module store handing back text `naga` will not parse (`Unreadable`), and
///   a real entrypoint whose row does not state the operand the plan supplies
///   (`Unplannable`). None needs weights, a pool or a shell.
///
///   The estimate that kept them here was "each needs a fire built to fail in
///   one specific way, which is a test apiece". The fire was the expensive
///   part in that estimate and it was the cheap part: `Unplannable` was first
///   met BY ACCIDENT, while checking whether a different test's guard could
///   fire at all. The four `Unread` refusals needed no fire whatever --
///   `serve::logits` decides all four from the `Readout` the plan states, and
///   nothing has to have run.
/// * **reachable only on other hardware** — `Ceiling::{StorageBinding,
///   UniformBinding, Invocations}`, `Failed::Unreachable` and
///   `Unopened::Unreachable` name limits this adapter is nowhere near.
///   `Ceiling::BufferSize` is the one of the four a test does reach, which is
///   why the enum is here rather than excused wholesale.
/// * **not reachable at all** — `Unstageable::MaskTooWide` fires on a mask row
///   longer than a `u32` can address, so provoking it means allocating four
///   gibibytes of mask to exercise one comparison.
///
/// # The selector is a naming convention, and it was too narrow
///
/// An enum is taken as a refusal if it is `Un…`, `Failed`, `Mismatch`,
/// `Misplaced` or `Ceiling`. `Misplaced` was NOT on that list, and it is a
/// refusal returned as one — so three variants sat outside a census whose
/// whole subject is coverage. They turned out to be tested, in `binding.rs`'s
/// own module, which is the lucky version of that mistake and not a reason to
/// keep making it.
///
/// So the selector is checked too: anything this crate returns as a `Result`
/// error must be selected, and dropping `Misplaced` again fails with
/// `{"Misplaced"} is returned as a `Result` error ... add it to the
/// predicate`. The converse is deliberately not required — `Ceiling` names
/// WHICH limit inside `Failed::PastLimit` and is never an error type itself,
/// so a rule built only on error position would have dropped it.
///
/// The point of the census is not that thirty-three is too many. It is that
/// thirty-seven was not a number anybody had — and the first probe written for
/// it said twenty-two, by swallowing non-test code from `src` that happened to
/// sit after a `#[cfg(test)]`. The number in a sweep is the sweep's until the
/// sweep has been checked; three of the extras were confirmed by hand.
#[test]
fn every_refusal_this_crate_builds_is_one_a_test_names() {
    /// Constructed, and named by no test.
    const UNNAMED: &[&str] = &[
        "Ceiling::Invocations",
        "Ceiling::StorageBinding",
        "Ceiling::UniformBinding",
        "Failed::Unreachable",
        // Unreachable today, and deliberately built anyway: no crossed body
        // states more than one dispatch, so `plan_one`'s narrow shape costs
        // nothing. A two-pass reduction is two entrypoints over one
        // statement, and the day one arrives this is a named refusal rather
        // than a silently dropped pass. Give it a test when a body does it.
        "Undispatchable::Multiple",
        "Unfired::Impossible",
        "Unforked::Device",
        "Unforked::Unhoused",
        "Unlaunched::Unserved",
        "Unlaunched::Unstepped",
        "Unopened::Absent",
        "Unopened::Device",
        "Unopened::Unreachable",
        "Unopened::Unservable",
        "Unresized::Device",
        "Unresized::Stranded",
        "Unstageable::MaskTooWide",
        "Unstepped::Failed",
        "Unstepped::Uncovered",
        "Unstepped::Unfired",
        "Unstepped::Unhoused",
        "Unstepped::Unread",
        "Unstepped::Unstageable",
    ];

    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let (mut sources, mut suites) = (Vec::new(), Vec::new());
    rust_files(&here.join("src"), &mut sources);
    rust_files(&here.join("tests"), &mut suites);
    assert!(
        sources.len() > 5 && suites.len() > 3,
        "{} sources and {} suites is not this crate",
        sources.len(),
        suites.len()
    );

    // A `#[cfg(test)]` module inside `src/` is a test too, and several
    // refusals are only ever named in one.
    let mut named = String::new();
    for file in suites.iter().chain(&sources) {
        // NOT this file. `UNNAMED` below is twenty-three string literals, so
        // counting it would have every refusal "named in a test" by the very
        // census that says they are not -- which is exactly what happened the
        // first time this ran, and the empty result is what gave it away.
        if file.file_name().is_some_and(|n| n == "citations.rs") {
            continue;
        }
        let text = std::fs::read_to_string(file).expect("a readable source file");
        let from = if sources.contains(file) {
            match text.find("#[cfg(test)]") {
                Some(at) => at,
                None => continue,
            }
        } else {
            0
        };
        for line in text[from..].lines() {
            named.push_str(line.split_once("//").map_or(line, |(before, _)| before));
            named.push('\n');
        }
    }

    let mut refusals = Vec::new();
    for file in &sources {
        let text = std::fs::read_to_string(file).expect("a readable source file");
        for (enum_name, variant) in refusal_variants(&text) {
            refusals.push(format!("{enum_name}::{variant}"));
        }
    }
    // PINNED, not floored. This was `>= 90` against a hundred, which is a
    // gate with ten variants of slack in the one direction that matters: a
    // refusal the code can still return but no test names is exactly what
    // this file exists to catch, and ten of them could go without a word.
    // A floor only guards the collapse of the PARSER; the census guards the
    // census. Both readings are wanted, so the number is the count.
    assert_eq!(
        refusals.len(),
        100,
        "the refusal census moved. If variants were added, name them in a \
         test and raise this; if they were removed, say which in the commit \
         and lower it. If it collapsed to a handful, the parser has stopped \
         reading the declarations it thinks it is, which is what the floor \
         this replaced was for and is the only reading a floor gave."
    );

    // And the SELECTOR is checked, because it is a naming convention and this
    // census is only as wide as it. `Misplaced` is a refusal, is returned as
    // one, and matched none of `Un…`/`Failed`/`Mismatch`/`Ceiling` -- so three
    // variants sat outside a census whose whole subject is coverage. They
    // turned out to be tested, in `binding.rs`'s own module, which is the
    // lucky version of that mistake and not a reason to keep making it.
    //
    // Anything this crate returns as a `Result` error must be selected. The
    // converse is NOT required: `Ceiling` names which limit inside
    // `Failed::PastLimit` and is never an error type itself, so a rule built
    // only on error position would have dropped it. Two shapes; the one that
    // can be checked is.
    let declared_locally: BTreeSet<String> = sources
        .iter()
        .flat_map(|f| {
            let text = std::fs::read_to_string(f).unwrap_or_default();
            text.match_indices("pub enum ")
                .map(|(at, _)| {
                    text[at + "pub enum ".len()..]
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let selected: BTreeSet<&str> = refusals
        .iter()
        .filter_map(|r| r.split("::").next())
        .collect();
    let mut missed = BTreeSet::new();
    for file in &sources {
        let text = std::fs::read_to_string(file).expect("a readable source file");
        for (at, _) in text.match_indices("Result<") {
            let rest = &text[at..];
            let Some(end) = rest.find('>') else { continue };
            let Some(err) = rest[..end].rsplit(',').next() else {
                continue;
            };
            let err = err.trim().rsplit("::").next().unwrap_or("").trim();
            if declared_locally.contains(err) && !selected.contains(err) {
                missed.insert(err.to_string());
            }
        }
    }
    assert!(
        missed.is_empty(),
        "{missed:?} is returned as a `Result` error by this crate and the \
         refusal selector does not pick it up, so its variants sit outside a \
         census about exactly that. Add it to the predicate."
    );

    let unnamed: BTreeSet<String> = refusals
        .iter()
        .filter(|r| !named.contains(r.as_str()))
        .cloned()
        .collect();
    let listed: BTreeSet<String> = UNNAMED.iter().map(|s| (*s).to_string()).collect();
    assert_eq!(
        unnamed, listed,
        "the refusals no test names are no longer the ones UNNAMED lists. If \
         you gave one a test, delete its line — that edit is what records the \
         progress. If you added a refusal, give it a test that asserts it BY \
         NAME, because a refusal nothing names is one whose condition could be \
         inverted with every suite still green."
    );

    // The two numbers the doc above quotes, ASSERTED rather than copied. A
    // count written into prose beside the assertion that produces it is the
    // rot this file has had to fix four times: the set check passes happily
    // while the sentence describing it goes stale.
    //
    // It had. The sentence said "sixty-four of ninety-seven" and the census
    // said a hundred and one, because the set check below can only see a
    // refusal that is added and left UNNAMED. One that is added WITH a test
    // moves both totals and changes nothing this test compares -- so the
    // headline drifted by three, silently, in the direction that looks like
    // less coverage than there is. Here the sentence fails.
    assert_eq!(
        (refusals.len(), refusals.len() - unnamed.len()),
        (100, 77),
        "this test's own doc says seventy-seven of one hundred refusal variants \
         are named by a test. Update the sentence with the number."
    );
}

/// `(enum, variant)` for every `pub enum` in a file whose name reads as a
/// refusal — `Un…`, `Failed`, `Mismatch`, `Ceiling`.
fn refusal_variants(text: &str) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let Some(rest) = lines[i].trim().strip_prefix("pub enum ") else {
            i += 1;
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let refusal = name.starts_with("Un")
            || name == "Failed"
            || name == "Mismatch"
            || name == "Misplaced"
            || name == "Ceiling";
        let mut depth = 0i32;
        for line in &lines[i..] {
            depth += line.matches('{').count() as i32;
            depth -= line.matches('}').count() as i32;
            i += 1;
            if refusal
                && let Some(body) = line.strip_prefix("    ")
                && body.starts_with(|c: char| c.is_ascii_uppercase())
            {
                let variant: String = body
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                let after = body[variant.len()..].trim_start();
                if after.starts_with('{') || after.starts_with('(') || after.starts_with(',') {
                    found.push((name.clone(), variant));
                }
            }
            if depth == 0 {
                break;
            }
        }
    }
    found
}

/// The seam sketches describe the seam that exists.
///
/// This crate's front door — `lib.rs`'s "**The seam is exactly this, and
/// nothing else**" — and `binding::Resolve`'s own "What the device half
/// implements" are both ```ignore blocks. They have to be: they are signature
/// sketches with `{ .. }` bodies, and they would not compile. So they are the
/// only code in these crates that `cargo test --doc` counts and never checks —
/// two ignored doctests, which is the entire doctest population of both
/// feature halves.
///
/// That makes them the exact shape this file exists for: a claim about the
/// code, written beside the code, with nothing comparing the two. A method
/// added to `Resolve` with a default would appear in neither sketch and break
/// nothing, and the crate's stated seam would quietly stop being the seam.
///
/// So the sketches are compared with the traits by name. Not by signature —
/// the sketches elide bodies and abbreviate types on purpose, and a control
/// that demanded they match textually would just make them useless. Names are
/// what a reader takes away and names are what is checked.
#[test]
fn the_seam_these_crates_sketch_is_the_seam_they_declare() {
    /// Every `fn <name>` in the first `pub trait <trait>` block of `text`.
    fn trait_methods(text: &str, name: &str) -> BTreeSet<String> {
        let head = format!("pub trait {name}");
        let at = text
            .find(&head)
            .unwrap_or_else(|| panic!("`{head}` is not in binding.rs any more"));
        let body = &text[at..];
        // The trait ends at the first line that is a lone `}`.
        let end = body
            .find("\n}")
            .unwrap_or_else(|| panic!("`{head}` never closes"));
        names_of_fns(&body[..end])
    }

    /// Every `fn <name>` in a chunk of text, however it is punctuated after.
    fn names_of_fns(text: &str) -> BTreeSet<String> {
        text.match_indices("fn ")
            .map(|(at, _)| {
                text[at + 3..]
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect::<String>()
            })
            .filter(|n| !n.is_empty())
            .collect()
    }

    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let binding = std::fs::read_to_string(here.join("src/binding.rs")).expect("binding.rs");
    let lib = std::fs::read_to_string(here.join("src/lib.rs")).expect("lib.rs");

    let mut declared = trait_methods(&binding, "Resolve");
    declared.extend(trait_methods(&binding, "Allocation"));
    assert!(
        declared.contains("weight") && declared.contains("size"),
        "the trait scan found {declared:?}, which is not this seam"
    );

    // Every ```ignore block in either file, and there should be exactly the two.
    let mut sketches = Vec::new();
    for (what, text) in [("lib.rs", &lib), ("binding.rs", &binding)] {
        for (at, _) in text.match_indices("```ignore") {
            let rest = &text[at + "```ignore".len()..];
            let end = rest.find("```").unwrap_or_else(|| {
                panic!("an ```ignore block in {what} never closes, so this scan read to the end")
            });
            sketches.push((what, names_of_fns(&rest[..end]), rest[..end].to_owned()));
        }
    }
    assert_eq!(
        sketches.len(),
        2,
        "this crate had two seam sketches and now has {}: {sketches:?}. A new \
         one is welcome, but it has to be added here or it is unchecked like \
         they were.",
        sketches.len()
    );

    for (what, signatures, whole) in &sketches {
        let invented: Vec<_> = signatures.difference(&declared).collect();
        assert!(
            invented.is_empty(),
            "{what}'s seam sketch writes a signature for {invented:?}, which \
             no trait in binding.rs declares. A reader implementing from that \
             sketch writes a method nothing calls."
        );
        // The other direction is checked against the whole block rather than
        // its signatures, and the difference is not pedantry: `binding.rs`
        // spells `kv`, `number` and `table` in a COMMENT -- "defaulted; a
        // resolver serving a text without paged attention need not state
        // them" -- which informs the reader exactly as well as a signature
        // would and is the better sketch. Demanding a `fn` line for each was
        // this control's first draft and it failed on its first run against
        // prose that was doing its job.
        let missed: Vec<_> = declared.iter().filter(|m| !whole.contains(*m)).collect();
        assert!(
            missed.is_empty(),
            "{what}'s seam sketch does not mention {missed:?} anywhere, and \
             the traits DO declare them. `lib.rs` says the seam is \"exactly \
             this, and nothing else\"; a defaulted method left out breaks no \
             build and makes that sentence false."
        );
    }
}
