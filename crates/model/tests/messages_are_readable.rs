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
            //
            // `examples/` holds PROBE PROGRAMS, whose stdout is a report: a
            // column of labels, a run of spaces, a column of values. That is
            // the alignment this file's header promises not to flag, and it
            // is indistinguishable from the slip by any predicate over one
            // line, because a probe's labels are ordinary lowercase words
            // (`device        {}`, `headers:       {} carried`). Nothing here
            // is a message an operator reads — an example is run by hand, by
            // the person who just built it. Every shipped message still sits
            // in `src/`, `tests/` or a binary, all of which are scanned.
            if path
                .file_name()
                .is_some_and(|n| n == "target" || n == "examples")
            {
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
/// `before` ends at the run and `after` begins at it.
///
/// # What this asks, and what it used to ask
///
/// It asks whether there is prose on BOTH sides. It used to ask whether
/// a word ABUTS the run, which is a different question and a much worse
/// one, because the character right before a line break is whatever the
/// sentence happened to end on. Two rounds of widening found five real
/// sites hiding behind that:
///
/// ```text
/// ...which no          scheme...      <- a two-letter word
/// ...with no '_scales'          beside it     <- a closing quote
/// ...having no rows —          delete...      <- an em dash
/// ...does not overlap,          so it...      <- a comma
/// ...is not shaped          [experts, ...     <- opens with a bracket
/// ```
///
/// Each was a real garbled message, and each was invisible to a
/// predicate looking at one character. Prose-on-both-sides sees all
/// five and does not care what punctuation the break landed on.
///
/// # The two things that are not slips
///
/// A run doing alignment sits in a literal that is a table or a
/// diagram, and those have no lowercase words on one side or the other
/// (`NAME    WIDTH`, `{:>8}    {}`, `+---+    +---+`) — or, when they do
/// have words, the run is followed by a COLUMN MARKER rather than by
/// prose: `avg_missing_at_fire    = {}` and `config    | final_mu` are
/// both real aligned output this must not flag.
///
/// A run inside generated source — the C and Metal `tensor-compiler`
/// emits — is an indent the author wrote, and is told apart by the `\n`
/// escape in front of it. A newline someone deleted leaves no newline
/// behind; that is the whole signature.
fn splits_prose(before: &str, after: &str) -> bool {
    if before.ends_with("\\n") {
        return false;
    }
    // Prose starts the right side: a letter, or a bracket or quote that
    // would open one. Not `=`, `|` or `:`, which are what a column of
    // aligned output puts there.
    //
    // A bare `{` is not one of those brackets. It opens a FORMAT
    // PLACEHOLDER, which is the commonest column marker there is -- an
    // aligned table's whole purpose is a label, a run of spaces, and an
    // interpolated cell:
    //
    // ```text
    // println!("storage      {} buffers per stage (WebGPU guarantees {})");
    // ```
    //
    // An escaped `{{` prints as a literal brace and really could open a
    // phrase, so it is kept. THE COST, the same one the tail rule pays at
    // the other end: a swallowed indent whose continuation begins with a
    // placeholder -- `"the tensor      {name} was not found"` -- is no
    // longer flagged. It is not distinguishable from a table cell by
    // anything visible in one literal, and the file's header promises
    // aligned output will not be flagged.
    let mut rest = after.chars();
    let opens_prose = match rest.next() {
        Some('{') => rest.next() == Some('{'),
        Some(c) => c.is_ascii_alphabetic() || "(`'[\"".contains(c),
        None => false,
    };
    opens_prose && has_word(before) && has_word(after)
}

/// Whether `text` holds a lowercase word of three or more letters,
/// NOT counting the inside of a format placeholder.
///
/// The evidence that a side is prose rather than a column of headings,
/// hex, or single-character cells.
///
/// A placeholder's identifier is not prose: the reader never sees
/// `limits`, they see whatever it interpolates. Counting it read this
///
/// ```text
/// println!("limits       {limits}");
/// ```
///
/// as a sentence split by an indent, when it is one row of a
/// column-aligned table -- the exact case this file's header promises
/// not to flag. Its neighbours `"adapter      {}"` and
/// `"kind         {:?}"` were already safe for the accidental reason
/// that a positional placeholder holds no letters; captured identifiers
/// made the same row unsafe, so the rule is stated rather than inherited
/// from the placeholder's spelling.
///
/// THE COST, stated because a control found it: a genuinely swallowed
/// indent whose tail is a lone placeholder and nothing else --
/// `"...checkpoint          {disagrees}"` -- is no longer flagged. It is
/// indistinguishable from a table cell by this rule, and a joined source
/// line almost never ends there: a continuation carries the rest of its
/// sentence, and a control confirms `"...          disagrees with {it}
/// entirely"` still bites.
fn has_word(text: &str) -> bool {
    let mut prose = String::with_capacity(text.len());
    let mut depth = 0usize;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            // `{{` and `}}` are escaped braces and print as themselves.
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
            }
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ if depth == 0 => prose.push(c),
            _ => {}
        }
    }
    prose
        .split(|c: char| !c.is_ascii_lowercase())
        .any(|word| word.len() >= 3)
}

/// The width at which a run of spaces stops being a gap and starts
/// being an indent.
///
/// Six. A sentence never has six spaces in it, and the shortest indent
/// that could be swallowed is a continuation line inside a nested
/// block, which is deeper than that in every case here — the nine
/// original sites carried runs of 14 to 22.
const INDENT_WIDTH: usize = 6;

/// Every string literal on one source line, as bodies without their quotes.
///
/// One line can hold SEVERAL, and what sits between two of them is code.
/// This used to take the first `"` to the last `"` and call everything
/// between it one body, which is right for the single-literal line the slip
/// produces and wrong for a table:
///
/// ```text
/// DeviceKernel { path: "attn::write_kv",        elem: "device::bf16" },
///                      ^------------------ "body" ------------------^
/// ```
///
/// The run being measured there is the alignment between two fields — the
/// exact case this file's header promises not to flag — and it read as prose
/// on both sides because `write_kv` and `elem` are both lowercase words.
fn literal_bodies(line: &str) -> Vec<&str> {
    let bytes = line.as_bytes();
    let mut bodies = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut j = start;
        while j < bytes.len() && bytes[j] != b'"' {
            // A backslash escapes the next byte, so `\"` does not close.
            j += if bytes[j] == b'\\' { 2 } else { 1 };
        }
        if j > bytes.len() {
            break;
        }
        if j < bytes.len() {
            bodies.push(&line[start..j]);
        }
        i = j + 1;
    }
    bodies
}

/// Does this literal read as a table ROW rather than a sentence?
///
/// A header labels columns and its own neighbours are code, so
/// [`aligned_with_a_neighbour`] cannot see the table it belongs to. What it has
/// instead is SEVERAL runs of padding, each between short labels: prose joined
/// from a source line has exactly one such run, because there was one newline.
/// Three or more is a layout.
fn reads_as_a_table_row(body: &str) -> bool {
    body.split("  ").filter(|part| !part.trim().is_empty()).count() >= 3
        && !body.contains(". ")
}

/// Does a neighbouring line put a word at the same column?
///
/// A COLUMN IS NOT A SWALLOWED INDENT, and the two look identical on one line.
/// A diagnostic table -- `driver-wgpu/tests/hybrid_probe.rs` prints several --
/// pads each row so its numbers line up, and the padding lands mid-sentence
/// exactly as a joined source line's indent does. What tells them apart is the
/// LINE ABOVE OR BELOW: a table's neighbour has a word starting at the same
/// column, and a joined line's does not.
fn aligned_with_a_neighbour(lines: &[&str], idx: usize, col: usize) -> bool {
    [idx.checked_sub(1), Some(idx + 1)]
        .into_iter()
        .flatten()
        .filter_map(|n| lines.get(n))
        .any(|other| {
            other.len() > col
                && other.as_bytes()[col] != b' '
                && col > 0
                && other.as_bytes()[col - 1] == b' '
        })
}

fn collapsed_literals(src: &str) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    let lines: Vec<&str> = src.lines().collect();
    for (idx, line) in src.lines().enumerate() {
        for body in literal_bodies(line) {
            let mut at = 0;
            let mut hit = false;
            while let Some(pos) = body[at..].find(&" ".repeat(INDENT_WIDTH)) {
                let start = at + pos;
                let end = start
                    + body[start..]
                        .find(|c: char| c != ' ')
                        .unwrap_or(body.len() - start);
                let at_col = line.find(&body).map_or(end, |o| o + end);
                if splits_prose(&body[..start], &body[end..])
                    && !aligned_with_a_neighbour(&lines, idx, at_col)
                    && !reads_as_a_table_row(&body)
                {
                    let shown: String = body.chars().take(90).collect();
                    found.push((idx + 1, shown));
                    hit = true;
                    break;
                }
                at = end.max(start + 1);
            }
            if hit {
                break;
            }
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

    // A scan that reaches less than it means to reports the same green as
    // a clean workspace, and this one has now found garbled messages in
    // `model`, `driver-metal` and `tensor-compiler` — three crates, one of
    // which is not even a library. A bare count is too weak a guard: the
    // largest crate alone would satisfy it. Name what has to be reachable.
    for (crate_name, sentinel) in [
        ("model", "src/catalog.rs"),
        ("engine", "src/lib.rs"),
        ("driver-metal", "src/lib.rs"),
        ("driver-cuda", "src/lib.rs"),
        ("tensor-compiler", "src/lib.rs"),
        ("model-loader", "src/lib.rs"),
    ] {
        let wanted = crates_dir().join(crate_name).join(sentinel);
        assert!(
            sources.contains(&wanted),
            "the scan does not reach {}; it has narrowed",
            wanted.display()
        );
    }
    assert!(
        sources.len() > 600,
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
    // Literal widths, not `INDENT_WIDTH`: a fixture written in terms of
    // the constant moves with it, and pins nothing. Six is caught and
    // five is not, which is the threshold said in the only way that
    // survives someone editing the constant.
    let edge = " ".repeat(6);
    for offender in [
        format!("    \"the row ships no v_proj{gap}and the text projects one\","),
        // Each of these hid behind a predicate that looked at the one
        // character in front of the run. What is in front of a line
        // break is whatever the sentence ended on.
        format!("    \"is a packed weight with no{gap}scales this scheme describes\","),
        format!("    \"is not quantized in groups of 64, which is{gap}what it reads\","),
        format!("    \"an MXFP4 block tensor with no '_scales'{gap}beside it\","),
        format!("    \"MXFP4 tensor is not shaped{gap}[experts, rows] against its scales\","),
        format!("    \"listed as having no rows —{gap}delete the line or ship one\","),
        format!("    \"the move does not overlap,{gap}so it cannot tell them apart\","),
        format!("    \"a run exactly this wide{edge}is already an indent\","),
    ] {
        assert_eq!(
            collapsed_literals(&offender).len(),
            1,
            "a swallowed indent must be caught: {offender}"
        );
    }

    for benign in [
        // A table: no lowercase words to make either side prose.
        "    \"NAME            WIDTH       PAGES\",",
        "    println!(\"{:>8}            {}\", a, b);",
        // A diagram.
        "    \"  +---+            +---+\",",
        // A short gap: two spaces after a period is not an indent.
        "    \"one sentence.  another sentence.\",",
        // One space under the threshold: still a gap, not an indent.
        "    \"a run one narrower than this     stays a gap\",",
        // Generated source: the run follows a newline the author put
        // there, so it is an indent and not a deletion.
        "    \"inline ulong salt(uint s) {{\\n  return splitmix64(\\n      ulong(s));\",",
        // Single-character cells: lowercase, but no word behind them.
        "    \"a       b       c\",",
        // Aligned output that DOES have words on both sides. The run is
        // followed by a column marker, which prose never is.
        "    \"avg_missing_at_fire            = {}   (dense all-ready fire)\",",
        "    \"[MIRO19] config              | final_mu  tail_S | distinct%\",",
    ] {
        assert!(
            collapsed_literals(benign).is_empty(),
            "flagged alignment as prose: {benign}"
        );
    }
}

/// The two predicates, over cases the workspace does not currently hold.
///
/// The tests above are CORPUS scans: they answer "does this build carry a
/// swallowed indent", which is the question worth failing on, but they
/// exercise only the shapes that happen to exist. A control undoing the
/// placeholder rule inside `has_word` was silent for exactly that reason --
/// no source here has a run whose LEFT side is a lone placeholder. The rule
/// is still load-bearing, so the cases are written down instead of waited
/// for.
///
/// Passed as two sides rather than as one literal on purpose: `splits_prose`
/// takes `before` and `after` already split, so nothing here has to contain
/// a run of spaces -- which matters, because this file is itself scanned by
/// the tests above.
#[test]
fn the_predicates_answer_the_shapes_the_corpus_does_not_hold() {
    // Flagged: prose on both sides, which is what a joined source line
    // leaves behind.
    for (before, after) in [
        (
            "the checkpoint declares one count and",
            "the row declares another",
        ),
        ("a tensor was missing", "(and the loader said nothing)"),
        ("the shard did not divide", "`local_extent` returned zero"),
    ] {
        assert!(
            splits_prose(before, after),
            "not flagged as a split sentence: {before:?} / {after:?}"
        );
    }

    // Not flagged: a column of aligned output. Every one of these is real
    // or near-real diagnostic printing.
    for (before, after) in [
        // The right side is a cell, not a continuation.
        ("storage", "{} buffers per stage"),
        ("limits", "{limits}"),
        ("kind", "{:?}"),
        // The left side is a cell. This is the case no source here has,
        // and the one whose control was silent.
        ("{count}", "items were dropped"),
        ("{:>8}", "rows carried"),
        // A column marker rather than prose.
        ("avg_missing_at_fire", "= {}"),
        ("config", "| final_mu"),
        // An indent the author wrote, told apart by the newline in front.
        ("a generated header\\n", "still indented"),
    ] {
        assert!(
            !splits_prose(before, after),
            "flagged as a split sentence: {before:?} / {after:?}"
        );
    }

    // An escaped brace prints as a brace and really can open a phrase, so
    // it is not treated as a placeholder.
    assert!(splits_prose(
        "the type resolved to",
        "{{unknown}} at that point"
    ));

    // `has_word` on its own: a placeholder's identifier is not prose,
    // because the reader never sees it.
    assert!(has_word("the row declares"));
    assert!(!has_word("{limits}"));
    assert!(!has_word("{:?}"));
    assert!(!has_word("{}"));
    assert!(has_word("{}{}rows"), "prose beside a placeholder is prose");
    assert!(
        !has_word("ab cd"),
        "words under three letters are not prose"
    );
    assert!(!has_word("NAME WIDTH"), "a heading is not prose");
}
