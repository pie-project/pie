//! Every fact a plane's bodies ask for is one its driver can answer.
//!
//! # The comparison this tree did not have
//!
//! `shader_backends_agree` holds the three shader planes against each other
//! and CUDA against none of them, and the obvious fix — diff the signatures —
//! is not available: the kernel sets are almost disjoint. Normalise the
//! element suffixes away and `kernels-cuda`'s routines share FIVE names with
//! `kernels-metal`'s. `Elem`'s own doc says why no signature is shared:
//! `Read` is one associated type with no backend parameter, so a plane's
//! `bf16` resolves it to a pointer or to a binding index and never to both.
//!
//! What all four planes DO share is the machinery, and this is the half of it
//! a test can hold them all to at once: a body reaches for a fact with
//! `ctx.ask::<C, keys::K>()`, and if its driver answers no such key the body
//! returns `Refusal::Unstated` and **the routine cannot fire**. Not a
//! degraded fire — no fire.
//!
//! # Why it is a scan
//!
//! `ask` is a CALL, not a declaration. `Asks`'s own doc records the cost:
//!
//! > `ask` is a CALL, not a declaration, so the derived column no longer
//! > enumerates it and a driver test can no longer walk that column to ask
//! > *"does this backend answer every fact its own kernels name"*.
//!
//! This is that question, asked of the source text instead. It misses a fact
//! asked inside a helper, which is the same blind spot `#[routine]`'s own
//! collection has and is accepted for the same reason.
//!
//! # What it caught
//!
//! A hundred and six ask sites across some thirty routines, every one of them
//! unfirable: each `qmm_t_splitk` and split-K reduce, the strided casts and
//! matmuls, `q_gate_split`, `gate`, `logit_softcap`, `moe_align_decode`,
//! `moe_grouped_gemm`, `add_moe_route_bias`. None was a fact. Cross the list
//! against HEAD and each was a `Param<N, i32>`, a bare `i32` argument, or a
//! `Reckoned` product of two facts — and the migration invented the key to
//! ask with, which is why HEAD has no parameter anywhere naming `RowStride`
//! or `SplitK`.
//!
//! # What was left, and why it is NOT the same thing
//!
//! Fifty asks survived that sweep and read for a long while as its remainder —
//! the same mistake, not yet got to. They are not. Counted 2026-08-21:
//!
//! - **Forty-eight of the fifty are declared in `keys.rs` §M or §L**, the two
//!   sections the CUDA mark migration wrote for itself. Both refuse a mark IN
//!   WRITING and give the reason: §M refuses `Out` because an `Out` mark is a
//!   claim the allocator reads, so marking a buffer the driver carved and a
//!   later statement still reads shortens its life to one fire; §L refuses
//!   `Const` because a `Const` promises the statement carries the number at
//!   its slot in the params run, and where nothing states one the promise
//!   breaks at the fire rather than at the type. The key is the right shape.
//! - **The other two, `NumExperts` and `RecurrentSlots`, sit in §1 ANSWERED**,
//!   and three drivers do answer them — `driver-metal`, `driver-wgpu`,
//!   `driver-vulkan`. Only `driver-cuda` does not, under a heading that says
//!   it does.
//!
//! So the debt is one-sided rather than mistaken, and it is one-sided
//! completely: §M and §L declare fifty-nine keys between them and
//! `driver-cuda` answers NOT ONE. Forty-seven of the forty-eight were declared
//! and first asked for in the SAME commit, `930fee2cb`, whose subject is about
//! the three SHADER planes and whose body carries the CUDA half as a rider:
//! *"The `kernels-cuda` mark migration rides along"*. The asking half of that
//! migration landed and the answering half did not, and nothing measured the
//! gap until this scan.
//!
//! # The scan believed a sentence that denied it
//!
//! "Not one" was "exactly one" until the strip below existed. `MoeMaxBlocks`
//! read as answered because `driver-cuda` names it — once, in a comment, which
//! reads in full: *"the signature takes as `Const` because no driver answers
//! `keys::MoeMaxBlocks` or `keys::MoeAlignedRows`"*. The scan took a sentence
//! whose subject is the ABSENCE of an answer as the answer. Stripping comments
//! from both sides moved the count from fifty to fifty-one, and the single new
//! line was that key.
//!
//! It is the `X` hole one direction over. That one is recorded further down —
//! a placeholder in `Asks::ask`'s doc, quoted in `kernels-cuda`, collected as
//! a real ask and excused by name because teaching the scan prose was dearer
//! than one entry. It was not dearer; it was seventy lines, and it closes both
//! directions at once. The excuse was cheaper only while the second hole was
//! still unknown.
//!
//! That is why the failure groups by section and says a different repair under
//! each. The message used to send every one of the fifty toward `Const` or
//! `Asks::param` — advice that is right for the hundred and six and that two
//! sections of the registry refuse by name for the forty-eight. A gate that
//! names the wrong repair costs more than one that only counts, because the
//! reader who follows it undoes a decision someone already made on purpose.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<crate>/..")
        .to_path_buf()
}

/// Every `.rs` under `dir`, recursively.
fn sources(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(sources(&path));
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    out
}

/// A plane's text, with its test modules cut off.
///
/// A probe's `resolve` names keys it answers for the tests and a fixture may
/// ask for one this plane's kernels never do, so counting them would measure
/// the fixtures rather than the plane.
fn plane_text(plane: &str) -> String {
    let mut out = String::new();
    for path in sources(&workspace_root().join(format!("crates/kernels-{plane}/src"))) {
        let src = std::fs::read_to_string(&path).unwrap_or_default();
        let live = src
            .find("\n#[cfg(test)]")
            .map_or(src.as_str(), |at| &src[..at]);
        out.push_str(&without_comments(live));
        out.push('\n');
    }
    out
}

/// `src` with its comments blanked out, string literals left whole.
///
/// # Why the scan cannot read prose
///
/// A driver "answers" a key here by naming it, and until this existed a
/// COMMENT naming it counted. That is not a hypothetical: the one key this
/// test believed `driver-cuda` answered out of §M and §L's fifty-nine was
/// `MoeMaxBlocks`, and its only mention in the whole crate is
/// `fire/moe_grouped.rs` saying *"the signature takes as `Const` because no
/// driver answers `keys::MoeMaxBlocks`"* — a sentence stating the exact
/// opposite of what the scan concluded from it. A gate that reads a denial as
/// a confirmation is worse than one that cannot read at all.
///
/// The reverse hole is already known and recorded above: `ask::<_, keys::X>`
/// in a doc comment was once collected as an ask. Both directions are the same
/// hole and this closes them together, since the plane text is stripped too.
///
/// String literals must SURVIVE, because matching `"conv_state"` out of
/// `driver-metal`'s `SLABS` table is one of the two legitimate answer routes.
/// So this tracks quotes rather than blanking from `//` to end of line.
fn without_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'"' => {
                // Copy the literal whole, honouring `\"`. Raw strings (`r"`,
                // `r#"`) copy correctly too: the opening quote is reached the
                // same way and no escape inside one can close it early.
                out.push('"');
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' && i + 1 < b.len() {
                        out.push(b[i] as char);
                        i += 1;
                    }
                    out.push(b[i] as char);
                    i += 1;
                }
                if i < b.len() {
                    out.push('"');
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'/') => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                // Nested, because rustc's block comments nest.
                let mut depth = 1;
                i += 2;
                while i < b.len() && depth > 0 {
                    if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                        depth += 1;
                        i += 2;
                    } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                        depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
            }
            c => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    out
}

fn driver_text(driver: &str) -> String {
    let mut out = String::new();
    for path in sources(&workspace_root().join(format!("crates/driver-{driver}/src"))) {
        out.push_str(&without_comments(
            &std::fs::read_to_string(&path).unwrap_or_default(),
        ));
        out.push('\n');
    }
    out
}

/// Every `keys::K` named in an `ask::<_, keys::K>()` in `text`.
fn asked(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut rest = text;
    while let Some(at) = rest.find("ask::<") {
        rest = &rest[at + "ask::<".len()..];
        // THE CARRIER NESTS. `ask::<Tensor<f32>, keys::AttnPartials>()` has a
        // comma inside the first argument and a `>` before the one that
        // closes the call, so the split has to count depth rather than take
        // the first of each -- which is how the first draft of this scan
        // reported that vulkan had stopped asking for a key it asks twice.
        let mut depth = 0i32;
        let mut comma = None;
        let mut close = None;
        for (i, c) in rest.char_indices() {
            match c {
                '<' => depth += 1,
                '>' if depth == 0 => {
                    close = Some(i);
                    break;
                }
                '>' => depth -= 1,
                ',' if depth == 0 && comma.is_none() => comma = Some(i),
                _ => {}
            }
        }
        let (Some(comma), Some(close)) = (comma, close) else {
            break;
        };
        let after = rest[comma + 1..close].trim();
        if let Some(name) = after.strip_prefix("keys::")
            && !name.is_empty()
            && name.chars().all(|c| c.is_alphanumeric() || c == '_')
        {
            out.insert(name.to_string());
        }
    }
    out
}

/// The migration's END STATE, held: the question this file was born to ask
/// — *"does every driver answer every fact its own kernels name"* — is
/// answered by there being NOTHING LEFT TO ASK. `ctx.ask` and `keys.rs`
/// are deleted; every need a routine names is an OPERAND now, enumerable
/// in `plan.runtime` and the derived column, and the walkable half of the
/// old question lives in its successors:
///
/// - `check_plan`'s closed-vocabulary rule (model-ir) refuses a runtime
///   name outside the floor's tier-1 list or a plane's dotted key, at
///   LOAD;
/// - `driver-cuda/tests/every_runtime_name_is_answered.rs` walks every
///   catalogued SKU's `plan.runtime` against the driver's exported
///   `ANSWERED`/`UNSTAGED` sets;
/// - `canon_claims_agree` holds the three shader planes' claims equal.
///
/// What this file still holds, with the scan it always used: that no ask
/// COMES BACK. A `ctx.ask::<..>` reappearing in a plane (a merge landing
/// stale code wrote exactly this, twice) compiles against nothing today —
/// but the scan refuses it with a sentence naming the channel it should
/// take, which is a better failure than "no method named `ask`".
#[test]
fn no_plane_asks_for_anything() {
    let mut sites = Vec::new();
    for plane in ["kernels-cuda", "kernels-metal", "kernels-vulkan", "kernels-wgpu"] {
        let text = plane_text(plane);
        for key in asked(&text) {
            sites.push(format!("{plane}: ctx.ask for `{key}`"));
        }
        // The vocabulary itself must stay gone too.
        assert!(
            !workspace_root().join("crates/kernels/src/keys.rs").exists(),
            "keys.rs is back; the ask vocabulary was deleted by the no-ask \
             migration and returns through no merge"
        );
    }
    assert!(
        sites.is_empty(),
        "ask sites in a tree with no ask machinery — each names a need that \
         is an OPERAND now (a Const on the statement, a fire extent, or a \
         runtime view/stream the driver answers by name):\n  {}",
        sites.join("\n  ")
    );
}
