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
        let live = src.find("\n#[cfg(test)]").map_or(src.as_str(), |at| &src[..at]);
        out.push_str(live);
        out.push('\n');
    }
    out
}

fn driver_text(driver: &str) -> String {
    let mut out = String::new();
    for path in sources(&workspace_root().join(format!("crates/driver-{driver}/src"))) {
        out.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
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
        let (Some(comma), Some(close)) = (comma, close) else { break };
        let after = rest[comma + 1..close].trim();
        if let Some(name) = after.strip_prefix("keys::") {
            if name.chars().all(|c| c.is_alphanumeric() || c == '_') && !name.is_empty() {
                out.insert(name.to_string());
            }
        }
    }
    out
}

/// Each key's declared name and whether its source is a positional SLOT.
///
/// A `Slot` key is resolved by the binder out of the statement — an operand's
/// width, a scalar's position — and reaches no fact table at all, so a driver
/// that never names it is answering it all the same.
fn key_table() -> BTreeMap<String, (String, bool)> {
    let src = std::fs::read_to_string(workspace_root().join("crates/kernels/src/keys.rs"))
        .expect("the key registry");
    let mut out = BTreeMap::new();
    for chunk in src.split("fact!(").skip(1) {
        // `Name = "string" => SOURCE => Ty);`
        let Some(eq) = chunk.find(" = \"") else { continue };
        let name: String = chunk[..eq]
            .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
            .next()
            .unwrap_or_default()
            .to_string();
        if name.is_empty() {
            continue;
        }
        let after = &chunk[eq + " = \"".len()..];
        let Some(quote) = after.find('"') else { continue };
        let string = after[..quote].to_string();
        let tail = &after[quote..];
        let source = tail.split("=>").nth(1).unwrap_or_default();
        out.insert(name, (string, source.contains("Source::Slot")));
    }
    out
}

/// The keys `plane` asks for that `driver` names by neither route.
///
/// Two routes because the drivers use both: `driver-cuda` writes
/// `<keys::RmsEps as keys::Fact>::KEY` and `driver-metal` matches the string
/// `"conv_state"` out of a `SLABS` table.
fn unanswered(plane: &str, driver: &str) -> Vec<String> {
    let keys = key_table();
    let text = driver_text(driver);
    asked(&plane_text(plane))
        .into_iter()
        .filter(|name| {
            let Some((string, slot)) = keys.get(name) else {
                return false;
            };
            !slot && !text.contains(&format!("keys::{name}")) && !text.contains(&format!("\"{string}\""))
        })
        .collect()
}

/// `kernels-vulkan` is measured against `driver-wgpu`, which is not its own.
///
/// There is no `driver-vulkan` in this tree — `.wiki/driver-vulkan.md` is the
/// design and not a crate — so the two attention facts below are unanswered
/// because the answering half does not exist here, not because a body reaches
/// for something that was never a fact. They are `Ask<..>` at HEAD too.
const NO_DRIVER_OF_ITS_OWN: &[&str] = &["AttnPartials", "AttnSplits"];

/// What a driver genuinely does not answer, with the reason.
///
/// Each entry is a DEBT and its own sentence has to say why, because the cost
/// is the same either way: the body refuses `Unstated` and the routine does
/// not fire.
const UNANSWERED: &[(&str, &str, &str)] = &[
    (
        "cuda",
        "RequestOfToken",
        "`attention_compressed_paged_bf16`'s, and `Env<keys::RequestOfToken>` \
         at HEAD as well -- `driver-cuda` has never answered it, so the \
         compressed-KV prefill this deepseek path states cannot fire on this \
         driver and could not before the marks either.",
    ),
];

// THE `X` EXCUSE IS GONE, AND IT WAS NEVER A DEBT.
//
// `X` is the placeholder identifier in `Asks::ask`'s own doc comment, and
// `kernels-cuda/src/lib.rs` quoted that line. A scan of source text cannot
// tell a doc example from a call, so the placeholder was excused by name --
// cheaper than teaching the scan prose. The quotation went with the CUDA
// comment sweep and the excuse became exactly what the gate below is for: an
// entry that reads as a known debt while the thing it excused has gone.
//
// It is deleted rather than kept, unlike the budget lines in
// `driver-cuda`'s family census: those record a permission a later file
// could inherit, and this records a parsing artefact that either recurs
// verbatim or does not. If some file quotes `ask::<_, keys::X>` again the
// entry comes back with it.

#[test]
fn every_fact_a_plane_asks_for_is_one_its_driver_answers() {
    // The pairing, and `vulkan` is the odd one: it has no driver here.
    const PAIRS: [(&str, &str); 4] = [
        ("cuda", "cuda"),
        ("metal", "metal"),
        ("vulkan", "wgpu"),
        ("wgpu", "wgpu"),
    ];

    let excused: BTreeSet<(&str, &str)> =
        UNANSWERED.iter().map(|(plane, key, _)| (*plane, *key)).collect();

    let mut asked_total = 0usize;
    let mut problems = Vec::new();
    for (plane, driver) in PAIRS {
        let missing = unanswered(plane, driver);
        asked_total += asked(&plane_text(plane)).len();
        for key in missing {
            if excused.contains(&(plane, key.as_str())) {
                continue;
            }
            if plane == "vulkan" && NO_DRIVER_OF_ITS_OWN.contains(&key.as_str()) {
                continue;
            }
            problems.push(format!(
                "  kernels-{plane} asks for `keys::{key}` and driver-{driver} names it nowhere"
            ));
        }
    }

    assert!(
        problems.is_empty(),
        "{} unanswerable ask(s). A body that reaches for a fact its driver \
         cannot answer returns `Refusal::Unstated`, so the routine does not \
         fire at all:\n{}\n\nBefore adding a key, check what the number was \
         BEFORE the marks. Every one of the hundred and six this test first \
         found was a `Param<N, i32>`, a bare `i32` argument or a `Reckoned` \
         product -- the statement's own geometry, which belongs on the \
         signature as a `Const` mark (or is read by index with \
         `Asks::param` where the params run is the shader's struct), not in a \
         fact table.",
        problems.len(),
        problems.join("\n"),
    );

    // A FLOOR, because a scan that stopped finding `ask::<` would pass. The
    // four planes ask for some 160 keys between them; half of that is a
    // margin wide enough for a family to retire and narrow enough to catch
    // the parse breaking.
    assert!(
        asked_total > 80,
        "only {asked_total} asks found across the four planes, so this scan \
         has stopped reading them rather than found them answered"
    );
}

/// Every excuse names a key some plane actually asks for.
///
/// A stale entry is worse than none: it reads as a known debt while the thing
/// it excused has gone, and the next real one to take that name is excused
/// silently.
#[test]
fn every_excuse_names_a_key_a_plane_still_asks_for() {
    let mut stale = Vec::new();
    for (plane, key, _) in UNANSWERED {
        if !asked(&plane_text(plane)).contains(*key) {
            stale.push(format!("  kernels-{plane} no longer asks for `keys::{key}`"));
        }
    }
    for key in NO_DRIVER_OF_ITS_OWN {
        if !asked(&plane_text("vulkan")).contains(*key) {
            stale.push(format!("  kernels-vulkan no longer asks for `keys::{key}`"));
        }
    }
    assert!(
        stale.is_empty(),
        "these excuses have outlived what they excused:\n{}",
        stale.join("\n"),
    );
}
