//! EVERY FACT A FAMILY DECLARES, READ BY THAT FAMILY'S TEXT.
//!
//! A fact is a promise that something about a deployment changes what the
//! model does. A fact nobody reads is a promise nobody keeps, and it
//! fails in the worst available way: the declaration looks complete, the
//! plan traces, and the answer is wrong on the axis the fact was supposed
//! to control.
//!
//! gemma-2 is why this file exists. It carried `tied_embeddings: true`,
//! correctly — the checkpoint ships no `lm_head.weight` — and its text
//! named `"lm_head"` unconditionally, because the
//! `if tied { "embed" } else { "lm_head" }` fork was written in four
//! other family texts and not in that one. The trace asked the binder for
//! a tensor that does not exist. Nothing caught it: the family is in
//! `NOT_YET_OPENABLE`, so no fire reaches it, and a golden pins whatever
//! the text says rather than what it should say.
//!
//! So this is the same closed-set discipline `UNARMED` and
//! `NOT_YET_OPENABLE` use, applied to facts: the unread set is STATED,
//! and a fact that joins it fails here rather than on a checkpoint.
//!
//! The scan is source-level, like `executor_bind`'s arm scan, and for the
//! same reason — the question is about what the code SAYS, and there is
//! no runtime handle on "did this field influence the trace".
//!
//! # Where a declaration lives
//!
//! It used to be `<family>/forward/facts.rs` and only that. The catalog
//! refactor moved the SHAPE up to `<family>/spec.rs` — ungated, because a
//! `const` catalog row is written in those words and a row must exist
//! under every aspect — and left `forward/facts.rs` behind as a re-export
//! plus whatever per-backend facts are genuinely about the tracer.
//!
//! So the scan reads BOTH, and it has to. A re-export file declares no
//! `pub struct`, so a scan that still looked only there would find an
//! empty declared set for every migrated family, agree with an empty
//! stated list, and pass — which is the same failure as the one this file
//! exists to catch, wearing the test's own clothes. A test that quietly
//! stops testing is worse than no test, because the list at the top keeps
//! claiming the set is known.

#![cfg(feature = "forward")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Facts declared and not read, by family, with why.
///
/// Most lines here belong to a family whose TEXT IS UNFINISHED — the
/// fact was read off a real config and is what the pass will need when
/// someone writes it. Deleting the field would throw away the reading.
/// kimi-k2's `rope_yarn_original` shows that shape: its text has no rope
/// statement at all — `mla_prepare` hands back `q_pe` and `k_pe` and
/// nothing rotates either, with `k_pe` dropped into `_`. The fact is not
/// forgotten, the pass is unwritten.
///
/// `gemma_2` was the OTHER kind, and it is the reason this list is a
/// test rather than a note: it traced a complete plan and silently
/// ignored one axis, which is exactly the defect the file is named for.
/// `gemma_4` was the second of that kind and left when its Metal
/// projection started stating the top-k its text routes on.
///
/// Both are gone now. `gemma_2: attn_logit_softcap` left last, and it
/// took four moves rather than one, which is what that kind costs: the
/// fact was a real measurement, its doc correctly said the attention
/// kernel takes the cap as a dispatch parameter, the kernel really does
/// take it — and `AttnCtx::logits_soft_cap` was the literal `0.0`. So
/// `Deployment` gained `attn_logit_softcap`, gemma-2's projection states
/// `50.0` beside the readout's `30.0`, the launch reads it, and
/// `launch_context_is_stated` widened from `Source::Ctx` to `Source::
/// Attn` because that is the scan the defect had been hiding behind.
///
/// A line LEAVES when the text starts reading the fact. A line JOINS only
/// with a commit that says which unfinished pass it belongs to.
///
/// # Two lines left this list while the catalog landed
///
/// `nemotron_h: tied_embeddings` and `deepseek_v4: sliding_window` were
/// both here, and both are now read — by `project.rs` files that did not
/// exist when they were written. That is the list working: the entries
/// are transient by construction, and the test fails just as loudly when
/// a fact starts being read as when one stops.
const DECLARED_BUT_UNREAD: &[(&str, &[&str])] = &[
    // The Mimi codec's base channel count. csm's `deployment()` and
    // `trace()` both refuse — its decoder is transcribed and its CODEC is
    // not — so the convolutional stack this width sizes has no statement
    // to be read by. It is here rather than deleted because it is a
    // measurement of `eustlb/csm-1b`'s codec, and the pass that needs it
    // is unwritten rather than the fact being wrong.
    ("csm", &["filters"]),
    // Hash routing, the output projection's grouping and the routed
    // MLP's clamp. `sliding_window` LEFT this list when `project.rs`
    // arrived: the deployment's per-layer `window_left` is stated from
    // it, so a V4 with a window and one without now differ before any
    // fire — which is the shape every line here is meant to leave by.
    (
        "deepseek_v4",
        &["hash_routed", "o_groups", "swiglu_limit_milli"],
    ),
    // The routed MLP's top-k. `gemma_4/forward/mod.rs` contains no
    // routing pass at all — no gate, no top-k, no expert dispatch — so
    // for as long as CUDA was the only text, the A4B's mixture was
    // declared and untraced. `moe_intermediate` LEFT this list because
    // the projection carries it to a planner for sizing, and
    // `experts_per_token` left the same way: `gemma_4::project::
    // metal_shape` states it on the `LlamaLikeFacts` that
    // `llama_like_metal` routes on, so the Metal text dispatches experts
    // by this number. The CUDA text still does not, and `Gemma4::
    // untraced` refuses the A4B row on both backends until it does —
    // which is a refusal a reader can find, and not a silent axis.
    // MLA's aligned MoE block. `index_topk` LEFT this list when the
    // indexer's statement started carrying it on the param channel —
    // which is the shape every line here is meant to leave by.
    ("glm_5", &["aligned_block"]),
    // The MXFP4 decode leg's route ceiling.
    ("gpt_oss", &["mxfp4_decode_max_routes"]),
    // No rope statement in the MLA text yet; see the header above.
    ("kimi_k2", &["rope_yarn_original"]),
    // KDA's gate floor.
    ("kimi_k3", &["gate_lower_bound_milli"]),
];

fn crate_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every family that declares a shape, and where its sources live.
///
/// A family qualifies on `spec.rs` OR `forward/facts.rs`: the first is
/// where the catalog refactor put the shape, the second is where the
/// per-backend facts stayed, and a family may have either or both.
fn families() -> BTreeMap<String, PathBuf> {
    let mut out = BTreeMap::new();
    // Both halves: a generation directory at the root, and a shared
    // family under `shared/`. The second was `src/families/`, and a
    // scan that skips a missing root in silence would have gone on
    // passing over half a crate after the rename — so neither is
    // optional and neither is skipped.
    for root in [crate_src(), crate_src().join("shared")] {
        let entries = std::fs::read_dir(&root)
            .unwrap_or_else(|e| panic!("{} is readable: {e}", root.display()));
        for e in entries.flatten() {
            let dir = e.path();
            if dir.join("spec.rs").is_file() || dir.join("forward/facts.rs").is_file() {
                let name = dir.file_name().unwrap().to_string_lossy().into_owned();
                out.insert(name, dir);
            }
        }
    }
    // Named rather than counted. A count would not have caught the
    // rename this comment is about: exactly one family lives under
    // `shared/`, so losing that root drops the total by one and any
    // threshold loose enough to survive a new generation is loose enough
    // to miss it. `llama_like` is the family a dozen generations bind,
    // which makes it the half worth being sure of.
    assert!(
        out.contains_key("llama_like"),
        "the shared half of the crate contributed nothing; found {:?}",
        out.keys().collect::<Vec<_>>()
    );
    assert!(out.len() >= 10, "found only {} families", out.len());
    out
}

/// The files a family's shape is declared in, in the order they are read.
///
/// Both, unioned. Splitting the shape across two files is fine; losing
/// half of it to a scan that knows about one is not.
fn declaring_files(dir: &Path) -> Vec<PathBuf> {
    [dir.join("spec.rs"), dir.join("forward/facts.rs")]
        .into_iter()
        .filter(|p| p.is_file())
        .collect()
}

/// Every `pub` field of every `pub struct` in one facts file.
fn declared_fields(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut in_struct = false;
    for line in src.lines() {
        let t = line.trim();
        if t.starts_with("pub struct ") && t.ends_with('{') {
            in_struct = true;
            continue;
        }
        if in_struct && t == "}" {
            in_struct = false;
            continue;
        }
        if !in_struct {
            continue;
        }
        // `pub name: Type,` — the field, not a doc comment or attribute.
        if let Some(rest) = t.strip_prefix("pub ")
            && let Some((name, _)) = rest.split_once(':')
            && name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit())
            && !name.is_empty()
        {
            out.insert(name.to_string());
        }
    }
    out
}

/// Every `.field` mention across a family's own sources, EXCLUDING its
/// tests and its prose.
///
/// Both exclusions are load-bearing, and the catalog refactor is what
/// proved it. `gpt_oss`'s `mxfp4_decode_max_routes`, `glm_5`'s
/// `aligned_block` and `kimi_k2`'s `rope_yarn_original` all dropped out
/// of the unread set when `project.rs` and `spec.rs` arrived — not
/// because any tracer started reading them, but because a `#[cfg(test)]`
/// assertion named them. `assert!(m.aligned_block > 0)` keeps no promise
/// about what a fire does; it is a test of the row, and counting it is
/// how a fact goes quiet.
///
/// Prose is excluded for the same reason one step further out: a doc
/// comment that says what a field is FOR is the strongest possible signal
/// that nothing yet does it.
fn read_fields(dir: &Path) -> BTreeSet<String> {
    let mut text = String::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                text.push_str(&without_tests_or_prose(
                    &std::fs::read_to_string(&p).unwrap_or_default(),
                ));
                text.push('\n');
            }
        }
    }
    let mut out = BTreeSet::new();
    let bytes: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == '.' && i + 1 < bytes.len() && (bytes[i + 1].is_ascii_lowercase()) {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == '_') {
                j += 1;
            }
            out.insert(bytes[start..j].iter().collect::<String>());
            i = j;
            continue;
        }
        i += 1;
    }
    out
}

/// One file with every `//`-comment line and every `#[cfg(test)]` item
/// removed.
///
/// The test item is cut by brace matching from the first `{` after the
/// attribute. A `#[cfg(test)] use …;` is braceless, and is detected by a
/// `;` arriving before any `{` — without that check the brace search runs
/// on to the NEXT item's body and deletes a real tracer with it.
fn without_tests_or_prose(src: &str) -> String {
    let code: String = src
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    const ATTR: &str = "#[cfg(test)]";
    let mut out = String::new();
    let mut rest = code.as_str();
    while let Some(at) = rest.find(ATTR) {
        out.push_str(&rest[..at]);
        let after = &rest[at + ATTR.len()..];
        let open = after.find('{');
        let semi = after.find(';');
        let Some(open) = open.filter(|o| semi.is_none_or(|s| s > *o)) else {
            // Braceless: `#[cfg(test)] use …;`. Drop the attribute, keep
            // everything else, and carry on from just past it.
            out.push_str(ATTR);
            rest = after;
            continue;
        };
        let mut depth = 0i32;
        let mut end = None;
        for (i, c) in after[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(open + i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        match end {
            Some(e) => rest = &after[e..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

#[test]
fn every_declared_fact_is_read_by_its_family() {
    let stated: BTreeMap<&str, BTreeSet<&str>> = DECLARED_BUT_UNREAD
        .iter()
        .map(|(fam, fields)| (*fam, fields.iter().copied().collect()))
        .collect();

    let families = families();
    assert!(
        families.len() > 8,
        "the family scan found {} families, so its shape assumption broke",
        families.len()
    );

    let mut found: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (fam, dir) in &families {
        let files = declaring_files(dir);
        assert!(
            !files.is_empty(),
            "{fam} qualified on a file that then vanished"
        );
        let mut declared = BTreeSet::new();
        for f in files {
            let src = std::fs::read_to_string(&f).expect("the family's declaration");
            declared.extend(declared_fields(&src));
        }
        let unread: BTreeSet<String> = declared.difference(&read_fields(dir)).cloned().collect();
        if !unread.is_empty() {
            found.insert(fam.clone(), unread);
        }
    }

    // A family whose stated list no longer matches is the interesting
    // case in both directions: a fact that LEFT means the text started
    // reading it, and a fact that JOINED is a promise nobody keeps.
    let found_view: BTreeMap<&str, BTreeSet<&str>> = found
        .iter()
        .map(|(k, v)| (k.as_str(), v.iter().map(String::as_str).collect()))
        .collect();
    assert_eq!(
        found_view, stated,
        "the declared-but-unread set moved.\n\
         A fact that LEFT means its text now reads it — delete the line.\n\
         A fact that JOINED is declared and never read, which is how \
         gemma-2 came to ask its binder for a `lm_head.weight` no gemma-2 \
         checkpoint ships. Either read it or say which unfinished pass it \
         belongs to."
    );
}

/// The scan must actually FIND declarations, in every family it counts.
///
/// This is the guard the file lacked when the shape moved to `spec.rs`.
/// A re-export declares no `pub struct`, so a scan pointed at
/// `forward/facts.rs` alone found an empty declared set for every
/// migrated family, subtracted it from a large read set, got an empty
/// unread set, agreed with an empty stated list, and PASSED — with the
/// list at the top still claiming the unread set was known.
///
/// A test that quietly stops testing is worse than no test. So the shape
/// of the scan's own input is asserted, not assumed.
#[test]
fn the_scan_finds_a_shape_in_every_family_it_counts() {
    for (fam, dir) in families() {
        let files = declaring_files(&dir);
        let mut declared = BTreeSet::new();
        for f in &files {
            declared.extend(declared_fields(&std::fs::read_to_string(f).unwrap()));
        }
        assert!(
            declared.len() >= 5,
            "{fam} declares only {} field(s) across {:?}. Either the family \
             genuinely has no shape — in which case it should not be in the \
             scan — or the shape moved to a file this scan does not read, \
             which is how this test disarmed itself once already.",
            declared.len(),
            files
                .iter()
                .map(|p| p.file_name().unwrap())
                .collect::<Vec<_>>()
        );
    }
}

/// A test assertion is not a reader, and neither is a doc comment.
///
/// Pinned directly because it is the whole difference between the stated
/// list and an empty one: three facts left the list on a `#[cfg(test)]`
/// mention alone when `project.rs` and `spec.rs` arrived.
#[test]
fn a_test_assertion_does_not_count_as_reading_a_fact() {
    let src = "\
fn trace(f: &F) -> u32 { f.traced }
#[cfg(test)]
mod tests {
    #[test]
    fn t() {
        let m = f();
        assert!(m.only_asserted > 0);
        if true { let _ = m.nested_in_a_block; }
    }
}
/// Prose naming `.only_documented`, which reads nothing.
fn tail(f: &F) -> u32 { f.after_the_test }
";
    let kept = without_tests_or_prose(src);
    assert!(kept.contains("f.traced"), "the tracer's read survives");
    assert!(
        kept.contains("f.after_the_test"),
        "code AFTER the test block survives"
    );
    assert!(
        !kept.contains("only_asserted"),
        "an assertion inside #[cfg(test)] is not a read"
    );
    assert!(
        !kept.contains("nested_in_a_block"),
        "brace matching must span the test's inner blocks, or the cut ends early \
         and everything after it counts again"
    );
    assert!(
        !kept.contains("only_documented"),
        "a doc comment that says what a field is FOR is evidence nothing does it"
    );
}

/// A `#[cfg(test)]` with no braces must not eat the rest of the file.
///
/// `#[cfg(test)] use …;` is common and reads nothing. Without the
/// semicolon check the brace search runs past it to the next unrelated
/// body and deletes a real tracer, which would put a read fact into the
/// unread list.
#[test]
fn a_braceless_cfg_test_attribute_cuts_nothing_after_it() {
    let src = "#[cfg(test)]\nuse std::x;\nfn trace(f: &F) -> u32 { f.traced }\n";
    let kept = without_tests_or_prose(src);
    assert!(
        kept.contains("f.traced"),
        "the item after a braceless `#[cfg(test)] use …;` must survive; got {kept:?}"
    );

    // And the two shapes in one file, in the order they really appear.
    let mixed = "#[cfg(test)]\nuse std::y;\n#[cfg(test)]\nmod t { fn a(m: &M) { m.asserted; } }\nfn f(x: &X) { x.real; }\n";
    let kept = without_tests_or_prose(mixed);
    assert!(kept.contains("x.real"), "the tracer after both survives");
    assert!(!kept.contains("m.asserted"), "the braced test is still cut");
}

/// Both declaring files are read, not just one.
#[test]
fn a_shape_split_across_spec_and_facts_is_read_whole() {
    let both: Vec<String> = families()
        .into_iter()
        .filter(|(_, d)| d.join("spec.rs").is_file() && d.join("forward/facts.rs").is_file())
        .map(|(f, _)| f)
        .collect();
    assert!(
        !both.is_empty(),
        "no family has both files, so the union in `declaring_files` is \
         untested and the next migration can drop half a shape unnoticed"
    );
    for fam in &both {
        let dir = families().remove(fam).unwrap();
        let spec = declared_fields(&std::fs::read_to_string(dir.join("spec.rs")).unwrap());
        let facts =
            declared_fields(&std::fs::read_to_string(dir.join("forward/facts.rs")).unwrap());
        let union = declaring_files(&dir)
            .iter()
            .flat_map(|f| declared_fields(&std::fs::read_to_string(f).unwrap()))
            .collect::<BTreeSet<_>>();
        assert!(
            union.len() >= spec.len().max(facts.len()),
            "{fam}: the union lost a field one of its two files declares"
        );
    }
}
