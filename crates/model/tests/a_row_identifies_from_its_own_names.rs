//! WHAT NAME DOES AN ARTIFACT CARRY? THE CATALOG'S OWN, AND HERE IS THE PROOF.
//!
//! Three vocabularies name the same weights. A trace asks for `layer.7.qkv`
//! because that is what the forward text says. A HuggingFace checkpoint
//! publishes `model.layers.7.self_attn.q_proj.weight`. llama.cpp writes
//! `blk.7.attn_q.weight`. `shared/weight_names.rs` holds the map from the
//! third-party spellings to the trace, and states why it lives in this crate
//! and not in a driver.
//!
//! The vocabulary that was never stated is the one in the MIDDLE: the names a
//! `.zt` artifact carries between import and serve. It was never chosen. It
//! is whatever the source happened to spell, inherited by an import pass that
//! renames nothing -- which is why importing a GGUF today yields an artifact
//! full of `blk.0.attn_q.bias`, and why that artifact matches no catalog row
//! at all.
//!
//! It works for HuggingFace inputs by ACCIDENT, and the accident is worth
//! naming precisely, because the whole of GGUF ingest is about to depend on
//! it: the catalog's rows are spelled HuggingFace-shaped, so a checkpoint
//! that is already HuggingFace-shaped passes through and matches. An ingest
//! pass with a genuinely foreign vocabulary has no such luck. It must be told
//! what to write, and until this file nothing said.
//!
//! So: **the artifact vocabulary is the catalog's own manifest vocabulary**,
//! as normalized by `Observed::logical`. That is now a decision rather than a
//! coincidence, and the two properties below are what makes it usable.
//!
//! Neither is circular, and the difference from a unit test matters. A unit
//! test over `logical` would hold this crate's rules against this crate's
//! expectations -- both sides written here, which is the failure mode
//! `the_two_name_maps_agree.rs` records for the MLX map ("it was
//! self-consistent, and the test that held the text against it passed,
//! because both sides were this file"). What is held here is one half of the
//! crate against the OTHER half: the catalog's 58 rows against the
//! normalizer, through the real `identify_observed`. Either can move without
//! the other, and when it does this fails.
//!
//! ## The property an ingest pass depends on
//!
//! A pass authoring `blk.7.attn_q` has to emit SOME name. This file says that
//! name is the row's own -- and proves the claim is worth acting on, because
//! a row's declared names are exactly what identifies it. Without that, "emit
//! the catalog's spelling" would be advice with no guarantee behind it.
//!
//! ## Two spellings, because pie writes one and reads the other
//!
//! `pie model build` does not write the checkpoint's globals verbatim: the
//! embedding table becomes `shared_embedding` and the final norm becomes
//! `final_norm`, and `Observed::global_spelling` translates them back. Its
//! doc records what the absence of that translation cost -- `qwen3-0.6b:
//! missing embed_tokens; missing norm`, "the whole catalog refusing the
//! output of the tool that reads the catalog". The second test below is that
//! bug, held down for every row rather than the one that was noticed.

use model::catalog::{Override, Variant, are_declared_twins, catalog, identify_observed};
use model::manifest::{Observed, Presence};

/// A row's manifest as a checkpoint that satisfies it.
///
/// `Absent` specs are skipped because the spec IS their absence -- a tied
/// model's row says `lm_head` is not published, and publishing it would
/// describe a different model. `Optional` ones are kept: a row that declares
/// one is a row that identifies with it present.
fn as_published(row: &dyn Variant, respell_globals: bool) -> Observed {
    let manifest = row.manifest();
    let mut pairs: Vec<(String, Vec<u64>)> = Vec::new();
    for spec in &manifest.tensors {
        if spec.presence == Presence::Absent {
            continue;
        }
        // A row with no stack still has model-level tensors, so the loop
        // runs once either way and a name with no `{}` breaks out of it.
        for index in 0..manifest.layers.max(1) {
            let mut name = spec.name.replace("{}", &index.to_string());
            if respell_globals {
                name = match name.as_str() {
                    "embed_tokens" => "shared_embedding".to_string(),
                    "norm" => "final_norm".to_string(),
                    other => other.to_string(),
                };
            }
            pairs.push((name, spec.extents.clone()));
            if !spec.name.contains("{}") {
                break;
            }
        }
    }
    Observed::from_pairs(pairs)
}

/// The row that a manifest built from `row`'s own names identifies as.
///
/// A declared [`model::catalog::GEOMETRIC_TWINS`] set answers as itself:
/// llama-3.1-70B and llama-3.3-70B are the same geometry retrained, the
/// catalog says so in a `const`, and a checkpoint matching both is REFUSED
/// rather than guessed at. That refusal is the catalog working. Reading it as
/// a vocabulary failure would be reading the one honest ambiguity in the
/// table as a defect.
fn identifies_as(row: &dyn Variant, respell_globals: bool) -> Result<(), String> {
    match identify_observed(&as_published(row, respell_globals), &Override::None) {
        Ok(hit) if hit.id() == row.id() => Ok(()),
        Ok(hit) if are_declared_twins(&[row.id(), hit.id()]) => Ok(()),
        Ok(hit) => Err(format!("identified as `{}`", hit.id())),
        Err(why) => {
            let ambiguous = model::catalog::GEOMETRIC_TWINS
                .iter()
                .any(|set| set.contains(&row.id()) && set.iter().all(|id| why.to_string().contains(id)));
            if ambiguous {
                return Ok(());
            }
            Err(why.to_string().lines().next().unwrap_or("").to_string())
        }
    }
}

/// Every row is found by the names it itself declares.
///
/// The guarantee behind "an ingest pass writes the catalog's spelling". A
/// failure here is not a row being wrong and not the normalizer being wrong;
/// it is the two having drifted, and it names which row noticed first.
#[test]
fn a_checkpoint_built_from_a_row_is_identified_as_that_row() {
    let faults: Vec<String> = catalog()
        .iter()
        .filter_map(|row| identifies_as(*row, false).err().map(|why| format!("{}: {why}", row.id())))
        .collect();
    assert!(
        faults.is_empty(),
        "{} of {} rows do not answer to their own names:\n  {}",
        faults.len(),
        catalog().len(),
        faults.join("\n  ")
    );
}

/// And is still found after pie's own lowering respells the globals.
///
/// The round trip that matters in practice: `pie model build` reads the
/// catalog, writes an artifact, and the artifact must still be readable BY
/// the catalog. `global_spelling` carries exactly two rules, so this is
/// cheap to state -- and it was worth stating, because those two rules were
/// added only after every model failed to match itself.
#[test]
fn an_artifact_pie_lowered_is_still_identified() {
    let faults: Vec<String> = catalog()
        .iter()
        .filter_map(|row| identifies_as(*row, true).err().map(|why| format!("{}: {why}", row.id())))
        .collect();
    assert!(
        faults.is_empty(),
        "{} of {} rows stop matching once lowered:\n  {}",
        faults.len(),
        catalog().len(),
        faults.join("\n  ")
    );
}
