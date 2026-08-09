//! A ROW THAT SHIPS A TOWER SAYS SO, AND A ROW THAT SAYS SO SHIPS ONE.
//!
//! `Advertised` is the handful of facts a driver puts in its capabilities
//! that are properties of the MODEL rather than of the device. They are
//! not read by any fire, which is exactly why getting one wrong is quiet:
//! nothing refuses, nothing is slower, and a whole entry point is simply
//! never built.
//!
//! `media_encode` has already been wrong once, in the direction that
//! costs the most. It was hardwired `false` while gemma-4's vision and
//! audio towers were ported, bound, and matching HuggingFace's embeddings
//! to cosine — and the worker refuses to build an encode executor at all
//! when the flag is clear, so none of that was reachable through the
//! engine. Four GPU tests fired the encode entry point and passed,
//! because they call the entry directly. The flag sits on the seam
//! between the model and the worker, and a test on either side of a seam
//! is a test that cannot see it.
//!
//! # Why a walk of the catalog
//!
//! Every generation asserts its own `media_encode`, and every one of
//! those assertions passed while the flag was wrong, because a row that
//! says `false` and means `true` is self-consistent. The question that
//! catches it is not what a row says but whether what it says matches
//! what it SHIPS, and only the two together answer it.
//!
//! gemma-4 is the one row that derives the flag from its towers instead
//! of stating it. That is the shape every row should have, and this file
//! is what makes stating it by hand safe: a family that grows a tower and
//! forgets the flag fails here.
//!
//! Ungated, because a catalog row is a `const` and `deployment()` is
//! answered in every build. The first draft of this file carried
//! `#![cfg(feature = "forward")]`, a feature this crate deleted, and ran
//! nothing at all while reporting success.

use model::catalog::{self, Deployed};
use std::collections::BTreeMap;

/// Rows this build cannot deploy at all, and what it lacks.
///
/// None of these is a rank problem -- they refuse at one, two, four and
/// eight alike -- and none is a property of the row. Each names a store or
/// a text this build does not provision, which is why the list belongs
/// here as words rather than as a `filter_map` that drops whatever fails.
/// The first draft of this file used the filter, and a negative control
/// that broke DeepSeek's advertised facts stayed green because DeepSeek
/// was never in the walk.
const THIS_BUILD_CANNOT_DEPLOY: &[(&str, &str)] = &[
    (
        "gemma-4-26b-a4b",
        // The text IS written; what is missing is the contract that
        // publishes `experts.switch_glu.*`, and the branch's own norms
        // and the router's two scales. `Gemma4::untraced` carries the
        // measurement -- this column used to say "no routed-expert
        // text", which named the wrong half.
        "no loadable routed bank for a gemma-4 block",
    ),
    ("glm-5-106b-a12b", "no MLA latent store"),
    ("kimi-k2", "no MLA latent store"),
    ("kimi-k3", "no MLA latent store"),
    ("deepseek-v4", "no compressed KV plane store"),
    (
        "csm-1b",
        "no speech decode loop, and three stacks to one Deployment",
    ),
];

/// Every row this build can deploy, at the first rank count it accepts.
///
/// Tried at one rank and then at two, four and eight, because a mixture
/// whose experts do not divide a single rank is a different thing from a
/// row this build has no store for, and only the second belongs in the
/// list above.
fn deployments() -> Vec<(&'static str, model::deployment::Deployment)> {
    let mut rows = Vec::new();
    let mut refused = Vec::new();
    for row in catalog::catalog() {
        let deployed = [1u32, 2, 4, 8].into_iter().find_map(|tp| {
            row.deployment(Deployed {
                tp_size: tp,
                ..Deployed::single()
            })
            .ok()
        });
        match deployed {
            Some(d) => rows.push((row.id(), d)),
            None => refused.push(row.id()),
        }
    }

    let expected: Vec<&str> = THIS_BUILD_CANNOT_DEPLOY.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        refused, expected,
        "the rows this build cannot deploy are not the ones \
         THIS_BUILD_CANNOT_DEPLOY lists. A row that started deploying is a \
         store this build grew, and one that stopped is a store it lost -- \
         either way the list says which, and nothing else in this file \
         holds these rows at all"
    );
    assert!(
        rows.len() >= 50,
        "only {} rows deployed, so this walk is too small to mean anything",
        rows.len()
    );
    rows
}

/// The flag and the towers are one fact stated twice.
///
/// # What this cannot catch
///
/// Only gemma-4 ships towers, and every gemma-4 row this build deploys
/// ships BOTH -- so hardwiring the flag `true`, or deriving it from the
/// vision tower alone and forgetting the audio one, are unfalsifiable
/// here. They become falsifiable the moment a row ships one tower and not
/// the other, which is the case this is really guarding.
#[test]
fn a_row_advertises_media_encode_exactly_when_it_ships_a_tower() {
    for (id, d) in deployments() {
        let ships = d.towers.vision.is_some() || d.towers.audio.is_some();
        assert_eq!(
            d.advertised.media_encode,
            ships,
            "{id} advertises media_encode={} but {} a tower. A row that \
             ships one and denies it has an encode entry point the worker \
             refuses to build, and a row that claims one and ships none \
             has an entry point with nothing behind it",
            d.advertised.media_encode,
            if ships { "ships" } else { "ships no" }
        );
    }
}

/// A tower a driver cannot size is a tower it cannot run.
///
/// Every field here is a divisor, a loop bound or an allocation, and a
/// zero in any of them is not a smaller tower — it is a division by zero,
/// an empty stack, or a buffer with no room. They are stated by rows
/// rather than derived, so the failure is a typo rather than a bug, and a
/// typo is what a walk like this is for.
#[test]
fn every_tower_a_row_ships_is_shaped_well_enough_to_run() {
    let mut towers = 0;
    for (id, d) in deployments() {
        if let Some(v) = d.towers.vision {
            towers += 1;
            for (what, n) in [
                ("layers", v.layers),
                ("hidden", v.hidden),
                ("heads", v.heads),
                ("intermediate", v.intermediate),
                ("pooling_kernel", v.pooling_kernel),
            ] {
                assert!(n > 0, "{id}'s vision tower states {what} as zero");
            }
            assert_eq!(
                v.hidden % v.heads,
                0,
                "{id}'s vision tower has {} heads over a width of {}, which \
                 do not divide -- a head is a slice of the residual stream \
                 and this one asks for a fractional slice",
                v.heads,
                v.hidden
            );
            assert!(
                v.norm_eps > 0.0 && v.rope_theta > 0.0,
                "{id}'s vision tower states a non-positive epsilon or rope \
                 base, which normalises to infinity or rotates by nothing"
            );
        }
        if let Some(a) = d.towers.audio {
            towers += 1;
            assert!(
                a.norm_eps > 0.0,
                "{id}'s audio tower states a non-positive epsilon"
            );
            assert!(
                a.residual_weight > 0.0,
                "{id}'s audio tower weights its residual at {}, so the \
                 conformer's half-step sum drops the branch it is summing",
                a.residual_weight
            );
        }
    }
    assert!(
        towers > 0,
        "no row in the catalog ships a tower, so every assertion in this \
         test held vacuously -- gemma-4 ships two"
    );
}

/// The two facts a guest program reads.
///
/// `arch` is what a guest matches on, so an empty one matches nothing and
/// is a row no program can select. `max_model_len` is allowed to be zero,
/// which states that the row publishes no ceiling — but a non-zero one
/// that is smaller than a single page is a ceiling no fire can respect.
#[test]
fn every_row_advertises_something_a_guest_can_match_on() {
    for (id, d) in deployments() {
        assert!(
            !d.advertised.arch.is_empty(),
            "{id} advertises an empty arch label, which no guest program \
             can match and no capability report can name"
        );
        assert!(
            d.advertised.max_model_len == 0 || d.advertised.max_model_len >= 1024,
            "{id} advertises a context ceiling of {}, which is not zero -- \
             the way a row says it publishes none -- and is shorter than a \
             single page",
            d.advertised.max_model_len
        );
    }
}

/// Two rows of one generation, two DIFFERENT vision encoders.
///
/// gemma-4's E-series ships 16 blocks of 768 and the A4B ships 27 of
/// 1152. That is why [`model::deployment::Deployment::towers`] is a
/// ROW's answer and not the generation's: a single tower hoisted to the
/// generation would give one of these two rows the other's encoder, and
/// an encoder of the wrong depth and width does not fail — it produces
/// embeddings the decoder happily attends to.
///
/// Asserted by VALUE rather than field by field, because the failure
/// this guards against is a shared constant, and a shared constant is
/// equal in every field at once.
#[test]
fn the_two_vision_towers_this_build_ships_are_not_the_same_tower() {
    let mut seen: Vec<(String, model::deployment::VisionTower)> = Vec::new();
    for (id, d) in deployments() {
        if let Some(v) = d.towers.vision {
            seen.push((id.to_string(), v));
        }
    }
    assert!(
        seen.len() >= 2,
        "fewer than two rows ship a vision tower, so this test compares \
         nothing -- gemma-4 ships two that differ"
    );
    assert!(
        seen.iter().any(|(_, a)| seen.iter().any(|(_, b)| a != b)),
        "every vision tower in the catalog is the same value, so the \
         per-row tower is a generation-wide constant wearing a row's \
         name: {seen:#?}"
    );

    // The rotary base is ONE HUNDRED, not the decoder's 10 000 or
    // 1 000 000. A tower that inherited the decoder's base would rotate a
    // 280-patch grid as if it were a 131 072-token context. The number is
    // small enough to look like a typo, so it is stated here as well as
    // in the row.
    for (id, v) in &seen {
        assert_eq!(
            v.rope_theta, 100.0,
            "{id}'s vision tower rotates at a decoder's base"
        );
    }
}

/// The audio tower's two subsampling widths are two numbers.
///
/// `subsampling_conv_channels: [128, 32]` -- the second convolution is
/// NARROWER than the first, which is why they are a pair and not a width
/// and a multiplier. A driver that derived the second from the first
/// would allocate four times the activation it needs and stride it
/// wrong.
#[test]
fn the_audio_towers_second_subsample_is_narrower_than_its_first() {
    let mut seen = 0;
    for (id, d) in deployments() {
        let Some(a) = d.towers.audio else { continue };
        seen += 1;
        assert!(
            a.subsample_channels_1 < a.subsample_channels_0,
            "{id}'s audio tower widens where it should narrow: {a:?}"
        );
        assert!(
            a.chunk_size > 0 && a.output_dims > 0 && a.feature_size > 0,
            "{id}'s audio tower states a zero where a stride or an \
             allocation goes: {a:?}"
        );
    }
    assert!(
        seen > 0,
        "no row ships an audio tower; gemma-4's E-series does"
    );
}

/// Family labels under which the rows do NOT agree, each with the reason
/// that is safe.
///
/// A label a guest sees is coarser than a row -- `qwen3` names twelve
/// checkpoints of six shapes -- so rows sharing one WILL differ. The
/// question the two tests below ask is whether anything reads the label
/// where it should have read the row, and the answer is a judgement per
/// label, not a rule. So the judgements are written down.
const LABELS_WHOSE_ROWS_DIFFER: &[(&str, &str)] = &[
    (
        "gemma3",
        "the 1b is text-only and the 4b ships a vision tower, so they \
         differ on `media_encode` and on `max_model_len`. Safe because \
         `media_encode` is read off the ROW: the worker builds an encode \
         executor for the 4b and not for the 1b, whatever label either \
         answers to.",
    ),
    (
        "gemma4",
        "the E-series and the A4B ship different vision encoders -- 16 \
         layers of 768 against 27 of 1152. Safe because \
         `VisionArch::from_arch_name` selects a PROCESSOR FAMILY (Gemma \
         against Qwen) and not a shape: the extents come from the row's \
         own tower, and the patch grid from `crate::multimodal`, which \
         computes it. One label over two encoders is not one front-end \
         over two shapes.",
    ),
];

/// Every label whose rows disagree is one that was thought about.
///
/// `arch` is the only thing a guest program has. `engine`'s
/// `model.arch_name()` is a host function inferlets call, and
/// `VisionArch::from_arch_name` picks an image front-end from the string
/// alone -- no id, no row. So a label covering rows that differ is a place
/// where a consumer COULD read the label and mean the row, and each one
/// has to be checked by hand once.
///
/// Compared as WHOLE VALUES rather than field by field. A field-by-field
/// version passes unchanged when a field is added, which is exactly when
/// this needs re-deciding.
#[test]
fn every_label_covering_rows_that_differ_is_one_that_was_thought_about() {
    let mut first_seen: BTreeMap<&'static str, (&'static str, model::deployment::Advertised)> =
        BTreeMap::new();
    let mut differ: BTreeMap<&'static str, String> = BTreeMap::new();
    for (id, d) in deployments() {
        match first_seen.get(d.advertised.arch) {
            None => {
                first_seen.insert(d.advertised.arch, (id, d.advertised.clone()));
            }
            Some((first, seen)) if *seen != d.advertised => {
                differ.insert(
                    d.advertised.arch,
                    format!("{first}: {seen:?}\n  {id}: {:?}", d.advertised),
                );
            }
            Some(_) => {}
        }
    }
    for (label, how) in &differ {
        assert!(
            LABELS_WHOSE_ROWS_DIFFER.iter().any(|(l, _)| l == label),
            "rows advertising `{label}` promise different things and no \
             reason is written down for it:\n  {how}\nEither make them \
             agree, or add `{label}` to LABELS_WHOSE_ROWS_DIFFER with the \
             argument for why a guest reading only the label is still \
             served correctly."
        );
    }
    for (label, _) in LABELS_WHOSE_ROWS_DIFFER {
        assert!(
            differ.contains_key(label) || !first_seen.contains_key(label),
            "`{label}` is excused for having rows that disagree, and its \
             rows now agree. Delete the entry -- an excuse nothing needs is \
             an excuse the next reader has to disprove."
        );
    }
}

/// And the towers behind those labels, which are not part of `Advertised`
/// and so can drift without the check above noticing.
///
/// The gemma-4 entry is the live one: two rows, one label, two genuinely
/// different vision encoders. It is safe for the reason written beside it,
/// and this test exists so that a THIRD tower appearing under a label with
/// no such reason is a failure rather than a discovery.
#[test]
fn every_label_covering_different_towers_is_one_that_was_thought_about() {
    let mut vision: BTreeMap<&'static str, (&'static str, model::deployment::VisionTower)> =
        BTreeMap::new();
    let mut audio: BTreeMap<&'static str, (&'static str, model::deployment::AudioTower)> =
        BTreeMap::new();
    let mut differ: BTreeMap<&'static str, String> = BTreeMap::new();
    let mut compared = 0usize;
    for (id, d) in deployments() {
        if let Some(v) = d.towers.vision {
            match vision.get(d.advertised.arch) {
                None => {
                    vision.insert(d.advertised.arch, (id, v));
                }
                Some((first, seen)) => {
                    compared += 1;
                    if *seen != v {
                        differ.insert(
                            d.advertised.arch,
                            format!("vision -- {first}: {seen:?}\n  {id}: {v:?}"),
                        );
                    }
                }
            }
        }
        if let Some(a) = d.towers.audio {
            match audio.get(d.advertised.arch) {
                None => {
                    audio.insert(d.advertised.arch, (id, a));
                }
                Some((first, seen)) => {
                    compared += 1;
                    if *seen != a {
                        differ.insert(
                            d.advertised.arch,
                            format!("audio -- {first}: {seen:?}\n  {id}: {a:?}"),
                        );
                    }
                }
            }
        }
    }
    assert!(
        compared > 0,
        "no two tower-bearing rows share a label, so this test compared \
         nothing"
    );
    for (label, how) in &differ {
        assert!(
            LABELS_WHOSE_ROWS_DIFFER.iter().any(|(l, _)| l == label),
            "rows advertising `{label}` ship different encoders and one \
             front-end is selected for both, with no reason written \
             down:\n  {how}"
        );
    }
}
