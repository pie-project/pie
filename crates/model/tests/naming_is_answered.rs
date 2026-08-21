//! EVERY ROW ANSWERS FOR EVERY NAME LAYOUT, AND SAYS SO IN WORDS.
//!
//! `Policy::naming` is the second backend-shaped axis. A checkpoint from
//! `mlx-community` names its tensors nothing like a `transformers`
//! publication of the same weights, so `Naming::Mlx` and `Naming::Hf`
//! are two different authoring problems, and a generation either solves
//! the second one or does not.
//!
//! The registry this catalog replaced answered "does not" by having no
//! row — and `None` from a table lookup is not a sentence. Eight
//! generations now state the refusal instead, in a message that names
//! the generation, and that is the property this file holds: asked to
//! author MLX names, a row either does something different from what it
//! does for HF names, or SAYS that it cannot.
//!
//! # Why an integration test
//!
//! The failure is a generation ADDED whose `author` never looks at
//! `naming()` — which is invisible from inside every module that
//! already exists, because each one only sees its own rows. So it walks
//! `catalog()`, and the lower bound on the row count is there because a
//! walk of an empty iterator passes every assertion in it.
#![cfg(feature = "contract")]

use model::catalog::{self, LoadShape};
use model::encoding::Encoding as StoredEncoding;
use model::shared::builder::Builder;
use model::shared::policy::{Naming, Policy};
use model_loader::checkpoint::{CheckpointMetadata, RawTensor};
use model_loader::error::Error;
use model_loader::plan::StorageTarget;
use model_loader::types::{DType, Encoding, FileId, TensorId};

/// A checkpoint with one tensor no generation claims.
///
/// It has to be non-empty: `Builder::finish` refuses an empty contract,
/// and that refusal would be the same sentence for every row, which
/// would make the comparison below vacuous. One unclaimable tensor lets
/// each generation's own pass get far enough to say its own thing.
fn a_checkpoint_no_generation_claims() -> CheckpointMetadata {
    CheckpointMetadata {
        files: Vec::new(),
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "model.embed_tokens.weight".to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 128,
            shape: vec![8, 8],
            encoding: Encoding::Raw(DType::BF16),
        }],
    }
}

fn authored_under(naming: Naming) -> Vec<(&'static str, Result<usize, String>)> {
    let metadata = a_checkpoint_no_generation_claims();
    let encoding = StoredEncoding::dense();
    let target = StorageTarget::default();
    let policy = Policy {
        naming,
        ..Policy::default()
    };
    catalog::catalog()
        .iter()
        .map(|row| {
            let mut builder = Builder::new(
                &metadata,
                row.id(),
                LoadShape::dense(1, 64, true),
                &encoding,
                &target,
                &policy,
            );
            let outcome = row
                .author(&mut builder)
                .and_then(|()| builder.finish())
                .map(|contract| contract.tensors.len())
                .map_err(|error| match error {
                    Error::Contract(message) => message,
                    other => format!("{other:?}"),
                });
            (row.id(), outcome)
        })
        .collect()
}

/// The generations whose `author` does not consult `naming()` at all.
///
/// Each one gets the SAME contract for both layouts, which means a
/// Metal load of one of these rows is authored against `transformers`
/// names and fails at bind rather than here. That is a real gap, and it
/// is listed rather than fixed because closing it needs an MLX pass per
/// generation — but it is listed so that the gap is a decision on the
/// record instead of a silence, and so that a NEW generation cannot
/// join it without this test being edited.
const NAMING_IS_NOT_CONSULTED: &[&str] = &["csm", "kimi-k2", "nemotron-h", "phi-3", "phi-4"];

#[test]
fn every_row_either_authors_mlx_differently_or_refuses_in_words() {
    let hf = authored_under(Naming::Hf);
    let mlx = authored_under(Naming::Mlx);
    assert_eq!(hf.len(), mlx.len());
    // The catalog census, third copy. `chat_surface_is_answered` keeps it at
    // 58 and `advertised_matches_what_is_shipped` keeps 58 minus five; this
    // file had the same number behind a 40 and its message even says so --
    // "the catalog has {} rows" -- while allowing eighteen of them to leave
    // without a word. Three walks over one list, three private floors, and
    // no two of them agreeing on what a healthy size looks like.
    assert_eq!(
        hf.len(),
        58,
        "the catalog has {} rows, not 58. Move this with the census in \
         `chat_surface_is_answered.rs`; they are the same list.",
        hf.len()
    );

    let mut silent = Vec::new();
    for ((id, hf_outcome), (mlx_id, mlx_outcome)) in hf.iter().zip(mlx.iter()) {
        assert_eq!(id, mlx_id, "the two walks visited the catalog in one order");
        if hf_outcome == mlx_outcome {
            silent.push(*id);
        }
    }

    let unexpected: Vec<&str> = silent
        .iter()
        .copied()
        .filter(|id| {
            !NAMING_IS_NOT_CONSULTED
                .iter()
                .any(|known| id.starts_with(known))
        })
        .collect();
    assert!(
        unexpected.is_empty(),
        "these rows answer `Naming::Mlx` with exactly what they answer `Naming::Hf`, \
         so a Metal load of one of them is authored against transformers names and \
         fails at bind instead of here: {unexpected:?}"
    );
}

/// A refusal that names the WRONG generation is worse than no refusal:
/// it sends the reader to the wrong module. Eight of these are
/// near-identical copies of one another, which is exactly the shape of
/// text that acquires that bug — and a row's id cannot catch it, because
/// a row is named for a model and not for its generation
/// (`embeddinggemma-300m` is a gemma-3).
///
/// What catches it is the SET. A copy of gemma-3's arm left in gemma-3n
/// makes both generations answer in gemma-3's name, and the set loses a
/// member.
#[test]
fn the_generations_that_state_the_refusal_each_state_their_own_name() {
    let mut named: Vec<String> = Vec::new();
    for (id, outcome) in authored_under(Naming::Mlx) {
        let Err(message) = outcome else {
            continue;
        };
        if !message.contains("no MLX authoring pass") {
            continue;
        }
        let (generation, _) = message
            .split_once(':')
            .unwrap_or_else(|| panic!("'{id}': the refusal states no generation: {message}"));
        assert!(
            message.contains("no name layout to author against"),
            "'{id}': the refusal says WHAT is missing, not just that something is: {message}"
        );
        if !named.iter().any(|seen| seen == generation) {
            named.push(generation.to_string());
        }
    }
    named.sort_unstable();
    assert_eq!(
        named,
        [
            "deepseek-v4",
            "gemma-2",
            "gemma-3",
            "gemma-3n",
            "glm-5",
            "kimi-k3",
            "olmo-2",
            "olmo-3",
        ],
        "either a generation gained or lost an MLX pass, or two of them are \
         refusing in the same name"
    );
}
