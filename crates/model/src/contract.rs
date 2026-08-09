//! The contract aspect: a row in, an authored contract out.
//!
//! This module used to BE the registry — two `const` tables keyed on the
//! `model_type` string out of a `config.json`, 52 rows between them, and
//! a `find` over whichever the driver's [`Naming`] selected. It is now a
//! single function, because the registry moved to
//! [`catalog`](crate::catalog) and there is only one of it.
//!
//! That is the whole change, and it is worth being precise about what it
//! bought. `tests/registry_agreement.rs` existed to hold THREE tables to
//! each other — this one, the CUDA arch table, and Metal's
//! `model_family_of` — because a `model_type` added to one and not the
//! others surfaced as two unrelated errors ("unsupported model_type"
//! from a driver, "no author" from here) whose common cause nothing
//! named. A test that checks two tables agree is a test that exists
//! because there are two tables. There is one now, every driver links
//! it, and the disagreement it was watching for cannot be expressed.
//!
//! What is left here is a pure function from
//! `(row, encoding, checkpoint, target, policy)` to the contract the
//! loader compiles. Nothing here opens a file or asks a device — both of
//! those are the caller's, which is what lets one row serve a driver
//! boot, an offline `pie model build`, and a test that authors against a
//! fixture.

use model_loader::checkpoint::CheckpointMetadata;
use model_loader::contract::ModelContract;
use model_loader::error::Error;
use model_loader::plan::StorageTarget;

use crate::builder::Builder;
use crate::policy::{Mxfp4MoePolicy, Policy};

/// What a row's `author` dispatches to: a family's authoring pass.
///
/// Still a named type, because the N:1 is still real — a dozen rows call
/// `author_llama_like` — but it is a CALL now and not a table column.
pub type Author = fn(&mut Builder<'_>) -> Result<(), Error>;

/// Author the load contract for one row.
///
/// # What this replaced
///
/// Two `const` tables — `HF_ROWS` (35 rows) and `MLX_ROWS` (17) — keyed
/// on the `model_type` string out of a `config.json`, plus a `find` over
/// whichever the [`Naming`] selected, plus an `Ok(None)` for "no row
/// claims this string".
///
/// All three are gone, and each for its own reason:
///
/// - The **tables** are gone because a row states its own author. That
///   is the same N:1 the table column expressed — a dozen generations
///   calling `author_llama_like` — spelled as a call, where nothing can
///   drift out of step with the other two answers the same row gives.
/// - The **string key** is gone because identity is not a string any
///   more. `"qwen3"` reached one author and `"qwen3_moe"` reached
///   another, and the second disagreed with what the same string
///   selected in `FACTS_ROWS`.
/// - The **`Option`** is gone because the caller cannot get here without
///   a row. [`identify`](crate::catalog::identify) either matched a
///   checkpoint to a variant or refused; "authored nothing, returned
///   `Ok`" was a third outcome that every caller had to remember to
///   check, and one of them turned it into `UnknownFamily` several
///   frames later, having lost the reason.
///
/// # Errors
///
/// The row's author refused: the checkpoint contradicts a shape it
/// asserts, or the contract came out empty.
pub fn author_with_policy(
    row: &dyn crate::catalog::Variant,
    encoding: &crate::encoding::Encoding,
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    policy: &Policy,
) -> Result<(ModelContract, Mxfp4MoePolicy), Error> {
    let mut builder = Builder::new(
        metadata,
        row.id(),
        row.load_shape(),
        encoding,
        target,
        policy,
    );
    row.author(&mut builder)?;
    let resolved = builder.mxfp4_moe();
    builder.finish().map(|contract| (contract, resolved))
}

/// The contract alone, for a caller with no interest in the MXFP4
/// resolution.
///
/// # Errors
///
/// As [`author_with_policy`].
pub fn author(
    row: &dyn crate::catalog::Variant,
    encoding: &crate::encoding::Encoding,
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    policy: &Policy,
) -> Result<ModelContract, Error> {
    author_with_policy(row, encoding, metadata, target, policy).map(|(contract, _)| contract)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Variant;
    use crate::encoding::Encoding as StoredEncoding;
    use crate::policy::Mxfp4MoeRequest;
    use model_loader::checkpoint::{FileId, RawTensor};
    use model_loader::types::{DType, Encoding as TensorEncoding, TensorId};

    /// A real row, so these test the function and not a stub of it.
    ///
    /// `qwen3-0.6b` because it is the exemplar the rest of the catalog
    /// was transcribed against, and because its author is
    /// `author_llama_like` — the N:1 that `HF_ROWS` spelled as a table
    /// column and that this module now spells as a call.
    fn row() -> &'static dyn Variant {
        crate::catalog::find("qwen3-0.6b").expect("the exemplar row")
    }

    /// A checkpoint with the handful of tensors the dense tail
    /// publishes directly.
    ///
    /// Deliberately not a whole qwen3: `author_with_policy` decides
    /// nothing about WHICH tensors exist, it decides that the row's
    /// author sees them and that the builder's two outputs come back
    /// paired. A 28-layer fixture would test `publish_remaining`, which
    /// has its own tests.
    fn checkpoint(names: &[(&str, &[i64])]) -> CheckpointMetadata {
        let mut offset = 0u64;
        let tensors = names
            .iter()
            .enumerate()
            .map(|(i, (name, shape))| {
                let elements: i64 = shape.iter().product();
                let span_bytes = u64::try_from(elements).unwrap() * DType::BF16.bytes();
                let raw = RawTensor {
                    id: TensorId(u32::try_from(i).unwrap()),
                    name: (*name).to_string(),
                    file_id: FileId(0),
                    file_offset: offset,
                    span_bytes,
                    shape: shape.to_vec(),
                    encoding: TensorEncoding::Raw(DType::BF16),
                };
                offset += span_bytes;
                raw
            })
            .collect();
        CheckpointMetadata { files: Vec::new(), tensors }
    }

    fn target() -> StorageTarget {
        StorageTarget { preferred_alignment: 256, ..StorageTarget::default() }
    }

    /// The row authors, and what comes back is the row's contract.
    #[test]
    fn a_row_authors_its_own_contract() {
        let meta = checkpoint(&[
            ("model.embed_tokens.weight", &[151_936, 1024]),
            ("model.norm.weight", &[1024]),
        ]);
        let contract = author(
            row(),
            &StoredEncoding::dense(),
            &meta,
            &target(),
            &Policy::default(),
        )
        .expect("a dense checkpoint the llama-like tail publishes");

        assert_eq!(
            contract.tensors.len(),
            2,
            "both tensors reached the contract; a name the author dropped is a \
             tensor the driver never binds"
        );
        assert_eq!(contract.alignment, 256, "the target's, carried through");
    }

    /// An empty checkpoint is refused HERE, not several frames later.
    ///
    /// This is the `Ok(None)` the old registry returned when no row
    /// claimed a `model_type`: a third outcome beside success and
    /// failure that every caller had to remember to check, and that one
    /// caller turned into `UnknownFamily` after the reason was gone.
    /// There is no third outcome now — a caller holding a row either
    /// gets a contract or gets a message naming the model.
    #[test]
    fn an_authorless_checkpoint_refuses_by_name() {
        let err = author(
            row(),
            &StoredEncoding::dense(),
            &checkpoint(&[]),
            &target(),
            &Policy::default(),
        )
        .expect_err("nothing was authored");

        let Error::Contract(message) = err else {
            panic!("an authoring failure is a contract error");
        };
        assert!(
            message.contains("qwen3-0.6b"),
            "the refusal names the MODEL, not a family string a dozen \
             checkpoints share: {message}"
        );
    }

    /// The MXFP4 resolution comes back beside the contract, from the
    /// same builder that made it.
    ///
    /// Paired rather than recomputed, because the device measurement it
    /// resolves against (`native_mxfp4_moe`) is the caller's and asking
    /// twice is how the two answers drift.
    #[test]
    fn the_moe_resolution_returns_with_the_contract_that_assumed_it() {
        let meta = checkpoint(&[("model.norm.weight", &[1024])]);
        let policy = Policy {
            moe_request: Mxfp4MoeRequest::Auto,
            ..Policy::default()
        };

        for (native, expected) in [
            (true, Mxfp4MoePolicy::NativeGemm),
            (false, Mxfp4MoePolicy::RoutedDecode),
        ] {
            let target = StorageTarget {
                preferred_alignment: 256,
                native_mxfp4_moe: native,
                ..StorageTarget::default()
            };
            let (_, resolved) = author_with_policy(
                row(),
                &StoredEncoding::dense(),
                &meta,
                &target,
                &policy,
            )
            .expect("a one-tensor contract");
            assert_eq!(resolved, expected, "native_mxfp4_moe={native}");
        }
    }

    /// [`author`] is [`author_with_policy`] with the second value
    /// dropped, and drops nothing else.
    #[test]
    fn the_two_entry_points_author_the_same_contract() {
        let meta = checkpoint(&[
            ("model.embed_tokens.weight", &[151_936, 1024]),
            ("model.norm.weight", &[1024]),
        ]);
        let enc = StoredEncoding::dense();
        let policy = Policy::default();

        let one = author(row(), &enc, &meta, &target(), &policy).expect("contract");
        let (two, _) =
            author_with_policy(row(), &enc, &meta, &target(), &policy).expect("contract");

        assert_eq!(one.alignment, two.alignment);
        assert_eq!(
            one.tensors.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
            two.tensors.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
        );
    }

    /// The row is asked for its id and its load shape, and the builder
    /// gets both.
    ///
    /// The point of the refactor in one assertion: these two answers
    /// used to come from different places — the id from a `model_type`
    /// string and the shape from a `ModelFacts` parsed out of a
    /// descriptor — and nothing held them to the same model.
    #[test]
    fn the_builder_is_told_which_model_it_is_authoring() {
        let shape = row().load_shape();
        assert_eq!(shape.layers, 28, "qwen3-0.6b's, from the row");
        assert_eq!(shape.head_dim, 128);
        assert!(shape.tied_embeddings, "the 0.6B ties its embeddings");
    }
}
