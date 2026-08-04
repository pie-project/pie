//! The reserved metadata namespace of a pie artifact.
//!
//! A pie `.zt` artifact carries more than weights: the compiled tokenizer, the
//! compiled model descriptor, and the provenance the two can be regenerated
//! from. zTensor has no non-tensor object — every manifest entry is a shape, a
//! layout and a set of parts — so those payloads are stored the only way the
//! format offers, as `dense` `u8` objects, and are told apart from weights by
//! **name**: everything under [`META_PREFIX`] is metadata, everything else is a
//! weight.
//!
//! That makes the prefix load-bearing rather than decorative. A metadata object
//! is byte-identical in kind to a raw `u8` weight, so any consumer that
//! enumerates `tensors` without filtering will happily plan a tokenizer vocab
//! into a decode, copy it into a contract, or upload it to a device. The guard
//! against that is *not* remembering to skip it at each site — it is that the
//! enumeration a weight consumer reaches for
//! ([`CheckpointMetadata::weights`](crate::checkpoint::CheckpointMetadata::weights))
//! already excludes it, and the writer refuses to put a weight in the namespace
//! in the first place.
//!
//! The prefix is `__meta__/`: doubled underscores make an accidental collision
//! with a checkpoint's own tensor names implausible, and the trailing slash
//! makes the namespace a directory-like tree
//! (`__meta__/tokenizer/vocab_bytes`, `__meta__/model/descriptor`) rather than
//! a flat prefix match that `__meta__extra` would also satisfy.

/// The name prefix every metadata object carries, and no weight may.
pub const META_PREFIX: &str = "__meta__/";

/// File attribute: the pie that wrote this artifact.
///
/// The provenance keys live here, beside the namespace, because they are the
/// same kind of thing: names an artifact is written under and read back by. A
/// writer and a reader that disagree about one of these produce no error —
/// the read simply finds nothing — so there is exactly one definition and
/// both sides take it from here.
pub const VERSION_KEY: &str = "pie_version";

/// File attribute: where the weights came from — a repo ID, or a path.
pub const SOURCE_KEY: &str = "pie_source";

/// Whether `name` addresses a metadata object rather than a weight.
pub fn is_meta(name: &str) -> bool {
    name.starts_with(META_PREFIX)
}

/// The metadata name `path` sits at, e.g. `meta_name("tokenizer/vocab_bytes")`.
pub fn meta_name(path: &str) -> String {
    format!("{META_PREFIX}{path}")
}

/// Rejects a weight name that would land in the reserved namespace.
///
/// Called by the writer for every weight it declares. A source checkpoint is
/// free to contain any name at all — including, in principle, one starting
/// `__meta__/` — and this is where that stops being representable, before the
/// artifact exists rather than after something reads a weight as a descriptor.
pub fn reject_reserved(name: &str) -> Result<(), crate::error::Error> {
    if is_meta(name) {
        return Err(crate::error::Error::Checkpoint(format!(
            "tensor {name:?} is in the reserved metadata namespace ({META_PREFIX}); \
             pie artifacts keep the compiled tokenizer and model descriptor there, \
             so a weight cannot be written under that prefix"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_namespace_is_a_tree_not_a_prefix_match() {
        assert!(is_meta("__meta__/tokenizer/vocab_bytes"));
        assert!(is_meta("__meta__/model/descriptor"));
        // The trailing slash is what keeps a name that merely *starts* with
        // the word out of the namespace.
        assert!(!is_meta("__meta__extra"));
        assert!(!is_meta("__meta__"));
        assert!(!is_meta("model.layers.0.mlp.down_proj.weight"));
        assert!(!is_meta(""));
    }

    #[test]
    fn a_weight_cannot_be_written_into_the_namespace() {
        assert!(reject_reserved("model.embed_tokens.weight").is_ok());
        let err = reject_reserved("__meta__/tokenizer/vocab_bytes").unwrap_err();
        assert!(err.to_string().contains("reserved metadata namespace"));
    }

    #[test]
    fn meta_name_composes_with_is_meta() {
        let name = meta_name("tokenizer/merge_table");
        assert_eq!(name, "__meta__/tokenizer/merge_table");
        assert!(is_meta(&name));
    }
}
