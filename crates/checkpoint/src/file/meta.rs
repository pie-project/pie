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
//! ([`Metadata::weights`](crate::file::Metadata::weights))
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

/// File attribute: how the SOURCE stored the numbers this artifact carries.
///
/// Written by `pie model import` — the distinct encodings of the checkpoint it
/// read, comma-separated and sorted (`q4_0`, `q4_k,q6_k`, `bf16`). Absent when
/// the archive it came from did not state it.
///
/// **Not the same fact as an encoding, and the difference is the point.**
/// `model::serve::encoding::Encoding` answers "how are THIS checkpoint's numbers
/// stored" by reading `config.json`'s `quantization_config`. An archive
/// `pie model import` wrote has no such block — GGUF states its quantization
/// per tensor and never in a config — so that question returns "not
/// quantized" for an archive whose weights are, tensor for tensor, Q4_K.
///
/// It answered "not quantized" before an archive kept its source packing too,
/// and back then it was even true: the import decoded every block on the way
/// in. Preservation did not create this gap, it only made the answer wrong as
/// well as unhelpful.
///
/// `Builder::runtime_quant_scheme` asks `encoding.is_none()` when what it
/// means is *"was this ever quantized"*, so that its refusal — "for FP8/INT8
/// only BF16 weights are re-quantized, never an already-quantized checkpoint"
/// — will fire. Against an archive it cannot, and no widening of that test
/// helps, because the fact is not in the place it looks. This is that fact,
/// kept where the question is actually answerable.
///
/// Whether reading it refuses or merely warns is `build::resolve_quant`'s
/// call, and it turns on whether the operator has another move. On CUDA they
/// do — the blocks decode to BF16 and bind — so a second rounding is refused.
/// On Metal, Vulkan and wgpu they do not: measured, those backends **require**
/// `--quant int4` (the MLX affine path binds every projection through it), so
/// refusing would not protect quality, it would make GGUF models unservable
/// there entirely. Same fact, two answers, one place that knows both.
pub const SOURCE_ENCODING_KEY: &str = "pie_source_encoding";

/// File attribute: the runtime quantization already baked into these weights.
///
/// Absent from every artifact this build writes: `pie model import` never
/// quantizes. It was written by the offline `pie model build`, and it is kept
/// because `src/local/store.rs` still reports it for a file an older pie left
/// in the store.
///
/// It exists because nothing about a weight says whether it has been
/// quantized already. An FP8 weight a checkpoint shipped and an FP8 weight a
/// quantizing build produced are both `Raw(Fp8E4m3)` at `[N, K]` with an
/// `_scale_inv` beside them; the first is a legitimate re-encoding source and
/// the second is a finished product, and they differ only in where they came
/// from. The
/// guard in `runtime_quant_scheme` tries to tell them apart by asking what
/// the weight *is*, which cannot work. This is the artifact saying so.
pub const RUNTIME_QUANT_KEY: &str = "pie_runtime_quant";

// NINE KEYS STOOD HERE, AND THEY WERE THE RUNTIME-ARTIFACT VOCABULARY:
// `pie_backend`, `pie_cache_key`, `pie_moe`, `pie_component`, `pie_tp_size`,
// `pie_source_stat`, `pie_model_id`, `pie_contract`, and the `CONTRACT_REVISION`
// the last of those stamped.
//
// They were one half of a conversation. `pie model build` wrote all of them
// into a `<name>/runtime/<key>.zt`; worker::weights::prefer_runtime read all
// of them back and bound the file only if every fact agreed. R3 deleted the
// writer, which left the readers matching against a directory nothing creates,
// and deleting the readers left these naming a protocol with no participants —
// nothing in this tree writes one, and nothing reads one.
//
// The keys above this line survive because their conversation still has both
// ends: `pie model import` writes `pie_version`, `pie_source`,
// `pie_source_encoding` and `pie_runtime_quant`, and `src/local/store.rs` and
// the import path read them.
//
// TWO THINGS WORTH KEEPING FROM WHAT WENT, because they were judgements and not
// facts, and a future runtime cache will face both again:
//
// * `pie_model_id` was BELIEVED, not checked, and that was correct. Catalog
//   identification answers "which model is this" by holding a checkpoint's
//   tensor names against each row's manifest, which works because a checkpoint
//   is written in HuggingFace's vocabulary. A built artifact is not — its
//   tensors are post-transform by construction, so identification refuses it,
//   correctly, given the question. The row had already been decided against the
//   archive, where the check meant something, so the build wrote the answer
//   down. Believing it was safe only because it was paired with the contract
//   revision: an artifact from a pie that laid tensors out differently is not
//   believed about anything.
// * `CONTRACT_REVISION` was hand-maintained rather than hashed, deliberately.
//   Hashing the pie version invalidates every artifact on every rebuild;
//   hashing nothing serves wrong bytes silently. A number a human must move is
//   the honest statement that it is a judgement. Its rule — bump when a change
//   to the authored contract or the compiled plan moves bytes — had one
//   recorded exception, and the exception is the useful part: a transform was
//   ADDED (`Builder::decode_bound_blocks` gained every published block a
//   `Cast`) and it did not bump, because built runtimes agreed byte for byte on
//   all 290 tensors of `qwen2.5-0.5b-instruct-q4_0`. The trigger is a proxy;
//   the question is whether an old artifact still describes the bytes the new
//   pie binds.
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
