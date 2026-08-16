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

/// File attribute: how the SOURCE stored the numbers this artifact carries.
///
/// Written by `pie model import` — the distinct encodings of the checkpoint it
/// read, comma-separated and sorted (`q4_0`, `q4_k,q6_k`, `bf16`). Absent from
/// a `pie model build` artifact's own provenance only when the archive it came
/// from did not state it.
///
/// **Not the same fact as an encoding, and the difference is the point.**
/// `model::encoding::Encoding` answers "how are THIS checkpoint's numbers
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
/// Written by `pie model build` when it was asked for one, absent otherwise —
/// `pie model import` never quantizes, so its artifacts never carry this.
///
/// It exists because nothing about a weight says whether it has been
/// quantized already. An FP8 weight a checkpoint shipped and an FP8 weight
/// `build` produced are both `Raw(F8E4M3)` at `[N, K]` with an `_scale_inv`
/// beside them; the first is a legitimate re-encoding source and the second
/// is a finished product, and they differ only in where they came from. The
/// guard in `runtime_quant_scheme` tries to tell them apart by asking what
/// the weight *is*, which cannot work. This is the artifact saying so.
pub const RUNTIME_QUANT_KEY: &str = "pie_runtime_quant";

/// File attribute: the backend this artifact's tensor layout is for.
///
/// Written by `pie model build`, absent from an imported archive — an archive
/// is general form and is for every backend, which is what makes it the one
/// artifact that can be served anywhere.
///
/// A build is not. `--backend` decides projections and naming, and both land
/// in the tensors and neither lands anywhere a reader could see them: a CUDA
/// build and a Metal build of one model are two files of the same shapes under
/// the same names holding different numbers. The store keeps them apart by
/// path, since the cache key covers the whole plan — but a path is a place and
/// this is a statement, and an artifact moved out of the store by `--out` or
/// copied between machines takes only the statement with it.
pub const BACKEND_KEY: &str = "pie_backend";

/// File attribute: the cache key the runtime artifact was named under.
///
/// The path already says this for an artifact still in the store. This is for
/// one that is not: it is what lets a copied file be matched against a freshly
/// computed key instead of trusted for its filename.
pub const CACHE_KEY_KEY: &str = "pie_cache_key";

/// File attribute: the MoE lowering baked into these weights.
///
/// `auto`, `routed` or `bf16`, as `pie model build --moe` spells them. A
/// mixture model's expert banks are a different set of tensors under each, and
/// like the backend the difference lands entirely in the bytes.
pub const MOE_KEY: &str = "pie_moe";

/// File attribute: which slice of the model these weights are.
///
/// `driver_api::ModelComponent`'s lowercase name. `pie model build` writes
/// `full` — it materializes the model entire — and a serve that wants a slice
/// must not be handed a whole model under the belief that it asked for one.
pub const COMPONENT_KEY: &str = "pie_component";

/// File attribute: how many tensor-parallel ranks this layout assumes.
///
/// Always `1` today: `pie model build` materializes one unsharded artifact,
/// because a sharded materialization is one file per rank and the store has
/// no way to say that yet. Stated rather than implied, so that the day it
/// stops being 1 the readers already refuse the mismatch instead of binding
/// a rank-0 shard on every rank.
pub const TP_SIZE_KEY: &str = "pie_tp_size";

/// File attribute: the archive this was derived from, as it stood then.
///
/// [`crate::cache_key::snapshot_stat`] of the archive's directory. A runtime
/// is a *cache* of an archive, so it is only valid for the archive it was
/// built from: re-importing a model rewrites `archive.zt` under the same path,
/// and every runtime hanging off it is then stale in the one way that has no
/// other symptom — the tensors are the right names and shapes and the wrong
/// numbers.
pub const SOURCE_STAT_KEY: &str = "pie_source_stat";

/// File attribute: the catalog row `pie model build` authored these tensors as.
///
/// **The one provenance key that is believed rather than checked**, and the
/// reason it has to exist.
///
/// `catalog::identify` answers "which model is this" by holding a checkpoint's
/// tensor names and extents against each row's manifest. That works because a
/// checkpoint is written in HuggingFace's vocabulary and the manifests are
/// stated in it. A *built* artifact is not: `pie model build` runs the load
/// transforms and writes the result, so its tensors are spelled the way the
/// bind path reads them — `experts.gate_up_proj.weight` where the checkpoint
/// had `experts.gate_up_proj_blocks`, a fused bank where the checkpoint had
/// three projections. Identification refuses it, and the refusal is correct
/// given the question: these are genuinely not the tensors a manifest
/// describes.
///
/// The question is what is wrong. The row was already decided, by this same
/// identification, at build time, against the archive — in HuggingFace's
/// vocabulary, where the check means something. Re-deriving it from the output
/// asks a settled question of evidence that can no longer answer it. So the
/// build writes down the answer, and a boot reads it.
///
/// The alternative — teaching the manifests to also accept every family's
/// post-transform spelling — is the one `manifest::Observed::global_spelling`
/// already argues against in its own doc: it would make the checks a reverse
/// map per family, maintained in two places, and it would weaken the check for
/// real checkpoints in order to admit artifacts that were never in doubt.
///
/// **Believing this is only safe because it is paired with
/// [`CONTRACT_KEY`].** The statement is trusted when the artifact also says it
/// was laid out by *this* pie's contract revision; an artifact from a pie that
/// laid tensors out differently is not believed about anything, because the
/// row it names no longer implies the tensors it holds.
pub const MODEL_ID_KEY: &str = "pie_model_id";

/// File attribute: the contract revision that laid these tensors out.
///
/// [`CONTRACT_REVISION`] at the time of the build. The five keys above cover
/// every fact the *request* carries, but not the one the request cannot: pie's
/// own idea of which tensors a contract declares and how they are laid out.
/// Change that and yesterday's runtime answers today's request with a layout
/// today's bind path does not read.
pub const CONTRACT_KEY: &str = "pie_contract";

/// What [`CONTRACT_KEY`] states, and the reason to bump it.
///
/// **Bump this when a change to the authored contract or the compiled plan
/// moves bytes** — a projection that fuses differently, a tensor that gains or
/// loses a transform, a name the bind path now spells another way. Every
/// runtime artifact built before the bump then stops matching and is rebuilt,
/// which costs one `pie model build` and buys the guarantee that a stale
/// layout is never bound.
///
/// It is a hand-maintained number and not a hash of the tree because the
/// alternative spellings are worse in both directions: hashing the pie version
/// invalidates every artifact on every rebuild of pie, and hashing nothing at
/// all makes a contract change silently serve the wrong bytes. A number a
/// human has to move is the honest statement that this is a judgement.
///
/// - `1` — the first layout a `pie model build` wrote.
/// - `2` — builds stopped persisting fused projection banks. A fusion is a
///   view arrangement the load path makes over the tensors a file holds, so
///   persisting it wrote the bank *and* the projections it aliases: measured
///   at 587,202,560 duplicated bytes on Qwen3-0.6B, on disk and again in
///   VRAM. Every `1` artifact carries those banks, and is now refused rather
///   than bound.
///
/// # A change that fires the literal rule and still did not bump this
///
/// Recorded because the rule above, read literally, says it should have —
/// and the next person to read both will otherwise conclude that one of them
/// is wrong.
///
/// `986252f35` stopped `pie model import` unpacking a self-contained block, so
/// an archive now keeps the packing its source shipped. To keep such a tensor
/// bindable, `Builder::decode_bound_blocks` gives every *published* block a
/// `Cast` it did not have before. That is a tensor gaining a transform, which
/// is this doc's own trigger.
///
/// It did not bump, because the trigger is a proxy for the thing that actually
/// matters — **whether a runtime artifact built by the old pie still describes
/// the bytes the new pie binds** — and here it does. Two facts, both measured:
///
/// - The bytes do not move. Built from a decoded archive and from a packed one,
///   `qwen2.5-0.5b-instruct-q4_0`'s runtimes agree on all 290 tensors byte for
///   byte. The decode changed *where* it happens, not what it produces.
/// - A stale artifact is already refused, by a stronger check than this one.
///   `worker::weights::answers` requires [`SOURCE_STAT_KEY`] to match, and
///   re-importing writes a different archive with a different stat, so every
///   runtime built from the old archive is rebuilt whether or not this number
///   moves.
///
/// Bumping anyway is not free and not neutral: it invalidates every artifact on
/// every machine, costing a full rebuild each, to protect against a mismatch
/// that cannot occur. The rule stays a rule — when in doubt, bump — but "a
/// transform was added" is the question, not the answer.
pub const CONTRACT_REVISION: u32 = 2;

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
