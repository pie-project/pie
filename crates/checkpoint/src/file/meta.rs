//! Reserved metadata namespace of a pie artifact: the compiled tokenizer,
//! model descriptor and provenance, stored as `dense` `u8` objects named
//! under [`META_PREFIX`]; everything else is a weight.

/// The name prefix every metadata object carries, and no weight may.
pub const META_PREFIX: &str = "__meta__/";

/// File attribute: the pie that wrote this artifact.
pub const VERSION_KEY: &str = "pie_version";

/// File attribute: where the weights came from — a repo ID, or a path.
pub const SOURCE_KEY: &str = "pie_source";

/// File attribute: how the source stored these numbers — comma-separated,
/// sorted encodings (`q4_0`, `q4_k,q6_k`, `bf16`); absent if unstated.
///
/// Distinct from `models::serve::encoding::Encoding`, which reads
/// `config.json` and so answers "not quantized" for an imported archive
/// (e.g. GGUF) whose weights actually are quantized. This is that fact,
/// kept where it's answerable.
pub const SOURCE_ENCODING_KEY: &str = "pie_source_encoding";

/// File attribute: the runtime quantization already baked into these
/// weights. Absent from anything `pie model import` writes (it never
/// quantizes); kept for older `pie model build` artifacts, since a weight's
/// dtype alone can't say whether it was already quantized.
pub const RUNTIME_QUANT_KEY: &str = "pie_runtime_quant";

/// Whether `name` addresses a metadata object rather than a weight.
pub fn is_meta(name: &str) -> bool {
    name.starts_with(META_PREFIX)
}

/// The metadata name `path` sits at, e.g. `meta_name("tokenizer/vocab_bytes")`.
pub fn meta_name(path: &str) -> String {
    format!("{META_PREFIX}{path}")
}

/// Rejects a weight name that would land in the reserved namespace. Called
/// by the writer for every weight it declares.
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

