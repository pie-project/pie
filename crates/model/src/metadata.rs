//! What a served model's compiled metadata *is* — the shape, not the cache.

/// The compiled metadata for the model being served, lifted by the worker.
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    /// The `pie.tokenizer/1` objects, by their name under `__meta__/`.
    /// `None` for a snapshot, whose tokenizer is a file beside the weights.
    pub tokenizer: Option<Vec<(String, Vec<u8>)>>,
    /// The checkpoint's own `config.json`, verbatim. Always present.
    pub config: Vec<u8>,
}
