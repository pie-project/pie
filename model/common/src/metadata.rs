//! What a served model's compiled metadata *is* — the shape, not the cache.
//!
//! This is the half of the old model service that stayed in `pie-model` when
//! the rest moved to `runtime/engine`. The split is on the crate's own
//! definition: a global `OnceLock` holding whatever this process booted is
//! runtime machinery, but "what a `.zt` artifact carries, and what the worker
//! normalizes a snapshot into" is a fact about models. The worker reads it
//! without linking the runtime at all.

/// The compiled metadata for the model being served, lifted by the worker.
///
/// Named for what it holds rather than where it came from. It was
/// `ArtifactMetadata` and it was optional: an artifact had one and a snapshot
/// did not, so the runtime kept a second path that parsed the files beside a
/// snapshot. There is no second path now — a snapshot's `config.json` is
/// normalized into a descriptor by `worker/src/weights.rs` before anything
/// downstream sees it, exactly as it is for the drivers.
///
/// The tokenizer half stays optional because it genuinely differs: an
/// artifact carries `pie.tokenizer/1` objects, and a snapshot has a
/// `tokenizer.json` on disk to load.
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    /// The `pie.tokenizer/1` objects, by their name under `__meta__/`.
    /// `None` for a snapshot, whose tokenizer is a file beside the weights.
    pub tokenizer: Option<Vec<(String, Vec<u8>)>>,
    /// The `pie.model/1` descriptor, as JSON. Always present.
    pub descriptor: Vec<u8>,
}
