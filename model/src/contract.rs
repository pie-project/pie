//! The contract aspect's registry: `model_type` in, authored contract out.
//!
//! The rows mirror `driver/cuda/src/model/registry.cpp` while the migration
//! runs — one row per `model_type`, every N:1 reuse written out — and shrink
//! that file as families move here. Like `instruct::create`, the match
//! dispatches on the *model type*; the family directories only organize the
//! implementations.
//!
//! `author` is the whole aspect: a pure function from
//! `(facts, checkpoint, target, policy)` to the contract the loader
//! compiles. Nothing here opens a file or asks a device — both of those are
//! the caller's, which is what lets the same row serve a driver boot, an
//! offline `pie model optimize`, and a test that authors against a fixture.

use pie_loader::checkpoint::CheckpointMetadata;
use pie_loader::contract::ModelContract;
use pie_loader::error::Error;
use pie_loader::plan::StorageTarget;

use crate::common::builder::Builder;
use crate::common::facts::ModelFacts;
use crate::common::policy::Policy;

/// Author the load contract for one model type.
///
/// `Ok(None)` when no family here authors this `model_type` yet — during the
/// migration that answer means "ask the C++ author", and afterwards it means
/// "unsupported model".
pub fn author(
    facts: &ModelFacts,
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    policy: &Policy,
) -> Result<Option<ModelContract>, Error> {
    let author = match facts.model_type.as_str() {
        // ── llama lineage: dense/GQA decoders sharing one storage schema.
        "qwen3" | "qwen2" | "llama" | "llama3" | "mistral" => crate::llama::contract::author_llama_like,
        "mistral3" | "ministral3" | "olmo2" | "olmo3" => crate::llama::contract::author_dense,
        "phi3" => crate::llama::contract::author_phi3,
        _ => return Ok(None),
    };
    let mut builder = Builder::new(metadata, facts, target, policy);
    author(&mut builder)?;
    builder.finish().map(Some)
}
