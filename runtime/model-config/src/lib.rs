//! HuggingFace `config.json` normalization, once, at import.
//!
//! The question "what is this model made of" is answered today at every serve
//! boot, by an ~850-line C++ normalizer with 25 `model_type` conditionals
//! (`driver/cuda/src/model/config.cpp`) *and* independently by two smaller
//! Rust probes that must agree with it by coincidence. This crate answers it
//! once, at `pie model convert`, and writes the answer into the artifact as
//! the `pie.model/1` descriptor.
//!
//! What makes that safe to do is not care, it is the differential test:
//! `tests/differential.rs` compares this normalizer against the C++ one over
//! 55 real and synthetic configs, field for field. See
//! `driver/cuda/tests/hf_config_dump/README.md` for how the oracle is built.

pub mod json;
pub mod normalize;
pub mod schema;

pub use normalize::normalize;
pub use schema::{HfConfig, RopeScaling};

/// The schema version the descriptor is written under.
pub const VERSION: &str = "pie.model/1";

/// Where the descriptor lives in an artifact's metadata namespace.
///
/// Named here, beside the schema it holds, for the same reason
/// `pie.tokenizer/1` names its own objects: a writer and a reader that
/// disagree about the name do not fail, they just find nothing.
pub const DESCRIPTOR_OBJECT: &str = "model/descriptor";

/// The `pie.model/1` descriptor for a HuggingFace `config.json`.
///
/// The normalized config, minus what is not a fact about the checkpoint, plus
/// the schema version. Readers get a flat struct with every HF defaulting rule
/// already resolved — no `text_config` nesting to step through, no key
/// alternatives to probe, no per-`model_type` conditionals.
///
/// **`head_dim_kernel` is excluded.** It rounds `head_dim` up to a head dim
/// the *driver build* instantiated in `kernels.def`, so it is a property of a
/// binary rather than of a model; baking it into an artifact would couple the
/// two. The driver recomputes it from `head_dim` at load, as it does today.
pub fn descriptor(root: &serde_json::Value, path: &str) -> anyhow::Result<serde_json::Value> {
    let config = normalize(root, path)?;
    let mut value = serde_json::to_value(&config)?;
    let map = value
        .as_object_mut()
        .expect("a struct serializes to an object");
    map.remove("head_dim_kernel");
    map.insert(
        "version".to_string(),
        serde_json::Value::String(VERSION.into()),
    );
    Ok(value)
}
