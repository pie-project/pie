//! HuggingFace `config.json` normalization — the only one there is.
//!
//! "What is this model made of" used to be answered at every serve boot, four
//! times over: an 855-line C++ normalizer with 25 `model_type` conditionals
//! (`crates/driver-cuda/csrc/src/model/config.cpp`), Metal's own `read_model_facts`, and
//! two smaller Rust probes in the runtime's model service — each walking
//! `text_config` nesting and key alternatives in its own order, agreeing with
//! the others by coincidence. All four are deleted. This module answers the
//! question once and writes the answer as the `pie.model/1` descriptor:
//! at `pie model import` for an artifact, and in `worker/src/weights.rs`
//! before any driver exists for a plain HF snapshot.
//!
//! What makes that safe is not care, it is the differential test:
//! `tests/differential.rs` compares this normalizer against the C++ one it
//! replaced over 56 real and synthetic configs, field for field. The C++ side
//! is gone, so those goldens are now a recording rather than a live oracle —
//! see `crates/driver-cuda/csrc/tests/hf_config_dump/README.md`. What keeps a second
//! normalizer from growing back is `tests/one_normalizer.rs`.
//!
//! It is a module of `model` because that is what it is: model knowledge, not
//! runtime machinery. It sat under `runtime/` while the runtime was its only
//! caller, then became a crate of its own under `model/`; the crate boundary
//! bought nothing once the only two consumers were this crate
//! ([`ModelFacts::from_descriptor`](crate::facts::ModelFacts::from_descriptor))
//! and `worker`, which already links `model`. The `config` feature is what
//! the boundary was really for: it turns `model` into the same serde-only
//! leaf — no loader, no tracer, no kernels table — so `pie model convert`
//! still runs without dragging that graph in.
//!
//! ## What does NOT move into a generation module
//!
//! The per-`model_type` rules in [`normalize`] look like they belong beside
//! the generation they name, and they do not. Two reasons, both structural:
//!
//! - The pass is a **transliteration** of the C++ normalizer, and the order
//!   in which it writes fields is load-bearing — several are written more
//!   than once and the last write wins. Splitting the conditionals across
//!   twenty modules would dissolve the one property `tests/differential.rs`
//!   pins.
//! - It runs *before* there is a generation to dispatch to. Normalization is
//!   what resolves `model_type` in the first place (outer vs `text_config`,
//!   the standalone Kimi-Linear spelling), and both registries in this crate
//!   dispatch on the answer.

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
