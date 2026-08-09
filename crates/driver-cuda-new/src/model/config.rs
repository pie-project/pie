//! The HF checkpoint configuration — gate-hf-config.
//!
//! There is no struct here to port, and that is the finding: `config.hpp`'s
//! tail records that `parse_hf_config` (855 lines, 25 `model_type`
//! conditionals) was deleted after `config.json` normalization moved to
//! Rust at import time. The normalizer's output type —
//! `model::config::schema::HfConfig`, generated from the same header and
//! differential-tested against the C++ parser over the 56-config corpus —
//! IS the config type, and this driver re-exports it rather than growing a
//! third copy. What this crate adds is the READ side: see
//! [`super::descriptor`] for the `pie.model/1` reader.
//!
//! An earlier slice of this gate carried a local 12-field `HfConfig` for
//! `apply_rope_config` and the prepare hook; the re-export replaced it with
//! no caller changes beyond the enum's spelling (`OriginalYarn`), because
//! the schema keeps the C++ field names.

pub use model::config::schema::{
    CsmConfig, CsmDepthDecoderConfig, GemmaAudioConfig, GemmaVisionConfig,
    HfConfig, MimiCodecConfig, Qwen3VLVisionConfig, RopeScaling,
};
