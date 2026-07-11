//! pie:core/model - Model and tokenizer global functions.
//!
//! The engine serves exactly one model, so these are free functions over the
//! single global [`pie_model::Model`] rather than resource methods.

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use anyhow::Result;
use pie_model as model;

impl pie::inferlet::model::Host for ProcessCtx {
    async fn name(&mut self) -> Result<String> {
        Ok(model::model().name().to_string())
    }

    async fn architecture(&mut self) -> Result<String> {
        Ok(model::model().arch_name().to_string())
    }

    async fn default_system_speculation(&mut self) -> Result<bool> {
        Ok(false)
    }

    /// Whether the bound model is linear/recurrent (carries a fused recurrent
    /// state that folds tokens irreversibly). TRUE iff the model has recurrent
    /// state — the same predicate the CUDA executor keys fold-commit on
    /// (`use_slots` / `rs_cache` present). Derived from the driver-handshake RS
    /// caps: a non-zero folded-state size means the model has recurrent state.
    /// The runtime uses this to select fold-commit (linear) vs KV-slot discard
    /// (attention) for speculative decode, so the inferlet stays model-agnostic.
    async fn is_linear(&mut self) -> Result<bool> {
        Ok(model::model().rs_caps().state_size > 0)
    }

    /// LM-head output dimension = `hf_config.vocab_size` (e.g. 151936 for
    /// qwen3), NOT the tokenizer vocab — the vocab the recognizer / program
    /// lowering targets. Sourced from the model config, not hardcoded.
    async fn output_vocab_size(&mut self) -> Result<u32> {
        Ok(model::model().vocab_size())
    }

    // ── Working-set / arena capabilities (global over the bound model) ──────
    //
    // Real values come from the driver handshake `DriverCapabilities`
    // (`rs_cache_slot_bytes` etc.), carried on the global `model::Model` via
    // `RsCaps` (populated at registration alongside the arena `ArenaConfig`).

    /// Bytes of one folded recurrent-state object (0 if the model has no RS).
    async fn rs_state_size(&mut self) -> Result<u64> {
        Ok(model::model().rs_caps().state_size)
    }

    /// Tokens per buffered RS page (0 if the model has no RS).
    async fn rs_buffer_page_size(&mut self) -> Result<u32> {
        Ok(model::model().rs_caps().buffer_page_size)
    }

    /// Fold granularity in tokens. 1 = unconstrained (token-causal: Qwen3.5 GDN,
    /// Nemotron-H Mamba2). `forward-pass.fold-buffered(n)` requires `n` to be a
    /// positive multiple of this.
    async fn rs_fold_granularity(&mut self) -> Result<u32> {
        Ok(model::model().rs_caps().fold_granularity)
    }

    /// KV page size (tokens) of the bound model.
    async fn arena_block_size(&mut self) -> Result<u64> {
        Ok(crate::store::registry::get(0, 0).kv_page_size as u64)
    }
}
