//! pie:core/model - Model and tokenizer global functions.
//!
//! The engine serves exactly one model, so these are free functions over the
//! single global [`crate::model::Model`] rather than resource methods.

use crate::api::pie;
use crate::instance::InstanceState;
use crate::model;
use anyhow::Result;

impl pie::core::model::Host for InstanceState {
    async fn name(&mut self) -> Result<String> {
        Ok(model::model().name().to_string())
    }

    async fn architecture(&mut self) -> Result<String> {
        Ok(model::model().arch_name().to_string())
    }

    async fn default_system_speculation(&mut self) -> Result<bool> {
        // The effective "speculate by default?" decision the SDK reflects: the
        // model must support a system drafter AND the operator must have opted
        // in (`enable_system_speculation`, default off). The runtime owns this
        // decision; the SDK only requests system drafts when both hold. (Manual
        // drafts are a separate path, gated in api/inference.rs.)
        let m = model::model();
        Ok(m.system_speculation_supported() && m.enable_system_speculation())
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

    /// Arena accounting block size. v1: one KV page == one arena block, so this
    /// is the bound model's KV page size (tokens).
    async fn arena_block_size(&mut self) -> Result<u64> {
        Ok(crate::page_size::tokens_per_page(0) as u64)
    }

    async fn encode(&mut self, text: String) -> Result<Vec<u32>> {
        Ok(model::model().tokenize(&text))
    }

    async fn decode(&mut self, tokens: Vec<u32>) -> Result<Result<String, String>> {
        Ok(Ok(model::model().detokenize(&tokens)))
    }

    async fn vocabs(&mut self) -> Result<(Vec<u32>, Vec<Vec<u8>>)> {
        Ok(model::model().get_vocabs())
    }

    async fn split_regex(&mut self) -> Result<String> {
        Ok(model::model().get_split_regex())
    }

    async fn special_tokens(&mut self) -> Result<(Vec<u32>, Vec<Vec<u8>>)> {
        Ok(model::model().get_special_tokens())
    }
}
