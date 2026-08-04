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

    /// Which forward-pass interface the bound model requires, over STATE
    /// SEMANTICS rather than architecture name. Recurrent state is present iff
    /// the driver handshake reports a non-zero folded-state size — the same
    /// predicate the CUDA executor keys fold-commit on (`use_slots` /
    /// `rs_cache` present); paged KV is present iff the model has a KV
    /// page size. Today every registered linear model (Qwen3.5 GDN dense/MoE,
    /// Nemotron-H Mamba2) interleaves attention layers and therefore reports
    /// `hybrid`; `recurrent` is reachable only for a model with no KV at all.
    async fn pass_kind(&mut self) -> Result<pie::inferlet::model::ForwardKind> {
        use pie::inferlet::model::ForwardKind;
        let model = model::model();
        let has_rs = model.rs_caps().state_size > 0;
        let has_kv = model.kv_page_size() > 0;
        Ok(match (has_kv, has_rs) {
            (_, false) => ForwardKind::Attention,
            (true, true) => ForwardKind::Hybrid,
            (false, true) => ForwardKind::Recurrent,
        })
    }

    /// LM-head output dimension = `hf_config.vocab_size` (e.g. 151936 for
    /// qwen3), NOT the tokenizer vocab — the vocab the recognizer / program
    /// lowering targets. Sourced from the model config, not hardcoded.
    async fn output_vocab_size(&mut self) -> Result<u32> {
        Ok(model::model().vocab_size())
    }

    async fn kv_page_size(&mut self) -> Result<u32> {
        Ok(model::model().kv_page_size())
    }

    /// Waves per frame (k) — the static deployment constant `forward.submit`
    /// sizes its slot list to. Fixed at engine start, like `kv-page-size`.
    async fn frame_size(&mut self) -> Result<u32> {
        Ok(crate::scheduler::configured_frame_size() as u32)
    }

    /// Bound on how long a pipeline may hold a frame's wait-set without
    /// submitting. See `scheduler::configured_submit_deadline`.
    async fn submit_deadline_us(&mut self) -> Result<u64> {
        Ok(crate::scheduler::configured_submit_deadline().as_micros() as u64)
    }

    /// Host-reader channel capacity, in cells, that sustains the engine's
    /// run-ahead for one lane. Includes the staging margin; see
    /// `scheduler::channel_capacity`.
    async fn channel_capacity(&mut self) -> Result<u32> {
        Ok(crate::scheduler::channel_capacity() as u32)
    }

    /// Max embed tokens in a single pass (C) — the guest-side prefill chunk
    /// budget, sourced from the bound driver's structural per-launch token
    /// capacity.
    async fn max_embed_length(&mut self) -> Result<u32> {
        Ok(crate::driver::get_spec(0)?.limits.max_forward_tokens as u32)
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
    /// Nemotron-H Mamba2). An RS fold of `n` tokens requires `n` to be a
    /// positive multiple of this.
    async fn rs_fold_granularity(&mut self) -> Result<u32> {
        Ok(model::model().rs_caps().fold_granularity)
    }

    /// KV page size (tokens) of the bound model.
    async fn arena_block_size(&mut self) -> Result<u64> {
        Ok(crate::store::registry::get(0, 0).kv_page_size as u64)
    }
}
