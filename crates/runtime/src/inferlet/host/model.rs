//! pie:core/model - Model and tokenizer global functions.
//!
//! The runtime serves exactly one model, so these are free functions over the
//! single global [`crate::model::Model`] rather than resource methods.

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::model;
use anyhow::Result;

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

    async fn mtp_depth(&mut self) -> Result<u32> {
        Ok(model::model().eta_caps().mtp_depth)
    }

    async fn draft_block(&mut self) -> Result<Option<pie::inferlet::model::BlockDrafter>> {
        let caps = model::model().eta_caps();
        Ok((caps.draft_block > 0).then(|| pie::inferlet::model::BlockDrafter {
            rows: caps.draft_block,
            mask_token: caps.draft_mask_token,
            bidirectional: caps.draft_bidirectional,
        }))
    }

    /// Which forward-pass interface the bound model requires, keyed on state
    /// semantics: recurrent state is present iff the engine handshake
    /// reports a non-zero folded-state size; paged KV is present iff the
    /// model has a KV page size.
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

    /// LM-head output dimension (`hf_config.vocab_size`), not the tokenizer
    /// vocab.
    async fn output_vocab_size(&mut self) -> Result<u32> {
        Ok(model::model().vocab_size())
    }

    async fn kv_page_size(&mut self) -> Result<u32> {
        Ok(model::model().kv_page_size())
    }

    /// Waves per frame (k) — the static deployment constant `forward.submit`
    /// sizes its slot list to. Fixed at runtime start, like `kv-page-size`.
    async fn frame_size(&mut self) -> Result<u32> {
        Ok(crate::scheduler::configured_frame_size() as u32)
    }

    /// Bound on how long a pipeline may hold a frame's wait-set without
    /// submitting. See `scheduler::configured_submit_deadline`.
    async fn submit_deadline_us(&mut self) -> Result<u64> {
        Ok(crate::scheduler::configured_submit_deadline().as_micros() as u64)
    }

    /// Host-reader channel capacity, in cells, that sustains the runtime's
    /// run-ahead for one lane. Includes the staging margin; see
    /// `scheduler::channel_capacity`.
    async fn channel_capacity(&mut self) -> Result<u32> {
        Ok(crate::scheduler::channel_capacity() as u32)
    }

    /// Max embed tokens in a single pass (C) — the guest-side prefill chunk
    /// budget, sourced from the bound engine's structural per-launch token
    /// capacity.
    async fn max_embed_length(&mut self) -> Result<u32> {
        Ok(crate::engine::get_spec(0)?.limits.max_forward_tokens as u32)
    }

    // working-set / arena capabilities, global over the bound model.

    /// Bytes of one folded recurrent-state object (0 if the model has no RS).
    async fn rs_state_size(&mut self) -> Result<u64> {
        Ok(model::model().rs_caps().state_size)
    }

    /// Tokens per buffered RS page (0 if the model has no RS).
    async fn rs_buffer_page_size(&mut self) -> Result<u32> {
        Ok(model::model().rs_caps().buffer_page_size)
    }

    /// Fold granularity in tokens; 1 = unconstrained. An RS fold of `n`
    /// tokens requires `n` to be a positive multiple of this.
    async fn rs_fold_granularity(&mut self) -> Result<u32> {
        Ok(model::model().rs_caps().fold_granularity)
    }

    /// KV page size (tokens) of the bound model.
    async fn arena_block_size(&mut self) -> Result<u64> {
        Ok(crate::store::registry::get(0, 0).kv_page_size as u64)
    }
}
