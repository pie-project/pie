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
