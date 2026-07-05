//! pie:inferlet/tokenizer - Tokenizer global functions.
//!
//! Split from `model` (§2.2): the engine serves exactly one model, so these are
//! free functions over the single global [`pie_model::Model`] rather than
//! resource methods.

use crate::api::pie;
use crate::inferlet::ProcessCtx;
use anyhow::Result;
use pie_model as model;

impl pie::inferlet::tokenizer::Host for ProcessCtx {
    async fn encode(&mut self, text: String) -> Result<Vec<u32>> {
        let ids = model::model().tokenize(&text);
        Ok(ids)
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
