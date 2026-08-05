//! pie:inferlet/tokenizer - Tokenizer global functions.
//!
//! Split from `model` (§2.2): the engine serves exactly one model, so these are
//! free functions over the single global [`crate::model::Model`] rather than
//! resource methods.

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use anyhow::Result;
use crate::model;

/// `pie_model` still hands token tables back as two parallel vectors; the WIT
/// surface names the pairing instead.
fn token_table((ids, bytes): (Vec<u32>, Vec<Vec<u8>>)) -> Vec<pie::inferlet::tokenizer::Token> {
    ids.into_iter()
        .zip(bytes)
        .map(|(id, bytes)| pie::inferlet::tokenizer::Token { id, bytes })
        .collect()
}

impl pie::inferlet::tokenizer::Host for ProcessCtx {
    async fn encode(&mut self, text: String) -> Result<Vec<u32>> {
        let ids = model::model().tokenize(&text);
        Ok(ids)
    }

    async fn decode(&mut self, tokens: Vec<u32>) -> Result<Result<String, String>> {
        Ok(Ok(model::model().detokenize(&tokens)))
    }

    async fn vocabs(&mut self) -> Result<Vec<pie::inferlet::tokenizer::Token>> {
        Ok(token_table(model::model().get_vocabs()))
    }

    async fn split_regex(&mut self) -> Result<String> {
        Ok(model::model().get_split_regex())
    }

    async fn special_tokens(&mut self) -> Result<Vec<pie::inferlet::tokenizer::Token>> {
        Ok(token_table(model::model().get_special_tokens()))
    }
}
