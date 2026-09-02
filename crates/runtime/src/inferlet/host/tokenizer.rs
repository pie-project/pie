//! pie:inferlet/tokenizer - Tokenizer global functions.
//!
//! The runtime serves exactly one model, so these are free functions over
//! the single global [`crate::model::Model`] rather than resource methods.

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::model;
use anyhow::Result;

/// `model` still hands token tables back as two parallel vectors; the WIT
/// surface names the pairing instead.
fn token_table((ids, bytes): (Vec<u32>, Vec<Vec<u8>>)) -> Vec<pie::inferlet::tokenizer::Token> {
    ids.into_iter()
        .zip(bytes)
        .map(|(id, bytes)| pie::inferlet::tokenizer::Token { id, bytes })
        .collect()
}

/// Above this size, tokenizer work moves to the blocking pool. Inline
/// (de)tokenization holds an async worker for the call's full duration;
/// small calls stay inline since spawn_blocking costs ~5-10 µs, which
/// would be pure overhead on per-token streaming decodes.
const TOKENIZER_OFFLOAD_THRESHOLD: usize = 64;

impl pie::inferlet::tokenizer::Host for ProcessCtx {
    async fn encode(&mut self, text: String) -> Result<Vec<u32>> {
        if text.len() >= TOKENIZER_OFFLOAD_THRESHOLD * 4 {
            return Ok(tokio::task::spawn_blocking(move || model::model().tokenize(&text)).await?);
        }
        let ids = model::model().tokenize(&text);
        Ok(ids)
    }

    async fn decode(&mut self, tokens: Vec<u32>) -> Result<Result<String, String>> {
        if tokens.len() >= TOKENIZER_OFFLOAD_THRESHOLD {
            return Ok(Ok(tokio::task::spawn_blocking(move || {
                model::model().detokenize(&tokens)
            })
            .await?));
        }
        Ok(Ok(model::model().detokenize(&tokens)))
    }

    /// The whole table: a quarter million records is ~20 ms, so it goes to
    /// the blocking pool unconditionally, unlike the threshold above. Built
    /// once per runtime ([`crate::model::Model::get_vocabs`]); what is left
    /// is the copy the WIT surface's owned list requires.
    async fn vocabs(&mut self) -> Result<Vec<pie::inferlet::tokenizer::Token>> {
        Ok(tokio::task::spawn_blocking(|| token_table(model::model().get_vocabs())).await?)
    }

    /// Inline, not gated by the threshold: a caller asks for the tokens it
    /// rolled back, which is one or two.
    async fn token_bytes(&mut self, tokens: Vec<u32>) -> Result<Vec<Vec<u8>>> {
        Ok(model::model().token_bytes(&tokens))
    }

    /// The byte-prefix query token healing needs. One pass over the cached
    /// table, on the blocking pool like a long detokenize.
    async fn tokens_with_prefix(&mut self, prefix: Vec<u8>) -> Result<Vec<u32>> {
        Ok(
            tokio::task::spawn_blocking(move || model::model().tokens_with_prefix(&prefix))
                .await?,
        )
    }

    async fn split_regex(&mut self) -> Result<String> {
        Ok(model::model().get_split_regex())
    }

    async fn special_tokens(&mut self) -> Result<Vec<pie::inferlet::tokenizer::Token>> {
        Ok(token_table(model::model().get_special_tokens()))
    }
}
