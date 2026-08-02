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

/// Above this size, tokenizer work moves to the blocking pool. Inline
/// (de)tokenization holds an async worker for the call's full duration —
/// a 512-token detokenize is ~ms-scale, and completions arrive in bursts,
/// so the workers all fill exactly when a planner-granted lane needs to
/// resume its submit (measured serve→collect p90 ~7 ms on the D cell).
/// Small calls stay inline: spawn_blocking costs ~5-10 µs, which would be
/// pure overhead on per-token streaming decodes.
const TOKENIZER_OFFLOAD_THRESHOLD: usize = 64;

impl pie::inferlet::tokenizer::Host for ProcessCtx {
    async fn encode(&mut self, text: String) -> Result<Vec<u32>> {
        if text.len() >= TOKENIZER_OFFLOAD_THRESHOLD * 4 {
            return Ok(tokio::task::spawn_blocking(move || {
                model::model().tokenize(&text)
            })
            .await?);
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
