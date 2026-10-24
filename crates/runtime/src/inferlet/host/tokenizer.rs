use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::model;
use anyhow::Result;

fn token_table<'a>(
    tokens: impl Iterator<Item = (u32, &'a [u8])>,
) -> Vec<pie::inferlet::tokenizer::Token> {
    tokens
        .map(|(id, bytes)| pie::inferlet::tokenizer::Token {
            id,
            bytes: bytes.to_vec(),
        })
        .collect()
}

const TOKENIZER_OFFLOAD_THRESHOLD: usize = 64;

impl pie::inferlet::tokenizer::Host for ProcessCtx {
    async fn encode(&mut self, text: String) -> Result<Vec<u32>> {
        if text.len() >= TOKENIZER_OFFLOAD_THRESHOLD * 4 {
            return Ok(crate::rt::spawn_blocking(move || model::model().tokenize(&text)).await?);
        }
        let ids = model::model().tokenize(&text);
        Ok(ids)
    }

    async fn decode(&mut self, tokens: Vec<u32>) -> Result<Result<String, String>> {
        if tokens.len() >= TOKENIZER_OFFLOAD_THRESHOLD {
            return Ok(Ok(crate::rt::spawn_blocking(move || {
                model::model().detokenize(&tokens)
            })
            .await?));
        }
        Ok(Ok(model::model().detokenize(&tokens)))
    }

    async fn vocabs(&mut self) -> Result<Vec<pie::inferlet::tokenizer::Token>> {
        Ok(crate::rt::spawn_blocking(|| token_table(model::model().vocabs())).await?)
    }

    async fn token_bytes(&mut self, tokens: Vec<u32>) -> Result<Vec<Vec<u8>>> {
        Ok(model::model().token_bytes(&tokens))
    }

    async fn tokens_with_prefix(&mut self, prefix: Vec<u8>) -> Result<Vec<u32>> {
        Ok(crate::rt::spawn_blocking(move || model::model().tokens_with_prefix(&prefix)).await?)
    }

    async fn split_regex(&mut self) -> Result<String> {
        // a guest reads "no single regex" as the empty string
        Ok(model::model().split_regex().unwrap_or_default().to_owned())
    }

    async fn special_tokens(&mut self) -> Result<Vec<pie::inferlet::tokenizer::Token>> {
        Ok(token_table(model::model().special_tokens()))
    }
}
