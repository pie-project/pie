use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::model;
use anyhow::Result;

fn token_table((ids, bytes): (Vec<u32>, Vec<Vec<u8>>)) -> Vec<pie::inferlet::tokenizer::Token> {
    ids.into_iter()
        .zip(bytes)
        .map(|(id, bytes)| pie::inferlet::tokenizer::Token { id, bytes })
        .collect()
}

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

    async fn vocabs(&mut self) -> Result<Vec<pie::inferlet::tokenizer::Token>> {
        Ok(tokio::task::spawn_blocking(|| token_table(model::model().get_vocabs())).await?)
    }

    async fn token_bytes(&mut self, tokens: Vec<u32>) -> Result<Vec<Vec<u8>>> {
        Ok(model::model().token_bytes(&tokens))
    }

    async fn tokens_with_prefix(&mut self, prefix: Vec<u8>) -> Result<Vec<u32>> {
        Ok(tokio::task::spawn_blocking(move || model::model().tokens_with_prefix(&prefix)).await?)
    }

    async fn split_regex(&mut self) -> Result<String> {
        Ok(model::model().get_split_regex())
    }

    async fn special_tokens(&mut self) -> Result<Vec<pie::inferlet::tokenizer::Token>> {
        Ok(token_table(model::model().get_special_tokens()))
    }
}
