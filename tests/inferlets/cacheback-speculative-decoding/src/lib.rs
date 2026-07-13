//! CacheBack-style speculative decoding with a prompt-lookup draft cache.
//!
//! The drafter searches the committed token history for the longest matching
//! suffix and reuses the tokens that followed its previous occurrence. One
//! target-model forward verifies the whole draft and supplies a correction or
//! bonus token. This reference implementation rebuilds the target KV for each
//! verification window so rejected draft state can never leak into later steps.

use inferlet::ptir::prelude::*;
use inferlet::{Result, chat, model as wit_model};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_draft_length")]
    draft_length: usize,
    #[serde(default = "default_max_ngram")]
    max_ngram: usize,
}

fn default_prompt() -> String {
    "Repeat this pattern: red green blue, red green blue, red green".into()
}

fn default_max_tokens() -> usize {
    256
}

fn default_draft_length() -> usize {
    4
}

fn default_max_ngram() -> usize {
    8
}

fn draft_from_cache(tokens: &[u32], draft_length: usize, max_ngram: usize) -> Vec<u32> {
    if draft_length == 0 || tokens.len() < 2 {
        return Vec::new();
    }

    let max_match = max_ngram.min(tokens.len() - 1);
    for width in (1..=max_match).rev() {
        let suffix = &tokens[tokens.len() - width..];
        for start in (0..tokens.len() - width).rev() {
            if &tokens[start..start + width] != suffix {
                continue;
            }
            let continuation = start + width;
            let end = (continuation + draft_length).min(tokens.len());
            if continuation < end {
                return tokens[continuation..end].to_vec();
            }
        }
    }
    Vec::new()
}

async fn verify(committed: &[u32], draft: &[u32], page_size: u32) -> Result<Vec<u32>> {
    if committed.is_empty() {
        return Err("cannot verify from an empty committed sequence".into());
    }

    let mut input = committed.to_vec();
    input.extend_from_slice(draft);
    let total = input.len() as u32;
    let rows = draft.len() as u32 + 1;
    let readout_start = committed.len() as u32 - 1;
    let readout = (readout_start..readout_start + rows).collect::<Vec<_>>();

    let ws = WorkingSet::new();
    ws.reserve(total.div_ceil(page_size))
        .map_err(|e| format!("reserve verification KV: {e}"))?;
    let tokens = Channel::from(input.iter().map(|&token| token as i32).collect::<Vec<_>>());
    let klen = Channel::from(vec![total]);
    let target_out = Channel::new([rows], dtype::i32).named("target_tokens");

    let fwd = ForwardPass::new();
    fwd.embed(&tokens, Tensor::constant(vec![0u32, total]));
    fwd.attn_working_set(&ws, &klen);
    fwd.readout(&Tensor::constant(readout));
    fwd.epilogue(|| {
        target_out.put(reduce_argmax(intrinsics::logits()));
    });

    let pipeline = Pipeline::new();
    fwd.submit(&pipeline)
        .map_err(|e| format!("verify cached draft: {e}"))?;
    let target = target_out
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("read verification result: {e}"))?
        .into_iter()
        .map(|token| token as u32)
        .collect();
    pipeline.close();
    Ok(target)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if input.max_tokens == 0 {
        return Ok(String::new());
    }

    let vocab = wit_model::output_vocab_size();
    let probe_ws = WorkingSet::new();
    let page_size = probe_ws.page_size();
    model::configure(vocab, page_size, 1);

    let mut committed = chat::system_user("Continue the requested text.", &input.prompt);
    committed.extend(chat::cue());
    if committed.is_empty() {
        committed.push(0);
    }
    let prompt_len = committed.len();
    let stop_tokens = chat::stop_tokens();
    let mut generated = Vec::with_capacity(input.max_tokens);
    let mut total_drafted = 0usize;
    let mut total_accepted = 0usize;
    let mut verification_steps = 0usize;
    let mut stopped = false;

    while generated.len() < input.max_tokens && !stopped {
        let draft = draft_from_cache(&committed, input.draft_length, input.max_ngram);
        total_drafted += draft.len();
        verification_steps += 1;

        let target = verify(&committed, &draft, page_size).await?;
        if target.len() != draft.len() + 1 {
            return Err(format!(
                "verification returned {} tokens for a {}-token draft",
                target.len(),
                draft.len()
            ));
        }

        let mut accepted = Vec::new();
        let mut rejected = false;
        for (index, &draft_token) in draft.iter().enumerate() {
            if target[index] == draft_token {
                accepted.push(draft_token);
                total_accepted += 1;
            } else {
                accepted.push(target[index]);
                rejected = true;
                break;
            }
        }
        if !rejected {
            accepted.push(target[draft.len()]);
        }

        for token in accepted {
            if stop_tokens.contains(&token) {
                stopped = true;
                break;
            }
            committed.push(token);
            generated.push(token);
            if generated.len() == input.max_tokens {
                break;
            }
        }
    }

    let acceptance_rate = if total_drafted == 0 {
        0.0
    } else {
        total_accepted as f64 / total_drafted as f64
    };
    eprintln!(
        "cacheback: prompt_tokens={prompt_len} generated={} verification_steps={verification_steps} \
         drafted={total_drafted} accepted={total_accepted} acceptance_rate={acceptance_rate:.3}",
        generated.len()
    );
    wit_model::decode(&generated)
}
