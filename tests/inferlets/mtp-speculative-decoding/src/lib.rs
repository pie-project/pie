//! Speculative decoding with a model's native multi-token-prediction heads.
//!
//! Each round obtains one target-model seed token and `k` native MTP drafts,
//! verifies all drafts in one target forward, accepts the matching prefix, and
//! appends a target correction or bonus token. This readable reference rebuilds
//! KV state for each round, so rejected speculative state is never retained.

use inferlet::ptir::prelude::*;
use inferlet::{Result, chat, model as wit_model};
use serde::Deserialize;

const PAGE_T: u32 = 16;
const NUM_LAYERS: u32 = 28;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_k")]
    k: u32,
}

fn default_prompt() -> String {
    "The quick brown fox jumps over".into()
}

fn default_max_tokens() -> usize {
    64
}

fn default_k() -> u32 {
    4
}

async fn draft_and_verify_window(context: &[u32], k: u32) -> Result<(u32, Vec<u32>, Vec<u32>)> {
    if context.is_empty() {
        return Err("cannot draft from an empty context".into());
    }
    let n = context.len() as u32;
    let ws = WorkingSet::new();
    ws.reserve(n.div_ceil(PAGE_T))
        .map_err(|e| format!("reserve draft KV: {e}"))?;

    let tokens = Channel::from(
        context
            .iter()
            .map(|&token| token as i32)
            .collect::<Vec<_>>(),
    );
    let klen = Channel::from(vec![n]);
    let seed_out = Channel::new([1], dtype::i32).named("seed");
    let drafts_out = Channel::new([k], dtype::i32).named("drafts");

    let fwd = ForwardPass::new();
    fwd.embed(&tokens, Tensor::constant(vec![0u32, n]));
    fwd.attn_working_set(&ws, &klen);
    fwd.epilogue(|| {
        seed_out.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
        drafts_out.put(reduce_argmax(intrinsics::mtp_logits(k)));
    });

    let verify_total = n + 1 + k;
    let verify_rows = k + 1;
    let verify_tokens = Channel::writer([verify_total], dtype::i32).named("verify_tokens");
    let verify_klen = Channel::from(vec![verify_total]);
    let target_out = Channel::new([verify_rows], dtype::i32).named("targets");
    let verify_ws = WorkingSet::new();
    verify_ws
        .reserve(verify_total.div_ceil(PAGE_T))
        .map_err(|e| format!("reserve verify KV: {e}"))?;
    let verify = ForwardPass::new();
    verify.embed(&verify_tokens, Tensor::constant(vec![0u32, verify_total]));
    verify.attn_working_set(&verify_ws, &verify_klen);
    verify.readout(&Tensor::constant((n..n + verify_rows).collect::<Vec<_>>()));
    verify.epilogue(|| {
        target_out.put(reduce_argmax(intrinsics::logits()));
    });

    let pipeline = Pipeline::new();
    fwd.submit(&pipeline)
        .map_err(|e| format!("native MTP draft: {e}"))?;
    verify
        .submit(&pipeline)
        .map_err(|e| format!("verify MTP draft: {e}"))?;

    let seed = seed_out
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("read target seed: {e}"))?[0] as u32;
    let drafts = drafts_out
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("read MTP drafts: {e}"))?
        .into_iter()
        .map(|token| token as u32)
        .collect::<Vec<_>>();
    let mut verify_input = context.to_vec();
    verify_input.push(seed);
    verify_input.extend_from_slice(&drafts);
    verify_tokens.put(
        verify_input
            .iter()
            .map(|&token| token as i32)
            .collect::<Vec<_>>(),
    );
    let targets = target_out
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("read MTP verification: {e}"))?
        .into_iter()
        .map(|token| token as u32)
        .collect::<Vec<_>>();
    pipeline.close();
    Ok((seed, drafts, targets))
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if input.max_tokens == 0 {
        return Ok(String::new());
    }
    if !(1..=32).contains(&input.k) {
        return Err("k must be between 1 and 32".into());
    }

    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);
    model::configure_gates(true, false);

    let mut committed = chat::system_user("Continue the requested text.", &input.prompt);
    committed.extend(chat::cue());
    if committed.is_empty() {
        committed.push(0);
    }

    let stop_tokens = chat::stop_tokens();
    let mut generated = Vec::with_capacity(input.max_tokens);
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let mut rounds = 0usize;
    let mut stopped = false;

    while generated.len() < input.max_tokens && !stopped {
        let (seed, drafts, targets) = draft_and_verify_window(&committed, input.k).await?;
        committed.push(seed);
        if stop_tokens.contains(&seed) {
            break;
        }
        generated.push(seed);
        if generated.len() == input.max_tokens {
            break;
        }
        drafted += drafts.len();
        rounds += 1;
        if targets.len() != drafts.len() + 1 {
            return Err(format!(
                "verification returned {} targets for {} drafts",
                targets.len(),
                drafts.len()
            ));
        }

        let mut commit = Vec::new();
        let mut rejected = false;
        for (index, &draft) in drafts.iter().enumerate() {
            if targets[index] == draft {
                commit.push(draft);
                accepted += 1;
            } else {
                commit.push(targets[index]);
                rejected = true;
                break;
            }
        }
        if !rejected {
            commit.push(targets[drafts.len()]);
        }

        for token in commit {
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

    let acceptance_rate = if drafted == 0 {
        0.0
    } else {
        accepted as f64 / drafted as f64
    };
    eprintln!(
        "mtp-speculative-decoding: rounds={rounds} drafted={drafted} accepted={accepted} \
         acceptance_rate={acceptance_rate:.3}"
    );
    wit_model::decode(&generated)
}
