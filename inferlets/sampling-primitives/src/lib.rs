//! Reads the core sampling measurements from one forward pass.
//!
//! A single PTIR epilogue publishes the greedy token, raw logits, entropy,
//! probabilities, and log-probabilities. The host then cross-checks that the
//! independently useful outputs agree.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};
use serde::Deserialize;

#[derive(Deserialize, Default)]
struct Input {}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[inferlet::main]
async fn main(_input: Input) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    let ws = WorkingSet::new();
    model::configure(vocab, ws.page_size(), 1);

    let mut prompt = wit_model::encode("The capital of France is");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;

    let toks = Channel::from(prompt.iter().map(|&token| token as i32).collect::<Vec<_>>());
    let klen = Channel::from(vec![n]);
    let token_out = Channel::new([1], dtype::i32).named("token");
    let logits_out = Channel::new([vocab], dtype::f32).named("logits");
    let entropy_out = Channel::new([1], dtype::f32).named("entropy");
    let probs_out = Channel::new([vocab], dtype::f32).named("probabilities");
    let logprobs_out = Channel::new([vocab], dtype::f32).named("log_probabilities");

    let fwd = ForwardPass::new();
    fwd.embed(&toks, Tensor::constant(vec![0u32, n]));
    fwd.attn_working_set(&ws, &klen);
    fwd.epilogue(|| {
        let logits = intrinsics::logits();
        let logprobs = log_softmax(&logits);
        let probabilities = exp(&logprobs);
        let entropy = reshape(neg(reduce_sum(mul(&probabilities, &logprobs))), [1]);
        let token = reshape(reduce_argmax(&logits), [1]);

        token_out.put(&token);
        logits_out.put(&logits);
        entropy_out.put(&entropy);
        probs_out.put(&probabilities);
        logprobs_out.put(&logprobs);
    });

    let pipeline = Pipeline::new();
    fwd.submit(&pipeline)
        .map_err(|e| format!("sampling-primitives submit: {e}"))?;

    let token = token_out
        .take()
        .get::<i32>()
        .map_err(|e| format!("read token: {e}"))?[0] as usize;
    let logits = logits_out
        .take()
        .get::<f32>()
        .map_err(|e| format!("read logits: {e}"))?;
    let entropy = entropy_out
        .take()
        .get::<f32>()
        .map_err(|e| format!("read entropy: {e}"))?[0];
    let probabilities = probs_out
        .take()
        .get::<f32>()
        .map_err(|e| format!("read probabilities: {e}"))?;
    let log_probabilities = logprobs_out
        .take()
        .get::<f32>()
        .map_err(|e| format!("read log-probabilities: {e}"))?;
    pipeline.close();

    if token != argmax(&logits) || token != argmax(&probabilities) {
        return Err("token, logits, and probability argmax values disagree".into());
    }
    if probabilities.len() != vocab as usize || log_probabilities.len() != vocab as usize {
        return Err("sampling output has the wrong vocabulary dimension".into());
    }
    if !entropy.is_finite() || entropy < 0.0 || entropy > (vocab as f32).ln() + 1e-3 {
        return Err(format!("entropy is outside the valid range: {entropy}"));
    }
    let token_logprob = log_probabilities[token];
    if (token_logprob.exp() - probabilities[token]).abs() > 1e-4 {
        return Err("probability and log-probability outputs disagree".into());
    }

    Ok(format!(
        "token={token} probability={:.6} log_probability={token_logprob:.6} entropy={entropy:.6}",
        probabilities[token]
    ))
}
