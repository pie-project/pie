//! Demonstrates output validation by computing normalized probabilities over
//! a fixed candidate list, given a shared context prefix.
//!
//! For each candidate, fork the context and walk one token at a time,
//! reading the per-position distribution via `Probe::Distribution` and
//! accumulating log probability. The cumulative log-prob sequence is then
//! softmax-normalized across candidates.

use inferlet::{
    Context, Result,
    model::Model,
    runtime,
    sample::Distribution,
};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {}

/// Calculate the normalized probability of each candidate string being
/// generated from the given context.
pub async fn validate_outputs(
    model: &Model,
    base: &Context,
    candidates: &[String],
) -> Result<Vec<(String, f32)>> {
    let tokenizer = model.tokenizer();
    let mut log_probs = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        let mut ctx = base.fork()?;
        let candidate_tokens = tokenizer.encode(candidate);

        // Bootstrap: feed an empty marker so the model produces a
        // distribution at the candidate's first token position. We use the
        // last token of the empty-string encoding as a no-op anchor.
        let mut pending = vec![*tokenizer.encode("").last().unwrap_or(&0)];
        let mut cumulative_log_prob = 0.0f32;

        for &target in &candidate_tokens {
            let mut pass = ctx.forward();
            pass.input(&pending);
            // Probe the distribution at the LAST input position (the slot
            // index is local to the auto-input window: `len-1`).
            let last_idx = (pending.len() - 1) as u32;
            // `k = 0` returns the full vocabulary so we can look up `target`.
            let h = pass.probe(last_idx, Distribution { temperature: 0.0, k: 0 });

            let out = pass.execute().await?;
            let (ids, probs) = out
                .distribution(h)
                .ok_or("Distribution probe missing")?;

            let p = ids
                .iter()
                .position(|&id| id == target)
                .map(|i| probs[i])
                .unwrap_or(0.0);

            if p > 0.0 {
                cumulative_log_prob += p.ln();
            } else {
                cumulative_log_prob = -1000.0;
                break;
            }

            // Feed the realized token forward for the next step.
            pending = vec![target];
        }
        log_probs.push(cumulative_log_prob);
    }

    let max_log_prob = log_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_log_prob.is_infinite() {
        let uniform = 1.0 / candidates.len() as f32;
        return Ok(candidates.iter().map(|c| (c.clone(), uniform)).collect());
    }

    let mut total = 0.0f32;
    let raw: Vec<f32> = log_probs
        .iter()
        .map(|&lp| {
            let p = (lp - max_log_prob).exp();
            total += p;
            p
        })
        .collect();

    Ok(candidates
        .iter()
        .zip(raw.iter())
        .map(|(c, &p)| (c.clone(), p / total))
        .collect())
}

#[inferlet::main]
async fn main(_input: Input) -> Result<String> {
    let start = Instant::now();
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    let mut ctx = Context::new(&model)?;
    ctx.system("You are an expert at information extraction.")
        .user(
            "From the sentence \"The financial report was prepared by David Chen.\", \
             extract the person's name.",
        )
        .cue();

    let prompt_tail = "The name of the person in the report is ";
    ctx.append(&model.tokenizer().encode(prompt_tail));
    ctx.flush().await?;

    let candidates = vec![
        "John Smith".to_string(),
        "Mary Anne".to_string(),
        "David Chen".to_string(),
        "Chen David".to_string(),
    ];

    println!("--- Context ---\n'{}'\n\n--- Candidates ---", prompt_tail);
    for c in &candidates {
        println!("- {}", c);
    }

    let results = validate_outputs(&model, &ctx, &candidates).await?;

    println!("\n--- Validation Results ---");
    for (candidate, probability) in results {
        println!(
            "- Candidate: {:<12} | Probability: {:.4}%",
            candidate,
            probability * 100.0
        );
    }

    println!("\nTotal elapsed: {:?}", start.elapsed());
    Ok(String::new())
}
