//! Demonstrates output validation by computing normalized probabilities over candidate strings.
//!
//! This example shows how to evaluate the likelihood of different candidate outputs
//! given a context.

use inferlet::{Args, Context, Result, anyhow};
use std::time::Instant;

const HELP: &str = "\
Usage: output-validation [OPTIONS]

A program to validate and rank candidate outputs based on their generation probability.

Options:
  -h, --help  Prints this help message";

/// Calculates the normalized probability of a list of candidate strings being generated from a given context.
///
/// This function evaluates each candidate string by calculating its conditional probability
/// given the initial state of the context. It then normalizes these probabilities so that they
/// sum to 1, providing a clear distribution of likelihood over the candidates.
///
/// # Arguments
/// * `ctx`: The initial context from which generation probabilities are calculated.
/// * `candidates`: A slice of strings representing the possible outputs to validate.
///
/// # Returns
/// A `Vec<(String, f32)>` where each tuple contains a candidate string and its normalized probability.
pub async fn validate_outputs(ctx: &Context, candidates: &[String]) -> Vec<(String, f32)> {
    let mut log_probs = Vec::new();

    for candidate in candidates.iter() {
        let mut candidate_ctx = ctx.fork();
        let candidate_tokens = candidate_ctx.tokenizer.tokenize(candidate);
        let mut current_log_prob = 0.0f32;

        // Calculate the cumulative log probability for the candidate token sequence
        for &token_id in &candidate_tokens {
            let dist = candidate_ctx.decode_step_dist().await;

            // Find the probability of the actual next token in our candidate
            if let Some(index) = dist.ids.iter().position(|&id| id == token_id) {
                let prob = dist.probs[index];
                if prob > 0.0 {
                    current_log_prob += prob.ln();
                } else {
                    current_log_prob = -1000.0;
                    break;
                }
            } else {
                current_log_prob = -1000.0;
                break;
            }

            // Fill the context with the current token to prepare for the next step
            candidate_ctx.fill_token(token_id);
        }
        log_probs.push(current_log_prob);
    }

    // Normalize the probabilities
    // Find the maximum log probability for numerical stability (softmax trick)
    let max_log_prob = log_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if max_log_prob.is_infinite() {
        // If all probabilities are zero, return a uniform distribution.
        let uniform_prob = 1.0 / candidates.len() as f32;
        return candidates
            .iter()
            .map(|c| (c.clone(), uniform_prob))
            .collect();
    }

    // Convert log probabilities to standard probabilities and sum them up
    let mut total_prob = 0.0;
    let probs: Vec<f32> = log_probs
        .iter()
        .map(|&log_p| {
            let p = (log_p - max_log_prob).exp();
            total_prob += p;
            p
        })
        .collect();

    // Normalize to get the final distribution
    candidates
        .iter()
        .zip(probs.iter())
        .map(|(candidate, &p)| (candidate.clone(), p / total_prob))
        .collect()
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let start = Instant::now();
    let model = inferlet::get_auto_model();
    let mut ctx = model.create_context();

    if !model.get_name().starts_with("llama-3") {
        return Err(anyhow!(
            "Output validation example is only implemented for Llama 3 models. Got: {}",
            model.get_name()
        ));
    }

    // 1. Set up the initial context (the "prompt")
    let prompt = "The name of the person in the report is ";
    ctx.fill("<|begin_of_text|>");
    ctx.fill(
        "<|start_header_id|>system<|end_header_id|>\n\n\
        You are an expert at information extraction.<|eot_id|>",
    );
    ctx.fill(&format!(
        "<|start_header_id|>user<|end_header_id|>\n\n\
        From the sentence \"The financial report was prepared by David Chen.\", \
        extract the person's name.<|eot_id|>"
    ));
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");
    ctx.fill(prompt);
    ctx.flush().await;

    // 2. Define the list of candidate outputs to validate
    let candidates = vec![
        "John Smith".to_string(),
        "Mary Anne".to_string(),
        "David Chen".to_string(),
        "Chen David".to_string(),
    ];

    println!("--- Context ---\n'{}'\n\n--- Candidates ---", prompt);
    for c in &candidates {
        println!("- {}", c);
    }

    // 3. Call the validation function
    let results = validate_outputs(&ctx, &candidates).await;

    println!("\n--- Validation Results ---");
    for (candidate, probability) in results {
        println!(
            "- Candidate: {:<12} | Probability: {:.4}%",
            candidate,
            probability * 100.0
        );
    }

    println!("\nTotal elapsed: {:?}", start.elapsed());

    Ok(())
}
