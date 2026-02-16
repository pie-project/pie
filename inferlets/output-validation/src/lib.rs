//! Demonstrates output validation by computing normalized probabilities over candidate strings.
//!
//! This example shows how to evaluate the likelihood of different candidate outputs
//! given a context, using the ForwardPass API with distribution sampling.

use inferlet::{
    context::Context, model::Model, runtime,
    ContextExt, InstructExt, ForwardPassExt, Result,
    inference::{ForwardPass, Output, Sampler},
};
use std::time::Instant;

const HELP: &str = "\
Usage: output-validation [OPTIONS]

A program to validate and rank candidate outputs based on their generation probability.

Options:
  -h, --help  Prints this help message";

/// Calculates the normalized probability of a list of candidate strings being generated
/// from a given context.
///
/// Uses ForwardPass with Sampler::Dist to retrieve token probability distributions,
/// then scores each candidate by its cumulative log probability.
pub async fn validate_outputs(
    model: &Model,
    ctx: &Context,
    candidates: &[String],
) -> Result<Vec<(String, f32)>> {
    let tokenizer = model.tokenizer();
    let mut log_probs = Vec::new();

    for (i, candidate) in candidates.iter().enumerate() {
        let candidate_ctx = ctx.fork(&format!("candidate-{}", i))?;

        let candidate_tokens = tokenizer.encode(candidate);
        let mut current_log_prob = 0.0f32;

        // Calculate the cumulative log probability for the candidate token sequence
        for &token_id in &candidate_tokens {
            // Use ForwardPass with Dist sampler to get probability distributions
            let seq_len = candidate_ctx.last_position().map(|p| p + 1).unwrap_or(0);
            let buffered = candidate_ctx.buffered_tokens();
            if buffered.is_empty() {
                current_log_prob = -1000.0;
                break;
            }

            let page_size = candidate_ctx.tokens_per_page();
            let cursor = candidate_ctx.cursor();
            let total_tokens_after = cursor + buffered.len() as u32;
            let total_pages_needed = (total_tokens_after + page_size - 1) / page_size;
            if total_pages_needed > 0 {
                candidate_ctx.reserve_pages(total_pages_needed)
                    .map_err(|e| format!("Failed to reserve pages: {}", e))?;
            }

            let pass = ForwardPass::new(model);
            pass.context(&candidate_ctx);
            let positions: Vec<u32> = (seq_len..seq_len + buffered.len() as u32).collect();
            pass.input_tokens(&buffered, &positions);

            // Sample distributions at the last token position
            let last_idx = (buffered.len() - 1) as u32;
            pass.sampler(&[last_idx], Sampler::Dist((0.0, 0)));

            let output = pass.execute_async().await
                .map_err(|e| format!("Forward pass failed: {}", e))?;

            // Commit pages and update cursor
            let new_cursor_abs = cursor + buffered.len() as u32;
            let pages_to_commit = new_cursor_abs / page_size;
            if pages_to_commit > 0 {
                let page_indices: Vec<u32> = (0..pages_to_commit).collect();
                candidate_ctx.commit_pages(&page_indices)
                    .map_err(|e| format!("Failed to commit pages: {}", e))?;
            }
            candidate_ctx.set_cursor(new_cursor_abs % page_size);

            // Extract distribution and find the probability of the target token
            if let Output::Distributions(dists) = output {
                if let Some((ids, probs)) = dists.first() {
                    if let Some(index) = ids.iter().position(|&id| id == token_id) {
                        let prob = probs[index];
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
                }
            }

            // Feed the current token for the next step
            candidate_ctx.set_buffered_tokens(&[token_id]);
        }
        log_probs.push(current_log_prob);
    }

    // Normalize the probabilities
    let max_log_prob = log_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if max_log_prob.is_infinite() {
        let uniform_prob = 1.0 / candidates.len() as f32;
        return Ok(candidates
            .iter()
            .map(|c| (c.clone(), uniform_prob))
            .collect());
    }

    let mut total_prob = 0.0;
    let probs: Vec<f32> = log_probs
        .iter()
        .map(|&log_p| {
            let p = (log_p - max_log_prob).exp();
            total_prob += p;
            p
        })
        .collect();

    Ok(candidates
        .iter()
        .zip(probs.iter())
        .map(|(candidate, &p)| (candidate.clone(), p / total_prob))
        .collect())
}

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let start = Instant::now();
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    let ctx = Context::create(&model, "validation", None)?;

    // Set up the initial context using InstructExt
    ctx.system("You are an expert at information extraction.");
    ctx.user(
        "From the sentence \"The financial report was prepared by David Chen.\", \
        extract the person's name.",
    );
    ctx.cue();

    let prompt = "The name of the person in the report is ";
    ctx.fill_tokens(&model.tokenizer().encode(prompt));
    ctx.flush().await?;

    // Define the list of candidate outputs to validate
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

    // Call the validation function
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
