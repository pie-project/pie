//! Demonstrates Best-of-N generation with diversity ranking.
//!
//! This example forks a context N times to generate N candidate responses
//! in parallel, then uses the `strsim` library to compute pairwise
//! similarity and select the most central (consensus) answer.

use futures::future;
use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Result, Sampler};
use std::time::Instant;

const HELP: &str = "\
Usage: best-of-n [OPTIONS]

Generates N candidate responses in parallel and selects the most
central one using string similarity (self-consistency).

Options:
  -q, --question <TEXT>        The question to answer
                               [default: What is 17 * 24 + 13?]
  -n, --num-candidates <N>     Number of parallel candidates [default: 5]
  -t, --max-tokens <N>         Max tokens per candidate [default: 1024]
  -h, --help                   Prints help information";

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant that solves problems step by step. \
Show your reasoning, then give your final answer on the last line \
in the format: Final Answer: <answer>";

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let question: String = args
        .value_from_str(["-q", "--question"])
        .unwrap_or_else(|_| "What is 17 * 24 + 13?".to_string());
    let num_candidates: usize = args.value_from_str(["-n", "--num-candidates"]).unwrap_or(5);
    let max_tokens: usize = args.value_from_str(["-t", "--max-tokens"]).unwrap_or(1024);

    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();
    let mut base_ctx = model.create_context();

    base_ctx.fill_system(SYSTEM_PROMPT);
    base_ctx.fill_user(&question);
    base_ctx.flush().await;

    let stop_condition =
        stop_condition::max_len(max_tokens).or(stop_condition::ends_with_any(eos_tokens));

    // --- Stage 1: Generate N candidates in parallel ---
    println!(
        "--- Generating {} candidates in parallel ---",
        num_candidates
    );

    let candidate_futures = (0..num_candidates).map(|_| {
        let mut ctx = base_ctx.fork();
        let stop_cond = stop_condition.clone();
        async move { ctx.generate(Sampler::top_p(0.6, 0.95), stop_cond).await }
    });

    let candidates: Vec<String> = future::join_all(candidate_futures).await;

    let generation_time = start.elapsed();
    println!(
        "Generated {} candidates in {:?}\n",
        candidates.len(),
        generation_time
    );

    // --- Stage 2: Extract final answers ---
    let answers: Vec<&str> = candidates.iter().map(|c| extract_final_answer(c)).collect();

    println!("--- Extracted Answers ---\n");
    for (i, answer) in answers.iter().enumerate() {
        println!("  Candidate {}: \"{}\"", i + 1, truncate(answer, 80));
    }
    println!();

    // --- Stage 3: Compute pairwise similarity on extracted answers ---
    println!("--- Computing pairwise similarity ---");

    let n = candidates.len();
    let mut similarity_matrix = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = strsim::normalized_levenshtein(answers[i], answers[j]);
            similarity_matrix[i][j] = sim;
            similarity_matrix[j][i] = sim;
        }
        similarity_matrix[i][i] = 1.0;
    }

    // --- Stage 4: Rank by centrality ---
    let centrality_scores: Vec<f64> = (0..n)
        .map(|i| {
            if n == 1 {
                return 1.0; // Single candidate is trivially the best
            }
            let sum: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| similarity_matrix[i][j])
                .sum();
            sum / (n - 1) as f64
        })
        .collect();

    let best_idx = centrality_scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    // --- Print results ---
    println!("--- Candidate Rankings ---\n");
    let mut ranked: Vec<(usize, f64)> = centrality_scores.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (rank, (idx, score)) in ranked.iter().enumerate() {
        let marker = if *idx == best_idx { " <-- BEST" } else { "" };
        println!(
            "  #{} (candidate {}, centrality: {:.4}){}\n     answer: \"{}\"",
            rank + 1,
            idx + 1,
            score,
            marker,
            truncate(answers[*idx], 80)
        );
    }

    println!("\n--- Consensus Answer (candidate {}) ---", best_idx + 1);
    println!("Final Answer: {}", answers[best_idx]);
    println!("\n--- Full Response ---");
    println!("{}", candidates[best_idx]);
    println!("\nTotal elapsed: {:?}", start.elapsed());

    Ok(())
}

/// Extracts the text after the last occurrence of "Final Answer:" in the response.
/// Falls back to the full text (trimmed) if the marker is not found.
fn extract_final_answer(response: &str) -> &str {
    response
        .rfind("Final Answer:")
        .map(|pos| response[pos + "Final Answer:".len()..].trim())
        .unwrap_or_else(|| response.trim())
}

/// Truncates a string to at most `max_len` characters, appending "..." if truncated.
fn truncate(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max_len {
        s
    } else {
        let truncated: String = s.chars().take(max_len).collect();
        format!("{}...", truncated)
    }
}
