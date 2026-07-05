//! Self-consistency: generates multiple independent proposals in parallel,
//! each solving the question fully, then takes a majority vote over the
//! extracted final numeric answers.

use futures::future;
use inferlet::{
    Context, sample::Sampler, model::Model,
    runtime, Result,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_question")]
    question: String,
    #[serde(default = "default_proposal_tokens")]
    proposal_tokens: Vec<usize>,
    #[serde(default = "default_aggregation_tokens")]
    aggregation_tokens: usize, // kept for schema compat; unused by voting
}

fn default_question() -> String { "Calculate (42 + 3) * 5 / 15.".to_string() }
fn default_proposal_tokens() -> Vec<usize> { vec![256, 256, 256, 256] }
fn default_aggregation_tokens() -> usize { 256 }

const SYSTEM_PROMPT: &str = "You are a helpful, respectful and honest assistant.";

const SOLVE_PROMPT_TEMPLATE: &str = "\
Solve the following question step by step. \
End your response with a line in exactly this format (no extra text after it): \
Final answer: <number>\n\
Question: {}";

/// Strips <think>...</think> blocks (Qwen3 reasoning traces) from model output.
fn strip_think_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        match rest.find("<think>") {
            Some(start) => {
                result.push_str(&rest[..start]);
                match rest[start..].find("</think>") {
                    Some(end_rel) => {
                        let end = start + end_rel + "</think>".len();
                        rest = &rest[end..];
                    }
                    None => {
                        rest = "";
                        break;
                    }
                }
            }
            None => {
                result.push_str(rest);
                break;
            }
        }
    }
    result
}

/// Extracts the number following "Final answer:" (case-insensitive),
/// normalizing commas/whitespace. Returns None if no marker is found
/// or the trailing text doesn't parse as a number.
fn extract_final_answer(text: &str) -> Option<f64> {
    let cleaned = strip_think_blocks(text);
    let lower = cleaned.to_lowercase();
    let marker = "final answer:";
    let idx = lower.rfind(marker)?;
    let after = &cleaned[idx + marker.len()..];

    let first_line = after.lines().next().unwrap_or("").trim();
    let numeric: String = first_line
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();

    if numeric.is_empty() {
        None
    } else {
        numeric.parse::<f64>().ok()
    }
}

/// Majority vote over extracted answers. Ties broken by first-occurrence
/// order among the tied values. Proposals with no parseable answer are
/// excluded from voting entirely.
fn majority_vote(answers: &[Option<f64>]) -> Option<f64> {
    let mut counts: Vec<(f64, usize)> = Vec::new();

    for ans in answers.iter().flatten() {
        if let Some(entry) = counts.iter_mut().find(|(v, _)| (*v - *ans).abs() < 1e-9) {
            entry.1 += 1;
        } else {
            counts.push((*ans, 1));
        }
    }

    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(v, _)| v)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let question = input.question;
    let proposal_tokens = input.proposal_tokens;

    let start = Instant::now();
    println!(
        "--- Starting self-consistency for question: \"{}\" ---",
        question
    );
    println!("Proposal tokens: {:?}", proposal_tokens);

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let mut ctx_root = Context::new(&model)?;
    ctx_root.system(SYSTEM_PROMPT);
    ctx_root.flush().await?;

    let solve_prompt = SOLVE_PROMPT_TEMPLATE.replace("{}", &question);
    ctx_root.user(&solve_prompt);
    ctx_root.flush().await?;

    const BATCH_SIZE: usize = 4;
    let mut proposal_texts: Vec<String> = Vec::new();
    let mut total_tokens: usize = 0;
    for chunk in proposal_tokens.chunks(BATCH_SIZE) {
        let proposal_futures = chunk
            .iter()
            .map(|&max_tokens| {
                let mut ctx = ctx_root.fork()?;
                Ok(async move {
                    ctx.cue();
                    let (text, n_tokens) = ctx
                        .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
                        .max_tokens(max_tokens)
                        .collect_text_with_tokens()
                        .await?;
                    Ok::<_, String>((text, n_tokens))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let batch_results: Vec<(String, usize)> = future::join_all(proposal_futures)
            .await
            .into_iter()
            .collect::<Result<_>>()?;
        for (text, n_tokens) in batch_results {
            total_tokens += n_tokens;
            proposal_texts.push(text);
        }
    }

    let extracted: Vec<Option<f64>> = proposal_texts
        .iter()
        .map(|t| extract_final_answer(t))
        .collect();

    let mut vote_counts: HashMap<String, usize> = HashMap::new();
    for ans in extracted.iter().flatten() {
        *vote_counts.entry(format!("{:.6}", ans)).or_insert(0) += 1;
    }

    let winner = majority_vote(&extracted);

    println!("\n--- Self-consistency complete in {:?} ---\n", start.elapsed());
    for (i, (text, ans)) in proposal_texts.iter().zip(extracted.iter()).enumerate() {
        println!(
            "Proposal #{}: extracted = {:?}\n{}\n",
            i + 1,
            ans,
            strip_think_blocks(text)
        );
    }
    println!("Vote counts: {:?}", vote_counts);
    println!("Majority answer: {:?}", winner);
    println!("Total tokens generated: {}", total_tokens);

    let answer_str = winner
        .map(|w| format!("{}", w))
        .unwrap_or_else(|| "NO_CONSENSUS".to_string());
    Ok(format!("{{\"answer\": \"{}\", \"tokens\": {}}}", answer_str, total_tokens))
}
