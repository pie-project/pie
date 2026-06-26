//! Demonstrates Graph-of-Thought (GoT) for hierarchical aggregation.
//!
//! This example generates multiple initial proposals concurrently, then
//! progressively aggregates them in pairs across multiple levels. The streaming
//! nature allows aggregation to begin as soon as pairs of proposals are ready,
//! maximizing parallelism.

use futures::stream::FuturesUnordered;
use futures::{StreamExt, future};
use inferlet::{Context, Result, sample::Sampler};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_question")]
    question: String,
    #[serde(default = "default_proposal_tokens")]
    proposal_tokens: Vec<usize>,
    #[serde(default = "default_aggregation_tokens")]
    aggregation_tokens: usize,
}

fn default_question() -> String { "Calculate (42 + 3) * 5 / 15.".to_string() }
fn default_proposal_tokens() -> Vec<usize> { vec![256, 256, 256, 256, 256, 256, 256, 256] }
fn default_aggregation_tokens() -> usize { 256 }

const SYSTEM_PROMPT: &str = "You are a helpful, respectful and honest assistant.";

const PROPOSAL_PROMPT_TEMPLATE: &str = "\
Could you suggest a method or approach to solve the following question? \
Please provide a high-level plan without doing the actual calculation. \
Keep it concise, around 80 words. Question: {}";

const AGGREGATE_PROMPT: &str = "\
Please compare the following solution with the one you just provided \
and aggregate their ideas into a single, improved solution:\n";


/// Main logic for running the hierarchical aggregation workflow.
async fn run_hierarchical_aggregation(
    base_context: &mut Context,
    question: &str,
    proposal_tokens: Vec<usize>,
    aggregation_tokens: usize,
) -> Result<Vec<String>> {
    // --- Stage 1: Generate Initial Proposals ---
    let propose_prompt = PROPOSAL_PROMPT_TEMPLATE.replace("{}", question);
    base_context.user(&propose_prompt);
    base_context.flush().await?;

    let mut proposal_tasks = proposal_tokens
        .into_iter()
        .map(|max_tokens| {
            let mut ctx = base_context.fork()?;
            Ok(async move {
                ctx.cue();
                let proposal_text = ctx
                    .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
                    .max_tokens(max_tokens)
                    .collect_text()
                    .await?;
                Ok::<_, String>((proposal_text, ctx))
            })
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .collect::<FuturesUnordered<_>>();

    // --- Stage 2: First-Level Aggregation (Pairing Proposals) ---
    let mut first_aggregation_tasks = FuturesUnordered::new();
    let mut pending_proposal: Option<(String, Context)> = None;

    while let Some(result) = proposal_tasks.next().await {
        let (proposal_text, mut proposal_ctx) = result?;
        if pending_proposal.is_none() {
            pending_proposal = Some((proposal_text, proposal_ctx));
        } else {
            let (previous_proposal_text, _) = pending_proposal.take().unwrap();
            let aggregation_prompt = format!("{}{}", AGGREGATE_PROMPT, previous_proposal_text);
            proposal_ctx.user(&aggregation_prompt);
            proposal_ctx.cue();

            first_aggregation_tasks.push(async move {
                let aggregation_text = proposal_ctx
                    .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
                    .max_tokens(aggregation_tokens)
                    .collect_text()
                    .await?;
                Ok::<_, String>((aggregation_text, proposal_ctx))
            });
        }
    }

    // --- Stage 3: Second-Level Aggregation (Pairing Aggregations) ---
    let mut second_aggregation_tasks = Vec::new();
    let mut pending_aggregation: Option<(String, Context)> = None;

    while let Some(result) = first_aggregation_tasks.next().await {
        let (aggregation_text, mut aggregation_ctx) = result?;
        if pending_aggregation.is_none() {
            pending_aggregation = Some((aggregation_text, aggregation_ctx));
        } else {
            let (previous_aggregation_text, _) = pending_aggregation.take().unwrap();
            let final_prompt = format!("{}{}", AGGREGATE_PROMPT, previous_aggregation_text);
            aggregation_ctx.user(&final_prompt);
            aggregation_ctx.cue();

            second_aggregation_tasks.push(async move {
                aggregation_ctx
                    .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
                    .max_tokens(aggregation_tokens)
                    .collect_text()
                    .await
            });
        }
    }

    // --- Stage 4: Collect Final Results ---
    let results = future::join_all(second_aggregation_tasks).await;
    results.into_iter().collect::<Result<Vec<_>>>()
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let question = input.question;
    let proposal_tokens = input.proposal_tokens;
    let aggregation_tokens = input.aggregation_tokens;

    let start = Instant::now();
    println!(
        "--- Starting hierarchical aggregation for question: \"{}\" ---",
        question
    );
    println!(
        "Proposal tokens: {:?}, Aggregation tokens: {}",
        proposal_tokens, aggregation_tokens
    );

    let mut ctx_root = Context::new()?;
    ctx_root.system(SYSTEM_PROMPT);
    ctx_root.flush().await?;

    let final_solutions = run_hierarchical_aggregation(
        &mut ctx_root,
        &question,
        proposal_tokens,
        aggregation_tokens,
    )
    .await?;

    println!("\n--- Aggregation complete in {:?} ---\n", start.elapsed());

    for (i, solution) in final_solutions.iter().enumerate() {
        println!("Final aggregated solution #{}:\n{}\n", i + 1, solution);
    }

    Ok(String::new())
}
