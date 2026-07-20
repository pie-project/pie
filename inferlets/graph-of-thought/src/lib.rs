//! Demonstrates Graph-of-Thought (GoT) for hierarchical aggregation.
//!
//! This example generates multiple initial proposals concurrently, then
//! progressively aggregates them in pairs across multiple levels.
//!
//! # Batched aggregation
//!
//! The original implementation processed proposals one at a time in a
//! `while let Some` loop, which meant aggregation pairs were launched
//! sequentially. This prevented the runtime scheduler from seeing all
//! aggregation requests simultaneously and coalescing them into a single
//! batched forward pass.
//!
//! This version collects all proposals first via `future::join_all`, then
//! launches all aggregation pairs in one shot. The scheduler now receives
//! all N/2 aggregation `flush()` calls concurrently and can batch them
//! together, reducing the number of GPU kernel invocations per level from
//! N/2 sequential firings to ideally 1 batched firing.

use futures::future;
use inferlet::{
    Context, sample::Sampler, model::Model,
    runtime, Result,
};
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
Solve the following math problem step by step. Show your work and give the final numeric answer. \
End your response with: ANSWER: <number>\n\
Question: {}";

const AGGREGATE_PROMPT: &str = "";  // unused - see format_aggregate_prompt below

/// Pair up a flat list of (text, context) results into aggregation tasks,
/// launching all pairs simultaneously so the scheduler can batch them.
///
/// If the input has an odd number of items, the last unpaired item is
/// dropped (its context is released immediately, freeing KV pages).
fn launch_aggregation_pairs(
    items: Vec<(String, Context)>,
    aggregation_tokens: usize,
    question: String,
    model: &Model,
) -> Result<Vec<impl std::future::Future<Output = Result<(String, Context)>> + 'static>> {
    let mut tasks = Vec::new();
    let mut iter = items.into_iter();

    while let Some((text_a, _ctx_a)) = iter.next() {
        if let Some((text_b, _ctx_b)) = iter.next() {
            // Both proposal contexts are dropped here — all their KV pages
            // are freed immediately. The aggregation uses a fresh context
            // with only the question + both proposals, so the model always
            // knows what problem it is solving.
            let mut ctx = Context::new(model)?;
            let prompt = format!(
                "You are solving this math problem: {}\n\nHere are two proposed solutions:\n\nSolution A:\n{}\n\nSolution B:\n{}\n\nPick the correct answer. Show your reasoning briefly. End with: ANSWER: <number>",
                question.clone(), text_a, text_b
            );
            ctx.system(SYSTEM_PROMPT);
            ctx.user(&prompt);
            ctx.cue();
            tasks.push(async move {
                let text = ctx
                    .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
                    .max_tokens(aggregation_tokens)
                    .collect_text()
                    .await?;
                Ok((text, ctx))
            });
        }
        // Odd item out: context dropped, KV pages freed.
    }

    Ok(tasks)
}

/// Main logic for running the hierarchical aggregation workflow.
async fn run_hierarchical_aggregation(
    base_context: &mut Context,
    question: &str,
    proposal_tokens: Vec<usize>,
    aggregation_tokens: usize,
    model: &Model,
) -> Result<Vec<String>> {
    // --- Stage 1: Generate Initial Proposals (all concurrent) ---
    let propose_prompt = PROPOSAL_PROMPT_TEMPLATE.replace("{}", question);
    base_context.user(&propose_prompt);
    base_context.flush().await?;

    let proposal_futures = proposal_tokens
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
        .collect::<Result<Vec<_>>>()?;

    // Collect ALL proposals before pairing. This lets Stage 2 launch all
    // aggregation pairs simultaneously rather than as each proposal dribbles in.
    let proposals: Vec<(String, Context)> = future::join_all(proposal_futures)
        .await
        .into_iter()
        .collect::<Result<_>>()?;

    // --- Stage 2: First-Level Aggregation (all pairs launched at once) ---
    // All N/2 aggregation flush() calls hit the scheduler simultaneously,
    // which can coalesce them into a single batched forward pass.
    let first_agg_futures = launch_aggregation_pairs(proposals, aggregation_tokens, question.to_string(), &model)?;
    let first_aggregations: Vec<(String, Context)> = future::join_all(first_agg_futures)
        .await
        .into_iter()
        .collect::<Result<_>>()?;

    // --- Stage 3: Second-Level Aggregation (all pairs launched at once) ---
    let second_agg_futures = launch_aggregation_pairs(first_aggregations, aggregation_tokens, question.to_string(), &model)?;
    let final_results: Vec<String> = future::join_all(second_agg_futures)
        .await
        .into_iter()
        .collect::<Result<Vec<(String, Context)>>>()?
        .into_iter()
        .map(|(text, _ctx)| text)
        .collect();

    Ok(final_results)
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

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let mut ctx_root = Context::new(&model)?;
    ctx_root.system(SYSTEM_PROMPT);
    ctx_root.flush().await?;

    let final_solutions = run_hierarchical_aggregation(
        &mut ctx_root,
        &question,
        proposal_tokens,
        aggregation_tokens,
        &model,
    )
    .await?;

    println!("\n--- Aggregation complete in {:?} ---\n", start.elapsed());

    for (i, solution) in final_solutions.iter().enumerate() {
        println!("Final aggregated solution #{}:\n{}\n", i + 1, solution);
    }

    Ok(String::new())
}
