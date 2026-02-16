//! Demonstrates Graph-of-Thought (GoT) for hierarchical aggregation.
//!
//! This example generates multiple initial proposals concurrently, then
//! progressively aggregates them in pairs across multiple levels. The streaming
//! nature allows aggregation to begin as soon as pairs of proposals are ready,
//! maximizing parallelism.

use futures::stream::FuturesUnordered;
use futures::{StreamExt, future};
use inferlet::{
    context::Context, inference::Sampler, model::Model,
    runtime, ContextExt, InstructExt, Result,
};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

const HELP: &str = "\
Usage: graph-of-thought [OPTIONS]

A program to test hierarchical aggregation by generating and combining multiple proposals for a given question.

Options:
  --question <QUESTION>          The question to process [default: Calculate (42 + 3) * 5 / 15.]
  --proposal-tokens <TOKENS>     Comma-separated list of max tokens for initial proposals
                                 [default: 256,256,256,256,256,256,256,256]
  --aggregation-tokens <TOKENS>  Max tokens for each aggregation step [default: 256]
  -h, --help                     Prints this help message";

const SYSTEM_PROMPT: &str = "You are a helpful, respectful and honest assistant.";

const PROPOSAL_PROMPT_TEMPLATE: &str = "\
Could you suggest a method or approach to solve the following question? \
Please provide a high-level plan without doing the actual calculation. \
Keep it concise, around 80 words. Question: {}";

const AGGREGATE_PROMPT: &str = "\
Please compare the following solution with the one you just provided \
and aggregate their ideas into a single, improved solution:\n";

static FORK_COUNTER: AtomicU32 = AtomicU32::new(0);

fn next_fork_name() -> String {
    format!("fork-{}", FORK_COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Main logic for running the hierarchical aggregation workflow.
async fn run_hierarchical_aggregation(
    base_context: &Context,
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
            let ctx = base_context.fork(&next_fork_name())?;
            Ok(async move {
                ctx.cue();
                let proposal_text = ctx
                    .generate(Sampler::TopP((0.6, 0.95)))
                    .with_max_tokens(max_tokens)
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
        let (proposal_text, proposal_ctx) = result?;
        if pending_proposal.is_none() {
            pending_proposal = Some((proposal_text, proposal_ctx));
        } else {
            let (previous_proposal_text, _) = pending_proposal.take().unwrap();
            let aggregation_prompt = format!("{}{}", AGGREGATE_PROMPT, previous_proposal_text);
            proposal_ctx.user(&aggregation_prompt);
            proposal_ctx.cue();

            first_aggregation_tasks.push(async move {
                let aggregation_text = proposal_ctx
                    .generate(Sampler::TopP((0.6, 0.95)))
                    .with_max_tokens(aggregation_tokens)
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
        let (aggregation_text, aggregation_ctx) = result?;
        if pending_aggregation.is_none() {
            pending_aggregation = Some((aggregation_text, aggregation_ctx));
        } else {
            let (previous_aggregation_text, _) = pending_aggregation.take().unwrap();
            let final_prompt = format!("{}{}", AGGREGATE_PROMPT, previous_aggregation_text);
            aggregation_ctx.user(&final_prompt);
            aggregation_ctx.cue();

            second_aggregation_tasks.push(async move {
                aggregation_ctx
                    .generate(Sampler::TopP((0.6, 0.95)))
                    .with_max_tokens(aggregation_tokens)
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
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let question: String = args
        .value_from_str("--question")
        .unwrap_or_else(|_| "Calculate (42 + 3) * 5 / 15.".to_string());

    let proposal_tokens_str: String = args
        .value_from_str("--proposal-tokens")
        .unwrap_or_else(|_| "256,256,256,256,256,256,256,256".to_string());

    let proposal_tokens: Vec<usize> = proposal_tokens_str
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to parse proposal tokens: {}", e))?;

    let aggregation_tokens: usize = args.value_from_str("--aggregation-tokens").unwrap_or(256);

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

    let ctx_root = Context::new(&model)?;
    ctx_root.system(SYSTEM_PROMPT);
    ctx_root.flush().await?;

    let final_solutions = run_hierarchical_aggregation(
        &ctx_root,
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
