//! Demonstrates Tree-of-Thought (ToT) as level-wise beam search.
//!
//! The default run uses breadth=3 and beam_width=2. At each level it
//! generates three candidates per frontier item, scores every candidate in
//! that level, keeps the global top two, and expands only those survivors.
//! Search details are optional debug output; the inferlet returns the selected
//! answer directly.

use futures::future;
use inferlet::{Context, Result, model::Model, runtime, sample::Sampler};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_question")]
    question: String,
    #[serde(default = "default_breadth")]
    breadth: usize,
    #[serde(default = "default_beam_width")]
    beam_width: usize,
    #[serde(default = "default_levels")]
    levels: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    show_search: bool,
}

fn default_question() -> String {
    "Calculate (42 + 3) * 5 / 15.".to_string()
}
fn default_breadth() -> usize {
    3
}
fn default_beam_width() -> usize {
    2
}
fn default_levels() -> usize {
    2
}
fn default_max_tokens() -> usize {
    384
}

const SYSTEM_PROMPT: &str = "\
You are a careful mathematical reasoning assistant. Follow the user question, \
keep intermediate candidates concise, and put the final answer on the last line.";

const PLAN_PROMPT: &str = "\
Generate one concise plan for solving the question. Do not calculate yet. \
End with the key operation you will perform.";

const SOLVE_PROMPT: &str = "\
Use the previous plan or partial solution to solve the question. Return a direct \
answer and keep the final line in the form: Final Answer: <answer>.";

const REFINE_PROMPT: &str = "\
Refine the previous solution, check arithmetic, and return a direct final \
answer. Keep the final line in the form: Final Answer: <answer>.";

const SCORE_PROMPT: &str = "\
Score the current candidate for correctness and usefulness on a 1 to 10 scale. \
Respond with only the integer score.";

struct Candidate {
    ctx: Context,
    content: String,
    score: u8,
    level: usize,
    parent_rank: usize,
    branch_index: usize,
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let question = input.question;
    let breadth = input.breadth.max(1);
    let beam_width = input.beam_width.max(1);
    let levels = input.levels.max(1);
    let max_tokens = input.max_tokens.max(64);
    let show_search = input.show_search;

    let start = Instant::now();

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let mut root = Context::new(&model)?;
    root.system(SYSTEM_PROMPT);
    root.user(&format!("Question: {question}"));
    root.flush().await?;

    let mut frontier = vec![Candidate {
        ctx: root,
        content: String::new(),
        score: 10,
        level: 0,
        parent_rank: 0,
        branch_index: 0,
    }];

    for level in 1..=levels {
        let mut expansions = Vec::with_capacity(frontier.len() * breadth);
        for (parent_rank, parent) in frontier.into_iter().enumerate() {
            for branch_index in 1..=breadth {
                let ctx = parent.ctx.fork()?;
                expansions.push(expand_and_score(
                    ctx,
                    level,
                    parent_rank + 1,
                    branch_index,
                    max_tokens,
                ));
            }
        }

        let results = future::join_all(expansions).await;
        let mut candidates = results.into_iter().collect::<Result<Vec<_>>>()?;
        candidates.sort_by(compare_candidates);

        if show_search {
            print_level(level, &candidates, beam_width);
        }

        let keep = select_beam(candidates, beam_width);
        frontier = keep;
    }

    let best = frontier
        .into_iter()
        .min_by(compare_candidates)
        .ok_or_else(|| "Tree-of-Thought search produced no candidates".to_string())?;

    if show_search {
        println!(
            "\nSelected L{} parent {} branch {} with score {} in {:?}",
            best.level,
            best.parent_rank,
            best.branch_index,
            best.score,
            start.elapsed()
        );
    }

    Ok(best.content.trim().to_string())
}

async fn expand_and_score(
    mut ctx: Context,
    level: usize,
    parent_rank: usize,
    branch_index: usize,
    max_tokens: usize,
) -> Result<Candidate> {
    ctx.user(prompt_for_level(level));
    ctx.cue();
    let content = ctx
        .generate(Sampler::TopP {
            temperature: 0.6,
            p: 0.95,
        })
        .max_tokens(max_tokens)
        .collect_text()
        .await?;

    let score = score_candidate(&ctx).await?;

    Ok(Candidate {
        ctx,
        content,
        score,
        level,
        parent_rank,
        branch_index,
    })
}

async fn score_candidate(ctx: &Context) -> Result<u8> {
    let mut score_ctx = ctx.fork()?;
    score_ctx.user(SCORE_PROMPT);
    score_ctx.cue();
    let raw_score = score_ctx
        .generate(Sampler::Argmax)
        .max_tokens(16)
        .collect_text()
        .await?;

    Ok(parse_score(&raw_score).unwrap_or(1))
}

fn prompt_for_level(level: usize) -> &'static str {
    match level {
        1 => PLAN_PROMPT,
        2 => SOLVE_PROMPT,
        _ => REFINE_PROMPT,
    }
}

fn select_beam(mut candidates: Vec<Candidate>, beam_width: usize) -> Vec<Candidate> {
    let keep = beam_width.min(candidates.len());
    candidates.truncate(keep);
    candidates
}

fn compare_candidates(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    b.score
        .cmp(&a.score)
        .then_with(|| a.level.cmp(&b.level))
        .then_with(|| a.parent_rank.cmp(&b.parent_rank))
        .then_with(|| a.branch_index.cmp(&b.branch_index))
}

fn parse_score(text: &str) -> Option<u8> {
    text.split(|ch: char| !ch.is_ascii_digit())
        .filter_map(|part| part.parse::<u8>().ok())
        .find(|score| (1..=10).contains(score))
}

fn print_level(level: usize, candidates: &[Candidate], beam_width: usize) {
    println!("\n--- Level {level}: all candidates scored; committing next frontier ---");
    for (rank, candidate) in candidates.iter().enumerate() {
        let marker = if rank < beam_width { "keep" } else { "prune" };
        println!(
            "  {marker}: rank {} score {} from parent {} branch {} — {}",
            rank + 1,
            candidate.score,
            candidate.parent_rank,
            candidate.branch_index,
            truncate(&candidate.content, 100)
        );
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max_len {
        s
    } else {
        let truncated: String = s.chars().take(max_len).collect();
        format!("{}...", truncated)
    }
}
