//! Unified inferlet for comparing reasoning workflows on math word problems.
//!
//! The inferlet never receives the reference answer. It returns candidates,
//! selections, and execution counters; the external benchmark harness owns
//! deterministic answer evaluation.

use std::collections::HashMap;
use std::time::Instant;

use futures::future;
use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::{Deserialize, Serialize};

const SYSTEM_PROMPT: &str = "\
You solve grade-school mathematical word problems carefully. Show concise \
reasoning and put the final numeric answer on the last line exactly as \
\"Final Answer: <number>\". Do not use that phrase anywhere else.";

const SCORE_SYSTEM: &str = "\
You evaluate candidate solutions to mathematical word problems. Check the \
problem interpretation and arithmetic. Respond with only one integer from 1 \
to 10, where 10 means fully correct.";

const AGGREGATE_SYSTEM: &str = "\
You synthesize candidate solutions to mathematical word problems. Compare \
their reasoning, resolve disagreements by recalculating, and return one \
concise solution. Put the final numeric answer on the last line exactly as \
\"Final Answer: <number>\".";

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_pattern")]
    pattern: String,
    question: String,
    #[serde(default = "default_num_candidates")]
    num_candidates: usize,
    #[serde(default = "default_beam_width")]
    beam_width: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_score_tokens")]
    score_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    thinking: bool,
}

fn default_pattern() -> String {
    "direct".into()
}
fn default_num_candidates() -> usize {
    4
}
fn default_beam_width() -> usize {
    2
}
fn default_max_tokens() -> usize {
    256
}
fn default_score_tokens() -> usize {
    16
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.95
}

#[derive(Clone, Serialize)]
struct Candidate {
    id: String,
    stage: &'static str,
    response: String,
    answer: Option<String>,
    score: Option<u8>,
    generated_tokens: usize,
    generator_steps: usize,
}

#[derive(Default, Serialize)]
struct ExecutionStats {
    elapsed_ms: u128,
    generated_tokens: usize,
    generator_steps: usize,
    context_forks: usize,
    generation_calls: usize,
    scoring_calls: usize,
}

impl ExecutionStats {
    fn add_score(&mut self, generated: &Generated) {
        self.generated_tokens += generated.tokens;
        self.generator_steps += generated.steps;
        self.scoring_calls += 1;
    }
}

#[derive(Serialize)]
struct Output {
    pattern: String,
    final_response: String,
    final_answer: Option<String>,
    selected_candidate_id: Option<String>,
    candidates: Vec<Candidate>,
    stats: ExecutionStats,
}

struct Generated {
    text: String,
    tokens: usize,
    steps: usize,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    validate(&input)?;
    let started = Instant::now();
    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    let mut answer_root = Context::new(&model)?;
    answer_root.system(SYSTEM_PROMPT);
    answer_root.user(&input.question);
    answer_root.flush().await?;

    let mut score_root = Context::new(&model)?;
    score_root.system(SCORE_SYSTEM);
    score_root.flush().await?;

    let mut aggregate_root = Context::new(&model)?;
    aggregate_root.system(AGGREGATE_SYSTEM);
    aggregate_root.flush().await?;

    let mut output = match input.pattern.to_ascii_lowercase().as_str() {
        "direct" => run_direct(&input, &model, &answer_root).await?,
        "best_of_n" | "best-of-n" | "self_consistency" => {
            run_best_of_n(&input, &model, &answer_root).await?
        }
        "tree_of_thought" | "tree-of-thought" | "tot" => {
            run_tree_of_thought(&input, &model, &answer_root, &score_root).await?
        }
        "graph_of_thought" | "graph-of-thought" | "got" => {
            run_graph_of_thought(&input, &model, &answer_root, &aggregate_root).await?
        }
        other => {
            return Err(format!(
                "unknown pattern '{other}': expected direct, best_of_n, \
                 tree_of_thought, or graph_of_thought"
            ));
        }
    };
    output.stats.elapsed_ms = started.elapsed().as_millis();
    Ok(output)
}

fn validate(input: &Input) -> Result<()> {
    if input.question.trim().is_empty() {
        return Err("question must not be empty".into());
    }
    if !(1..=16).contains(&input.num_candidates) {
        return Err("num_candidates must be in [1, 16]".into());
    }
    if !(1..=input.num_candidates).contains(&input.beam_width) {
        return Err("beam_width must be in [1, num_candidates]".into());
    }
    if input.max_tokens == 0 || input.score_tokens == 0 {
        return Err("token budgets must be at least 1".into());
    }
    if !(input.temperature.is_finite() && (0.0..=2.0).contains(&input.temperature)) {
        return Err("temperature must be in [0.0, 2.0]".into());
    }
    if !(input.top_p.is_finite() && input.top_p > 0.0 && input.top_p <= 1.0) {
        return Err("top_p must be in (0.0, 1.0]".into());
    }
    Ok(())
}

async fn run_direct(input: &Input, model: &Model, root: &Context) -> Result<Output> {
    let generated = generate_answer(
        root.fork()?,
        model,
        input.max_tokens,
        input.temperature,
        input.top_p,
        input.thinking,
        None,
    )
    .await?;
    let candidate = candidate("direct-0", "answer", generated, None);
    let mut stats = ExecutionStats {
        context_forks: 1,
        ..Default::default()
    };
    add_candidate_stats(&mut stats, &candidate);

    Ok(Output {
        pattern: "direct".into(),
        final_response: candidate.response.clone(),
        final_answer: candidate.answer.clone(),
        selected_candidate_id: Some(candidate.id.clone()),
        candidates: vec![candidate],
        stats,
    })
}

async fn run_best_of_n(input: &Input, model: &Model, root: &Context) -> Result<Output> {
    let generated = generate_candidates(input, model, root, "sample", None).await?;
    let candidates: Vec<Candidate> = generated
        .into_iter()
        .enumerate()
        .map(|(idx, generated)| candidate(&format!("sample-{idx}"), "sample", generated, None))
        .collect();
    let selected = majority_vote_index(&candidates).unwrap_or(0);
    let chosen = &candidates[selected];
    let mut stats = ExecutionStats {
        context_forks: candidates.len(),
        ..Default::default()
    };
    for candidate in &candidates {
        add_candidate_stats(&mut stats, candidate);
    }

    Ok(Output {
        pattern: "best_of_n".into(),
        final_response: chosen.response.clone(),
        final_answer: chosen.answer.clone(),
        selected_candidate_id: Some(chosen.id.clone()),
        candidates,
        stats,
    })
}

async fn run_tree_of_thought(
    input: &Input,
    model: &Model,
    answer_root: &Context,
    score_root: &Context,
) -> Result<Output> {
    let initial = generate_candidates(input, model, answer_root, "initial", None).await?;
    let mut candidates: Vec<Candidate> = initial
        .into_iter()
        .enumerate()
        .map(|(idx, generated)| candidate(&format!("initial-{idx}"), "initial", generated, None))
        .collect();
    let mut stats = ExecutionStats {
        context_forks: candidates.len(),
        ..Default::default()
    };
    for candidate in &candidates {
        add_candidate_stats(&mut stats, candidate);
    }

    score_candidates(
        &input.question,
        input.thinking,
        input.score_tokens,
        model,
        score_root,
        &mut candidates,
        &mut stats,
    )
    .await?;

    let mut ranked: Vec<usize> = (0..candidates.len()).collect();
    ranked.sort_by_key(|&idx| std::cmp::Reverse(candidates[idx].score.unwrap_or(0)));
    ranked.truncate(input.beam_width);

    let refine_futures = ranked.iter().map(|&idx| {
        let previous = candidates[idx].response.clone();
        let prompt = format!(
            "Previous candidate solution:\n{previous}\n\n\
             Critique its interpretation and arithmetic, then produce a corrected \
             complete solution to the original problem."
        );
        let ctx = answer_root.fork();
        async move {
            generate_answer(
                ctx?,
                model,
                input.max_tokens,
                input.temperature,
                input.top_p,
                input.thinking,
                Some(&prompt),
            )
            .await
        }
    });
    let refined = future::join_all(refine_futures)
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    stats.context_forks += refined.len();

    let first_refined = candidates.len();
    for (idx, generated) in refined.into_iter().enumerate() {
        let candidate = candidate(&format!("refined-{idx}"), "refined", generated, None);
        add_candidate_stats(&mut stats, &candidate);
        candidates.push(candidate);
    }
    score_candidates(
        &input.question,
        input.thinking,
        input.score_tokens,
        model,
        score_root,
        &mut candidates[first_refined..],
        &mut stats,
    )
    .await?;

    let selected = (first_refined..candidates.len())
        .max_by_key(|&idx| candidates[idx].score.unwrap_or(0))
        .unwrap_or(0);
    let chosen = &candidates[selected];

    Ok(Output {
        pattern: "tree_of_thought".into(),
        final_response: chosen.response.clone(),
        final_answer: chosen.answer.clone(),
        selected_candidate_id: Some(chosen.id.clone()),
        candidates,
        stats,
    })
}

async fn run_graph_of_thought(
    input: &Input,
    model: &Model,
    answer_root: &Context,
    aggregate_root: &Context,
) -> Result<Output> {
    let generated = generate_candidates(input, model, answer_root, "proposal", None).await?;
    let mut candidates: Vec<Candidate> = generated
        .into_iter()
        .enumerate()
        .map(|(idx, generated)| candidate(&format!("proposal-{idx}"), "proposal", generated, None))
        .collect();
    let mut stats = ExecutionStats {
        context_forks: candidates.len() + 1,
        ..Default::default()
    };
    for candidate in &candidates {
        add_candidate_stats(&mut stats, candidate);
    }

    let proposals = candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| format!("Candidate {}:\n{}", idx + 1, candidate.response))
        .collect::<Vec<_>>()
        .join("\n\n");
    let prompt = format!(
        "Original problem:\n{}\n\nCandidate solutions:\n{}\n\n\
         Synthesize the most reliable final solution.",
        input.question, proposals
    );
    let generated = generate_answer(
        aggregate_root.fork()?,
        model,
        input.max_tokens,
        0.0,
        1.0,
        input.thinking,
        Some(&prompt),
    )
    .await?;
    let aggregate = candidate("aggregate-0", "aggregate", generated, None);
    add_candidate_stats(&mut stats, &aggregate);
    let final_response = aggregate.response.clone();
    let final_answer = aggregate.answer.clone();
    let selected_candidate_id = Some(aggregate.id.clone());
    candidates.push(aggregate);

    Ok(Output {
        pattern: "graph_of_thought".into(),
        final_response,
        final_answer,
        selected_candidate_id,
        candidates,
        stats,
    })
}

async fn generate_candidates(
    input: &Input,
    model: &Model,
    root: &Context,
    _stage: &'static str,
    prompt: Option<&str>,
) -> Result<Vec<Generated>> {
    let contexts = (0..input.num_candidates)
        .map(|_| root.fork())
        .collect::<Result<Vec<_>>>()?;
    let futures = contexts.into_iter().map(|ctx| {
        generate_answer(
            ctx,
            model,
            input.max_tokens,
            input.temperature,
            input.top_p,
            input.thinking,
            prompt,
        )
    });
    future::join_all(futures)
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()
}

async fn score_candidates(
    question: &str,
    thinking: bool,
    max_tokens: usize,
    model: &Model,
    root: &Context,
    candidates: &mut [Candidate],
    stats: &mut ExecutionStats,
) -> Result<()> {
    let futures =
        candidates.iter().map(|candidate| {
            let prompt = format!(
                "Problem:\n{question}\n\nCandidate solution:\n{}\n\nScore:",
                candidate.response
            );
            let ctx = root.fork();
            async move {
                generate_answer(ctx?, model, max_tokens, 0.0, 1.0, thinking, Some(&prompt)).await
            }
        });
    let scores = future::join_all(futures)
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    stats.context_forks += scores.len();
    for (candidate, generated) in candidates.iter_mut().zip(scores) {
        candidate.score = parse_score(&generated.text);
        stats.add_score(&generated);
    }
    Ok(())
}

async fn generate_answer(
    mut ctx: Context,
    model: &Model,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    thinking: bool,
    prompt: Option<&str>,
) -> Result<Generated> {
    if let Some(prompt) = prompt {
        ctx.user(prompt);
    }
    ctx.cue();
    if !thinking {
        ctx.append(&model.tokenizer().encode("<think>\n\n</think>\n\n"));
    }
    let sampler = if temperature <= 0.0 {
        Sampler::Argmax
    } else {
        Sampler::TopP {
            temperature,
            p: top_p,
        }
    };
    let stops = chat::stop_tokens(model);
    let mut generator = ctx.generate(sampler).max_tokens(max_tokens).stop(&stops);
    let mut decoder = chat::Decoder::new(model);
    let mut text = String::new();
    let mut tokens = 0;
    let mut steps = 0;

    while let Some(step) = generator.next()? {
        let output = step.execute().await?;
        tokens += output.tokens.len();
        steps += 1;
        match decoder.feed(&output.tokens)? {
            chat::Event::Delta(delta) => text.push_str(&delta),
            chat::Event::Done(done) => {
                text = done;
                break;
            }
            chat::Event::Idle | chat::Event::Interrupt(_) => {}
        }
    }
    Ok(Generated {
        text,
        tokens,
        steps,
    })
}

fn candidate(id: &str, stage: &'static str, generated: Generated, score: Option<u8>) -> Candidate {
    Candidate {
        id: id.into(),
        stage,
        answer: extract_final_answer(&generated.text),
        response: generated.text,
        score,
        generated_tokens: generated.tokens,
        generator_steps: generated.steps,
    }
}

fn add_candidate_stats(stats: &mut ExecutionStats, candidate: &Candidate) {
    stats.generated_tokens += candidate.generated_tokens;
    stats.generator_steps += candidate.generator_steps;
    stats.generation_calls += 1;
}

fn majority_vote_index(candidates: &[Candidate]) -> Option<usize> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for answer in candidates
        .iter()
        .filter_map(|candidate| candidate.answer.as_deref())
    {
        *counts.entry(answer).or_default() += 1;
    }
    candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, candidate)| {
            let answer = candidate.answer.as_deref()?;
            Some((idx, counts.get(answer).copied().unwrap_or(0)))
        })
        .max_by_key(|(idx, count)| (*count, std::cmp::Reverse(*idx)))
        .map(|(idx, _)| idx)
}

fn extract_final_answer(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let marker = "final answer:";
    let pos = lower.rfind(marker)?;
    let tail = &text[pos + marker.len()..];
    extract_last_number(tail)
}

fn extract_last_number(text: &str) -> Option<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | ',' | '/') {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
        .into_iter()
        .rev()
        .find_map(|token| normalize_number(&token))
}

fn normalize_number(raw: &str) -> Option<String> {
    let mut value = raw
        .trim_matches(|c: char| matches!(c, '+' | '.' | ',' | '/'))
        .replace(',', "");
    if value.is_empty() || !value.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }
    if value.ends_with(".0") {
        value.truncate(value.len() - 2);
    }
    Some(value)
}

fn parse_score(text: &str) -> Option<u8> {
    text.split(|c: char| !c.is_ascii_digit())
        .filter_map(|part| part.parse::<u8>().ok())
        .find(|score| (1..=10).contains(score))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_marked_numeric_answer() {
        assert_eq!(
            extract_final_answer("Work.\nFinal Answer: $1,234.0"),
            Some("1234".into())
        );
    }

    #[test]
    fn rejects_unmarked_numeric_answer() {
        assert_eq!(extract_final_answer("Work.\nThe answer is 1234."), None);
    }

    #[test]
    fn rejects_final_answer_without_colon() {
        assert_eq!(extract_final_answer("Work.\nFinal Answer is 1234."), None);
    }

    #[test]
    fn vote_prefers_first_candidate_on_tie() {
        let make = |id: &str, answer: &str| Candidate {
            id: id.into(),
            stage: "sample",
            response: String::new(),
            answer: Some(answer.into()),
            score: None,
            generated_tokens: 0,
            generator_steps: 0,
        };
        let candidates = vec![make("a", "1"), make("b", "2")];
        assert_eq!(majority_vote_index(&candidates), Some(0));
    }
}
