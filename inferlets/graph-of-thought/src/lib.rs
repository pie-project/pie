//! Graph-of-Thought (GoT) hierarchical aggregation — **raw-WIT keep-core** rewrite.
//!
//! Generates multiple initial proposals concurrently (via `kv.fork()` shared-prefix
//! decoding), then progressively aggregates them in pairs across levels. Written
//! directly on the low-level WIT surface (In Gim's SDK-minimize directive): no
//! `Context`/`Generator`/`Sampler` facade. Each branch's decode LOOP is the
//! visible `decode_chat`; only the thin keep-core primitives it calls are SDK
//! surface (`carrier::submit_pass`/`discard_pass`, `prefill::tokens`,
//! `sampler::sampler_program`, the kept `chat`/`model` bindings). Fork is a raw
//! `KvWorkingSet::fork()` (COW-shared prefix) + cursor copy.

use futures::stream::FuturesUnordered;
use futures::{future, StreamExt};
use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, prefill, Result};
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

const PROPOSAL_SAMPLER: SamplerSpec = SamplerSpec::TopP { temperature: 0.6, p: 0.95 };

/// A raw-WIT decode context: its own KV working set + cursor + first-pass flag.
/// This is a minimal state bundle (not a facade) — the decode loop is the
/// visible `decode_chat` free function; the only methods here are one-line
/// delegations to keep-core primitives.
struct Ctx {
    kv: KvWorkingSet,
    seq_len: u32,
    fresh: bool,
    /// The last sampled token of the previous decode, held un-prefilled until
    /// the next `prefill`/`decode_chat` (mirrors the facade's `ctx.buffer`
    /// residual: a chat generate commits all-but-the-last token to KV and
    /// re-prefills the last on the next turn, so it stays in the running
    /// conversation context across fork/aggregation stages).
    pending: Vec<u32>,
}

impl Ctx {
    fn new() -> Self {
        Self { kv: KvWorkingSet::new(), seq_len: 0, fresh: true, pending: Vec::new() }
    }

    /// COW-shared-prefix fork (raw `KvWorkingSet::fork`) + cursor/residual copy.
    /// The fork starts a new generation, so its first pass is `fresh` (#26 clear).
    fn fork(&self) -> Result<Self> {
        Ok(Self {
            kv: self.kv.fork().map_err(|e| format!("fork: {e}"))?,
            seq_len: self.seq_len,
            fresh: true,
            pending: self.pending.clone(),
        })
    }

    /// Non-sampling prefill of the pending residual + `tokens` (the facade's
    /// `flush`, which drains `ctx.buffer` before the fill).
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        let mut t = std::mem::take(&mut self.pending);
        t.extend_from_slice(tokens);
        prefill::tokens(&self.kv, &mut self.seq_len, &t)
    }
}

/// Read the sampled token off a finalized pass's single-`Token` output tensor.
async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

/// Pipelined chat decode over `tail` (prompt suffix), up to `max_tokens` tokens.
/// The GoT branches configure NO stop set, so termination is either the chat
/// template's end-of-turn (`chat::Event::Done`, detected by the streaming detok)
/// or the token budget. Each step eagerly speculates the next forward (run-ahead
/// overlap) and rolls the speculation back with `discard_pass` on the terminal
/// step. The loop is hand-written and visible; it calls only keep-core primitives.
async fn decode_chat(c: &mut Ctx, spec: SamplerSpec, tail: &[u32], max_tokens: usize) -> Result<String> {
    let vocab = inferlet::model::output_vocab_size();
    let s = sampler::sampler_program(spec, vocab)?;
    let mut dec = chat::Decoder::new();
    let mut text = String::new();
    if max_tokens == 0 {
        return Ok(text);
    }
    // Drain the residual (the facade's `next()` takes `ctx.buffer` first), then
    // append this turn's prompt suffix.
    let mut head = std::mem::take(&mut c.pending);
    head.extend_from_slice(tail);
    if head.is_empty() {
        head = vec![0u32];
    }

    let mut producer = carrier::submit_pass(&c.kv, &mut c.seq_len, &mut c.fresh, &s, &head, true)?;
    let mut generated = 0usize;
    let mut last_token = 0u32;
    loop {
        let speculate = generated + 1 < max_tokens;
        let consumer = if speculate {
            Some(carrier::submit_pass(&c.kv, &mut c.seq_len, &mut c.fresh, &s, &[0u32], true)?)
        } else {
            None
        };

        let token = read_token(producer).await?;
        generated += 1;
        last_token = token;

        let mut done = false;
        match dec.feed(&[token])? {
            chat::Event::Delta(x) => text.push_str(&x),
            chat::Event::Done(x) => {
                text = x;
                done = true;
            }
            _ => {}
        }
        if generated >= max_tokens {
            done = true;
        }

        if done {
            if let Some(cc) = consumer {
                carrier::discard_pass(cc, &mut c.seq_len).await;
            }
            break;
        }
        producer = consumer.expect("consumer speculated when not terminal");
    }
    // Hold the last sampled token un-prefilled, exactly as the facade leaves it
    // in `ctx.buffer` for the next turn's flush.
    c.pending = vec![last_token];
    Ok(text)
}

/// Main logic for running the hierarchical aggregation workflow.
async fn run_hierarchical_aggregation(
    base: &mut Ctx,
    question: &str,
    proposal_tokens: Vec<usize>,
    aggregation_tokens: usize,
) -> Result<Vec<String>> {
    // --- Stage 1: Generate Initial Proposals ---
    let propose_prompt = PROPOSAL_PROMPT_TEMPLATE.replace("{}", question);
    base.prefill(&chat::user(&propose_prompt))?;

    let mut proposal_tasks = proposal_tokens
        .into_iter()
        .map(|max_tokens| {
            let mut ctx = base.fork()?;
            Ok(async move {
                let tail = chat::cue();
                let proposal_text = decode_chat(&mut ctx, PROPOSAL_SAMPLER, &tail, max_tokens).await?;
                Ok::<_, String>((proposal_text, ctx))
            })
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .collect::<FuturesUnordered<_>>();

    // --- Stage 2: First-Level Aggregation (Pairing Proposals) ---
    let mut first_aggregation_tasks = FuturesUnordered::new();
    let mut pending_proposal: Option<(String, Ctx)> = None;

    while let Some(result) = proposal_tasks.next().await {
        let (proposal_text, mut proposal_ctx) = result?;
        if pending_proposal.is_none() {
            pending_proposal = Some((proposal_text, proposal_ctx));
        } else {
            let (previous_proposal_text, _) = pending_proposal.take().unwrap();
            let aggregation_prompt = format!("{}{}", AGGREGATE_PROMPT, previous_proposal_text);
            let mut tail = chat::user(&aggregation_prompt);
            tail.extend(chat::cue());

            first_aggregation_tasks.push(async move {
                let aggregation_text =
                    decode_chat(&mut proposal_ctx, PROPOSAL_SAMPLER, &tail, aggregation_tokens).await?;
                Ok::<_, String>((aggregation_text, proposal_ctx))
            });
        }
    }

    // --- Stage 3: Second-Level Aggregation (Pairing Aggregations) ---
    let mut second_aggregation_tasks = Vec::new();
    let mut pending_aggregation: Option<(String, Ctx)> = None;

    while let Some(result) = first_aggregation_tasks.next().await {
        let (aggregation_text, mut aggregation_ctx) = result?;
        if pending_aggregation.is_none() {
            pending_aggregation = Some((aggregation_text, aggregation_ctx));
        } else {
            let (previous_aggregation_text, _) = pending_aggregation.take().unwrap();
            let final_prompt = format!("{}{}", AGGREGATE_PROMPT, previous_aggregation_text);
            let mut tail = chat::user(&final_prompt);
            tail.extend(chat::cue());

            second_aggregation_tasks.push(async move {
                decode_chat(&mut aggregation_ctx, PROPOSAL_SAMPLER, &tail, aggregation_tokens).await
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

    let mut ctx_root = Ctx::new();
    ctx_root.prefill(&chat::system(SYSTEM_PROMPT))?;

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
