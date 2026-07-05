//! Skeleton-of-Thought (SoT) parallel elaboration — **raw-WIT keep-core** rewrite.
//!
//! First generates a high-level plan (skeleton) of key points, then elaborates
//! each point concurrently via `kv.fork()` shared-prefix decoding. Written
//! directly on the low-level WIT surface (In Gim's SDK-minimize directive): no
//! `Context`/`Generator`/`Sampler` facade. Each branch's decode LOOP is the
//! visible `decode_chat`; only the thin keep-core primitives it calls are SDK
//! surface (`carrier`, `prefill`, `sampler`, kept `chat`/`model`).

use futures::future;
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
    #[serde(default = "default_num_points")]
    num_points: usize,
    #[serde(default = "default_plan_tokens")]
    plan_tokens: usize,
    #[serde(default = "default_elab_tokens")]
    elab_tokens: usize,
}

fn default_question() -> String { "What are the defining characteristics of Rome?".to_string() }
fn default_num_points() -> usize { 3 }
fn default_plan_tokens() -> usize { 256 }
fn default_elab_tokens() -> usize { 256 }

const SAMPLER: SamplerSpec = SamplerSpec::TopP { temperature: 0.6, p: 0.95 };

/// A raw-WIT decode context: its own KV working set + cursor + first-pass flag +
/// the un-prefilled residual token (mirrors the facade's `ctx.buffer`). Minimal
/// state bundle, not a facade — the decode loop is the visible `decode_chat`.
struct Ctx {
    kv: KvWorkingSet,
    seq_len: u32,
    fresh: bool,
    pending: Vec<u32>,
}

impl Ctx {
    fn new() -> Self {
        Self { kv: KvWorkingSet::new(), seq_len: 0, fresh: true, pending: Vec::new() }
    }

    fn fork(&self) -> Result<Self> {
        Ok(Self {
            kv: self.kv.fork().map_err(|e| format!("fork: {e}"))?,
            seq_len: self.seq_len,
            fresh: true,
            pending: self.pending.clone(),
        })
    }

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

/// Pipelined chat decode over `tail`, up to `max_tokens` tokens (no stop set —
/// terminates on the chat template's end-of-turn `Done` or the budget). Drains
/// the residual first and leaves the last sampled token as the new residual,
/// mirroring the facade's `ctx.buffer` across turns. Hand-written, visible loop
/// over keep-core primitives.
async fn decode_chat(c: &mut Ctx, spec: SamplerSpec, tail: &[u32], max_tokens: usize) -> Result<String> {
    let vocab = inferlet::model::output_vocab_size();
    let s = sampler::sampler_program(spec, vocab)?;
    let mut dec = chat::Decoder::new();
    let mut text = String::new();
    if max_tokens == 0 {
        return Ok(text);
    }
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
    c.pending = vec![last_token];
    Ok(text)
}

/// Generates a high-level plan and elaborates on each point in parallel.
async fn plan_and_generate_parallel(
    ctx: &mut Ctx,
    question: &str,
    max_points: usize,
    plan_max_tokens: usize,
    elab_max_tokens: usize,
) -> Result<Vec<String>> {
    // 1. Fork a context for generating the plan.
    let mut plan_ctx = ctx.fork()?;
    let plan_prompt = format!(
        "Generate up to {} key points that outline the answer to the following question: {}. \
        Each point must be enclosed between the <point> and </point> tags.",
        max_points, question
    );
    let mut plan_tail = chat::user(&plan_prompt);
    plan_tail.extend(chat::cue());

    let output = decode_chat(&mut plan_ctx, SAMPLER, &plan_tail, plan_max_tokens).await?;

    // 2. Robustly parse points from the output.
    let points: Vec<String> = output
        .split("<point>")
        .skip(1)
        .filter_map(|s| s.split("</point>").next())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if points.is_empty() {
        return Ok(Vec::new());
    }

    // 3. Fork from the original base context for a clean state for each elaboration.
    let leaf_futures = points
        .into_iter()
        .map(|point| {
            let mut elab_ctx = ctx.fork()?;
            let complete_prompt = format!(
                "Elaborate on the following point: {}. \
                Your response should be complete and only concerned with this point.",
                point
            );
            let mut tail = chat::user(&complete_prompt);
            tail.extend(chat::cue());

            Ok(async move {
                decode_chat(&mut elab_ctx, SAMPLER, &tail, elab_max_tokens).await
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let results = future::join_all(leaf_futures).await;
    results.into_iter().collect::<Result<Vec<_>>>()
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let question = input.question;
    let num_points = input.num_points;
    let plan_max_tokens = input.plan_tokens;
    let elab_max_tokens = input.elab_tokens;

    let start = Instant::now();

    let mut ctx = Ctx::new();
    ctx.prefill(&chat::system("You are a helpful, respectful and honest assistant."))?;

    println!(
        "--- Starting plan and generate (plan: {} points, {} tokens; elab: {} tokens) ---",
        num_points, plan_max_tokens, elab_max_tokens
    );

    let elaborations = plan_and_generate_parallel(
        &mut ctx,
        &question,
        num_points,
        plan_max_tokens,
        elab_max_tokens,
    )
    .await?;

    println!("\n--- Completed in {:?} ---\n", start.elapsed());

    if elaborations.is_empty() {
        println!("No points were generated or elaborated upon.");
    } else {
        for (i, elaboration) in elaborations.iter().enumerate() {
            println!("Elaboration {}:\n{}\n", i + 1, elaboration);
        }
    }

    Ok(String::new())
}
