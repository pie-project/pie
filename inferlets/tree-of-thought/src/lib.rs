//! Tree-of-Thought (ToT) multi-branch reasoning — **raw-WIT keep-core** rewrite.
//!
//! A 3-level tree search (Propose → Execute → Reflect); each level spawns
//! `num_branches` branches, explored concurrently via `kv.fork()` shared-prefix
//! decoding. Written directly on the low-level WIT surface (In Gim's SDK-minimize
//! directive): no `Context`/`Generator`/`Sampler` facade. Each branch's decode
//! LOOP is the visible `decode_chat`; only the thin keep-core primitives it calls
//! are SDK surface (`carrier`, `prefill`, `sampler`, kept `chat`/`model`).

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
    #[serde(default = "default_num_branches")]
    num_branches: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_question() -> String { "Calculate (42 + 3) * 5 / 15.".to_string() }
fn default_num_branches() -> usize { 2 }
fn default_max_tokens() -> usize { 512 }

const PROPOSE_PROMPT_TEMPLATE: &str = "\
Please generate a high-level plan for solving the following question. \
First, just state the method you will use. Do not do the actual calculation. \
Keep your response concise and within 80 words. Question: ";

const EXECUTE_PROMPT: &str = "\
The plan looks good! Now, use real numbers and do the calculation. \
Please solve the question step-by-step according to the plan. \
Give me the final answer. Make your response short.";

const REFLECT_PROMPT: &str = "\
Okay. Now, evaluate your own solution and give it a score on a scale of 1 to 5. \
Please rigorously check the correctness of the calculations and the final answer.";

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

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let question = input.question;
    let num_branches = input.num_branches;
    let max_tokens_per_step = input.max_tokens;

    let total_leaves = num_branches.pow(3);
    println!(
        "--- Starting Tree of Thought (Branches={}, Leaves={}, MaxTokens/Step={}) ---",
        num_branches, total_leaves, max_tokens_per_step
    );
    let start = Instant::now();

    let mut ctx_root = Ctx::new();
    ctx_root.prefill(&chat::system(
        "You are a helpful, respectful, and honest assistant that excels at \
        mathematical reasoning. Please follow the user's instructions precisely.",
    ))?;

    // Build and execute tree in parallel
    let level1_futures = (0..num_branches)
        .map(|_| {
            let mut propose_ctx = ctx_root.fork()?;
            let question_ = question.clone();
            Ok(async move {
                // Level 1: Propose Plan
                let propose_prompt = format!("{}{}", PROPOSE_PROMPT_TEMPLATE, question_);
                let mut tail = chat::user(&propose_prompt);
                tail.extend(chat::cue());
                decode_chat(&mut propose_ctx, SAMPLER, &tail, max_tokens_per_step).await?;

                // Level 2: Execute Plan
                propose_ctx.prefill(&chat::user(EXECUTE_PROMPT))?;

                let level2_futures = (0..num_branches)
                    .map(|_| {
                        let mut execute_ctx = propose_ctx.fork()?;
                        Ok(async move {
                            decode_chat(&mut execute_ctx, SAMPLER, &chat::cue(), max_tokens_per_step)
                                .await?;

                            // Level 3: Reflect on Solution
                            execute_ctx.prefill(&chat::user(REFLECT_PROMPT))?;

                            let level3_futures = (0..num_branches)
                                .map(|_| {
                                    let mut reflect_ctx = execute_ctx.fork()?;
                                    Ok(async move {
                                        decode_chat(
                                            &mut reflect_ctx,
                                            SAMPLER,
                                            &chat::cue(),
                                            max_tokens_per_step,
                                        )
                                        .await?;
                                        Ok::<_, String>(())
                                    })
                                })
                                .collect::<Result<Vec<_>>>()?;
                            let results = future::join_all(level3_futures).await;
                            for r in results {
                                r?;
                            }
                            Ok::<_, String>(())
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let results = future::join_all(level2_futures).await;
                for r in results {
                    r?;
                }
                Ok::<_, String>(())
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let results = future::join_all(level1_futures).await;
    for r in results {
        r?;
    }

    println!("\n--- All leaf nodes generated in {:?} ---", start.elapsed());

    Ok(String::new())
}
