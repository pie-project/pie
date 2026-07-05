//! Demonstrates Best-of-N generation with diversity ranking.
//!
//! Forks a context N times to generate N candidate responses in parallel,
//! then uses `strsim` to compute pairwise similarity between the extracted
//! answers and selects the most central (consensus) answer.

//! Low-level ① rewrite (chat-EOS, pipelined, FORK): the common prefix is
//! prefilled once via `prefill::tokens`, forked N times with `kv.fork()`
//! (keep-core), and each candidate decodes concurrently on the run-ahead carrier
//! (`sampler_program(TopP)` + `submit_pass`/`discard_pass` depth-1 EOS rollback,
//! `chat::` templating) — NO `Context`/`Generator`/`Sampler` facade. Like the
//! original, the fork happens while the prefix prefill is in-flight (Context::flush
//! also only enqueues) — stream ordering serializes prefix-write → candidate-read.

use futures::future;
use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, model, prefill, Result};
use serde::Deserialize;
use std::time::Instant;

async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

fn pass_carries(stop_empty: bool, max_tokens: usize, produced_token_index: usize) -> bool {
    !(stop_empty && max_tokens == produced_token_index)
}

/// Run-ahead (pipelined) chat-EOS decode with depth-1 rollback, continuing from
/// the current `*seq_len` (so a forked/prefilled prefix on `kv` is reused).
async fn decode_pipelined(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    s: &sampler::LoweredSampler,
    prompt: Vec<u32>,
    max_tokens: usize,
    stop: &[u32],
) -> Result<Vec<u32>> {
    let pending = if prompt.is_empty() { vec![0u32] } else { prompt };
    let mut out: Vec<u32> = Vec::with_capacity(max_tokens);
    if max_tokens == 0 {
        return Ok(out);
    }
    let prime_carry = pass_carries(stop.is_empty(), max_tokens, 1);
    let mut producer = carrier::submit_pass(kv, seq_len, fresh, s, &pending, prime_carry)?;
    let mut generated = 0usize;
    loop {
        let speculate = generated + 1 < max_tokens;
        let consumer = if speculate {
            let carry = pass_carries(stop.is_empty(), max_tokens, generated + 2);
            Some(carrier::submit_pass(kv, seq_len, fresh, s, &[0u32], carry)?)
        } else {
            None
        };
        let token = read_token(producer).await?;
        if stop.contains(&token) {
            if let Some(c) = consumer {
                carrier::discard_pass(c, seq_len).await;
            }
            break;
        }
        out.push(token);
        generated += 1;
        match consumer {
            Some(c) => producer = c,
            None => break,
        }
    }
    Ok(out)
}

fn decode_text(tokens: &[u32]) -> Result<String> {
    let mut dec = chat::Decoder::new();
    let mut text = String::new();
    match dec.feed(tokens)? {
        chat::Event::Delta(s) | chat::Event::Done(s) => text.push_str(&s),
        _ => {}
    }
    Ok(text)
}

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_question")]
    question: String,
    #[serde(default = "default_num_candidates")]
    num_candidates: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_question() -> String { "What is 17 * 24 + 13?".to_string() }
fn default_num_candidates() -> usize { 5 }
fn default_max_tokens() -> usize { 1024 }

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant that solves problems step by step. \
Show your reasoning, then give your final answer on the last line \
in the format: Final Answer: <answer>";

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let question = input.question;
    let num_candidates = input.num_candidates;
    let max_tokens = input.max_tokens;

    let start = Instant::now();
    let vocab = model::output_vocab_size();
    let s = sampler::sampler_program(SamplerSpec::TopP { temperature: 0.6, p: 0.95 }, vocab)?;
    let stop = chat::stop_tokens();

    // Build + prefill the common prefix: system + user (deferred system → the
    // combined `system_user` form). The prefill is in-flight; the forks below
    // inherit it (stream-ordered), matching the original's flush-then-fork.
    let base_kv = KvWorkingSet::new();
    let mut base_seq = 0u32;
    let prefix = chat::system_user(SYSTEM_PROMPT, &question);
    prefill::tokens(&base_kv, &mut base_seq, &prefix)?;

    // --- Stage 1: Fork N candidates, decode concurrently ---
    println!("--- Generating {} candidates in parallel ---", num_candidates);

    let mut forks: Vec<(KvWorkingSet, u32)> = Vec::with_capacity(num_candidates);
    for _ in 0..num_candidates {
        let fk = base_kv.fork().map_err(|e| format!("fork: {e}"))?;
        forks.push((fk, base_seq)); // each inherits the prefilled prefix + cursor
    }

    let cue = chat::cue();
    let s_ref = &s;
    let stop_ref: &[u32] = &stop;
    let futs = forks.into_iter().map(|(fk, seq)| {
        let cue = cue.clone();
        async move {
            let mut fseq = seq;
            let mut ffresh = false; // inherited (prefilled) context, not a fresh generate
            let toks =
                decode_pipelined(&fk, &mut fseq, &mut ffresh, s_ref, cue, max_tokens, stop_ref)
                    .await?;
            decode_text(&toks)
        }
    });

    let results: Vec<Result<String>> = future::join_all(futs).await;
    let candidates: Vec<String> = results.into_iter().collect::<Result<Vec<_>>>()?;

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

    // --- Stage 3: Pairwise similarity on extracted answers ---
    println!("--- Computing pairwise similarity ---");

    let n = candidates.len();
    let mut sim = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let s = strsim::normalized_levenshtein(answers[i], answers[j]);
            sim[i][j] = s;
            sim[j][i] = s;
        }
        sim[i][i] = 1.0;
    }

    // --- Stage 4: Rank by centrality (mean similarity to peers) ---
    let centrality: Vec<f64> = (0..n)
        .map(|i| {
            if n <= 1 {
                return 1.0;
            }
            let sum: f64 = (0..n).filter(|&j| j != i).map(|j| sim[i][j]).sum();
            sum / (n - 1) as f64
        })
        .collect();

    let best_idx = centrality
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    // --- Print results ---
    println!("--- Candidate Rankings ---\n");
    let mut ranked: Vec<(usize, f64)> = centrality.iter().copied().enumerate().collect();
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

    Ok(String::new())
}

/// Extract the text after the last occurrence of "Final Answer:" in the response.
/// Fall back to the full trimmed text if the marker is missing.
fn extract_final_answer(response: &str) -> &str {
    response
        .rfind("Final Answer:")
        .map(|pos| response[pos + "Final Answer:".len()..].trim())
        .unwrap_or_else(|| response.trim())
}

/// Truncate to at most `max_len` characters, appending "..." if clipped.
fn truncate(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max_len {
        s
    } else {
        let truncated: String = s.chars().take(max_len).collect();
        format!("{}...", truncated)
    }
}
