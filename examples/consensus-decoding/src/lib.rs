//! Parallel candidate generation with consensus ranking.
//!
//! Generates N candidate responses from a shared prompt prefix, then uses
//! `strsim` to compute pairwise similarity between the extracted answers and
//! selects the most central (consensus) answer.

use inferlet::chat;
use inferlet::eta::prelude::*;
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_question")]
    question: String,
    #[serde(default = "default_num_candidates")]
    num_candidates: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Sampling temperature. Self-consistency needs genuinely diverse samples;
    /// too low a value collapses every candidate onto the same greedy chain.
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
}

fn default_question() -> String {
    "What is 17 * 24 + 13?".to_string()
}
fn default_num_candidates() -> usize {
    5
}
fn default_max_tokens() -> usize {
    1024
}
fn default_temperature() -> f32 {
    0.9
}
fn default_top_p() -> f32 {
    0.95
}

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant that solves problems step by step. \
Show your reasoning, then give your final answer on the last line \
in the format: Final Answer: <answer>";

fn decode_text(tokens: &[u32]) -> Result<String> {
    if tokens.is_empty() {
        return Ok(String::new());
    }
    let dec = chat::Decoder::new();
    let mut text = String::new();
    match dec.feed(tokens)? {
        chat::Event::Delta(s) | chat::Event::Done(s) => text.push_str(&s),
        _ => {}
    }
    Ok(text)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let question = input.question;
    let num_candidates = input.num_candidates;
    let temperature = input.temperature.max(1e-4);
    let top_p = input.top_p.clamp(0.0, 1.0);
    let max_tokens = input.max_tokens;
    if num_candidates == 0 {
        return Err("num_candidates must be at least 1".into());
    }
    let b = num_candidates as u32;

    let start = Instant::now();
    let vocab = model::output_vocab_size();
    let stop = chat::stop_tokens();

    // Shared prefix: system + question via the deferred-system `system_user`
    // form, plus the assistant cue. The cue is identical for every candidate so
    // it folds into the shared prefill.
    let mut prefix = chat::system_user(SYSTEM_PROMPT, &question);
    prefix.extend(chat::cue());
    let prefix: Vec<i32> = prefix.into_iter().map(|t| t as i32).collect();
    let (&last, head) = prefix.split_last().ok_or("empty prompt")?;

    println!(
        "--- Generating {} candidates in parallel ---",
        num_candidates
    );

    let mut cand_tokens: Vec<Vec<u32>> = vec![Vec::new(); num_candidates];

    let ws = WorkingSet::new();
    let pipe = Pipeline::new();
    if !head.is_empty() {
        let fill = ForwardPass::new();
        fill.embed(
            &Channel::from(head),
            &Channel::from([0u32, head.len() as u32]),
        )?;
        fill.state(&[&ws], StateGeometry::folding())?;
        fill.submit(&pipe).context("prefill submit")?;
    }
    let candidates_ws = (0..num_candidates)
        .map(|_| ws.fork(&pipe))
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("fork the shared prefix")?;
    let lanes_ws: Vec<&WorkingSet> = candidates_ws.iter().collect();

    // ──────────────── BATCHED DECODE (B lanes = candidates) ────────────────
    let tok_in = Channel::from(vec![last; num_candidates]); // [B] device loop-carried
    let out = Channel::new([b], dtype::i32).capacity(channel_capacity() as u32);
    let rng = Channel::from([0x9e37_u32, 0]);
    let lanes = Channel::from((0..=b).collect::<Vec<u32>>());

    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &lanes)?;
    fwd.state(&lanes_ws, StateGeometry::folding())?;
    fwd.epilogue(move || {
        // TAKES + compute first, PUTS last (value-id discipline).
        let r = rng.take();

        // Per-lane top-p + temperature sample over [B, vocab] logits
        // (row-wise nucleus, independent Gumbel noise per lane).
        // B=1 arrives squeezed to `[vocab]`; B>=2 must stay the bare intrinsic,
        // or the fused nucleus fast path is skipped and the sample comes back empty.
        let logits = if b == 1 {
            reshape(intrinsics::output(0, "logits"), [b, vocab])
        } else {
            intrinsics::output(0, "logits")
        }; // [B, vocab]
        let scaled = &logits / temperature;
        let probs = softmax(&scaled);
        let keep = pivot_threshold(&probs, cummass_le(top_p));
        let masked = mask_apply(&scaled, &keep);
        let g = gumbel(&r, [b, vocab]);
        let toks = reduce_argmax(&masked + &g); // [B] i32
        let r_next = &r + iota(2);

        tok_in.put(&toks);
        out.put(&toks);
        rng.put(&r_next);
    });
    let mut done = vec![false; num_candidates];
    run_ahead(&pipe, &fwd, max_tokens, async || {
        let step: Vec<i32> = out.take_host::<Vec<i32>>().await?;
        for (c, &t) in step.iter().enumerate().take(num_candidates) {
            if done[c] {
                continue; // lane keeps firing; its output is ignored
            }
            let t = t as u32;
            if stop.contains(&t) {
                done[c] = true;
            } else {
                cand_tokens[c].push(t);
            }
        }
        if done.iter().all(|d| *d) {
            return Ok(ControlFlow::Break(()));
        }
        Ok(ControlFlow::Continue(()))
    })
    .await?;
    // Any fire still in flight after an early stop is left untaken; close
    // releases the scheduler wait-set and reclaims them.
    pipe.close();

    let candidates: Vec<String> = cand_tokens
        .iter()
        .map(|t| decode_text(t))
        .collect::<Result<Vec<_>>>()?;

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

    Ok(candidates[best_idx].clone())
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
