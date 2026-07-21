//! Parallel candidate generation with consensus ranking.
//!
//! Generates N candidate responses from a shared prompt prefix, then uses
//! `strsim` to compute pairwise similarity between the extracted answers and
//! selects the most central (consensus) answer.
//!
//! The common prefix is prefilled once. Candidate lanes share those KV cells,
//! while per-lane attention masks isolate their divergent continuations.
//! Independent Gumbel noise drives top-p sampling in each lane.

use inferlet::ptir::prelude::*;
use inferlet::{Result, chat, model as wit_model};
use serde::Deserialize;
use std::time::Instant;

const PAGE_T: u32 = 16; // tokens per pool page
// Qwen3-0.6B
const TEMPERATURE: f32 = 0.6;
const TOP_P: f32 = 0.95;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_question")]
    question: String,
    #[serde(default = "default_num_candidates")]
    num_candidates: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
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

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant that solves problems step by step. \
Show your reasoning, then give your final answer on the last line \
in the format: Final Answer: <answer>";

fn decode_text(tokens: &[u32]) -> Result<String> {
    if tokens.is_empty() {
        return Ok(String::new());
    }
    let mut dec = chat::Decoder::new();
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
    let max_tokens = input.max_tokens;
    if num_candidates == 0 {
        return Err("num_candidates must be at least 1".into());
    }
    let b = num_candidates as u32;

    let start = Instant::now();
    let vocab = wit_model::output_vocab_size();
    let stop = chat::stop_tokens();

    // Shared prefix: system + question via the deferred-system `system_user`
    // form, plus the assistant cue. The cue is identical for every candidate so
    // it folds into the shared prefill.
    let mut prefix = chat::system_user(SYSTEM_PROMPT, &question);
    prefix.extend(chat::cue());
    if prefix.is_empty() {
        prefix.push(0);
    }
    let n = prefix.len() as u32;

    println!(
        "--- Generating {} candidates in parallel ---",
        num_candidates
    );

    let mut cand_tokens: Vec<Vec<u32>> = vec![Vec::new(); num_candidates];

    if max_tokens > 0 {
        // Shared logical page pool: prefix + all candidates' appends.
        let pool_pages = (n + b * max_tokens as u32 + 2).div_ceil(PAGE_T);
        let pool = pool_pages * PAGE_T;
        let ws = WorkingSet::new();
        let slots = ws
            .reserve(pool_pages)
            .map_err(|e| format!("ws.reserve: {e}"))?;
        let pool_ids = slots.ids().to_vec();

        // ─────────────── 1. SHARED-PREFIX PREFILL FIRE (N-wide) ───────────────
        // One fire writes the prefix KV cells 0..n every candidate shares, and
        // samples B INDEPENDENT first tokens off the read-out row (per-lane
        // Gumbel noise over the shared nucleus keep-mask).
        let prefix_i32: Vec<i32> = prefix.iter().map(|&t| t as i32).collect();
        let toks_p = Channel::from(prefix_i32).named("toks_p"); // [N] i32 (seeded)
        let embed_indptr_p = Channel::from(vec![0u32, n]).named("embed_indptr_p");
        let positions_p = Channel::from((0..n).collect::<Vec<_>>()).named("positions_p");

        // Explicit N-cell write descriptor: cell c → pool_ids[c/PAGE_T] @ c%PAGE_T.
        let w_slot_pv: Vec<u32> = (0..n).map(|c| pool_ids[(c / PAGE_T) as usize]).collect();
        let w_off_pv: Vec<u32> = (0..n).map(|c| c % PAGE_T).collect();
        let w_slot_p = Channel::from(w_slot_pv).named("w_slot_p");
        let w_off_p = Channel::from(w_off_pv).named("w_off_p");
        let klen_p = Channel::from(vec![n; 1]).named("klen_p");
        let pages_p = Channel::from(pool_ids.clone()).named("pages_p");
        let page_indptr_p = Channel::from_shaped([2], vec![0u32, pool_pages]).named("pidx_p");

        // Causal prefill mask [N, POOL]: query row i attends KV cols j <= i.
        let mask_pv: Vec<bool> = (0..n)
            .flat_map(|i| (0..pool).map(move |j| j <= i))
            .collect();
        let mask_p = Channel::from_shaped([n, pool], mask_pv).named("mask_p");
        let rng_p = Channel::from(vec![0x51ed_u32, 0]).named("rng_p");
        let g0s_ch = Channel::new([b], dtype::i32).named("g0s");

        let fwd_p = ForwardPass::new();
        fwd_p.embed(&toks_p, &embed_indptr_p)?;
        fwd_p.attention(
            &ws,
            ..,
            ..,
            &klen_p,
            &pages_p,
            &page_indptr_p,
            &w_slot_p,
            &w_off_p,
            &positions_p,
            Some(&mask_p),
        )?;
        fwd_p.epilogue(move || {
            let r = rng_p.take();
            // Shared nucleus keep-mask over the read-out row, then B independent
            // Gumbel draws → B distinct first tokens.
            let logits = intrinsics::logits(); // [vocab] (single read-out row)
            let scaled = div(&logits, TEMPERATURE.max(1e-4));
            let probs = softmax(&scaled);
            let keep = pivot_threshold(&probs, cummass_le(TOP_P));
            let masked = mask_apply(&scaled, &keep); // [vocab]
            let wide = broadcast(reshape(&masked, [1, vocab]), [b, vocab]); // [B, vocab]
            let g = gumbel(&r, [b, vocab]); // independent per-lane noise
            let toks0 = reduce_argmax(add(&wide, &g)); // [B] i32
            let r_next = add(&r, iota(2)); // advance ctr: [key, ctr+1]
            g0s_ch.put(&toks0);
            rng_p.put(&r_next);
        });

        // ONE pipeline, ONE stream (R4-4): the shared-prefix prefill and the
        // batched decode are sequential phases of the same stream. With
        // `max_tokens == 1` the prefill's sample IS the whole stream, so
        // finish() lands right after its submit (F7).
        let pipe = Pipeline::new();
        fwd_p
            .submit(&pipe)
            .map_err(|e| format!("prefill submit: {e}"))?;
        let g0s: Vec<i32> = g0s_ch
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("g0s take: {e}"))?;

        let mut done = vec![false; num_candidates];
        for (c, &t) in g0s.iter().enumerate().take(num_candidates) {
            let t = t as u32;
            if stop.contains(&t) {
                done[c] = true;
            } else {
                cand_tokens[c].push(t);
            }
        }

        // ──────────────── 2. BATCHED DECODE (B lanes = candidates) ────────────
        // Lane c embeds its own previous token (device loop-carried) at logical
        // position n + step and appends its KV at flat pool cell fill + c. All
        // lanes share the pool pages; each lane's mask row admits the shared
        // prefix plus its own cells only.
        let tok_in = Channel::from(g0s.clone()).named("tok_in"); // [B] device loop-carried
        let pos = Channel::from(vec![n; num_candidates]).named("pos");
        let fill = Channel::from(vec![n + b; 1]).named("fill"); // next free flat cell
        let klen = Channel::from(vec![n + b; num_candidates]).named("klen");
        let w_slot_v: Vec<u32> = (0..b)
            .map(|c| pool_ids[((n + c) / PAGE_T) as usize])
            .collect();
        let w_off_v: Vec<u32> = (0..b).map(|c| (n + c) % PAGE_T).collect();
        let w_slot = Channel::from(w_slot_v).named("w_slot");
        let w_off = Channel::from(w_off_v).named("w_off");
        // Lane c's seed mask: the shared prefix (j < n) plus its own fire-0 cell.
        let seed_mask: Vec<bool> = (0..b)
            .flat_map(|c| (0..pool).map(move |j| j < n || j == n + c))
            .collect();
        let mask = Channel::from_shaped([b, pool], seed_mask).named("mask");
        let tiled: Vec<u32> = (0..b).flat_map(|_| pool_ids.iter().copied()).collect();
        let pages = Channel::from(tiled).named("pages"); // [B*POOL_PAGES]
        let pidx_v: Vec<u32> = (0..=b).map(|c| c * pool_pages).collect();
        let page_indptr = Channel::from_shaped([b + 1], pidx_v).named("page_indptr");
        let pool_ids_ch = Channel::from(pool_ids.clone()).named("pool_ids");
        let out = Channel::new([b], dtype::i32)
            .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
            .named("out");
        let rng = Channel::from(vec![0x9e37_u32, 0]).named("rng");
        let lanes = Channel::from((0..=b).collect::<Vec<u32>>()).named("embed_indptr");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lanes)?;
        fwd.attention(
            &ws,
            ..,
            (n / ws.page_size())..,
            &klen,
            &pages,
            &page_indptr,
            &w_slot,
            &w_off,
            &pos,
            Some(&mask),
        )?;
        fwd.epilogue(move || {
            // TAKES + compute first, PUTS last (value-id discipline).
            let base = fill.take().tensor(); // [1] u32 — next fire's first append cell
            let pids = pool_ids_ch.take().tensor();
            let r = rng.take();

            // Per-lane top-p + temperature sample over [B, vocab] logits
            // (row-wise nucleus, independent Gumbel noise per lane).
            let logits = intrinsics::logits(); // [B, vocab]
            let scaled = div(&logits, TEMPERATURE.max(1e-4));
            let probs = softmax(&scaled);
            let keep = pivot_threshold(&probs, cummass_le(TOP_P));
            let masked = mask_apply(&scaled, &keep);
            let g = gumbel(&r, [b, vocab]);
            let toks = reduce_argmax(add(&masked, &g)); // [B] i32
            let r_next = add(&r, iota(2));

            // Flat append cells for the NEXT fire: wpos = base + lane.
            let lane = iota(b);
            let base_b = broadcast(reshape(&base, [1]), [b]);
            let wpos = add(&base_b, &lane); // [B]

            // Mask evolution: each lane keeps its own ancestry + its new cell.
            let col = broadcast(reshape(iota(pool), [1, pool]), [b, pool]);
            let wpos_c = broadcast(reshape(&wpos, [b, 1]), [b, pool]);
            let new_mask = or(mask.take(), eq(col, wpos_c)); // [B, POOL]

            // Explicit write descriptor via the host-fed pool ids.
            let w_slot_n = gather(&pids, div(&wpos, PAGE_T)); // [B]
            let w_off_n = rem(&wpos, PAGE_T); // [B]
            let filled = add(&base, b); // [1] span after the next fire's appends
            let klen_n = broadcast(reshape(&filled, [1]), [b]);
            let pos_n = add(pos.take(), 1u32);
            let pages_n = reshape(
                broadcast(reshape(&pids, [1, pool_pages]), [b, pool_pages]),
                [b * pool_pages],
            );
            let pidx_n = mul(iota(b + 1), pool_pages);

            tok_in.put(&toks);
            out.put(&toks);
            mask.put(&new_mask);
            w_slot.put(&w_slot_n);
            w_off.put(&w_off_n);
            klen.take();
            klen.put(&klen_n);
            pos.put(&pos_n);
            fill.put(&filled);
            pages.take();
            pages.put(&pages_n);
            page_indptr.take();
            page_indptr.put(&pidx_n);
            rng.put(&r_next);
            pool_ids_ch.put(&pids);
        });
        let budget = if done.iter().any(|d| !d) {
            max_tokens.saturating_sub(1) // the prefill's g0s already emitted
        } else {
            0
        };
        let mut submitted = 0usize;
        let mut in_flight = 0usize;
        while in_flight < DEFAULT_RUNAHEAD_DEPTH && submitted < budget {
            fwd.submit(&pipe)
                .map_err(|e| format!("decode submit: {e}"))?;
            submitted += 1;
            in_flight += 1;
        }
        while in_flight > 0 && done.iter().any(|d| !d) {
            let step: Vec<i32> = out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("out.take: {e}"))?;
            in_flight -= 1;
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
            if submitted < budget && done.iter().any(|d| !d) {
                fwd.submit(&pipe)
                    .map_err(|e| format!("decode submit: {e}"))?;
                submitted += 1;
                in_flight += 1;
            }
        }
        while in_flight > 0 {
            out.take()
                .get::<i32>()
                .await
                .map_err(|e| format!("drain run-ahead candidates: {e}"))?;
            in_flight -= 1;
        }
        // Fully drained above, so close only releases the scheduler wait-set.
        pipe.close();
    }

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
