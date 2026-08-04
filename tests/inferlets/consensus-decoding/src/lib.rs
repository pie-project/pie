//! Parallel candidate generation with consensus ranking.
//!
//! Generates N candidate responses from a shared prompt prefix, then uses
//! `strsim` to compute pairwise similarity between the extracted answers and
//! selects the most central (consensus) answer.
//!
//! The common prefix is prefilled once. Candidate lanes share those KV cells,
//! while per-lane attention masks isolate their divergent continuations.
//! Independent Gumbel noise drives top-p sampling in each lane.

use inferlet::chat;
use inferlet::ptir::attention::prelude::*;
use serde::Deserialize;
use std::time::Instant;

const PAGE_T: u32 = 16; // tokens per pool page

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
        let slots = ws.reserve(pool_pages).context("ws.reserve")?;
        let pool_ids = slots.ids().to_vec();

        // ─────────────── 1. SHARED-PREFIX PREFILL FIRE (N-wide) ───────────────
        // One fire writes the prefix KV cells 0..n every candidate shares, and
        // samples B INDEPENDENT first tokens off the read-out row (per-lane
        // Gumbel noise over the shared nucleus keep-mask).
        let prefix_i32: Vec<i32> = prefix.iter().map(|&t| t as i32).collect();
        let toks_p = Channel::from(prefix_i32).named("toks_p"); // [N] i32 (seeded)
        let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
        let positions_p = Channel::from_iter(0..n).named("positions_p");

        // Explicit N-cell write descriptor: cell c → pool_ids[c/PAGE_T] @ c%PAGE_T.
        let w_slot_pv: Vec<u32> = (0..n).map(|c| pool_ids[(c / PAGE_T) as usize]).collect();
        let w_off_pv: Vec<u32> = (0..n).map(|c| c % PAGE_T).collect();
        let w_slot_p = Channel::from(w_slot_pv).named("w_slot_p");
        let w_off_p = Channel::from(w_off_pv).named("w_off_p");
        let klen_p = Channel::from([n]).named("klen_p");
        let pages_p = Channel::from(pool_ids.clone()).named("pages_p");
        // The page CSR is the wire's source of truth for kv_len: the driver derives
        // `last_page_len = ((kv_len-1) % PAGE_T) + 1` and reads back a span of
        // `(page_count-1)*PAGE_T + last_page_len` cells. A pool-wide constant count
        // inflates that span past the live prefix and attends uninitialized KV, so
        // the count must track `kv_len` exactly.
        let page_indptr_p = Channel::from([0u32, n.div_ceil(PAGE_T)]).named("pidx_p");

        // Causal prefill mask [N, POOL]: query row i attends KV cols j <= i.
        let mask_pv: Vec<bool> = (0..n)
            .flat_map(|i| (0..pool).map(move |j| j <= i))
            .collect();
        let mask_p = Channel::from_shaped([n, pool], mask_pv).named("mask_p");
        let rng_p = Channel::from([0x51ed_u32, 0]).named("rng_p");
        let g0s_ch = Channel::new([1], dtype::i32).named("g0s");

        let fwd_p = ForwardPass::new();
        fwd_p.embed(&toks_p, &embed_indptr_p)?;
        fwd_p.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &klen_p,
                pages: &pages_p,
                page_indptr: &page_indptr_p,
                w_slot: &w_slot_p,
                w_off: &w_off_p,
                positions: &positions_p,
                mask: Some(&mask_p),
            },
        )?;
        fwd_p.epilogue(move || {
            let r = rng_p.take();
            // Nucleus sample the single read-out row. The B candidates all start
            // from this same token; divergence comes from the decode loop, whose
            // logits really are [B, vocab] and therefore carry B independent
            // Gumbel draws.
            //
            // Do NOT `broadcast` the read-out row to [B, vocab] here. The
            // compiler still matches `LibraryOp::NucleusSample` on the widened
            // form, but the driver's library fast path assumes its logits input
            // is the logits intrinsic (behind at most a reshape) and elides the
            // scratch for it (`fused_runtime.cuh` nucleus prep). A `broadcast`
            // producer still executes and writes a full [B, vocab] tensor into
            // that elided 4-byte slot, corrupting the sampler's scratch — the
            // failure is silent and yields plausible-looking junk tokens.
            let logits = intrinsics::logits(); // [1, vocab] (single read-out row)
            let scaled = &reshape(&logits, [1, vocab]) / temperature;
            let probs = softmax(&scaled);
            let keep = pivot_threshold(&probs, cummass_le(top_p));
            let masked = mask_apply(&scaled, &keep); // [1, vocab]
            let g = gumbel(&r, [1, vocab]);
            let toks0 = reduce_argmax(&masked + &g); // [1] i32
            let r_next = &r + iota(2); // advance ctr: [key, ctr+1]
            g0s_ch.put(&toks0);
            rng_p.put(&r_next);
        });

        // ONE pipeline, ONE stream (R4-4): the shared-prefix prefill and the
        // batched decode are sequential phases of the same stream. With
        // `max_tokens == 1` the prefill's sample IS the whole stream, so
        // finish() lands right after its submit (F7).
        let pipe = Pipeline::new();
        fwd_p.submit(&pipe).context("prefill submit")?;
        let g0s: Vec<i32> = g0s_ch.take_host::<Vec<i32>>().await?;
        // All B candidates share the prefill's token; they diverge in the decode
        // loop, where each lane draws its own Gumbel noise.
        let g0s: Vec<i32> = vec![g0s[0]; num_candidates];

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
        let fill = Channel::from([n + b]).named("fill"); // next free flat cell
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
        // Lane stride is the LIVE page count, not the pool size; `pages` keeps its
        // [B*POOL_PAGES] capacity but only the first `b * pc0` entries are read.
        let pc0 = (n + b).div_ceil(PAGE_T);
        let tiled: Vec<u32> = (0..b * pool_pages)
            .map(|i| pool_ids[(i % pc0) as usize])
            .collect();
        let pages = Channel::from(tiled).named("pages"); // [B*POOL_PAGES]
        let pidx_v: Vec<u32> = (0..=b).map(|c| c * pc0).collect();
        let page_indptr = Channel::from_shaped([b + 1], pidx_v).named("page_indptr");
        let pool_ids_ch = Channel::from(pool_ids.clone()).named("pool_ids");
        let out = Channel::new([b], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("out");
        let rng = Channel::from([0x9e37_u32, 0]).named("rng");
        let lanes = Channel::from((0..=b).collect::<Vec<u32>>()).named("embed_indptr");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lanes)?;
        fwd.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: (n / kv_page_size())..,
                kv_len: &klen,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &pos,
                mask: Some(&mask),
            },
        )?;
        fwd.epilogue(move || {
            // TAKES + compute first, PUTS last (value-id discipline).
            let base = fill.take(); // [1] u32 — next fire's first append cell
            let pids = pool_ids_ch.take();
            let r = rng.take();

            // Per-lane top-p + temperature sample over [B, vocab] logits
            // (row-wise nucleus, independent Gumbel noise per lane).
            let logits = intrinsics::logits(); // [B, vocab]
            let scaled = &logits / temperature;
            let probs = softmax(&scaled);
            let keep = pivot_threshold(&probs, cummass_le(top_p));
            let masked = mask_apply(&scaled, &keep);
            let g = gumbel(&r, [b, vocab]);
            let toks = reduce_argmax(&masked + &g); // [B] i32
            let r_next = &r + iota(2);

            // Flat append cells for the NEXT fire: wpos = base + lane.
            let lane = iota(b);
            let base_b = broadcast(reshape(&base, [1]), [b]);
            let wpos = &base_b + &lane; // [B]

            // Mask evolution: each lane keeps its own ancestry + its new cell.
            let col = broadcast(reshape(iota(pool), [1, pool]), [b, pool]);
            let wpos_c = broadcast(reshape(&wpos, [b, 1]), [b, pool]);
            let new_mask = or(mask.take(), eq(col, wpos_c)); // [B, POOL]

            // Explicit write descriptor via the host-fed pool ids.
            let w_slot_n = gather(&pids, &wpos / PAGE_T); // [B]
            let w_off_n = &wpos % PAGE_T; // [B]
            let filled = &base + b; // [1] span after the next fire's appends
            let klen_n = broadcast(reshape(&filled, [1]), [b]);
            let pos_n = pos.take() + 1u32;
            let page_count = filled.div_ceil(PAGE_T);
            let pages_n = gather(
                &pids,
                iota(b * pool_pages) % broadcast(&page_count, [b * pool_pages]),
            );
            let pidx_n = iota(b + 1) * broadcast(&page_count, [b + 1]);

            // Device-resolved geometry is loop-carried: the host never drains
            // these rings, so every fire's values are re-put here.
            tok_in.put(&toks);
            out.put(&toks);
            mask.put(&new_mask);
            w_slot.put(&w_slot_n);
            w_off.put(&w_off_n);
            klen.put(&klen_n);
            pos.put(&pos_n);
            fill.put(&filled);
            pages.put(&pages_n);
            page_indptr.put(&pidx_n);
            rng.put(&r_next);
            pool_ids_ch.put(&pids);
        });
        let budget = if done.iter().any(|d| !d) {
            max_tokens.saturating_sub(1) // the prefill's g0s already emitted
        } else {
            0
        };
        run_ahead(&pipe, &fwd, budget, async || {
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
