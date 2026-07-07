//! Hierarchical-attention inferlet.
//!
//! A standalone example of **application-controlled, structured attention**.
//! The repo already ships `attention-sink` (keep the first few tokens + a
//! recent window) and `windowed-attention` (keep only a recent window). This
//! inferlet keeps a *hierarchy* visible during generation:
//!
//! ```text
//!   global instructions            (sink tokens at the very start)
//!     chunk summaries              (a short header range per chunk)
//!       selected chunk detail      (the full body of the most relevant chunk)
//!         recent generated text    (a sliding local window at the end)
//! ```
//!
//! A long prompt is split into word chunks. Each chunk contributes a short
//! header (its "summary") and a full body. We record the token ranges of every
//! header and body, pick the chunk(s) most relevant to the task by lexical
//! overlap, then run a **manual decode loop**. At each step we build a BRLE
//! attention mask whose *true* runs are exactly: the sink, every chunk header,
//! the selected chunk body, and the recent window. Everything else is masked.
//!
//! How this differs from the existing examples:
//!
//! ```text
//!   attention-sink         = first sink tokens            + recent window
//!   windowed-attention     = recent window only
//!   hierarchical-attention = sink + ALL chunk summaries + selected full chunk + recent window
//! ```
//!
//! MVP scope: "summaries" are the first N tokens of each chunk header, and
//! relevance is lexical overlap. One shared mask is applied to every query
//! token in a pass (the pseudocode's documented simplification). The point is
//! to demonstrate Pie's programmable attention-mask interface
//! (`ctx.forward()` + `pass.attention_mask(...)`), not to ship a tuned policy.
//! No speedup is claimed — masked KV pages still occupy memory; the mask only
//! controls what the model *attends to*.

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, Result};
use serde::Deserialize;
use std::collections::HashSet;

#[derive(Debug, Clone, Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_chunk_words")]
    chunk_size_words: usize,
    #[serde(default = "default_sink")]
    sink_tokens: u32,
    #[serde(default = "default_summary")]
    summary_tokens_per_chunk: u32,
    #[serde(default = "default_window")]
    local_window_tokens: u32,
    #[serde(default = "default_selected_chunks")]
    selected_chunks: usize,
    #[serde(default = "default_selection_mode")]
    selection_mode: String,
}

fn default_prompt() -> String {
    "Explain how LLM serving systems use KV cache, batching, scheduling, and \
     attention masks. Include one practical example."
        .into()
}
fn default_max_tokens() -> usize { 128 }
fn default_chunk_words() -> usize { 80 }
fn default_sink() -> u32 { 64 }
fn default_summary() -> u32 { 24 }
fn default_window() -> u32 { 128 }
fn default_selected_chunks() -> usize { 1 }
fn default_selection_mode() -> String { "lexical".into() }

/// A half-open `[start, end)` token range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Range {
    start: u32,
    end: u32,
}

impl Range {
    fn new(start: u32, end: u32) -> Self {
        Range { start, end }
    }
}

/// Finalize a pass and read its sampled token (the low 4 bytes of the output
/// tensor, LE). `None` on a short/empty tensor.
async fn read_token(pass: ForwardPass) -> Result<Option<u32>> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32)
    } else {
        None
    })
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let stop_tokens = inferlet::chat::stop_tokens();

    // Minimum chunk size guards against degenerate 1-word chunks blowing up the
    // header/body bookkeeping.
    let chunk_words = input.chunk_size_words.max(8);
    let chunks = split_words(&input.prompt, chunk_words);

    let selected = select_relevant_chunks(&chunks, &input.prompt, input.selected_chunks.max(1));

    // Build one prompt token stream, recording the header (summary) and body
    // (full) token ranges for each chunk as we go.
    let mut prompt_tokens: Vec<u32> = Vec::new();
    let mut summary_ranges: Vec<Range> = Vec::new();
    let mut full_ranges: Vec<Range> = Vec::new();

    prompt_tokens.extend(inferlet::chat::system(
        "You are a concise assistant. Use the visible hierarchy: global \
         instructions, the chunk summaries, and the selected local chunk.",
    ));

    for (i, chunk) in chunks.iter().enumerate() {
        let header = format!("Chunk {} summary: {}\n", i, summarize_words(chunk, 20));
        let body = format!("Chunk {} full text:\n{}\n", i, chunk);

        // Header → keep only its first `summary_tokens_per_chunk` tokens as a
        // global summary range.
        let header_start = prompt_tokens.len() as u32;
        prompt_tokens.extend(inferlet::chat::user(&header));
        let header_end = prompt_tokens.len() as u32;
        let summary_end = header_start + input.summary_tokens_per_chunk.min(header_end - header_start);
        summary_ranges.push(Range::new(header_start, summary_end));

        // Body → record the full range so a selected chunk can be kept whole.
        let body_start = prompt_tokens.len() as u32;
        prompt_tokens.extend(inferlet::chat::user(&body));
        let body_end = prompt_tokens.len() as u32;
        full_ranges.push(Range::new(body_start, body_end));
    }

    prompt_tokens.extend(inferlet::chat::user(
        "Answer the original request using the selected local chunk(s) and the \
         global chunk summaries.",
    ));
    prompt_tokens.extend(inferlet::chat::cue());

    println!("--- hierarchical-attention-rust ---");
    println!("chunks={}", chunks.len());
    println!("selected_chunk={:?} (mode={})", selected, input.selection_mode);
    println!("summary_ranges={}", fmt_ranges(&summary_ranges));
    println!("full_ranges={}", fmt_ranges(&full_ranges));

    // Raw keep-core decode (was owned by `Context`): a KV working set + cursor;
    // each step attaches the per-token hierarchical mask in the `submit_pass_with`
    // bind seam (after input_tokens, before the sampler/execute tail).
    let vocab = inferlet::model::output_vocab_size();
    let sampler = sampler::sampler_program(SamplerSpec::Argmax, vocab)?;
    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;
    let mut pending = prompt_tokens;
    let mut generated: Vec<u32> = Vec::new();
    let mut logged_mask = false;

    for _ in 0..input.max_tokens {
        if pending.is_empty() {
            break;
        }

        let total_seq_after = seq_len + pending.len() as u32;

        // Assemble the keep-set for this step.
        let mut keep: Vec<Range> = Vec::new();
        // 1. Sink: the very beginning stays globally visible.
        keep.push(Range::new(0, input.sink_tokens.min(total_seq_after)));
        // 2. Summary/header tokens from every chunk.
        keep.extend(summary_ranges.iter().copied());
        // 3. The full body of each selected chunk.
        for &idx in &selected {
            if let Some(r) = full_ranges.get(idx) {
                keep.push(*r);
            }
        }
        // 4. Recent window over the committed + pending sequence.
        let win_start = total_seq_after.saturating_sub(input.local_window_tokens);
        keep.push(Range::new(win_start, total_seq_after));

        let mask = build_brle_mask(total_seq_after, &keep);

        // Log the first step's mask occupancy so the effect is visible without
        // spamming one line per generated token.
        if !logged_mask {
            println!(
                "mask_true_tokens={} / total={}",
                mask_true_count(&mask),
                total_seq_after
            );
            logged_mask = true;
        }

        // One shared mask per query token in this pass (MVP simplification).
        let masks: Vec<Vec<u32>> = (0..pending.len()).map(|_| mask.clone()).collect();

        let pass = carrier::submit_pass_with(
            &kv,
            &mut seq_len,
            &mut fresh,
            &sampler,
            &pending,
            false, // sequential masked decode — no run-ahead carrier
            |pass| {
                pass.attention_mask(&masks);
            },
        )?;

        let token = match read_token(pass).await? {
            Some(t) => t,
            None => break,
        };
        if stop_tokens.contains(&token) {
            break;
        }

        generated.push(token);
        pending = vec![token];
    }

    println!("generated_tokens={}", generated.len());
    Ok(inferlet::model::decode(&generated)?)
}

// =============================================================================
// Pure helpers (unit-tested)
// =============================================================================

/// Split text into chunks of at most `chunk_words` whitespace-separated words.
/// An empty prompt yields a single empty chunk so downstream range bookkeeping
/// always has something to point at.
fn split_words(text: &str, chunk_words: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![String::new()];
    }
    let n = chunk_words.max(1);
    words.chunks(n).map(|w| w.join(" ")).collect()
}

/// First `n` words of `text` — the stand-in "summary" for a chunk.
fn summarize_words(text: &str, n: usize) -> String {
    text.split_whitespace().take(n).collect::<Vec<_>>().join(" ")
}

/// Pick the `k` chunks with the highest lexical overlap with `query`.
///
/// Overlap = count of query words (longer than 3 chars, lowercased) that appear
/// in the chunk. Ties break toward the earlier chunk. Always returns at least
/// one index (chunk 0) so the keep-set is never empty.
fn select_relevant_chunks(chunks: &[String], query: &str, k: usize) -> Vec<usize> {
    if chunks.is_empty() {
        return vec![];
    }
    let q: HashSet<String> = query
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 3)
        .collect();

    let mut scored: Vec<(usize, usize)> = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let score = c
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .filter(|w| q.contains(w))
                .count();
            (i, score)
        })
        .collect();

    // Sort by score desc, then index asc (stable tie-break toward earlier).
    scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let take = k.clamp(1, chunks.len());
    let mut out: Vec<usize> = scored.into_iter().take(take).map(|(i, _)| i).collect();
    out.sort_unstable(); // keep ranges in positional order
    out
}

/// Clip ranges to `[0, total)`, drop empties, sort, and merge overlapping or
/// touching ranges into a minimal disjoint set.
fn merge_ranges(ranges: &[Range], total: u32) -> Vec<Range> {
    let mut clipped: Vec<Range> = ranges
        .iter()
        .map(|r| Range::new(r.start.min(total), r.end.min(total)))
        .filter(|r| r.start < r.end)
        .collect();
    clipped.sort_by_key(|r| r.start);

    let mut merged: Vec<Range> = Vec::new();
    for r in clipped {
        match merged.last_mut() {
            // `<=` merges touching ranges (e.g. [0,4) and [4,8) -> [0,8)).
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => merged.push(r),
        }
    }
    merged
}

/// Build a BRLE attention mask over `[0, total)` keeping `ranges`.
///
/// BRLE alternates run lengths starting with a **false** run:
/// `[false, true, false, true, ...]`. `[0, total]` means "all true".
///
/// Contract enforced here:
/// * ranges are clipped to `total` and merged (overlaps collapse, no crash on
///   empty ranges),
/// * an empty keep-set returns `[0, total]` (all-true / no restriction) rather
///   than an all-false mask, which the runtime would reject as attending to
///   nothing.
fn build_brle_mask(total: u32, ranges: &[Range]) -> Vec<u32> {
    let merged = merge_ranges(ranges, total);
    if merged.is_empty() {
        // No keep-ranges => impose no restriction (all visible). Never emit a
        // pure all-false mask.
        return vec![0, total];
    }

    let mut out: Vec<u32> = Vec::new();
    let mut cursor = 0u32;
    for r in merged {
        out.push(r.start - cursor); // false run up to this range
        out.push(r.end - r.start); // true run covering this range
        cursor = r.end;
    }
    if cursor < total {
        out.push(total - cursor); // trailing false run
    }
    out
}

/// Number of *attended* positions in a BRLE mask = sum of the true runs (the
/// odd-indexed entries).
fn mask_true_count(mask: &[u32]) -> u32 {
    mask.iter().skip(1).step_by(2).sum()
}

fn fmt_ranges(ranges: &[Range]) -> String {
    let parts: Vec<String> = ranges.iter().map(|r| format!("[{},{})", r.start, r.end)).collect();
    format!("[{}]", parts.join(", "))
}
