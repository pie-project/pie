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
//! A long user turn is split into word chunks. We map those chunks onto the
//! tokenized turn, treat the first few tokens of each chunk as its summary,
//! select the chunk(s) most relevant to the task by lexical overlap, then run a
//! **manual decode loop**. At each step we build a BRLE attention mask whose
//! *true* runs are exactly: the sink, every chunk summary, the selected chunk
//! detail, and the recent window. Everything else is masked.
//!
//! How this differs from the existing examples:
//!
//! ```text
//!   attention-sink         = first sink tokens            + recent window
//!   windowed-attention     = recent window only
//!   hierarchical-attention = sink + ALL chunk summaries + selected full chunk + recent window
//! ```
//!
//! MVP scope: "summaries" are the first N tokens of each chunk, and relevance
//! is lexical overlap. The runtime clips each query's hierarchy mask to its
//! causal bound. The point is to demonstrate Pie's programmable attention-mask
//! interface
//! (`ctx.forward()` + `pass.attention_mask(...)`), not to ship a tuned policy.
//! No speedup is claimed — masked KV pages still occupy memory; the mask only
//! controls what the model *attends to*.

use inferlet::{Context, Result, model::Model, runtime, sample::Sampler};
use serde::Deserialize;
use std::collections::HashSet;

// Large per-query custom-mask prefills can fail as an empty driver future.
const PREFILL_CHUNK_TOKENS: usize = 256;

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
fn default_max_tokens() -> usize {
    128
}
fn default_chunk_words() -> usize {
    80
}
fn default_sink() -> u32 {
    64
}
fn default_summary() -> u32 {
    24
}
fn default_window() -> u32 {
    128
}
fn default_selected_chunks() -> usize {
    1
}
fn default_selection_mode() -> String {
    "lexical".into()
}

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

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let model = Model::load(runtime::models().first().ok_or("No models available")?)?;
    let stop_tokens = inferlet::chat::stop_tokens(&model);
    let (system_text, user_text) = split_chat_prompt(&input.prompt);

    // Minimum chunk size guards against degenerate 1-word chunks blowing up the
    // range bookkeeping.
    let chunk_words = input.chunk_size_words.max(8);
    let chunks = split_words(user_text, chunk_words);

    let selected = select_chunks(
        &chunks,
        user_text,
        input.selected_chunks.max(1),
        &input.selection_mode,
    )?;

    // Keep the model-facing prompt as one valid chat turn. The prior
    // implementation emitted adjacent user turns for every header and body,
    // which caused chat models to immediately emit a stop token.
    let mut prompt_tokens: Vec<u32> = Vec::new();
    prompt_tokens.extend(inferlet::chat::system(&model, system_text));

    let user_start = prompt_tokens.len() as u32;
    let user_tokens = inferlet::chat::user(&model, user_text);
    let user_len = user_tokens.len() as u32;
    prompt_tokens.extend(user_tokens);

    // Map the lexical word chunks onto contiguous ranges of the single
    // tokenized user turn. Boundaries are proportional because tokenizer
    // pieces do not correspond one-to-one with whitespace-separated words.
    let full_ranges = partition_range(user_start, user_len, chunks.len());
    let summary_ranges: Vec<Range> = full_ranges
        .iter()
        .map(|range| {
            Range::new(
                range.start,
                range.start + input.summary_tokens_per_chunk.min(range.end - range.start),
            )
        })
        .collect();

    prompt_tokens.extend(inferlet::chat::cue(&model));

    println!("--- hierarchical-attention-rust ---");
    println!("chunks={}", chunks.len());
    println!(
        "selected_chunk={:?} (mode={})",
        selected, input.selection_mode
    );
    println!("summary_ranges={}", fmt_ranges(&summary_ranges));
    println!("full_ranges={}", fmt_ranges(&full_ranges));

    let mut ctx = Context::new(&model)?;
    let mut logged_mask = false;

    // Keep custom-mask prefills bounded. This also gives the host a chance to
    // validate and commit KV lineage before processing the rest of a long
    // multi-step agent transcript.
    let mut prompt_offset = 0;
    while prompt_tokens.len() - prompt_offset > PREFILL_CHUNK_TOKENS {
        let end = prompt_offset + PREFILL_CHUNK_TOKENS;
        let chunk = &prompt_tokens[prompt_offset..end];
        let mut fwd = ctx.forward();
        let start_position = fwd.start_position();
        let masks = build_hierarchical_masks(
            start_position,
            chunk.len(),
            input.sink_tokens,
            &summary_ranges,
            &full_ranges,
            &selected,
            input.local_window_tokens,
        );
        if !logged_mask {
            println!(
                "mask_true_tokens={} / total={}",
                mask_true_count(masks.last().expect("chunk is non-empty")),
                start_position + chunk.len() as u32
            );
            logged_mask = true;
        }
        fwd.input(chunk);
        fwd.attention_mask(&masks);
        fwd.execute().await?;
        prompt_offset = end;
    }

    let mut pending = prompt_tokens[prompt_offset..].to_vec();
    let mut generated: Vec<u32> = Vec::new();

    for _ in 0..input.max_tokens {
        if pending.is_empty() {
            break;
        }

        let mut fwd = ctx.forward();
        let start_position = fwd.start_position();
        let total_seq_after = start_position + pending.len() as u32;
        let masks = build_hierarchical_masks(
            start_position,
            pending.len(),
            input.sink_tokens,
            &summary_ranges,
            &full_ranges,
            &selected,
            input.local_window_tokens,
        );

        // Log the first step's mask occupancy so the effect is visible without
        // spamming one line per generated token.
        if !logged_mask {
            println!(
                "mask_true_tokens={} / total={}",
                mask_true_count(masks.last().expect("pending is non-empty")),
                total_seq_after
            );
            logged_mask = true;
        }

        fwd.input(&pending);
        fwd.attention_mask(&masks);

        let h = fwd.sample(&[(pending.len() - 1) as u32], Sampler::Argmax);
        let out = fwd.execute().await?;
        let token = match out.token(h) {
            Some(t) => t,
            None => return Err("hierarchical-attention: sampler returned no token".into()),
        };
        if stop_tokens.contains(&token) {
            break;
        }

        generated.push(token);
        pending = vec![token];
    }

    println!("generated_tokens_total={}", generated.len());
    Ok(model.tokenizer().decode(&generated)?)
}

// =============================================================================
// Pure helpers (unit-tested)
// =============================================================================

/// Split text into chunks of at most `chunk_words` whitespace-separated words.
/// An empty prompt yields a single empty chunk so downstream range bookkeeping
/// always has something to point at.
fn split_chat_prompt(prompt: &str) -> (&str, &str) {
    prompt
        .split_once("\n\n")
        .unwrap_or(("You are a concise assistant.", prompt))
}

fn partition_range(start: u32, len: u32, parts: usize) -> Vec<Range> {
    let parts = parts.max(1);
    (0..parts)
        .filter_map(|index| {
            let range_start = start + len * index as u32 / parts as u32;
            let range_end = start + len * (index + 1) as u32 / parts as u32;
            (range_start < range_end).then(|| Range::new(range_start, range_end))
        })
        .collect()
}

fn split_words(text: &str, chunk_words: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![String::new()];
    }
    let n = chunk_words.max(1);
    words.chunks(n).map(|w| w.join(" ")).collect()
}

fn select_chunks(chunks: &[String], query: &str, k: usize, mode: &str) -> Result<Vec<usize>> {
    match mode {
        "lexical" => Ok(select_relevant_chunks(chunks, query, k)),
        "all-visible" | "all-visible-baseline" => Ok((0..chunks.len()).collect()),
        other => Err(format!(
            "unsupported selection_mode {other:?}; expected lexical or all-visible-baseline"
        )),
    }
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

fn build_hierarchical_masks(
    start_position: u32,
    pending_len: usize,
    sink_tokens: u32,
    summary_ranges: &[Range],
    full_ranges: &[Range],
    selected: &[usize],
    local_window_tokens: u32,
) -> Vec<Vec<u32>> {
    let total = start_position + pending_len as u32;
    let mut keep = vec![Range::new(0, sink_tokens.min(total))];
    keep.extend(summary_ranges.iter().copied());
    for &index in selected {
        if let Some(range) = full_ranges.get(index) {
            keep.push(*range);
        }
    }
    keep.push(Range::new(total.saturating_sub(local_window_tokens), total));

    // The runtime expects one full-request mask per query token. It applies
    // each query position's causal bound after decoding the BRLE mask.
    let mask = build_brle_mask(total, &keep);
    vec![mask; pending_len]
}

/// Number of *attended* positions in a BRLE mask = sum of the true runs (the
/// odd-indexed entries).
fn mask_true_count(mask: &[u32]) -> u32 {
    mask.iter().skip(1).step_by(2).sum()
}

fn fmt_ranges(ranges: &[Range]) -> String {
    let parts: Vec<String> = ranges
        .iter()
        .map(|r| format!("[{},{})", r.start, r.end))
        .collect();
    format!("[{}]", parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_chat_prompt_preserves_roles() {
        assert_eq!(
            split_chat_prompt("system rules\n\nuser task"),
            ("system rules", "user task")
        );
        assert_eq!(
            split_chat_prompt("user-only task"),
            ("You are a concise assistant.", "user-only task")
        );
    }

    #[test]
    fn partition_range_is_contiguous_and_complete() {
        assert_eq!(
            partition_range(10, 10, 3),
            vec![Range::new(10, 13), Range::new(13, 16), Range::new(16, 20)]
        );
    }

    #[test]
    fn partition_range_drops_empty_partitions() {
        assert_eq!(
            partition_range(5, 2, 4),
            vec![Range::new(5, 6), Range::new(6, 7)]
        );
    }

    #[test]
    fn selection_mode_controls_visible_chunks() {
        let chunks = vec!["alpha beta".into(), "gamma delta".into()];
        assert_eq!(
            select_chunks(&chunks, "gamma", 1, "lexical").unwrap(),
            vec![1]
        );
        assert_eq!(
            select_chunks(&chunks, "gamma", 1, "all-visible-baseline").unwrap(),
            vec![0, 1]
        );
        assert!(select_chunks(&chunks, "gamma", 1, "unknown").is_err());
    }

    #[test]
    fn hierarchical_prefill_masks_cover_the_full_request() {
        let summaries = vec![Range::new(2, 4), Range::new(8, 10)];
        let full = vec![Range::new(2, 8), Range::new(8, 14)];
        let masks = build_hierarchical_masks(10, 3, 2, &summaries, &full, &[1], 4);

        assert_eq!(masks.len(), 3);
        assert_eq!(
            masks
                .iter()
                .map(|mask| mask.iter().sum::<u32>())
                .collect::<Vec<_>>(),
            vec![13, 13, 13]
        );
    }
}
