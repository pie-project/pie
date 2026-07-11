//! Hierarchical-attention inferlet.
//!
//! A standalone example of **application-controlled, structured attention**. The
//! repo already ships `attention-sink` (keep the first few tokens + a recent
//! window) and `windowed-attention` (keep only a recent window). This inferlet
//! keeps a *hierarchy* visible during generation:
//!
//! ```text
//!   global instructions            (sink tokens at the very start)
//!     chunk summaries              (a short header range per chunk)
//!       selected chunk detail      (the full body of the most-recent chunk(s))
//!         recent generated text    (a sliding local window at the end)
//! ```
//!
//! **A4 PTIR rewrite** (classic `forward-pass` retirement). The classic
//! `attention_mask(list<brle>)` func + the `carrier::submit_pass_with` keep-core
//! decode helper are gone. This authors the decode loop directly on the
//! `inferlet::ptir` bridge, on the SAME device-proven one-token-per-pass B=1
//! explicit-write geometry as `windowed-attention`/`attention-sink`, with the
//! hierarchical `attn_mask` evolved IN-GRAPH in the epilogue. For the query at
//! flat position `p`, KV position `j` (over the fixed pool) is admitted iff
//! `j <= p` (causal) AND any of:
//!   * `j < sink`                          — the global-instruction sink,
//!   * `j % chunk < summary`               — every chunk's header (summary),
//!   * `j + window > p`                     — the recent local window,
//!   * `j/chunk + 1 == p/chunk` (± `selected`) — the most-recently-COMPLETED
//!     `selected` chunk bodies.
//! Everything else is masked. KV pages are never freed (no eviction), only masked.
//!
//! DESIGN NOTE (why in-graph, not a host-written mask): a hierarchical keep-set
//! looks "arbitrary," but once prompt-lexical chunk selection is removed it is a
//! pure function of the query position and the (constant) chunk parameters — so it
//! rides the proven in-graph pattern with no per-pass host mask. The original
//! LEXICAL relevance selection required a real prompt in the KV; the ptir path
//! can't do variable-length prompt prefill (the embed token MUST be device
//! loop-carried — the runtime resolves EmbedTokens from the device channel value,
//! and a host-fed embed leaves token_ids empty), so the mask is demonstrated over
//! the GENERATED sequence with recency-based chunk selection. A genuinely arbitrary
//! per-pass HOST-WRITTEN attention mask is a DISTINCT, currently-unproven device
//! pattern (no existing inferlet needs it) — see plan.md.

use inferlet::ptir::prelude::*;
use inferlet::{chat, model as wit_model, Result};
use serde::Deserialize;
use std::time::Instant;

#[derive(Debug, Clone, Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Tokens per chunk over the generated sequence (was word-chunking of the
    /// prompt; prompt prefill is unsupported on the ptir path — see the note).
    #[serde(default = "default_chunk_tokens")]
    chunk_size_words: u32,
    #[serde(default = "default_sink")]
    sink_tokens: u32,
    #[serde(default = "default_summary")]
    summary_tokens_per_chunk: u32,
    #[serde(default = "default_window")]
    local_window_tokens: u32,
    #[serde(default = "default_selected_chunks")]
    selected_chunks: u32,
    #[serde(default = "default_selection_mode")]
    selection_mode: String,
}

fn default_prompt() -> String { "Tell me a long story about a cat.".to_string() }
fn default_max_tokens() -> usize { 128 }
fn default_chunk_tokens() -> u32 { 8 }
fn default_sink() -> u32 { 2 }
fn default_summary() -> u32 { 1 }
fn default_window() -> u32 { 8 }
fn default_selected_chunks() -> u32 { 1 }
fn default_selection_mode() -> String { "recent".into() }

const PAGE_T: u32 = 16; // tokens per pool page
const NUM_LAYERS: u32 = 28; // Qwen3-0.6B
const MAX_SEQ: u32 = 1024; // pool capacity (KV kept; the mask only masks)
const POOL_PAGES: u32 = MAX_SEQ / PAGE_T; // physical pool pages
const POOL: u32 = POOL_PAGES * PAGE_T; // flat pool token positions

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// Seed hierarchical mask for the first query (flat position 0): only KV position
/// 0 is present, so exactly one cell is admitted.
fn seed_mask() -> Vec<bool> {
    (0..POOL).map(|j| j == 0).collect()
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let start = Instant::now();

    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);

    // Mask parameters (constants captured by the epilogue; the whole keep-set is a
    // pure function of the query position and these).
    let sink = input.sink_tokens;
    let summary = input.summary_tokens_per_chunk;
    let window = input.local_window_tokens.max(1);
    let chunk = input.chunk_size_words.max(1);
    let selected = input.selected_chunks.max(1);

    let mut prompt: Vec<u32> = Vec::new();
    prompt.extend(chat::system("You are a concise assistant."));
    prompt.extend(chat::user(&input.prompt));
    prompt.extend(chat::cue());
    let stop_tokens = chat::stop_tokens();

    if prompt.is_empty() {
        return Err("empty prompt after chat templating".to_string());
    }
    if input.max_tokens as u32 >= POOL {
        return Err(format!(
            "max_tokens ({}) exceeds hierarchical-attention pool ({POOL})",
            input.max_tokens
        ));
    }

    println!(
        "--- hierarchical-attention-rust (chunk={chunk}, sink={sink}, summary={summary}, window={window}, selected={selected}, mode={}) ---",
        input.selection_mode,
    );

    // Fixed physical page pool: allocate POOL_PAGES real page slots ONCE. The flat
    // pool position `p` maps to physical page `pool_ids[p / PAGE_T]` at offset
    // `p % PAGE_T`. KV is never evicted; the hierarchical mask does all attention
    // restriction.
    let ws: &'static WorkingSet = bx(WorkingSet::new());
    let pool = ws.reserve(POOL_PAGES).map_err(|e| format!("ws.reserve pool: {e}"))?;
    let pool_ids: &'static Vec<u32> = bx(pool.ids().to_vec()); // [POOL_PAGES] physical
    let phys0 = pool_ids[0];

    // Loop-carried decode state, DEVICE loop-carried exactly like windowed-
    // attention: every channel is guest-SEEDED for fire-0 and the epilogue re-emits
    // it for the next pass as a pure function of the pass index. The embed token
    // MUST be device loop-carried. `fill` = the next free flat pool position = the
    // position the NEXT fire writes/queries.
    let seed_tok = prompt[0] as i32;
    let tok_in = bx(Channel::from(vec![seed_tok; 1]).named("tok_in")); // device loop-carried
    let pos = bx(Channel::from(vec![0u32; 1]).named("pos"));
    let fill = bx(Channel::from(vec![1u32; 1]).named("fill")); // next free flat position
    let klen = bx(Channel::from(vec![1u32; 1]).named("klen"));
    let mask = bx(Channel::from_shaped([1, POOL], seed_mask()).named("mask")); // [1,POOL] bool
    let w_slot = bx(Channel::from(vec![phys0; 1]).named("w_slot")); // physical page id
    let w_off = bx(Channel::from(vec![0u32; 1]).named("w_off"));
    let pages = bx(Channel::from(pool_ids.clone()).named("pages")); // [POOL_PAGES] physical
    let page_indptr = bx(Channel::from_shaped([2], vec![0u32, POOL_PAGES]).named("page_indptr"));
    // Physical pool ids [POOL_PAGES], host-fed each fire, gathered in-graph to map
    // a flat pool-page index -> physical page id for the write descriptor.
    let pool_ids_ch = bx(Channel::new([POOL_PAGES], dtype::u32).named("pool_ids"));
    let out = bx(Channel::new([1], dtype::i32).named("out"));

    // One token per pass (single lane): embed indptr = [0, 1].
    let lane1 = Tensor::constant(vec![0u32, 1u32]);

    let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd.embed(tok_in, lane1);
    fwd.positions(pos);
    fwd.attn_working_set(ws, klen);
    fwd.port_channel(Port::Pages, pages);
    fwd.port_channel(Port::PageIndptr, page_indptr);
    fwd.port_channel(Port::WSlot, w_slot);
    fwd.port_channel(Port::WOff, w_off);
    fwd.attn_mask(mask);
    fwd.epilogue(move || {
        // Takes-first / puts-last discipline (beam-designb/windowed). `base` = fill
        // = the flat position THIS next fire writes/queries.
        let base = fill.take().tensor(); // [1] u32 — next free flat position
        let pids = pool_ids_ch.take(); // [POOL_PAGES] physical page ids

        // 1. greedy argmax → the next input token (device loop-carried) + host read.
        let tok = reduce_argmax(intrinsics::logits()); // [1] i32

        // 2. hierarchical mask for the query at position `base`, over pool columns
        //    j in [0, POOL): causal AND (sink OR header OR window OR selected body).
        let col = iota(POOL); // [POOL] u32
        let base_b = broadcast(reshape(&base, [1]), [POOL]); // [POOL]
        let causal = le(&col, &base_b); // j <= base
        let is_sink = lt(&col, sink); // j < sink
        let is_header = lt(rem(&col, chunk), summary); // (j % chunk) < summary
        let in_window = gt(add(&col, window), &base_b); // j + window > base
        // selected: the `selected` most-recently-COMPLETED chunks, i.e. chunk
        // indices [base/chunk - selected, base/chunk - 1]. `add` avoids u32
        // underflow when base < chunk (then base/chunk = 0 → no completed chunk).
        let jchunk = div(&col, chunk); // j / chunk
        let basechunk = div(&base_b, chunk); // base / chunk
        let is_sel = and(
            lt(&jchunk, &basechunk), // completed (strictly before the current chunk)
            ge(add(&jchunk, selected), &basechunk), // within the last `selected`
        );
        let admit = or(or(&is_sink, &is_header), or(&in_window, &is_sel));
        let new_mask = reshape(and(&causal, &admit), [1, POOL]); // [1,POOL] bool

        // 3. explicit write descriptor. WSlot is a PHYSICAL page id: map the flat
        //    pool-page index (base / PAGE_T) through the host-fed physical pool ids.
        let logical_slot = div(&base, PAGE_T); // [1] pool page index
        let w_slot_v = gather(&pids, &logical_slot); // [1] physical page id
        let w_off_v = rem(&base, PAGE_T); // [1]

        // 4. KV physical extent after this fire writes: positions 0..=base present.
        let klen_v = add(&base, 1u32);
        let next_free = add(&base, 1u32); // next pass's fill

        // 5. re-emit the fixed pool geometry each fire (page_indptr as iota(2)*
        //    POOL_PAGES so it is a compute NODE, not a lazy vector const).
        let pages_v = reshape(&pids, [POOL_PAGES]);
        let pidx_v = mul(&iota(2), POOL_PAGES); // [0, POOL_PAGES]

        // -- puts last --
        tok_in.put(&tok);
        out.put(&tok);
        mask.put(&new_mask);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);
        klen.put(&klen_v);
        pos.put(&base);
        fill.put(&next_free);
        pages.put(&pages_v);
        page_indptr.put(&pidx_v);
    });

    // Generation loop: fire-0 embeds the seed token; the epilogue feeds each argmax
    // back into `tok_in` (device loop-carried) and evolves the hierarchical mask +
    // geometry in-graph. The host only re-supplies the physical pool ids and reads
    // out each committed token.
    let pipeline = Pipeline::new();
    let mut generated_tokens: Vec<u32> = Vec::new();

    for _ in 0..input.max_tokens {
        pool_ids_ch.put(pool_ids.clone());
        fwd.submit(&pipeline).map_err(|e| format!("submit: {e}"))?;
        let sampled = out
            .take()
            .get::<i32>()
            .map_err(|e| format!("out.take: {e}"))?;
        let token = match sampled.first() {
            Some(&t) => t as u32,
            None => break,
        };
        if stop_tokens.contains(&token) {
            break;
        }
        generated_tokens.push(token);
    }

    let text = wit_model::decode(&generated_tokens)?;
    println!(
        "Generated {} tokens in {:?}",
        generated_tokens.len(),
        start.elapsed()
    );
    println!("Output:\n{text}");

    let preview: Vec<u32> = generated_tokens.iter().copied().take(8).collect();
    let result = format!(
        "HIERARCHICAL_ATTENTION chunk={chunk} sink={sink} window={window} selected={selected} generated={} tokens={preview:?}",
        generated_tokens.len(),
    );
    println!("HIERARCHICAL_ATTENTION_E2E {result}");
    Ok(result)
}
