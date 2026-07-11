//! Demonstrates attention sink — bounded KV cache with preserved initial
//! tokens, on PTIR.
//!
//! Maintains an "attention sink" of initial tokens plus a sliding window of the
//! most recent tokens. Tokens between the sink and the window are masked via a
//! per-step attention mask, preventing the model from attending to them.
//!
//! NOTE: full KV cache eviction is not yet supported by the runtime — the mask
//! only prevents the model from *attending* to masked tokens; the KV pages stay
//! in memory.
//!
//! **A4 PTIR rewrite** (classic `forward-pass` retirement). The classic
//! `attention_mask(list<brle>)` func + the `carrier::submit_pass_with` keep-core
//! decode helper are gone. This authors the decode loop directly on the
//! `inferlet::ptir` bridge, mirroring `windowed-attention`: a fixed
//! one-token-per-pass `ForwardPass` (device-proven explicit-write wire-form,
//! B=1) over a fixed physical page POOL. All loop-carried state is DEVICE
//! loop-carried (guest-seeded for fire-0, re-emitted by the epilogue) — the embed
//! token feeds the epilogue's argmax back into `tok_in` (generation from a seed
//! token), and ALL geometry — position, KV length, write descriptor, and the
//! sink+window `attn_mask` — evolves IN-GRAPH as a pure function of the pass
//! index. The mask for the query at flat position `p` admits KV position `j` iff
//! `j <= p` AND (`j < sink` OR `j + window > p`): the initial sink tokens plus the
//! most recent `window`. KV pages are never freed (no eviction), only masked out.
//!
//! NOTE: the embed token MUST be device loop-carried (the runtime resolves the
//! EmbedTokens port from the device channel value; a host-fed embed leaves the
//! token_ids empty and KV-prepare rejects it). Host-injected prompt prefill on the
//! ptir token-at-a-time path is a separate, unproven concern; this demonstrates
//! the sink+window geometry via generation from a seed token.

use inferlet::ptir::prelude::*;
use inferlet::{chat, model as wit_model, Result};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_sink_size")]
    sink_size: u32,
    #[serde(default = "default_window_size")]
    window_size: u32,
}

fn default_prompt() -> String { "Tell me a long story about a cat.".to_string() }
fn default_max_tokens() -> usize { 512 }
fn default_sink_size() -> u32 { 4 }
fn default_window_size() -> u32 { 64 }

const PAGE_T: u32 = 16; // tokens per pool page
const NUM_LAYERS: u32 = 28; // Qwen3-0.6B
const MAX_SEQ: u32 = 1024; // pool capacity (KV kept; the mask only masks)
const POOL_PAGES: u32 = MAX_SEQ / PAGE_T; // physical pool pages
const POOL: u32 = POOL_PAGES * PAGE_T; // flat pool token positions

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// Seed sink+window mask for the first query (flat position 0): only KV
/// position 0 is present, so exactly one cell is admitted.
fn seed_mask() -> Vec<bool> {
    (0..POOL).map(|j| j == 0).collect()
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let start = Instant::now();

    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);
    let sink = input.sink_size;
    let window = input.window_size.max(1);

    let mut prompt: Vec<u32> = Vec::new();
    prompt.extend(chat::system("You are a helpful assistant."));
    prompt.extend(chat::user(&input.prompt));
    prompt.extend(chat::cue());
    let stop_tokens = chat::stop_tokens();

    if prompt.len() as u32 >= POOL {
        return Err(format!(
            "prompt ({} tokens) exceeds attention-sink pool ({POOL})",
            prompt.len()
        ));
    }

    println!(
        "--- Attention Sink (sink={sink}, window={window}, page_size={PAGE_T}, pool={POOL}) ---",
    );

    // Fixed physical page pool: allocate POOL_PAGES real page slots ONCE. The
    // flat pool position `p` maps to physical page `pool_ids[p / PAGE_T]` at
    // offset `p % PAGE_T`. KV is never evicted; the mask does all attention
    // restriction.
    let ws: &'static WorkingSet = bx(WorkingSet::new());
    let pool = ws.reserve(POOL_PAGES).map_err(|e| format!("ws.reserve pool: {e}"))?;
    let pool_ids: &'static Vec<u32> = bx(pool.ids().to_vec()); // [POOL_PAGES] physical
    let phys0 = pool_ids[0];

    // Loop-carried decode state, DEVICE loop-carried exactly like the beam-designb
    // reference: every channel is guest-SEEDED for fire-0 and the epilogue re-emits
    // it for the next pass as a pure function of the pass index. The embed token
    // MUST be device loop-carried (the runtime resolves EmbedTokens from the device
    // channel value; a host-fed embed leaves token_ids empty). `fill` = the next
    // free flat pool position = the position the NEXT fire writes/queries.
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
        // Structured exactly like the device-proven beam-designb epilogue: take
        // all loop-carried inputs first, compute, then PUT everything last (an
        // early put interleaved with compute — or a lazy vector Tensor::constant
        // materialized at put-time — desyncs the traced value-id stream).
        //
        // `base` = fill = the flat position THIS next fire writes/queries. Every
        // descriptor is a pure function of `base`.
        let base = fill.take().tensor(); // [1] u32 — next free flat position
        let pids = pool_ids_ch.take(); // [POOL_PAGES] physical page ids

        // 1. greedy argmax over this query's logits [1, V] → the next input token
        //    (device loop-carried into `tok_in`) and the host-read output.
        let tok = reduce_argmax(intrinsics::logits()); // [1] i32

        // 2. sink+window mask for the query at position `base`: admit KV position j
        //    iff (j <= base) AND (j < sink OR j + window > base) — the initial sink
        //    tokens plus the most recent `window`.
        let col = iota(POOL); // [POOL] u32
        let base_b = broadcast(reshape(&base, [1]), [POOL]); // [POOL]
        let causal = le(&col, &base_b); // j <= base
        let is_sink = lt(&col, sink); // j < sink
        let in_window = gt(add(&col, window), &base_b); // j + window > base
        let new_mask = reshape(and(&causal, or(&is_sink, &in_window)), [1, POOL]); // [1,POOL]

        // 3. explicit write descriptor. WSlot is a PHYSICAL page id: map the flat
        //    pool-page index (base / PAGE_T) through the host-fed physical pool ids.
        let logical_slot = div(&base, PAGE_T); // [1] pool page index
        let w_slot_v = gather(&pids, &logical_slot); // [1] physical page id
        let w_off_v = rem(&base, PAGE_T); // [1]

        // 4. KV physical extent after this fire writes: positions 0..=base present.
        let klen_v = add(&base, 1u32);
        let next_free = add(&base, 1u32); // next pass's fill

        // 5. re-emit the fixed pool geometry each fire (page_indptr as iota(2)*
        //    POOL_PAGES so it is a compute-section NODE, not a lazy vector const).
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
    // back into `tok_in` (device loop-carried) and evolves the sink+window mask +
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
        "ATTENTION_SINK sink={sink} window={window} generated={} tokens={preview:?}",
        generated_tokens.len()
    );
    println!("ATTENTION_SINK_E2E {result}");
    Ok(result)
}
