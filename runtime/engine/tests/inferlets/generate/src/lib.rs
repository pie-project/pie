//! **Harness decode guest — PTIR rewrite** (bravo). Was: raw classic-WIT
//! `inference::ForwardPass` + a `sampling::Graph` argmax program; both surfaces
//! are deleted, so this drives the same observable loop on the `inferlet::ptir`
//! bridge (the templates' wire form): an N-wide prompt-prefill fire followed by
//! a 1-token-per-pass decode loop with author-bound, loop-carried `KvLen`
//! (the host projects the working-set pages into every launch — exactly the
//! per-request `kv_page_indices` runs the `kv_retention` probe inspects).
//!
//! TOKEN SEMANTICS. The classic mock (`EchoBehavior(42)`) fabricated the
//! `ForwardResponse` and echoed 42 as every fire's sampled token — the old
//! argmax program never actually ran on these suites. The direct dummy driver
//! executes the guest's own epilogue instead (its `deterministic_logits` argmax
//! varies per instance/fire and its vocab is 32, so no logits program can yield
//! a constant 42). The echo constant therefore moves IN-GRAPH (the
//! `direct-channel-e2e` idiom): a device loop-carried channel seeded 42 that
//! every epilogue re-emits and feeds back as the next embed token. The suites'
//! determinism oracle — every concurrent pipeline returns exactly
//! `generated 5 tokens: [42, 42, 42, 42, 42]`, divergence ⇒ a host race — is
//! unchanged.
//!
//! Input: an optional token budget (default 5), e.g. `"16"` or `{"lane":N}`
//! (ignored → default).

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const DEFAULT_MAX_TOKENS: usize = 5;

/// The per-fire "sampled" token (the mock-era `EchoBehavior(42)` constant,
/// carried in-graph — see the module docs).
const ECHO_TOKEN: i32 = 42;

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    // guru's discriminator: NAME the launch-input encoding (rules out stale-wasm /
    // quote-wrap — a bare int here ⇒ the budget reached the inferlet live).
    eprintln!("[generate] input={input:?}");
    // Bare-integer input → budget; anything else (e.g. `{"lane":N}`) → default.
    let max_tokens: usize = input.trim().parse().unwrap_or(DEFAULT_MAX_TOKENS);

    let ws = WorkingSet::new();
    let page_size = ws.page_size();

    if max_tokens == 0 {
        let result = "generated 0 tokens: []".to_string();
        eprintln!("[GENERATE] {result}");
        return Ok(result);
    }

    let prompt = wit_model::encode("hello world");
    let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(ws.page_size());
    let reserve_to_tokens = |tokens: u32| -> std::result::Result<(), String> {
        let target = tokens.div_ceil(page_size).saturating_add(1).min(max_pages);
        let current = ws.page_len();
        if current < target {
            ws.reserve(target - current)?;
        }
        Ok(())
    };
    reserve_to_tokens(n.max(1)).map_err(|e| format!("ws.reserve prompt: {e}"))?;

    // ───────────────────────── 1. PREFILL FIRE (N-wide) ─────────────────────
    // Seeded prompt and every descriptor channel are explicit. The pages
    // channel spans the declared pool; page_indptr selects its live prefix.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p"); // [N] i32 (seeded)
    let embed_indptr_p = Channel::from(vec![0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from((0..n).collect::<Vec<_>>()).named("positions_p");
    let pages_p = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages_p");
    let page_indptr_p = Channel::from(vec![0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p = Channel::from(
        (0..n)
            .map(|position| position / page_size)
            .collect::<Vec<_>>(),
    )
    .named("w_slot_p");
    let w_off_p = Channel::from(
        (0..n)
            .map(|position| position % page_size)
            .collect::<Vec<_>>(),
    )
    .named("w_off_p");
    let echo_p = Channel::from(vec![ECHO_TOKEN; 1]).named("echo_p"); // [1] i32 seed
    let g0_ch = Channel::new([1], dtype::i32).named("g0"); // host-read first gen token

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    let kv_len_p = Channel::from(vec![n]).named("kv_len_p");
    fwd_p.attention(
        &ws,
        ..,
        ..,
        &kv_len_p,
        &pages_p,
        &page_indptr_p,
        &w_slot_p,
        &w_off_p,
        &positions_p,
        None,
    )?;
    fwd_p.epilogue(move || {
        // The fire's "sampled" token = the loop-carried echo constant.
        let t = echo_p.take().tensor(); // [1] i32
        g0_ch.put(&t);
    });

    // ONE pipeline for the whole prefill→decode stream.
    let pipe = Pipeline::new();
    fwd_p
        .submit(&pipe)
        .map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = g0_ch
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("g0 take: {e}"))?[0];

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    generated.push(g0 as u32);

    // ───────────────────────── 2. DECODE LOOP (1-wide) ──────────────────────
    // Seeded with g0 at position N, one committed token per fire. All
    // loop-carried state is DEVICE loop-carried (guest-seeded for fire-0,
    // re-emitted by the epilogue): the embed token MUST ride a seeded channel —
    // the runtime resolves EmbedTokens from the device channel value; a
    // host-fed embed leaves `token_ids` empty and KV-prepare rejects it.
    if generated.len() < max_tokens {
        let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
        let echo = Channel::from(vec![ECHO_TOKEN; 1]).named("echo");
        let out = Channel::new([1], dtype::i32).named("out");
        let lane1 = Channel::from(vec![0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from(vec![n]).named("positions");
        let pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages");
        let page_indptr =
            Channel::from(vec![0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from(vec![n / page_size]).named("w_slot");
        let w_off = Channel::from(vec![n % page_size]).named("w_off");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        let kv_len = Channel::from(vec![n + 1]).named("kv_len");
        fwd.attention(
            &ws,
            ..,
            (n / page_size)..,
            &kv_len,
            &pages,
            &page_indptr,
            &w_slot,
            &w_off,
            &positions,
            None,
        )?;
        fwd.epilogue(move || {
            let length = kv_len.take().tensor();
            let t = echo.take().tensor(); // [1] i32 — the echo token
            let next_length = add(&length, 1u32);
            let page_count = div(add(&next_length, page_size - 1), page_size);
            let next_page_indptr = mul(iota(2), broadcast(&page_count, [2]));
            tok_in.put(&t);
            echo.put(&t);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(div(&length, page_size));
            w_off.put(rem(&length, page_size));
            page_indptr.take();
            page_indptr.put(&next_page_indptr);
            out.put(&t);
        });

        for step in 1..max_tokens {
            reserve_to_tokens(n + step as u32)
                .map_err(|e| format!("reserve decode @{step}: {e}"))?;
            fwd.submit(&pipe)
                .map_err(|e| format!("decode submit @{step}: {e}"))?;
            let t = out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("out.take @{step}: {e}"))?;
            let Some(&t0) = t.first() else {
                return Err(format!("out.take @{step}: empty tensor"));
            };
            generated.push(t0 as u32);
        }
    }
    pipe.close();

    let result = format!("generated {} tokens: {:?}", generated.len(), generated);
    eprintln!("[GENERATE] {result}");
    Ok(result)
}
