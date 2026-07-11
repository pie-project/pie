//! **Harness decode guest — PTIR rewrite** (bravo). Was: raw classic-WIT
//! `inference::ForwardPass` + a `sampling::Graph` argmax program; both surfaces
//! are deleted, so this drives the same observable loop on the `inferlet::ptir`
//! bridge (the templates' wire form): an N-wide prompt-prefill fire followed by
//! a 1-token-per-pass decode loop, sugar `attn_working_set(&ws, &klen)` arity
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

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    // guru's discriminator: NAME the launch-input encoding (rules out stale-wasm /
    // quote-wrap — a bare int here ⇒ the budget reached the inferlet live).
    eprintln!("[generate] input={input:?}");
    // Bare-integer input → budget; anything else (e.g. `{"lane":N}`) → default.
    let max_tokens: usize = input.trim().parse().unwrap_or(DEFAULT_MAX_TOKENS);

    let vocab = wit_model::output_vocab_size();
    let ws: &'static WorkingSet = bx(WorkingSet::new());
    model::configure(vocab, ws.page_size(), 1);

    if max_tokens == 0 {
        let result = "generated 0 tokens: []".to_string();
        eprintln!("[GENERATE] {result}");
        return Ok(result);
    }

    let prompt = wit_model::encode("hello world");
    let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let n = prompt.len() as u32;

    // ───────────────────────── 1. PREFILL FIRE (N-wide) ─────────────────────
    // Seeded prompt channel → EmbedTokens (seeding, not host `.put`, is the path
    // the runtime resolves for a host-known embed). qo_indptr = [0, N] (one
    // lane, N query rows); positions default to [0..N]; read-out defaults to
    // row N-1. No device-geometry ports: the sugar arity lets the host derive +
    // project the KV pages (`ptir_kv`), which is the surface the harness probes.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = bx(Channel::from(prompt_i32).named("toks_p")); // [N] i32 (seeded)
    let embed_indptr_p = Tensor::constant(vec![0u32, n]); // [2] qo_indptr
    let klen_p = bx(Channel::from(vec![n; 1]).named("klen_p")); // [1] cells after this fire
    let echo_p = bx(Channel::from(vec![ECHO_TOKEN; 1]).named("echo_p")); // [1] i32 seed
    let g0_ch = bx(Channel::new([1], dtype::i32).named("g0")); // host-read first gen token

    let fwd_p: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd_p.embed(toks_p, embed_indptr_p);
    fwd_p.attn_working_set(ws, klen_p);
    fwd_p.epilogue(move || {
        // The fire's "sampled" token = the loop-carried echo constant.
        let t = echo_p.take().tensor(); // [1] i32
        g0_ch.put(&t);
    });

    let prefill = Pipeline::new();
    prefill
        .submit(fwd_p)
        .map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = g0_ch.take().get::<i32>().map_err(|e| format!("g0 take: {e}"))?[0];
    prefill.close();

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    generated.push(g0 as u32);

    // ───────────────────────── 2. DECODE LOOP (1-wide) ──────────────────────
    // Seeded with g0 at position N, one committed token per fire. All
    // loop-carried state is DEVICE loop-carried (guest-seeded for fire-0,
    // re-emitted by the epilogue): the embed token MUST ride a seeded channel —
    // the runtime resolves EmbedTokens from the device channel value; a
    // host-fed embed leaves `token_ids` empty and KV-prepare rejects it.
    if generated.len() < max_tokens {
        let tok_in = bx(Channel::from(vec![g0; 1]).named("tok_in"));
        let pos = bx(Channel::from(vec![n; 1]).named("pos"));
        let klen = bx(Channel::from(vec![n + 1; 1]).named("klen")); // cells 0..=n present
        let fill = bx(Channel::from(vec![n + 1; 1]).named("fill")); // position the NEXT fire writes
        let echo = bx(Channel::from(vec![ECHO_TOKEN; 1]).named("echo"));
        let out = bx(Channel::new([1], dtype::i32).named("out"));
        let lane1 = Tensor::constant(vec![0u32, 1u32]); // embed indptr [0,1]

        let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
        fwd.embed(tok_in, lane1);
        fwd.positions(pos);
        fwd.attn_working_set(ws, klen);
        fwd.epilogue(move || {
            // Takes + compute first, PUTS last (value-id discipline).
            let base = fill.take().tensor(); // [1] u32 — position the NEXT fire writes
            let t = echo.take().tensor(); // [1] i32 — the echo token

            let klen_v = add(&base, 1u32); // cells 0..=base after the next fire
            let next_free = add(&base, 1u32);

            tok_in.put(&t);
            echo.put(&t);
            out.put(&t);
            pos.put(&base);
            klen.put(&klen_v);
            fill.put(&next_free);
        });

        let decode = Pipeline::new();
        for step in 1..max_tokens {
            decode
                .submit(fwd)
                .map_err(|e| format!("decode submit @{step}: {e}"))?;
            let t = out
                .take()
                .get::<i32>()
                .map_err(|e| format!("out.take @{step}: {e}"))?;
            let Some(&t0) = t.first() else {
                return Err(format!("out.take @{step}: empty tensor"));
            };
            generated.push(t0 as u32);
        }
        decode.close();
    }

    let result = format!("generated {} tokens: {:?}", generated.len(), generated);
    eprintln!("[GENERATE] {result}");
    Ok(result)
}
