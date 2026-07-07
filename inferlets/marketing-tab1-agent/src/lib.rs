//! Minimal tool-calling agent — **raw-WIT keep-core** rewrite.
//!
//! Written directly on the low-level WIT surface (In Gim's SDK-minimize
//! directive): no `Context`/`Generator`/`Sampler` facade. The tool-call loop is
//! hand-written and visible here; only the thin keep-core primitives it calls
//! are the SDK surface (`carrier::submit_pass`, `sampler::sampler_program`, the
//! kept `chat`/`tools`/`model` thin bindings).
//!
//! This is a **tool-terminated** loop (not chat-EOS): each round decodes
//! sequentially until the streaming `tools::Decoder` detects a completed call
//! or the per-round token budget is hit. On a call, the result is framed with
//! `tools::answer_prefix` and fed back as the next round's prefill. Decoding is
//! sequential (carry=false) — faithful to the original facade loop, which
//! inspected every token to detect the call boundary.

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, LoweredSampler, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, model, tool, tools, Result};

/// Search the web for current information.
#[tool]
async fn web_search(query: String) -> Result<String> {
    Ok(format!("(stub result for: {query})"))
}

/// Per-round token budget (mirrors the facade's `.max_tokens(512)`).
const MAX_TOKENS: usize = 512;

/// Read the sampled token off a finalized pass's single-`Token` output tensor.
async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

#[inferlet::main]
async fn main(prompt: String) -> Result<String> {
    let vocab = model::output_vocab_size();
    let s: LoweredSampler = sampler::sampler_program(SamplerSpec::Argmax, vocab)?;

    // Prompt tokens (kept thin bindings): system turn, then the tool schemas
    // registered via the `tools::equip` primitive, then the user turn. Mirrors
    // `Context::new().system(..).equip(..)?.user(..)` — `equip` flushes the
    // system into the buffer first, so the user turn is a plain `chat::user`.
    let mut prompt_tokens = chat::system("Use web_search if you need fresh facts, then answer.");
    prompt_tokens.extend(tools::equip(&[&web_search])?);
    prompt_tokens.extend(chat::user(&prompt));

    // One decode context on the raw WIT surface: its own KV working set + cursor.
    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;

    // The next round's prefill: the initial prompt, then each tool answer.
    let mut pending = prompt_tokens;

    loop {
        let mut tdec = tools::Decoder::new();
        let mut full: Vec<u32> = Vec::new();
        let mut generated = 0usize;

        // Sequential decode this round (carry=false: every token is inspected to
        // detect the tool-call boundary, so there is nothing to speculate past).
        let call = loop {
            if generated >= MAX_TOKENS {
                break None;
            }
            let pass = carrier::submit_pass(&kv, &mut seq_len, &mut fresh, &s, &pending, false)?;
            let token = read_token(pass).await?;
            generated += 1;
            full.push(token);
            pending = vec![token];

            if let tools::Event::Call(name, args) = tdec.feed(&[token])? {
                break Some((name, args));
            }
        };

        let Some((name, args)) = call else {
            return model::decode(&full);
        };

        let result = match name.as_str() {
            "web_search" => web_search::call(&args).await?,
            _ => return Err(format!("unknown tool: {name}")),
        };
        // Frame the result for the next turn; it becomes the next round's prefill.
        pending = tools::answer_prefix(&name, &result);
    }
}
