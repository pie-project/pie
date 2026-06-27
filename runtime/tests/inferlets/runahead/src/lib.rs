//! Run-ahead carryover test inferlet (Seam A §2.1).
//!
//! Drives the device-side run-ahead carrier via
//! [`collect_tokens_pipelined`](inferlet::generation::Generator::collect_tokens_pipelined):
//! each forward pass's sampled token is carried into the *next* pass's input by
//! the carrier (producer `source_link` → consumer `carried_input` +
//! `inject_link` + `free_link`), instead of the guest reading it back and
//! re-feeding it. This exercises the executor-hook path (retain → inject → free)
//! in a real fire.
//!
//! The 1a milestone runs this **sequentially** (each pass awaited before the
//! next is submitted), so the carrier's RETAIN strictly precedes its INJECT.
//! delta's carrier instrumentation GPU-verifies `consumer.pi.tokens[dest] ==
//! producer's sample`.
//!
//! Deterministic greedy (argmax) so the carried token stream is reproducible
//! and MUST equal the synchronous `collect_tokens` stream on the same prompt.
//!
//! JSON/plain input: an optional token budget (defaults to 8), e.g. `"16"`.

use inferlet::{Context, Result, model, sample::Sampler};

const PROMPT: &str = "hello world";

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let max_tokens: usize = input.trim().parse().unwrap_or(8);

    let prompt_tokens = model::encode(PROMPT);
    eprintln!("[RUNAHEAD] encoded prompt: {} tokens", prompt_tokens.len());

    // Pipelined (run-ahead carrier) greedy stream.
    let mut ctx_p = Context::new()?;
    ctx_p.append(&prompt_tokens);
    let tokens_p = ctx_p
        .generate(Sampler::TopK {
            temperature: 0.0,
            k: 1,
        })
        .max_tokens(max_tokens)
        .collect_tokens_pipelined()
        .await?;

    // Synchronous reference: same greedy, same prompt, fresh context. The
    // carried stream MUST equal this (greedy ⇒ deterministic; any divergence is
    // a real carryover bug, not sampling noise).
    let mut ctx_s = Context::new()?;
    ctx_s.append(&prompt_tokens);
    let tokens_s = ctx_s
        .generate(Sampler::TopK {
            temperature: 0.0,
            k: 1,
        })
        .max_tokens(max_tokens)
        .collect_tokens()
        .await?;

    let matched = tokens_p == tokens_s;
    let result = format!("MATCH={matched} pipelined={tokens_p:?} sync={tokens_s:?}");
    eprintln!("[RUNAHEAD] {result}");
    Ok(result)
}
