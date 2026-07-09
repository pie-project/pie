//! Self-spec greedy loop e2e harness (#31) — the **token-identical-to-greedy**
//! land gate, with a host-draft-inject mode for drafter-less validation.
//!
//! Drives alpha's `SpecMode::System` host loop
//! ([`Generator::collect_tokens_speculative_instrumented`]) and a plain-greedy
//! decode from the SAME prompt, then asserts the two token streams are
//! **identical**. Losslessness ⟹ the KV/commit choreography (commit-only-accepted
//! · truncate-rejected · `t_j` rollover · no `ctx.buffer` double-feed) is correct;
//! ANY commit-model bug diverges the streams → `SPEC_GREEDY_IDENTICAL=false`.
//! Emits the per-block accepted-token counts (`block_accept_lens`) — the accept
//! rate measurement (LOG, not gate).
//!
//! `inject` input selects the draft SOURCE:
//!  * `"none"` — the real MTP drafter (`pass.draft_output`); needs a drafter-capable
//!    model (gemma-4-E4B-it, MTP head in-repo). VERIFIED gate.
//!  * `"greedy"` — host-inject the greedy continuation → ALL-ACCEPT (commit path,
//!    accept rate k/block).
//!  * `"reject"` — host-inject greedy with a per-block mismatch at r=1 → PARTIAL
//!    reject (truncate + correction path, accept rate ~2/block).
//!  * `"garbage"` — host-inject all-wrong → FULL reject every block (correction-only
//!    path, accept rate 1/block).
//! The inject modes drive the KV/commit + losslessness gate on a drafter-less model
//! (e.g. qwen3-0.6b); the OUTPUT must be token-identical-to-greedy in EVERY mode.

use inferlet::{Context, Result, model, sample::Sampler, serde_json};

/// Draft block size — MUST match the SDK loop's `SPEC_DRAFT_LEN`.
const K: usize = 4;

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let params: serde_json::Value =
        serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let max_tokens = params
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as usize;
    let inject = params
        .get("inject")
        .and_then(|v| v.as_str())
        .unwrap_or("none")
        .to_string();

    let prompt = model::encode("The quick brown fox jumps over");

    // Plain greedy reference (fresh context).
    let greedy: Vec<u32> = {
        let mut ctx = Context::new()?;
        ctx.append(&prompt);
        ctx.generate(Sampler::Argmax)
            .max_tokens(max_tokens)
            .collect_tokens()
            .await?
    };

    // Self-spec run — real drafter (`inject=none`) or host-injected drafts.
    let run = {
        let mut ctx = Context::new()?;
        ctx.append(&prompt);
        let generator = ctx
            .generate(Sampler::Argmax)
            .max_tokens(max_tokens)
            .system_speculation();
        let generator = if inject == "none" {
            generator
        } else {
            // Host-construct each block's drafts from the greedy continuation at
            // `pos = tokens_generated` (== the greedy index, by losslessness).
            let g = greedy.clone();
            let mode = inject.clone();
            generator.inject_spec_drafts(move |_anchor, pos| {
                (0..K)
                    .map(|r| {
                        let t = g.get(pos + r).copied().unwrap_or(1);
                        match mode.as_str() {
                            // all-wrong → reject at r=0 → accept just the correction.
                            "garbage" => t.wrapping_add(1),
                            // wrong at r=1 → accept d0 then correct → 2/block.
                            "reject" if r == 1 => t.wrapping_add(1),
                            // greedy continuation → all-accept.
                            _ => t,
                        }
                    })
                    .collect()
            })
        };
        generator.collect_tokens_speculative_instrumented().await?
    };

    let identical = run.tokens == greedy;
    let result = format!(
        "SPEC_GREEDY_IDENTICAL={identical} inject={inject} n_greedy={} n_spec={} \
         block_accept_lens={:?} greedy={greedy:?} spec={:?}",
        greedy.len(),
        run.tokens.len(),
        run.block_accept_lens,
        run.tokens
    );
    eprintln!("[SPECGREEDY] {result}");
    Ok(result)
}
