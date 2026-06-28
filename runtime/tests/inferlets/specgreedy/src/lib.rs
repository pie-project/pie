//! Self-spec greedy loop e2e harness (#31) — the **token-identical-to-greedy**
//! land gate.
//!
//! Drives alpha's `SpecMode::System` host loop
//! ([`Generator::collect_tokens_speculative`]) and a plain-greedy decode from
//! the SAME prompt, then asserts the two token streams are **identical**.
//! Losslessness ⟹ the KV/commit choreography (commit-only-accepted ·
//! truncate-rejected · `t_j` rollover · no `ctx.buffer` double-feed) is correct;
//! ANY commit-model bug diverges the streams → `SPEC_GREEDY_IDENTICAL=false`.
//!
//! This is the explicit, OPT-IN invocation surface (not the default decode
//! path): `generate(..).system_speculation().collect_tokens_speculative()`.
//! Requires a drafter-capable model (e.g. Qwen3.5-0.8B + `mtp_num_drafts=4`) so
//! the propose-forward's `pass.draft_output(k)` yields real MTP drafts — the
//! draft window `k` (`SPEC_DRAFT_LEN`) must match the drafter's `mtp_num_drafts`.

use inferlet::{Context, Result, model, sample::Sampler, serde_json};

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let params: serde_json::Value =
        serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let max_tokens = params
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as usize;

    let prompt = model::encode("The quick brown fox jumps over");

    // Plain greedy reference (one fresh context).
    let greedy: Vec<u32> = {
        let mut ctx = Context::new()?;
        ctx.append(&prompt);
        ctx.generate(Sampler::Argmax)
            .max_tokens(max_tokens)
            .collect_tokens()
            .await?
    };

    // Self-spec greedy: MTP drafts → verify → accept → commit. Lossless greedy
    // speculation MUST emit exactly the same tokens as plain greedy.
    let spec: Vec<u32> = {
        let mut ctx = Context::new()?;
        ctx.append(&prompt);
        ctx.generate(Sampler::Argmax)
            .max_tokens(max_tokens)
            .system_speculation()
            .collect_tokens_speculative()
            .await?
    };

    let identical = spec == greedy;
    let result = format!(
        "SPEC_GREEDY_IDENTICAL={identical} n_greedy={} n_spec={} greedy={greedy:?} spec={spec:?}",
        greedy.len(),
        spec.len()
    );
    eprintln!("[SPECGREEDY] {result}");
    Ok(result)
}
