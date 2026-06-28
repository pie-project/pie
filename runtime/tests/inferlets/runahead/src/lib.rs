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
//! **Scenario A** (`MATCH` + `ANCHOR_OK`): the pipelined carrier stream equals
//! the sync stream AND positively equals the verified milestone tokens (an
//! absolute anchor — a self-consistency-only check false-passes when both paths
//! read the same broken `output()`).
//! **Scenario B** (`CLEAR_OK`): the #26 dangling-carry clear. A stop-terminated
//! pipelined generate-1 leaves a dangling carrier link; a 2nd `generate()` on the
//! same context must drop it (`fresh-generate`) so gen-2 is token-exact vs a
//! context whose gen-1 was sequential (no carrier). Positively GPU-verifies the
//! host clear's device free-link path, which the 454 host tests only mock.
//!
//! JSON/plain input: an optional token budget (defaults to 8), e.g. `"16"`.

use inferlet::{Context, Result, model, sample::Sampler};

const PROMPT: &str = "hello world";
/// Fixed 2nd-turn continuation for the #26 clear probe (Scenario B).
const CONT: &str = " Tell me more.";

/// Deterministic greedy sampler (argmax) — reproducible token streams.
fn greedy() -> Sampler {
    Sampler::TopK {
        temperature: 0.0,
        k: 1,
    }
}

/// The known-good greedy decode of `PROMPT` on qwen3-0.6b — the verified #6/#21
/// milestone stream. An ABSOLUTE anchor: a degenerate / all-zeros `output()`
/// fails the gate even when pipelined and sync agree by both reading the same
/// broken path (the false-pass blind spot a self-consistency-only check has —
/// the gap that let the cut #1 fast-path's zeros slip to green).
const MILESTONE: [u32; 8] = [198, 9707, 1879, 374, 264, 4285, 2025, 429];

/// Whether `tokens` positively matches the milestone anchor: exact over the
/// known `MILESTONE` prefix, and non-degenerate (not all-zeros) beyond it.
fn anchor_ok(tokens: &[u32]) -> bool {
    match tokens.len() {
        0 => false,
        n if n <= MILESTONE.len() => tokens == &MILESTONE[..n],
        _ => tokens[..MILESTONE.len()] == MILESTONE && tokens.iter().any(|&t| t != 0),
    }
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let max_tokens: usize = input.trim().parse().unwrap_or(8);

    let prompt_tokens = model::encode(PROMPT);
    eprintln!("[RUNAHEAD] encoded prompt: {} tokens", prompt_tokens.len());

    // ── Scenario A — carrier token-exactness (1a milestone regression) ──
    // The pipelined run-ahead carrier stream MUST equal the synchronous greedy
    // stream on the same prompt (greedy ⇒ deterministic; any divergence is a
    // real carryover bug, not sampling noise).
    let mut ctx_p = Context::new()?;
    ctx_p.append(&prompt_tokens);
    let tokens_p = ctx_p
        .generate(greedy())
        .max_tokens(max_tokens)
        .collect_tokens_pipelined()
        .await?;

    let mut ctx_s = Context::new()?;
    ctx_s.append(&prompt_tokens);
    let tokens_s = ctx_s
        .generate(greedy())
        .max_tokens(max_tokens)
        .collect_tokens()
        .await?;

    let matched = tokens_p == tokens_s;

    // ── Scenario B — #26 dangling-carry CLEAR (fresh-generate host-clear) ──
    // A STOP-terminated PIPELINED generate leaves a dangling carrier link on its
    // context — the terminal pass emitted `next-inputs` but no successor consumes
    // it. The next `generate()` on the SAME context must DROP that carry
    // (`fresh-generate`, #26) before its prime injects, else the prime's first
    // input row is overwritten by the stale retained token (and the retained
    // device buffer leaks). The 454 host tests MOCK the free-link path; this
    // positively exercises it on a real fire — the wrong-link / double-free /
    // free-a-live-buffer class that host-green can't see.
    //
    // Isolation: stop gen-1 on its FIRST sampled token, so gen-1 keeps NO tokens
    // and commits ONLY the prompt — identical committed state, seq_len, and
    // history on both sides. ctx_a runs gen-1 PIPELINED (leaves a dangling
    // carry); ctx_b runs it SEQUENTIALLY (no carrier). Both then append the same
    // continuation and decode gen-2 the same way. The ONLY difference is the
    // dangling link on ctx_a, so gen-2 matches iff the clear drops it; a broken
    // clear injects the stale token into ctx_a's gen-2 prime → divergence.
    let cont_tokens = model::encode(CONT);
    let clear_ok = if let Some(&stop_tok) = tokens_s.first() {
        // ctx_a: pipelined gen-1 (→ dangling carry), then gen-2.
        let mut ctx_a = Context::new()?;
        ctx_a.append(&prompt_tokens);
        let g1a = ctx_a
            .generate(greedy())
            .stop(&[stop_tok])
            .max_tokens(max_tokens)
            .collect_tokens_pipelined()
            .await?;
        ctx_a.append(&cont_tokens);
        let g2a = ctx_a
            .generate(greedy())
            .max_tokens(max_tokens)
            .collect_tokens_pipelined()
            .await?;

        // ctx_b: sequential gen-1 (no carrier, no dangling), then gen-2 (same path).
        let mut ctx_b = Context::new()?;
        ctx_b.append(&prompt_tokens);
        let g1b = ctx_b
            .generate(greedy())
            .stop(&[stop_tok])
            .max_tokens(max_tokens)
            .collect_tokens()
            .await?;
        ctx_b.append(&cont_tokens);
        let g2b = ctx_b
            .generate(greedy())
            .max_tokens(max_tokens)
            .collect_tokens_pipelined()
            .await?;

        eprintln!(
            "[RUNAHEAD] clear: stop={stop_tok} g1a={g1a:?} g1b={g1b:?} g2a={g2a:?} g2b={g2b:?}"
        );
        // gen-1 halts on its first sample ⇒ no kept tokens; gen-2 must be
        // non-empty, non-degenerate (not all-zeros — the broken-output() blind
        // spot), and identical across the two paths iff the clear worked.
        !g2a.is_empty() && g2a.iter().any(|&t| t != 0) && g2a == g2b
    } else {
        eprintln!("[RUNAHEAD] clear: no greedy tokens — scenario B skipped");
        false
    };

    // Non-degeneracy anchor: the stream must positively equal the verified
    // milestone, not merely self-agree — `pipelined == sync` FALSE-passes if
    // BOTH read a broken, all-zeros `output()` (the gate gap that let the cut #1
    // fast-path slip to green).
    let anchor_ok = anchor_ok(&tokens_p);
    let result = format!(
        "MATCH={matched} ANCHOR_OK={anchor_ok} CLEAR_OK={clear_ok} \
         pipelined={tokens_p:?} sync={tokens_s:?}"
    );
    eprintln!("[RUNAHEAD] {result}");
    Ok(result)
}
