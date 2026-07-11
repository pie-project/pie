//! `[k]`-Token channel e2e verify — **PTIR keep-core** rewrite.
//!
//! Migrated off the classic `inferlet::sampling` eDSL + `ForwardPass::sampler`
//! surface onto `inferlet::ptir` (the `Pipeline`/`ForwardPass`/`Channel` wire
//! form of direct-channel-e2e / ptir-prefill-e2e). Each verify window is ONE
//! seeded fire — a fresh `ForwardPass` + `WorkingSet` per window, exactly the
//! old guest's fresh-`KvWorkingSet`-per-arm shape (and the host geometry
//! prefill resolves embed/draft from SEEDS, the §3 single-fire contract). The
//! traced program carries both of the old inferlet's layers:
//!
//!  1. **ARGMAX-MATRIX PROBE:** the `[k, vocab]` logits matrix (a k-row
//!     `Readout` port over `prompt + (k-1)` fillers ⇒ `intrinsics::logits()`
//!     declares `[k, vocab]`) → per-row `argmax` → a `[k]`-Token `target`
//!     channel with NO sentinel → ALL k values publish. `CHANNEL_EMITS_K` (the
//!     asserted headline) proves the bound reader channel carries the complete
//!     `[k]` value.
//!  2. **SPEC-VERIFY DAG:** the same fire verifies a seeded `[k]` draft
//!     (`eq → cumprod` cross-row prefix-AND → `select`) into a sentinel-coded
//!     `[k]`-Token `verify` channel (accepted prefix, then `-1`).
//!
//! The old guest's value diagnostics compared against a separately-decoded
//! greedy continuation `g` — degenerate on the mock (each fire's synthetic
//! logits are seeded per-launch, so a prior decode's argmax says nothing about
//! this fire's). The PTIR rewrite gets a STRONGER, mock-honest check by
//! emitting the per-row argmax `target` alongside the verify result from the
//! SAME fire: `SPEC_DAG_OK` recomputes the accept-prefix host-side from
//! (`target`, draft) and demands value-exact equality — a batched per-block
//! cumprod that leaks draft rows past the first reject fails it (the old
//! REJECT_MID detector, no 0-sentinel variant needed: the PTIR channel emits
//! all k rows, there is no marshal truncation to mask the leak).
//! `MATRIX_ROWS_DISTINCT` guards the `[k, vocab]` matrix intrinsic actually
//! carrying per-row values (a single-row broadcast would collapse all rows).
//!
//! Window 2 re-fires with a reject-mid-shaped draft (window 1's target, one
//! mid row perturbed) — on a real driver (stable logits) that is exactly the
//! old reject-at-j arm; on the mock the DAG recompute stays exact against
//! window 2's own target.
//!
//! JSON input: `{"k": 4}` (draft-window size, default 4, min 2).

use inferlet::ptir::prelude::*;
use inferlet::{model as wit_model, serde_json, Result};

const PAGE_T: u32 = 16; // tokens per KV page (mock env page size)
const NUM_LAYERS: u32 = 1;

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// Host recompute of the spec-verify DAG (the golden accept-prefix): accept
/// `draft[r]` while every row `0..=r` matched `target`, `-1` from the first
/// mismatch on — including all rows PAST it (the cross-row cumprod zeroes the
/// whole suffix; a per-block leak would resurface `draft[r]` there).
fn expected_verdict(target: &[i32], draft: &[i32]) -> Vec<i32> {
    let mut ok = true;
    target
        .iter()
        .zip(draft)
        .map(|(&t, &d)| {
            ok = ok && t == d;
            if ok { d } else { -1 }
        })
        .collect()
}

/// Fire ONE seeded `[k, vocab]` verify window (fresh pass + working set, the
/// old fresh-`KvWorkingSet`-per-arm shape): input = prompt + `(k-1)` fillers,
/// read-out rows `l-1 .. l-1+k`, epilogue = per-row `argmax` (probe) + the
/// spec-verify DAG over the seeded `[k]` draft. Returns
/// `(target [k], verdict [k])` off the two bound reader channels.
fn verify_window(
    prompt: &[u32],
    k: u32,
    draft: &[i32],
) -> Result<(Vec<i32>, Vec<i32>)> {
    let l = prompt.len() as u32;
    let n = l + k - 1;
    let input_toks: Vec<i32> = prompt
        .iter()
        .map(|&t| t as i32)
        .chain(std::iter::repeat(0).take((k - 1) as usize))
        .collect();

    let ws: &'static WorkingSet = bx(WorkingSet::new());
    ws.alloc(n.div_ceil(PAGE_T))
        .map_err(|e| format!("ws.alloc: {e}"))?;

    // Seeded inputs (single fire: the host geometry prefill reads seeds) +
    // terminal [k]-Token reader outputs.
    let toks = bx(Channel::from(input_toks).named("toks"));
    let klen = bx(Channel::from(vec![n]).named("klen"));
    let draft_ch = bx(Channel::from(draft.to_vec()).named("draft"));
    let target_out = bx(Channel::new([k], dtype::i32).named("target_out"));
    let verify_out = bx(Channel::new([k], dtype::i32).named("verify_out"));

    // k read-out rows ⇒ intrinsics::logits() declares [k, vocab].
    let readout: Vec<u32> = (0..k).map(|i| l - 1 + i).collect();

    let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd.embed(toks, Tensor::constant(vec![0u32, n]));
    fwd.attn_working_set(ws, klen);
    fwd.readout(&Tensor::constant(readout));
    fwd.epilogue(move || {
        // Takes + compute first, PUTS last (value-id discipline).
        let d = draft_ch.take().tensor(); // [k] i32 submit draft
        let logits = intrinsics::logits(); // [k, vocab] f32 matrix intrinsic
        let tgt = reduce_argmax(logits); // [k] i32 per-row argmax
        let hit = eq(&tgt, &d); // [k] bool
        let ones = broadcast(Tensor::constant(1.0f32), [k]);
        let zeros = broadcast(Tensor::constant(0.0f32), [k]);
        // Cross-row prefix-AND: reject at j zeroes the WHOLE suffix.
        let keep = gt(cumprod(select(&hit, &ones, &zeros)), 0.5f32);
        let neg1 = broadcast(Tensor::constant(-1i32), [k]);
        let ver = select(&keep, &d, &neg1); // accepted prefix, then -1
        target_out.put(&tgt);
        verify_out.put(&ver);
    });

    let pipeline = Pipeline::new();
    pipeline.submit(fwd).map_err(|e| format!("submit: {e}"))?;
    let tgt = target_out
        .take()
        .get::<i32>()
        .map_err(|e| format!("target take: {e}"))?;
    let ver = verify_out
        .take()
        .get::<i32>()
        .map_err(|e| format!("verify take: {e}"))?;
    pipeline.close();
    Ok((tgt, ver))
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let params: serde_json::Value =
        serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let k: u32 = params.get("k").and_then(|v| v.as_u64()).unwrap_or(4).max(2) as u32;
    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);

    let mut prompt = wit_model::encode("The quick brown fox jumps over");
    if prompt.is_empty() {
        prompt.push(0);
    }

    // ── Window 1 (probe): a fixed nonzero draft — the [k]-Token plumbing
    //    headline plus the value-exact DAG recompute against this window's own
    //    target. ──
    let draft_a: Vec<i32> = (1..=k as i32).collect();
    let (target_a, verdict_a) = verify_window(&prompt, k, &draft_a)?;

    // ── Window 2 (reject-mid-shaped): window 1's target with row j perturbed
    //    to a wrong, NONZERO token (≠ target_a[j]). On a stable-logits driver
    //    this is the old reject-at-j arm; the DAG check recomputes from window
    //    2's target either way. ──
    let j = (k / 2).max(1) as usize;
    let mut draft_b = target_a.clone();
    if let Some(x) = draft_b.get_mut(j) {
        *x = ((*x + 1).rem_euclid(vocab as i32)).max(1);
    }
    let (target_b, verdict_b) = verify_window(&prompt, k, &draft_b)?;

    // ── Verdicts ──
    let channel_emits_k = target_a.len() == k as usize
        && verdict_a.len() == k as usize
        && target_b.len() == k as usize
        && verdict_b.len() == k as usize;
    let dag_a_ok = verdict_a == expected_verdict(&target_a, &draft_a);
    let dag_b_ok = verdict_b == expected_verdict(&target_b, &draft_b);
    let spec_dag_ok = dag_a_ok && dag_b_ok;
    let distinct: std::collections::BTreeSet<i32> = target_a.iter().copied().collect();
    let matrix_rows_distinct = distinct.len() > 1;

    let result = format!(
        "CHANNEL_EMITS_K={channel_emits_k} SPEC_DAG_OK={spec_dag_ok} \
         MATRIX_ROWS_DISTINCT={matrix_rows_distinct} k={k} j={j} \
         target_a={target_a:?} draft_a={draft_a:?} verdict_a={verdict_a:?} \
         target_b={target_b:?} draft_b={draft_b:?} verdict_b={verdict_b:?}"
    );
    eprintln!("[K-MARSHAL] {result}");
    Ok(result)
}
