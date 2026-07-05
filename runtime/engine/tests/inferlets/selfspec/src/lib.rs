//! #31 greedy-v0 **self-spec verify** e2e — DEVICE-ALIAS draft source (delta).
//!
//! The de-hardwiring thesis: the verify's `[k]` draft tokens come NOT from a host
//! upload but DEVICE-RESIDENT off `pi.tokens + sample_row + 1` (echo's
//! `SelfSpecDraftInput` resolver), source-selected by the manifest flag — the
//! drafts ARE forward-N's verify INPUT (the tokens after the anchor). This inferlet
//! drives `mtp_self_spec_greedy_observable` (foxtrot's A2 keystone) through that
//! binding and asserts the COMPLETE accept set incl. the target correction `t_j`.
//!
//! ## What's new vs the #35-A `specverify` (host-injected `Submit` draft):
//!  1. **Device-alias draft, not host-injected.** The draft is bound
//!     `Readiness::SelfSpecDraftInput` → the resolver reads `pi.tokens+sample_row+1`
//!     (alpha's ballast lets `resolve_bindings(.., &[])` synthesize the no-value
//!     slot). So the drafts are the INPUT tokens after the anchor: we feed
//!     `input = prompt ++ D`, and the verify device-aliases `D` from `pi.tokens`.
//!     A wrong-buffer bind (`pi.sampled`, wrong offset) ⇒ the reject lands at the
//!     wrong row ⇒ RED. This is the e2e proof of echo's bind.
//!  2. **A2 correction.** `mtp_self_spec_greedy` emits `[d0..d_{j-1}, t_j, sentinel..]`
//!     — the target's greedy token `t_j` spliced at the first-reject boundary (the
//!     free correct token that kills the full-reject stall). The observable variant
//!     uses a `0` sentinel (non-truncating) so the cross-row prefix + the correction
//!     are OBSERVABLE (not masked by the `-1` marshal truncation).
//!
//! ## Independent witness (no degenerate green):
//! the reference `A` is the greedy continuation `g` computed via the **Argmax
//! SAMPLER** — a DIFFERENT argmax code path than the verify's matrix intrinsic, so
//! `V == greedy_verify_a2(D, g)` catches a verify matrix-argmax regression (the
//! #35-A class) too, not just the eq/cumprod/select DAG. Using the verify's own
//! argmax as the reference would be a shared-reference degenerate green.
//!
//! ## Arms (the inferlet asserts each):
//!  * **ACCEPT-ALL:** `D = g` ⇒ every row matches ⇒ `V == g` (all k drafts, no
//!    correction, no sentinel).
//!  * **REJECT-MID (the load-bearing gate):** `D = g` with `D[j]` perturbed
//!    (`j=k/2`, nonzero, `≠ g[j]`) ⇒ `V == [g0..g_{j-1}, g_j, 0..]` — the accepted
//!    prefix `g[0..j]`, the correction `t_j == g[j]` spliced at `j`, then 0s. The
//!    perturbation forces the verify to READ the device-aliased draft (a wrong bind
//!    rejects elsewhere) AND exercises the A2 boundary correction.
//!
//! ## 0-sentinel collision invariant (extended `DRAFTS_NONZERO`):
//! under A2 the boundary emits the REAL token `t_j`; a `t_j == 0` would collide with
//! the past-boundary `0`-sentinels, re-masking the reject. So `{d0..d_{j-1}, t_j}`
//! must ALL be `∈ [1, vocab)` — ensured by construction (a prompt whose greedy
//! continuation incl. the boundary correction is nonzero); flagged if violated.

use inferlet::program::resolve_bindings;
use inferlet::sample::Sampler;
use inferlet::sampling::program as edsl;
use inferlet::serde_json;
use inferlet::{Context, Result, model};

/// Host reference for `mtp_self_spec_greedy_observable`'s A2 DAG
/// (`argmax -> eq -> cumprod -> select(draft, select(boundary, target, 0))`): the
/// accepted draft prefix, the target correction `a[j]` spliced at the first reject,
/// then `0`-sentinels. `a` is the INDEPENDENT argmax reference (`g`).
fn greedy_verify_a2_obs(d: &[u32], a: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(d.len());
    let mut keep = true;
    for i in 0..d.len() {
        if keep && d[i] == a[i] {
            out.push(d[i]); // accepted draft (== target)
        } else if keep {
            out.push(a[i]); // first-reject boundary → target correction t_i
            keep = false;
        } else {
            out.push(0); // past the boundary → 0-sentinel
        }
    }
    out
}

/// Fire `mtp_self_spec_greedy_observable` with `input = prompt ++ drafts`; the
/// verify device-aliases the k drafts off `pi.tokens+sample_row+1` (NO host upload —
/// `resolve_bindings(.., &[])`, alpha's ballast fills the slot + carries the flag).
/// Returns the `[k]`-Token accept set via the #32 `program_tokens` accessor.
async fn fire_verify(
    context: &Context,
    prompt: &[u32],
    vocab: u32,
    k: u32,
    drafts: &[u32],
) -> Result<Vec<u32>> {
    let l = prompt.len() as u32;
    let mut sctx = context.fork()?;
    let mut pass = sctx.forward();
    let start = pass.start_position();
    let mut inp = prompt.to_vec();
    inp.extend_from_slice(drafts); // drafts = input tokens after the anchor
    pass.input(&inp);
    let (built, _keys) = edsl::mtp_self_spec_greedy_observable(vocab, k)
        .map_err(|e| format!("mtp_self_spec build: {e:?}"))?;
    let program =
        inferlet::emit::emit_program(&built.program).map_err(|e| format!("emit: {e}"))?;
    // Matrix logit positions [start+l-1 .. start+l+k-2]: row r argmax conditioned on
    // prompt ++ D[0..r). Empty submit_values: the SelfSpecDraftInput draft is driver
    // source-selected (echo's resolver, pi.tokens+sample_row+1), the ballast fills it.
    let positions: Vec<u32> = (0..k).map(|i| start + l - 1 + i).collect();
    let bindings = resolve_bindings(&built.bindings, &built.host_inputs, &positions, &[])?;
    let handles = pass.sampler(&program, bindings, built.outputs.len() as u32);
    let out = pass.execute().await?;
    let v: Vec<u32> = out
        .tokens(handles[0])
        .await
        .map_err(|e| format!("self-spec tokens: {e}"))?;
    Ok(v)
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let params: serde_json::Value =
        serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let k: u32 = params
        .get("k")
        .and_then(|v| v.as_u64())
        .unwrap_or(4)
        .max(2) as u32;
    let vocab = model::output_vocab_size();
    let j = (k / 2).max(1) as usize; // reject position

    let mut prompt = model::encode("The quick brown fox jumps over");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let context = Context::new()?;

    // ── Greedy continuation g[0..k): k sequential Argmax decodes. This is the
    //    INDEPENDENT argmax witness (Argmax sampler ≠ the verify's matrix intrinsic).
    let g: Vec<u32> = {
        let mut gctx = context.fork()?;
        gctx.append(&prompt);
        let mut decoder = gctx.generate(Sampler::Argmax).max_tokens(k as usize);
        let mut acc = Vec::new();
        while let Some(step) = decoder.next()? {
            let o = step.execute().await?;
            acc.extend(o.tokens.iter().copied());
        }
        acc
    };
    if (g.len() as u32) < k {
        return Err(format!("greedy produced {} < k={} tokens", g.len(), k).into());
    }
    let g: Vec<u32> = g[..k as usize].to_vec();

    // ── ACCEPT-ALL: drafts = g → all rows match → V == g (device-alias read of the
    //    greedy continuation; no reject, no correction, no sentinel). ──
    let accept_out = fire_verify(&context, &prompt, vocab, k, &g).await?;

    // ── REJECT-MID (load-bearing): perturb draft j (≠ g[j], nonzero). The verify
    //    must READ the device-aliased perturbed draft → reject at j → splice the
    //    correction t_j = g[j] → V == [g0..g_{j-1}, g_j, 0..]. ──
    let mut rj_draft = g.clone();
    rj_draft[j] = ((g[j] + 1) % vocab).max(1); // ≠ g[j], ∈ [1,vocab)
    let reject_out = fire_verify(&context, &prompt, vocab, k, &rj_draft).await?;

    // ── Verdicts ──
    let expect_accept = greedy_verify_a2_obs(&g, &g); // == g
    let expect_reject = greedy_verify_a2_obs(&rj_draft, &g); // [g0..g_{j-1}, g_j, 0..]

    let accept_all_ok = accept_out == expect_accept;
    let reject_mid_ok = reject_out == expect_reject;
    // The A2 boundary correction: the rejected row j emits the TARGET token g[j]
    // (NOT the perturbed draft, NOT 0). This is the spliced correction that kills the
    // full-reject stall and keeps v0 greedy-exact.
    let a2_correction_ok = reject_out.get(j).copied() == Some(g[j])
        && reject_out[j] != rj_draft[j]
        && reject_out[j] != 0;
    // Device-alias bind proof: the perturbed draft forced a reject EXACTLY at j and
    // the prefix < j read the (correct) device-aliased drafts == g. A wrong-buffer
    // bind would reject at the wrong row / read garbage.
    let bind_device_alias_ok = reject_out.len() == k as usize
        && reject_out[..j] == g[..j]
        && reject_out.get(j).copied() != Some(rj_draft[j]);
    let crosscheck_ok = accept_all_ok && reject_mid_ok;
    // Extended 0-sentinel invariant (A2): the accepted drafts AND the boundary
    // correction t_j must be ∈ [1,vocab) — a 0 anywhere there collides with the
    // past-boundary 0-sentinels and re-masks the reject.
    let drafts_nonzero = g.iter().all(|&t| t != 0) && rj_draft.iter().all(|&t| t != 0);

    let result = format!(
        "BIND_DEVICE_ALIAS_OK={bind_device_alias_ok} ACCEPT_ALL_OK={accept_all_ok} \
         REJECT_MID_OK={reject_mid_ok} A2_CORRECTION_OK={a2_correction_ok} \
         CROSSCHECK_OK={crosscheck_ok} DRAFTS_NONZERO={drafts_nonzero} k={k} j={j} \
         g={g:?} rj_draft={rj_draft:?} accept_out={accept_out:?} reject_out={reject_out:?} \
         expect_reject={expect_reject:?}"
    );
    eprintln!("[SELF-SPEC] {result}");
    Ok(result)
}
