//! #31 greedy-v0 **self-spec verify** e2e — DEVICE-ALIAS draft source
//! (`inferlet::ptir` bridge rewrite).
//!
//! The de-hardwiring thesis: the verify's `[k]` draft tokens come NOT from a
//! host upload but DEVICE-RESIDENT off the model's OWN embedded input tokens —
//! the drafts ARE forward-N's verify INPUT (the tokens after the anchor). This
//! inferlet embeds `input = prompt ++ D`, peeks the SAME `toks` channel back
//! inside the epilogue (`toks.read()`, non-consuming — the device-alias read,
//! the ptir-bridge equivalent of the deleted `Readiness::SelfSpecDraftInput`
//! resolver reading `pi.tokens+sample_row+1`) to recover `D` as an in-program
//! tensor, and verifies it against the target's per-row argmax — the A2 DAG
//! (`argmax -> eq -> cumprod -> select`): the accepted draft prefix, the
//! target's correction spliced at the first reject, then `0`-sentinels.
//!
//! ## Independent witness (no degenerate green):
//! the reference `g` is the greedy continuation computed via a SEPARATE
//! sequential argmax decode loop (its own working set) — a different code
//! path than the verify's matrix `reduce_argmax`, so `V == greedy_verify_a2(D,
//! g)` catches a verify-matrix argmax regression too, not just the
//! eq/cumprod/select DAG.
//!
//! ## Arms (the inferlet asserts each):
//!  * **ACCEPT-ALL:** `D = g` ⇒ every row matches ⇒ `V == g`.
//!  * **REJECT-MID (load-bearing):** `D = g` with `D[j]` perturbed (`j=k/2`,
//!    nonzero, `≠ g[j]`) ⇒ `V == [g0..g_{j-1}, g_j, 0..]` — the accepted
//!    prefix, the correction `t_j == g[j]` spliced at `j`, then 0s.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model, serde_json};

const PAGE_T: u32 = 16;
const NUM_LAYERS: u32 = 1;
const PROMPT: &str = "The quick brown fox jumps over";

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// Host reference for the A2 DAG: the accepted draft prefix, the target
/// correction `a[j]` spliced at the first reject, then `0`-sentinels.
fn greedy_verify_a2_obs(d: &[u32], a: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(d.len());
    let mut keep = true;
    for i in 0..d.len() {
        if keep && d[i] == a[i] {
            out.push(d[i]);
        } else if keep {
            out.push(a[i]);
            keep = false;
        } else {
            out.push(0);
        }
    }
    out
}

/// The independent greedy-argmax reference continuation `g[0..k)`: `k`
/// sequential single-token argmax decode fires on a fresh working set.
fn greedy_reference(prompt: &[u32], k: u32) -> Result<Vec<u32>> {
    let ws: &'static WorkingSet = bx(WorkingSet::new());
    let n = prompt.len() as u32;
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();

    let toks_p = bx(Channel::from(prompt_i32).named("g_toks_p"));
    let klen_p = bx(Channel::from(vec![n; 1]).named("g_klen_p"));
    let g0_ch = bx(Channel::new([1], dtype::i32).named("g0"));

    let fwd_p: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd_p.embed(toks_p, Tensor::constant(vec![0u32, n]));
    fwd_p.attn_working_set(ws, klen_p);
    fwd_p.epilogue(move || {
        let t = reduce_argmax(intrinsics::logits());
        g0_ch.put(&t);
    });
    let prefill = Pipeline::new();
    fwd_p
        .submit(&prefill)
        .map_err(|e| format!("greedy prefill: {e}"))?;
    let g0 = g0_ch
        .take()
        .get::<i32>()
        .map_err(|e| format!("g0 take: {e}"))?[0];
    prefill.close();

    let mut g: Vec<u32> = vec![g0 as u32];
    if k > 1 {
        let tok_in = bx(Channel::from(vec![g0; 1]).named("g_tok_in"));
        let pos = bx(Channel::from(vec![n; 1]).named("g_pos"));
        let klen = bx(Channel::from(vec![n + 1; 1]).named("g_klen"));
        let fill = bx(Channel::from(vec![n + 1; 1]).named("g_fill"));
        let out = bx(Channel::new([1], dtype::i32).named("g_out"));
        let lane1 = Tensor::constant(vec![0u32, 1u32]);

        let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
        fwd.embed(tok_in, lane1);
        fwd.positions(pos);
        fwd.attn_working_set(ws, klen);
        fwd.epilogue(move || {
            let base = fill.take().tensor();
            let t = reduce_argmax(intrinsics::logits());
            let klen_v = add(&base, 1u32);
            let next_free = add(&base, 1u32);
            tok_in.put(&t);
            out.put(&t);
            pos.put(&base);
            klen.put(&klen_v);
            fill.put(&next_free);
        });
        let decode = Pipeline::new();
        for step in 1..k {
            fwd.submit(&decode)
                .map_err(|e| format!("greedy decode @{step}: {e}"))?;
            let t = out
                .take()
                .get::<i32>()
                .map_err(|e| format!("greedy out @{step}: {e}"))?[0];
            g.push(t as u32);
        }
        decode.close();
    }
    Ok(g)
}

/// Fire `input = prompt ++ drafts`; the verify device-aliases the k drafts by
/// peeking (`.read()`, non-consuming) the SAME `toks` channel already embedded
/// — NO separate draft submission. Returns the `[k]` accept set.
fn fire_verify(prompt: &[u32], k: u32, drafts: &[u32]) -> Result<Vec<u32>> {
    let l = prompt.len() as u32;
    let mut inp: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    inp.extend(drafts.iter().map(|&t| t as i32));
    let n = l + k;

    let ws: &'static WorkingSet = bx(WorkingSet::new());
    let toks = bx(Channel::from(inp).named("v_toks"));
    let klen = bx(Channel::from(vec![n]).named("v_klen"));
    let verify_out = bx(Channel::new([k], dtype::i32).named("v_out"));

    // k read-out rows (the last k positions, prompt-tail .. prompt-tail+k-1)
    // ⇒ intrinsics::logits() declares [k, vocab].
    let readout: Vec<u32> = (0..k).map(|i| l - 1 + i).collect();
    // The k draft token indices WITHIN `toks` (the positions after the anchor).
    let draft_positions: Vec<u32> = (l..l + k).collect();

    let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd.embed(toks, Tensor::constant(vec![0u32, n]));
    fwd.attn_working_set(ws, klen);
    fwd.readout(&Tensor::constant(readout));
    fwd.epilogue(move || {
        // Device-alias read: peek the embedded input tokens (NOT a separate
        // submitted draft channel) and gather the k draft positions out of it.
        let toks_val = toks.read().tensor(); // [n] i32
        let draft = gather(&toks_val, Tensor::constant(draft_positions.clone())); // [k] i32
        let target = reduce_argmax(intrinsics::logits()); // [k] i32
        let hit = eq(&target, &draft); // [k] bool
        let ones = broadcast(Tensor::constant(1.0f32), [k]);
        let zeros = broadcast(Tensor::constant(0.0f32), [k]);
        let n_acc = cast(reduce_sum(cumprod(select(&hit, &ones, &zeros))), DType::U32);
        let keep = ge(broadcast(&n_acc, [k]), iota(k)); // [k] bool: i <= n_acc
        let zeros_i32 = broadcast(Tensor::constant(0i32), [k]);
        let verify = select(&keep, &target, &zeros_i32); // accepted prefix + correction + 0s
        verify_out.put(&verify);
    });

    let pipeline = Pipeline::new();
    fwd.submit(&pipeline)
        .map_err(|e| format!("verify submit: {e}"))?;
    let v = verify_out
        .take()
        .get::<i32>()
        .map_err(|e| format!("verify take: {e}"))?;
    pipeline.close();
    Ok(v.iter().map(|&x| x as u32).collect())
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let params: serde_json::Value = serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let k: u32 = params.get("k").and_then(|v| v.as_u64()).unwrap_or(4).max(2) as u32;
    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);
    let j = (k / 2).max(1) as usize;

    let mut prompt = wit_model::encode(PROMPT);
    if prompt.is_empty() {
        prompt.push(0);
    }

    // ── Greedy continuation g[0..k): the INDEPENDENT argmax witness. ──
    let g = greedy_reference(&prompt, k)?;
    if (g.len() as u32) < k {
        return Err(format!("greedy produced {} < k={} tokens", g.len(), k));
    }
    let g: Vec<u32> = g[..k as usize].to_vec();

    // ── ACCEPT-ALL: drafts = g → all rows match → V == g. ──
    let accept_out = fire_verify(&prompt, k, &g)?;

    // ── REJECT-MID (load-bearing): perturb draft j (≠ g[j], nonzero). ──
    let mut rj_draft = g.clone();
    rj_draft[j] = ((g[j] + 1) % vocab).max(1);
    let reject_out = fire_verify(&prompt, k, &rj_draft)?;

    // ── Verdicts ──
    let expect_accept = greedy_verify_a2_obs(&g, &g);
    let expect_reject = greedy_verify_a2_obs(&rj_draft, &g);

    let accept_all_ok = accept_out == expect_accept;
    let reject_mid_ok = reject_out == expect_reject;
    let a2_correction_ok = reject_out.get(j).copied() == Some(g[j])
        && reject_out[j] != rj_draft[j]
        && reject_out[j] != 0;
    let bind_device_alias_ok = reject_out.len() == k as usize
        && reject_out[..j] == g[..j]
        && reject_out.get(j).copied() != Some(rj_draft[j]);
    let crosscheck_ok = accept_all_ok && reject_mid_ok;
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
