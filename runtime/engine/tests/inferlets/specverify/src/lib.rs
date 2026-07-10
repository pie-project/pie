//! `[k]`-Token channel e2e verify.
//!
//! Two layers are separated on purpose:
//!  1. **ARGMAX-MATRIX PROBE:** a `[k,vocab]` intrinsic →
//!     per-row `argmax` → `[k]`-Token, with NO `-1` sentinel → ALL k argmax values
//!     publish regardless of their correctness. `CHANNEL_EMITS_K` proves the
//!     bound reader channel carries the complete value.
//!  2. **MATRIX-ARGMAX VALUES (diagnostic):** `probe_out == g` (the target's greedy
//!     continuation) iff the `[k,vocab]` matrix intrinsic feeds each row r the
//!     logits at position L-1+r. `MATRIX_ARGMAX_OK` = this. (k=1 only ever
//!     exercised row 0 — row≥1 is the new path.)
//!  3. **SPEC-VERIFY arm (bonus):** `spec_verify_greedy(k)` mixed draft ⇒
//!     accept-prefix + `-1` sentinels (the truncation semantics).
//!
//! Runs on the raw low-level WIT (keep-core): each arm's `Context::fork` +
//! `forward` is a fresh `KvWorkingSet` + `geometry::*` write + raw `ForwardPass`,
//! and the `[k]`-Token channel is read back with `output().read()`. The greedy
//! `g` arm is a raw k-step argmax decode loop.

use inferlet::inference::{ForwardPass, InputBinding};
use inferlet::program::{encode_i32, resolve_bindings};
use inferlet::sampling::program as edsl;
use inferlet::sampling::{DType, Graph, OutputKind, Readiness, dselect};
use inferlet::serde_json;
use inferlet::working_set::KvWorkingSet;
use inferlet::{Result, geometry, model, sampler};

/// Fire a single-output `[k]`-Token program and read the bound reader channel.
async fn fire_token_channel(
    program: &inferlet::tensor::Program,
    bindings: Vec<InputBinding>,
    input: &[u32],
    page: u32,
) -> Result<Vec<u32>> {
    let kv = KvWorkingSet::new();
    let n = input.len() as u32;
    let pass = ForwardPass::new();
    pass.fresh_generate();
    let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(0, n, page))?;
    geometry::attach_kv_write(&pass, &kv, &geom);
    let positions: Vec<u32> = (0..n).collect();
    pass.input_tokens(input, &positions);
    pass.sampler(program, bindings);
    pass.execute();
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("read: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let params: serde_json::Value = serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let k: u32 = params.get("k").and_then(|v| v.as_u64()).unwrap_or(4).max(2) as u32;
    let vocab = model::output_vocab_size();
    let page = KvWorkingSet::new().page_size();

    let mut prompt = model::encode("The quick brown fox jumps over");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let l = prompt.len() as u32;

    // ── 1. Target greedy continuation g[0..k): k sequential Argmax decodes. ──
    let g: Vec<u32> = {
        let kv = KvWorkingSet::new();
        let greedy = sampler::sampler_program(sampler::SamplerSpec::Argmax, vocab)?;
        let mut seq_len = 0u32;
        let mut fresh = true;
        let mut pending = prompt.clone();
        let mut acc = Vec::new();
        for _ in 0..k {
            let n = pending.len() as u32;
            let pass = ForwardPass::new();
            if fresh {
                pass.fresh_generate();
                fresh = false;
            }
            let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq_len, n, page))?;
            geometry::attach_kv_write(&pass, &kv, &geom);
            let positions: Vec<u32> = (seq_len..seq_len + n).collect();
            pass.input_tokens(&pending, &positions);
            pass.sampler(&greedy.program, greedy.bindings(seq_len + n - 1)?);
            pass.execute();
            seq_len += n;
            let out = pass.output().await.map_err(|e| format!("g output: {e}"))?;
            let bytes = out.read().map_err(|e| format!("g read: {e:?}"))?;
            let t = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            acc.push(t);
            pending = vec![t];
        }
        acc
    };
    if (g.len() as u32) < k {
        return Err(format!("greedy produced {} < k={} tokens", g.len(), k).into());
    }
    let g: Vec<u32> = g[..k as usize].to_vec();

    // forward input = prompt + cont[0..k-1) so the k logit positions [L-1..L+k-2]
    // exist with each row's argmax conditioned on cont's prefix.
    let base_inp = |cont: &[u32]| {
        let mut inp = prompt.clone();
        inp.extend_from_slice(&cont[..(k as usize - 1)]);
        inp
    };
    // Each arm forks a FRESH (empty) working set, so the pass start position is 0
    // and the k logit rows land at [L-1 .. L-1+k).
    let positions = |start: u32| -> Vec<u32> { (0..k).map(|i| start + l - 1 + i).collect() };

    // ── 2. ARGMAX-MATRIX PROBE: [k,vocab] intrinsic → per-row argmax → [k]-Token.
    //       No draft, no -1 sentinel → all k argmax values emit. ──
    let probe_built = {
        let gp = Graph::new(vocab);
        let logits = gp.intrinsic_logits_matrix_dyn(k); // [k, vocab]
        let am = logits.argmax(); // [k] i32 per-row argmax
        gp.output(&am, OutputKind::Token);
        gp.build().map_err(|e| format!("probe build: {e:?}"))?
    };
    let probe_program = inferlet::emit::emit_program(&probe_built.program)
        .map_err(|e| format!("probe emit: {e}"))?;
    let probe_out: Vec<u32> = {
        let bindings = resolve_bindings(
            &probe_built.bindings,
            &probe_built.host_inputs,
            &positions(0),
            &[],
        )?;
        fire_token_channel(&probe_program, bindings, &base_inp(&g), page).await?
    };

    // ── 3. SPEC-VERIFY mixed arm (bonus): reject at j=k/2 ⇒ accept-prefix + -1. ──
    let (sv_built, keys) = edsl::spec_verify_greedy(vocab, k)
        .map_err(|e| format!("spec_verify_greedy build: {e:?}"))?;
    let sv_program =
        inferlet::emit::emit_program(&sv_built.program).map_err(|e| format!("spec emit: {e}"))?;
    let j = (k / 2).max(1) as usize;
    let mut mixed_draft = g.clone();
    // Perturb to a wrong SUBMIT draft at j that is ≠ g[j] AND ≠ 0 — the 0-sentinel
    // reject-MID detector REQUIRES all drafts ∈ [1, vocab) (foxtrot's invariant): a
    // token-0 draft is indistinguishable from the reject→0 sentinel, re-introducing
    // the mask. `.max(1)` only bumps the wrap-to-0 case (g[j]==vocab-1) to 1.
    mixed_draft[j] = ((g[j] + 1) % vocab).max(1);
    let mixed_out: Vec<u32> = {
        // INPUT = the greedy continuation (so each row's argmax == g); ONLY the
        // submit draft is perturbed → the spec-verify rejects at j (argmax g[j] !=
        // draft[j]) → accept-prefix g[0..j] + sentinels. (Perturbing the INPUT
        // instead would change the argmax conditioning — not a draft-rejection test.)
        let draft_i32: Vec<i32> = mixed_draft.iter().map(|&t| t as i32).collect();
        let bindings = resolve_bindings(
            &sv_built.bindings,
            &sv_built.host_inputs,
            &positions(0),
            &[(keys.draft, encode_i32(&draft_i32))],
        )?;
        fire_token_channel(&sv_program, bindings, &base_inp(&g), page).await?
    };

    // ── ACCEPT-ALL spec-verify arm: draft == g → all rows match → output == g (no
    //    -1). The probe proves the matrix argmax is value-exact; this adds the
    //    spec-verify DAG (eq/cumprod/select) on top. If this == g, the DAG is
    //    value-exact; if truncated, the DAG (not the matrix) is the residual. ──
    let accept_sv_out: Vec<u32> = {
        let draft_i32: Vec<i32> = g.iter().map(|&t| t as i32).collect();
        let bindings = resolve_bindings(
            &sv_built.bindings,
            &sv_built.host_inputs,
            &positions(0),
            &[(keys.draft, encode_i32(&draft_i32))],
        )?;
        fire_token_channel(&sv_program, bindings, &base_inp(&g), page).await?
    };

    // ── REJECT-MID arm (the NON-DEGENERATE cumprod-collapse detector). The marshal
    //    truncates `spec_verify_greedy`'s output at the first `-1`, so its reject-mid
    //    output is `[d0,d1]` for BOTH the correct cross-row scan (`[d0,d1,-1,-1]`)
    //    AND the batched per-block leak (`[d0,d1,-1,d3]`) — the truncation MASKS the
    //    row-3 leak. So we build a 0-SENTINEL variant (reject → 0, NOT -1): 0 ≥ 0 so
    //    the marshal emits ALL k → the cross-row accept-prefix is OBSERVABLE.
    //    matched=[1,1,0,1] ⇒ correct cumprod `[1,1,0,0]` → `[g0,g1,0,0]`; a batched
    //    per-block cumprod `[1,1,0,1]` → `[g0,g1,0,g3]` (row 3 LEAKS draft past the
    //    reject). Input = greedy g (argmax==g); only the SUBMIT draft is perturbed.
    let rj_draft = mixed_draft.clone(); // [g0, g1, g2+1, g3] — reject exactly at j=2
    let (rj_built, rj_key) = {
        let gp = Graph::new(vocab);
        let logits = gp.intrinsic_logits_matrix_dyn(k);
        let draft = gp.host_vector_dyn(DType::I32, k, Readiness::Submit);
        let target = logits.argmax();
        let matched = target.eq(&draft);
        let ones = gp.constant_f32_dyn(1.0).broadcast_vec(k);
        let zeros = gp.constant_f32_dyn(0.0).broadcast_vec(k);
        let acc = dselect(&matched, &ones, &zeros).cumprod(); // CROSS-ROW prefix-AND
        let keep = acc.gt(&gp.constant_f32_dyn(0.5));
        let zero_sentinel = gp.constant_i32_dyn(0).broadcast_vec(k); // ≥0 → NO truncation
        let out = dselect(&keep, &draft, &zero_sentinel);
        gp.output(&out, OutputKind::Token);
        let key = draft.input_key().expect("rj draft input");
        (gp.build().map_err(|e| format!("rj build: {e:?}"))?, key)
    };
    let rj_program =
        inferlet::emit::emit_program(&rj_built.program).map_err(|e| format!("rj emit: {e}"))?;
    let reject_mid_out: Vec<u32> = {
        // greedy conditioning ⇒ argmax == g; only the SUBMIT draft is perturbed
        let draft_i32: Vec<i32> = rj_draft.iter().map(|&t| t as i32).collect();
        let bindings = resolve_bindings(
            &rj_built.bindings,
            &rj_built.host_inputs,
            &positions(0),
            &[(rj_key, encode_i32(&draft_i32))],
        )?;
        fire_token_channel(&rj_program, bindings, &base_inp(&g), page).await?
    };

    // ── Verdicts ──
    let channel_emits_k = probe_out.len() == k as usize;
    let matrix_argmax_ok = probe_out == g; // [k,vocab] feeds each row its own position
    let pj = j.min(mixed_out.len());
    let spec_prefix_ok = pj > 0 && mixed_out[..pj] == g[..pj];
    let spec_accept_all_ok = accept_sv_out == g; // full DAG value-exact when all-accept
    // The cross-row accept-prefix: accept [0,j) == g, reject at j → 0, and rows > j
    // ALSO 0 (the cross-row cumprod zeros the whole suffix). A batched per-block
    // cumprod LEAKS draft at matched rows past j → reject_mid_out[r>j] == g[r] != 0.
    let expect_rj: Vec<u32> = (0..k as usize)
        .map(|i| if i < j { g[i] } else { 0 })
        .collect();
    let reject_mid_ok = reject_mid_out == expect_rj;
    // The 0-sentinel detector's invariant (foxtrot): a token-0 draft is
    // indistinguishable from a reject→0 sentinel → the mask returns. So ALL drafts
    // (the greedy accept-prefix g AND the perturbed wrong token) must be ∈ [1,vocab).
    // If g itself contains a 0 (model-dependent), the gate is ambiguous — flag it.
    let drafts_nonzero = g.iter().all(|&t| t != 0) && rj_draft.iter().all(|&t| t != 0);

    let result = format!(
        "CHANNEL_EMITS_K={channel_emits_k} MATRIX_ARGMAX_OK={matrix_argmax_ok} \
         SPEC_PREFIX_OK={spec_prefix_ok} SPEC_ACCEPT_ALL_OK={spec_accept_all_ok} \
         REJECT_MID_OK={reject_mid_ok} DRAFTS_NONZERO={drafts_nonzero} k={k} j={j} \
         g={g:?} probe_len={} probe_out={probe_out:?} mixed_out={mixed_out:?} \
         accept_sv_out={accept_sv_out:?} reject_mid_out={reject_mid_out:?} expect_rj={expect_rj:?}",
        probe_out.len()
    );
    eprintln!("[K-MARSHAL] {result}");
    Ok(result)
}
