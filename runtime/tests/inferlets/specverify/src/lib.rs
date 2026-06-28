//! `[k]`-Token marshal e2e verify (#32/#33) — k>1 OFF `spec_tokens`.
//!
//! The 2a `specverify` used **k=1** to dodge the `[k]`-Token marshal truncation.
//! #32/#33 route a `[k]`-Token (`elem_count > 1`) OFF the system-drafter
//! `spec_tokens` channel into the per-(request,output) two-level CSR
//! `program_tokens`, read back via [`forward::Output::tokens`].
//!
//! Two layers, separated on purpose:
//!  1. **ARGMAX-MATRIX PROBE (the marshal headline):** a `[k,vocab]` intrinsic →
//!     per-row `argmax` → `[k]`-Token, with NO `-1` sentinel → ALL k argmax values
//!     emit regardless of their correctness. So `probe_out.len() == k` PROVES the
//!     `[k]` marshal routes all k off `program_tokens` (the #32 claim) independent
//!     of any upstream value bug. `MARSHAL_EMITS_K` = this.
//!  2. **MATRIX-ARGMAX VALUES (diagnostic):** `probe_out == g` (the target's greedy
//!     continuation) iff the `[k,vocab]` matrix intrinsic feeds each row r the
//!     logits at position L-1+r. `MATRIX_ARGMAX_OK` = this. (k=1 only ever
//!     exercised row 0 — row≥1 is the new path.)
//!  3. **SPEC-VERIFY arm (bonus):** `spec_verify_greedy(k)` mixed draft ⇒
//!     accept-prefix + `-1` sentinels (the truncation semantics).

use inferlet::program::{encode_i32, resolve_bindings};
use inferlet::sample::Sampler;
use inferlet::sampling::program as edsl;
use inferlet::sampling::{Graph, OutputKind};
use inferlet::serde_json;
use inferlet::{Context, Result, model};

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

    let mut prompt = model::encode("The quick brown fox jumps over");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let l = prompt.len() as u32;
    let context = Context::new()?;

    // ── 1. Target greedy continuation g[0..k): k sequential Argmax decodes. ──
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

    // forward input = prompt + cont[0..k-1) so the k logit positions [L-1..L+k-2]
    // exist with each row's argmax conditioned on cont's prefix.
    let base_inp = |cont: &[u32]| {
        let mut inp = prompt.clone();
        inp.extend_from_slice(&cont[..(k as usize - 1)]);
        inp
    };
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
        let mut sctx = context.fork()?;
        let mut pass = sctx.forward();
        let start = pass.start_position();
        pass.input(&base_inp(&g));
        let bindings = resolve_bindings(
            &probe_built.bindings,
            &probe_built.host_inputs,
            &positions(start),
            &[],
        )?;
        let handles = pass.sampler(&probe_program, bindings, probe_built.outputs.len() as u32);
        let out = pass.execute().await?;
        out.tokens(handles[0])
            .await
            .map_err(|e| format!("probe tokens: {e}"))?
    };

    // ── 3. SPEC-VERIFY mixed arm (bonus): reject at j=k/2 ⇒ accept-prefix + -1. ──
    let (sv_built, keys) = edsl::spec_verify_greedy(vocab, k)
        .map_err(|e| format!("spec_verify_greedy build: {e:?}"))?;
    let sv_program =
        inferlet::emit::emit_program(&sv_built.program).map_err(|e| format!("spec emit: {e}"))?;
    let j = (k / 2).max(1) as usize;
    let mut mixed_draft = g.clone();
    mixed_draft[j] = (g[j] + 1) % vocab; // a wrong draft at j (≠ g[j])
    let mixed_out: Vec<u32> = {
        let mut sctx = context.fork()?;
        let mut pass = sctx.forward();
        let start = pass.start_position();
        pass.input(&base_inp(&mixed_draft));
        let draft_i32: Vec<i32> = mixed_draft.iter().map(|&t| t as i32).collect();
        let bindings = resolve_bindings(
            &sv_built.bindings,
            &sv_built.host_inputs,
            &positions(start),
            &[(keys.draft, encode_i32(&draft_i32))],
        )?;
        let handles = pass.sampler(&sv_program, bindings, sv_built.outputs.len() as u32);
        let out = pass.execute().await?;
        out.tokens(handles[0])
            .await
            .map_err(|e| format!("mixed tokens: {e}"))?
    };

    // ── Verdicts ──
    let marshal_emits_k = probe_out.len() == k as usize; // #32 headline (value-independent)
    let matrix_argmax_ok = probe_out == g; // [k,vocab] feeds each row its own position
    let pj = j.min(mixed_out.len());
    let spec_prefix_ok = pj > 0 && mixed_out[..pj] == g[..pj];

    let result = format!(
        "MARSHAL_EMITS_K={marshal_emits_k} MATRIX_ARGMAX_OK={matrix_argmax_ok} \
         SPEC_PREFIX_OK={spec_prefix_ok} k={k} j={j} \
         g={g:?} probe_len={} probe_out={probe_out:?} mixed_out={mixed_out:?}",
        probe_out.len()
    );
    eprintln!("[K-MARSHAL] {result}");
    Ok(result)
}
