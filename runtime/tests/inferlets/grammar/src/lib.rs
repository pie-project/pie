//! Grammar mask-apply OP verify (cut #2(a), `MASK_OP_OK`).
//!
//! Verifies the de-hardwired grammar-masking OP (Sampling-IR `0x65 mask-apply`)
//! end-to-end through the SUBMIT-mask path: a host matcher computes a per-step
//! allowed-token set, packs it into a `[ceil(vocab/32)]` u32 bitmask (bit 1 =
//! allowed), binds it as a submit-bound program input, and fires
//! `grammar_submit_with_logits` = `argmax(mask_apply(logits, mask))` returning
//! `[token, raw_logits]`. Two non-degenerate asserts (the cut #1 discipline — an
//! all-allowed no-op mask cannot pass):
//!   1. CONFORM: every step, the device token == `apply_mask_argmax(raw_logits,
//!      mask)` recomputed host-side with the byte-identical CPU reference
//!      (`inferlet::mask`) — proves device mask-apply == host semantics (closes
//!      the host<->device drift class that bit cut #1).
//!   2. FORCED-OUT: at step 0 the raw logits ARE the unconstrained logits (the
//!      history is the prompt only), so the natural argmax `u0` is well-defined;
//!      assert `u0` is DISALLOWED (bit 0) in the mask AND the constrained token
//!      != `u0` — positively proves the `-inf` fired (not passthrough). The
//!      matcher's small alphabet disallows the model's natural (large-id) pick,
//!      forcing the divergence; this subsumes "constrained != unconstrained".
//!
//! SCOPE — op-semantics ONLY. This verifies the `0x65` mask-apply OP via the
//! Submit supply path. It does NOT exercise the Late-channel supply path
//! (`tensor.write` -> device-alias carrier -> `HostLate`); that production path
//! requires its OWN GPU verify when the Late wiring lands (the cut #1
//! host<->device supply-drift class). `MASK_OP_OK` MUST NOT stand in for the
//! Late-supply verify.

use inferlet::inference::ForwardPass;
use inferlet::mask::{all_allowed, apply_mask_argmax, bf16_hi_to_f32, bit_allowed, pack_allowed};
use inferlet::serde_json;
use inferlet::working_set::KvWorkingSet;
use inferlet::{geometry, model, sampler, Result};

/// Constraint alphabet: the only token ids the grammar ever allows. Small ids
/// the model never naturally argmaxes, so the natural pick is always forced out.
const ALPHABET: [u32; 4] = [10, 11, 12, 13];
/// Default tokens to generate; override via `{"max_tokens":N}`.
const MAX_TOKENS: usize = 12;

/// Tiny DFA: allow any alphabet token except the one just emitted (no repeats).
struct NoRepeatMatcher {
    last: Option<u32>,
}

impl NoRepeatMatcher {
    fn new() -> Self {
        Self { last: None }
    }
    fn allowed(&self) -> Vec<u32> {
        ALPHABET
            .iter()
            .copied()
            .filter(|&t| Some(t) != self.last)
            .collect()
    }
    fn accept(&mut self, token: u32) {
        self.last = Some(token);
    }
}

/// Read a raw-logits output tensor as `f32`, converting bf16 storage exactly as
/// the driver does (`bf16_hi_to_f32`) so the host CPU reference is byte-identical
/// to the device's `0x65` f32-from-bf16 compute.
fn logits_as_f32(bytes: &[u8], vocab: usize) -> Vec<f32> {
    if bytes.len() == vocab * 2 {
        bytes
            .chunks_exact(2)
            .map(|c| bf16_hi_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let params: serde_json::Value = serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let max_tokens = params
        .get("max_tokens")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(MAX_TOKENS);

    // Logits/output vocab (= hf_config.vocab_size); the program masks + samples
    // over the logits dim.
    let vocab = model::output_vocab_size();

    let mut prompt = model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }

    // Keep-core masked grammar sampler with raw logits: `argmax(mask_apply(logits,
    // mask))` → outputs [0]=constrained token, [1]=raw (unmasked) logits (read via
    // `ForwardPass::outputs()`). Off the `Context`/`Forward`/`program` facade.
    let g = sampler::grammar_program_with_logits(vocab)?;

    let kv = KvWorkingSet::new();
    let mut seq = 0u32;
    let mut fresh = true;

    let mut matcher = NoRepeatMatcher::new();
    let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut pending: Vec<u32> = prompt;

    let mut conform_ok = true;
    let mut forced_out_ok = false;
    let mut natural0: i64 = -1;

    for step in 0..max_tokens {
        let allowed = matcher.allowed();
        let packed = pack_allowed(vocab as usize, &allowed);

        // Raw grammar fire (geometry + input + masked sampler + execute), reading
        // BOTH outputs (Token, raw Logits) via the raw `outputs()` multi-output.
        let n = pending.len() as u32;
        let pass = ForwardPass::new();
        if fresh {
            pass.fresh_generate();
            fresh = false;
        }
        let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq, n, kv.page_size()))?;
        geometry::attach_kv_write(&pass, &kv, &geom);
        let positions: Vec<u32> = (seq..seq + n).collect();
        pass.input_tokens(&pending, &positions);
        let decode_pos = seq + n - 1;
        pass.sampler(&g.program, g.bindings(decode_pos, &packed)?);
        pass.execute();
        seq += n;

        let outs = pass
            .outputs()
            .await
            .map_err(|e| format!("outputs @{step}: {e}"))?;
        let tok_bytes = outs[0].read().map_err(|e| format!("read token @{step}: {e:?}"))?;
        let token = if tok_bytes.len() >= 4 {
            i32::from_le_bytes([tok_bytes[0], tok_bytes[1], tok_bytes[2], tok_bytes[3]]) as u32
        } else {
            0
        };
        let logit_bytes = outs[1].read().map_err(|e| format!("read logits @{step}: {e:?}"))?;
        let logits = logits_as_f32(&logit_bytes, vocab as usize);

        // Assert #1 CONFORM: device token == host apply_mask_argmax(raw logits, mask).
        let host_token = apply_mask_argmax(&logits, &packed);
        if token != host_token {
            conform_ok = false;
            eprintln!(
                "[GRAMMAR-OP] CONFORM mismatch @{step}: device={token} host={host_token}"
            );
        }
        // Grammar conformance invariant: the constrained token is in the alphabet.
        if !ALPHABET.contains(&token) {
            conform_ok = false;
            eprintln!("[GRAMMAR-OP] grammar violated @{step}: {token} not in alphabet");
        }

        // Assert #2 FORCED-OUT @ step 0: the natural argmax is disallowed + forced out.
        if step == 0 {
            // Step-0 raw logits ARE the unconstrained logits (history = prompt only).
            let u0 = apply_mask_argmax(&logits, &all_allowed(vocab as usize));
            natural0 = u0 as i64;
            let disallowed = !bit_allowed(&packed, u0 as usize);
            forced_out_ok = disallowed && token != u0;
            eprintln!(
                "[GRAMMAR-OP] forced-out @0: natural={u0} disallowed={disallowed} constrained={token}"
            );
        }

        matcher.accept(token);
        tokens.push(token);
        pending = vec![token];
    }

    let mask_op_ok = conform_ok && forced_out_ok;
    // JSON result, matching the sibling capability-inferlet convention (mirostat):
    // `sampler` names the path, `conformant` is the grammar-enforcement verdict
    // (CONFORM ∧ FORCED-OUT), `count` is the tokens generated. The harness
    // (`programmable_sampler_e2e`) asserts these fields.
    let tokens_json = serde_json::to_string(&tokens).unwrap_or_else(|_| "[]".to_string());
    let result = format!(
        "{{\"sampler\":\"grammar\",\"conformant\":{mask_op_ok},\"conform\":{conform_ok},\
         \"forced_out\":{forced_out_ok},\"natural0\":{natural0},\"count\":{},\
         \"tokens\":{tokens_json}}}",
        tokens.len(),
    );
    eprintln!("[GRAMMAR-OP] {result}");
    Ok(result)
}
