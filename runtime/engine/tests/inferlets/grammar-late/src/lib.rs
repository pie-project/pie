//! Late-channel grammar masking verify (cut #2 production supply, `LATE_MASK_OK`).
//!
//! The Late-supply counterpart to the `grammar` (`MASK_OP_OK`) op-verify: it
//! drives the de-hardwired grammar-masking OP (Sampling-IR `0x65 mask-apply`)
//! end-to-end through the **production Late-channel supply** — the packed mask is
//! a `Readiness::Late` host input (`grammar_with_logits`), so it rides bravo's
//! **device-alias carrier** (`sampling_late_device_*`) → @ingim's direct
//! WASM-slice→device memcpy (`pie_tensor_write_async`, no IPC staging) → echo's
//! `HostLate` resolve, instead of the submit-staged blob path.
//!
//! The mask supply is identical at the inferlet level (`resolve_bindings` builds
//! `Tensor::from_data(mask_bytes)` for the mask slot) — the HOST routes it by the
//! program's declared `InputDecl.ready == Late` into the device-alias channel.
//! So this exercises the cut-#1-supply-drift class on the **real** production
//! channel, which `MASK_OP_OK` (Submit) explicitly does NOT.
//!
//! Two non-degenerate asserts (the cut #1 discipline; an all-allowed no-op mask
//! cannot pass):
//!   1. CONFORM: every step, the device token == `apply_mask_argmax(raw_logits,
//!      mask)` recomputed host-side with the byte-identical CPU reference
//!      (`inferlet::mask`). A misaligned/dropped device-alias carrier ⇒ wrong or
//!      empty mask ⇒ conform fails (a dropped carrier ⇒ `SkippedLateBindMiss` ⇒
//!      the constrained token is the unconstrained argmax ⇒ also fails #2).
//!   2. FORCED-OUT @ step 0: the natural (unconstrained) argmax is DISALLOWED in
//!      the mask AND absent from the output (the `-inf` actually fired through the
//!      device-alias-supplied mask). The small alphabet disallows the model's
//!      natural (large-id) pick, so a transparent/dropped mask is caught loud.
//!
//! Run it on the MERGED run-ahead path (delta drives): the late carrier rides the
//! batch-merged request (`extend_sampling_programs_from` concat — the exact cut #1
//! drop site, now with the durable carrier-preservation guard).
//!
//! Build/run (GPU):
//!   cargo build --target wasm32-wasip2 -p grammar-late
//!   cargo test -p pie-bin --features driver-cuda --test cuda_grammar_late -- --ignored --nocapture

use inferlet::inference::ForwardPass;
use inferlet::mask::{all_allowed, apply_mask_argmax, bf16_hi_to_f32, bit_allowed, pack_allowed};
use inferlet::serde_json;
use inferlet::working_set::KvWorkingSet;
use inferlet::{geometry, model, sampler, Result};

/// Default constraint alphabet: the only token ids the grammar ever allows. Small
/// ids the model never naturally argmaxes, so the natural pick is always forced
/// out. Override via `{"alphabet":[..]}` — the gate-1 R=2 barrier launches two
/// procs with DISJOINT alphabets so their constrained tokens are distinguishable
/// (`tok_A ∈ alphabet_A` ≠ `tok_B ∈ alphabet_B` by construction, ruling out a
/// `per_req[0]→both` mask aliasing on the merged `forward_R=2` fire).
const ALPHABET: [u32; 4] = [10, 11, 12, 13];
/// Default tokens to generate; override via `{"max_tokens":N}`.
const MAX_TOKENS: usize = 8;

/// Tiny DFA: allow any alphabet token except the one just emitted (no repeats).
struct NoRepeatMatcher {
    alphabet: Vec<u32>,
    last: Option<u32>,
}

impl NoRepeatMatcher {
    fn new(alphabet: Vec<u32>) -> Self {
        Self {
            alphabet,
            last: None,
        }
    }
    fn allowed(&self) -> Vec<u32> {
        self.alphabet
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
    // The grammar's allowed alphabet. The gate-1 R=2 barrier passes two DISJOINT
    // alphabets (A=[10..13], B=[20..23]) so the two procs' constrained tokens are
    // distinguishable on the merged `forward_R=2` fire (no `per_req[0]→both` alias).
    let alphabet: Vec<u32> = params
        .get("alphabet")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect())
        .filter(|a: &Vec<u32>| !a.is_empty())
        .unwrap_or_else(|| ALPHABET.to_vec());

    // Logits/output vocab (= hf_config.vocab_size); the program masks + samples
    // over the logits dim.
    let vocab = model::output_vocab_size();

    let mut prompt = model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }

    // Keep-core masked grammar sampler with raw logits + a **Late** mask (the
    // production device-alias supply channel): `argmax(mask_apply(logits, mask))`
    // → outputs [0]=constrained token, [1]=raw logits (read via `outputs()`). Off
    // the `Context`/`Forward`/`program` facade; the mask value is supplied
    // identically (`Tensor::from_data`) — the program's `Readiness::Late` routes it
    // host-side into the device-alias carrier.
    let g = sampler::grammar_program_with_logits_late(vocab)?;

    let kv = KvWorkingSet::new();
    let mut seq = 0u32;
    let mut fresh = true;

    let mut matcher = NoRepeatMatcher::new(alphabet.clone());
    let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut pending: Vec<u32> = prompt;

    let mut conform_ok = true;
    let mut forced_out_ok = false;
    let mut natural0: i64 = -1;

    for step in 0..max_tokens {
        let allowed = matcher.allowed();
        let packed = pack_allowed(vocab as usize, &allowed);

        // Raw grammar fire (geometry + input + Late-masked sampler + execute),
        // reading BOTH outputs (Token, raw Logits) via the raw `outputs()`.
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
                "[GRAMMAR-LATE] CONFORM mismatch @{step}: device={token} host={host_token}"
            );
        }
        // Grammar conformance invariant: the constrained token is in the alphabet.
        if !alphabet.contains(&token) {
            conform_ok = false;
            eprintln!("[GRAMMAR-LATE] grammar violated @{step}: {token} not in alphabet");
        }

        // Assert #2 FORCED-OUT @ step 0: the natural argmax is disallowed + forced out.
        if step == 0 {
            // Step-0 raw logits ARE the unconstrained logits (history = prompt only).
            let u0 = apply_mask_argmax(&logits, &all_allowed(vocab as usize));
            natural0 = u0 as i64;
            let disallowed = !bit_allowed(&packed, u0 as usize);
            forced_out_ok = disallowed && token != u0;
            eprintln!(
                "[GRAMMAR-LATE] forced-out @0: natural={u0} disallowed={disallowed} constrained={token}"
            );
        }

        matcher.accept(token);
        tokens.push(token);
        pending = vec![token];
    }

    let late_mask_ok = conform_ok && forced_out_ok;
    let result = format!(
        "LATE_MASK_OK={late_mask_ok} conform={conform_ok} forced_out={forced_out_ok} \
         natural0={natural0} tokens={tokens:?}"
    );
    eprintln!("[GRAMMAR-LATE] {result}");
    Ok(result)
}
