//! Grammar-constrained decode on the PTIR surface (`MASK_OP_OK`, §6.1 slice).
//!
//! Migrated off the deleted classic surface (`ForwardPass` + submit-bound
//! Sampling-IR `grammar_program_with_logits`) onto `inferlet::ptir`: ONE
//! loop-carried decode pass whose epilogue enforces the grammar in-graph —
//! `reduce_argmax(mask_apply(logits, mask))` — with the per-step allowed-token
//! mask supplied as a host-writer bool channel (`gmask.put` before each submit,
//! the D2 instance-data path) and the RAW (unmasked) logits published back to
//! the host on a reader channel. Two non-degenerate asserts (the cut #1
//! discipline — an all-allowed no-op mask cannot pass):
//!   1. CONFORM: every step, the device token == `apply_mask_argmax(raw_logits,
//!      mask)` recomputed host-side with the byte-identical CPU reference
//!      (`inferlet::mask`) — proves device mask-apply == host semantics.
//!   2. FORCED-OUT: at some step the natural (unconstrained) argmax is
//!      DISALLOWED by the mask AND the constrained token diverges from it —
//!      positively proves the `-inf` fired (not passthrough). The matcher's
//!      small alphabet disallows the model's natural pick almost every step;
//!      evaluating the divergence at EVERY step (any-step ∃, not step-0-only)
//!      keeps the probe deterministic on the eval-mock's 32-token vocab, where
//!      a single step's natural pick lands inside the 4-token alphabet ~12% of
//!      the time. `natural0` still reports step 0's natural argmax.
//!
//! The device mask semantics are identical to the old `0x65 mask-apply`: the
//! PTIR `mask_apply` expands to `select(mask, logits, -inf)` and the host
//! reference packs the same allowed set into the `inferlet::mask` bitmask.

use inferlet::mask::{all_allowed, apply_mask_argmax, bit_allowed, pack_allowed};
use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model, serde_json};

/// Constraint alphabet: the only token ids the grammar ever allows. Small ids
/// the model rarely naturally argmaxes, so the natural pick is forced out.
const ALPHABET: [u32; 4] = [10, 11, 12, 13];
/// Default tokens to generate; override via `{"max_tokens":N}`.
const MAX_TOKENS: usize = 12;
/// Tokens per KV page (matches the mock env / test drivers).
const PAGE_T: u32 = 16;
const NUM_LAYERS: u32 = 1;

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

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let params: serde_json::Value = serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let max_tokens = params
        .get("max_tokens")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(MAX_TOKENS);

    // Logits/output vocab; the program masks + argmaxes over the logits dim.
    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);

    // Seed the loop-carried token chain from the prompt tail. (The classic
    // guest prefilled the whole prompt in pass 1; the constraint verdicts are
    // input-independent — the mask/argmax act on the read-out logits — so the
    // ptir form seeds the 1-wide chain directly.)
    let mut prompt = wit_model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let seed_tok = *prompt.last().unwrap() as i32;

    let ws = WorkingSet::new();
    let max_pages = (max_tokens as u32 + 1).div_ceil(PAGE_T).max(1);
    ws.reserve(max_pages)
        .map_err(|e| format!("ws.reserve: {e}"))?;

    // Channels: tok_in is the device loop-carried token (seeded; each fire's
    // embed takes it, the epilogue re-puts the constrained pick); gmask is the
    // per-step host-fed allowed mask; tok_out/raw are host-reader outputs
    // (constrained token + RAW unmasked logits for the CPU reference).
    let tok_in = Channel::from(vec![seed_tok]).named("tok_in");
    let kv_len = Channel::from(vec![1u32]).named("kv_len");
    let gmask = Channel::new([vocab], dtype::bool).named("gmask");
    let tok_out = Channel::new([1], dtype::i32).named("tok_out");
    let raw = Channel::new([vocab], dtype::f32).named("raw");

    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, Tensor::constant(vec![0u32, 1]));
    fwd.port_channel(Port::KvLen, &kv_len);
    fwd.attn_working_set(&ws, .., ..)?;
    fwd.derive_dense_geometry();
    fwd.epilogue(move || {
        let length = kv_len.take().tensor();
        // Takes + compute first, puts last (value-id discipline). `tok` is
        // pre-reshaped to the channel cell shape so the terminal-output puts
        // (tok_out/raw, host-read) emit NO value-defining ops — their auto-
        // drain takes are trailing when the assembly drops them.
        let m = gmask.take(); // [V] bool, host-fed per step
        let lg = intrinsics::logits(); // [V] f32 (read-out row)
        let tok = reshape(reduce_argmax(mask_apply(&lg, &m)), [1]); // [1] i32
        tok_in.put(&tok);
        kv_len.put(add(&length, 1u32));
        tok_out.put(&tok);
        raw.put(&lg);
    });

    let mut matcher = NoRepeatMatcher::new();
    let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens);

    let mut conform_ok = true;
    let mut forced_out_ok = false;
    let mut natural0: i64 = -1;

    let pipeline = Pipeline::new();
    for step in 0..max_tokens {
        let allowed = matcher.allowed();
        let packed = pack_allowed(vocab as usize, &allowed);
        let mask_bool: Vec<bool> = (0..vocab as usize)
            .map(|j| bit_allowed(&packed, j))
            .collect();

        gmask.put(mask_bool);
        fwd.submit(&pipeline)
            .map_err(|e| format!("submit @{step}: {e}"))?;
        let token = tok_out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("tok_out.take @{step}: {e}"))?[0] as u32;
        let logits = raw
            .take()
            .get::<f32>()
            .await
            .map_err(|e| format!("raw.take @{step}: {e}"))?;

        // Assert #1 CONFORM: device token == host apply_mask_argmax(raw, mask).
        let host_token = apply_mask_argmax(&logits, &packed);
        if token != host_token {
            conform_ok = false;
            eprintln!("[GRAMMAR-OP] CONFORM mismatch @{step}: device={token} host={host_token}");
        }
        // Grammar conformance invariant: the constrained token is in the alphabet.
        if !ALPHABET.contains(&token) {
            conform_ok = false;
            eprintln!("[GRAMMAR-OP] grammar violated @{step}: {token} not in alphabet");
        }

        // Assert #2 FORCED-OUT: the natural argmax is disallowed + forced out
        // (evaluated every step; must fire at least once over the loop).
        let natural = apply_mask_argmax(&logits, &all_allowed(vocab as usize));
        if step == 0 {
            natural0 = natural as i64;
        }
        let disallowed = !bit_allowed(&packed, natural as usize);
        if disallowed && token != natural {
            if !forced_out_ok {
                eprintln!(
                    "[GRAMMAR-OP] forced-out @{step}: natural={natural} disallowed={disallowed} \
                     constrained={token}"
                );
            }
            forced_out_ok = true;
        }

        matcher.accept(token);
        tokens.push(token);
    }
    pipeline.close();

    let mask_op_ok = conform_ok && forced_out_ok;
    // JSON result, matching the sibling capability-inferlet convention:
    // `sampler` names the path, `conformant` is the grammar-enforcement verdict
    // (CONFORM ∧ FORCED-OUT), `count` is the tokens generated. The harness
    // (`north_star_grammar_constrained_decode`) asserts these fields.
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
