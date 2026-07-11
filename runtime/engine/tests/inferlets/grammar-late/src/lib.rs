//! Late-channel grammar masking verify (cut #2 production supply, `LATE_MASK_OK`)
//! — **`inferlet::ptir` bridge rewrite**. The classic split between a
//! submit-staged blob mask and a `Readiness::Late` device-alias mask no
//! longer exists at the guest surface: every host input rides a first-class
//! `Channel`, and a host-writer channel's `put` (staged before `submit`,
//! coalesced into the fire) IS the production device-alias supply path — the
//! single channel mechanism now serves both the `grammar` (`MASK_OP_OK`) and
//! this `grammar-late` (`LATE_MASK_OK`) verify. Kept as its own inferlet (own
//! JSON knobs + result field names) so the R=2 barrier (`cuda_grammar_r2`)
//! keeps its two DISJOINT-alphabet, distinguishable-token proof.
//!
//! Two non-degenerate asserts (the cut #1 discipline — an all-allowed no-op
//! mask cannot pass):
//!   1. CONFORM: every step, the device token == `apply_mask_argmax(raw_logits,
//!      mask)` recomputed host-side with the byte-identical CPU reference
//!      (`inferlet::mask`).
//!   2. FORCED-OUT @ step 0: the natural (unconstrained) argmax is DISALLOWED
//!      in the mask AND absent from the output (the `-inf` actually fired
//!      through the channel-supplied mask).
//!
//! Loop-carried `pos`/`klen`/`fill` (the `isolatedtopp`/`mirostat` split: `klen`
//! is the port-bound channel `attn_working_set` reads, `fill` is the internal
//! carry the epilogue advances) grow the absolute RoPE position and the
//! attended KV length by 1 every fire, so each fire embeds at the growing
//! absolute position and attends the FULL committed KV — not a fixed 1-token
//! window re-fired at position 0 forever.
//!
//! JSON input: `{"alphabet":[..],"max_tokens":N}` (defaults `[10,11,12,13]`, 8).

use inferlet::mask::{all_allowed, apply_mask_argmax, bit_allowed, pack_allowed};
use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model, serde_json};

/// Default constraint alphabet: the only token ids the grammar ever allows.
const ALPHABET: [u32; 4] = [10, 11, 12, 13];
/// Default tokens to generate; override via `{"max_tokens":N}`.
const MAX_TOKENS: usize = 8;
const PAGE_T: u32 = 16;
const NUM_LAYERS: u32 = 1;

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

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let params: serde_json::Value = serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let max_tokens = params
        .get("max_tokens")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(MAX_TOKENS);
    let alphabet: Vec<u32> = params
        .get("alphabet")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect()
        })
        .filter(|a: &Vec<u32>| !a.is_empty())
        .unwrap_or_else(|| ALPHABET.to_vec());

    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);

    let mut prompt = wit_model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let seed_tok = *prompt.last().unwrap() as i32;

    let ws: &'static WorkingSet = bx(WorkingSet::new());
    ws.reserve(1).map_err(|e| format!("ws.reserve: {e}"))?;

    // Channels: tok_in is the device loop-carried token (seeded; each fire's
    // embed takes it, the epilogue re-puts the constrained pick); pos is the
    // loop-carried absolute RoPE position of THIS fire's token; klen/fill grow
    // the attended KV length by 1 every fire (klen is the port-bound channel
    // read by attn_working_set; fill is the internal loop-carry advancing it
    // — same split as `isolatedtopp`/`mirostat`) so each fire embeds at the
    // growing absolute position and attends the FULL committed KV, not a
    // fixed 1-token window. gmask is the per-step host-writer allowed mask
    // (the production channel supply path); tok_out/raw are host-reader
    // outputs (constrained token + RAW logits).
    let tok_in = bx(Channel::from(vec![seed_tok]).named("tok_in"));
    let pos = bx(Channel::from(vec![0u32]).named("pos"));
    let klen = bx(Channel::from(vec![1u32]).named("klen"));
    let fill = bx(Channel::from(vec![1u32]).named("fill"));
    let gmask = bx(Channel::new([vocab], dtype::bool).named("gmask"));
    let tok_out = bx(Channel::new([1], dtype::i32).named("tok_out"));
    let raw = bx(Channel::new([vocab], dtype::f32).named("raw"));

    let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd.embed(tok_in, Tensor::constant(vec![0u32, 1]));
    fwd.positions(pos);
    fwd.attn_working_set(ws, klen);
    fwd.epilogue(move || {
        // Takes + compute first, PUTS last (value-id discipline).
        let base = fill.take().tensor(); // [1] u32 — this fire's absolute position
        let m = gmask.take(); // [V] bool, host-fed per step
        let lg = intrinsics::logits(); // [V] f32 (read-out row)
        let tok = reshape(reduce_argmax(mask_apply(&lg, &m)), [1]); // [1] i32

        let klen_v = add(&base, 1u32); // cells 0..=base after this fire
        let next_free = add(&base, 1u32); // position the NEXT fire writes

        tok_in.put(&tok);
        tok_out.put(&tok);
        raw.put(&lg);
        pos.put(&base);
        klen.put(&klen_v);
        fill.put(&next_free);
    });

    let mut matcher = NoRepeatMatcher::new(alphabet.clone());
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
            .map_err(|e| format!("tok_out.take @{step}: {e}"))?[0] as u32;
        let logits = raw
            .take()
            .get::<f32>()
            .map_err(|e| format!("raw.take @{step}: {e}"))?;

        // Assert #1 CONFORM: device token == host apply_mask_argmax(raw, mask).
        let host_token = apply_mask_argmax(&logits, &packed);
        if token != host_token {
            conform_ok = false;
            eprintln!("[GRAMMAR-LATE] CONFORM mismatch @{step}: device={token} host={host_token}");
        }
        if !alphabet.contains(&token) {
            conform_ok = false;
            eprintln!("[GRAMMAR-LATE] grammar violated @{step}: {token} not in alphabet");
        }

        // Assert #2 FORCED-OUT @ step 0: the natural argmax is disallowed + forced out.
        if step == 0 {
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
    }
    pipeline.close();

    let late_mask_ok = conform_ok && forced_out_ok;
    let result = format!(
        "LATE_MASK_OK={late_mask_ok} conform={conform_ok} forced_out={forced_out_ok} \
         natural0={natural0} tokens={tokens:?}"
    );
    eprintln!("[GRAMMAR-LATE] {result}");
    Ok(result)
}
