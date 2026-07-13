//! Single-`[Token]` mask grammar inferlet for the #34 M-batch occupancy verify
//! — **`inferlet::ptir` bridge rewrite**. The classic M-batch-eligibility split
//! (a single-`Token`-output program vs a rich `[Token, Logits]` program) was an
//! executor batching-optimization detail of the old sampler-program surface;
//! the ptir bridge has no such shape distinction — every host input (here, the
//! grammar mask) rides a first-class host-writer [`Channel`], and only the
//! OUTPUTS an inferlet chooses to `put` cross back to the host. This inferlet
//! publishes only the constrained token (no raw-logits reader), mirroring the
//! old single-`[Token]` shape's OBSERVABLE surface.
//!
//! Self-check: each constrained token MUST be in the alphabet (the mask's
//! `−∞` fired). With DISJOINT alphabets across co-batched requests, a
//! wrong-grouping / scatter bug (request i gets request j's mask) yields a
//! token in `alphabet_j ∉ alphabet_i` → caught here, AND by the harness's
//! per-request ON==OFF token comparison.
//!
//! Loop-carried `pos`/`klen`/`fill` (the `isolatedtopp`/`mirostat` split) grow
//! the absolute RoPE position and the attended KV length by 1 every fire, so
//! each fire embeds at the growing absolute position and attends the FULL
//! committed KV — not a fixed 1-token window re-fired at position 0 forever.

use inferlet::mask::pack_allowed;
use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model, serde_json};

const ALPHABET: [u32; 4] = [10, 11, 12, 13];
const MAX_TOKENS: usize = 6;
const PAGE_T: u32 = 16;
const NUM_LAYERS: u32 = 1;

/// Tiny DFA: allow any alphabet token except the one just emitted (no repeats),
/// so the per-step mask varies (a non-trivial mask each step).
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
        .map(|n| n as usize)
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

    // tok_in: device loop-carried token (seeded; each fire's embed takes it,
    // the epilogue re-puts the constrained pick). pos: loop-carried absolute
    // RoPE position of THIS fire's token. klen/fill grow the attended KV
    // length by 1 every fire (klen is the port-bound channel `attn_working_set`
    // reads, fill is the internal carry the epilogue advances — the
    // `isolatedtopp`/`mirostat` split), so each fire embeds at the growing
    // absolute position and attends the FULL committed KV. gmask: per-step
    // host-writer allowed mask. tok_out: the SOLE host-reader output (no
    // raw-logits reader — mirrors the old single-`[Token]` M-batch-eligible
    // shape).
    let tok_in = bx(Channel::from(vec![seed_tok]).named("tok_in"));
    let pos = bx(Channel::from(vec![0u32]).named("pos"));
    let klen = bx(Channel::from(vec![1u32]).named("klen"));
    let fill = bx(Channel::from(vec![1u32]).named("fill"));
    let gmask = bx(Channel::new([vocab], dtype::bool).named("gmask"));
    let tok_out = bx(Channel::new([1], dtype::i32).named("tok_out"));

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
        pos.put(&base);
        klen.put(&klen_v);
        fill.put(&next_free);
    });

    let mut matcher = NoRepeatMatcher::new(alphabet.clone());
    let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut conform_ok = true;

    let pipeline = Pipeline::new();
    for step in 0..max_tokens {
        let allowed = matcher.allowed();
        let packed = pack_allowed(vocab as usize, &allowed);
        let mask_bool: Vec<bool> = (0..vocab as usize)
            .map(|j| inferlet::mask::bit_allowed(&packed, j))
            .collect();

        gmask.put(mask_bool);
        fwd.submit(&pipeline)
            .map_err(|e| format!("submit @{step}: {e}"))?;
        let token = tok_out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("tok_out.take @{step}: {e}"))?[0] as u32;

        // Grammar conformance: the masked argmax MUST be in this request's alphabet.
        if !alphabet.contains(&token) {
            conform_ok = false;
            eprintln!("[GRAMMARMB] grammar violated @{step}: {token} not in {alphabet:?}");
        }

        matcher.accept(token);
        tokens.push(token);
    }
    pipeline.close();

    let result = format!("GRAMMARMB_OK={conform_ok} tokens={tokens:?} alphabet={alphabet:?}");
    eprintln!("[GRAMMARMB] {result}");
    Ok(result)
}
