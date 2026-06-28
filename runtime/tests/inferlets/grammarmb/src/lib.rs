//! Single-`[Token]` late-mask grammar inferlet for the #34 M-batch occupancy
//! verify. Uses `edsl::grammar(vocab)` — `argmax(mask_apply(logits, mask))` with
//! `outputs = [Token]` (single) and a `Readiness::Late` mask → **M-batch ELIGIBLE**
//! (the `*_with_logits` variants emit `[Token, Logits]` = rich → ineligible; the
//! `grammar`/`grammar-late` test inferlets use those for the host-reference
//! conform, so neither M-batches).
//!
//! Self-check: each constrained token MUST be in the alphabet (the mask's `−∞`
//! fired). With DISJOINT alphabets across co-batched requests, a wrong-grouping /
//! scatter bug (request i gets request j's gathered mask) yields a token in
//! `alphabet_j ∉ alphabet_i` → caught here, AND by the harness's per-request
//! ON==OFF token comparison. A dropped mask ⇒ the natural (large-id) argmax ∉ the
//! small alphabet ⇒ also caught (the forced-out property).

use inferlet::mask::pack_allowed;
use inferlet::program::{encode_u32, resolve_bindings};
use inferlet::sampling::program as edsl;
use inferlet::serde_json;
use inferlet::{Context, Result, model};

const ALPHABET: [u32; 4] = [10, 11, 12, 13];
const MAX_TOKENS: usize = 6;

/// Tiny DFA: allow any alphabet token except the one just emitted (no repeats),
/// so the per-step mask varies (a non-trivial Late mask each step).
struct NoRepeatMatcher {
    alphabet: Vec<u32>,
    last: Option<u32>,
}

impl NoRepeatMatcher {
    fn new(alphabet: Vec<u32>) -> Self {
        Self { alphabet, last: None }
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

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let params: serde_json::Value =
        serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let max_tokens = params
        .get("max_tokens")
        .and_then(|x| x.as_u64())
        .map(|n| n as usize)
        .unwrap_or(MAX_TOKENS);
    let alphabet: Vec<u32> = params
        .get("alphabet")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect())
        .filter(|a: &Vec<u32>| !a.is_empty())
        .unwrap_or_else(|| ALPHABET.to_vec());

    let vocab = model::output_vocab_size();
    let mut context = Context::new()?;
    let mut prompt = model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }

    // grammar(vocab): [Token]-only, Readiness::Late mask → M-batch eligible.
    let (built, keys) = edsl::grammar(vocab).map_err(|e| format!("grammar build: {e:?}"))?;
    let program =
        inferlet::emit::emit_program(&built.program).map_err(|e| format!("grammar emit: {e}"))?;
    let n_out = built.outputs.len() as u32;

    let mut matcher = NoRepeatMatcher::new(alphabet.clone());
    let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut pending: Vec<u32> = prompt;
    let mut conform_ok = true;

    for step in 0..max_tokens {
        let allowed = matcher.allowed();
        let packed = pack_allowed(vocab as usize, &allowed);

        let mut pass = context.forward();
        let start = pass.start_position();
        pass.input(&pending);
        let decode_pos = start + pending.len() as u32 - 1;
        let bindings = resolve_bindings(
            &built.bindings,
            &built.host_inputs,
            &[decode_pos],
            &[(keys.mask, encode_u32(&packed))],
        )?;
        let handles = pass.sampler(&program, bindings, n_out);
        let out = pass.execute().await?;
        let token = out
            .token(handles[0])
            .await
            .map_err(|e| format!("read token @{step}: {e}"))?;

        // Grammar conformance: the masked argmax MUST be in this request's alphabet.
        if !alphabet.contains(&token) {
            conform_ok = false;
            eprintln!("[GRAMMARMB] grammar violated @{step}: {token} not in {alphabet:?}");
        }

        matcher.accept(token);
        tokens.push(token);
        pending = vec![token];
    }

    let result = format!("GRAMMARMB_OK={conform_ok} tokens={tokens:?} alphabet={alphabet:?}");
    eprintln!("[GRAMMARMB] {result}");
    Ok(result)
}
