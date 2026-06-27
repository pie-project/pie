//! Grammar / constrained-decoding test inferlet (Phase 2, WS3).
//!
//! Demonstrates the **sequential late-bind** loop for constrained decoding: a
//! host-side matcher computes a per-step allowed-token set, lowers it to an
//! additive logit-bias mask (`0.0` for allowed, `-inf` for disallowed), binds
//! it as a **submit-bound** host input, and fires a Sampling-IR `grammar`
//! program that returns `argmax(logits + mask)`. The matcher then advances on
//! the sampled token and the loop repeats — closing the constraint loop
//! entirely through the IR (no hardwired logit-mask path).
//!
//! The matcher here is a tiny DFA over a fixed token alphabet that forbids
//! immediate repeats, so the output is provably (a) always within the alphabet
//! and (b) never twice the same token in a row — assertable end-to-end.

use inferlet::program::{encode_f32, resolve_bindings};
use inferlet::sampling::program as edsl;
use inferlet::serde_json;
use inferlet::{Context, Result, model};

/// Constraint alphabet: the only token ids the grammar ever allows.
const ALPHABET: [u32; 4] = [10, 11, 12, 13];
/// Default number of tokens to generate; override via `_input` `"max_tokens"`.
const MAX_TOKENS: usize = 12;

/// Tiny DFA: allow any alphabet token except the one just emitted (no immediate
/// repeats). `last == None` at the start allows the whole alphabet.
struct NoRepeatMatcher {
    last: Option<u32>,
}

impl NoRepeatMatcher {
    fn new() -> Self {
        Self { last: None }
    }

    /// Allowed token ids at the current state.
    fn allowed(&self) -> Vec<u32> {
        ALPHABET
            .iter()
            .copied()
            .filter(|&t| Some(t) != self.last)
            .collect()
    }

    /// Advance on the sampled token.
    fn accept(&mut self, token: u32) {
        self.last = Some(token);
    }
}

/// Build an additive logit-bias mask: `0.0` for allowed ids, a large negative
/// bias elsewhere (drives those logits below any real value so `argmax` never
/// picks them).
fn additive_mask(vocab: usize, allowed: &[u32]) -> Vec<f32> {
    let mut mask = vec![f32::NEG_INFINITY; vocab];
    for &id in allowed {
        if (id as usize) < vocab {
            mask[id as usize] = 0.0;
        }
    }
    mask
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    // Optional `{"max_tokens":N}` run param; defaults to the built-in so `"{}"`
    // reproduces the standard config and lets a harness lengthen the run.
    let params: serde_json::Value = serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let max_tokens = params
        .get("max_tokens")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(MAX_TOKENS);

    // Logits/output vocab (= hf_config.vocab_size), not the tokenizer token
    // count: the sampler program is lowered + sampled over the logits dim.
    let vocab = model::output_vocab_size();

    let mut context = Context::new()?;
    // Prime the sequence with a short prompt so the first fire has a query
    // position; tokens are fed directly into the forward pass (no chat
    // template). Fall back to a single seed token if the prompt encodes empty
    // (keeps the first fire's query position well-defined on any tokenizer).
    let mut prompt = model::encode("start");
    if prompt.is_empty() {
        prompt.push(0);
    }

    // Build the grammar program once (binding-free; reusable). `keys.mask` is
    // the submit-bound additive-mask input refreshed every fire. Emit the WIT
    // `tensor::program` once.
    let (built, keys) =
        edsl::grammar(vocab).map_err(|e| format!("grammar program build: {e:?}"))?;
    let program =
        inferlet::emit::emit_program(&built.program).map_err(|e| format!("grammar emit: {e}"))?;
    let n_out = built.outputs.len() as u32;

    let mut matcher = NoRepeatMatcher::new();
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    // First fire processes the prompt and samples at its last position; each
    // later fire feeds back the single token just sampled.
    let mut pending: Vec<u32> = prompt;

    for _ in 0..max_tokens {
        let allowed = matcher.allowed();
        let mask = additive_mask(vocab as usize, &allowed);

        let mut pass = context.forward();
        let start = pass.start_position();
        pass.input(&pending);
        let decode_pos = start + pending.len() as u32 - 1;
        let bindings = resolve_bindings(
            &built.bindings,
            &built.host_inputs,
            &[decode_pos],
            &[(keys.mask, encode_f32(&mask))],
        )?;
        let handles = pass.sampler(&program, bindings, n_out);
        let out = pass.execute().await?;

        let token = out
            .token(handles[0])
            .await
            .map_err(|e| format!("grammar: read token: {e}"))?;

        // Invariant checks — the IR-constrained output must obey the grammar.
        if !ALPHABET.contains(&token) {
            return Err(format!("grammar violated: {token} not in alphabet"));
        }
        if matcher.last == Some(token) {
            return Err(format!("grammar violated: immediate repeat of {token}"));
        }

        matcher.accept(token);
        generated.push(token);
        pending = vec![token];
    }

    // Emit a structured result so a harness can assert conformance
    // programmatically (the per-step invariants above already hold, else we
    // returned Err). `conformant` is always true on the success path.
    let tokens_json = serde_json::to_string(&generated).unwrap_or_else(|_| "[]".to_string());
    let result = format!(
        "{{\"sampler\":\"grammar\",\"conformant\":true,\"count\":{},\"tokens\":{}}}",
        generated.len(),
        tokens_json
    );
    eprintln!("[GRAMMAR] {result}");
    Ok(result)
}
