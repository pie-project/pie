//! EBNF grammar-constrained decoding — **low-level ① rewrite (grammar, SEQUENTIAL)**.
//!
//! Off the `Context`/`Generator`/`Sampler`/`constrain_with` facade onto the
//! keep-core (`ptir-grammar-tranche-conversion-spec`):
//!   - `Ebnf(grammar).build_constraint()` → a kept `constraint::GrammarConstraint`
//!     (host `Matcher`): `advance(accepted)` + `mask()` (packed allowed-token bitmask);
//!   - `sampler::grammar_program` — the masked greedy sampler `argmax(mask_apply(
//!     logits, mask))` (the packed mask a per-fire submit tensor);
//!   - `geometry::*` KV split + raw `ForwardPass`.
//!
//! **Grammar decode is SEQUENTIAL** — the mask for token N+1 depends on token N
//! (the matcher advances on it), so the run-ahead carrier does NOT apply. This is
//! a real data dependency, not an avoidable bubble → the conversion is
//! facade-removal, not pipelining (same wall-clock the facade's constrained path
//! had). The mask is host-computed before each fire.

use inferlet::inference::ForwardPass;
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, sampler, Constrain, Ebnf, Result, Schema};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_num_tokens")]
    num_tokens: usize,
}

fn default_num_tokens() -> usize { 512 }

async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

/// One SEQUENTIAL grammar fire: geometry + input + the masked grammar sampler +
/// execute, advancing the cursor. No run-ahead carrier (grammar can't speculate:
/// the next mask depends on this token). `packed_mask` = the matcher's per-step
/// allowed bitmask.
fn grammar_fire(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    g: &sampler::LoweredGrammar,
    tokens: &[u32],
    packed_mask: &[u32],
) -> Result<ForwardPass> {
    let n = tokens.len() as u32;
    let pass = ForwardPass::new();
    if *fresh {
        pass.fresh_generate();
        *fresh = false;
    }
    let geom = geometry::ensure_pages(kv, geometry::kv_write_geometry(*seq_len, n, kv.page_size()))?;
    geometry::attach_kv_write(&pass, kv, &geom);
    let positions: Vec<u32> = (*seq_len..*seq_len + n).collect();
    pass.input_tokens(tokens, &positions);
    let decode_pos = *seq_len + n - 1;
    pass.sampler(&g.program, g.bindings(decode_pos, packed_mask)?);
    pass.execute();
    *seq_len += n;
    Ok(pass)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_tokens = input.num_tokens;

    let grammar = r#"
root ::= "[" person ("," person)* "]"
person ::= "{" "\"name\"" ":" string "," "\"age\"" ":" number "}"
string ::= "\"" [^"]+ "\""
number ::= [0-9]+
"#;

    let vocab = model::output_vocab_size();
    let g = sampler::grammar_program(vocab)?;
    let mut matcher = Ebnf(grammar).build_constraint()?;
    let stop = chat::stop_tokens();

    let mut prompt = chat::system_user(
        "You are a helpful assistant that outputs structured data in JSON format.",
        "List three famous scientists with their approximate birth years. \
         Format the output as a JSON array of objects with 'name' and 'age' fields. \
         For 'age', use their approximate birth year.",
    );
    prompt.extend(chat::cue());

    let start = std::time::Instant::now();

    let kv = KvWorkingSet::new();
    let mut seq = 0u32;
    let mut fresh = true;
    let mut pending = prompt; // first fire processes the whole prompt + samples token 0
    let mut tokens: Vec<u32> = Vec::new();

    for _ in 0..max_tokens {
        // The allowed set for THIS step (matcher state = tokens accepted so far).
        // An empty mask means "no restriction" → an all-ones (transparent) mask.
        let m = matcher.mask();
        let packed: Vec<u32> = if m.is_empty() { vec![u32::MAX; g.mask_words] } else { m };

        let pass = grammar_fire(&kv, &mut seq, &mut fresh, &g, &pending, &packed)?;
        let token = read_token(pass).await?;
        if stop.contains(&token) {
            break;
        }
        tokens.push(token);
        matcher.advance(&[token]); // advance the grammar for the next step's mask
        pending = vec![token];
    }

    let text = model::decode(&tokens).unwrap_or_default();
    println!("Generated (constrained):\n{}", text);
    println!("\nElapsed: {:?}", start.elapsed());

    Ok(String::new())
}
