//! #18-class lock: a **single `[Entropy]` (Scalar)** measurement program.
//!
//! Fires `probe(Probe::Entropy)` — a lone `OutputKind::Scalar` program (Shannon
//! entropy `H = -Σ p·log p` of the softmax). Before the #19 fast-path gate fix,
//! `populate_output_fastpath` was eligible for ANY single fast-path-eligible
//! output (Token∧Scalar∧Entropy), so a lone `[Scalar]`/`[Entropy]` was wrongly
//! routed into the driver's a2 eager-D2H whose source is `pi.sampled[N-1]` (a
//! TOKEN) → the scalar slot got a token id's int-bits-as-f32 (a denormal ≈0).
//! After the fix (gate = exactly one `Token` output), this lone-Scalar program
//! takes the proven rich path → the real entropy. A plausible positive entropy
//! (NOT a ~1e-40 denormal) proves the #18-class is locked.

use inferlet::forward::Probe;
use inferlet::{model, Context, Result};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let mut context = Context::new()?;
    let mut prompt = model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }

    let mut pass = context.forward();
    pass.input(&prompt);
    // A lone [Entropy] (Scalar) output — the #18-class shape.
    let handles = pass
        .probe(Probe::Entropy)
        .map_err(|e| format!("probe(Entropy) build: {e}"))?;
    let out = pass.execute().await?;
    let entropy = out
        .scalar(handles[0])
        .await
        .map_err(|e| format!("read entropy scalar: {e}"))?;

    eprintln!("[ENTROPYCHECK] entropy={entropy}");
    Ok(format!("{{\"entropy\":{entropy}}}"))
}
