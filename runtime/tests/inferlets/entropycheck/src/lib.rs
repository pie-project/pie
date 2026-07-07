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

use inferlet::inference::ForwardPass;
use inferlet::sampler::{probe_program, ProbeKind};
use inferlet::working_set::KvWorkingSet;
use inferlet::{geometry, model, Result};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let mut prompt = model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let vocab = model::output_vocab_size();

    // Raw keep-core single fire (no `Context` facade): one KV working set +
    // cursor, a lone [Entropy] (Scalar) measurement at the last input position
    // via the keep-core `probe_program` (the de-hardwired `forward::Probe`).
    let kv = KvWorkingSet::new();
    let seq_len = 0u32;
    let n = prompt.len() as u32;
    let pass = ForwardPass::new();
    pass.fresh_generate();
    let geom = geometry::ensure_pages(
        &kv,
        geometry::kv_write_geometry(seq_len, n, kv.page_size()),
    )?;
    geometry::attach_kv_write(&pass, &kv, &geom);
    let positions: Vec<u32> = (seq_len..seq_len + n).collect();
    pass.input_tokens(&prompt, &positions);
    let probe = probe_program(ProbeKind::Entropy, vocab)
        .map_err(|e| format!("probe(Entropy) build: {e}"))?;
    pass.sampler(&probe.program, probe.bindings(seq_len + n - 1)?);
    pass.execute();

    // A lone [Entropy] (Scalar) output takes the rich path (#19 gate = exactly
    // one Token output); read the scalar off the raw output tensor.
    let out = pass.output().await.map_err(|e| format!("execute: {e}"))?;
    let bytes = out.read().map_err(|e| format!("read entropy: {e:?}"))?;
    let entropy = if bytes.len() >= 4 {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    } else {
        0.0
    };

    eprintln!("[ENTROPYCHECK] entropy={entropy}");
    Ok(format!("{{\"entropy\":{entropy}}}"))
}
