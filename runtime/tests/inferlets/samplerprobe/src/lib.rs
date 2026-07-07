//! **Sampler-lowering keep-core primitive exercise** (echo). Proves an inferlet
//! author gets a PARAMETRIC sampler (top-p, with temperature / top-p riding as
//! host-submit tensors) on the RAW WIT surface via `sampler::sampler_program`
//! — no hand-built Sampling-IR `Graph`, no `Context`/`Generator` facade. The
//! sampler analog of the `geometry`/`carrier` keep-core exercises
//! (`ptir-sdk-minimization-audit`):
//!
//!   - `sampler::sampler_program(spec, vocab)` — lower a standard sampler spec
//!     to an attachable `tensor::Program` + its binding template / submit params
//!     (the param-invariant `standard_program`, NOT baked immediates);
//!   - `LoweredSampler::bindings(decode_pos)` — resolve the per-fire
//!     `InputBinding` list (logits row + the top-p param submit tensors);
//!   - `geometry::*` (keep-core) — the KV read/write page split per step;
//!   - raw `ForwardPass` — `input_tokens` / `sampler` / `execute` / `output`.
//!
//! On the mock `EchoBehavior(42)` every fire echoes token 42, so a 3-step decode
//! returns `[42, 42, 42]` — the clean signal that the full lowering + parametric
//! binding-resolution path runs end-to-end.

use inferlet::geometry;
use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{model, Result};

const MAX_TOKENS: usize = 3;

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = model::output_vocab_size();

    // The keep-core sampler-lowering primitive: standard spec → attachable
    // program + per-fire bindings. Built once, reused across decode steps.
    let s = sampler::sampler_program(SamplerSpec::TopP { temperature: 0.8, p: 0.9 }, vocab)?;

    let kv = KvWorkingSet::new();
    let page = kv.page_size();

    let prompt = model::encode("hello world");
    let mut pending: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let mut seq_len: u32 = 0;
    let mut generated: Vec<u32> = Vec::with_capacity(MAX_TOKENS);

    for step in 0..MAX_TOKENS {
        let n = pending.len() as u32;

        let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq_len, n, page))?;
        let pass = ForwardPass::new();
        geometry::attach_kv_write(&pass, &kv, &geom);

        let positions: Vec<u32> = (seq_len..seq_len + n).collect();
        pass.input_tokens(&pending, &positions);

        // Attach the lowered sampler: bind its logits slot to the decode row and
        // its param slots (T / p) to the per-fire submit tensors.
        let decode_pos = seq_len + n - 1;
        pass.sampler(&s.program, s.bindings(decode_pos)?);

        pass.execute();
        let out = pass
            .output()
            .await
            .map_err(|e| format!("output @{step}: {e}"))?;
        let bytes = out.read().map_err(|e| format!("tensor read @{step}: {e:?}"))?;
        let token = if bytes.len() >= 4 {
            i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
        } else {
            return Err(format!("output @{step}: short tensor ({} bytes)", bytes.len()));
        };

        generated.push(token);
        seq_len += n;
        pending = vec![token];
    }

    let result = format!("sampled {} tokens: {:?}", generated.len(), generated);
    eprintln!("[SAMPLERPROBE] {result}");
    Ok(result)
}
