//! Isolated top-p inferlet — a SINGLE TopP fire on a fresh context.
//!
//! Phase-1 #12 token gate (un-confounded): the `multisamp` harness shares one
//! context across 4 kinds, so top-p fires after top-k's tokens — and in phase-1
//! top-k stays CustomJIT, polluting top-p's input context (the full-sequence
//! `[…]×4` parity is therefore the phase-1+phase-2 done-bar). This inferlet
//! fires top-p ALONE on bare `"hello world"`, so its tokens depend only on the
//! prompt — a clean token-identity check: recognize TopP → extract(T=0.8,
//! p=0.9) → FlashInfer, vs the slot-surface baseline captured off `70e8082d`.
//!
//! Run with `PIE_FIXED_SAMPLING_SEED=12345` for reproducibility (ambient seed),
//! and `PIE_SAMPLING_IR_TRACE=1` to confirm the FlashInfer dispatch flip.

use inferlet::inference::ForwardPass;
use inferlet::sampler::{sampler_program, LoweredSampler, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{geometry, model, Result};

/// Raw keep-core decode context (no `Context`/`Generator` facade): one KV
/// working set + a cursor. The first sampling fire clears the run-ahead carrier.
struct Ctx {
    kv: KvWorkingSet,
    seq_len: u32,
    fresh: bool,
}
impl Ctx {
    fn new() -> Self {
        Self { kv: KvWorkingSet::new(), seq_len: 0, fresh: true }
    }
    /// One sequential sampler fire: geometry + input + sampler + execute; return
    /// the sampled token. Token-identical to the `Generator` run-ahead path.
    async fn sample_fire(&mut self, s: &LoweredSampler, tokens: &[u32]) -> Result<u32> {
        let n = tokens.len() as u32;
        let pass = ForwardPass::new();
        if self.fresh {
            pass.fresh_generate();
            self.fresh = false;
        }
        let geom = geometry::ensure_pages(
            &self.kv,
            geometry::kv_write_geometry(self.seq_len, n, self.kv.page_size()),
        )?;
        geometry::attach_kv_write(&pass, &self.kv, &geom);
        let positions: Vec<u32> = (self.seq_len..self.seq_len + n).collect();
        pass.input_tokens(tokens, &positions);
        pass.sampler(&s.program, s.bindings(self.seq_len + n - 1)?);
        pass.execute();
        self.seq_len += n;
        let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
        let bytes = out.read().map_err(|e| format!("read token: {e:?}"))?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    // P3 single-model: the engine serves exactly one model — no handle to
    // load; tokenizer is the global `model::*` API.
    let vocab = model::output_vocab_size();
    let s = sampler_program(SamplerSpec::TopP { temperature: 0.8, p: 0.9 }, vocab)
        .map_err(|e| format!("sampler_program(TopP): {e}"))?;

    let mut ctx = Ctx::new();
    let mut pending = model::encode("hello world");

    let mut got = Vec::new();
    for _ in 0..4 {
        let t = ctx.sample_fire(&s, &pending).await?;
        got.push(t);
        pending = vec![t];
    }
    eprintln!("[ISOLATED_TOPP] tokens: {got:?}");
    Ok(format!("{{\"tokens\": {got:?}}}"))
}
