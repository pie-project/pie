//! Multi-sampler #7 per-kind coverage inferlet (echo).
//!
//! Generates a few tokens with each standard DedicatedKernel sampler kind in
//! sequence — TopK, TopP, MinP, TopKTopP — so one run exercises the executor's
//! recognizer→dispatch path across every kind that routes to a dedicated kernel
//! (FlashInfer top-k/p/joint + the temperature group). Used by the #7 cutover
//! per-kind `gate-on≡gate-off` verify: run with `PIE_RECOGNIZER_DISPATCH=1`
//! (recognizer drives the flag-set) vs unset (legacy flags), assert identical
//! tokens. Each sampler kind continues the same context, so the kinds appear as
//! distinct decode fires.

use inferlet::inference::ForwardPass;
use inferlet::sampler::{sampler_program, LoweredSampler, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{geometry, model, Result};

/// Raw keep-core decode context (no `Context`/`Generator` facade): one KV
/// working set + a cursor, threaded across all sampler kinds. The first sampling
/// fire clears the run-ahead carrier.
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

    let samplers: [(&str, SamplerSpec); 4] = [
        ("topk", SamplerSpec::TopK { temperature: 0.8, k: 40 }),
        ("topp", SamplerSpec::TopP { temperature: 0.8, p: 0.9 }),
        ("minp", SamplerSpec::MinP { temperature: 0.8, p: 0.05 }),
        ("joint", SamplerSpec::TopKTopP { temperature: 0.8, k: 40, p: 0.9 }),
    ];

    let mut ctx = Ctx::new();
    // The prompt feeds the FIRST fire; every subsequent fire (across all kinds)
    // feeds the previously sampled token — one continuous growing context.
    let mut pending = model::encode("hello world");

    let mut all = Vec::new();
    for (name, spec) in samplers {
        let s = sampler_program(spec, vocab)
            .map_err(|e| format!("sampler_program({name}): {e}"))?;
        let mut got = Vec::new();
        for _ in 0..4 {
            let t = ctx.sample_fire(&s, &pending).await?;
            got.push(t);
            pending = vec![t];
        }
        eprintln!("[MULTISAMP] {name} tokens: {got:?}");
        all.extend(got);
    }

    Ok(format!("{{\"tokens\": {all:?}}}"))
}
