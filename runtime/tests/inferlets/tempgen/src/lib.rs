//! Temperature-sampling test inferlet (echo, Task #4 verify).
//!
//! Drives `SamplerSpec::Multinomial { temperature }`, which lowers to
//! `Spec::Multinomial { temperature }` → per-row params `(T>0, top_k=0,
//! top_p=1, min_p=0)` → the host recognizer classifies it **Temperature** →
//! the BakedIR de-hardwiring path (`PIE_DEHARDWIRE_STD_SAMPLERS`). Used by the
//! 4090 executor-integration verify to prove temp fires route to the baked IR
//! `standard_sampler_program(Temperature, V)` and produce valid tokens.

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
    let prompt_tokens = model::encode("hello world");
    eprintln!("[TEMPGEN] encoded prompt: {} tokens", prompt_tokens.len());

    let s = sampler_program(SamplerSpec::Multinomial { temperature: 0.8 }, vocab)
        .map_err(|e| format!("sampler_program(Multinomial): {e}"))?;

    let max_tokens: usize = 8;
    let mut ctx = Ctx::new();
    let mut pending = prompt_tokens;

    let mut generated = Vec::new();
    for _ in 0..max_tokens {
        let t = ctx.sample_fire(&s, &pending).await?;
        eprintln!("[TEMPGEN] got token: {t}");
        generated.push(t);
        pending = vec![t];
    }

    let text = model::decode(&generated);
    eprintln!("[TEMPGEN] generated {} tokens: {:?}", generated.len(), text);
    Ok(format!("{{\"tokens\": {generated:?}, \"text\": {text:?}}}"))
}
