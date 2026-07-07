use inferlet::inference::ForwardPass;
use inferlet::sampler::{probe_program, LoweredProbe, ProbeKind};
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, prefill, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_max_tokens")] max_tokens: usize,
    #[serde(default = "default_delta")] delta: f32,
    #[serde(default = "default_gamma")] gamma: f32,
}
fn default_max_tokens() -> usize { 256 }
fn default_delta() -> f32 { 4.0 }
fn default_gamma() -> f32 { 0.5 }

/// Raw keep-core decode context (no `Context` facade): one KV working set + a
/// sequence cursor. The first sampling fire clears the dangling run-ahead
/// carrier (`fresh_generate`); prefill materializes KV without sampling.
struct Ctx {
    kv: KvWorkingSet,
    seq_len: u32,
    fresh: bool,
}
impl Ctx {
    fn new() -> Self {
        Self { kv: KvWorkingSet::new(), seq_len: 0, fresh: true }
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        prefill::tokens(&self.kv, &mut self.seq_len, tokens)
    }
    /// One measurement fire: geometry + input + probe sampler + execute.
    fn probe_fire(&mut self, probe: &LoweredProbe, tokens: &[u32]) -> Result<ForwardPass> {
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
        pass.sampler(&probe.program, probe.bindings(self.seq_len + n - 1)?);
        pass.execute();
        self.seq_len += n;
        Ok(pass)
    }
}

/// Read a `[vocab]` f32 measurement (logits / distribution) off the raw output.
async fn read_f32(pass: ForwardPass) -> Result<Vec<f32>> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let vocab = model::output_vocab_size();
    let probe = probe_program(ProbeKind::Logits, vocab)
        .map_err(|e| format!("probe(Logits) build: {e}"))?;

    // Prompt: first user turn + generation cue, prefilled in full (the flush
    // twin). The decode loop then feeds the last cue token at the next position.
    let cue = chat::cue();
    let mut prompt = chat::first_user(&input.prompt);
    prompt.extend_from_slice(&cue);

    let mut ctx = Ctx::new();
    ctx.prefill(&prompt)?;

    let mut tokens = Vec::new();
    let mut last = *cue.last().unwrap();

    for _ in 0..input.max_tokens {
        let pass = ctx.probe_fire(&probe, &[last])?;
        let mut logits = read_f32(pass).await?;
        let vocab = logits.len() as u32;
        for t in green_list(last, vocab, input.gamma) { logits[t as usize] += input.delta; }
        last = argmax(&logits);
        tokens.push(last);
    }
    Ok(model::decode(&tokens)?)
}

fn green_list(seed: u32, vocab: u32, gamma: f32) -> impl Iterator<Item = u32> {
    let mut s = seed.wrapping_mul(0x9E3779B1).wrapping_add(1);
    let n = (vocab as f32 * gamma) as u32;
    (0..n).map(move |_| {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s % vocab
    })
}
fn argmax(xs: &[f32]) -> u32 {
    xs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
}
