//! Demonstrates output validation by computing normalized probabilities over
//! a fixed candidate list, given a shared context prefix.
//!
//! For each candidate, fork the context and walk one token at a time,
//! reading the per-position distribution via `Probe::Distribution` and
//! accumulating log probability. The cumulative log-prob sequence is then
//! softmax-normalized across candidates.

use inferlet::inference::ForwardPass;
use inferlet::sampler::{probe_program, LoweredProbe, ProbeKind};
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, prefill, Result};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {}

/// Raw keep-core decode context (no `Context` facade): a KV working set +
/// sequence cursor, forkable with a COW-shared prefix.
struct Ctx {
    kv: KvWorkingSet,
    seq_len: u32,
    fresh: bool,
}
impl Ctx {
    fn new() -> Self {
        Self { kv: KvWorkingSet::new(), seq_len: 0, fresh: true }
    }
    /// COW-shared-prefix fork + cursor copy; the fork's first pass is `fresh`.
    fn fork(&self) -> Result<Self> {
        Ok(Self {
            kv: self.kv.fork().map_err(|e| format!("fork: {e}"))?,
            seq_len: self.seq_len,
            fresh: true,
        })
    }
    /// Non-sampling prefill (the facade's `flush`).
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

/// Read a dense `[vocab]` f32 distribution off the raw output tensor (indexed
/// by the implicit identity ids `0..vocab`).
async fn read_f32(pass: ForwardPass) -> Result<Vec<f32>> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Calculate the normalized probability of each candidate string being
/// generated from the given context.
async fn validate_outputs(
    base: &Ctx,
    probe: &LoweredProbe,
    candidates: &[String],
) -> Result<Vec<(String, f32)>> {
    let mut log_probs = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        let mut ctx = base.fork()?;
        let candidate_tokens = model::encode(candidate);

        // Bootstrap: feed an empty marker so the model produces a
        // distribution at the candidate's first token position. We use the
        // last token of the empty-string encoding as a no-op anchor.
        let mut pending = vec![*model::encode("").last().unwrap_or(&0)];
        let mut cumulative_log_prob = 0.0f32;

        for &target in &candidate_tokens {
            // The full softmax distribution (over all vocab) at the decode
            // position (the last input token); dense identity ids ⇒ the target's
            // probability is `probs[target]`.
            let pass = ctx.probe_fire(probe, &pending)?;
            let probs = read_f32(pass).await?;
            let p = probs.get(target as usize).copied().unwrap_or(0.0);

            if p > 0.0 {
                cumulative_log_prob += p.ln();
            } else {
                cumulative_log_prob = -1000.0;
                break;
            }

            // Feed the realized token forward for the next step.
            pending = vec![target];
        }
        log_probs.push(cumulative_log_prob);
    }

    let max_log_prob = log_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_log_prob.is_infinite() {
        let uniform = 1.0 / candidates.len() as f32;
        return Ok(candidates.iter().map(|c| (c.clone(), uniform)).collect());
    }

    let mut total = 0.0f32;
    let raw: Vec<f32> = log_probs
        .iter()
        .map(|&lp| {
            let p = (lp - max_log_prob).exp();
            total += p;
            p
        })
        .collect();

    Ok(candidates
        .iter()
        .zip(raw.iter())
        .map(|(c, &p)| (c.clone(), p / total))
        .collect())
}

#[inferlet::main]
async fn main(_input: Input) -> Result<String> {
    let start = Instant::now();
    let vocab = model::output_vocab_size();
    let probe = probe_program(ProbeKind::Distribution, vocab)
        .map_err(|e| format!("probe(Distribution) build: {e}"))?;

    // Prompt: system + first user turn + cue + the fixed prompt tail, prefilled
    // in full (the flush twin).
    let prompt_tail = "The name of the person in the report is ";
    let mut prompt = chat::system_user(
        "You are an expert at information extraction.",
        "From the sentence \"The financial report was prepared by David Chen.\", \
         extract the person's name.",
    );
    prompt.extend(chat::cue());
    prompt.extend(model::encode(prompt_tail));

    let mut ctx = Ctx::new();
    ctx.prefill(&prompt)?;

    let candidates = vec![
        "John Smith".to_string(),
        "Mary Anne".to_string(),
        "David Chen".to_string(),
        "Chen David".to_string(),
    ];

    println!("--- Context ---\n'{}'\n\n--- Candidates ---", prompt_tail);
    for c in &candidates {
        println!("- {}", c);
    }

    let results = validate_outputs(&ctx, &probe, &candidates).await?;

    println!("\n--- Validation Results ---");
    for (candidate, probability) in results {
        println!(
            "- Candidate: {:<12} | Probability: {:.4}%",
            candidate,
            probability * 100.0
        );
    }

    println!("\nTotal elapsed: {:?}", start.elapsed());
    Ok(String::new())
}
