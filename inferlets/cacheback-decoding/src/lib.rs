//! CacheBack speculative decoding on the raw low-level WIT (keep-core), off the
//! `Context`/`Forward` facade.
//!
//! A separate lightweight "drafter" (its own `KvWorkingSet`) greedily proposes
//! candidate tokens; the main sequence verifies them in ONE forward pass
//! (`sampler::argmax_matrix_program` — argmax at every `[anchor, drafts]`
//! position). Accepted tokens commit as working KV; the rejected suffix is rolled
//! back from BOTH sequences via the inline raw `kv_truncate` (the retired
//! `Context::truncate`: `seq_len -= n` + `kv.free` the trailing empty pages).

use inferlet::inference::ForwardPass;
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, sampler, Result};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_draft_length")]
    draft_length: usize,
}

fn default_prompt() -> String { "Explain quantum computing.".to_string() }
fn default_max_tokens() -> usize { 256 }
fn default_draft_length() -> usize { 4 }

/// Raw n-token KV rollback (the keep-core equivalent of `Context::truncate`):
/// drop the trailing `n` materialized tokens + free the pages that no longer
/// hold a live token. `n` is clamped to `*seq_len`.
fn kv_truncate(kv: &KvWorkingSet, seq_len: &mut u32, n: u32) {
    let n = n.min(*seq_len);
    if n == 0 {
        return;
    }
    *seq_len -= n;
    let page = kv.page_size();
    let live_pages = seq_len.div_ceil(page);
    let have = kv.size();
    if have > live_pages {
        let drop: Vec<u32> = (live_pages..have).collect();
        let _ = kv.free(&drop);
    }
}

/// Read a `[n]`-Token output tensor as `u32` ids.
async fn read_tokens(pass: ForwardPass) -> Result<Vec<u32>> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("read: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect())
}

/// A lightweight greedy drafter on its own `KvWorkingSet` — accumulates tokens
/// across `draft` calls; `rollback` truncates the rejected suffix.
struct GreedyDrafter {
    kv: KvWorkingSet,
    seq_len: u32,
    fresh: bool,
}

impl GreedyDrafter {
    fn new() -> Self {
        Self { kv: KvWorkingSet::new(), seq_len: 0, fresh: true }
    }

    /// Generate `draft_length` greedy tokens starting from `seed`. Returns
    /// whatever completes before any fire fails.
    async fn draft(
        &mut self,
        seed: u32,
        draft_length: usize,
        greedy: &sampler::LoweredSampler,
    ) -> Result<Vec<u32>> {
        let page = self.kv.page_size();
        let mut tokens = Vec::with_capacity(draft_length);
        let mut current = seed;
        for _ in 0..draft_length {
            let pass = ForwardPass::new();
            if self.fresh {
                pass.fresh_generate();
                self.fresh = false;
            }
            let geom =
                geometry::ensure_pages(&self.kv, geometry::kv_write_geometry(self.seq_len, 1, page))?;
            geometry::attach_kv_write(&pass, &self.kv, &geom);
            pass.input_tokens(&[current], &[self.seq_len]);
            pass.sampler(&greedy.program, greedy.bindings(self.seq_len)?);
            pass.execute();
            self.seq_len += 1;
            match read_tokens(pass).await?.first() {
                Some(&t) => {
                    current = t;
                    tokens.push(t);
                }
                None => break,
            }
        }
        Ok(tokens)
    }

    fn rollback(&mut self, n: u32) {
        kv_truncate(&self.kv, &mut self.seq_len, n);
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_tokens = input.max_tokens;
    let draft_length = input.draft_length;

    let start = Instant::now();
    let stop_tokens = chat::stop_tokens();
    let vocab = model::output_vocab_size();

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue());

    let kv = KvWorkingSet::new();
    let page = kv.page_size();
    let mut seq_len: u32 = 0;

    let greedy = sampler::sampler_program(sampler::SamplerSpec::Argmax, vocab)?;

    // Bootstrap: materialize the prompt KV and read the first token = argmax of
    // the last prompt position.
    let first_token = {
        let n = prompt.len() as u32;
        let pass = ForwardPass::new();
        pass.fresh_generate();
        let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq_len, n, page))?;
        geometry::attach_kv_write(&pass, &kv, &geom);
        let positions: Vec<u32> = (seq_len..seq_len + n).collect();
        pass.input_tokens(&prompt, &positions);
        pass.sampler(&greedy.program, greedy.bindings(seq_len + n - 1)?);
        pass.execute();
        seq_len += n;
        *read_tokens(pass).await?.first().ok_or("bootstrap produced no token")?
    };

    let mut drafter = GreedyDrafter::new();
    let mut all_generated: Vec<u32> = vec![first_token];
    let mut anchor = first_token;
    let mut total_accepted = 1usize;
    let mut total_steps = 0usize;

    while total_accepted < max_tokens {
        // Step 1: draft off the secondary sequence.
        let draft_tokens = drafter.draft(anchor, draft_length, &greedy).await?;
        if draft_tokens.is_empty() {
            break;
        }

        // Step 2: verify [anchor] + drafts in one fire — argmax at every position.
        let mut verify_input = vec![anchor];
        verify_input.extend_from_slice(&draft_tokens);
        let n = verify_input.len() as u32;
        let verify = sampler::argmax_matrix_program(vocab, n)?;

        let pass = ForwardPass::new();
        let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq_len, n, page))?;
        geometry::attach_kv_write(&pass, &kv, &geom);
        let positions: Vec<u32> = (seq_len..seq_len + n).collect();
        pass.input_tokens(&verify_input, &positions);
        pass.sampler(&verify.program, verify.bindings(&positions)?);
        pass.execute();
        let verified = read_tokens(pass).await?;
        if verified.is_empty() {
            break;
        }

        // Step 3: accepted = anchor's own prediction + each matching draft.
        let mut accepted_count = 1usize;
        for i in 1..verified.len().min(draft_tokens.len() + 1) {
            if i - 1 < draft_tokens.len() && verified[i - 1] == draft_tokens[i - 1] {
                accepted_count += 1;
            } else {
                break;
            }
        }
        let newly_accepted: Vec<u32> = verified[..accepted_count.min(verified.len())].to_vec();

        // Step 4: roll back the rejected suffix from the main sequence — keep
        // `accepted_count` of the `n` written tokens.
        kv_truncate(&kv, &mut seq_len, n - accepted_count as u32);

        // Roll back the drafter too: it wrote `draft_length`; `accepted_count-1`
        // drafts were accepted.
        let drafter_rejected = draft_length as u32 - (accepted_count.saturating_sub(1) as u32);
        drafter.rollback(drafter_rejected);

        // Stop on the first stop token.
        let mut hit_stop = false;
        for &t in &newly_accepted {
            if stop_tokens.contains(&t) {
                hit_stop = true;
                break;
            }
            all_generated.push(t);
            total_accepted += 1;
        }
        if hit_stop || total_accepted >= max_tokens {
            break;
        }

        anchor = *newly_accepted.last().unwrap_or(&anchor);
        total_steps += 1;
    }

    let text = model::decode(&all_generated)?;
    println!("--- CacheBack Decoding (draft_length={draft_length}, steps={total_steps}) ---");
    println!("Generated in {:?}", start.elapsed());
    println!("Output:\n{text}");

    Ok(String::new())
}
