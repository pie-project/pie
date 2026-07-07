//! LoRA-adapted n-gram speculative decoding on the raw low-level WIT
//! (keep-core), off the `Context`/`Generator`/`Sampler` facade.
//!
//! A guest `NGram` drafter proposes candidate tokens from a learned
//! prev→next table; the main sequence verifies `[anchor] + drafts` in ONE
//! adapter-attached forward pass (`sampler::argmax_matrix_program` — argmax
//! at every position). Accepted tokens commit as working KV; the rejected
//! suffix is rolled back with the inline raw `kv_truncate` (the retired
//! `Context::truncate`). The LoRA adapter is attached on every fire via the
//! raw `ForwardPass::adapter`.

use inferlet::adapter::Adapter;
use inferlet::inference::ForwardPass;
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, sampler, Result};
use std::collections::HashMap;

/// Raw n-token KV rollback (the keep-core equivalent of `Context::truncate`):
/// drop the trailing `n` materialized tokens and free the page slots that no
/// longer hold a live token. `n` is clamped to `*seq_len`. Byte-identical to
/// the jacobi/cacheback helper.
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
        // Best-effort: a stale trailing page is harmless — the next forward
        // overwrites it. Matches the facade's non-fatal free.
        let drop: Vec<u32> = (live_pages..have).collect();
        let _ = kv.free(&drop);
    }
}

/// Read a Token output tensor as `u32` ids.
async fn read_tokens(pass: ForwardPass) -> Result<Vec<u32>> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("read: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect())
}

#[inferlet::main]
async fn main(prompt: String) -> Result<String> {
    let bytes = inferlet::http::fetch("https://example.com/loras/math-tutor.safetensors").await?;
    std::fs::write("/scratch/lora.safetensors", &bytes).map_err(|e| e.to_string())?;
    let lora = Adapter::create("math-tutor")?;
    lora.load("/scratch/lora.safetensors")?;

    let max_tokens = 512usize;
    let stop_tokens = chat::stop_tokens();
    let vocab = model::output_vocab_size();

    let mut tokens = chat::system_user("Solve the problem step by step.", &prompt);
    tokens.extend(chat::cue());

    let kv = KvWorkingSet::new();
    let page = kv.page_size();
    let mut seq_len: u32 = 0;

    let greedy = sampler::sampler_program(sampler::SamplerSpec::Argmax, vocab)?;

    // Bootstrap: materialize the prompt KV (adapter-attached) and read the
    // first token = argmax of the last prompt position.
    let first_token = {
        let n = tokens.len() as u32;
        let pass = ForwardPass::new();
        pass.fresh_generate();
        let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq_len, n, page))?;
        geometry::attach_kv_write(&pass, &kv, &geom);
        pass.adapter(&lora);
        let positions: Vec<u32> = (seq_len..seq_len + n).collect();
        pass.input_tokens(&tokens, &positions);
        pass.sampler(&greedy.program, greedy.bindings(seq_len + n - 1)?);
        pass.execute();
        seq_len += n;
        *read_tokens(pass)
            .await?
            .first()
            .ok_or("bootstrap produced no token")?
    };

    // The guest drafter learns from every accepted token; seed it with the
    // bootstrap token so its prev→next table starts growing.
    let mut drafter = NGram::new(seq_len);
    drafter.accept(&[first_token]);

    let mut all_generated: Vec<u32> = vec![first_token];
    let mut anchor = first_token;
    let mut total_accepted = 1usize;

    while total_accepted < max_tokens {
        // Step 1: draft off the guest n-gram table (pure CPU — no forward).
        let (draft_tokens, _positions) = drafter.draft();

        // Step 2: verify [anchor] + drafts in one adapter-attached fire —
        // argmax at every position. With no drafts this degrades to a plain
        // single-token decode of the anchor's next-token.
        let mut verify_input = vec![anchor];
        verify_input.extend_from_slice(&draft_tokens);
        let n = verify_input.len() as u32;
        let verify = sampler::argmax_matrix_program(vocab, n)?;

        let pass = ForwardPass::new();
        let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq_len, n, page))?;
        geometry::attach_kv_write(&pass, &kv, &geom);
        pass.adapter(&lora);
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

        // Step 4: roll back the rejected suffix — keep `accepted_count` of the
        // `n` written tokens.
        kv_truncate(&kv, &mut seq_len, n - accepted_count as u32);

        // Grow the drafter's table with the accepted tokens.
        drafter.accept(&newly_accepted);

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
    }

    Ok(model::decode(&all_generated)?)
}

struct NGram {
    cursor: u32,
    history: Vec<u32>,
    table: HashMap<u32, u32>,
}
impl NGram {
    fn new(start: u32) -> Self {
        Self {
            cursor: start,
            history: Vec::new(),
            table: HashMap::new(),
        }
    }
    fn draft(&mut self) -> (Vec<u32>, Vec<u32>) {
        let Some(&last) = self.history.last() else {
            return (Vec::new(), Vec::new());
        };
        let mut drafts = Vec::new();
        let mut probe = last;
        for _ in 0..4 {
            let Some(&t) = self.table.get(&probe) else {
                break;
            };
            drafts.push(t);
            probe = t;
        }
        let positions = (self.cursor..self.cursor + drafts.len() as u32).collect();
        (drafts, positions)
    }
    fn accept(&mut self, accepted: &[u32]) {
        for &t in accepted {
            if let Some(&prev) = self.history.last() {
                self.table.insert(prev, t);
            }
            self.history.push(t);
        }
        self.cursor += accepted.len() as u32;
    }
}
