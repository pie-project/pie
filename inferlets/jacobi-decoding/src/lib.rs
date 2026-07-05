//! Demonstrates Jacobi decoding — parallel speculation via fixed-point iteration,
//! on the raw low-level WIT (keep-core), off the `Context`/`Forward` facade.
//!
//! Jacobi decoding initializes N positions with guessed tokens, then runs a
//! forward over `[anchor, guesses]` and reads the model's argmax at EVERY input
//! position (`sampler::argmax_matrix_program` — one fire, N greedy picks). The
//! longest converged prefix (`predicted[i] == guess[i]`) is accepted; the
//! unconverged suffix is rolled back with a raw KV truncate.
//!
//! Raw rollback (the retired `Context::truncate`): reduce `seq_len` by `n` and
//! `kv.free` the trailing pages that no longer hold a live token — a ~10-line
//! inline helper on the raw `KvWorkingSet` (In Gim's SDK-minimize thesis: the
//! spec-decode bookkeeping stays visible in the inferlet). Accepted tokens stay
//! materialized as working KV for the next iteration.

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
    #[serde(default = "default_window_size")]
    window_size: usize,
}

fn default_prompt() -> String { "Write a poem about the ocean.".to_string() }
fn default_max_tokens() -> usize { 256 }
fn default_window_size() -> usize { 5 }

/// Raw n-token KV rollback (the keep-core equivalent of `Context::truncate`):
/// drop the trailing `n` materialized tokens and free the page slots that no
/// longer hold a live token. `n` is clamped to `*seq_len`.
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

/// Read a `[n]`-Token output tensor as `u32` ids.
async fn read_tokens(pass: ForwardPass) -> Result<Vec<u32>> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("read: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect())
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_tokens = input.max_tokens;
    let window_size = input.window_size;

    let start = Instant::now();
    let stop_tokens = chat::stop_tokens();
    let vocab = model::output_vocab_size();

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue());

    let kv = KvWorkingSet::new();
    let page = kv.page_size();
    let mut seq_len: u32 = 0;
    let mut fresh = true;

    println!("--- Jacobi Decoding (window_size={window_size}, page_size={page}) ---");

    // The verify sampler: one fire → argmax at each of the (1 + window_size)
    // input positions. Built once (rows trace-known from the fixed window).
    let verify_rows = (1 + window_size) as u32;
    let verify = sampler::argmax_matrix_program(vocab, verify_rows)?;
    // The single-position greedy for the bootstrap first token.
    let greedy = sampler::sampler_program(sampler::SamplerSpec::Argmax, vocab)?;

    // Bootstrap: materialize the prompt KV and read the first token = argmax of
    // the last prompt position. One fire over the whole prompt.
    let mut all_generated: Vec<u32> = Vec::new();
    let first_token = {
        let n = prompt.len() as u32;
        let pass = ForwardPass::new();
        pass.fresh_generate();
        fresh = false;
        let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq_len, n, page))?;
        geometry::attach_kv_write(&pass, &kv, &geom);
        let positions: Vec<u32> = (seq_len..seq_len + n).collect();
        pass.input_tokens(&prompt, &positions);
        pass.sampler(&greedy.program, greedy.bindings(seq_len + n - 1)?);
        pass.execute();
        seq_len += n;
        let toks = read_tokens(pass).await?;
        *toks.first().ok_or("bootstrap produced no token")?
    };
    let _ = fresh;
    all_generated.push(first_token);

    let mut anchor = first_token;
    let mut window: Vec<u32> = vec![anchor; window_size];
    let mut total_accepted = 1usize; // anchor

    while total_accepted < max_tokens {
        // Verifier fire: [anchor] + window guesses at [seq_len, seq_len+n); read
        // the model's argmax at every position. All n tokens are written to KV;
        // the rejected suffix is truncated below (kept in working pages until
        // accepted — that is what makes the output window-size-independent).
        let mut input_all = vec![anchor];
        input_all.extend_from_slice(&window);
        let n = input_all.len() as u32;

        let pass = ForwardPass::new();
        let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq_len, n, page))?;
        geometry::attach_kv_write(&pass, &kv, &geom);
        let positions: Vec<u32> = (seq_len..seq_len + n).collect();
        pass.input_tokens(&input_all, &positions);
        pass.sampler(&verify.program, verify.bindings(&positions)?);
        pass.execute();
        let predicted = read_tokens(pass).await?;
        if predicted.is_empty() {
            break;
        }

        // Jacobi verification: longest converged prefix. predicted[0] is the
        // anchor's own next-token (always accepted); each later guess is
        // accepted iff the model reproduced it.
        let mut accepted_count = 1usize;
        for i in 1..predicted.len().min(window.len() + 1) {
            if i - 1 < window.len() && predicted[i - 1] == window[i - 1] {
                accepted_count += 1;
            } else {
                break;
            }
        }
        let newly_accepted: Vec<u32> = predicted[..accepted_count.min(predicted.len())].to_vec();

        // Stop-token check within the accepted prefix.
        let mut stop_at = newly_accepted.len();
        for (i, &t) in newly_accepted.iter().enumerate() {
            if stop_tokens.contains(&t) {
                stop_at = i;
                break;
            }
        }
        let final_accepted = &newly_accepted[..stop_at];
        all_generated.extend_from_slice(final_accepted);
        total_accepted += final_accepted.len();

        // Commit the accepted prefix, roll back the rest. The fire wrote all n
        // tokens at [seq_len, seq_len+n); keep the anchor + the accepted prefix
        // (`1 + final_accepted.len()`, matching the facade's `truncate`) and free
        // the unconverged suffix as working pages.
        let committed = (1 + final_accepted.len()) as u32; // anchor + accepted
        let mut new_seq = seq_len + n;
        kv_truncate(&kv, &mut new_seq, n - committed);
        seq_len = new_seq;

        if stop_at < newly_accepted.len() || total_accepted >= max_tokens {
            break;
        }

        let last_accepted = *final_accepted.last().unwrap();
        anchor = last_accepted;
        // Next window: fresh predictions past the accepted prefix, padded with
        // the last accepted token.
        window = if accepted_count < predicted.len() {
            let mut w: Vec<u32> = predicted[accepted_count..].to_vec();
            w.truncate(window_size);
            while w.len() < window_size {
                w.push(last_accepted);
            }
            w
        } else {
            vec![last_accepted; window_size]
        };
    }

    let text = model::decode(&all_generated)?;
    println!(
        "Generated {} tokens in {:?} ({:.1} tokens/s)",
        all_generated.len(),
        start.elapsed(),
        all_generated.len() as f64 / start.elapsed().as_secs_f64()
    );
    println!("Output:\n{text}");

    Ok(String::new())
}
