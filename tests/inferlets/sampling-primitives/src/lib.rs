//! Reads the core sampling measurements from one forward pass.
//!
//! A single PTIR epilogue publishes the greedy token, raw logits, entropy,
//! probabilities, log-probabilities, and a top-p nucleus keep-mask. The host
//! then cross-checks that the independently useful outputs agree.
//!
//! The keep-mask is deliberately published as a standalone value rather than
//! consumed by a `select`/`gumbel`/`argmax` tail, so it CANNOT match
//! `LibraryOp::NucleusSample` (compiler.rs:1305) and must be lowered through
//! the generated `pivot_threshold` implementation. That path had no coverage:
//! its `cummass_le` arm ran the M1 reference on thread 0 alone, an O(len^3)
//! selection sort that never returned at a 151936-token vocabulary and hung
//! the GPU. The host assertions below pin the exact nucleus contract
//! (interp.rs `Predicate::CummassLe`) without depending on float summation
//! order.

use inferlet::ptir::attention::prelude::*;
use serde::Deserialize;

const TOP_P: f32 = 0.9;

#[derive(Deserialize, Default)]
struct Input {}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[inferlet::main]
async fn main(_input: Input) -> Result<String> {
    let vocab = model::output_vocab_size();
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    let mut prompt = model::encode("The capital of France is");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = n.div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;

    let toks = Channel::from_iter(prompt.iter().map(|&token| token as i32));
    let embed_indptr = Channel::from([0u32, n]).named("embed_indptr");
    let positions = Channel::from_iter(0..n).named("positions");
    let pages = Channel::from_iter(0..max_pages).named("pages");
    let page_indptr = Channel::from([0u32, max_pages]).named("page_indptr");
    let w_slot = Channel::from_iter((0..n).map(|p| p / page_size)).named("w_slot");
    let w_off = Channel::from_iter((0..n).map(|p| p % page_size)).named("w_off");
    let kv_len = Channel::from([n]).named("kv_len");
    let token_out = Channel::new([1], dtype::i32).named("token");
    let logits_out = Channel::new([vocab], dtype::f32).named("logits");
    let entropy_out = Channel::new([1], dtype::f32).named("entropy");
    let probs_out = Channel::new([vocab], dtype::f32).named("probabilities");
    let logprobs_out = Channel::new([vocab], dtype::f32).named("log_probabilities");
    let keep_out = Channel::new([vocab], dtype::i32).named("nucleus_keep");

    let fwd = ForwardPass::new();
    fwd.embed(&toks, &embed_indptr)?;
    fwd.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &kv_len,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &positions,
            mask: None,
        },
    )?;
    fwd.epilogue(move || {
        let logits = intrinsics::logits();
        let logprobs = log_softmax(&logits);
        let probabilities = exp(&logprobs);
        let entropy = reshape(entropy_from_logprobs(&probabilities, &logprobs), [1]);
        let token = reshape(reduce_argmax(&logits), [1]);
        // Generated-path `pivot_threshold`: no gumbel/argmax tail follows, so
        // the library nucleus pattern cannot claim it.
        let keep = pivot_threshold(&probabilities, cummass_le(TOP_P));
        let keep_mask = reshape(cast(&keep, dtype::i32), [vocab]);

        keep_out.put(&keep_mask);
        token_out.put(&token);
        logits_out.put(&logits);
        entropy_out.put(&entropy);
        probs_out.put(&probabilities);
        logprobs_out.put(&logprobs);
    });

    let pipeline = Pipeline::new();
    fwd.submit(&pipeline)
        .context("sampling-primitives submit")?;

    let token = token_out.take_host::<i32>().await? as usize;
    let logits = logits_out.take_host::<Vec<f32>>().await?;
    let entropy = entropy_out.take_host::<f32>().await?;
    let probabilities = probs_out.take_host::<Vec<f32>>().await?;
    let log_probabilities = logprobs_out.take_host::<Vec<f32>>().await?;
    let keep_mask = keep_out.take_host::<Vec<i32>>().await?;
    pipeline.close();

    if token != argmax(&logits) || token != argmax(&probabilities) {
        return Err("token, logits, and probability argmax values disagree".into());
    }
    if probabilities.len() != vocab as usize || log_probabilities.len() != vocab as usize {
        return Err("sampling output has the wrong vocabulary dimension".into());
    }
    if !entropy.is_finite() || entropy < 0.0 || entropy > (vocab as f32).ln() + 1e-3 {
        return Err(format!("entropy is outside the valid range: {entropy}"));
    }
    let token_logprob = log_probabilities[token];
    if (token_logprob.exp() - probabilities[token]).abs() > 1e-4 {
        return Err("probability and log-probability outputs disagree".into());
    }

    let kept: Vec<usize> = keep_mask
        .iter()
        .enumerate()
        .filter(|&(_, &m)| m != 0)
        .map(|(index, _)| index)
        .collect();
    if keep_mask.len() != vocab as usize {
        return Err("nucleus keep-mask has the wrong vocabulary dimension".into());
    }
    if kept.is_empty() {
        return Err("nucleus keep-mask is empty".into());
    }
    if keep_mask.iter().any(|&m| m != 0 && m != 1) {
        return Err("nucleus keep-mask is not boolean-valued".into());
    }
    // The nucleus is exactly a descending prefix, so the smallest kept
    // probability must dominate every dropped one.
    let min_kept = kept
        .iter()
        .map(|&i| probabilities[i])
        .fold(f32::INFINITY, f32::min);
    let max_dropped = (0..vocab as usize)
        .filter(|i| keep_mask[*i] == 0)
        .map(|i| probabilities[i])
        .fold(f32::NEG_INFINITY, f32::max);
    if max_dropped > min_kept {
        return Err(format!(
            "nucleus keep-mask is not a descending prefix: dropped {max_dropped:.9} > kept {min_kept:.9}"
        ));
    }
    // `cummass_le` keeps the maximal prefix whose EXCLUSIVE mass stays below p
    // (interp.rs), so the kept mass must reach p while the mass excluding the
    // last-admitted element must not.
    let kept_mass: f32 = kept.iter().map(|&i| probabilities[i]).sum();
    let tolerance = 1e-4_f32;
    if kept.len() < vocab as usize && kept_mass + tolerance < TOP_P {
        return Err(format!(
            "nucleus keep-mask retains only {kept_mass:.6} mass, below top_p={TOP_P}"
        ));
    }
    if kept.len() > 1 && kept_mass - min_kept - tolerance > TOP_P {
        return Err(format!(
            "nucleus keep-mask is not minimal: {:.6} already exceeds top_p={TOP_P}",
            kept_mass - min_kept
        ));
    }

    Ok(format!(
        "token={token} probability={:.6} log_probability={token_logprob:.6} entropy={entropy:.6} nucleus_kept={} nucleus_mass={kept_mass:.6}",
        probabilities[token],
        kept.len()
    ))
}
