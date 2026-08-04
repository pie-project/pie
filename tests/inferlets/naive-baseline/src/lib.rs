//! Naive text completion — the performance control for the algorithm inferlets.
//!
//! Structurally identical to every truncation/penalty inferlet in this
//! directory: one N-wide prefill fire, then a device-carried decode loop
//! driven by `ptir::run_ahead`, which keeps the engine's run-ahead window
//! (`model.channel-capacity()`) full ahead of the host drain. The only
//! difference is the epilogue, which does nothing but temperature-scale the
//! logits and draw a Gumbel-max sample. Whatever an algorithm inferlet costs
//! above this number is the algorithm.
//!
//! `stats` adds the two extra `[1]`-shaped f32 drains that every algorithm
//! inferlet carries for its self-verification metrics, without adding any
//! algorithm compute. Running with and without it separates the cost of the
//! extra host round-trip channels from the cost of the algorithm itself.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_seed")]
    seed: u32,
    /// Drain the two instrumentation channels the algorithm inferlets carry.
    #[serde(default)]
    stats: bool,
    /// STRUCTURAL v0: run only the first k transformer layers (the
    /// layerskip-draft / logit-lens class). None = the full model.
    #[serde(default)]
    max_layers: Option<u32>,
}

fn default_prompt() -> String {
    "Write a short paragraph about naive sampling.".into()
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> usize {
    32
}

fn default_seed() -> u32 {
    0x7ce1
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    stats: bool,
}

/// One sampling step: temperature, then a Gumbel-max draw over the full vocab.
fn step(logits: Tensor, temperature: f32, rng_state: impl AsTensor + Copy) -> Tensor {
    let scaled = if temperature == 1.0 {
        logits
    } else {
        div(&logits, temperature)
    };
    gumbel_max(scaled, rng_state)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }
    let max_tokens = input.max_tokens;
    let temperature = input.temperature;
    let want_stats = input.stats;
    let ws = WorkingSet::new();
    let page_size = ws.page_size();

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "naive-baseline",
            text: String::new(),
            count: 0,
            stats: want_stats,
        });
    }

    let mut prompt = wit_model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages)
        .map_err(|e| format!("reserve KV: {e}"))?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // ── PREFILL (chunked, C-wide): first sampled token comes off the prompt. ──
    //
    // Split into `ceil(n / C)` chunks, `C = max_embed_length()` -- the driver's
    // structural per-launch token capacity. A one-shot fire cannot exceed it,
    // which capped this reference at 8192 prompt tokens and so capped every
    // policy measured against it. When the prompt fits in one chunk the loop
    // runs once and builds exactly the pass this used to build.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    // The split is `prefill_chunks` (SDK), which spreads the remainder over the
    // FIRST chunks so the last one is never a sliver. Every inferlet in this
    // tree uses it, which is what makes their chunk boundaries identical: a
    // text difference above the ceiling is then a policy difference, not a
    // difference in the attention tile decomposition (§11.4).
    let spans = prefill_chunks(n, None);
    let pipe = Pipeline::new();

    let mut g0 = 0i32;
    for &(base, end) in &spans {
        let len = end - base;

        let toks_p =
            Channel::from(prompt_i32[base as usize..end as usize].to_vec()).named("toks_p");
        let embed_indptr_p = Channel::from(vec![0u32, len]).named("embed_indptr_p");
        let positions_p = Channel::from((base..end).collect::<Vec<_>>()).named("positions_p");
        let pages_p = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages_p");
        let page_indptr_p =
            Channel::from(vec![0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let w_slot_p =
            Channel::from((base..end).map(|p| p / page_size).collect::<Vec<_>>()).named("w_slot_p");
        let w_off_p =
            Channel::from((base..end).map(|p| p % page_size).collect::<Vec<_>>()).named("w_off_p");
        let kv_len_p = Channel::from(vec![end]).named("kv_len_p");
        let rng_p = Channel::from(vec![input.seed, 0]).named("rng_p");
        let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
        let s1_out_p = Channel::new([1], dtype::f32).named("s1_out_p");
        let s2_out_p = Channel::new([1], dtype::f32).named("s2_out_p");

        let fwd_p = ForwardPass::new();
        if let Some(k) = input.max_layers {
            fwd_p.set_max_layers(k)?;
        }
        fwd_p.embed(&toks_p, &embed_indptr_p)?;
        fwd_p.attention(
            &ws,
            ..,
            ..,
            &kv_len_p,
            &pages_p,
            &page_indptr_p,
            &w_slot_p,
            &w_off_p,
            &positions_p,
            None,
        )?;
        fwd_p.epilogue(move || {
            let r = rng_p.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);
            let r_next = add(&r, iota(2));
            tok_out_p.put(&token);
            if want_stats {
                let mirror = reshape(cast(&token, DType::F32), [1]);
                s1_out_p.put(&mirror);
                s2_out_p.put(&mirror);
            }
            rng_p.put(&r_next);
        });

        fwd_p
            .submit(&pipe)
            .map_err(|e| format!("prefill submit @{base}: {e}"))?;

        // Every chunk samples; only the last chunk's token continues the
        // prompt. The intermediate takes cannot be skipped -- an epilogue put
        // has to be drained or the channel fills.
        g0 = tok_out_p
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("g0 take @{base}: {e}"))?[0];
        if want_stats {
            s1_out_p
                .take()
                .get::<f32>()
                .await
                .map_err(|e| format!("s1 take @{base}: {e}"))?;
            s2_out_p
                .take()
                .get::<f32>()
                .await
                .map_err(|e| format!("s2 take @{base}: {e}"))?;
        }
    }
    generated.push(g0 as u32);

    // ── DECODE LOOP (1-wide, run-ahead). ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
        let rng = Channel::from(vec![input.seed ^ 0x5bd1, 0]).named("rng");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let s1_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("s1_out");
        let s2_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("s2_out");
        let lane1 = Channel::from(vec![0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from(vec![n]).named("positions");
        let pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages");
        let page_indptr =
            Channel::from(vec![0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from(vec![n / page_size]).named("w_slot");
        let w_off = Channel::from(vec![n % page_size]).named("w_off");
        let kv_len = Channel::from(vec![n + 1]).named("kv_len");

        let fwd = ForwardPass::new();
        if let Some(k) = input.max_layers {
            fwd.set_max_layers(k)?;
        }
        fwd.embed(&tok_in, &lane1)?;
        fwd.attention(
            &ws,
            ..,
            (n / page_size)..,
            &kv_len,
            &pages,
            &page_indptr,
            &w_slot,
            &w_off,
            &positions,
            None,
        )?;
        fwd.epilogue(move || {
            let length = kv_len.take().tensor();
            let r = rng.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);

            let r_next = add(&r, iota(2));
            let next_length = add(&length, 1u32);
            let page_count = div(add(&next_length, page_size - 1), page_size);

            tok_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(div(&length, page_size));
            w_off.put(rem(&length, page_size));
            page_indptr.take();
            page_indptr.put(mul(iota(2), broadcast(&page_count, [2])));
            tok_out.put(&token);
            if want_stats {
                let mirror = reshape(cast(&token, DType::F32), [1]);
                s1_out.put(&mirror);
                s2_out.put(&mirror);
            }
            rng.put(&r_next);
        });

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget, async || {
            let t = tok_out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("tok_out.take @{}: {e}", generated.len()))?[0];
            if want_stats {
                s1_out
                    .take()
                    .get::<f32>()
                    .await
                    .map_err(|e| format!("s1_out.take @{}: {e}", generated.len()))?;
                s2_out
                    .take()
                    .get::<f32>()
                    .await
                    .map_err(|e| format!("s2_out.take @{}: {e}", generated.len()))?;
            }
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    Ok(Output {
        sampler: "naive-baseline",
        text: wit_model::decode(&generated)?,
        count: generated.len(),
        stats: want_stats,
    })
}
