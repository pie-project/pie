//! Naive text completion — the performance control for the algorithm inferlets.
//!
//! Structurally identical to every truncation/penalty inferlet in this
//! directory: one N-wide prefill fire, then a device-carried decode loop
//! driven by `eta::run_ahead`, which keeps the runtime's run-ahead window
//! (`model.channel-capacity()`) full ahead of the host drain. The only
//! difference is the epilogue, which does nothing but temperature-scale the
//! logits and draw a Gumbel-max sample. Whatever an algorithm inferlet costs
//! above this number is the algorithm.
//!
//! `stats` adds the two extra `[1]`-shaped f32 drains that every algorithm
//! inferlet carries for its self-verification metrics, without adding any
//! algorithm compute. Running with and without it separates the cost of the
//! extra host round-trip channels from the cost of the algorithm itself.

use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_seed")]
    seed: u32,
    /// Drain the two instrumentation channels the algorithm inferlets carry.
    #[serde(default)]
    stats: bool,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_stream_every")]
    stream_every: usize,
    #[serde(default)]
    park_every: usize,
    #[serde(default)]
    park_ms: u64,
    #[serde(default)]
    use_park: bool,
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

fn default_stream_every() -> usize {
    1
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    /// The generated ids, so an identity gate can find WHERE two runs part.
    tokens: Vec<u32>,
    count: usize,
    stats: bool,
}

/// One sampling step: temperature, then a Gumbel-max draw over the full vocab.
fn step(logits: Tensor, temperature: f32, rng_state: &Tensor) -> Tensor {
    let scaled = if temperature == 1.0 {
        logits
    } else {
        &logits / temperature
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

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "naive-baseline",
            text: String::new(),
            tokens: Vec::new(),
            count: 0,
            stats: want_stats,
        });
    }

    let mut prompt = match &input.prompt_tokens {
        Some(pt) => pt.clone(),
        None => {
            let mut p = inferlet::chat::prefix();
            p.extend(model::encode(&input.prompt));
            p
        }
    };
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // The runtime cuts every program's prompt at the same rows (`PIE_FIRE_ROWS` pins them).
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let pipe = Pipeline::new();

    let toks_p = Channel::from(&prompt_i32[..]);
    let embed_indptr_p = Channel::from([0u32, n]);
    let rng_p = Channel::from([input.seed, 0]);
    let tok_out_p = Channel::new([1], dtype::i32);
    let s1_out_p = Channel::new([1], dtype::f32);
    let s2_out_p = Channel::new([1], dtype::f32);

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    fwd_p.state(&[&ws], StateGeometry::folding())?;
    fwd_p.epilogue(move || {
        let r = rng_p.take();
        let logits = intrinsics::output(0, "logits");
        let token = step(logits, temperature, &r);
        let r_next = &r + iota(2);
        tok_out_p.put(&token);
        if want_stats {
            let mirror = reshape(cast(&token, dtype::f32), [1]);
            s1_out_p.put(&mirror);
            s2_out_p.put(&mirror);
        }
        rng_p.put(&r_next);
    });

    fwd_p.submit(&pipe).context("prefill submit")?;

    let g0 = tok_out_p
        .take_host::<i32>()
        .await
        .context("prefill drain")?;
    if want_stats {
        s1_out_p
            .take_host::<Vec<f32>>()
            .await
            .context("prefill drain")?;
        s2_out_p
            .take_host::<Vec<f32>>()
            .await
            .context("prefill drain")?;
    }
    if input.stream {
        inferlet::session::send("t:1");
    }
    generated.push(g0 as u32);

    // ── DECODE LOOP (1-wide, run-ahead). ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from([g0]);
        let rng = Channel::from([input.seed ^ 0x5bd1, 0]);
        let tok_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
        let s1_out = Channel::new([1], dtype::f32).capacity(channel_capacity() as u32);
        let s2_out = Channel::new([1], dtype::f32).capacity(channel_capacity() as u32);
        let lane1 = Channel::from([0u32, 1u32]);

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        fwd.state(&[&ws], StateGeometry::folding())?;
        fwd.epilogue(move || {
            let r = rng.take();
            let logits = intrinsics::output(0, "logits");
            let token = step(logits, temperature, &r);

            let r_next = &r + iota(2);

            tok_in.put(&token);
            tok_out.put(&token);
            if want_stats {
                let mirror = reshape(cast(&token, dtype::f32), [1]);
                s1_out.put(&mirror);
                s2_out.put(&mirror);
            }
            rng.put(&r_next);
        });

        let budget = max_tokens - 1;
        if input.park_every > 0 && input.park_ms > 0 {
            let r = frame_size().max(1);
            let window_fires = inferlet::model::run_ahead_window() as usize;
            let window_frames = (window_fires / r).max(1);
            let mut total_done = 0usize;
            while total_done < budget {
                let chunk = input.park_every.min(budget - total_done);
                let mut sub = 0usize;
                let mut con = 0usize;
                let submit_one = |sub: &mut usize| -> std::result::Result<(), String> {
                    let live = r.min(chunk - *sub);
                    if live == 0 {
                        return Ok(());
                    }
                    let slots: Vec<Option<&ForwardPass>> = vec![Some(&fwd); live];
                    submit_frame(&pipe, &slots)?;
                    *sub += live;
                    Ok(())
                };
                for _ in 0..window_frames {
                    if sub >= chunk {
                        break;
                    }
                    submit_one(&mut sub)?;
                }
                while con < sub {
                    let t = tok_out
                        .take_host::<i32>()
                        .await
                        .with_context(|| format!("@{}", generated.len()))?;
                    if want_stats {
                        s1_out.take_host::<Vec<f32>>().await?;
                        s2_out.take_host::<Vec<f32>>().await?;
                    }
                    if input.stream {
                        inferlet::session::send("t:1");
                    }
                    generated.push(t as u32);
                    con += 1;
                    if sub < chunk && sub - con <= (window_frames - 1) * r {
                        submit_one(&mut sub)?;
                    }
                }
                total_done += chunk;
                if total_done < budget {
                    if input.use_park {
                        pipe.park();
                    }
                    inferlet::sleep(std::time::Duration::from_millis(input.park_ms)).await;
                }
            }
        } else {
            let step_emit = input.stream_every.max(1);
            let mut stream_acc = 0usize;
            run_ahead(&pipe, &fwd, budget, async || {
                let t = tok_out
                    .take_host::<i32>()
                    .await
                    .with_context(|| format!("@{}", generated.len()))?;
                if want_stats {
                    s1_out.take_host::<Vec<f32>>().await?;
                    s2_out.take_host::<Vec<f32>>().await?;
                }
                if input.stream {
                    stream_acc += 1;
                    if stream_acc >= step_emit {
                        inferlet::session::send(&format!("t:{stream_acc}"));
                        stream_acc = 0;
                    }
                }
                generated.push(t as u32);
                Ok(ControlFlow::Continue(()))
            })
            .await?;
            if input.stream && stream_acc > 0 {
                inferlet::session::send(&format!("t:{stream_acc}"));
            }
        }
    }
    pipe.close();

    Ok(Output {
        sampler: "naive-baseline",
        text: if input.stream {
            String::new()
        } else {
            model::decode(&generated)?
        },
        count: generated.len(),
        tokens: if input.stream { Vec::new() } else { generated },
        stats: want_stats,
    })
}
