//! A single tool-using agent, for agent traces.
//!
//! Each turn generates a short answer, then "calls" a fake tool that sleeps
//! 0.5–2 s (varied per turn and per `seed`), and feeds a short result back
//! through `tools::answer` before the next turn. The working set is kept
//! across turns, so every turn after the first prefills only the tool's
//! result on top of the context the agent already holds.
//!
//! The device's view of it is the point: prefill, decode, a tool wait with no
//! device work, again. Run it under `pie run --trace` to see that, and with
//! `--parallel 3` to see three agents share the device.

use inferlet::eta::prelude::*;
use inferlet::{chat, tools};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(default)]
struct Input {
    /// Turns: generate, call the tool, feed the result back.
    turns: usize,
    /// Most tokens one turn generates.
    max_tokens: usize,
    /// The fake tool's shortest and longest sleep.
    tool_min_ms: u64,
    tool_max_ms: u64,
    /// Varies the tool's sleeps (and the sampling) between copies.
    seed: u64,
    question: String,
    temperature: f32,
}

impl Default for Input {
    fn default() -> Self {
        Input {
            turns: 10,
            max_tokens: 80,
            tool_min_ms: 500,
            tool_max_ms: 2_000,
            seed: 0,
            question: "Plan a three-day trip to Kyoto. Use the search tool for each step, \
                       and keep every reply to two sentences."
                .into(),
            temperature: 0.7,
        }
    }
}

fn mix(mut x: u64) -> u64 {
    // splitmix64
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn sample(r: &Tensor, temperature: f32) -> Tensor {
    let logits = intrinsics::logits();
    if temperature == 0.0 {
        return reduce_argmax(&logits);
    }
    nucleus_sample(&(&logits / temperature.max(1e-4)), 0.95, r)
}

/// Prefills `tokens` onto `ws`, then decodes up to `max_tokens`; the number
/// of tokens generated.
async fn turn(
    ws: &WorkingSet,
    tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    seed: u32,
    stop: &[u32],
) -> Result<usize> {
    let pipe = Pipeline::new();
    let n = tokens.len() as u32;
    let first = {
        let toks = Channel::from(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>());
        let indptr = Channel::from([0u32, n]);
        let rng = Channel::from([seed, 0]);
        let out = Channel::new([1], dtype::i32);
        let fwd = ForwardPass::new();
        fwd.embed(&toks, &indptr)?;
        fwd.state(&[ws], StateGeometry::folding())?;
        fwd.epilogue(move || {
            let r = rng.take();
            let tok = sample(&r, temperature);
            out.put(&tok);
            rng.put(&(&r + iota(2)));
        });
        fwd.submit(&pipe).context("prefill submit")?;
        out.take_host::<i32>().await?
    };
    if stop.contains(&(first as u32)) || max_tokens <= 1 {
        pipe.close();
        return Ok(1);
    }

    let tok_in = Channel::from([first]);
    let out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
    let rng = Channel::from([seed ^ 0x9e37, 0]);
    let lane = Channel::from([0u32, 1u32]);
    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &lane)?;
    fwd.state(&[ws], StateGeometry::folding())?;
    fwd.epilogue(move || {
        let r = rng.take();
        let tok = sample(&r, temperature);
        tok_in.put(&tok);
        out.put(&tok);
        rng.put(&(&r + iota(2)));
    });
    let generated = run_ahead(&pipe, &fwd, max_tokens - 1, async || {
        let t = out.take_host::<Vec<i32>>().await?;
        let token = *t.first().unwrap_or(&0) as u32;
        Ok(if stop.contains(&token) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        })
    })
    .await?;
    Ok(1 + generated)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let stop = chat::stop_tokens();
    let ws = WorkingSet::new();

    let mut next =
        chat::system("You are a travel agent with one tool, `search`. Answer briefly.");
    next.extend(chat::user(&input.question));
    next.extend(chat::cue_without_thinking());

    let span = input.tool_max_ms.saturating_sub(input.tool_min_ms) + 1;
    let mut generated = 0usize;
    let mut waited_ms = 0u64;
    for turn_index in 0..input.turns {
        let seed = mix(input.seed.wrapping_mul(1_000).wrapping_add(turn_index as u64));
        generated += turn(
            &ws,
            &next,
            input.max_tokens,
            input.temperature,
            seed as u32,
            &stop,
        )
        .await?;

        // The tool: no device work while it runs.
        let ms = input.tool_min_ms + seed % span;
        inferlet::sleep(std::time::Duration::from_millis(ms)).await;
        waited_ms += ms;

        let result = format!(
            "{{\"results\": {}, \"took_ms\": {ms}, \"step\": {}}}",
            1 + seed % 5,
            turn_index + 1
        );
        next = tools::answer("search", &result);
        next.extend(chat::cue_without_thinking());
    }
    Ok(format!(
        "{{\"turns\": {}, \"generated_tokens\": {generated}, \"tool_ms\": {waited_ms}}}",
        input.turns
    ))
}
