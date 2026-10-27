//! Generates with attention sinks and a sliding recent-token window.
//!
//! The prompt is prefilled once with normal causal attention. During decoding,
//! each query attends to the first `sink_size` positions and the most recent
//! `window_size` positions; `sink_size: 0` is a plain sliding window. Masked
//! KV remains allocated in this example.
//!
//! On a model with recurrent layers too, the sink/window mask is a mask over
//! the ATTENTION layers only, and the recurrent layers fold every token as
//! usual. There is nothing to emulate there: a recurrence has no key it can
//! decline to read.

use inferlet::chat;
use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_sink_size")]
    sink_size: u32,
    #[serde(default = "default_window_size")]
    window_size: u32,
}

fn default_prompt() -> String {
    "Tell me a long story about a cat.".into()
}

fn default_max_tokens() -> usize {
    512
}

fn default_sink_size() -> u32 {
    4
}

fn default_window_size() -> u32 {
    64
}

/// The report every KV-policy program here answers with (`tova-attention`,
/// `snapkv-eviction`, ...): the text, the counts, and what the policy did.
/// `scripts/bench/pie_bench.py` reads `count`; a bare string is not a reply
/// it can parse.
#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    /// The continuation, greedily decoded.
    text: String,
    /// How many tokens it is.
    count: usize,
    /// Prompt tokens, chat-templated.
    prompt_len: u32,
    /// Leading positions every decode query may see.
    sink_size: u32,
    /// Recent positions a decode query may see (the input, floored at 1).
    window_size: u32,
    /// KV positions live at the last fire: the prompt plus every token fed
    /// back.
    kv_len: u32,
    /// Keys the last fire's query could see, counted from the same rule that
    /// built the first decode row (the rows after it evolve on the device).
    /// The prefill is causal and sees the whole prompt; a decode row sees at
    /// most `sink_size + window_size`.
    visible_kv: u32,
}

fn host_sink_window_mask(span: u32, query: u32, sink: u32, window: u32) -> Vec<bool> {
    (0..span)
        .map(|key| key <= query && (key < sink || key.saturating_add(window) > query))
        .collect()
}

/// The report after `fires` decode fires over an `n`-token prompt.
fn report(generated: &[u32], n: u32, sink: u32, window: u32, fires: u32) -> Result<Output> {
    let kv_len = n - 1 + fires;
    let visible_kv = if fires <= 1 {
        kv_len
    } else {
        host_sink_window_mask(kv_len, kv_len - 1, sink, window)
            .iter()
            .filter(|visible| **visible)
            .count() as u32
    };
    Ok(Output {
        sampler: "attention-sink",
        text: model::decode(generated)?,
        count: generated.len(),
        prompt_len: n,
        sink_size: sink,
        window_size: window,
        kv_len,
        visible_kv,
    })
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let sink = input.sink_size;
    let window = input.window_size.max(1);

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue());
    let n = prompt.len() as u32;
    let (&last, head) = prompt.split_last().ok_or("empty prompt")?;
    let stop_tokens = chat::stop_tokens();
    let span = n + input.max_tokens as u32 + 1;

    let ws = WorkingSet::new();
    let pipeline = Pipeline::new();
    if !head.is_empty() {
        let h = head.len() as u32;
        let causal = Channel::from_shaped(
            [h, span],
            (0..h)
                .flat_map(|query| (0..span).map(move |key| key <= query))
                .collect::<Vec<_>>(),
        );
        let prefill = ForwardPass::new();
        prefill.embed(
            &Channel::from_iter(head.iter().map(|&t| t as i32)),
            &Channel::from([0u32, h]),
        )?;
        prefill.mask(&causal)?;
        prefill.state(&[&ws], StateGeometry::folding())?;
        prefill
            .submit(&pipeline)
            .context("attention-sink prefill")?;
    }

    // The prompt's last row is causal like the rest of the prompt.
    let token_in = Channel::from([last as i32]);
    let fill = Channel::from([n]);
    let mask = Channel::from_shaped([1, span], (0..span).map(|key| key < n).collect::<Vec<_>>());
    let token_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);

    let decode = ForwardPass::new();
    decode.embed(&token_in, &Channel::from([0u32, 1]))?;
    decode.mask(&mask)?;
    decode.state(&[&ws], StateGeometry::folding())?;
    decode.epilogue(move || {
        let base = fill.take();
        let token = reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]);
        let next_mask = reshape(sink_window_mask(&base, span, sink, window), [1, span]);

        token_in.put(&token);
        token_out.put(&token);
        fill.put(&base + 1u32);
        mask.put(&next_mask);
    });

    let mut generated = Vec::with_capacity(input.max_tokens);
    // Fires whose token came back — a stop token's fire wrote its KV
    // position like any other, so it counts toward `kv_len`.
    let mut fires = 0u32;
    run_ahead(&pipeline, &decode, input.max_tokens, async || {
        let token = token_out.take_host::<i32>().await? as u32;
        fires += 1;
        if stop_tokens.contains(&token) {
            return Ok(ControlFlow::Break(()));
        }
        generated.push(token);
        Ok(ControlFlow::Continue(()))
    })
    .await?;
    // Any fire still in flight after an early stop is left untaken; close
    // releases the scheduler wait-set and reclaims them.
    pipeline.close();
    report(&generated, n, sink, window, fires)
}
