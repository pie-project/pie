//! Generates with attention sinks and a sliding recent-token window.
//!
//! The prompt is prefilled once with normal causal attention. During decoding,
//! each query attends to the first `sink_size` positions and the most recent
//! `window_size` positions; `sink_size: 0` is a plain sliding window.
//!
//! The rows a query no longer sees are let go with a keep (`Fold::Keep`)
//! every `keep_every` decodes: the first keep commits the sink, each drops
//! the rows that left the window and leaves the window buffered. Between
//! keeps the mask hides the rows that left the window; it speaks of the
//! buffer (column `c` is slot `folded + c`), and once the sink is committed
//! every query reads it without a bit. So the set holds the sink, the window
//! and at most `keep_every` rows past it, and a row keeps its position.
//!
//! On a model with recurrent layers too, the sink/window mask is a mask over
//! the ATTENTION layers only: a recurrence has no key it can decline to read.
//! The keep folds the record through every row it drops (the window it
//! leaves buffered is a suffix), so a fire replays the window, not the
//! history.

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
    /// Decodes between keeps.
    #[serde(default = "default_keep_every")]
    keep_every: u32,
}

fn default_keep_every() -> u32 {
    256
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
/// A benchmark client (pie-project/benchmarks' `pie_bench.py`) reads `count`; a
/// bare string is not a reply it can parse.
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
    /// Keeps stated, and the rows the set held at the end.
    keeps: u32,
    held: u32,
    /// The decodes' wall time, the first decode on.
    decode_ms: f64,
}

fn host_sink_window_mask(span: u32, query: u32, sink: u32, window: u32) -> Vec<bool> {
    (0..span)
        .map(|key| key <= query && (key < sink || key.saturating_add(window) > query))
        .collect()
}

/// The report after `fires` decode fires over an `n`-token prompt.
fn report(
    generated: &[u32],
    n: u32,
    sink: u32,
    window: u32,
    fires: u32,
    held: (u32, u32, f64),
) -> Result<Output> {
    let kv_len = n + fires;
    let visible_kv = if fires == 0 {
        n
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
        keeps: held.0,
        held: held.1,
        decode_ms: held.2,
    })
}

/// A keep's outcome for each column of a set folded at `folded` whose
/// query, the keeping fire's row, is slot `query`: the sink committed
/// (while it is buffered), the window and the query left buffered, the rest
/// dropped.
fn sink_window_keep(folded: u32, query: u32, sink: u32, window: u32) -> Vec<u32> {
    use inferlet::eta::keep::{BUFFER, COMMIT, DROP};
    (folded..=query)
        .map(|slot| {
            if slot < sink {
                COMMIT
            } else if slot + window > query {
                BUFFER
            } else {
                DROP
            }
        })
        .collect()
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let sink = input.sink_size;
    let window = input.window_size.max(1);

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    if input.max_tokens == 0 {
        return Ok(Output {
            sampler: "attention-sink",
            text: String::new(),
            count: 0,
            prompt_len: n,
            sink_size: sink,
            window_size: window,
            kv_len: 0,
            visible_kv: 0,
            keeps: 0,
            held: 0,
            decode_ms: 0.0,
        });
    }
    let stop_tokens = chat::stop_tokens();
    let span = n + input.max_tokens as u32 + 1;
    let every = input.keep_every.max(1);

    let ws = WorkingSet::new();

    let pipeline = Pipeline::new();
    let tokens = Channel::from_iter(prompt.iter().map(|&t| t as i32));
    let embed_indptr = Channel::from([0u32, n]);
    let causal = Channel::from_shaped(
        [n, span],
        (0..n)
            .flat_map(|query| (0..span).map(move |key| key <= query))
            .collect::<Vec<_>>(),
    );
    let first_out = Channel::new([1], dtype::i32);

    let prefill = ForwardPass::new();
    prefill.embed(&tokens, &embed_indptr)?;
    prefill.mask(&causal)?;
    prefill.state(&[&ws], StateGeometry::default())?;
    prefill.epilogue(move || {
        first_out.put(reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]));
    });
    prefill
        .submit(&pipeline)
        .context("attention-sink prefill")?;
    let first = first_out.take_host::<i32>().await? as u32;
    // The decodes post through pipelines of their own (a segment's, a
    // keep's), the set passing from one to the next.
    pipeline.close();

    let mut generated = Vec::with_capacity(input.max_tokens);
    if !stop_tokens.contains(&first) {
        generated.push(first);
    }
    if generated.len() >= input.max_tokens || stop_tokens.contains(&first) {
        pipeline.close();
        return report(&generated, n, sink, window, 0, (0, n, 0.0));
    }

    // The set: rows `0..end`, committed below `folded`; the next query is
    // slot `end`. The sink is committed by the first keep.
    let began = std::time::Instant::now();
    let (mut folded, mut end, mut keeps) = (0u32, n, 0u32);
    let mut token = first as i32;
    let mut fires = 0u32;
    let mut stopped = false;
    while !stopped && generated.len() < input.max_tokens {
        // A segment of decodes: the mask speaks of the buffer; the sink is
        // in it until it is committed.
        let left = (input.max_tokens - generated.len()) as u32;
        let steps = every.min(left).saturating_sub(1);
        let width = end - folded + steps + 1;
        let sink_here = sink.saturating_sub(folded);
        if steps > 0 {
            let token_in = Channel::from([token]);
            let fill = Channel::from([end - folded + 1]);
            let mask = Channel::from_shaped(
                [1, width],
                host_sink_window_mask(width, end - folded, sink_here, window),
            );
            let token_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
            let decode = ForwardPass::new();
            let decode_indptr = Channel::from([0u32, 1]);
            decode.embed(&token_in, &decode_indptr)?;
            decode.mask(&mask)?;
            decode.state(&[&ws], StateGeometry::default())?;
            decode.epilogue(move || {
                let base = fill.take();
                let token = reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]);
                let next_mask = reshape(sink_window_mask(&base, width, sink_here, window), [1, width]);
                token_in.put(&token);
                token_out.put(&token);
                fill.put(&base + 1u32);
                mask.put(&next_mask);
            });
            // `run_ahead` closes its pipeline once it has posted its budget:
            // a segment takes one of its own.
            let segment = Pipeline::new();
            let mut taken = 0u32;
            run_ahead(&segment, &decode, steps as usize, async || {
                let next = token_out.take_host::<i32>().await?;
                taken += 1;
                fires += 1;
                token = next;
                if stop_tokens.contains(&(next as u32)) {
                    stopped = true;
                    return Ok(ControlFlow::Break(()));
                }
                generated.push(next as u32);
                Ok(ControlFlow::Continue(()))
            })
            .await?;
            end += taken;
            if stopped || generated.len() >= input.max_tokens {
                break;
            }
        }
        // The keeping decode: its query is slot `end`; it lets go of what
        // left the window and commits the sink.
        let outcomes = sink_window_keep(folded, end, sink, window);
        let extent = outcomes.len() as u32;
        let keep = Channel::from_shaped([1, extent], outcomes.clone());
        let mask = Channel::from_shaped(
            [1, extent],
            host_sink_window_mask(extent, end - folded, sink_here, window),
        );
        let token_in = Channel::from([token]);
        let token_out = Channel::new([1], dtype::i32);
        let decode = ForwardPass::new();
        decode.embed(&token_in, &Channel::from([0u32, 1]))?;
        decode.mask(&mask)?;
        decode.state(
            &[&ws],
            StateGeometry {
                fold: Fold::Keep(&keep),
                ..StateGeometry::default()
            },
        )?;
        decode.epilogue(move || {
            token_out.put(reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]));
        });
        let keeping = Pipeline::new();
        decode.submit(&keeping).context("attention-sink keep")?;
        let next = token_out.take_host::<i32>().await?;
        keeping.close();
        fires += 1;
        keeps += 1;
        use inferlet::eta::keep::{COMMIT, DROP};
        let commits = outcomes.iter().filter(|&&at| at == COMMIT).count() as u32;
        let dropped = outcomes.iter().filter(|&&at| at == DROP).count() as u32;
        folded += commits;
        end = end + 1 - dropped;
        token = next;
        if stop_tokens.contains(&(next as u32)) {
            stopped = true;
        } else {
            generated.push(next as u32);
        }
    }
    let decode_ms = began.elapsed().as_secs_f64() * 1e3;
    report(&generated, n, sink, window, fires, (keeps, end, decode_ms))
}
