//! stanza-score: the log-probability pie gives each next token of a token
//! sequence, under the base model or under a trained adapter bundle. Entry
//! `i` is about predicting `tokens[i + 1]`. With `"adapter": true` the
//! inferlet sends "ready" and takes the bundle's `adapter.safetensors` bytes
//! by the chunked file transfer. One prefill pass reads out every row of a
//! span of at most `SPAN` rows, so a longer sequence takes one pass per span.

use inferlet::eta::Adapter;
use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

/// The most rows one pass reads out. The epilogue holds about eleven
/// `[rows, vocab]` f32 copies of the logits, and one fire's scratch is capped
/// at 4 GiB: 699 rows of Qwen3's 151,936-token vocabulary asked for 4.67 GB.
const SPAN: u32 = 128;

/// The request.
#[derive(Deserialize)]
struct Input {
    /// The rendered sequence to score.
    tokens: Vec<u32>,
    /// Whether a bundle follows by file transfer once the inferlet is ready.
    #[serde(default)]
    adapter: bool,
}

/// The reply; both lists have `len(tokens) - 1` entries.
#[derive(Serialize)]
struct Output {
    /// `log_softmax(logits[i])[tokens[i + 1]]`.
    logprobs: Vec<f32>,
    /// The most likely token after `tokens[..=i]`.
    argmax: Vec<i32>,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let adapter = if input.adapter {
        inferlet::session::send("ready");
        let bytes = inferlet::session::receive_file()
            .await
            .ok_or("no adapter bundle was sent")?;
        Some(Adapter::from_bundle(&bytes)?)
    } else {
        None
    };
    let mut out = Output {
        logprobs: Vec::new(),
        argmax: Vec::new(),
    };
    let rows = input.tokens.len().saturating_sub(1) as u32;
    let vocab = model::output_vocab_size();
    let max = model::lane_of(&[])?
        .outputs
        .iter()
        .find(|output| output.name == "logits")
        .map_or(1, |output| output.max_rows)
        .min(SPAN);
    let ids: Vec<i32> = input.tokens.iter().map(|&t| t as i32).collect();
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();
    // Every pass stays bound to the end, so the adapter's slot is written once.
    let mut passes = Vec::new();
    for (base, end) in readout_spans(rows, max) {
        let (base, end, len) = (base as usize, end as usize, end - base);
        let toks = Channel::from(&ids[base..end]);
        let indptr = Channel::from([0u32, len]);
        let readout = Channel::from_iter(0..len);
        let targets = Channel::from(&input.tokens[base + 1..end + 1]);
        let logprobs = Channel::new([len], dtype::f32);
        let argmax = Channel::new([len], dtype::i32);
        let fwd = ForwardPass::new();
        fwd.embed(&toks, &indptr)?;
        fwd.readout(&readout)?;
        fwd.state(&[&ws], StateGeometry::folding())?;
        if let Some(adapter) = &adapter {
            fwd.adapter(adapter)?;
        }
        fwd.epilogue(move || {
            let logits = reshape(intrinsics::output(0, "logits"), [len, vocab]);
            let flat = reshape(log_softmax(&logits), [len * vocab]);
            let at = iota(len) * Tensor::constant(vocab) + targets.read();
            logprobs.put(gather(&flat, &at));
            argmax.put(cast(reshape(reduce_argmax(&logits), [len]), dtype::i32));
        });
        fwd.submit(&pipe)?;
        out.logprobs.extend(logprobs.take_host::<Vec<f32>>().await?);
        out.argmax.extend(argmax.take_host::<Vec<i32>>().await?);
        passes.push(fwd);
    }
    pipe.close();
    Ok(out)
}
