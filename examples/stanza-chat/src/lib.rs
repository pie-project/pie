//! stanza-chat: greedy generation from token ids, under the base model or
//! under a trained adapter bundle. The client renders the prompt, so the chat
//! template is never applied here. With `"adapter": true` the inferlet sends
//! "ready" and takes the bundle's `adapter.safetensors` bytes by the chunked
//! file transfer, which carries bundles of any size.

use inferlet::eta::Adapter;
use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

/// The request.
#[derive(Deserialize)]
struct Input {
    /// The rendered prompt.
    tokens: Vec<u32>,
    /// The most tokens to generate.
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Tokens that end the reply; the one that ends it is not returned.
    #[serde(default)]
    stop: Vec<u32>,
    /// Whether a bundle follows by file transfer once the inferlet is ready.
    #[serde(default)]
    adapter: bool,
}

fn default_max_tokens() -> usize {
    128
}

/// The reply.
#[derive(Serialize)]
struct Output {
    /// The generated tokens, greedy, without the prompt.
    tokens: Vec<u32>,
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
    let mut tokens = Vec::new();
    if input.tokens.is_empty() || input.max_tokens == 0 {
        return Ok(Output { tokens });
    }
    let prompt: Vec<i32> = input.tokens.iter().map(|&t| t as i32).collect();
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();

    let toks = Channel::from(&prompt[..]);
    let indptr = Channel::from([0u32, prompt.len() as u32]);
    let first = Channel::new([1], dtype::i32);
    let fwd = ForwardPass::new();
    fwd.embed(&toks, &indptr)?;
    fwd.state(&[&ws], StateGeometry::folding())?;
    if let Some(adapter) = &adapter {
        fwd.adapter(adapter)?;
    }
    fwd.epilogue(move || {
        first.put(reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]));
    });
    fwd.submit(&pipe)?;
    let token = first.take_host::<i32>().await? as u32;

    if !input.stop.contains(&token) {
        tokens.push(token);
    }
    if tokens.len() == 1 && input.max_tokens > 1 {
        let tok_in = Channel::from([token as i32]);
        let tok_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
        let indptr = Channel::from([0u32, 1]);
        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &indptr)?;
        fwd.state(&[&ws], StateGeometry::folding())?;
        if let Some(adapter) = &adapter {
            fwd.adapter(adapter)?;
        }
        fwd.epilogue(move || {
            let token = reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]);
            tok_in.put(&token);
            tok_out.put(&token);
        });
        run_ahead(&pipe, &fwd, input.max_tokens - 1, async || {
            let token = tok_out.take_host::<i32>().await? as u32;
            if input.stop.contains(&token) {
                return Ok(ControlFlow::Break(()));
            }
            tokens.push(token);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();
    Ok(Output { tokens })
}
