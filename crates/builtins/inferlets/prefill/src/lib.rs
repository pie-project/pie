//! Prefills token ids and exits.
//!
//! A worker runs this for a pass another worker offloaded to it. The pages
//! the run fills seal under the model's content keys and stay in this
//! worker's cache when the program exits, for the asker to take.
//!
//! Input: `{"tokens": [u32, ...]}`. Output: `{"tokens": n}`. A launch that
//! moves a handle in as `ws` gets it back holding the tokens: a checkpoint
//! no model code wrote.

use inferlet::eta::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    tokens: Vec<u32>,
}

#[inferlet::main]
async fn main(input: String) -> std::result::Result<String, String> {
    let input: Input = serde_json::from_str(&input).map_err(|e| format!("prefill input: {e}"))?;
    let n = input.tokens.len() as u32;
    let moved = WorkingSet::take("ws");
    if n == 0 {
        if let Some(ws) = moved {
            ws.give("ws")?;
        }
        return Ok(serde_json::json!({"tokens": 0}).to_string());
    }
    let tokens: Vec<i32> = input.tokens.iter().map(|&t| t as i32).collect();
    let keep = moved.is_some();
    let ws = moved.unwrap_or_default();
    let pipe = Pipeline::new();
    let toks = Channel::from(&tokens[..]);
    let indptr = Channel::from([0u32, n]);
    let done = Channel::new([1], dtype::i32);
    let landed = done;

    let fwd = ForwardPass::new();
    fwd.embed(&toks, &indptr).map_err(|e| e.to_string())?;
    fwd.state(&[&ws], StateGeometry::folding())
        .map_err(|e| e.to_string())?;
    // The run's end is awaited through one token: nothing is sampled.
    fwd.epilogue(move || {
        done.put(reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]));
    });
    fwd.submit(&pipe).map_err(|e| e.to_string())?;
    landed.take_host::<i32>().await?;
    pipe.close();
    drop(fwd);
    if keep {
        ws.give("ws")?;
    }
    Ok(serde_json::json!({"tokens": n}).to_string())
}
