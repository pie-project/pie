//! Prefills token ids and exits.
//!
//! A worker runs this for a pass another worker offloaded to it. The pages
//! the run fills seal under the model's content keys and stay in this
//! worker's cache when the program exits, for the asker to take.
//!
//! Input: `{"tokens": [u32, ...]}`. Output: `{"tokens": n, "cached": k}`,
//! `k` of the `n` found already cached here.

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
    if n == 0 {
        return Ok(serde_json::json!({"tokens": 0, "cached": 0}).to_string());
    }
    let tokens: Vec<i32> = input.tokens.iter().map(|&t| t as i32).collect();
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();
    let toks = Channel::from(&tokens[..]).named("toks");
    let indptr = Channel::from([0u32, n]).named("embed_indptr");
    let done = Channel::new([1], dtype::i32).named("done");
    let landed = done;

    let fwd = ForwardPass::new();
    fwd.embed(&toks, &indptr).map_err(|e| e.to_string())?;
    fwd.state(&[&ws], StateGeometry::folding())
        .map_err(|e| e.to_string())?;
    // The run's end is awaited through one token: nothing is sampled.
    fwd.epilogue(move || {
        done.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
    });
    fwd.submit(&pipe).map_err(|e| e.to_string())?;
    landed.take_host::<i32>().await?;
    let cached = ws.cached_tokens();
    pipe.close();
    Ok(serde_json::json!({"tokens": n, "cached": cached}).to_string())
}
