//! The keep's A/B: decode throughput of a lane that dropped part of its
//! context with one keep, against the same lane unfolded with a mask that
//! hides those rows on every decode (an eviction guest's way before keeps).
//!
//! The context is `context` synthetic tokens, prefilled unfolded. The keeping
//! lane's first decode masks the dropped rows (`[drop_from, drop_to)`) and
//! keeps: drops them, commits the rest. Its later decodes state nothing. The
//! masking lane masks the same rows on every decode and folds nothing. Each
//! times `steps` decodes after the first, wall clock in the guest.

use inferlet::eta::prelude::*;
use inferlet::Result;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_context")]
    context: u32,
    #[serde(default = "default_steps")]
    steps: usize,
    /// The dropped rows; default the middle half.
    drop_from: Option<u32>,
    drop_to: Option<u32>,
    /// "keep" or "mask".
    #[serde(default = "default_mode")]
    mode: String,
}

fn default_context() -> u32 {
    4096
}

fn default_steps() -> usize {
    64
}

fn default_mode() -> String {
    "keep".into()
}

#[derive(Serialize)]
struct Output {
    mode: String,
    context: u32,
    dropped: u32,
    steps: usize,
    decode_ms: f64,
    tokens_per_s: f64,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let n = input.context;
    let keep = match input.mode.as_str() {
        "keep" => true,
        "mask" => false,
        other => return Err(format!("mode is keep or mask, not {other}")),
    };
    let drop = (
        input.drop_from.unwrap_or(n / 4),
        input.drop_to.unwrap_or(3 * n / 4),
    );
    if !(drop.0 < drop.1 && drop.1 < n) {
        return Err(format!("drop {drop:?} must lie inside the {n}-token context"));
    }
    let prompt: Vec<i32> = (0..n).map(|at| 1000 + (at * 7919 % 20000) as i32).collect();
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();
    let toks = Channel::from(prompt);
    let indptr = Channel::from(vec![0u32, n]);
    let first = Channel::new([1], dtype::i32);
    let fwd = ForwardPass::new();
    fwd.embed(&toks, &indptr)?;
    fwd.state(&[&ws], StateGeometry::default())?;
    fwd.epilogue(move || {
        first.put(reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]));
    });
    fwd.submit(&pipe).map_err(|e| format!("prefill: {e}"))?;
    let mut token = first.take_host::<i32>().await?;

    let mut began = None;
    for step in 0..=input.steps {
        if step == 1 {
            began = Some(std::time::Instant::now());
        }
        let extent = n + step as u32 + 1;
        let tok = Channel::from(vec![token]);
        let lane = Channel::from(vec![0u32, 1]);
        let out = Channel::new([1], dtype::i32);
        let fwd = ForwardPass::new();
        fwd.embed(&tok, &lane)?;
        let masks = !keep || step == 0;
        let mask = Channel::from_shaped(
            [1, extent],
            (0..extent)
                .map(|c| !(drop.0 <= c && c < drop.1))
                .collect::<Vec<bool>>(),
        );
        if masks {
            fwd.mask(&mask)?;
        }
        let kept = Channel::from_shaped(
            [1, extent],
            (0..extent)
                .map(|c| u32::from(drop.0 <= c && c < drop.1) * 2)
                .collect::<Vec<u32>>(),
        );
        let geometry = if keep && step == 0 {
            StateGeometry {
                fold: Fold::Keep(&kept),
                ..StateGeometry::default()
            }
        } else {
            StateGeometry::default()
        };
        fwd.state(&[&ws], geometry)?;
        fwd.epilogue(move || {
            out.put(reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]));
        });
        fwd.submit(&pipe).map_err(|e| format!("decode {step}: {e}"))?;
        token = out.take_host::<i32>().await?;
    }
    let decode_ms = began.map_or(0.0, |at| at.elapsed().as_secs_f64() * 1e3);
    pipe.close();
    Ok(Output {
        mode: input.mode,
        context: n,
        dropped: drop.1 - drop.0,
        steps: input.steps,
        decode_ms,
        tokens_per_s: input.steps as f64 / (decode_ms / 1e3).max(1e-9),
    })
}
