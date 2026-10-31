//! What the simulator drivers share: the simulator platform, opening a
//! compiled program on it, argument parsing and the batched build's shape.

#![allow(dead_code, reason = "each driver uses its own part of this module")]

use std::path::Path;

use engine_cerebras::sdk::{Platform, Rect, Simulator, Target};
use engine_cerebras::spatial::paged::Shape;
use engine_cerebras::spatial::wafer::WaferModel;

/// The fabric simulator on WSE-3, with `PIE_SIM_THREADS` worker threads
/// (else `default_threads`).
pub fn simulator(default_threads: u32) -> Platform {
    Platform::Simulator(Simulator {
        num_threads: std::env::var("PIE_SIM_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_threads),
        ..Simulator::new(Target::Wse3)
    })
}

/// The compiled program at `program` on the simulator (5 threads unless
/// `PIE_SIM_THREADS` says otherwise), `ckpt` placed.
pub fn open(program: &Path, ckpt: &Path) -> Result<WaferModel, String> {
    WaferModel::open(program, ckpt, simulator(5))
}

/// A batched serve2d-family build's launch lanes and KV pool.
pub fn batched(m: &WaferModel) -> Result<(usize, Shape), String> {
    let grid = m.design();
    let shape = grid.paged().ok_or("not a batched build")?;
    Ok((grid.launch_lanes(), shape))
}

/// Token ids as `1,2,3`.
pub fn token_ids(list: &str) -> Vec<u32> {
    list.split(',')
        .map(|t| t.parse().expect("a token id"))
        .collect()
}

/// A JSON file.
pub fn read_json(path: impl AsRef<Path>) -> Result<serde_json::Value, String> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// A logits row's argmax.
pub fn argmax(row: &[f32]) -> usize {
    (0..row.len())
        .max_by(|&i, &j| row[i].total_cmp(&row[j]))
        .unwrap_or(0)
}

/// A rectangle of PEs.
pub fn rect(x: usize, y: usize, w: usize, h: usize) -> Rect {
    Rect {
        x: x as i32,
        y: y as i32,
        w: w as i32,
        h: h as i32,
    }
}
