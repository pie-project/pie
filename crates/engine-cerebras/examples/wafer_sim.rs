//! The spatial wafer model on the fabric simulator, in process:
//! `wafer_sim <compiled program dir> <ckpt dir> <prompt ids,..> <new>`
//! runs a compiled one-sequence program (serve2d `GRID` 0), places the
//! checkpoint, runs the prompt then greedy on the wafer, and continues the
//! cache with a second request; prints the tokens as JSON.

mod common;

use std::path::PathBuf;

use engine_cerebras::spatial::wafer::{Logits, WaferModel};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    // A compiled program's directory.
    let program = PathBuf::from(&args[1]);
    if !program.join("out.json").is_file() {
        return Err(format!("{}: not a compiled program directory", args[1]));
    }
    let prompt = common::token_ids(&args[3]);
    let new: usize = args[4].parse().expect("count");
    let started = std::time::Instant::now();
    let mut m = WaferModel::open(&program, &PathBuf::from(&args[2]), common::simulator(16))?;
    let load_s = started.elapsed().as_secs_f32();
    // Request 1: the prompt, each step's argmax; greedy by feeding them back.
    let mut fed = prompt.clone();
    let (argmax, _) = m.forward(&prompt, 0, Logits::None)?;
    let mut generated = vec![*argmax.last().ok_or("no argmax")?];
    for _ in 1..new {
        let pos = fed.len();
        let tok = *generated.last().expect("one");
        fed.push(tok);
        let (a, _) = m.forward(&[tok], pos, Logits::None)?;
        generated.push(a[0]);
    }
    // Request 2: the last token with its logits.
    let pos = fed.len();
    let (a, logits) = m.forward(&[*generated.last().expect("one")], pos, Logits::Last)?;
    // The token loop from scratch: the prompt, then the rest in one launch.
    let looped = m.greedy(&prompt, 0, new)?;
    let looped = looped[prompt.len() - 1..].to_vec();
    let row = &logits[0];
    let best = common::argmax(row);
    println!(
        "{}",
        serde_json::json!({"program": program, "gen": generated, "loop": looped, "next": a[0], "logits_argmax": best,
                           "logits_len": row.len(), "load_s": load_s})
    );
    Ok(())
}
