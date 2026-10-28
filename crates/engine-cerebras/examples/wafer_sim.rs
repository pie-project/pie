//! The spatial wafer model on the fabric simulator, in process:
//! `wafer_sim <serve28 params> <ckpt dir> <prompt ids,..> <new>` compiles
//! (or reuses from the cache) the serve28 program for `params`, places the
//! checkpoint, runs the prompt then greedy on the wafer, and continues the
//! cache with a second request; prints the tokens as JSON.

use std::path::PathBuf;

use engine_cerebras::sdk::{Platform, Simulator, Target};
use engine_cerebras::spatial::wafer::{Logits, WaferModel, compile};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let program = compile(&args[1], Target::Wse3, false)?;
    let prompt: Vec<u32> = args[3].split(',').map(|t| t.parse().expect("token id")).collect();
    let new: usize = args[4].parse().expect("count");
    let started = std::time::Instant::now();
    let mut m = WaferModel::open(
        &program,
        &PathBuf::from(&args[2]),
        Platform::Simulator(Simulator::new(Target::Wse3)),
        None,
    )?;
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
    let best = (0..row.len()).max_by(|&i, &j| row[i].total_cmp(&row[j])).unwrap_or(0);
    println!(
        "{}",
        serde_json::json!({"program": program, "gen": generated, "loop": looped, "next": a[0], "logits_argmax": best,
                           "logits_len": row.len(), "load_s": load_s})
    );
    Ok(())
}
