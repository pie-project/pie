//! The batched serve2d build (GRID 1) on the fabric simulator, in process:
//! `batch_sim <compiled program dir> <ckpt dir> <prompt a> <prompt b> <new>`.
//! Two lanes on their own pages: one launch prefills both prompts, a second
//! continues both greedily `new` steps on the wafer's own argmaxes, a third
//! reads the logits of one more step; prints each lane's tokens as JSON.

use std::path::Path;

use engine_cerebras::sdk::{Platform, Simulator, Target};
use engine_cerebras::spatial::serve2d::BatchLane;
use engine_cerebras::spatial::wafer::{Design, WaferModel};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let ids = |s: &str| -> Vec<u32> { s.split(',').map(|t| t.parse().expect("token id")).collect() };
    let prompts = [ids(&args[3]), ids(&args[4])];
    let new: usize = args[5].parse().expect("count");
    let mut m = WaferModel::open(
        Path::new(&args[1]),
        Path::new(&args[2]),
        Platform::Simulator(Simulator {
            num_threads: std::env::var("PIE_SIM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(5),
            ..Simulator::new(Target::Wse3)
        }),
        None,
    )?;
    let Design::Grid(grid) = m.design() else { return Err("not a serve2d build".into()) };
    let shape = grid.paged.ok_or("not a batched build")?;
    let ps = shape.page_size as u32;
    // Lane b's pages: enough for prompt + new + 1, on rows apart from the other lane's.
    let need = (prompts[0].len().max(prompts[1].len()) + new + 2) as u32;
    let npages = need.div_ceil(ps);
    let pages: Vec<Vec<u32>> = (0..2u32).map(|b| (0..npages).map(|i| b * npages + i).collect()).collect();
    let lane = |b: usize, pos: u32, forced: Vec<u32>| BatchLane {
        pos,
        slot: pos,
        pages: pages[b].clone(),
        visible: None,
        forced,
    };
    // Prefill: the prompts must be the same length to share a launch.
    assert_eq!(prompts[0].len(), prompts[1].len());
    let n = prompts[0].len();
    let (pre, _) = m.fire_batch(&[lane(0, 0, prompts[0].clone()), lane(1, 0, prompts[1].clone())], n, false)?;
    let first: Vec<u32> = pre.iter().map(|t| *t.last().expect("a token")).collect();
    let (dec, _) = m.fire_batch(
        &[lane(0, n as u32, vec![first[0]]), lane(1, n as u32, vec![first[1]])],
        new,
        false,
    )?;
    let at = (n + new) as u32;
    let (next, logits) = m.fire_batch(
        &[lane(0, at, vec![*dec[0].last().unwrap()]), lane(1, at, vec![*dec[1].last().unwrap()])],
        1,
        true,
    )?;
    let best: Vec<usize> = logits
        .iter()
        .map(|row| (0..row.len()).max_by(|&i, &j| row[i].total_cmp(&row[j])).unwrap_or(0))
        .collect();
    let generated: Vec<Vec<u32>> = (0..2)
        .map(|b| std::iter::once(first[b]).chain(dec[b].iter().copied()).chain(next[b].iter().copied()).collect())
        .collect();
    println!("{}", serde_json::json!({"prefill_argmax": pre, "gen": generated, "logits_argmax": best}));
    Ok(())
}
