//! The pipelined serve2d build (`serve2d_pipe`, MMAX micro-batches round the
//! ring) on the fabric simulator, in process:
//! `pipe_sim <compiled program dir> <ckpt dir> <laps1> <laps2>`.
//! MMAX x B lanes, each its own prompt and pages: one launch of `laps1`
//! laps forces the prompts, a second goes on `laps2` laps from there and
//! reads the logits; prints each lane's argmaxes as JSON.

mod common;

use std::path::Path;

use engine_cerebras::spatial::serve2d::BatchLane;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let (l1, l2): (usize, usize) = (
        args[3].parse().expect("laps"),
        args[4].parse().expect("laps"),
    );
    let mut m = common::open(Path::new(&args[1]), Path::new(&args[2]))?;
    let (n, shape) = common::batched(&m)?;
    let base: [&[u32]; 6] = [&[3, 1], &[7], &[11, 8], &[6], &[5, 9], &[2]];
    let prompts: Vec<Vec<u32>> = (0..n).map(|g| base[g % base.len()].to_vec()).collect();
    let ps = shape.page_size;
    let npg = (l1 + l2).div_ceil(ps);
    let lane = |g: usize, pos: usize, forced: Vec<u32>| BatchLane {
        pos: pos as u32,
        slot: pos as u32,
        pages: (0..npg).map(|i| (g * npg + i) as u32).collect(),
        visible: None,
        forced,
    };
    let (out1, _) = m.fire_batch(
        &(0..n)
            .map(|g| lane(g, 0, prompts[g].clone()))
            .collect::<Vec<_>>(),
        l1,
        false,
    )?;
    eprintln!("launch 1 {out1:?}");
    let (out2, logits) = m.fire_batch(
        &(0..n)
            .map(|g| lane(g, l1, vec![*out1[g].last().unwrap()]))
            .collect::<Vec<_>>(),
        l2,
        true,
    )?;
    eprintln!("launch 2 {out2:?}");
    let best: Vec<usize> = logits.iter().map(|row| common::argmax(row)).collect();
    let got: Vec<Vec<u32>> = (0..n)
        .map(|g| out1[g].iter().chain(&out2[g]).copied().collect())
        .collect();
    println!(
        "{}",
        serde_json::json!({"prompts": prompts, "got": got, "logits_argmax": best})
    );
    Ok(())
}
