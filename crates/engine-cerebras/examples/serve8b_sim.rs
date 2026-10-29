//! serve8b through the engine's own wafer path on the fabric simulator (the
//! checks of `serve8b/run.py --mode serve`): `serve8b_sim <program dir>
//! <ckpt dir> <new> <steps2>`. B lanes of prompts [3,1,4], [7,2], [11,8,5],
//! [6,6] (repeated), lane b on pool pages from b·NB·PY/B: one launch of
//! max prompt + `new` passes, a continuation of `steps2` passes from the
//! last token with the logits rows. Then, on any build, a page copy: lane 0's
//! pages copied to free ones and one more pass of lane 0 on the copies next
//! to lane 1 continuing lane 0 on its own pages (the two must agree). Prints
//! one JSON line.

use std::path::Path;

use engine_cerebras::sdk::{Platform, Simulator, Target};
use engine_cerebras::spatial::serve2d::BatchLane;
use engine_cerebras::spatial::wafer::{Design, WaferModel};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let new: usize = args[3].parse().map_err(|e| format!("{e}"))?;
    let steps2: usize = args[4].parse().map_err(|e| format!("{e}"))?;
    let mut m = WaferModel::open(
        Path::new(&args[1]),
        Path::new(&args[2]),
        Platform::Simulator(Simulator {
            num_threads: std::env::var("PIE_SIM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(5),
            ..Simulator::new(Target::Wse3)
        }),
        None,
    )?;
    let Design::Grid(grid) = m.design() else { return Err("not a grid build".into()) };
    let shape = grid.paged.ok_or("not a batched build")?;
    let (b_n, pool, ps) = (shape.lanes, shape.pages() as u32, shape.page_size);
    let base: [&[u32]; 4] = [&[3, 1, 4], &[7, 2], &[11, 8, 5], &[6, 6]];
    let prompts: Vec<Vec<u32>> = (0..b_n).map(|b| base[b % 4].to_vec()).collect();
    let steps1 = prompts.iter().map(Vec::len).max().unwrap_or(0) + new;
    let total = steps1 + steps2;
    let npg = (total + 1).div_ceil(ps) as u32;
    let pages: Vec<Vec<u32>> = (0..b_n as u32)
        .map(|b| {
            let first = (b * pool / b_n as u32) % pool;
            (0..npg).map(|i| (first + i) % pool).collect()
        })
        .collect();
    let lane = |pos: usize, pages: Vec<u32>, forced: Vec<u32>| BatchLane {
        pos: pos as u32,
        slot: pos as u32,
        pages,
        visible: None,
        forced,
    };
    let lanes: Vec<BatchLane> = (0..b_n).map(|b| lane(0, pages[b].clone(), prompts[b].clone())).collect();
    let (out1, _) = m.fire_batch(&lanes, steps1, false)?;
    eprintln!("fire 1 {out1:?}");
    let lanes: Vec<BatchLane> = (0..b_n)
        .map(|b| lane(steps1, pages[b].clone(), vec![*out1[b].last().unwrap()]))
        .collect();
    let (out2, logits) = m.fire_batch(&lanes, steps2, true)?;
    eprintln!("fire 2 {out2:?}");
    let argmax: Vec<usize> = logits
        .iter()
        .map(|row| row.iter().enumerate().fold(0, |best, (i, &x)| if x > row[best] { i } else { best }))
        .collect();
    let got: Vec<Vec<u32>> = (0..b_n).map(|b| [out1[b].clone(), out2[b].clone()].concat()).collect();
    // Lane 0's pages copied to pages no lane holds, then one more pass of
    // lane 0 on the copies (lane 0) and on its own pages (lane 1).
    let held: Vec<u32> = pages.concat();
    let free: Vec<u32> = (0..pool).filter(|g| !held.contains(g)).take(pages[0].len()).collect();
    let pairs: Vec<(u32, u32)> = pages[0].iter().copied().zip(free.iter().copied()).collect();
    m.copy_pages(&pairs)?;
    let last = *out2[0].last().unwrap();
    let mut lanes = vec![lane(total, free.clone(), vec![last])];
    if b_n > 1 {
        lanes.push(lane(total, pages[0].clone(), vec![last]));
    }
    let (out3, _) = m.fire_batch(&lanes, 1, false)?;
    let copy_agrees = out3.len() < 2 || out3[0] == out3[1];
    println!(
        "{}",
        serde_json::json!({
            "prompts": prompts, "got": got, "logits_argmax": argmax, "copy": pairs,
            "after_copy": out3, "copy_agrees": copy_agrees,
        })
    );
    Ok(())
}
