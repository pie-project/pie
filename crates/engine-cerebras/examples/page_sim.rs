//! The host tier's page round trip on the fabric simulator: `page_sim
//! <program dir> <ckpt dir>`. A batched build's page images out
//! ([`WaferModel::page_out`]), a made-up image into another page
//! ([`WaferModel::page_in`]) and back out word for word, the pages around it
//! as they were; prints the time a page takes each way.

use std::path::Path;
use std::time::Instant;

use engine_cerebras::sdk::{Platform, Simulator, Target};
use engine_cerebras::spatial::wafer::{Design, WaferModel};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let (program, ckpt) = (Path::new(&args[1]), Path::new(&args[2]));
    let mut m = WaferModel::open(
        program,
        ckpt,
        Platform::Simulator(Simulator {
            num_threads: std::env::var("PIE_SIM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(5),
            ..Simulator::new(Target::Wse3)
        }),
        None,
    )?;
    let Design::Grid(grid) = m.design() else { return Err("not a serve2d build".into()) };
    let shape = grid.paged.ok_or("not a batched build")?;
    // The host copy's words: its cells at the KV's width (bf16 two a word).
    let words = m.page_bytes().ok_or("the host keeps no pages of this build")? / 4;
    let pages: Vec<u32> = (0..shape.pages() as u32).collect();
    let before = m.page_out(&pages)?;
    // A made-up image (distinct words) into page `dst`; its neighbours on
    // the same row and the same block must not move.
    let dst = (shape.pages() as u32 / 2) | 1;
    let image: Vec<u32> = (0..words as u32).map(|i| 0x3f80_0000 ^ i.wrapping_mul(2_654_435_761)).collect();
    let t0 = Instant::now();
    m.page_in(&[(dst, image.clone())])?;
    let t_in = t0.elapsed();
    let t0 = Instant::now();
    let back = m.page_out(&[dst])?.remove(0);
    let t_out = t0.elapsed();
    let after = m.page_out(&pages)?;
    let same = back == image;
    let others = pages.iter().filter(|&&p| p != dst).all(|&p| after[p as usize] == before[p as usize]);
    println!(
        "{{\"pages\": {}, \"page_words\": {words}, \"round_trip\": {same}, \"others_unmoved\": {others}, \"in_ms\": {:.1}, \"out_ms\": {:.1}}}",
        pages.len(),
        t_in.as_secs_f64() * 1e3,
        t_out.as_secs_f64() * 1e3
    );
    if same && others { Ok(()) } else { Err("the page did not round-trip".into()) }
}
