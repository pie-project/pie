//! A per-row table three ways, on the simulator: `rowbcast_bench <compiled
//! program dir> <symbol> <x> <y> <w> <h> <n>` writes `h` rows of `n` words
//! (row r's words the same on every PE of row r) into `symbol` over the
//! rectangle with `memcpy_h2d_rowbcast`, with one `memcpy_h2d` a row (the row
//! replicated `w` times), and with one `memcpy_h2d` of the whole rectangle;
//! checks each by reading the rectangle back and prints the times as JSON.

use std::path::Path;
use std::time::Instant;

use engine_cerebras::sdk::{DataType, Order, Platform, Rect, Runtime, Sdk, Simulator, Target};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let num = |i: usize| -> usize { args[i].parse().expect("a number") };
    let (symbol, x, y, w, h, n) = (&args[2], num(3), num(4), num(5), num(6), num(7));
    let sdk = Sdk::open().map_err(|e| e.to_string())?;
    let mut rt = Runtime::new(
        sdk,
        Path::new(&args[1]),
        Platform::Simulator(Simulator::new(Target::Wse3)),
        "WARNING",
    )
    .map_err(|e| e.to_string())?;
    rt.load();
    rt.run();
    let e = |e: engine_cerebras::sdk::Error| e.to_string();
    let rect = Rect { x: x as i32, y: y as i32, w: w as i32, h: h as i32 };
    let row = |r: usize, round: u32| -> Vec<u32> {
        (0..n).map(|k| (round << 24) | ((r as u32) << 12) | k as u32).collect()
    };
    let check = |rt: &mut Runtime, round: u32| -> Result<bool, String> {
        let mut back = vec![0u32; w * h * n];
        rt.memcpy_d2h(symbol, &mut back, rect, n, Order::RowMajor, DataType::Bits32)
            .map_err(e)?;
        Ok((0..h).all(|r| (0..w).all(|c| back[(r * w + c) * n..(r * w + c + 1) * n] == row(r, round)[..])))
    };
    // Warm the copy paths once.
    let all: Vec<u32> = (0..h).flat_map(|r| (0..w).flat_map(move |_| row(r, 0))).collect();
    rt.memcpy_h2d(symbol, &all, rect, n, Order::RowMajor, DataType::Bits32).map_err(e)?;

    let t = Instant::now();
    let rows: Vec<u32> = (0..h).flat_map(|r| row(r, 1)).collect();
    rt.memcpy_h2d_rowbcast(symbol, &rows, rect, n, DataType::Bits32).map_err(e)?;
    let rowbcast_s = t.elapsed().as_secs_f64();
    let rowbcast_ok = check(&mut rt, 1)?;

    let t = Instant::now();
    for r in 0..h {
        let line: Vec<u32> = (0..w).flat_map(|_| row(r, 2)).collect();
        let one = Rect { x: x as i32, y: (y + r) as i32, w: w as i32, h: 1 };
        rt.memcpy_h2d(symbol, &line, one, n, Order::RowMajor, DataType::Bits32).map_err(e)?;
    }
    let per_row_s = t.elapsed().as_secs_f64();
    let per_row_ok = check(&mut rt, 2)?;

    let t = Instant::now();
    let all: Vec<u32> = (0..h).flat_map(|r| (0..w).flat_map(move |_| row(r, 3))).collect();
    rt.memcpy_h2d(symbol, &all, rect, n, Order::RowMajor, DataType::Bits32).map_err(e)?;
    let whole_s = t.elapsed().as_secs_f64();
    let whole_ok = check(&mut rt, 3)?;

    println!(
        "{}",
        serde_json::json!({
            "rect": [x, y, w, h], "n": n,
            "rowbcast": {"s": rowbcast_s, "ok": rowbcast_ok},
            "per_row_h2d": {"s": per_row_s, "ok": per_row_ok},
            "whole_h2d": {"s": whole_s, "ok": whole_ok},
        })
    );
    rt.stop();
    Ok(())
}
