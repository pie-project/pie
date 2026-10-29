//! The batched per-pass table for lanes given as JSON, for checking against
//! `serve_lib2d.WaferModel.tables`: `paged_table <B> <P> <PS> <NB> <E> <steps>`
//! reads `[{"pos","wslot"?,"pages","visible"?}, ..]` on stdin and prints the
//! table (rows of u32) as JSON.

use std::io::Read;

use engine_cerebras::spatial::paged::{Lane, Shape, table};

fn main() -> Result<(), String> {
    let a: Vec<usize> = std::env::args().skip(1).map(|v| v.parse().expect("a number")).collect();
    let shape = Shape { lanes: a[0], rows: a[1], page_size: a[2], blocks: a[3], entries: a[4] };
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let num = |x: &serde_json::Value| x.as_u64().unwrap_or(0) as u32;
    let lanes: Vec<Lane> = v
        .as_array()
        .ok_or("a list of lanes")?
        .iter()
        .map(|l| Lane {
            pos: num(&l["pos"]),
            slot: l.get("wslot").map_or(num(&l["pos"]), num),
            pages: l["pages"].as_array().map(|p| p.iter().map(num).collect()).unwrap_or_default(),
            visible: l.get("visible").and_then(|v| v.as_array()).map(|v| v.iter().map(num).collect()),
        })
        .collect();
    let tab = table(&shape, &lanes, a[5] as u32)?;
    let rows: Vec<&[u32]> = tab.chunks(shape.row_words()).collect();
    println!("{}", serde_json::to_string(&rows).map_err(|e| e.to_string())?);
    Ok(())
}
