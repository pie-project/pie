//! Lists every host-to-device copy PIE's serve8b (or serve8b_region)
//! placement makes (the routes
//! before `init`, then the weights), one line a copy:
//! `symbol x y w h n bits sha256`, `bits` 16 or 32 and the hash over the
//! copy's words as little-endian u32. Nothing runs on a device: for a
//! byte-level comparison with `serve_lib8b.py` under a recording runtime.
//!
//! `serve8b_h2d <K:V,... params> <checkpoint dir> [rope_theta]`

use engine_cerebras::sdk::Rect;
use engine_cerebras::spatial::ckpt::Ckpt;
use engine_cerebras::spatial::serve2d::{self, Layout2d};
use engine_cerebras::spatial::serve8b;
use engine_cerebras::spatial::wafer::Sink;
use sha2::{Digest, Sha256};

struct Record;

impl Record {
    fn line(symbol: &str, r: Rect, n: usize, bits: u32, words: &[u32]) -> Result<(), String> {
        if words.len() != (r.w * r.h) as usize * n {
            return Err(format!("{symbol}: {} words for {r:?} x {n}", words.len()));
        }
        let mut h = Sha256::new();
        for w in words {
            h.update(w.to_le_bytes());
        }
        let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        println!("{symbol} {} {} {} {} {n} {bits} {hex}", r.x, r.y, r.w, r.h);
        Ok(())
    }
}

impl Sink for Record {
    fn h2d(&mut self, symbol: &str, rect: Rect, n: usize, words: &[u32]) -> Result<(), String> {
        Record::line(symbol, rect, n, 32, words)
    }

    fn h2d16(&mut self, symbol: &str, rect: Rect, n: usize, words: &[u32]) -> Result<(), String> {
        Record::line(symbol, rect, n, 16, words)
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let params: Vec<(String, usize)> = args
        .get(1)
        .ok_or("K:V,... params")?
        .split(',')
        .filter_map(|kv| kv.split_once(':'))
        .map(|(k, v)| (k.to_string(), v.parse().unwrap_or(0)))
        .collect();
    let lay = Layout2d::of(|k| params.iter().find(|p| p.0 == k).map(|p| p.1)).ok_or("not a serve8b build")??;
    let ckpt = Ckpt::open(std::path::Path::new(args.get(2).ok_or("a checkpoint dir")?))?;
    let theta = match args.get(3) {
        Some(t) => t.parse().map_err(|e| format!("{e}"))?,
        None => ckpt.config()["rope_theta"].as_f64().unwrap_or(1.0e6),
    };
    let mut rec = Record;
    // serve8b_region with HROUTE: a stand-in for the program's route
    // encoding (bit i), as the Python recorder's.
    let rbits: Vec<u32> = (0..10).map(|i| 1 << i).collect();
    serve8b::put_routes(&lay, &mut rec, Some(&rbits))?;
    serve2d::place(&lay, &ckpt, theta, &mut rec)
}
