//! Lists every host-to-device copy PIE's serve8b (or serve8b_region)
//! placement makes (the routes
//! before `init`, then the weights), one line a copy:
//! `symbol x y w h n bits sha256`, `bits` 16 or 32 and the hash over the
//! copy's words as little-endian u32. Nothing runs on a device: for a
//! byte-level comparison with `serve_lib8b.py` under a recording runtime.
//!
//! `serve8b_h2d <K:V,... params> <checkpoint dir> [rope_theta]`: the
//! checkpoint read by transformers names (as the Python hosts read it);
//! `serve8b_h2d <params> <checkpoint> --landed <deployment>`: through PIE's
//! checkpoint path (the deployment's contract, poem names), as serving
//! reads it. The two must list the same copies.

use engine_cerebras::sdk::Rect;
use engine_cerebras::spatial::ckpt::Ckpt;
use engine_cerebras::spatial::serve2d::{self, Layout2d};
use engine_cerebras::spatial::serve8b;
use engine_cerebras::spatial::serve8b_region;
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
    let dir = std::path::Path::new(args.get(2).ok_or("a checkpoint dir")?);
    let (ckpt, theta) = if args.get(3).is_some_and(|a| a == "--landed") {
        let name = args.get(4).ok_or("a deployment")?;
        let deployment = poem_compiler::catalog::repository().parse(name).ok_or("no such deployment")?;
        let platform = poem_ir::Platform::Cerebras;
        let trace = deployment.trace(platform);
        let source = checkpoint::file::snapshot::open(dir).map_err(|e| e.to_string())?;
        let contract = deployment.contract(&source, platform).map_err(|e| e.to_string())?;
        let spec = engine_cerebras::spatial::plan::ModelSpec::read(&trace).map_err(|e| e.to_string())?;
        (Ckpt::landed(&trace, &contract, dir)?, f64::from(spec.rope_theta))
    } else {
        let ckpt = Ckpt::open(dir)?;
        let theta = match args.get(3) {
            Some(t) => t.parse().map_err(|e| format!("{e}"))?,
            None => ckpt.config()["rope_theta"].as_f64().unwrap_or(1.0e6),
        };
        (ckpt, theta)
    };
    let mut rec = Record;
    // serve8b_region with HROUTE: a stand-in for the program's route
    // encoding (bit i), as the Python recorder's.
    let rbits: Vec<u32> = (0..10).map(|i| 1 << i).collect();
    serve8b::put_routes(&lay, &mut rec, Some(&rbits))?;
    serve2d::place(&lay, &ckpt, theta, &mut rec)?;
    // serve8b_region with the eta VM: SERVE8B_VM_CASES (etagen case JSONs, a
    // lane each, `-` an empty lane) loaded as `vmhost.load_tables` does.
    if let Ok(list) = std::env::var("SERVE8B_VM_CASES") {
        let mut owned = Vec::new();
        for f in list.split(',') {
            owned.push(if f == "-" {
                None
            } else {
                let text = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
                let case: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{f}: {e}"))?;
                let rows = case["rows"].as_u64().unwrap_or(1) as u32;
                let (table, seeds) = engine_cerebras::spatial::guest4::wafer::case_table(&case)?;
                Some((table, seeds, rows))
            });
        }
        let lanes: Vec<Option<serve8b_region::VmLane>> = owned
            .iter()
            .map(|o| o.as_ref().map(|(table, seeds, rows)| serve8b_region::VmLane { table, seeds, rows: *rows }))
            .collect();
        serve8b_region::vm_load(&lay, &mut rec, &lanes)?;
    }
    Ok(())
}
