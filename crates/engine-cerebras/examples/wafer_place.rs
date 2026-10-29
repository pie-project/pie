//! Records the spatial wafer model's weight placement without a device:
//! `wafer_place <program dir with out.json> <ckpt dir> <out dir> [--landed
//! <deployment>]` writes
//! every copy as `calls.jsonl` (symbol, rect, elements a PE, 16- or 32-bit) and the words
//! as `calls.bin`, for checking against the bring-up's serve_lib28.py.

use std::io::Write;
use std::path::Path;

use engine_cerebras::sdk::Rect;
use engine_cerebras::spatial::ckpt::Ckpt;
use engine_cerebras::spatial::wafer::{Design, Sink, place};

struct Recorder {
    index: std::fs::File,
    words: std::fs::File,
}

impl Recorder {
    fn record(&mut self, symbol: &str, rect: Rect, n: usize, bits: u32, words: &[u32]) -> Result<(), String> {
        writeln!(
            self.index,
            "{}",
            serde_json::json!({"sym": symbol, "x": rect.x, "y": rect.y, "w": rect.w, "h": rect.h, "n": n, "bits": bits})
        )
        .map_err(|e| e.to_string())?;
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.words.write_all(&bytes).map_err(|e| e.to_string())
    }
}

impl Sink for Recorder {
    fn h2d(&mut self, symbol: &str, rect: Rect, n: usize, words: &[u32]) -> Result<(), String> {
        self.record(symbol, rect, n, 32, words)
    }

    /// A 16-bit copy (serve8b's bf16 blocks): still a u32 word an element.
    fn h2d16(&mut self, symbol: &str, rect: Rect, n: usize, words: &[u32]) -> Result<(), String> {
        self.record(symbol, rect, n, 16, words)
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let design = Design::read(Path::new(&args[1]))?;
    // `--landed <deployment>` as a 5th and 6th argument: through PIE's
    // checkpoint path (the deployment's contract, poem names), as serving
    // reads it; else the directory by transformers names.
    let (ckpt, theta) = if args.get(4).is_some_and(|a| a == "--landed") {
        let deployment = poem_compiler::catalog::repository().parse(&args[5]).ok_or("no such deployment")?;
        let trace = deployment.trace(poem_ir::Platform::Cerebras);
        let source = checkpoint::file::snapshot::open(Path::new(&args[2])).map_err(|e| e.to_string())?;
        let contract = deployment.contract(&source, poem_ir::Platform::Cerebras).map_err(|e| e.to_string())?;
        let spec = engine_cerebras::spatial::plan::ModelSpec::read(&trace).map_err(|e| e.to_string())?;
        (Ckpt::landed(&trace, &contract, Path::new(&args[2]))?, f64::from(spec.rope_theta))
    } else {
        let ckpt = Ckpt::open(Path::new(&args[2]))?;
        let theta = ckpt.config()["rope_theta"].as_f64().unwrap_or(1.0e6);
        (ckpt, theta)
    };
    let out = Path::new(&args[3]);
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let mut rec = Recorder {
        index: std::fs::File::create(out.join("calls.jsonl")).map_err(|e| e.to_string())?,
        words: std::fs::File::create(out.join("calls.bin")).map_err(|e| e.to_string())?,
    };
    match &design {
        Design::Strips(lay) => {
            place(lay, &ckpt, theta, &mut rec)?;
            println!("hbe {:?} width {} height {}", lay.hbe(), lay.width, lay.height);
        }
        Design::Grid(grid) => {
            engine_cerebras::spatial::serve2d::place(grid, &ckpt, theta, &mut rec)?;
            println!("root {:?}", grid.root());
        }
    }
    Ok(())
}
