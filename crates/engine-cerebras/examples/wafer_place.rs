//! Records the spatial wafer model's weight placement without a device:
//! `wafer_place <program dir with out.json> <ckpt dir> <out dir>` writes
//! every copy as `calls.jsonl` (symbol, rect, elements a PE) and the words
//! as `calls.bin`, for checking against the bring-up's serve_lib28.py.

use std::io::Write;
use std::path::Path;

use engine_cerebras::sdk::Rect;
use engine_cerebras::spatial::ckpt::Ckpt;
use engine_cerebras::spatial::wafer::{Layout, Sink, place};

struct Recorder {
    index: std::fs::File,
    words: std::fs::File,
}

impl Sink for Recorder {
    fn h2d(&mut self, symbol: &str, rect: Rect, n: usize, words: &[u32]) -> Result<(), String> {
        writeln!(
            self.index,
            "{}",
            serde_json::json!({"sym": symbol, "x": rect.x, "y": rect.y, "w": rect.w, "h": rect.h, "n": n})
        )
        .map_err(|e| e.to_string())?;
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.words.write_all(&bytes).map_err(|e| e.to_string())
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let lay = Layout::read(Path::new(&args[1]))?;
    let ckpt = Ckpt::open(Path::new(&args[2]))?;
    let theta = ckpt.config()["rope_theta"].as_f64().unwrap_or(1.0e6);
    let out = Path::new(&args[3]);
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let mut rec = Recorder {
        index: std::fs::File::create(out.join("calls.jsonl")).map_err(|e| e.to_string())?,
        words: std::fs::File::create(out.join("calls.bin")).map_err(|e| e.to_string())?,
    };
    place(&lay, &ckpt, theta, &mut rec)?;
    println!("hbe {:?} width {} height {}", lay.hbe(), lay.width, lay.height);
    Ok(())
}
