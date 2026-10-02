//! Compiles one rendered program directory (`layout.csl` + `pe.csl`, as the
//! program cache keeps them) with the SDK compiler, to measure a program
//! against a PE by hand: `cslc_probe DIR W H` writes `DIR/out`.
use engine_cerebras::sdk::{Arch, Compile, Cslc};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("a program directory"));
    let w: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(1);
    let h: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(1);
    let out = dir.join("out");
    let _ = std::fs::remove_dir_all(&out);
    let compile = Compile::memcpy(Arch::Wse3, dir.join("layout.csl"), w, h, &out);
    match Cslc::find().and_then(|c| c.compile(&compile)) {
        Ok(()) => println!("linked"),
        Err(e) => {
            let text = e.to_string();
            println!("failed:\n{}", text.lines().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n"));
        }
    }
}
