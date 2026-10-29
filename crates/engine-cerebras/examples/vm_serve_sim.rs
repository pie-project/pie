//! serve2d with the eta VM (layout `VM` 1) driven through the engine's own
//! wafer path ([`WaferModel::vm_load`], [`WaferModel::vm_step`]) on the
//! fabric simulator, as `payload/serve2d/vm_check.py` drives it:
//! `vm_serve_sim <compiled program dir> <ckpt dir> <etagen case json>
//! <steps> <prompt ids,..> <out dir>`. The prompt but its last token is
//! prefilled with the head's argmax (gmode 0), the case's program is loaded,
//! then every step is one VM launch (gmode 1). Writes `<out dir>/report.json`
//! (the VM's tokens and out-ring rows, in check_vm_dump.py's shape less the
//! planes) and `<out dir>/planes.f32` (the rows it sampled from, a step
//! after another).

use std::path::Path;

use engine_cerebras::sdk::{Platform, Simulator, Target};
use engine_cerebras::spatial::guest4::wafer;
use engine_cerebras::spatial::wafer::{Logits, WaferModel};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let (program, ckpt, case_path) = (Path::new(&args[1]), Path::new(&args[2]), &args[3]);
    let steps: usize = args[4].parse().expect("steps");
    let prompt: Vec<u32> = args[5].split(',').map(|t| t.parse().expect("token id")).collect();
    let out = Path::new(&args[6]);
    let case: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(case_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let (table, seeds) = wafer::case_table(&case)?;
    let mut m = WaferModel::open(
        program,
        ckpt,
        Platform::Simulator(Simulator {
            num_threads: std::env::var("PIE_SIM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(5),
            ..Simulator::new(Target::Wse3)
        }),
        None,
    )?;
    let n = prompt.len();
    if n > 1 {
        m.forward(&prompt[..n - 1], 0, Logits::None)?;
    }
    m.vm_load(&table, &seeds)?;
    let (mut tokens, mut rows, mut planes) = (Vec::new(), Vec::new(), Vec::<f32>::new());
    let mut tok = prompt[n - 1];
    for k in 0..steps {
        let (sampled, row, _pool) = m.vm_step(tok, n - 1 + k)?;
        eprintln!("step {k}: {sampled} {row:?}");
        planes.extend(m.vm_plane()?);
        tokens.push(sampled);
        rows.push(row);
        tok = sampled;
    }
    let name = case["name"].as_str().unwrap_or("case").to_string();
    let bytes: Vec<u8> = planes.iter().flat_map(|v| v.to_le_bytes()).collect();
    let report = serde_json::json!({
        "prompt": prompt, "steps": steps,
        "cases": {name: {"tokens_a": tokens, "ring_rows": rows}},
    });
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    std::fs::write(out.join("planes.f32"), bytes).map_err(|e| e.to_string())?;
    std::fs::write(out.join("report.json"), report.to_string()).map_err(|e| e.to_string())?;
    println!("{}", serde_json::json!({"tokens": tokens}));
    Ok(())
}
