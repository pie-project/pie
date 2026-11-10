//! Catalog models with random weights fired through the engine: every
//! readout of a fixed scenario, bit for bit. With `PIE_XLA_GOLDEN_DUMP=<dir>`
//! the readouts are written there (one file a model); with
//! `PIE_XLA_GOLDEN=<dir>` they are compared against what a build wrote
//! there. Skipped without a PJRT plugin.

use crate::common_llm::{Model, scenario};

fn file(dir: &str, name: &str) -> std::path::PathBuf {
    std::path::Path::new(dir).join(format!("{name}.bits"))
}

fn encode(readouts: &[Vec<u32>]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in readouts {
        out.extend_from_slice(&(r.len() as u32).to_le_bytes());
        for v in r {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn run(name: &str, layers: Option<u32>, context: u32, long: Option<u32>) {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load(name, layers, context, 7);
    let readouts = scenario(&mut model, long);
    assert!(
        readouts.iter().all(|r| !r.is_empty()),
        "{name}: every fire reads out"
    );
    let bytes = encode(&readouts);
    if let Ok(dir) = std::env::var("PIE_XLA_GOLDEN_DUMP") {
        std::fs::create_dir_all(&dir).expect("the dump directory");
        std::fs::write(file(&dir, name), &bytes).expect("the dump writes");
    }
    if let Ok(dir) = std::env::var("PIE_XLA_GOLDEN") {
        let want = std::fs::read(file(&dir, name)).expect("the golden file reads");
        assert!(
            want == bytes,
            "{name}: the readouts differ from the golden build's ({} vs {} bytes)",
            bytes.len(),
            want.len()
        );
        eprintln!("{name}: {} readouts bit-identical", readouts.len());
    }
}

#[test]
fn a_dense_model_answers_bit_for_bit() {
    run("qwen3-micro-text", None, 256, None);
}

#[test]
fn a_gated_delta_hybrid_answers_bit_for_bit() {
    run("qwen35-tiny", None, 256, None);
}

#[test]
fn a_windowed_model_answers_bit_for_bit_past_its_window() {
    run("gemma4-e4b-mini-l1", None, 1024, Some(700));
}
