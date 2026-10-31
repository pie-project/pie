//! The guest4 translator's tables for a handful of the corpus's real programs
//! (`guest4_corpus/<example>_<hash>.ptir`: each program's reference PTIR as
//! the engine registered it) are
//! pinned: each program's table, homes, outputs, feature bits and bank sets,
//! as `json_of` writes them, must match `guest4_corpus/<same>.json` exactly.
//! A translator change that moves any of them fails here. The set covers
//! top-p, a plain sampler, a grammar mask written every step, a per-step
//! scalar, typical (top-k 128 and short ops), a two-row mask channel, asap's
//! sentinel-row gather, a window mask, per-row top-k and two-row beam search.
//!
//! When a change is meant to move a table, re-bless the goldens on purpose:
//! `GUEST4_BLESS=1 cargo test -p engine-cerebras --test engine_cerebras guest4_translation`.

use std::path::{Path, PathBuf};

use engine_cerebras::spatial::guest4;
use eta_compiler::plan::compile_bound;
use eta_exec::ExecPlan;
use eta_ir::Dtype;
use eta_ir::container::HostRole;
use eta_ir::registry::{Axis, EngineCaps, OutputDecl};

/// Qwen3's vocabulary, the size the programs were dumped at.
const VOCAB: u32 = 151_936;

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/engine_cerebras/guest4_corpus")
}

/// The program's plan as the engine builds it: bound to one logits output,
/// compiled, adopted by eta-exec.
fn plan_of(bytes: &[u8]) -> (ExecPlan, Vec<u32>) {
    let container = eta_ir::container::decode(bytes).expect("the .ptir decodes");
    let host_takes = container
        .channels
        .iter()
        .enumerate()
        .filter(|(_, d)| d.host_role == HostRole::Reader)
        .map(|(i, _)| i as u32)
        .collect();
    let logits = vec![vec![OutputDecl {
        name: "logits".into(),
        axis: Axis::Row,
        width: VOCAB,
        dtype: Dtype::F32,
        layers: 0,
        readout: true,
    }]];
    let bound = eta_ir::validate::bind(
        container,
        logits,
        EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
    .expect("binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let plan = eta_exec::adopt_launch_package(launch).expect("adopts");
    assert!(plan.executable, "not executable: {:?}", plan.reject_reason);
    (plan, host_takes)
}

#[test]
fn guest4_translation_is_pinned() {
    let bless = std::env::var_os("GUEST4_BLESS").is_some();
    let mut files: Vec<PathBuf> = std::fs::read_dir(corpus())
        .expect("the corpus directory")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "ptir"))
        .collect();
    files.sort();
    assert!(files.len() >= 10, "the pinned corpus is missing programs");
    let mut moved = Vec::new();
    for ptir in &files {
        let (plan, host_takes) = plan_of(&std::fs::read(ptir).expect("reads"));
        // The corpus is pinned at 24 registers and a 4096-word pool.
        let table =
            guest4::translate(&plan, VOCAB, &host_takes, 24, 4096, false).expect("translates");
        let got = guest4::json_of(&table);
        let golden = ptir.with_extension("json");
        if bless {
            let text = serde_json::to_string_pretty(&got).expect("json") + "\n";
            std::fs::write(&golden, text).expect("writes the golden");
            continue;
        }
        // Compared as values, not text: a workspace build turns on
        // serde_json's `preserve_order`, which changes how keys print.
        let want: Option<serde_json::Value> = std::fs::read_to_string(&golden)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok());
        if want.as_ref() != Some(&got) {
            moved.push(ptir.file_stem().unwrap().to_string_lossy().into_owned());
        }
    }
    assert!(
        moved.is_empty(),
        "guest4 tables moved (re-bless if intended): {moved:?}"
    );
}
